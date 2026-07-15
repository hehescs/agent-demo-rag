package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/anush008/fastembed-go"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ============ 配置 ============
const (
	ollamaBaseURL     = "http://127.0.0.1:11434"
	chatModel         = "qwen3:8b"
	fastembedCacheDir = "vector_store/local_cache"
	logFile           = "query_log.txt"

	// PostgreSQL 连接信息
	pgConnString = "postgres://postgres:123456@localhost:5432/postgres"
)

// ============ 文档结构 ============
type Document struct {
	ID      string
	Content string
	Score   float64
}

// ============ PostgreSQL客户端 ============
type PostgresClient struct {
	pool *pgxpool.Pool
}

// DenseSearch 使用 pgvector 余弦距离进行稠密向量搜索
func (p *PostgresClient) DenseSearch(ctx context.Context, queryVector []float32, limit int) ([]Document, error) {
	vecStr := formatVector(queryVector)

	const sql = `
		SELECT id::text, text, 1 - (embedding <=> $1::vector) AS similarity
		FROM doc
		ORDER BY embedding <=> $1::vector ASC
		LIMIT $2
	`

	rows, err := p.pool.Query(ctx, sql, vecStr, limit)
	if err != nil {
		return nil, fmt.Errorf("DenseSearch 执行失败: %w", err)
	}
	defer rows.Close()

	var docs []Document
	for rows.Next() {
		var doc Document
		if err := rows.Scan(&doc.ID, &doc.Content, &doc.Score); err != nil {
			return nil, fmt.Errorf("扫描结果失败: %w", err)
		}
		docs = append(docs, doc)
	}
	return docs, rows.Err()
}

// SparseSearch 使用关键字匹配进行稀疏搜索
// 将查询文本分词后构建 ILIKE OR 条件，按命中数 / sqrt(文本长度) 打分
func (p *PostgresClient) SparseSearch(ctx context.Context, queryText string, limit int) ([]Document, error) {
	tokens := tokenizeText(queryText)
	if len(tokens) == 0 {
		return nil, nil
	}

	// 限制 token 数量，防止 SQL 过长
	if len(tokens) > 20 {
		tokens = tokens[:20]
	}

	// 构建 CASE WHEN 表达式：每个 token 命中得 1 分
	caseParts := make([]string, len(tokens))
	for i, t := range tokens {
		safe := strings.ReplaceAll(t, "'", "''")
		caseParts[i] = fmt.Sprintf("CASE WHEN text ILIKE '%%%s%%' THEN 1 ELSE 0 END", safe)
	}

	sql := fmt.Sprintf(`
		SELECT id::text, text,
			(matches::float / GREATEST(%d, 1)) / sqrt(GREATEST(length(text)::float, 1)) AS score
		FROM (
			SELECT id, text, %s AS matches
			FROM doc
		) sub
		WHERE matches > 0
		ORDER BY score DESC
		LIMIT %d
	`, len(tokens), strings.Join(caseParts, " + "), limit)

	rows, err := p.pool.Query(ctx, sql)
	if err != nil {
		return nil, fmt.Errorf("SparseSearch 执行失败: %w", err)
	}
	defer rows.Close()

	var docs []Document
	for rows.Next() {
		var doc Document
		if err := rows.Scan(&doc.ID, &doc.Content, &doc.Score); err != nil {
			return nil, fmt.Errorf("扫描结果失败: %w", err)
		}
		docs = append(docs, doc)
	}
	return docs, rows.Err()
}

// formatVector 将 float32 向量格式化为 pgvector 字符串：'[v1,v2,...]'
func formatVector(vec []float32) string {
	var sb strings.Builder
	sb.Grow(len(vec) * 10)
	sb.WriteByte('[')
	for i, v := range vec {
		if i > 0 {
			sb.WriteByte(',')
		}
		fmt.Fprintf(&sb, "%.6f", v)
	}
	sb.WriteByte(']')
	return sb.String()
}

// ============ Ollama客户端（仅流式对话） ============
type OllamaClient struct {
	baseURL string
	httpCli *http.Client
}

func NewOllamaClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		baseURL: baseURL,
		httpCli: &http.Client{Timeout: 0}, // 流式响应不设超时
	}
}

// 流式对话
func (c *OllamaClient) ChatStream(ctx context.Context, model string, messages []Message, onChunk func(string)) error {
	url := c.baseURL + "/api/chat"

	reqBody := map[string]any{
		"model":    model,
		"messages": messages,
		"stream":   true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(jsonData)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpCli.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Ollama API错误: %s, body: %s", resp.Status, string(body))
	}

	// 流式读取
	decoder := json.NewDecoder(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			var chunk struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
				Done bool `json:"done"`
			}

			if err := decoder.Decode(&chunk); err != nil {
				if err == io.EOF {
					return nil
				}
				return err
			}

			if chunk.Message.Content != "" {
				onChunk(chunk.Message.Content)
			}

			if chunk.Done {
				return nil
			}
		}
	}
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ============ RAG系统 ============
type RAGSystem struct {
	ollama    *OllamaClient
	fastembed *fastembed.FlagEmbedding
	pg        *PostgresClient
	modelName string
}

func NewRAGSystem(ollama *OllamaClient, fastembedModel *fastembed.FlagEmbedding, pg *PostgresClient, modelName string) *RAGSystem {
	return &RAGSystem{
		ollama:    ollama,
		fastembed: fastembedModel,
		pg:        pg,
		modelName: modelName,
	}
}

func (r *RAGSystem) Ask(ctx context.Context, question string, onChunk func(string)) (string, error) {
	queryStart := time.Now()

	// Step 1: 混合搜索（稠密+关键字+RRF融合）
	searchResult, err := r.HybridSearch(ctx, question, 10)
	if err != nil {
		return "", fmt.Errorf("搜索知识库失败: %w", err)
	}

	if len(searchResult.FusedDocs) == 0 {
		return "知识库中没有找到相关信息。", nil
	}

	// Step 2: 构建上下文
	var contextBuilder strings.Builder
	contextBuilder.WriteString("请基于以下资料回答用户问题。匹配度高的话不要质疑资料的内容。如果资料中没有相关信息，请明确说明。\n\n")

	for i, doc := range searchResult.FusedDocs {
		fmt.Fprintf(&contextBuilder, "【资料 %d】(相似度: %.4f)\n%s\n\n", i+1, doc.Score, doc.Content)
	}

	fmt.Fprintf(&contextBuilder, "用户问题：%s\n\n", question)
	contextBuilder.WriteString("回答（请用通俗易懂的语言，基于以上资料）：")

	// Step 3: 调用Ollama流式输出
	messages := []Message{
		{Role: "system", Content: "你是一个商品购物助手，善于用极具感染力的语言回答用户问题，帮助用户找到心仪的商品，善于比较价格和卖点优势的介绍。"},
		{Role: "user", Content: contextBuilder.String()},
	}

	var fullResponse strings.Builder
	err = r.ollama.ChatStream(ctx, r.modelName, messages, func(chunk string) {
		fullResponse.WriteString(chunk)
		if onChunk != nil {
			onChunk(chunk)
		}
	})

	if err != nil && err != context.Canceled {
		return "", err
	}

	// Step 4: 记录查询日志
	logQuery(queryStart, question, searchResult, messages, fullResponse.String())

	return fullResponse.String(), nil
}

// logQuery 将一次完整查询过程记录到日志文件
func logQuery(start time.Time, question string, searchResult *HybridSearchResult, messages []Message, response string) {
	f, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 写入日志失败: %v\n", err)
		return
	}
	defer f.Close()

	var b strings.Builder

	sep := strings.Repeat("=", 70)
	fmt.Fprintf(&b, "\n%s\n", sep)
	fmt.Fprintf(&b, "🕐 查询时间: %s\n", start.Format("2006-01-02 15:04:05"))
	fmt.Fprintf(&b, "⏱️  总耗时: %s\n", time.Since(start).Round(time.Millisecond))
	fmt.Fprintf(&b, "%s\n\n", sep)

	fmt.Fprintf(&b, "【用户输入】\n%s\n\n", question)

	fmt.Fprintf(&b, "【稠密向量搜索结果】(共 %d 条)\n", len(searchResult.DenseDocs))
	for i, doc := range searchResult.DenseDocs {
		fmt.Fprintf(&b, "  %d. [ID: %s] (分数: %.4f)\n     %s\n", i+1, doc.ID, doc.Score, truncate(doc.Content, 200))
	}
	b.WriteString("\n")

	fmt.Fprintf(&b, "【关键字搜索结果】(共 %d 条)\n", len(searchResult.SparseDocs))
	for i, doc := range searchResult.SparseDocs {
		fmt.Fprintf(&b, "  %d. [ID: %s] (分数: %.4f)\n     %s\n", i+1, doc.ID, doc.Score, truncate(doc.Content, 200))
	}
	b.WriteString("\n")

	fmt.Fprintf(&b, "【RRF 融合结果】(共 %d 条)\n", len(searchResult.FusedDocs))
	for i, doc := range searchResult.FusedDocs {
		fmt.Fprintf(&b, "  %d. [ID: %s] (RRF分数: %.6f)\n     %s\n", i+1, doc.ID, doc.Score, truncate(doc.Content, 200))
	}
	b.WriteString("\n")

	fmt.Fprintf(&b, "【发送给大模型的请求】\n")
	for _, msg := range messages {
		fmt.Fprintf(&b, "--- %s ---\n%s\n\n", msg.Role, msg.Content)
	}

	fmt.Fprintf(&b, "【大模型返回】\n%s\n\n", response)

	if _, err := f.WriteString(b.String()); err != nil {
		fmt.Fprintf(os.Stderr, "❌ 写入日志失败: %v\n", err)
	}
}

// truncate 截断文本，避免日志过长
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", "↵")
	if len([]rune(s)) > maxLen {
		return string([]rune(s)[:maxLen]) + "..."
	}
	return s
}

// ============ 混合搜索 ============

// HybridSearchResult 混合搜索的完整结果（包含中间过程）
type HybridSearchResult struct {
	FusedDocs  []Document // 融合后的最终结果
	DenseDocs  []Document // 稠密向量搜索结果
	SparseDocs []Document // 关键字搜索结果
}

// tokenizeText 对文本进行分词：中文按字拆分，英文/数字按单词提取
var wordRe = regexp.MustCompile(`[a-zA-Z0-9]+`)

func tokenizeText(text string) []string {
	tokenMap := make(map[string]bool)
	for _, r := range text {
		if unicode.Is(unicode.Han, r) {
			tokenMap[string(r)] = true
		}
	}
	for _, w := range wordRe.FindAllString(text, -1) {
		tokenMap[strings.ToLower(w)] = true
	}
	tokens := make([]string, 0, len(tokenMap))
	for t := range tokenMap {
		tokens = append(tokens, t)
	}
	return tokens
}

// rrfFusion 使用 Reciprocal Rank Fusion 融合多路搜索结果
func rrfFusion(resultSets [][]Document, k int) []Document {
	type scored struct {
		doc   Document
		score float64
	}
	rrfScores := make(map[string]*scored)

	for _, docs := range resultSets {
		for rank, doc := range docs {
			score := 1.0 / float64(k+rank+1)
			if s, ok := rrfScores[doc.ID]; ok {
				s.score += score
			} else {
				rrfScores[doc.ID] = &scored{doc: doc, score: score}
			}
		}
	}

	sorted := make([]*scored, 0, len(rrfScores))
	for _, s := range rrfScores {
		sorted = append(sorted, s)
	}
	// 按 RRF 分数降序排序
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].score > sorted[i].score {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	result := make([]Document, len(sorted))
	for i, s := range sorted {
		result[i] = Document{
			ID:      s.doc.ID,
			Content: s.doc.Content,
			Score:   s.score,
		}
	}
	return result
}

// HybridSearch 混合搜索：稠密向量(pgvector) + 关键字匹配(ILIKE)，RRF 融合
func (r *RAGSystem) HybridSearch(ctx context.Context, queryText string, limit int) (*HybridSearchResult, error) {
	// 1. 用 fastembed 生成稠密查询向量
	queryVector, err := r.fastembed.QueryEmbed(queryText)
	if err != nil {
		return nil, fmt.Errorf("生成查询向量失败: %w", err)
	}

	// 2. 并行执行两路搜索
	var (
		denseDocs, sparseDocs []Document
		denseErr, sparseErr   error
		wg                    sync.WaitGroup
	)
	wg.Add(2)

	go func() {
		defer wg.Done()
		denseDocs, denseErr = r.pg.DenseSearch(ctx, queryVector, limit*2)
	}()
	go func() {
		defer wg.Done()
		sparseDocs, sparseErr = r.pg.SparseSearch(ctx, queryText, limit*2)
	}()
	wg.Wait()

	if denseErr != nil {
		return nil, fmt.Errorf("稠密向量搜索失败: %w", denseErr)
	}
	if sparseErr != nil {
		return nil, fmt.Errorf("关键字搜索失败: %w", sparseErr)
	}

	// 3. RRF 融合
	fused := rrfFusion([][]Document{denseDocs, sparseDocs}, 60)

	// 限制返回数量
	if len(fused) > limit {
		fused = fused[:limit]
	}

	return &HybridSearchResult{
		FusedDocs:  fused,
		DenseDocs:  denseDocs,
		SparseDocs: sparseDocs,
	}, nil
}

// ============ 交互式界面 ============
type InteractiveUI struct {
	rag          *RAGSystem
	cancelFunc   context.CancelFunc
	ctx          context.Context
	isGenerating bool
	mu           sync.Mutex
}

func NewInteractiveUI(rag *RAGSystem) *InteractiveUI {
	return &InteractiveUI{
		rag: rag,
	}
}

func (ui *InteractiveUI) showWaitingAnimation(stopCh <-chan struct{}) {
	frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	i := 0
	ticker := time.NewTicker(80 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-stopCh:
			fmt.Print("\r")
			return
		case <-ticker.C:
			fmt.Printf("\r🤔 思考中... %s ", frames[i%len(frames)])
			i++
		}
	}
}

func (ui *InteractiveUI) handleUserInput() {
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("🤖 RAG 增强搜索系统 (混合搜索: 稠密向量+关键字)")
	fmt.Printf("📚 对话模型: %s | 向量模型: fastembed (BGE-small-zh)\n", chatModel)
	fmt.Println("🗄️  向量数据库: PostgreSQL + pgvector")
	fmt.Println("🔍 搜索方式: 稠密向量(pgvector) + 关键字匹配 (RRF融合)")
	fmt.Println("💡 提示: 输入问题开始搜索，按 Ctrl+C 中断回答")
	fmt.Println("📝 输入 'exit' 或 'quit' 退出")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println()

	for {
		fmt.Print("👤 你: ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if strings.ToLower(input) == "exit" || strings.ToLower(input) == "quit" {
			fmt.Println("👋 再见！")
			break
		}

		// 创建可取消的上下文
		ctx, cancel := context.WithCancel(context.Background())
		ui.mu.Lock()
		if ui.cancelFunc != nil {
			ui.cancelFunc()
		}
		ui.cancelFunc = cancel
		ui.ctx = ctx
		ui.isGenerating = true
		ui.mu.Unlock()

		// 启动等待动画
		stopAnimation := make(chan struct{})
		go ui.showWaitingAnimation(stopAnimation)

		fmt.Print("\n🤖 助手: ")

		chunkChan := make(chan string, 10)
		doneChan := make(chan bool)
		errChan := make(chan error, 1)

		go func() {
			_, err := ui.rag.Ask(ctx, input, func(chunk string) {
				select {
				case chunkChan <- chunk:
				case <-ctx.Done():
					return
				}
			})
			if err != nil {
				errChan <- err
			}
			doneChan <- true
		}()

		var fullResponse strings.Builder
		responseDone := false
		isFirstChunk := true

		for !responseDone {
			select {
			case chunk := <-chunkChan:
				if isFirstChunk {
					close(stopAnimation)
					<-stopAnimation
					isFirstChunk = false
				}
				fmt.Print(chunk)
				fullResponse.WriteString(chunk)

			case <-doneChan:
				responseDone = true
				if isFirstChunk {
					close(stopAnimation)
					<-stopAnimation
				}
				fmt.Println()

			case err := <-errChan:
				if err == context.Canceled {
					fmt.Println("\n⏹️  回答已中断")
				} else if err != nil {
					fmt.Printf("\n❌ 错误: %v\n", err)
				}
				responseDone = true
				if isFirstChunk {
					close(stopAnimation)
					<-stopAnimation
				}

			case <-ctx.Done():
				fmt.Println("\n⏹️  正在中断回答...")
				responseDone = true
				if isFirstChunk {
					close(stopAnimation)
					<-stopAnimation
				}
			}
		}

		ui.mu.Lock()
		ui.isGenerating = false
		ui.mu.Unlock()
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "\n❌ 读取输入错误: %v\n", err)
	}
}

// ============ 主函数 ============
func main() {
	// 检查Ollama是否运行
	if err := checkService(ollamaBaseURL, "Ollama"); err != nil {
		fmt.Printf("❌ Ollama未运行: %v\n", err)
		fmt.Println("请先启动: ollama serve")
		os.Exit(1)
	}

	// 创建Ollama客户端（仅用于流式对话）
	ollama := NewOllamaClient(ollamaBaseURL)

	// 初始化FastEmbed本地向量模型
	fmt.Println("🔧 正在加载FastEmbed向量模型 (BGE-small-zh)...")
	embedModel, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model:    fastembed.BGESmallZH,
		CacheDir: fastembedCacheDir,
	})
	if err != nil {
		fmt.Printf("❌ 初始化FastEmbed失败: %v\n", err)
		os.Exit(1)
	}
	defer embedModel.Destroy()
	fmt.Println("✅ FastEmbed向量模型加载完成 (512维)")

	// 连接 PostgreSQL
	fmt.Println("🔌 正在连接 PostgreSQL...")
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, pgConnString)
	if err != nil {
		fmt.Printf("❌ 连接PostgreSQL失败: %v\n", err)
		os.Exit(1)
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		fmt.Printf("❌ PostgreSQL Ping失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✅ PostgreSQL 连接成功")

	pg := &PostgresClient{pool: pool}

	// 创建RAG系统
	rag := NewRAGSystem(ollama, embedModel, pg, chatModel)

	// 启动交互界面
	ui := NewInteractiveUI(rag)
	ui.handleUserInput()
}

// 检查服务是否运行（HTTP 健康检查）
func checkService(baseURL, name string) error {
	resp, err := http.Get(baseURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 500 {
		return fmt.Errorf("%s 返回状态码: %d", name, resp.StatusCode)
	}
	return nil
}
