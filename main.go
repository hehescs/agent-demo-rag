package main

import (
	"bufio"
	"bytes"
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
)

// ============ 配置 ============
const (
	ollamaBaseURL     = "http://127.0.0.1:11434"
	qdrantBaseURL     = "http://localhost:6333"
	qdrantCollection  = "doc" // Qdrant 集合名称，按实际修改
	chatModel         = "qwen3:8b"
	fastembedCacheDir = "vector_store/local_cache"
	logFile           = "query_log.txt"

	// 向量字段名称
	denseVectorName  = "dense"  // 稠密向量字段名
	sparseVectorName = "sparse" // 稀疏向量字段名
)

// ============ Qdrant客户端 ============
type QdrantClient struct {
	baseURL    string
	collection string
	httpCli    *http.Client
}

type QdrantDocument struct {
	ID      string
	Content string
	Score   float64
}

func NewQdrantClient(baseURL, collection string) *QdrantClient {
	return &QdrantClient{
		baseURL:    baseURL,
		collection: collection,
		httpCli:    &http.Client{Timeout: 30 * time.Second},
	}
}

// DenseSearch 稠密向量搜索
func (q *QdrantClient) DenseSearch(ctx context.Context, queryVector []float32, limit int) ([]QdrantDocument, error) {
	url := fmt.Sprintf("%s/collections/%s/points/search", q.baseURL, q.collection)

	reqBody := map[string]any{
		"vector": map[string]any{
			"name":   denseVectorName,
			"vector": queryVector,
		},
		"limit":        limit,
		"with_payload": true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("序列化请求失败: %w", err)
	}

	// 调试：打印请求体（可选）
	// fmt.Printf("请求体: %s\n", string(jsonData))

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := q.httpCli.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Qdrant API错误: %s, body: %s", resp.Status, string(body))
	}

	var result struct {
		Result []struct {
			ID      json.RawMessage `json:"id"`
			Payload map[string]any  `json:"payload"`
			Score   float64         `json:"score"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析响应失败: %w", err)
	}

	var docs []QdrantDocument
	for _, r := range result.Result {
		content, _ := r.Payload["text"].(string)
		idStr := strings.Trim(string(r.ID), `"`)
		docs = append(docs, QdrantDocument{
			ID:      idStr,
			Content: content,
			Score:   r.Score,
		})
	}
	return docs, nil
}

// SparseSearch 稀疏向量搜索
func (q *QdrantClient) SparseSearch(ctx context.Context, indices []int, values []float32, limit int) ([]QdrantDocument, error) {
	url := fmt.Sprintf("%s/collections/%s/points/search", q.baseURL, q.collection)

	reqBody := map[string]any{
		"vector": map[string]any{
			"name": sparseVectorName,
			"vector": map[string]any{
				"indices": indices,
				"values":  values,
			},
		},
		"limit":        limit,
		"with_payload": true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("序列化请求失败: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := q.httpCli.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Qdrant API错误: %s, body: %s", resp.Status, string(body))
	}

	var result struct {
		Result []struct {
			ID      json.RawMessage `json:"id"`
			Payload map[string]any  `json:"payload"`
			Score   float64         `json:"score"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析响应失败: %w", err)
	}

	var docs []QdrantDocument
	for _, r := range result.Result {
		content, _ := r.Payload["text"].(string)
		idStr := strings.Trim(string(r.ID), `"`)
		docs = append(docs, QdrantDocument{
			ID:      idStr,
			Content: content,
			Score:   r.Score,
		})
	}
	return docs, nil
}

// ============ Ollama客户端（仅流式对话） ============
type OllamaClient struct {
	baseURL string
	httpCli *http.Client
}

func NewOllamaClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		baseURL: baseURL,
		httpCli: &http.Client{Timeout: 0}, // 流式响应不设超时，由模型生成速度决定
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

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
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
	qdrant    *QdrantClient
	modelName string
}

func NewRAGSystem(ollama *OllamaClient, fastembedModel *fastembed.FlagEmbedding, qdrant *QdrantClient, modelName string) *RAGSystem {
	return &RAGSystem{
		ollama:    ollama,
		fastembed: fastembedModel,
		qdrant:    qdrant,
		modelName: modelName,
	}
}

func (r *RAGSystem) Ask(ctx context.Context, question string, onChunk func(string)) (string, error) {
	queryStart := time.Now()

	// Step 1: 混合搜索（稠密+稀疏+RRF融合）
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
		// {Role: "system", Content: "你是一个知识渊博的助手，善于用通俗易懂的语言解释专业问题。"},
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

// saveQueryLog 将一次完整查询过程记录到日志文件
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

	// 用户输入
	fmt.Fprintf(&b, "【用户输入】\n%s\n\n", question)

	// 稠密向量搜索结果
	fmt.Fprintf(&b, "【稠密向量搜索结果】(共 %d 条)\n", len(searchResult.DenseDocs))
	for i, doc := range searchResult.DenseDocs {
		fmt.Fprintf(&b, "  %d. [ID: %s] (分数: %.4f)\n     %s\n", i+1, doc.ID, doc.Score, truncate(doc.Content, 200))
	}
	b.WriteString("\n")

	// 稀疏向量搜索结果
	fmt.Fprintf(&b, "【稀疏向量搜索结果】(共 %d 条)\n", len(searchResult.SparseDocs))
	for i, doc := range searchResult.SparseDocs {
		fmt.Fprintf(&b, "  %d. [ID: %s] (分数: %.4f)\n     %s\n", i+1, doc.ID, doc.Score, truncate(doc.Content, 200))
	}
	b.WriteString("\n")

	// RRF 融合结果
	fmt.Fprintf(&b, "【RRF 融合结果】(共 %d 条)\n", len(searchResult.FusedDocs))
	for i, doc := range searchResult.FusedDocs {
		fmt.Fprintf(&b, "  %d. [ID: %s] (RRF分数: %.6f)\n     %s\n", i+1, doc.ID, doc.Score, truncate(doc.Content, 200))
	}
	b.WriteString("\n")

	// 发送给大模型的请求
	fmt.Fprintf(&b, "【发送给大模型的请求】\n")
	for _, msg := range messages {
		fmt.Fprintf(&b, "--- %s ---\n%s\n\n", msg.Role, msg.Content)
	}

	// 大模型返回
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
	FusedDocs  []QdrantDocument // 融合后的最终结果
	DenseDocs  []QdrantDocument // 稠密向量搜索结果
	SparseDocs []QdrantDocument // 稀疏向量搜索结果
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

// tokenToIndex 全局 token→index 映射（简单哈希，覆盖足够范围）
var (
	tokenIndexMu sync.Mutex
	tokenIndex   = make(map[string]int)
	nextIndex    = 1
)

func tokenToIndex(token string) int {
	tokenIndexMu.Lock()
	defer tokenIndexMu.Unlock()
	if idx, ok := tokenIndex[token]; ok {
		return idx
	}
	idx := nextIndex
	nextIndex++
	tokenIndex[token] = idx
	return idx
}

// buildSparseVector 根据文本构建稀疏向量（TF 权重）
func buildSparseVector(text string) (indices []int, values []float32) {
	tokens := tokenizeText(text)
	if len(tokens) == 0 {
		return nil, nil
	}
	// 计算词频
	tf := make(map[string]float32)
	for _, t := range tokens {
		tf[t]++
	}
	n := float32(len(tokens))
	for token, count := range tf {
		indices = append(indices, tokenToIndex(token))
		values = append(values, count/n) // TF 归一化
	}
	return
}

// rrfFusion 使用 Reciprocal Rank Fusion 融合多路搜索结果
func rrfFusion(resultSets [][]QdrantDocument, k int) []QdrantDocument {
	type scored struct {
		doc   QdrantDocument
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

	result := make([]QdrantDocument, len(sorted))
	for i, s := range sorted {
		result[i] = QdrantDocument{
			ID:      s.doc.ID,
			Content: s.doc.Content,
			Score:   s.score,
		}
	}
	return result
}

// HybridSearch 混合搜索：稠密向量 + 稀疏向量，RRF 融合
func (r *RAGSystem) HybridSearch(ctx context.Context, queryText string, limit int) (*HybridSearchResult, error) {
	// 1. 用 fastembed 生成稠密查询向量
	queryVector, err := r.fastembed.QueryEmbed(queryText)
	if err != nil {
		return nil, fmt.Errorf("生成查询向量失败: %w", err)
	}

	// 2. 构建稀疏查询向量
	sparseIndices, sparseValues := buildSparseVector(queryText)

	// 3. 并行执行两路搜索
	var (
		denseDocs, sparseDocs []QdrantDocument
		denseErr, sparseErr   error
		wg                    sync.WaitGroup
	)
	wg.Add(2)

	go func() {
		defer wg.Done()
		denseDocs, denseErr = r.qdrant.DenseSearch(ctx, queryVector, limit*2)
	}()
	go func() {
		defer wg.Done()
		if len(sparseIndices) > 0 {
			sparseDocs, sparseErr = r.qdrant.SparseSearch(ctx, sparseIndices, sparseValues, limit*2)
		}
	}()
	wg.Wait()

	if denseErr != nil {
		return nil, fmt.Errorf("稠密向量搜索失败: %w", denseErr)
	}
	if sparseErr != nil {
		return nil, fmt.Errorf("稀疏向量搜索失败: %w", sparseErr)
	}

	// 4. RRF 融合
	fused := rrfFusion([][]QdrantDocument{denseDocs, sparseDocs}, 60)

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
	fmt.Println("🤖 RAG 增强搜索系统 (混合搜索: 稠密+稀疏)")
	fmt.Printf("📚 对话模型: %s | 向量模型: fastembed (BGE-small-zh)\n", chatModel)
	fmt.Printf("🗄️  向量数据库: Qdrant @ %s (集合: %s)\n", qdrantBaseURL, qdrantCollection)
	fmt.Println("🔍 搜索方式: 稠密向量 + 稀疏向量 (RRF融合)")
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

// ============ 辅助函数：创建支持混合搜索的集合 ============
func createHybridCollection(ctx context.Context, qdrant *QdrantClient) error {
	// 创建集合的 API 请求
	url := fmt.Sprintf("%s/collections/%s", qdrant.baseURL, qdrantCollection)

	reqBody := map[string]any{
		"vectors": map[string]any{
			denseVectorName: map[string]any{
				"size":     512,      // BGE-small-zh 的向量维度
				"distance": "Cosine", // 使用余弦相似度
			},
		},
		"sparse_vectors": map[string]any{
			sparseVectorName: map[string]any{
				"modifier": "idf", // 启用 IDF 加权
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "PUT", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := qdrant.httpCli.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("创建集合失败: %s, body: %s", resp.Status, string(body))
	}

	fmt.Println("✅ 混合搜索集合创建成功 (支持稠密+稀疏向量)")
	return nil
}

// ============ 主函数 ============
func main() {
	// 检查Ollama是否运行
	if err := checkService(ollamaBaseURL, "Ollama"); err != nil {
		fmt.Printf("❌ Ollama未运行: %v\n", err)
		fmt.Println("请先启动: ollama serve")
		os.Exit(1)
	}

	// 检查Qdrant是否运行
	if err := checkService(qdrantBaseURL, "Qdrant"); err != nil {
		fmt.Printf("❌ Qdrant未运行: %v\n", err)
		fmt.Println("请先启动Qdrant向量数据库")
		os.Exit(1)
	}

	// 创建Ollama客户端（仅用于流式对话）
	ollama := NewOllamaClient(ollamaBaseURL)

	// 初始化FastEmbed本地向量模型（用于数据入库）
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

	// 创建Qdrant客户端
	qdrant := NewQdrantClient(qdrantBaseURL, qdrantCollection)

	// 提示：确保集合已创建并支持混合搜索
	fmt.Printf("⚠️  请确保集合 '%s' 已配置支持稠密+稀疏向量\n", qdrantCollection)
	fmt.Println("   如果未创建，请运行以下代码或使用Qdrant API创建:")
	fmt.Printf("   - 稠密向量字段: '%s' (512维, Cosine)\n", denseVectorName)
	fmt.Printf("   - 稀疏向量字段: '%s' (BM25)\n", sparseVectorName)
	fmt.Println()

	// 创建RAG系统
	rag := NewRAGSystem(ollama, embedModel, qdrant, chatModel)

	// 启动交互界面
	ui := NewInteractiveUI(rag)
	ui.handleUserInput()
}

// 检查服务是否运行
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
