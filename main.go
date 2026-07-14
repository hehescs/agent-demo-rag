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
	"strings"
	"sync"
	"time"

	"github.com/anush008/fastembed-go"
)

// ============ 配置 ============
const (
	ollamaBaseURL     = "http://127.0.0.1:11434"
	qdrantBaseURL     = "http://localhost:6333"
	qdrantCollection  = "doc" // Qdrant 集合名称，按实际修改
	chatModel         = "qwen3:8b"
	fastembedCacheDir = "vector_store/local_cache"
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

// Search 在Qdrant中搜索最相似的文档
func (q *QdrantClient) Search(ctx context.Context, vector []float64, limit int) ([]QdrantDocument, error) {
	url := fmt.Sprintf("%s/collections/%s/points/search", q.baseURL, q.collection)

	reqBody := map[string]any{
		"vector":       vector,
		"limit":        limit,
		"with_payload": true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
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
		return nil, err
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
		httpCli: &http.Client{Timeout: 120 * time.Second},
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
	// Step 1: 用fastembed生成问题向量
	queryVector, err := r.fastembed.QueryEmbed(question)
	if err != nil {
		return "", fmt.Errorf("生成问题向量失败: %w", err)
	}

	// 将float32转为float64（Qdrant需要float64）
	vectorF64 := make([]float64, len(queryVector))
	for i, v := range queryVector {
		vectorF64[i] = float64(v)
	}

	// Step 2: 在Qdrant中搜索相关文档
	relevantDocs, err := r.qdrant.Search(ctx, vectorF64, 3)
	if err != nil {
		return "", fmt.Errorf("搜索知识库失败: %w", err)
	}

	if len(relevantDocs) == 0 {
		return "知识库中没有找到相关信息。", nil
	}

	// Step 3: 构建上下文
	var contextBuilder strings.Builder
	contextBuilder.WriteString("请基于以下资料回答用户问题。匹配度高的话不要质疑资料的内容。如果资料中没有相关信息，请明确说明。\n\n")

	for i, doc := range relevantDocs {
		fmt.Fprintf(&contextBuilder, "【资料 %d】(相似度: %.4f)\n%s\n\n", i+1, doc.Score, doc.Content)
	}

	fmt.Fprintf(&contextBuilder, "用户问题：%s\n\n", question)
	contextBuilder.WriteString("回答（请用通俗易懂的语言，基于以上资料）：")

	// Step 4: 调用Ollama流式输出
	messages := []Message{
		{Role: "system", Content: "你是一个知识渊博的助手，善于用通俗易懂的语言解释专业问题。"},
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

	return fullResponse.String(), nil
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
	fmt.Println("🤖 RAG 增强搜索系统")
	fmt.Printf("📚 对话模型: %s | 向量模型: fastembed (BGE-small-zh)\n", chatModel)
	fmt.Printf("🗄️  向量数据库: Qdrant @ %s (集合: %s)\n", qdrantBaseURL, qdrantCollection)
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

	// 检查Qdrant是否运行
	if err := checkService(qdrantBaseURL, "Qdrant"); err != nil {
		fmt.Printf("❌ Qdrant未运行: %v\n", err)
		fmt.Println("请先启动Qdrant向量数据库")
		os.Exit(1)
	}

	// 创建Ollama客户端（仅用于流式对话）
	ollama := NewOllamaClient(ollamaBaseURL)

	// 初始化FastEmbed本地向量模型
	fmt.Println("🔧 正在加载FastEmbed向量模型 (BGE-small-zh)...")
	embedModel, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model:    fastembed.BGESmallZH, // 使用中文模型，向量维度通常是 512 维
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
