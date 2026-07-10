package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// ============ 向量存储（内存版） ============
type VectorStore struct {
	mu         sync.RWMutex
	documents  []Document
	maxResults int
}

type Document struct {
	ID        string
	Content   string
	Embedding []float64
}

func NewVectorStore() *VectorStore {
	return &VectorStore{
		documents:  []Document{},
		maxResults: 3,
	}
}

func (vs *VectorStore) Add(id, content string, embedding []float64) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.documents = append(vs.documents, Document{
		ID:        id,
		Content:   content,
		Embedding: embedding,
	})
}

func (vs *VectorStore) Search(queryEmbedding []float64, topK int) []Document {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	if len(vs.documents) == 0 {
		return []Document{}
	}

	type result struct {
		doc   Document
		score float64
	}

	var results []result
	for _, doc := range vs.documents {
		score := cosineSimilarity(queryEmbedding, doc.Embedding)
		results = append(results, result{doc, score})
	}

	// 按相似度排序
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if len(results) > topK {
		results = results[:topK]
	}

	var docs []Document
	for _, r := range results {
		docs = append(docs, r.doc)
	}
	return docs
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ============ Ollama客户端（使用HTTP） ============
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

// 获取嵌入向量
func (c *OllamaClient) Embeddings(ctx context.Context, model, text string) ([]float64, error) {
	url := c.baseURL + "/api/embed"

	reqBody := map[string]interface{}{
		"model":  model,
		"prompt": text,
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

	resp, err := c.httpCli.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama API错误: %s, body: %s", resp.Status, string(body))
	}

	var result struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Embedding, nil
}

// 流式对话
func (c *OllamaClient) ChatStream(ctx context.Context, model string, messages []Message, onChunk func(string)) error {
	url := c.baseURL + "/api/chat"

	reqBody := map[string]interface{}{
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
	ollama      *OllamaClient
	vectorStore *VectorStore
	modelName   string
}

func NewRAGSystem(ollama *OllamaClient, modelName string) *RAGSystem {
	return &RAGSystem{
		ollama:      ollama,
		vectorStore: NewVectorStore(),
		modelName:   modelName,
	}
}

func (r *RAGSystem) PrepareKnowledgeBase(ctx context.Context, docs []string) error {
	fmt.Println("📚 正在准备知识库...")

	for i, doc := range docs {
		embedding, err := r.ollama.Embeddings(ctx, r.modelName, doc)
		if err != nil {
			return fmt.Errorf("生成文档 %d 向量失败: %w", i, err)
		}

		r.vectorStore.Add(fmt.Sprintf("doc_%d", i), doc, embedding)
		fmt.Printf("  ✅ 文档 %d 已加载 (向量维度: %d)\n", i+1, len(embedding))
	}

	fmt.Printf("📚 知识库准备完成，共 %d 个文档\n", len(docs))
	return nil
}

func (r *RAGSystem) Ask(ctx context.Context, question string, onChunk func(string)) (string, error) {
	// Step 1: 生成问题向量
	questionEmbedding, err := r.ollama.Embeddings(ctx, r.modelName, question)
	if err != nil {
		return "", fmt.Errorf("生成问题向量失败: %w", err)
	}

	// Step 2: 搜索相关文档
	relevantDocs := r.vectorStore.Search(questionEmbedding, 3)

	if len(relevantDocs) == 0 {
		return "知识库中没有找到相关信息。", nil
	}

	// Step 3: 构建上下文
	var contextBuilder strings.Builder
	contextBuilder.WriteString("请基于以下资料回答用户问题。如果资料中没有相关信息，请明确说明。\n\n")

	for i, doc := range relevantDocs {
		contextBuilder.WriteString(fmt.Sprintf("【资料 %d】\n%s\n\n", i+1, doc.Content))
	}

	contextBuilder.WriteString(fmt.Sprintf("用户问题：%s\n\n", question))
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
	fmt.Println("📚 模型: qwen3:8b")
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
				fmt.Println("\n")

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
}

// ============ 主函数 ============
func main() {
	// 检查Ollama是否运行
	if err := checkOllama(); err != nil {
		fmt.Printf("❌ Ollama未运行: %v\n", err)
		fmt.Println("请先启动: ollama serve")
		os.Exit(1)
	}

	// 创建Ollama客户端
	ollama := NewOllamaClient("http://127.0.0.1:11434")

	// 创建RAG系统
	rag := NewRAGSystem(ollama, "qwen3:8b")
	ctx := context.Background()

	// // 准备知识库
	// knowledgeBase := []string{
	// 	"Go语言是由Google开发的静态强类型、编译型、并发型编程语言，具有垃圾回收功能。",
	// 	"goroutine是Go语言中的轻量级线程，由Go运行时管理。它比操作系统线程更轻量，创建成本很低，可以同时运行成千上万个。",
	// 	"Go语言支持垃圾回收（GC），自动管理内存分配和释放，开发者不需要手动管理内存。",
	// 	"Go的并发模型基于CSP（Communicating Sequential Processes）理论，通过goroutine和channel实现并发编程。",
	// 	"Go语言的设计目标是提供一种简单、高效、可靠的编程语言，特别适合构建大规模分布式系统。",
	// 	"Go语言的编译速度非常快，可以快速将代码编译成单一的可执行文件，便于部署。",
	// 	"Go标准库提供了丰富的网络编程支持，包括HTTP服务器、客户端、WebSocket等。",
	// 	"Go的包管理工具go mod支持模块化开发，可以方便地管理项目依赖。",
	// 	"Go语言中通过关键字go来启动一个goroutine，例如：go func() { fmt.Println('hello') }()。",
	// 	"Go的channel是goroutine之间通信的主要方式，可以分为有缓冲和无缓冲两种类型。",
	// }

	// 准备知识库
	knowledgeBase := []string{
		"猫喜欢看瑞克和莫蒂的动画片。",
	}

	if err := rag.PrepareKnowledgeBase(ctx, knowledgeBase); err != nil {
		fmt.Printf("❌ 准备知识库失败: %v\n", err)
		os.Exit(1)
	}

	// 启动交互界面
	ui := NewInteractiveUI(rag)
	ui.handleUserInput()
}

// 检查Ollama是否运行
func checkOllama() error {
	resp, err := http.Get("http://127.0.0.1:11434/api/tags")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}
