package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anush008/fastembed-go"
	"golang.org/x/text/encoding/simplifiedchinese"
)

// ============ 配置 ============
const (
	qdrantBaseURL     = "http://localhost:6333"
	qdrantCollection  = "doc"
	fastembedCacheDir = "local_cache"
	docsFile          = "../knowledge/docs.txt"
)

// 稠密向量维度（BGE-small-zh 是 512 维）
const denseVectorSize = 512

func main() {
	// 1. 初始化 FastEmbed 模型（稠密向量）
	fmt.Println("🔧 正在加载 FastEmbed 向量模型 (BGE-small-zh)...")
	denseModel, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model:    fastembed.BGESmallZH,
		CacheDir: fastembedCacheDir,
	})
	if err != nil {
		log.Fatal("初始化模型失败: ", err)
	}
	defer denseModel.Destroy()
	fmt.Println("✅ 模型加载完成 (512维)")

	// 2. 读取文档（自动处理GBK编码）
	data, err := os.ReadFile(docsFile)
	if err != nil {
		log.Fatal("读取文档文件失败: ", err)
	}

	// 如果不是合法UTF-8，按GBK解码
	content := string(data)
	if !utf8.Valid(data) {
		decoded, decErr := simplifiedchinese.GBK.NewDecoder().Bytes(data)
		if decErr != nil {
			log.Fatal("GBK解码失败: ", decErr)
		}
		content = string(decoded)
		fmt.Println("⚠️  检测到非UTF-8编码，已按GBK解码")
	}

	var documents []string
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			documents = append(documents, line)
		}
	}
	if len(documents) == 0 {
		log.Fatal("文档为空，没有可导入的内容")
	}
	fmt.Printf("📄 读取到 %d 条文档\n", len(documents))

	// ===== 修改点1: 创建集合时支持混合搜索 =====
	if err := createHybridCollection(); err != nil {
		log.Fatal("创建集合失败: ", err)
	}

	// 3. 生成稠密向量
	const embedBatchSize = 32
	fmt.Println("🔄 正在生成稠密向量...")
	var allDenseEmbeddings [][]float32
	for i := 0; i < len(documents); i += embedBatchSize {
		end := min(i+embedBatchSize, len(documents))
		batch := documents[i:end]
		batchEmb, err := denseModel.PassageEmbed(batch, len(batch))
		if err != nil {
			log.Printf("⚠️  批次 [%d-%d) 生成向量失败，逐条重试...", i, end)
			for j, doc := range batch {
				emb, err2 := denseModel.PassageEmbed([]string{doc}, 1)
				if err2 != nil {
					log.Printf("⚠️  文档 %d 编码失败，跳过: %v", i+j, err2)
					allDenseEmbeddings = append(allDenseEmbeddings, nil)
					continue
				}
				allDenseEmbeddings = append(allDenseEmbeddings, emb...)
			}
			continue
		}
		allDenseEmbeddings = append(allDenseEmbeddings, batchEmb...)
		fmt.Printf("  已生成 %d/%d 条稠密向量\n", len(allDenseEmbeddings), len(documents))
	}
	fmt.Printf("✅ 已生成 %d 个稠密向量\n", len(allDenseEmbeddings))

	// ===== 修改点2: 写入时同时存储稠密和稀疏向量 =====
	if err := upsertHybridToQdrant(documents, allDenseEmbeddings); err != nil {
		log.Fatal("写入 Qdrant 失败: ", err)
	}
	fmt.Printf("🎉 成功将文档写入 Qdrant (集合: %s)\n", qdrantCollection)

	// ===== 修改点3: 测试混合搜索 =====
	fmt.Println("\n🔍 测试混合搜索...")
	if err := hybridSearch("洗地机"); err != nil {
		log.Printf("搜索失败: %v", err)
	}
}

// ===== 新增函数1: 创建支持混合搜索的集合 =====
func createHybridCollection() error {
	// 1. 删除旧集合（如果存在）
	deleteURL := fmt.Sprintf("%s/collections/%s", qdrantBaseURL, qdrantCollection)
	req, _ := http.NewRequest("DELETE", deleteURL, nil)
	client := &http.Client{Timeout: 30 * time.Second}
	resp, _ := client.Do(req)
	if resp != nil {
		resp.Body.Close()
	}

	// 2. 创建新集合（支持稠密+稀疏向量）
	createBody := map[string]any{
		"vectors": map[string]any{
			"dense": map[string]any{
				"size":     denseVectorSize,
				"distance": "Cosine",
			},
		},
		"sparse_vectors": map[string]any{
			"sparse": map[string]any{
				"modifier": "idf", // 启用 IDF 加权
			},
		},
	}

	jsonData, err := json.Marshal(createBody)
	if err != nil {
		return err
	}

	createURL := fmt.Sprintf("%s/collections/%s", qdrantBaseURL, qdrantCollection)
	req, err = http.NewRequest("PUT", createURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err = client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("创建集合失败: %s\n%s", resp.Status, string(body))
	}

	fmt.Println("✅ 集合已创建（支持密集+稀疏向量）")
	return nil
}

// ===== 修改函数2: 混合写入 =====
// 原来: upsertToQdrant
// 现在: upsertHybridToQdrant
func upsertHybridToQdrant(documents []string, denseEmbeddings [][]float32) error {
	// 获取当前最大 ID
	startID, err := getMaxID()
	if err != nil {
		fmt.Printf("⚠️  获取最大ID失败，从1开始: %v\n", err)
		startID = 0
	}

	type Point struct {
		ID      int                    `json:"id"`
		Vector  map[string]interface{} `json:"vector"` // 改为 map，支持多个向量
		Payload map[string]any         `json:"payload"`
	}

	var points []Point
	idCounter := 0
	for i, doc := range documents {
		if denseEmbeddings[i] == nil {
			log.Printf("警告: 文档 %d 稠密向量为空，跳过\n", i)
			continue
		}

		parts := strings.Split(doc, " ") // 注意：你的数据是用 Tab 分隔的，不是空格！
		if len(parts) < 4 {
			log.Printf("警告: 文档 %d 格式错误，跳过\n", i)
			continue
		}

		idCounter++
		pointID := startID + idCounter

		// ===== 关键修改：构建包含两个向量的 Vector =====
		// 1. 稠密向量：直接用生成的向量
		// 2. 稀疏向量：使用 Document 结构，让 Qdrant 服务端用 BM25 生成
		point := Point{
			ID: pointID,
			Vector: map[string]interface{}{
				"dense": denseEmbeddings[i], // 稠密向量
				"sparse": map[string]string{ // 稀疏向量：Qdrant 服务端生成
					"text":  doc,
					"model": "qdrant/bm25",
				},
			},
			Payload: map[string]any{
				"text":       doc,
				"id":         pointID,
				"name":       parts[0],
				"price":      parts[1],
				"category":   parts[2],
				"source":     docsFile,
				"sell_point": parts[3],
			},
		}
		points = append(points, point)
	}

	reqBody := map[string]any{
		"points": points,
		"wait":   true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("序列化请求失败: %w", err)
	}

	url := fmt.Sprintf("%s/collections/%s/points", qdrantBaseURL, qdrantCollection)
	req, err := http.NewRequest("PUT", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Qdrant API错误: %s\n%s", resp.Status, string(body))
	}

	fmt.Printf("✅ 成功写入 %d 条混合向量数据\n", len(points))
	return nil
}

// ===== 新增函数3: 混合搜索 =====
func hybridSearch(queryText string) error {
	// 1. 加载模型生成查询的稠密向量
	denseModel, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model:    fastembed.BGESmallZH,
		CacheDir: fastembedCacheDir,
	})
	if err != nil {
		return err
	}
	defer denseModel.Destroy()

	// 2. 生成查询的稠密向量（QueryEmbed）
	queryDenseVec, err := denseModel.QueryEmbed(queryText)
	if err != nil {
		return err
	}

	// 3. 构建混合搜索请求
	searchBody := map[string]any{
		"prefetch": []map[string]any{
			{
				"query": queryDenseVec[0],
				"using": "dense",
				"limit": 20,
			},
			{
				"query": map[string]string{
					"text":  queryText,
					"model": "qdrant/bm25",
				},
				"using": "sparse",
				"limit": 20,
			},
		},
		"query": map[string]string{
			"fusion": "rrf", // 倒数排名融合
		},
		"limit":        10,
		"with_payload": true,
		"with_vector":  false,
	}

	jsonData, err := json.Marshal(searchBody)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("%s/collections/%s/points/query", qdrantBaseURL, qdrantCollection)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("搜索失败: %s\n%s", resp.Status, string(body))
	}

	// 4. 解析结果
	var result struct {
		Result struct {
			Points []struct {
				ID      int            `json:"id"`
				Score   float64        `json:"score"`
				Payload map[string]any `json:"payload"`
			} `json:"points"`
		} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return err
	}

	fmt.Printf("\n🔍 搜索结果: %d 条\n", len(result.Result.Points))
	for i, point := range result.Result.Points {
		name, _ := point.Payload["name"].(string)
		price, _ := point.Payload["price"].(string)
		fmt.Printf("%d. ID: %d, 相似度: %.4f\n", i+1, point.ID, point.Score)
		fmt.Printf("   名称: %s\n", name)
		fmt.Printf("   价格: %s\n", price)
		fmt.Printf("   类目: %v\n\n", point.Payload["category"])
	}

	return nil
}

// ===== 以下是原有的辅助函数（保持不变）=====

// getMaxID 查询集合中当前最大的 point ID
func getMaxID() (int, error) {
	url := fmt.Sprintf("%s/collections/%s/points/scroll", qdrantBaseURL, qdrantCollection)
	maxID := 0
	limit := 100

	client := &http.Client{Timeout: 30 * time.Second}
	var nextOffset interface{}

	for {
		reqBody := map[string]any{
			"limit":        limit,
			"with_payload": false,
			"with_vector":  false,
		}
		if nextOffset != nil {
			reqBody["offset"] = nextOffset
		}

		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			return 0, err
		}

		req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
		if err != nil {
			return 0, err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			return 0, err
		}

		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			if resp.StatusCode == http.StatusNotFound {
				return 0, nil
			}
			body, _ := io.ReadAll(resp.Body)
			return 0, fmt.Errorf("Qdrant API错误: %s, body: %s", resp.Status, string(body))
		}

		var result struct {
			Result struct {
				Points []struct {
					ID json.RawMessage `json:"id"`
				} `json:"points"`
				NextOffset json.RawMessage `json:"next_offset"`
			} `json:"result"`
		}

		decodeErr := json.NewDecoder(resp.Body).Decode(&result)
		resp.Body.Close()
		if decodeErr != nil {
			return 0, decodeErr
		}

		for _, point := range result.Result.Points {
			if id := parseIntID(point.ID); id > maxID {
				maxID = id
			}
		}

		if result.Result.NextOffset == nil || len(result.Result.Points) < limit {
			break
		}
		nextOffset = result.Result.NextOffset
	}

	if maxID > 0 {
		fmt.Printf("🔍 查询到当前最大ID: %d\n", maxID)
	} else {
		fmt.Println("🔍 集合为空或不存在，将从ID 1开始")
	}

	return maxID, nil
}

// parseIntID 从 json.RawMessage 中解析出整数 ID
func parseIntID(raw json.RawMessage) int {
	if raw == nil {
		return 0
	}
	var intVal int
	if err := json.Unmarshal(raw, &intVal); err == nil {
		return intVal
	}
	return 0
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
