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

func main() {
	// 1. 初始化 FastEmbed 模型
	fmt.Println("🔧 正在加载 FastEmbed 向量模型 (BGE-small-zh)...")
	model, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model:    fastembed.BGESmallZH,
		CacheDir: fastembedCacheDir,
	})
	if err != nil {
		log.Fatal("初始化模型失败: ", err)
	}
	defer model.Destroy()
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

	// 3. 生成向量（PassageEmbed 用于文档入库，会自动加 "passage: " 前缀）
	fmt.Println("🔄 正在生成向量...")
	embeddings, err := model.PassageEmbed(documents, len(documents))
	if err != nil {
		log.Fatal("生成向量失败: ", err)
	}
	fmt.Printf("✅ 已生成 %d 个向量 (维度: %d)\n", len(embeddings), len(embeddings[0]))

	// 4. 写入 Qdrant
	if err := upsertToQdrant(documents, embeddings); err != nil {
		log.Fatal("写入 Qdrant 失败: ", err)
	}
	fmt.Printf("🎉 成功将 %d 条文档写入 Qdrant (集合: %s)\n", len(documents), qdrantCollection)
}

// upsertToQdrant 将文档和对应向量写入 Qdrant
func upsertToQdrant(documents []string, embeddings [][]float32) error {
	// 获取当前最大 ID，避免覆盖已有数据
	startID, err := getMaxID()
	if err != nil {
		fmt.Printf("⚠️  获取最大ID失败，从1开始: %v\n", err)
		startID = 0
	}

	type Point struct {
		ID      int            `json:"id"`
		Vector  []float32      `json:"vector"`
		Payload map[string]any `json:"payload"`
	}

	var points []Point
	for i, doc := range documents {
		points = append(points, Point{
			ID:     startID + i + 1,
			Vector: embeddings[i],
			Payload: map[string]any{
				"text":     doc,
				"id":       startID + i + 1,
				"category": "default",
				"source":   docsFile,
			},
		})
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

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Qdrant API错误: %s\n%s", resp.Status, string(body))
	}

	return nil
}

// getMaxID 查询集合中当前最大的 point ID
// 通过滚动遍历所有点来找到最大ID
func getMaxID() (int, error) {
	url := fmt.Sprintf("%s/collections/%s/points/scroll", qdrantBaseURL, qdrantCollection)
	maxID := 0
	limit := 100 // 每次获取100个点

	client := &http.Client{Timeout: 30 * time.Second}

	// 首次请求不带 offset，后续使用上一次返回的 next_offset 作为游标
	var nextOffset interface{}

	for {
		reqBody := map[string]any{
			"limit":        limit,
			"with_payload": false,
			"with_vector":  false,
		}
		// 只有非首次请求才带 offset（Qdrant scroll 的 offset 是点 ID 游标，不是数字偏移）
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
			// 如果是集合不存在（404），返回0
			if resp.StatusCode == http.StatusNotFound {
				return 0, nil
			}
			body, _ := io.ReadAll(resp.Body)
			return 0, fmt.Errorf("Qdrant API错误: %s, body: %s", resp.Status, string(body))
		}

		// 使用 json.RawMessage 接收 ID 和 next_offset，兼容 int / string(UUID) 两种类型
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

		// 遍历当前批次的所有点，解析 ID 并找到最大值
		for _, point := range result.Result.Points {
			if id := parseIntID(point.ID); id > maxID {
				maxID = id
			}
		}

		// 没有更多点则退出
		if result.Result.NextOffset == nil || len(result.Result.Points) < limit {
			break
		}

		// 将 next_offset 原样保留作为下次请求的游标（可能是 int 或 string）
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
// 兼容 Qdrant 返回 int 或 string(UUID) 两种格式
func parseIntID(raw json.RawMessage) int {
	if raw == nil {
		return 0
	}
	// 尝试直接解析为 int
	var intVal int
	if err := json.Unmarshal(raw, &intVal); err == nil {
		return intVal
	}
	// 尝试解析为 string（UUID 格式），取哈希值或忽略
	// UUID 无法转为有意义的递增整数，此时返回 0 表示无法比较
	return 0
}
