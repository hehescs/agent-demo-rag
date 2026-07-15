package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/anush008/fastembed-go"
)

// ============ 配置 ============
const (
	// FastEmbed 模型缓存目录（相对于项目根目录）
	fastembedCacheDir = "../local_cache"

	// 向量维度（BGE-small-zh = 512 维）
	denseVectorSize = 512
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "用法: go run vector_store/embed/text_to_vector.go <文本>")
		fmt.Fprintln(os.Stderr, "示例: go run vector_store/embed/text_to_vector.go \"洗地机推荐\"")
		os.Exit(1)
	}

	text := strings.Join(os.Args[1:], " ")

	// 加载模型
	fmt.Printf("🔧 加载模型 (BGE-small-zh, %d维)...\n", denseVectorSize)
	model, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model:    fastembed.BGESmallZH,
		CacheDir: fastembedCacheDir,
	})
	if err != nil {
		log.Fatal("初始化模型失败: ", err)
	}
	defer model.Destroy()

	// 生成查询向量
	vec, err := model.QueryEmbed(text)
	if err != nil {
		log.Fatal("生成向量失败: ", err)
	}

	// 输出 pgvector 格式字符串，可直接用于 SQL INSERT
	fmt.Printf("\n📝 输入文本: %q\n", text)
	fmt.Printf("📐 向量维度: %d\n", len(vec))
	fmt.Printf("\n📦 pgvector 格式（可直接用于 SQL）:\n\n")
	fmt.Println(formatVector(vec))
}

// formatVector 将 float32 向量转换为 pgvector 字符串格式：'[v1,v2,...]'
func formatVector(vec []float32) string {
	var sb strings.Builder
	sb.Grow(len(vec) * 10) // 预分配，减少扩容
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
