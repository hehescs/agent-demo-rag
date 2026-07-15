package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/anush008/fastembed-go"
	"github.com/jackc/pgx/v5"
	"golang.org/x/text/encoding/simplifiedchinese"
)

// ============ 配置 ============
const (
	// PostgreSQL 连接信息（请根据实际情况修改用户名、密码、数据库名）
	pgConnString = "postgres://postgres:123456@localhost:5432/postgres"

	// FastEmbed 模型缓存目录（与 vector_transfer.go 共享）
	fastembedCacheDir = "../local_cache"

	// 文档文件路径（相对于运行目录）
	docsFile = "../../knowledge/docs.txt"

	// 向量维度（BGE-small-zh = 512 维）
	denseVectorSize = 512

	// 嵌入批量大小
	embedBatchSize = 32
)

// DocRecord 文档记录结构体
type DocRecord struct {
	Name      string
	Price     float32
	Category  string
	SellPoint string
	Source    string
	Text      string // 原始完整文本
}

func main() {
	ctx := context.Background()

	// 1. 初始化 FastEmbed 向量模型（稠密向量，512维）
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

	// 2. 读取文档（自动处理 GBK 编码）
	data, err := readDocsFile(docsFile)
	if err != nil {
		log.Fatal("读取文档文件失败: ", err)
	}

	content := string(data)

	// 解析文档行（跳过表头，按 Tab 分隔）
	var documents []string
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// 跳过表头行
		if i == 0 && strings.Contains(line, "商品名称") {
			fmt.Println("⏭️  已跳过表头行")
			continue
		}
		documents = append(documents, line)
	}

	if len(documents) == 0 {
		log.Fatal("文档为空，没有可导入的内容")
	}
	fmt.Printf("📄 读取到 %d 条文档\n", len(documents))

	// 解析每条文档为结构体（用于提取各字段）
	var docRecords []DocRecord
	for _, doc := range documents {
		rec, ok := parseDocLine(doc, docsFile)
		if !ok {
			continue
		}
		docRecords = append(docRecords, rec)
	}
	fmt.Printf("📋 有效文档记录: %d 条\n", len(docRecords))
	if len(docRecords) == 0 {
		log.Fatal("没有有效的文档记录")
	}

	// 3. 生成稠密向量（512维）
	fmt.Println("🔄 正在生成稠密向量...")
	var allDenseEmbeddings [][]float32
	// 只用有效记录对应的原文进行嵌入
	validTexts := make([]string, len(docRecords))
	for i, rec := range docRecords {
		validTexts[i] = rec.Text
	}

	for i := 0; i < len(validTexts); i += embedBatchSize {
		end := min(i+embedBatchSize, len(validTexts))
		batch := validTexts[i:end]

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
		fmt.Printf("  已生成 %d/%d 条稠密向量\n", len(allDenseEmbeddings), len(validTexts))
	}
	fmt.Printf("✅ 已生成 %d 个稠密向量\n", len(allDenseEmbeddings))

	// 4. 连接 PostgreSQL 并写入数据
	fmt.Println("\n🔌 正在连接 PostgreSQL...")
	conn, err := pgx.Connect(ctx, pgConnString)
	if err != nil {
		log.Fatal("连接 PostgreSQL 失败: ", err)
	}
	defer conn.Close(ctx)

	// 验证连接
	if err := conn.Ping(ctx); err != nil {
		log.Fatal("PostgreSQL Ping 失败: ", err)
	}
	fmt.Println("✅ PostgreSQL 连接成功")

	// 确保 pgvector 扩展已启用
	if _, err := conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
		log.Printf("⚠️  创建 pgvector 扩展失败（可能已存在或无权限）: %v", err)
	}

	// 获取当前最大 ID
	startID, err := getMaxPGID(ctx, conn)
	if err != nil {
		log.Printf("⚠️  获取最大 ID 失败，从 1 开始: %v", err)
		startID = 0
	}
	fmt.Printf("🔍 当前最大 ID: %d，将从 %d 开始写入\n", startID, startID+1)

	// 批量插入数据
	if err := insertDocsToPG(ctx, conn, docRecords, allDenseEmbeddings, startID); err != nil {
		fmt.Println("打印变量值：", docRecords)
		log.Fatal("写入 PostgreSQL 失败: ", err)
	}

	// 5. 测试余弦相似度搜索
	fmt.Println("\n🔍 测试余弦相似度搜索...")
	if err := searchSimilarPG(ctx, conn, denseModel, "洗地机"); err != nil {
		log.Printf("搜索失败: %v", err)
	}
}

// ============ 文件读取 ============

// readDocsFile 读取文档文件，自动处理 GBK 编码
func readDocsFile(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("读取文件失败: %w", err)
	}

	if utf8.Valid(data) {
		return data, nil
	}

	// 非 UTF-8，尝试 GBK 解码
	decoded, err := simplifiedchinese.GBK.NewDecoder().Bytes(data)
	if err != nil {
		return nil, fmt.Errorf("GBK 解码失败: %w", err)
	}
	fmt.Println("⚠️  检测到非 UTF-8 编码，已按 GBK 解码")
	return decoded, nil
}

// parseDocLine 按空格分隔解析一条文档行，返回 DocRecord
// 数据格式：商品名称（可含空格） SPU优惠价 后台类目 卖点
// 策略：从右往左解析，因为价格（纯数字）是可靠的锚点
//
//	parts[N-1] → 卖点（最后一个空格后的内容）
//	parts[N-2] → 类目
//	parts[N-3] → 价格（最后一个纯数字字段）
//	parts[0..N-4] → 商品名称（用空格拼接回来）
func parseDocLine(line, source string) (DocRecord, bool) {
	parts := strings.Fields(line) // 按任意空白字符切分，自动忽略多余空格
	if len(parts) < 4 {
		log.Printf("⚠️  文档格式错误（字段数不足4），跳过: %s", truncate(line, 50))
		return DocRecord{}, false
	}

	n := len(parts)
	sellPoint := parts[n-1]
	category := parts[n-2]
	// 转换成数字
	price, _ := strconv.ParseFloat(parts[n-3], 32)
	name := strings.Join(parts[:n-3], "\t")

	return DocRecord{
		Name:      name,
		Price:     float32(price),
		Category:  category,
		SellPoint: sellPoint,
		Source:    source,
		Text:      line,
	}, true
}

// ============ PostgreSQL 写入 ============

// getMaxPGID 查询 doc 表中当前最大的 id
func getMaxPGID(ctx context.Context, conn *pgx.Conn) (int, error) {
	var maxID int
	err := conn.QueryRow(ctx, "SELECT COALESCE(MAX(id), 0)::int FROM doc").Scan(&maxID)
	return maxID, err
}

// insertDocsToPG 批量将文档和向量写入 PostgreSQL doc 表
func insertDocsToPG(
	ctx context.Context,
	conn *pgx.Conn,
	records []DocRecord,
	embeddings [][]float32,
	startID int,
) error {
	tx, err := conn.Begin(ctx)
	if err != nil {
		return fmt.Errorf("开启事务失败: %w", err)
	}
	defer func() {
		// 如果还没提交，回滚事务（忽略已提交时的错误）
		_ = tx.Rollback(ctx)
	}()

	const insertSQL = `
		INSERT INTO doc (id, name, category, price, sell_point, source, text, embedding)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`

	insertCount := 0
	for i, rec := range records {
		if i >= len(embeddings) || embeddings[i] == nil {
			log.Printf("⚠️  文档 %d 向量为空，跳过", i)
			continue
		}

		newID := startID + 1 + insertCount
		embeddingStr := formatEmbedding(embeddings[i])

		_, err = tx.Exec(ctx, insertSQL,
			newID,
			rec.Name,
			rec.Category,
			rec.Price,
			rec.SellPoint,
			rec.Source,
			rec.Text,
			embeddingStr, // pgvector 接受字符串格式 '[v1, v2, ...]'
		)
		if err != nil {
			return fmt.Errorf("插入第 %d 条文档 (ID=%d) 失败: %w", i, newID, err)
		}

		insertCount++
		if insertCount%50 == 0 || insertCount == len(records) {
			fmt.Printf("  已写入 %d/%d 条文档\n", insertCount, len(records))
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("提交事务失败: %w", err)
	}

	fmt.Printf("✅ 成功写入 %d 条文档到 PostgreSQL (doc 表)\n", insertCount)
	return nil
}

// ============ 余弦相似度搜索 ============

// searchSimilarPG 使用 pgvector 的余弦距离操作符 <=> 进行相似度搜索
func searchSimilarPG(
	ctx context.Context,
	conn *pgx.Conn,
	model *fastembed.FlagEmbedding,
	queryText string,
) error {
	// 生成查询向量（QueryEmbed 会对文本做查询侧预处理）
	queryVec, err := model.QueryEmbed(queryText)
	if err != nil {
		return fmt.Errorf("生成查询向量失败: %w", err)
	}
	queryEmbeddingStr := formatEmbedding(queryVec)

	// 使用 <=> 余弦距离操作符，ORDER BY ASC 得到最相似的结果
	const searchSQL = `
		SELECT id, name, price, category, text,
		       embedding <=> $1::vector AS distance
		FROM doc
		ORDER BY distance ASC
		LIMIT 10
	`

	rows, err := conn.Query(ctx, searchSQL, queryEmbeddingStr)
	if err != nil {
		return fmt.Errorf("执行搜索 SQL 失败: %w", err)
	}
	defer rows.Close()

	fmt.Printf("\n🔍 查询词: %q\n", queryText)
	fmt.Println(strings.Repeat("─", 60))

	var count int
	for rows.Next() {
		var (
			id       int
			name     string
			price    string
			category string
			text     string
			distance float64
		)
		if err := rows.Scan(&id, &name, &price, &category, &text, &distance); err != nil {
			return fmt.Errorf("扫描结果行失败: %w", err)
		}
		count++
		similarity := 1.0 - distance // 转为相似度分数
		fmt.Printf("%d. ID: %d | 相似度: %.4f\n", count, id, similarity)
		fmt.Printf("   名称: %s\n", name)
		fmt.Printf("   类目: %s\n", category)
		fmt.Printf("   价格: %s\n", price)
		fmt.Printf("   卖点: %s\n", truncate(text, 80))
		fmt.Println()
	}

	if count == 0 {
		fmt.Println("⚠️  未找到任何匹配结果（doc 表可能为空）")
	} else {
		fmt.Printf("共找到 %d 条相似文档\n", count)
	}

	return rows.Err()
}

// ============ 辅助函数 ============

// formatEmbedding 将 float32 向量格式化为 pgvector 可识别的字符串：'[v1, v2, ...]'
func formatEmbedding(vec []float32) string {
	var sb strings.Builder
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

// truncate 截断字符串到指定长度（按字节），超长部分用省略号表示
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
