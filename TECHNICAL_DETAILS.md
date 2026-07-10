# Agent Demo RAG — 技术细节文档

## 1. 项目概述

本项目是一个**基于 Go 语言的 RAG（Retrieval-Augmented Generation，检索增强生成）系统**。通过本地部署的 Ollama 服务调用大语言模型（`qwen3:8b`），实现了从知识文档向量化、语义检索到流式问答的完整 RAG 流程，并提供终端交互式界面。

### 技术栈一览

| 层面 | 技术选型 |
|------|----------|
| 编程语言 | Go 1.26 |
| LLM 运行时 | Ollama（本地部署，默认端口 11434） |
| LLM 模型 | `qwen3:8b` |
| 向量存储 | 纯内存实现（无外部数据库） |
| HTTP 通信 | Go 标准库 `net/http`（手写 REST 调用） |
| 并发模型 | goroutine + channel + `sync.RWMutex` |

---

## 2. 系统架构

项目采用**单文件架构**（`main.go`），内部划分为 4 个逻辑模块：

```
┌─────────────────────────────────────────────────┐
│                 InteractiveUI                    │
│   终端交互 / 流式输出 / 等待动画 / 中断控制       │
├─────────────────────────────────────────────────┤
│                   RAGSystem                      │
│   知识库准备 / 提问编排 (Embed → Retrieve → Gen)  │
├──────────────────┬──────────────────────────────┤
│   VectorStore    │       OllamaClient            │
│  内存向量存储     │   Embedding API / Chat API    │
│  余弦相似度检索   │   流式 JSON 解码              │
└──────────────────┴──────────────────────────────┘
```

---

## 3. 核心模块详解

### 3.1 VectorStore — 内存向量存储

#### 数据结构

```go
type VectorStore struct {
    mu         sync.RWMutex   // 读写锁，保证并发安全
    documents  []Document     // 文档切片
    maxResults int            // 默认最大返回数：3
}

type Document struct {
    ID        string
    Content   string
    Embedding []float64       // 向量表示
}
```

#### 检索算法

采用**余弦相似度（Cosine Similarity）**度量向量间的语义距离：

```
similarity(A, B) = dot(A, B) / (‖A‖ × ‖B‖)
                 = Σ(Ai × Bi) / (√Σ(Ai²) × √Σ(Bi²))
```

- 值域 `[0, 1]`，值越大表示语义越相近
- 对零向量做了防御处理（返回 0）
- 对维度不一致的向量做了防御处理（返回 0）

#### 排序与截断

检索时对全部文档计算相似度后，通过冒泡排序按降序排列，再截取 Top-K 条结果返回。

#### 并发安全

- `Add()` 使用写锁（`Lock`）
- `Search()` 使用读锁（`RLock`）
- 支持多读单写的并发访问模式

---

### 3.2 OllamaClient — LLM HTTP 客户端

封装了 Ollama 本地服务的两个核心 REST API：

#### Embedding API（文本向量化）

| 项目 | 详情 |
|------|------|
| 端点 | `POST /api/embed` |
| 请求体 | `{"model": "qwen3:8b", "prompt": "待向量化的文本"}` |
| 响应体 | `{"embedding": [0.123, ...]}` |
| 用途 | 将文档和用户问题转换为高维向量 |

#### Chat API（流式对话）

| 项目 | 详情 |
|------|------|
| 端点 | `POST /api/chat` |
| 请求体 | `{"model": "qwen3:8b", "messages": [...], "stream": true}` |
| 响应格式 | 逐行 JSON，每行包含 `message.content` 和 `done` 标志 |
| 解码方式 | `json.NewDecoder` 流式逐条解码 |
| 超时 | HTTP 客户端全局超时 120 秒 |
| 取消 | 支持 `context.Context` 取消 |

#### 流式解码流程

```
HTTP Response → json.Decoder → 循环 Decode
  ├── chunk.Message.Content ≠ "" → 回调 onChunk()
  ├── chunk.Done == true → 返回 nil（结束）
  └── err == io.EOF → 返回 nil（连接关闭）
```

---

### 3.3 RAGSystem — RAG 核心编排

#### Ask() 方法的四步流程

```
Step 1: 向量化      用户问题 → Ollama Embedding API → 问题向量
Step 2: 语义检索    问题向量 → VectorStore.Search() → Top-3 相关文档
Step 3: Prompt 构建  系统提示 + 检索文档 + 用户问题 → 完整 Prompt
Step 4: 流式生成    Prompt → Ollama Chat API → 逐 token 流式输出
```

#### Prompt 模板

```
请基于以下资料回答用户问题。如果资料中没有相关信息，请明确说明。

【资料 1】
{文档内容1}

【资料 2】
{文档内容2}

【资料 3】
{文档内容3}

用户问题：{用户输入}

回答（请用通俗易懂的语言，基于以上资料）：
```

#### 角色设定

- **System Message**: "你是一个知识渊博的助手，善于用通俗易懂的语言解释专业问题。"
- 引导模型严格基于检索到的资料回答，避免幻觉

---

### 3.4 InteractiveUI — 交互式终端界面

#### 并发架构

```
main goroutine          ─── 用户输入循环
  ├── animation goroutine ─── Braille 字符旋转动画（80ms 刷新）
  ├── query goroutine     ─── RAG 问答（Ask 调用）
  └── select loop         ─── chunkChan / doneChan / errChan / ctx.Done()
```

#### 等待动画

使用 Braille 字符实现终端旋转效果：

```
⠋ → ⠙ → ⠹ → ⠸ → ⠼ → ⠴ → ⠦ → ⠧ → ⠇ → ⠏
```

每帧间隔 80ms，首个 chunk 到达时自动停止动画并切换到流式文本输出。

#### 中断与取消机制

- 每次用户提问创建新的 `context.WithCancel`
- 新提问自动取消上一次未完成的请求（`cancelFunc()` 覆盖调用）
- 支持 `Ctrl+C` 中断当前回答
- 通过 `isGenerating` 标志 + `sync.Mutex` 跟踪生成状态

#### 通道设计

| 通道 | 缓冲大小 | 用途 |
|------|----------|------|
| `chunkChan` | 10 | 传递 LLM 生成的文本片段 |
| `doneChan` | 1 | 标记问答完成 |
| `errChan` | 1 | 传递错误信息 |
| `stopAnimation` | 0 | 通知动画 goroutine 停止 |

---

## 4. 启动流程

```
main()
  ├── checkOllama()          → GET /api/tags 检查 Ollama 服务是否运行
  ├── NewOllamaClient()      → 创建 HTTP 客户端（baseURL: 127.0.0.1:11434）
  ├── NewRAGSystem()         → 初始化 RAG 系统（模型: qwen3:8b）
  ├── PrepareKnowledgeBase() → 逐条文档调用 Embedding API → 存入 VectorStore
  └── InteractiveUI.Run()    → 启动交互式问答循环
```

---

## 5. 知识库

### 当前配置

代码中当前仅加载 **1 条知识**：

```
"猫喜欢看瑞克和莫蒂的动画片。"
```

### 备份知识（`knowledge/docs.txt`）

文件中包含 5 条 Go 语言相关知识，但代码中未读取该文件：

1. Go 语言基础（Google 开发、静态强类型、编译型、并发型、GC）
2. goroutine（轻量级线程、Go 运行时管理）
3. 垃圾回收（自动内存管理）
4. CSP 并发模型（goroutine + channel）
5. 设计目标（简单、高效、可靠、大规模分布式系统）

### 原始硬编码知识（已注释）

代码中注释掉了 10 条 Go 语言知识，涵盖：语言特性、goroutine、GC、CSP、编译速度、标准库、go mod、channel 等。

---

## 6. 依赖说明

`go.mod` 声明了以下依赖：

| 依赖 | 版本 | 实际使用情况 |
|------|------|-------------|
| `github.com/ollama/ollama` | v0.31.2 | ⚠️ 已声明但代码中**未 import**，实际通过 `net/http` 手写调用 |
| `github.com/google/uuid` | v1.6.0 | 间接依赖（未直接使用） |
| `golang.org/x/crypto` | v0.43.0 | 间接依赖 |
| `gopkg.in/yaml.v3` | v3.0.1 | 间接依赖 |

> **注意**: Ollama SDK 依赖未实际使用，可考虑清理。

---

## 7. 技术特点与改进建议

### 当前特点

| 特点 | 说明 |
|------|------|
| ✅ 零外部数据库依赖 | 纯内存向量存储，启动即用 |
| ✅ 流式输出 | 逐 token 实时打印，用户体验好 |
| ✅ 请求可中断 | 基于 context 的取消机制 |
| ✅ 并发安全 | 读写锁保护共享状态 |

### 改进建议

| 问题 | 当前实现 | 建议方案 |
|------|----------|----------|
| 排序效率 | 冒泡排序 O(n²) | 使用 `sort.Slice` O(n log n) |
| 文档分块 | 整条文档作为一个 chunk | 引入文本分块（按段落/句子/滑动窗口） |
| 向量持久化 | 纯内存，重启丢失 | 接入 SQLite-vec / Milvus / Chroma 等 |
| 维度校验 | 仅在计算时检查 | 存储时统一维度校验 |
| 知识加载 | 硬编码在代码中 | 从文件/目录动态加载（如 `knowledge/docs.txt`） |
| 未使用的依赖 | `ollama` SDK 已声明未 import | `go mod tidy` 清理 |
| 健康检查 | 仅检查 HTTP 连通性 | 验证响应状态码和模型可用性 |
| 检索策略 | 固定 Top-3 | 支持相似度阈值过滤 |
