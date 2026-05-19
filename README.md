# LegalRAG 私有合同PDF智能问答工具

## 一、项目简介
基于 DeepSeek 大模型 + 混合检索（BM25 + BGE 稠密向量）+ BGE Reranker 精排的私有合同 PDF 智能问答工具。文档不上传第三方，检索可持久化（Chroma），低幻觉、可溯源，适合法律合同、办公文档等私有知识库快速搭建。

## 二、版本更新 V2.0
本次迭代核心目标：**增强模型能力**（检索 + 生成 + 工程化）

- 🔍 **混合检索升级**：从纯 TF-IDF 升级为 BM25（中文 jieba 分词）+ BGE 稠密向量双路召回，RRF 算法融合，召回更全；
- 🎯 **BGE Reranker 精排**：粗排 20 条 → CrossEncoder 重排取 5 条，命中精度显著提升；
- 💾 **向量库持久化**：接入 Chroma 本地落盘，启动只需加载，无需每次重算向量；
- ♻️ **多轮 Query 改写**：基于历史对话自动消解指代（如「它呢」「上面说的那条」），生成独立完整查询再检索；
- 📑 **引用溯源 Prompt**：要求模型输出 `[片段N]` 编号作为依据，配合切分时分配的 chunk_id，可定位原文位置；
- 🏗️ **架构解耦**：单文件拆为 `config / retriever / chain / main / api` 五个模块，便于后续迭代；
- 🔁 **可切换检索后端**：`RETRIEVER_MODE` 支持 `hybrid | dense | bm25 | tfidf` 四种，可做 A/B 与回归对照；
- ⚡ **API 懒加载**：首次请求才初始化模型与索引，避免 import 阶段卡顿。

历史版本：
- V1.1：通用化 + 多轮对话 + 链路解耦（仅 TF-IDF 检索）
- V1.0：基础 RAG 链路 + 关键词增强

## 三、核心功能
- 读取本地合同 PDF，严格基于文档原文精准问答；
- 多轮对话 + 上下文记忆，多轮指代自动改写为独立 query；
- 本地化 Embedding + Reranker（BGE 系列），国内可直连 hf-mirror.com 拉模型；
- 引用溯源：回答附 `[片段N]` 编号，可与文档片段对照核查；
- 前后端分离 + 简约聊天页面，FastAPI 健康检查端点；
- 检索后端可一键切换，方便对比与回归。

## 四、技术栈
- **开发语言**：Python 3.10+
- **大模型**：DeepSeek Chat（通过 OpenAI 兼容接口）
- **向量化**：BAAI/bge-small-zh-v1.5（本地 sentence-transformers）
- **重排**：BAAI/bge-reranker-base（本地 CrossEncoder）
- **稀疏检索**：BM25 + jieba 中文分词
- **向量库**：Chroma（本地文件持久化）
- **检索融合**：Reciprocal Rank Fusion（RRF）
- **链路框架**：LangChain
- **后端框架**：FastAPI + Uvicorn
- **前端**：原生 HTML

## 五、快速启动指南

### 5.1 安装依赖
```bash
pip install -r requirements.txt
```
> 首次会自动下载 BGE 模型（约 100MB + 280MB），默认走 `https://hf-mirror.com`，国内可直连。

### 5.2 配置 API Key
复制 `.env.example` 重命名为 `.env`，填入 DeepSeek API Key：
```env
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

### 5.3 替换 PDF 文档
将合同 PDF 放入项目根目录，通过环境变量指定（或保留默认 `劳动合同.pdf`）：
```env
PDF_FILE=你的合同.pdf
FORCE_REBUILD_INDEX=true   # 换 PDF 后跑一次，之后改回 false
```

### 5.4 启动后端
```bash
python api.py
```
首次启动会下载模型并构建向量库（约 1-3 分钟），之后启动只需几秒。

### 5.5 启动前端
双击 `index.html` 即可进入聊天页面。

### 5.6 健康检查
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","retriever_mode":"hybrid","reranker":true,...}
```

## 六、可调参数（.env）

| 参数 | 默认 | 说明 |
|---|---|---|
| `RETRIEVER_MODE` | hybrid | 检索模式：hybrid / dense / bm25 / tfidf |
| `USE_RERANKER` | true | 是否启用 BGE Reranker 精排（tfidf 模式自动跳过） |
| `ENABLE_QUERY_REWRITE` | true | 是否启用多轮 query 改写（多耗一次 LLM 调用） |
| `TOP_K_RECALL` | 20 | 粗排召回数量 |
| `TOP_K_RERANK` | 5 | 精排后保留并喂给 LLM 的片段数量 |
| `CHUNK_SIZE` | 4000 | 切分块大小 |
| `CHUNK_OVERLAP` | 800 | 切分块重叠 |
| `FORCE_REBUILD_INDEX` | false | 强制重建向量库（替换 PDF 后用一次） |
| `EMBEDDING_MODEL` | BAAI/bge-small-zh-v1.5 | Embedding 模型 |
| `RERANKER_MODEL` | BAAI/bge-reranker-base | Reranker 模型 |

## 七、项目结构
```text
LegalRAG/
├── config.py          # 全局配置（路径、模型、检索参数、HF 镜像）
├── retriever.py       # 文档加载、切分、四种检索后端、RRF、Reranker
├── chain.py           # RAG 链构建、多轮 query 改写、引用溯源 prompt
├── main.py            # CLI 入口（python main.py）
├── api.py             # FastAPI 后端 + /health 端点
├── index.html         # 前端聊天页面
├── .env               # API Key 与可选配置
├── .env.example       # 配置模板
├── requirements.txt   # 依赖列表
├── .chroma/           # 向量库持久化目录（自动生成，已 gitignore）
└── 合同PDF文件         # 可自行替换任意 PDF
```

## 八、注意事项
- **首次启动较慢**：要下载约 380MB 模型 + 构建向量库；之后启动只需几秒
- **国内网络**：代码已设置 `HF_ENDPOINT=https://hf-mirror.com`，可直连；如仍失败可手动 `huggingface-cli download BAAI/bge-small-zh-v1.5`
- **隐私性**：PDF、Embedding、Reranker 全部本地处理；仅 LLM 推理走 DeepSeek API
- **更换 PDF**：必须设置一次 `FORCE_REBUILD_INDEX=true`，否则会复用旧向量库
- **回归对照**：用 `RETRIEVER_MODE=tfidf` 启动可还原 V1.1 行为，便于对比新旧效果

## 九、检索效果对比建议
| 模式 | 适用场景 |
|---|---|
| `hybrid`（默认） | 综合最优，兼顾关键词命中（如条款编号）与语义匹配（如口语化提问） |
| `dense` | 语义优先，口语化提问表现更好 |
| `bm25` | 精确关键词命中，适合"违约金"、"押金"等条款定位 |
| `tfidf` | V1.1 兼容，仅用于对照基准 |

## 十、后续可扩展方向（Roadmap）

**V2.1 法律领域生成增强**
- Parent-Child 结构化切分（按"第X条/第X章"识别边界）
- JSON 结构化输出（answer + citations[]）
- 法律专业 Prompt + 少量 few-shot 示例

**V2.2 对话与工程化**
- `RunnableWithMessageHistory` 真正的消息历史（按 session_id）
- 文档上传 API（`/upload`）+ 多文档隔离
- SSE 流式输出 + LangChain LLM Cache

**V2.3 评估与观测**
- 自建评估集（30-50 条 Q&A）
- 检索指标（Hit@k、MRR）+ 生成指标（RAGAS faithfulness）
- LangSmith 或本地 Phoenix 链路观测

**进阶**
- 法律实体抽取 → 知识图谱
- Agent 化（法条查询 / 合同对比工具）
- 小模型 LoRA 微调本地兜底
