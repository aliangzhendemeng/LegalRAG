import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# HuggingFace 国内镜像（必须在 import sentence_transformers / huggingface_hub 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ========== 文档与持久化路径 ==========
PDF_FILE = os.path.join(BASE_DIR, os.getenv("PDF_FILE", "劳动合同.pdf"))
CHROMA_DIR = os.path.join(BASE_DIR, ".chroma")

# ========== LLM ==========
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ========== Embedding / Reranker ==========
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# ========== 切分 ==========
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "800"))

# ========== 检索 ==========
# hybrid | dense | bm25 | tfidf
RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "hybrid").lower()
TOP_K_RECALL = int(os.getenv("TOP_K_RECALL", "20"))   # 粗排召回
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))    # 精排保留
RRF_K = int(os.getenv("RRF_K", "60"))                 # RRF 融合常数

# tfidf 模式不走 reranker
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

# 强制重建向量库（PDF 替换后置 true 一次即可）
FORCE_REBUILD_INDEX = os.getenv("FORCE_REBUILD_INDEX", "false").lower() == "true"

# 多轮改写 query（关闭可省一次 LLM 调用）
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
