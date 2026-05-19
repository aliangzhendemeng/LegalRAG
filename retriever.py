import os
import shutil
from typing import List, Optional

import jieba
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PDF_FILE, CHROMA_DIR, EMBEDDING_MODEL, RERANKER_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_MODE,
    TOP_K_RECALL, TOP_K_RERANK, RRF_K,
    USE_RERANKER, FORCE_REBUILD_INDEX,
)


def _load_and_split(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n第", "\n\n", "\n", "。", "；", " ", ""],
    )
    splits = splitter.split_documents(raw_docs)
    for i, d in enumerate(splits):
        d.metadata["chunk_id"] = i
    return splits


def _chinese_tokenizer(text: str) -> List[str]:
    return [w for w in jieba.lcut(text) if w.strip()]


class HybridRetriever:
    """支持 hybrid / dense / bm25 / tfidf 四种模式，可选 bge-reranker 精排。"""

    def __init__(self, pdf_path: str = PDF_FILE, mode: str = RETRIEVER_MODE):
        if mode not in {"hybrid", "dense", "bm25", "tfidf"}:
            raise ValueError(f"未知检索模式：{mode}")
        self.mode = mode
        print(f"[Retriever] 加载并切分文档：{pdf_path}")
        self.docs = _load_and_split(pdf_path)
        self.texts = [d.page_content for d in self.docs]
        print(f"[Retriever] 切分完成，共 {len(self.docs)} 个片段")

        self._embeddings = None
        self._vector_store = None
        self._bm25 = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._reranker = None

        if mode in ("hybrid", "dense"):
            self._build_dense()
        if mode in ("hybrid", "bm25"):
            self._build_bm25()
        if mode == "tfidf":
            self._build_tfidf()
        if USE_RERANKER and mode != "tfidf":
            self._build_reranker()

    # ---------- 构建 ----------
    def _build_dense(self) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        print(f"[Retriever] 加载 Embedding 模型：{EMBEDDING_MODEL}")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

        if FORCE_REBUILD_INDEX and os.path.exists(CHROMA_DIR):
            print(f"[Retriever] FORCE_REBUILD_INDEX=true，清空 {CHROMA_DIR}")
            shutil.rmtree(CHROMA_DIR)

        has_index = os.path.exists(CHROMA_DIR) and any(os.scandir(CHROMA_DIR))
        if has_index:
            print(f"[Retriever] 复用已有向量库：{CHROMA_DIR}")
            self._vector_store = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self._embeddings,
                collection_name="legal_rag",
            )
        else:
            print(f"[Retriever] 构建新向量库到：{CHROMA_DIR}")
            self._vector_store = Chroma.from_documents(
                documents=self.docs,
                embedding=self._embeddings,
                persist_directory=CHROMA_DIR,
                collection_name="legal_rag",
            )

    def _build_bm25(self) -> None:
        from langchain_community.retrievers import BM25Retriever
        print("[Retriever] 构建 BM25 索引（jieba 中文分词）")
        self._bm25 = BM25Retriever.from_documents(
            self.docs,
            preprocess_func=_chinese_tokenizer,
        )
        self._bm25.k = TOP_K_RECALL

    def _build_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("[Retriever] 构建 TF-IDF 索引（对照组）")
        self._tfidf_vectorizer = TfidfVectorizer(tokenizer=_chinese_tokenizer)
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self.texts)

    def _build_reranker(self) -> None:
        from sentence_transformers import CrossEncoder
        print(f"[Retriever] 加载 Reranker 模型：{RERANKER_MODEL}")
        self._reranker = CrossEncoder(RERANKER_MODEL)

    # ---------- 检索 ----------
    def _dense_search(self, query: str, k: int) -> List[Document]:
        return self._vector_store.similarity_search(query, k=k)

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        self._bm25.k = k
        return self._bm25.invoke(query)

    def _tfidf_search(self, query: str, k: int) -> List[Document]:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._tfidf_matrix)[0]
        idx = np.argsort(sims)[-k:][::-1]
        return [self.docs[i] for i in idx]

    @staticmethod
    def _rrf(rank_lists: List[List[Document]], k_const: int = RRF_K) -> List[Document]:
        scores: dict = {}
        keep: dict = {}
        for ranks in rank_lists:
            for rank, doc in enumerate(ranks):
                cid = doc.metadata.get("chunk_id", id(doc))
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank)
                keep[cid] = doc
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [keep[cid] for cid, _ in ordered]

    def _rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        if not self._reranker or not docs:
            return docs[:k]
        pairs = [[query, d.page_content] for d in docs]
        scores = self._reranker.predict(pairs)
        order = np.argsort(scores)[::-1][:k]
        return [docs[i] for i in order]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        k_final = top_k or TOP_K_RERANK

        if self.mode == "tfidf":
            return self._tfidf_search(query, k_final)

        if self.mode == "dense":
            recalled = self._dense_search(query, TOP_K_RECALL)
        elif self.mode == "bm25":
            recalled = self._bm25_search(query, TOP_K_RECALL)
        else:  # hybrid
            dense = self._dense_search(query, TOP_K_RECALL)
            bm25 = self._bm25_search(query, TOP_K_RECALL)
            recalled = self._rrf([dense, bm25])[:TOP_K_RECALL]

        return self._rerank(query, recalled, k_final)

    @staticmethod
    def format_context(docs: List[Document]) -> str:
        return "\n---\n".join(
            f"[片段{d.metadata.get('chunk_id', '?')}] (page={d.metadata.get('page', '?')})\n{d.page_content}"
            for d in docs
        )
