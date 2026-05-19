from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="法律合同RAG智能问答系统 V2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_chain():
    from config import PDF_FILE, RETRIEVER_MODE
    from retriever import HybridRetriever
    from chain import build_rag_chain
    return build_rag_chain(HybridRetriever(pdf_path=PDF_FILE, mode=RETRIEVER_MODE))


class QueryRequest(BaseModel):
    question: str
    history: str = ""


@app.post("/ask")
def ask(request: QueryRequest):
    try:
        answer = get_chain().invoke({
            "history": request.history,
            "question": request.question,
        })
        return {
            "code": 200,
            "question": request.question,
            "answer": answer,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    from config import RETRIEVER_MODE, USE_RERANKER, EMBEDDING_MODEL, RERANKER_MODEL
    return {
        "status": "ok",
        "retriever_mode": RETRIEVER_MODE,
        "reranker": USE_RERANKER,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL if USE_RERANKER else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
