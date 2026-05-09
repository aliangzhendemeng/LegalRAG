from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import rag_chain  # 注意现在是 rag_chain，不是原来的 chain！

app = FastAPI(title="法律合同RAG智能问答系统")

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    history: str = ""  # 支持多轮对话历史


@app.post("/ask")
def ask(request: QueryRequest):
    try:
        # 现在必须传入字典：history + question
        answer = rag_chain.invoke({
            "history": request.history,
            "question": request.question
        })

        return {
            "code": 200,
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)