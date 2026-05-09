from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import chain

app = FastAPI(title="法律合同RAG智能问答系统")

# 跨域，解决网页请求失败
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QueryRequest):
    try:
        answer = chain.invoke(request.question)
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