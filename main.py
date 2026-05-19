from config import RETRIEVER_MODE, USE_RERANKER, PDF_FILE
from retriever import HybridRetriever
from chain import build_rag_chain


retriever = HybridRetriever(pdf_path=PDF_FILE, mode=RETRIEVER_MODE)
rag_chain = build_rag_chain(retriever)


if __name__ == "__main__":
    print(f"\nLegalRAG V2.0  |  检索模式={RETRIEVER_MODE}  |  Reranker={USE_RERANKER}")
    print("输入问题开始对话，输入 exit 退出。\n")
    history = ""
    while True:
        try:
            q = input("请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        ans = rag_chain.invoke({"history": history, "question": q})
        print("\nAI回答：\n", ans, "\n")
        history += f"用户：{q}\nAI：{ans}\n"
