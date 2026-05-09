import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========== 路径配置 ==========
BASE_DIR = r"E:\LegalRAG"
PDF_FILE = os.path.join(BASE_DIR, "劳动合同.pdf")
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# ========== 1. 大模型 ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.1
)

# ========== 2. 加载PDF（超大块，保证条款不被切断） ==========
loader = PyPDFLoader(PDF_FILE)
docs = loader.load()

# 超大分块，确保一条合同条款永远不会被切断
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
splits = splitter.split_documents(docs)
texts = [s.page_content for s in splits]

# ========== 3. 纯TF-IDF检索（零关键词增强！！！） ==========
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(texts)


def retrieve_func(inputs):
    question = inputs["question"]

    # 纯自然语言检索，不增强、不加词、不干预
    q_vec = vectorizer.transform([question])
    similarities = cosine_similarity(q_vec, doc_vectors)[0]

    # 召回全部文档里最相关的前 8 段
    top_k_idx = np.argsort(similarities)[-8:][::-1]
    top_context = "\n---\n".join([texts[i] for i in top_k_idx])
    return top_context


# ========== 4. 通用提示词 ==========
prompt = ChatPromptTemplate.from_template("""
你是文档智能问答助手，请根据提供的文档内容回答问题，不要编造信息。
如果文档中没有相关信息，请回答：未找到相关内容。

历史对话：
{history}

文档内容：
{context}

用户问题：{question}
""")

# ========== 5. RAG 链 ==========
from langchain_core.runnables import RunnableLambda

rag_chain = (
        {
            "context": RunnableLambda(retrieve_func),
            "history": lambda x: x["history"],
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
)

# ========== 6. 多轮对话 ==========
if __name__ == "__main__":
    print("✅ 纯净通用RAG（零关键词增强）")
    history = ""
    while True:
        q = input("\n请输入问题：")
        if q.lower() == "exit":
            break

        ans = rag_chain.invoke({
            "history": history,
            "question": q
        })

        print("\n📝 AI回答：\n", ans)
        history += f"用户：{q}\nAI：{ans}\n"