import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========== 路径配置 ==========
BASE_DIR = r"E:\LegalRAG"
PDF_FILE = os.path.join(BASE_DIR, "law.pdf")
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# ========== 1. 大模型 DeepSeek ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.1
)

# ========== 2. 加载并切分 PDF ==========
loader = PyPDFLoader(PDF_FILE)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = splitter.split_documents(docs)
texts = [s.page_content for s in splits]

# ========== 3. 本地检索（不联网、不下载、零依赖） ==========
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(texts)

def retrieve(query):
    q_vec = vectorizer.transform([query])
    similarities = cosine_similarity(q_vec, doc_vectors)[0]
    best_idx = np.argmax(similarities)
    return texts[best_idx]

# ========== 4. 法律提示词 ==========
prompt = ChatPromptTemplate.from_template("""
你是专业法律AI助手，只能根据提供的合同内容回答问题，严禁编造。
如果没有相关内容，请直接回答：未找到相关内容。

合同内容：
{context}

用户问题：{question}
""")

# ========== 5. RAG 链 ==========
chain = (
    {"context": retrieve, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========== 运行 ==========
if __name__ == "__main__":
    print("✅ 法律合同RAG问答系统（本地检索版）")
    while True:
        q = input("\n请输入问题：")
        if q.lower() == "exit":
            break
        ans = chain.invoke(q)
        print("\n📝 AI回答：\n", ans)