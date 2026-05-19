from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from config import (
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TEMPERATURE,
    ENABLE_QUERY_REWRITE,
)
from retriever import HybridRetriever


llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=LLM_TEMPERATURE,
)


_rewrite_prompt = ChatPromptTemplate.from_template(
    """基于以下历史对话，将用户的最新提问改写为一个**独立完整、信息自包含**的检索查询。
要求：
- 若提问含指代（"它/这个/上面说的"等），用历史里的具体对象替换；
- 若提问已经独立完整，原样返回；
- 只输出改写后的查询本身，不要任何解释、引号或前缀。

历史对话：
{history}

用户最新提问：{question}

改写后的查询："""
)

_rewrite_chain = _rewrite_prompt | llm | StrOutputParser()


_qa_prompt = ChatPromptTemplate.from_template(
    """你是一名专业的法律合同问答助手。请严格依据下方"文档内容"回答用户问题，不得编造文档外的信息。
若文档中没有相关条款，请明确回答："未找到相关内容"。

历史对话：
{history}

文档内容：
{context}

用户问题：{question}

回答要求：
1. 先给出简洁明确的结论；
2. 引用具体的文档片段编号（如 [片段3]）作为依据；
3. 如涉及金额、期限、违约责任等关键条款，原文摘录关键句；
4. 必要时给出风险提示，但说明这不构成正式法律意见。"""
)


def _rewrite_if_needed(history: str, question: str) -> str:
    if not ENABLE_QUERY_REWRITE or not history.strip():
        return question
    try:
        rewritten = _rewrite_chain.invoke({
            "history": history,
            "question": question,
        }).strip()
        return rewritten or question
    except Exception as e:
        print(f"[Chain] query 改写失败，回退原句：{e}")
        return question


def build_rag_chain(retriever: HybridRetriever):
    def _retrieve_step(inputs: dict) -> str:
        history = inputs.get("history", "") or ""
        question = inputs["question"]
        search_query = _rewrite_if_needed(history, question)
        if search_query != question:
            print(f"[Chain] 改写 query: {question}  →  {search_query}")
        docs = retriever.retrieve(search_query)
        return retriever.format_context(docs)

    chain = (
        {
            "context": RunnableLambda(_retrieve_step),
            "history": lambda x: x.get("history", "") or "",
            "question": lambda x: x["question"],
        }
        | _qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain
