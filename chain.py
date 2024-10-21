import os
from operator import itemgetter
from dotenv import load_dotenv
from retriever import get_retriever
from typing import List, Literal
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from langchain_community.vectorstores import FAISS

# from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def format_docs(docs: Document):
    def get_content(doc: Document):
        res = doc.page_content
        
        return res
    return "\n\n".join(
        get_content(doc) if isinstance(doc, Document) else get_content(doc[0])
        for doc in docs
    )

class BasicRAG:
    def __init__(self, retriever: VectorStoreRetriever, llm):
        self.retriever = retriever
        self.llm = llm
        template = """Giả sử bạn là chuyên gia tư vấn tuyển sinh của trường Đại học Công nghệ. Bạn có nhiệm vụ tư vấn, trả lời câu hỏi liên quan đến tuyển sinh của trường dựa trên những tài liệu mà bạn được cung cấp dưới đây.
        Đây là tài liệu được cung cấp: {context}

        Đây là câu hỏi, nội dung tư vấn bạn cần trả lời: {question}
        
        Hãy trả lời câu hỏi trên bằng 1 đoạn văn trả lời không quá dài 
        Đương nhiên, nếu tài liệu không liên quan đến câu hỏi, hoặc không giải quyết được vấn đề câu hỏi đưa ra thì xin lỗi người dùng và trả lời họ theo định dạng: "Hiện tại, với kho dữ liệu đang được hoàn thiện, tôi rất tiếc chưa thể cung cấp câu trả lời thoả đáng cho câu hỏi của bạn. Chúng tôi đang nỗ lực nghiên cứu và phát triển để mở rộng phạm vi hỗ trợ trong tương lai gần.
        """
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def answer(self, question: str) -> str:
        return self.chain.invoke(question)
if __name__ == "__main__":
    pass