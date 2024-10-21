import os
from dotenv import load_dotenv
from pydantic import BaseModel
from chain import BasicRAG
from retriever import get_retriever
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
from embedding import get_openai_embedding
from langchain_openai import ChatOpenAI

load_dotenv()

embedding = get_openai_embedding(api_key="")
retrievers = get_retriever(source="data", embedding=embedding)
# gg_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
gpt_llm = ChatOpenAI(temperature=0, openai_api_key="")
basicrag = BasicRAG(retrievers, gpt_llm)
print(basicrag.answer(question="điểm thi trung học phổ thông quốc gia của em là 27 điểm, em có nên đăng ký vào ngành khoa học máy tính không??"))