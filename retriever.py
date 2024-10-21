from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from embedding import get_openai_embedding


def get_retriever(source: str, embedding: Embeddings):
    return FAISS.load_local(
        f"./faiss/{source}",
        embedding,
        allow_dangerous_deserialization=True,
    ).as_retriever()

def get_vectorstrore(source: str, embedding: Embeddings):
    return FAISS.load_local(
        f"./faiss/{source}",
        embedding,
        allow_dangerous_deserialization=True,
    )

if __name__ == "__main__":
    embedding = get_openai_embedding(api_key="")
    vectorstore = get_vectorstrore("data", embedding)
    relevant_documents = vectorstore.similarity_search_with_relevance_scores(
        "điểm thi trung học phổ thông quốc gia của em là 27 điểm, em có nên đăng ký vào ngành khoa học máy tính không?"
    )
    for doc in relevant_documents:
        print(doc)
        print()