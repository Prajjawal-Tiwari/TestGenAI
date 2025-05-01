from langchain.vectorstores import FAISS
from langchain.embeddings import CTransformerEmbeddings

def create_vector_store(doc_chunks):
    embedding = CTransformerEmbeddings(model_name="intfloat/e5-small")
    return FAISS.from_documents(doc_chunks, embedding)