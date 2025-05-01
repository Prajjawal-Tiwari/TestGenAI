# from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile

def load_documents(files):
    all_docs = []
    for file in files:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if suffix == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == "txt":
            loader = TextLoader(tmp_path)
        elif suffix == "docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == "csv":
            loader = CSVLoader(tmp_path)
        else:
            continue

        docs = loader.load()
        all_docs.extend(docs)

    return all_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)
