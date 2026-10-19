from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from loading_chunking import texts

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print("Vector DB saved in ./chroma_db")
