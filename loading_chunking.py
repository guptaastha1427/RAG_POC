
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# document loading
loader = PyPDFLoader("./data/BEFORE MR.pdf")
documents = loader.load()
print("Total pages:", len(documents))

# text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)


print("Total chunks:", len(texts))
# print(texts)