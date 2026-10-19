import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def load_vector_db():
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=get_embeddings(),
    )

st.title("PDF Chatbot")

if not Path(DB_PATH).exists():
    st.error("No vector DB found. First run: `python create_vector_db.py`")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5,
)
vector_db = load_vector_db()

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Working..."):
            docs = vector_db.similarity_search(question, k=5)
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = f"""Use only the context below from the PDF to answer.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
"""
            response = llm.invoke(prompt)
            answer = response.text.replace("*", "")

        st.write("**Answer:**")
        st.write(answer)


        st.subheader("Sources")
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            page_label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
            with st.expander(f"{i}. {page_label}"):
                st.text(doc.page_content)