# app.py
import os
import glob
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# Optionally set your OpenAI key here (local dev only)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "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")

# Load documents
def load_documents(path):
    docs = []
    for file in glob.glob(f"{path}/*.pdf"):
        loader = PyPDFLoader(file)
        docs.extend(loader.load())
    for file in glob.glob(f"{path}/*.txt"):
        loader = TextLoader(file)
        docs.extend(loader.load())
    return docs

# Split, embed, and store
@st.cache_resource
def setup_vectorstore():
    raw_docs = load_documents("data")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

st.title("AI Deal Room Assistant")
st.write("Ask a question about the documents in the data room.")

vectorstore = setup_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

user_query = st.text_input("Your question:")

if user_query:
    with st.spinner("Thinking..."):
        result = qa({"query": user_query})
        st.subheader("Answer")
        st.write(result['result'])

        st.subheader("Sources")
        for doc in result.get('source_documents', []):
            st.markdown(f"**{os.path.basename(doc.metadata['source'])}**: {doc.page_content[:300]}...")

        feedback = st.radio("Was this helpful?", ("👍", "👎"))
        if feedback == "👎":
            with st.spinner("Trying again with different context..."):
                retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)
                result = qa({"query": user_query})
                st.subheader("Retry Answer")
                st.write(result['result'])