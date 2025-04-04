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

# Set Streamlit page config with custom title and favicon
st.set_page_config(page_title="Honeyroom AI", page_icon="🍯", layout="wide")

# Inject custom CSS for theme
st.markdown("""
    <style>
    body {
        background-color: #fde68a;
        color: black !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #fde68a;
        color: black !important;
    }
    .stTextInput>div>div>input {
        background-color: #fff3c4;
        color: black;
    }
    .stButton>button {
        background-color: #facc15;
        color: black;
        font-weight: bold;
    }
    .stMarkdown, .stRadio label, .stSubheader, .stTextInput label, .stTextInput>div>label, .stRadio>div>label {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# Show logo
st.image("image.png", use_column_width=False, width=400)

# Use Streamlit secrets for API key (secure deployment)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

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

st.title("Ask Honeyroom 🐝")
st.write("AI-powered search for effortless buy-side diligence.")

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