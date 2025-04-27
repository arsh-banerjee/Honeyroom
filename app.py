# app.py
import os
import glob
import streamlit as st
import base64
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain_core.documents import Document

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
st.image("image.png", width=300)

# Use Streamlit secrets for API key (secure deployment)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Load documents
def load_documents(path):
    docs = []
    for file in glob.glob(f"{path}/*.pdf"):
        reader = PdfReader(file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + ""
        if full_text.strip():
            docs.append(Document(page_content=full_text, metadata={"source": file}))

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
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

st.title("Ask Honeyroom 🐝")
st.write("AI-powered search for effortless buy-side diligence.")

vectorstore = setup_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever, return_source_documents=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    # Check if it's a follow-up asking for source
    lowered = user_input.lower()
    if any(keyword in lowered for keyword in ["source", "yes", "where", "show me"]):
        last_result = st.session_state.last_result
        if last_result and last_result.get("source_documents"):
            sources = last_result["source_documents"]
            source_response = "Here are the sources I used:"  
            st.session_state.chat_history.append(("ai", source_response))

            unique_sources = {}
            for doc in sources:
                file_path = doc.metadata['source']
                if file_path not in unique_sources:
                    unique_sources[file_path] = doc.page_content

            for file_path, content in list(unique_sources.items())[:1]:
                filename = os.path.basename(file_path)
                st.chat_message("ai").markdown(f"**Preview: {filename}**")
                if file_path.lower().endswith(".pdf"):
                    try:
                        with open(file_path, "rb") as f:
                            data = f.read()
                            base64_pdf = base64.b64encode(data).decode("utf-8")
                            iframe_html = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="80%" height="600px" type="application/pdf"></iframe>'
                            st.markdown(iframe_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning("Could not preview this PDF. You can still download it below.")
                        with open(file_path, "rb") as f:
                            st.download_button(label=f"Download {filename}", data=f, file_name=filename)
                else:
                    st.markdown(content[:500])
        else:
            st.session_state.chat_history.append(("ai", "Sorry, I don’t have a source to show right now."))

    else:
        with st.spinner("Thinking..."):
            result = qa({"query": user_input})
            st.session_state.last_result = result
            answer = result['result']
            sources = result.get('source_documents', [])

            response = answer
            if sources:
                response += "\n\nWould you like to see where this came from? Just ask."
            else:
                response += "\n\nLet me know if you'd like me to try again with a different approach."

            st.session_state.chat_history.append(("ai", response))

# Render conversation
for role, message in st.session_state.chat_history:
    st.chat_message(role).markdown(message)
