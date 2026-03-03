import streamlit as st
import os
import uuid
from dotenv import load_dotenv

# Google Gemini & LangChain 2026 Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

import warnings
warnings.filterwarnings("ignore")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- INITIALIZE BACKEND ---
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_llm(temp):
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp, max_output_tokens=2048)

def load_vector_store():
    try:
        return FAISS.load_local("faiss_index", gemini_embeddings, allow_dangerous_deserialization=True)
    except: return None

# --- UI STYLING (Clean & Minimalist) ---
def apply_custom_design():
    st.markdown("""
        <style>
        /* Remove Sidebar Background Colors */
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        /* White Main Area */
        .stApp {
            background-color: #ffffff;
        }

        /* Stat Cards - White with Grey Border and Orange Accent */
        .stat-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 5px solid #f97316; 
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .stat-label { font-size: 0.8rem; color: #64748b; font-weight: bold; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #1e293b; }
        
        /* Orange Button Styling */
        .stButton>button {
            background-color: #f97316 !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            font-weight: bold;
            width: 100%;
            padding: 10px;
        }

        /* Sidebar Text Color */
        [data-testid="stSidebar"] .stMarkdown p { color: #1e293b; }
        </style>
    """, unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="🤖 RAG Assistant", layout="wide", initial_sidebar_state="expanded")
    apply_custom_design()

    if "messages" not in st.session_state: st.session_state.messages = []
    if "num_chunks" not in st.session_state: st.session_state.num_chunks = 0

    # --- LEFT PANEL (SIDEBAR) ---
    with st.sidebar:
        st.markdown("### 🤖 Document Center")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        
        if uploaded_file and st.button("🚀 Process Documents"):
            with st.spinner("Indexing..."):
                temp_path = f"temp_{uuid.uuid4()}.pdf"
                with open(temp_path, "wb") as f: f.write(uploaded_file.getvalue())
                loader = PyPDFLoader(temp_path)
                pages = loader.load_and_split()
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(pages)
                st.session_state.num_chunks = len(chunks)
                FAISS.from_documents(chunks, gemini_embeddings).save_local("faiss_index")
                os.remove(temp_path)
                st.success("🤖 Analysis Ready!")

        st.markdown("---")
        st.markdown("### 📊 System Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-label">CHUNKS</div><div class="stat-value">{st.session_state.num_chunks}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="stat-card"><div class="stat-label">TOP-K</div><div class="stat-value">5</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        temp = st.slider("🤖 Temperature", 0.0, 1.0, 0.3)

        st.markdown("---")
        if st.button("🗑️ Clear All Content"):
            st.session_state.messages = []
            st.session_state.num_chunks = 0
            st.rerun()

    # --- MAIN CHAT AREA ---
    st.title("🤖 Intelligent PDF Assistant")
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask 🤖 about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)

        vector_store = load_vector_store()
        if not vector_store:
            st.warning("🤖 Please upload a PDF first!")
        else:
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Thinking..."):
                        llm = get_llm(temp)
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", "Answer based on context:\n\n{context}"),
                            ("human", "{input}"),
                        ])
                        chain = create_retrieval_chain(
                            vector_store.as_retriever(search_kwargs={"k": 5}), 
                            create_stuff_documents_chain(llm, prompt_template)
                        )
                        res = chain.invoke({"input": prompt})
                        st.markdown(res["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
        st.rerun()

if __name__ == "__main__":
    main()