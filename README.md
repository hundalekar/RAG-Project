# 🤖 Gemini 2.5 Flash: Intelligent RAG Dashboard

A high-performance, professional-grade **Retrieval-Augmented Generation (RAG)** application. This platform allows users to transform static PDF documents into an interactive knowledge base, powered by **Google Gemini 2.5 Flash** and **FAISS** vector indexing.

---

## 🛠️ System Architecture

The application follows a modular RAG pipeline designed for speed and precision. 



### The Workflow:
1.  **Ingestion Layer**: PyPDFLoader extracts raw text from uploaded PDF files.
2.  **Processing Layer**: `RecursiveCharacterTextSplitter` breaks text into 1000-character chunks with 150-character overlaps to maintain semantic context.
3.  **Embedding Layer**: Text chunks are converted into high-dimensional vectors using `models/gemini-embedding-001`.
4.  **Vector Store**: FAISS (Facebook AI Similarity Search) indexes these vectors locally for millisecond-latency retrieval.
5.  **Retrieval Loop**: When a user queries 🤖, the system performs a similarity search to find the Top-5 most relevant chunks.
6.  **Augmented Generation**: The retrieved context + the user prompt are sent to **Gemini 2.5 Flash** to generate a grounded, factual response.

---

## ✨ Key Features

* **Clean Pro UI**: A minimalist white-and-grey dashboard with vibrant orange accents for high readability.
* **Real-Time System Stats**: Dynamic cards tracking document "Chunks" and "Top-K" retrieval settings.
* **Auto-Clearing Chat**: Powered by `st.chat_input`, ensuring a seamless "type-and-enter" experience without manual backspacing.
* **Adjustable Intelligence**: Sidebar slider to control Model Temperature (creativity vs. logic).
* **Source Transparency**: Built-in "Reference Context" expander to verify AI answers against the original text.

---

🤖 High-Level RAG System Architecture

🛠️ Technical Breakdown of the FlowThe system is divided into two distinct pipelines that work together to provide accurate answers:

A) The Ingestion Pipeline (The "Memory" Creation)

This happens when you click " Process Documents" in the sidebar:

1. PDF Loading: PyPDFLoader breaks the document into raw pages.

2. Recursive Chunking: Text is split into $1000$ character blocks with a $150$ character "overlap." This overlap ensures that if a sentence is cut in half, the meaning is preserved in the next chunk.

3. Vectorization: Each chunk is sent to the models/gemini-embedding-001. This converts text into a mathematical vector (a list of numbers) representing its semantic meaning.

4. FAISS Storage: These vectors are stored in a local FAISS (Facebook AI Similarity Search) index. This acts as a high-speed search engine for "meanings" rather than just keywords.

B) The Retrieval & Generation Pipeline (The "Thinking" Loop)

This happens every time you type a question into the Chat Input:

1. Query Embedding: Your question is also converted into a vector using the same Gemini embedding model.

2. Similarity Search: The system compares your "Question Vector" against all "Chunk Vectors" in FAISS to find the Top-5 (K=5) most mathematically similar matches.

3. Prompt Augmentation: The system constructs a "Super Prompt" that looks like this:"Answer the user based ONLY on this context: [Chunk 1, Chunk 2... Chunk 5]. Question: [User Question]"

4. LLM Synthesis: Gemini 2.5 Flash reads the context and the question to generate a human-like, factual response that is grounded in your PDF.


---
## 🚀 Technical Setup

### 1. Prerequisites
* Python 3.10+
* Google AI Studio API Key (Gemini API)

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-link>
cd <project-folder>

# Install dependencies
pip install -U streamlit langchain-google-genai langchain-community pypdf faiss-cpu python-dotenv
3. Configuration
Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY=your_gemini_api_key_here
4. Running the App
Bash
streamlit run app.py
⚙️ Model Specifications (2026 Stable)
Core LLM: Gemini 2.5 Flash

Embeddings: Gemini-001 Embedding

Vector Engine: FAISS (Local Deserialization enabled)

Context Window: Optimized for 2048 output tokens