# 🚀 RAGMaster: Production-Ready RAG System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Keyword-BM25-orange.svg)](https://github.com/dorianbrown/rank_bm25)
[![Chainlit](https://img.shields.io/badge/UI-Chainlit-red.svg)](https://github.com/Chainlit/chainlit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RAGMaster** is a high-performance Retrieval-Augmented Generation (RAG) system designed to help you interact with your documents seamlessly. It combines the precision of keyword-based search (BM25) with the semantic understanding of dense vector embeddings (FAISS) to provide highly accurate, grounded answers.

## 🌟 The "Why"
I built this tool to solve a personal frustration: **scouring through thousands of pages of documentation to find specific commands or configuration details was taking too much time.** 

Instead of manual searching, I wanted an intelligent system that could:
1.  **Read and understand** complex documents (PDFs, Word, HTML, Confluence).
2.  **Retrieve precise answers** immediately, complete with clickable citations.
3.  **Provide a "Debug View"** to show exactly why a specific document was chosen, making the retrieval process transparent.

## ✨ Key Features
-   **🔍 Hybrid Search Architecture**: Combines BM25 and FAISS with weighted score fusion (Alpha tuning) for superior retrieval performance.
-   **📚 Intelligent Document Parsing**: Robust support for PDF, DOCX, HTML, Text, and Confluence JSON exports.
-   **🏗️ Advanced Chunking**: Implements both Semantic (sentence-aware) and Hierarchical (structure-aware) chunking strategies.
-   **📊 Real-time Observability**: Built-in Prometheus metrics and structured JSON logging for monitoring query latency, token usage, and costs.
-   **🤖 LLM Agnostic**: Seamlessly switch between OpenAI (GPT-4o) and local LLMs (Ollama/Llama 2).
-   **💬 Interactive UI**: A clean, modern chat interface powered by Chainlit with streaming responses and citation tracking.

## 🛠️ Tech Stack
-   **Logic**: Python 3.9+
-   **Retrieval**: FAISS (Dense), BM25 (Sparse)
-   **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
-   **LLM Integration**: OpenAI SDK / Ollama
-   **UI**: Chainlit
-   **Config/Metrics**: Pydantic Settings, Prometheus

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/mk-elbaz/RAGMaster.git
cd RAGMaster

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Copy the `.env.example` file to `.env` and add your settings:
```bash
cp .env.example .env
```

**Key variables in `.env`:**
```text
OPENAI_API_KEY=your_key_here
# or for local use:
USE_LOCAL_LLM=true
LOCAL_LLM_BASE_URL=http://localhost:11434
```

### 3. Build the Index
Add your documents to the `data/` directory, then run the index builder:
```bash
python src/build_index.py --data-dir ./data --index-dir ./indexes
```

### 4. Run the Application
```bash
chainlit run app.py -w
```
Visit `http://localhost:8000` to start chatting!

## 🔮 Future Improvements
-   [ ] **Cross-Encoders**: Implement a re-ranking step using Cross-Encoders to further improve precision.
-   [ ] **GraphRAG**: Integrate knowledge graphs to better handle complex relationships between documents.
-   [ ] **Multi-user Support**: Add authentication and user-specific document workspaces.
-   [ ] **Streaming Retrieval**: Visualize the retrieval process in real-time as the index is searched.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
