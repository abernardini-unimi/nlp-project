# 📚⚡ RAG-DOCUMENT

An **optimized Retrieval-Augmented Generation (RAG) pipeline** featuring:

* 🔍 **Hybrid retriever** → BM25 + semantic embeddings
* 🧩 **Advanced chunking** → hybrid document segmentation strategy
* 🧠 **End-to-end pipeline** → ingestion, retrieval, and generation
* ⚡ **Modular architecture** → independent and reusable components

---

## 📂 Project Structure

```bash
├── api/                    # Api component functions
├── config/                 # Configurations and environment variables
├── db/                     # Database component functions 
├── docs/                   # Source documents for ingestion
├── logs/                   # Local Logs folder by days
├── mcp_server/             # MCP Server folder
├── src/                    # Core source code
│   ├── chunker.py          # Hybrid chunking strategy
│   ├── classes.py          # Complete RAG pipeline implementation
│   ├── pipeline.py         # Semantic search pipeline
│   ├── service_manager.py  # Multi Service Handler
│   ├── retriever.py        # Hybrid retriever (BM25 + embeddings)
│   ├── utils.py            # Utility functions
│   └── text_processor.py   # Optimized text preprocessor
├── tests/                  # Unit & integration tests
│   └── test_muliservice.py # Multi service test system
├── vectorstore/            # Persisted vector indexes and chunks
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
├── setup_ubuntu.sh         # Script for automatic setup ubuntu environment
├── setup_windows.sh        # Script for automatic setup windows environment
├── Dockerfile
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Mode 1 (Automatic)

1. **Clone the repository**

   ```bash
   git clone https://github.com/inxide-srl/inx-mcp-document.git
   cd inx-mcp-document
   ```

2. **Install dependencies and set up the virtual environment**

   You can either run the platform-specific setup script:

   **Linux / Mac**
   
   ```bash
   chmod +x setup_ubuntu.sh
   ./setup_ubuntu.sh
   ```

   **Windows**

   ```bash
   chmod +x setup_windows.sh
   ./setup_windows.sh
   ```
   
3. **activate a virtual environment**

   ```bash
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Add your API keys inside `.env` (e.g., `OPENAI_API_KEY`).

### Mode 2 (Manual)

1. **Clone the repository**

   ```bash
   git clone https://github.com/inxide-srl/inx-mcp-document.git
   cd inx-mcp-document
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   python -m spacy download it_core_news_sm
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Add your API keys inside `.env` (e.g., `OPENAI_API_KEY`).

---

## ▶️ Usage

After upload doc in '/docs'

### Test multi service functions

```bash
python -m tests.test_multi_service
```

---

## 🖥 MCP Server

Start the MCP server in http or sse mode:

```bash
python -m mcp_server.server                  # Default sse
python -m mcp_server.server --transport sse
python -m mcp_server.server --transport http
```

Start the MCP Inspector (Other bash):

Open a new Bash and:

```bash
npx @modelcontextprotocol/inspector
```

### 🔗 Integration with Flowise

The MCP server can be used as a data source in Flowise with the following configuration:

If start the server with sse transport:

```bash
{
  "url": "http://<container-port>:8000/sse" 
}
```

If start the server with http transport:

```bash
{
  "url": "http://<container-port>:8000/mcp" 
}
```

Replace <container-port> with the actual port mapping of your Docker container.

---

## 🐳 Docker

Build the image:

```bash
docker-compose up -d
```

---

## 📊 Technical Highlights

* **Hybrid chunking** → balances semantic coherence and token size
* **Hybrid retriever** → combines BM25 keyword search with embeddings
* **Persistent vectorstore** → supports incremental updates
* **Modular pipeline** → easy to extend with new retrievers or LLMs

---
