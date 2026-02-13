# 🧠 RAG-Document

Un sistema avanzato di Retrieval-Augmented Generation (RAG) progettato per gestire multipli flussi documentali. Il progetto espone un'architettura ibrida che supporta sia API REST tradizionali sia un **Server MCP (Model Context Protocol)** per l'interazione diretta con agenti AI autonomi.

## 🏗️ Architettura del Sistema

<p align="center">
  <img src="img/img.png" width="400">
</p>

Il sistema è orchestrato da un componente centrale (**Manager**) che gestisce il flusso di dati tra gli input, gli agenti e i database.

* **🤖 MCP Server:** Autentica e instrada le richieste di agenti multipli utilizzando un sistema di token verification.
* **🧠 Core Manager:** Il "cervello" dell'applicazione. Crea e gestisce le pipeline, valuta le query e decide quale pipeline di retrieval o elaborazione attivare.
* **🌐 API Server:** Espone gli endpoint tramite FastAPI per la creazione ed eliminazione dei vector store.
* **🗄️ Neo4j Database:** Utilizza un database a grafi per memorizzare l'istantanea dei vector store creati.

---

## ✨ Funzionalità Principali

* Gestione di più richieste in parallelo da parte degli agenti.
* Sistema di cache intelligente per caricare in memoria solo i vector store più usati dagli agenti.
* Sistema di confronto tra varie tecniche di RAG avanzate, tra cui:
  * **Sparse (BM25)**
  * **Semantic**
  * **Hybrid Retriever**
  * **Contextual Header**
  * **Hierarchical Indices**
  * **Multi-Query RAG**
  * **Parent Document**
  * **Query Transformations**
  * **Relevant Segment Extraction**
  * **Reranking**

---

## 📁 Struttura del Progetto

```text
rag_document/
├── api/                # Endpoint FastAPI, schemi e router
├── config/             # Impostazioni di sistema e configurazione logger
├── db/                 # Connessioni e script di setup per Neo4j/Vector DB
├── docs/               # Cartella di ingestione documenti (PDF, Docx, TXT)
├── llm/                # Integrazioni con i provider LLM (Groq)
├── mcp_server/         # Server Model Context Protocol e verifica token
├── src/                # Core logic:
│   ├── retrievers/     # Tutte le strategie di ricerca RAG sopra elencate
│   ├── cache.py 
│   ├── chuncker.py 
│   ├── embedder.py
│   ├── text_processor.py
│   ├── pipeline.py
│   └── manager.py
└── test/                  # Script di benchmark per i retriever disponibili
│   ├── results/           # Risultati dei test di comparione tra i retriever     
│   ├── faqs.json          # Json con tutte le faq utilizzate nei test           
│   ├── run_judge.py       # Codice per eseguire la valutazione LLM-AS-JUDGE delle risposte date dai retriever 
│   ├── test_pipeline.py   # Codice per creare e confrontare i tempi di creazione delle pipeline per ogni retriever 
│   └── test_retriever.py  # Codice per effettuare le domande di test presenti in faqs ai retriver per poter confrontare la latenza 
```

---

## 🚀 Setup e Installazione

Puoi avviare il sistema RAG-Document in due modi: tramite **Docker** (metodo consigliato per test rapidi) o tramite un'**installazione locale** classica. Altrimenti è possibile replicare i test eseguiti, ovvero confrontare tutti i retriver andando direttamente alla sezione Test e Benchmark, i risultati saranno presenti nella cartella test/results.

### 1️⃣ Metodo Consigliato: Docker + MCP Inspector

Questo metodo genera un ambiente testabile dell'intero sistema caricando i documenti presenti nella cartella `docs/`.

Costruisci e avvia i container:

```bash
docker-compose up -d --build
```

In un nuovo terminale, avvia l'MCP Inspector per testare le richieste:

```bash
npx @modelcontextprotocol/inspector
```


### 2️⃣ Metodo Locale: Installazione da sorgente

1. **Clona il repository ed entra nella cartella:**
```bash
cd rag-document
```


2. **Crea e attiva un ambiente virtuale:**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```


3. **Installa le dipendenze:**
```bash
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt
python -m spacy download it_core_news_sm
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
```


4. **Configura le variabili d'ambiente:**
```bash
cp .env.example .env
```

*Assicurati di inserire le tue chiavi API (es. GROQ) all'interno del file `.env`.*

5. **Fai partire il Sistema:**
```bash
python main.py
```

Una volta che hai fatto partire il sistema devi:
   1. Creare un db in Neo4j per tenere traccia dei vectorstore che crei.
   1. Aggiungere i tuoi documenti nella cartella docs/
   2. Creare un nuovo un vectorstore utilizzando le API costruito con i documenti che hai aggiunto.
   3. Collegarti al server MCP tramite l'inspector per effettuare le richieste al sistema(ricorda che ad ogni vectorstore creato è collegata una chiave di autenticazione da dover fornire per connetterti al server mcp)


---

### Avvio dell'MCP Server in locale

Puoi avviare il server MCP in modalità HTTP o SSE:

```bash
python -m mcp_server.server                 # Default (SSE)
python -m mcp_server.server --transport sse
python -m mcp_server.server --transport http
```

Anche in questo caso, puoi usare l'Inspector aprendo un altro terminale:

```bash
npx @modelcontextprotocol/inspector
```

---

## 🧪 Test e Benchmark

È possibile riprodurre i test di confronto effettuati tra i vari retriever. Tutti i risultati (in formato `.xlsx`) verranno salvati automaticamente nella cartella `test/results/`.

**Testare la velocità di creazione delle pipeline di tutti i retriever:**

```bash
python -m test.test_pipeline
```

**Eseguire le domande di test per ogni retriever:**

```bash
python -m test.test_retriever
```

**Valutare la correttezza delle risposte (LLM-as-a-Judge):**

```bash
python -m test.run_judge
```

---