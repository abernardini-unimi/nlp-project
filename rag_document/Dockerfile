# --- Immagine base leggera ---
FROM python:3.11-slim

# --- Argomento di build ---
ARG RETRIEVER_TYPE=HybridRetriever

# --- Variabili di ambiente ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RETRIEVER_TYPE=${RETRIEVER_TYPE}

# --- Installazione dipendenze di sistema ---
RUN apt-get update && \
    apt-get install -y build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# --- Creazione della cartella di lavoro ---
WORKDIR /app

# --- Copia requirements.txt per caching ---
COPY requirements.txt /app/requirements.txt

# --- Installazione delle dipendenze Python (CPU-only) ---
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install sentence-transformers && \
    pip install --no-cache-dir -r requirements.txt

# --- Copia del codice dell'applicazione ---
COPY . /app

# --- Download dei dataset NLTK necessari ---
RUN python -m nltk.downloader punkt stopwords punkt_tab

# --- Download modello SpaCy italiano ---
RUN python -m spacy download it_core_news_sm

# --- Esposizione porta del server ---
EXPOSE 8080

# --- Comando di avvio del server ---
CMD ["sh", "-c", "python main.py --retriever ${RETRIEVER_TYPE}"]