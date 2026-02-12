from dotenv import load_dotenv #type: ignore
import os

load_dotenv()  

try:
    DOCS_FOLDER_PATH = os.getenv("DOCS_FOLDER_PATH")
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH")
    SERVICE_CONFIG_PATH = os.getenv("SERVICE_CONFIG_PATH")
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
    MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE"))
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    TOP_K = int(os.getenv("TOP_K"))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT"))
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    LOGGER_LEVEL = os.getenv("LOGGER_LEVEL").upper()
    LOGGER_DIRECTORY = os.getenv("LOGGER_DIRECTORY")

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    API_PORT = int(os.getenv("API_PORT"))
    MCP_PORT = int(os.getenv("MCP_PORT"))

    BASE_VECTORSTORE_PATH = os.getenv("BASE_VECTORSTORE_PATH")
    TEST_NUM_QUERIES = int(os.getenv("TEST_NUM_QUERIES"))

except ValueError as e:
    print(f"❌ Errore nel parsing delle variabili d'ambiente: {e}")
    raise