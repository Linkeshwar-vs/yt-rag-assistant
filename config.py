from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO = "meta-llama/Llama-3.1-8B-Instruct"

TEMPERATURE = 0.2

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4