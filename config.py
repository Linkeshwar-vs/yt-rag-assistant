from dotenv import load_dotenv
import os
import logging
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO = "meta-llama/Llama-3.1-8B-Instruct"

TEMPERATURE = 0.2

CHUNK_SIZE = 700
CHUNK_OVERLAP = 150


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 20

FINAL_K = 5