import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


ASR_MODEL_NAME = "whisper-large-v3"
ASR_LANGUAGE = "ar"
AUDIO_SAMPLE_RATE = 16000

SUMMARIZATION_MODEL_NAME = "llama-3.3-70b-versatile" 
SUMMARIZATION_MAX_TOKENS = 1024
SUMMARIZATION_TEMPERATURE = 0.3


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384
SEARCH_TOP_K = 5
SEARCH_CHUNK_SIZE = 2 

SUPPORTED_AUDIO_FORMATS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm")
