import time
import re
import numpy as np
# pyrefly: ignore [missing-import]
import faiss
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    SEARCH_TOP_K,
    SEARCH_CHUNK_SIZE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import spaces
except ImportError:
    # Mock decorator for local CPU development
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# Global cached model
_model = None

def get_model():
    global _model
    if _model is None:
        # On ZeroGPU, the GPU is only visible inside the @spaces.GPU decorated functions
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Search] Initializing SentenceTransformer model on {device}...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _model

@spaces.GPU
def encode_chunks_gpu(chunks):
    model = get_model()
    return model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

@spaces.GPU
def encode_query_gpu(query):
    model = get_model()
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


class SemanticSearch:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL_NAME
        self.is_loaded = False

        self.index = None
        self.chunks = []
        self.embeddings = None
        self.is_indexed = False

    def load_model(self):
        if self.is_loaded:
            return

        print(f"[Search] Pre-loading embedding model...")
        get_model()
        self.is_loaded = True


    def _split_into_chunks(self, text: str, chunk_size: int = None) -> list[str]:
        if chunk_size is None:
            chunk_size = SEARCH_CHUNK_SIZE

        sentences = re.split(r'(?<=[.!?،؟؛\n])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks

    def index_text(self, text: str, chunk_size: int = None) -> dict:
        self.load_model()

        if not text or not text.strip():
            return {
                "num_chunks": 0,
                "embedding_dim": EMBEDDING_DIMENSION,
                "processing_time": 0.0,
            }

        print(f"[Search] Indexing text ({len(text)} characters)...")
        start = time.time()

        self.chunks = self._split_into_chunks(text, chunk_size)
        print(f"[Search] Created {len(self.chunks)} chunks")

        if not self.chunks:
            return {
                "num_chunks": 0,
                "embedding_dim": EMBEDDING_DIMENSION,
                "processing_time": 0.0,
            }

        self.embeddings = encode_chunks_gpu(self.chunks)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

        self.is_indexed = True
        elapsed = time.time() - start
        print(f"[Search] Indexing completed in {elapsed:.1f}s")

        return {
            "num_chunks": len(self.chunks),
            "embedding_dim": dim,
            "processing_time": round(elapsed, 2),
        }

    def search(self, query: str, top_k: int = None) -> list[dict]:
        if not self.is_indexed:
            print("[Search] Warning: No text has been indexed yet. Call index_text() first.")
            return []

        if top_k is None:
            top_k = SEARCH_TOP_K

        top_k = min(top_k, len(self.chunks))

        query_embedding = encode_query_gpu(query)

        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append({
                    "rank": rank + 1,
                    "text": self.chunks[idx],
                    "score": round(float(score), 4),
                    "chunk_id": int(idx),
                })

        return results

    def get_stats(self) -> dict:
        return {
            "is_indexed": self.is_indexed,
            "num_chunks": len(self.chunks),
            "embedding_model": self.model_name,
            "embedding_dim": EMBEDDING_DIMENSION,
            "model_loaded": self.is_loaded,
        }
