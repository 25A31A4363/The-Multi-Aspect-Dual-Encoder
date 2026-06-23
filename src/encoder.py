import logging
import os
import pickle
import sqlite3
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    candidate_id TEXT NOT NULL,
    aspect TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY(candidate_id, aspect)
);
"""


def load_model(model_name: str) -> SentenceTransformer:
    """Load the sentence-transformers model by name."""
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts into dense vectors."""
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)


class EmbeddingCache:
    """Disk-backed cache for candidate aspect embeddings."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(CACHE_SCHEMA)
        self.conn.commit()

    def get(self, candidate_id: str, aspect: str) -> Optional[np.ndarray]:
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE candidate_id = ? AND aspect = ?",
            (candidate_id, aspect),
        ).fetchone()
        if row is None:
            return None
        return pickle.loads(row[0])

    def set(self, candidate_id: str, aspect: str, vector: np.ndarray) -> None:
        blob = pickle.dumps(vector, protocol=pickle.HIGHEST_PROTOCOL)
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings(candidate_id, aspect, vector) VALUES (?, ?, ?)",
            (candidate_id, aspect, blob),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def embed_aspect(
    model: SentenceTransformer,
    cache: EmbeddingCache,
    candidate_id: str,
    aspect: str,
    text: str,
) -> np.ndarray:
    """Embed a single candidate aspect, using cache if available."""
    cached = cache.get(candidate_id, aspect)
    if cached is not None:
        logger.debug("Cache hit for %s:%s", candidate_id, aspect)
        return cached

    embedding = embed_texts(model, [text])
    if embedding.shape[0] == 0:
        return np.zeros((model.get_sentence_embedding_dimension(),), dtype=np.float32)
    vector = embedding[0]
    cache.set(candidate_id, aspect, vector)
    return vector


if __name__ == "__main__":
    import tempfile

    model = load_model("BAAI/bge-large-en-v1.5")
    cache_path = os.path.join(tempfile.gettempdir(), "embedding_cache_test.sqlite")
    cache = EmbeddingCache(cache_path)

    sample_text = "Senior Software Engineer with experience in Python, AWS, Kubernetes"
    emb1 = embed_aspect(model, cache, "test-1", "skills", sample_text)
    emb2 = embed_aspect(model, cache, "test-1", "skills", sample_text)

    print("embedding shape:", emb1.shape)
    print("cache reused:", np.allclose(emb1, emb2))
    cache.close()
