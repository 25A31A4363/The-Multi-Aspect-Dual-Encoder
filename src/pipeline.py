import logging
import os
import json
from typing import Any, Dict, List
import numpy as np
import faiss
from tqdm import tqdm

try:
    from .encoder import EmbeddingCache, embed_aspect, load_model
    from .extractor import extract_structural_aspects
except ImportError:
    from encoder import EmbeddingCache, embed_aspect, load_model
    from extractor import extract_structural_aspects

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS index for fast inner product (cosine similarity) searches.

    Args:
        embeddings: Candidate aspect embeddings of shape (N, D).

    Returns:
        A FAISS IndexFlatIP index populated with normalized vectors.
    """
    dimension = embeddings.shape[1]
    # L2-normalize vectors for cosine similarity via inner product
    normalized_embeddings = embeddings.copy()
    norms = np.linalg.norm(normalized_embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized_embeddings = normalized_embeddings / norms

    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings.astype('float32'))
    return index


def process_dataset(
    dataset_path: str,
    model_name: str,
    cache_path: str,
    force_recompute: bool = False,
    limit: int = -1,
) -> Dict[str, Any]:
    """Load JSONL dataset line-by-line, parse structural aspects, and load/compute cached embeddings.

    Args:
        dataset_path: Path to the JSONL dataset.
        model_name: Configurable embedding model name.
        cache_path: Path to SQLite embedding cache.
        force_recompute: If True, bypass cache and recompute embeddings.
        limit: Limit number of candidate records loaded (-1 for unlimited).

    Returns:
        A dictionary containing lists of candidate metadata and stacked embedding matrices.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load embedding model and cache
    model = load_model(model_name)
    cache = EmbeddingCache(cache_path)

    candidate_ids: List[str] = []
    candidate_names: List[str] = []
    title_embeddings: List[np.ndarray] = []
    project_embeddings: List[np.ndarray] = []
    skill_embeddings: List[np.ndarray] = []
    raw_records: List[Dict[str, Any]] = []

    logger.info("Reading dataset line-by-line from %s...", dataset_path)

    # Count total lines for tqdm
    total_lines = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1
            if limit > 0 and total_lines >= limit:
                break

    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=total_lines, desc="Processing candidates")):
            if limit > 0 and idx >= limit:
                break
            
            line = line.strip()
            if not line:
                continue

            try:
                raw_record = json.loads(line)
            except Exception as e:
                logger.warning("Skipping invalid JSON line at index %d: %s", idx, e)
                continue

            # Infer candidate ID and name
            candidate_id = raw_record.get("candidate_id")
            if not candidate_id:
                candidate_id = f"CAND_{idx:07d}"
            
            profile = raw_record.get("profile", {}) or {}
            candidate_name = profile.get("anonymized_name", profile.get("name", ""))
            if not candidate_name:
                candidate_name = f"Candidate {candidate_id}"

            # Extract aspects
            parsed = extract_structural_aspects(raw_record, candidate_id=candidate_id)
            title_text = " \n ".join(parsed["job_titles"]) if parsed["job_titles"] else ""
            project_text = " \n ".join(parsed["projects"]) if parsed["projects"] else ""
            skill_text = " \n ".join(parsed["skills"]) if parsed["skills"] else ""

            # Check force_recompute bypass
            if force_recompute:
                # Force database clear for this candidate
                cache.conn.execute("DELETE FROM embeddings WHERE candidate_id = ?", (candidate_id,))
                cache.conn.commit()

            # Embed aspects (uses cache internally)
            title_vector = embed_aspect(model, cache, candidate_id, "job_titles", title_text)
            project_vector = embed_aspect(model, cache, candidate_id, "projects", project_text)
            skill_vector = embed_aspect(model, cache, candidate_id, "skills", skill_text)

            candidate_ids.append(candidate_id)
            candidate_names.append(candidate_name)
            title_embeddings.append(title_vector)
            project_embeddings.append(project_vector)
            skill_embeddings.append(skill_vector)
            
            # Keep raw candidate record for metadata lookups later
            raw_records.append(raw_record)

    cache.close()

    # Stack embeddings
    dim = model.get_sentence_embedding_dimension()
    title_matrix = np.vstack(title_embeddings) if title_embeddings else np.zeros((0, dim), dtype=np.float32)
    project_matrix = np.vstack(project_embeddings) if project_embeddings else np.zeros((0, dim), dtype=np.float32)
    skill_matrix = np.vstack(skill_embeddings) if skill_embeddings else np.zeros((0, dim), dtype=np.float32)

    logger.info("Loaded %d candidates. Embedding dimension: %d", len(candidate_ids), dim)

    # Scaffold FAISS indexes
    logger.info("Scaffolding FAISS indexes per aspect...")
    title_index = create_faiss_index(title_matrix)
    project_index = create_faiss_index(project_matrix)
    skill_index = create_faiss_index(skill_matrix)

    return {
        "candidate_ids": candidate_ids,
        "candidate_names": candidate_names,
        "raw_records": raw_records,
        "title_embeddings": title_matrix,
        "project_embeddings": project_matrix,
        "skill_embeddings": skill_matrix,
        "title_index": title_index,
        "project_index": project_index,
        "skill_index": skill_index,
    }


if __name__ == "__main__":
    dataset_path = "src/dataset/candidates.jsonl"
    model_name = "BAAI/bge-large-en-v1.5"
    cache_path = "outputs/embeddings_cache.sqlite"
    try:
        result = process_dataset(dataset_path, model_name, cache_path, force_recompute=False, limit=10)
        print("Processed candidates:", len(result["candidate_ids"]))
        print("Embeddings shape:", result["title_embeddings"].shape)
    except FileNotFoundError as exc:
        print(exc)
