from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

from ..config.settings import get_settings


@lru_cache(maxsize=1)
def _load_model():
    # Lazy import so the app still starts even if the optional deps are missing.
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    return SentenceTransformer(s.embeddings_model_name)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns float32 matrix of shape [len(texts), dim].
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model = _load_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between 1D vectors.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def best_match_via_faiss(
    query_vecs: np.ndarray,
    key_vecs: np.ndarray,
    key_labels: List[str],
    top_k: int = 1,
) -> List[List[Tuple[str, float]]]:
    """
    Returns, for each query: list of (label, similarity).
    similarity is cosine similarity because we L2-normalize and use inner product.
    """
    if query_vecs.size == 0 or key_vecs.size == 0:
        return [[] for _ in range(len(query_vecs))]

    q = normalize(query_vecs)
    k = normalize(key_vecs)

    try:
        import faiss  # type: ignore

        dim = k.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(k)
        sims, idxs = index.search(q, top_k)

        results: List[List[Tuple[str, float]]] = []
        for row_sims, row_idxs in zip(sims, idxs):
            row: List[Tuple[str, float]] = []
            for sim, idx in zip(row_sims, row_idxs):
                if idx < 0 or idx >= len(key_labels):
                    continue
                row.append((key_labels[int(idx)], float(sim)))
            results.append(row)
        return results
    except Exception:
        # Fallback: no FAISS available.
        results = []
        for qv in q:
            sims = k @ qv
            best_idxs = np.argsort(-sims)[:top_k]
            row = [(key_labels[int(i)], float(sims[int(i)])) for i in best_idxs]
            results.append(row)
        return results

