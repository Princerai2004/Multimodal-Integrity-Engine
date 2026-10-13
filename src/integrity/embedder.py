"""Dense vector embedding engine with disk caching using sentence-transformers."""

from __future__ import annotations
import hashlib
import os
import pickle
from typing import List, Optional, Tuple
import numpy as np


class Embedder:
    """Manages sentence-transformer model loading, batch encoding, and disk caching."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = ".cache/embeddings",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self._model = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def model(self):
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _hash_text(self, text: str) -> str:
        """Generate SHA-256 hash for cache keying."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _get_cache_path(self, text_hash: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into normalized dense vectors (dim=384 for MiniLM-L6-v2).
        
        Leverages disk cache to avoid redundant computation on re-runs.
        Returns:
            np.ndarray of shape (len(texts), embedding_dim), normalized to L2 norm 1.0.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        cached_vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        texts_to_compute: List[Tuple[int, str]] = []

        # Check cache
        for idx, text in enumerate(texts):
            h = self._hash_text(text)
            cpath = self._get_cache_path(h)
            if cpath and os.path.exists(cpath):
                try:
                    with open(cpath, "rb") as f:
                        cached_vectors[idx] = pickle.load(f)
                    continue
                except Exception:
                    pass
            texts_to_compute.append((idx, text))

        # Compute missing embeddings in batches
        if texts_to_compute:
            raw_texts = [item[1] for item in texts_to_compute]
            new_embeddings = self.model.encode(
                raw_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            # Store computed vectors and update disk cache
            for (idx, text), vec in zip(texts_to_compute, new_embeddings):
                cached_vectors[idx] = vec
                h = self._hash_text(text)
                cpath = self._get_cache_path(h)
                if cpath:
                    try:
                        with open(cpath, "wb") as f:
                            pickle.dump(vec, f)
                    except Exception:
                        pass

        # Stack into final matrix
        result = np.vstack([v for v in cached_vectors if v is not None])
        return result.astype(np.float32)

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encode individual sentences without caching for fine-grained alignment."""
        if not sentences:
            return np.empty((0, 384), dtype=np.float32)
        return self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
