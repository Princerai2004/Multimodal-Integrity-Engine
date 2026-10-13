"""Vector similarity index with exact cosine matrix multiplication and same-document masking."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.integrity.models import Chunk


class VectorIndex:
    """In-memory vector index computing exact pairwise cosine similarities for submission chunks."""

    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Initialize vector index with chunks and their L2-normalized embeddings.
        
        Args:
            chunks: List of Chunk objects with metadata.
            embeddings: Normalized float32 ndarray of shape (len(chunks), embedding_dim).
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.num_chunks = len(chunks)
        self._similarity_matrix: Optional[np.ndarray] = None
        self._doc_id_to_indices: Dict[str, List[int]] = {}

        for idx, chunk in enumerate(chunks):
            self._doc_id_to_indices.setdefault(chunk.doc_id, []).append(idx)

    def compute_masked_similarity_matrix(self) -> np.ndarray:
        """Compute all-pairs cosine similarity matrix with strict same-document masking.
        
        Mathematical basis:
        Since embeddings are L2-normalized (||e||_2 = 1.0),
        Cosine Similarity(u, v) = u . v = E @ E.T
        
        Masking guarantee:
        All entries S[i, j] where chunks[i].doc_id == chunks[j].doc_id are set to -inf.
        This completely eliminates:
        1. Identical self-matches (diagonal i == j).
        2. Overlapping chunk artifacts from within the same document.
        """
        if self._similarity_matrix is not None:
            return self._similarity_matrix

        if self.num_chunks == 0:
            self._similarity_matrix = np.empty((0, 0), dtype=np.float32)
            return self._similarity_matrix

        # Exact dot product for normalized vectors
        sim = np.matmul(self.embeddings, self.embeddings.T).astype(np.float32)

        # Apply strict same-document mask
        for doc_id, indices in self._doc_id_to_indices.items():
            if len(indices) == 1:
                idx = indices[0]
                sim[idx, idx] = -np.inf
            else:
                ix_grid = np.ix_(indices, indices)
                sim[ix_grid] = -np.inf

        self._similarity_matrix = sim
        return self._similarity_matrix

    def get_candidate_matches_for_chunk(
        self,
        chunk_idx: int,
        threshold: float = 0.80,
    ) -> List[Tuple[int, float]]:
        """Retrieve all foreign chunks matching chunk_idx with cosine similarity >= threshold.
        
        Returns:
            List of (other_chunk_idx, cosine_similarity) tuples sorted by score descending.
        """
        sim_matrix = self.compute_masked_similarity_matrix()
        if self.num_chunks == 0 or chunk_idx >= self.num_chunks:
            return []

        row = sim_matrix[chunk_idx]
        candidate_indices = np.where(row >= threshold)[0]
        if len(candidate_indices) == 0:
            return []

        candidates = [(int(c_idx), float(row[c_idx])) for c_idx in candidate_indices]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates


def match_external_corpus(
    submission_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    threshold: float = 0.80,
) -> List[Optional[Tuple[int, float]]]:
    """Find the top reference match for each submission chunk against an external corpus.
    
    Args:
        submission_embeddings: Normalized (N_sub, dim) ndarray.
        reference_embeddings: Normalized (N_ref, dim) ndarray.
        threshold: Minimum cosine similarity.
        
    Returns:
        List of length N_sub, where each element is either (ref_chunk_idx, score) or None.
    """
    if submission_embeddings.shape[0] == 0 or reference_embeddings.shape[0] == 0:
        return [None] * submission_embeddings.shape[0]

    # Cross-matrix similarity: (N_sub, N_ref)
    cross_sim = np.matmul(submission_embeddings, reference_embeddings.T)

    best_match_indices = np.argmax(cross_sim, axis=1)
    best_match_scores = np.max(cross_sim, axis=1)

    results: List[Optional[Tuple[int, float]]] = []
    for idx in range(len(best_match_scores)):
        score = float(best_match_scores[idx])
        if score >= threshold:
            results.append((int(best_match_indices[idx]), score))
        else:
            results.append(None)

    return results
