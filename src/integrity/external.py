"""External reference and web plagiarism simulator."""

from __future__ import annotations
import glob
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.integrity.models import Chunk, MatchRecord, Segment
from src.integrity.chunking import chunk_segments
from src.integrity.embedder import Embedder
from src.integrity.cleaning import compute_word_jaccard
from rapidfuzz import fuzz


class ExternalEngine:
    """Simulates web/external reference search by indexing a local reference knowledge base."""

    def __init__(self, reference_dir: str = "reference_corpus", embedder: Optional[Embedder] = None):
        self.reference_dir = reference_dir
        self.embedder = embedder or Embedder()
        self.ref_chunks: List[Chunk] = []
        self.ref_embeddings: Optional[np.ndarray] = None
        self._is_indexed = False

    def build_reference_index(self):
        """Load and index all reference documents in reference_dir."""
        if not os.path.exists(self.reference_dir):
            os.makedirs(self.reference_dir, exist_ok=True)

        txt_files = glob.glob(os.path.join(self.reference_dir, "*.txt"))
        all_chunks: List[Chunk] = []

        for fpath in txt_files:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Segment text
                seg = Segment(text=content, doc_id=fname, modality="reference", page=1)
                chunks = chunk_segments([seg], doc_id=fname, target_words=180, overlap_sentences=2)
                all_chunks.extend(chunks)
            except Exception:
                continue

        self.ref_chunks = all_chunks
        if all_chunks:
            texts = [c.normalized_text for c in all_chunks]
            self.ref_embeddings = self.embedder.encode(texts)
        else:
            self.ref_embeddings = np.empty((0, 384), dtype=np.float32)

        self._is_indexed = True

    def match_document(
        self,
        submission_chunks: List[Chunk],
        threshold: float = 0.80,
    ) -> Tuple[List[MatchRecord], float]:
        """Match non-template chunks of a submission against the external reference index.
        
        Returns:
            (external_matches, external_similarity_percentage)
        """
        if not self._is_indexed:
            self.build_reference_index()

        non_template_chunks = [c for c in submission_chunks if not c.is_template]
        total_words = sum(c.word_count for c in non_template_chunks)

        if not non_template_chunks or self.ref_embeddings is None or len(self.ref_chunks) == 0:
            return [], 0.0

        sub_texts = [c.normalized_text for c in non_template_chunks]
        sub_embeddings = self.embedder.encode(sub_texts)

        # Cross dot-product: (N_sub, N_ref)
        cross_sim = np.matmul(sub_embeddings, self.ref_embeddings.T)

        matched_words = 0
        matches: List[MatchRecord] = []
        matched_chunk_set = set()

        for i, q_chunk in enumerate(non_template_chunks):
            scores = cross_sim[i]
            best_ref_idx = int(np.argmax(scores))
            best_score = float(scores[best_ref_idx])

            if best_score >= threshold:
                ref_chunk = self.ref_chunks[best_ref_idx]
                matched_chunk_set.add(i)

                lex_ratio = fuzz.ratio(q_chunk.normalized_text, ref_chunk.normalized_text) / 100.0
                jaccard = compute_word_jaccard(q_chunk.text, ref_chunk.text, n=3)

                m_type = "exact" if best_score >= 0.92 and lex_ratio >= 0.70 else "paraphrase"

                # Best sentence alignment
                best_q_sent = q_chunk.sentences[0] if q_chunk.sentences else q_chunk.text
                best_ref_sent = ref_chunk.sentences[0] if ref_chunk.sentences else ref_chunk.text

                record = MatchRecord(
                    matched_file=ref_chunk.doc_id,
                    similarity_score=best_score,
                    query_passage=best_q_sent,
                    matched_passage=best_ref_sent,
                    location={"reference_source": ref_chunk.doc_id, **q_chunk.formatted_location()},
                    match_type=m_type,
                    lexical_score=lex_ratio,
                    jaccard_score=jaccard,
                    is_multimodal=False,
                    query_chunk_id=q_chunk.chunk_id,
                    matched_chunk_id=ref_chunk.chunk_id,
                )
                matches.append(record)

        for idx in matched_chunk_set:
            matched_words += non_template_chunks[idx].word_count

        ext_percentage = round((matched_words / total_words) * 100.0, 1) if total_words > 0 else 0.0
        return matches, ext_percentage
