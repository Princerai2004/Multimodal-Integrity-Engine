"""Peer-to-peer and cross-modal matching with sentence-level explainability."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from rapidfuzz import fuzz

from src.integrity.models import Chunk, MatchRecord, Modality
from src.integrity.cleaning import compute_word_jaccard
from src.integrity.index import VectorIndex
from src.integrity.embedder import Embedder


def align_best_sentence_pair(
    query_chunk: Chunk,
    matched_chunk: Chunk,
    embedder: Embedder,
) -> Tuple[str, str, float]:
    """Find the most aligned sentence pair inside two matched chunks for explainability.
    
    Returns:
        (query_sentence, matched_sentence, sentence_similarity_score)
    """
    q_sents = [s.strip() for s in query_chunk.sentences if s.strip()]
    m_sents = [s.strip() for s in matched_chunk.sentences if s.strip()]

    if not q_sents or not m_sents:
        return query_chunk.text, matched_chunk.text, 0.0

    # If both have only 1 sentence, return them directly
    if len(q_sents) == 1 and len(m_sents) == 1:
        return q_sents[0], m_sents[0], 0.0

    try:
        q_emb = embedder.encode_sentences(q_sents)
        m_emb = embedder.encode_sentences(m_sents)

        sent_sim = np.matmul(q_emb, m_emb.T)
        best_q_idx, best_m_idx = np.unravel_index(np.argmax(sent_sim), sent_sim.shape)
        return q_sents[best_q_idx], m_sents[best_m_idx], float(sent_sim[best_q_idx, best_m_idx])
    except Exception:
        return q_sents[0], m_sents[0], 0.0


def run_peer_matching(
    chunks: List[Chunk],
    vector_index: VectorIndex,
    embedder: Embedder,
    semantic_threshold: float = 0.80,
    exact_threshold: float = 0.92,
    paraphrase_lexical_max: float = 0.70,
) -> Tuple[Dict[str, List[MatchRecord]], Dict[str, float], np.ndarray, List[str]]:
    """Execute pairwise chunk matching, classify match types, and compute peer similarity percentages.
    
    Returns:
        - matches_per_doc: doc_id -> List of MatchRecord
        - peer_percentages: doc_id -> float percentage (0.0 to 100.0)
        - doc_matrix: (num_docs, num_docs) pair similarity matrix
        - doc_id_list: ordered list of document IDs corresponding to matrix rows/cols
    """
    sim_matrix = vector_index.compute_masked_similarity_matrix()
    num_chunks = len(chunks)

    # Group chunks by doc_id
    doc_ids = sorted(list(dict.fromkeys(c.doc_id for c in chunks)))
    doc_to_chunks: Dict[str, List[int]] = {d: [] for d in doc_ids}
    for idx, c in enumerate(chunks):
        doc_to_chunks[c.doc_id].append(idx)

    matches_per_doc: Dict[str, List[MatchRecord]] = {d: [] for d in doc_ids}
    matched_chunk_indices_per_doc: Dict[str, set] = {d: set() for d in doc_ids}

    # Pairwise document matched words: doc_pairs[doc_a][doc_b] = set of chunk indices in A matched to B
    doc_pair_matches: Dict[str, Dict[str, set]] = {d1: {d2: set() for d2 in doc_ids} for d1 in doc_ids}

    for i in range(num_chunks):
        chunk_i = chunks[i]
        # Ignore template chunks from peer matches
        if chunk_i.is_template:
            continue

        candidates = vector_index.get_candidate_matches_for_chunk(i, threshold=semantic_threshold)
        if not candidates:
            continue

        # Keep the best hit per unique foreign document
        seen_foreign_docs = set()
        for j, cos_sim in candidates:
            chunk_j = chunks[j]
            if chunk_j.is_template:
                continue
            if chunk_j.doc_id in seen_foreign_docs:
                continue
            seen_foreign_docs.add(chunk_j.doc_id)

            # Record match for document pair matrix
            doc_pair_matches[chunk_i.doc_id][chunk_j.doc_id].add(i)
            matched_chunk_indices_per_doc[chunk_i.doc_id].add(i)

            # Compute lexical similarity metrics
            lex_ratio = fuzz.ratio(chunk_i.normalized_text, chunk_j.normalized_text) / 100.0
            jaccard = compute_word_jaccard(chunk_i.text, chunk_j.text, n=3)

            # Classify match type
            if cos_sim >= exact_threshold and lex_ratio >= paraphrase_lexical_max:
                m_type = "exact"
            elif cos_sim >= semantic_threshold and lex_ratio < paraphrase_lexical_max:
                m_type = "paraphrase"
            else:
                m_type = "near-exact"

            # Sentence-level alignment for 30-second audit clarity
            best_q_sent, best_m_sent, _ = align_best_sentence_pair(chunk_i, chunk_j, embedder)

            # Modality check: is it cross-modal (video involved)?
            is_cross_modal = (chunk_i.modality == Modality.VIDEO.value) or (chunk_j.modality == Modality.VIDEO.value)

            # Build location metadata dict
            q_loc = chunk_i.formatted_location()
            m_loc = chunk_j.formatted_location()
            combined_loc: Dict[str, object] = {}
            if "page" in q_loc:
                combined_loc["query_page"] = q_loc["page"]
            if "page" in m_loc:
                combined_loc["matched_page"] = m_loc["page"]
            if "timestamp" in q_loc:
                combined_loc["query_timestamp"] = q_loc["timestamp"]
            if "timestamp" in m_loc:
                combined_loc["matched_timestamp"] = m_loc["timestamp"]
            if "paragraph" in q_loc:
                combined_loc["query_paragraph"] = q_loc["paragraph"]
            if "paragraph" in m_loc:
                combined_loc["matched_paragraph"] = m_loc["paragraph"]

            record = MatchRecord(
                matched_file=chunk_j.doc_id,
                similarity_score=float(cos_sim),
                query_passage=best_q_sent,
                matched_passage=best_m_sent,
                location=combined_loc,
                match_type=m_type,
                lexical_score=float(lex_ratio),
                jaccard_score=float(jaccard),
                is_multimodal=is_cross_modal,
                query_chunk_id=chunk_i.chunk_id,
                matched_chunk_id=chunk_j.chunk_id,
            )
            matches_per_doc[chunk_i.doc_id].append(record)

    # Compute overall peer similarity percentages
    peer_percentages: Dict[str, float] = {}
    for d in doc_ids:
        doc_chunk_indices = doc_to_chunks[d]
        non_template_chunks = [chunks[idx] for idx in doc_chunk_indices if not chunks[idx].is_template]
        total_non_template_words = sum(c.word_count for c in non_template_chunks)

        matched_indices = matched_chunk_indices_per_doc[d]
        matched_words = sum(chunks[idx].word_count for idx in matched_indices if not chunks[idx].is_template)

        if total_non_template_words > 0:
            percentage = min(100.0, (matched_words / total_non_template_words) * 100.0)
        else:
            percentage = 0.0

        peer_percentages[d] = round(percentage, 1)

    # Build document-by-document similarity matrix
    n_docs = len(doc_ids)
    doc_matrix = np.zeros((n_docs, n_docs), dtype=np.float32)

    for i, doc_a in enumerate(doc_ids):
        non_tmpl_a = [chunks[idx] for idx in doc_to_chunks[doc_a] if not chunks[idx].is_template]
        total_words_a = sum(c.word_count for c in non_tmpl_a)
        if total_words_a == 0:
            continue

        for j, doc_b in enumerate(doc_ids):
            if i == j:
                doc_matrix[i, j] = 100.0
            else:
                matched_chunks_in_a = doc_pair_matches[doc_a][doc_b]
                words_covered = sum(chunks[idx].word_count for idx in matched_chunks_in_a if not chunks[idx].is_template)
                doc_matrix[i, j] = round((words_covered / total_words_a) * 100.0, 1)

    return matches_per_doc, peer_percentages, doc_matrix, doc_ids
