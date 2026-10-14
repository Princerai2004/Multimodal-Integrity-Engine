"""Template, institutional boilerplate, and citation discounting engine."""

from __future__ import annotations
import os
from typing import Dict, List, Set, Tuple
import numpy as np
from src.integrity.models import Chunk


def detect_and_discount_templates(
    chunks: List[Chunk],
    embeddings: np.ndarray,
    threshold: float = 0.90,
    min_doc_frequency: int = 3,
    template_files: List[str] = None,
    short_text_threshold: int = 25,
) -> Tuple[List[Chunk], Dict[str, List[str]]]:
    """Identify and mark common prompt templates and institutional boilerplate.
    
    Rules applied:
    1. Cross-Document Frequency: Chunks matching chunks in >= min_doc_frequency other documents
       with cosine similarity >= threshold are classified as benign template text.
    2. Explicit Template Files: Any chunk matching text from an uploaded template file
       is marked as template text.
    3. Short-text Rule: Chunks under short_text_threshold words are flagged to prevent
       spurious inflation from standard one-line instructions.
       
    Returns:
        (updated_chunks, discounted_texts_per_doc): Chunks with is_template set, and a mapping
        of doc_id to list of discounted text snippets for report audit trails.
    """
    if template_files is None:
        template_files = []

    discounted_per_doc: Dict[str, List[str]] = {}
    for c in chunks:
        discounted_per_doc.setdefault(c.doc_id, [])

    if len(chunks) == 0:
        return chunks, discounted_per_doc

    num_chunks = len(chunks)

    # 1. Frequency-based detection across documents
    # Compute similarity matrix
    sim = np.matmul(embeddings, embeddings.T)

    for i in range(num_chunks):
        chunk_i = chunks[i]

        # Short text rule: headers under 12 words are discounted
        if chunk_i.modality != "video" and chunk_i.word_count < 12:
            chunk_i.is_template = True
            discounted_per_doc[chunk_i.doc_id].append(
                f"[Short Header/Title <12 words]: {chunk_i.text[:60]}..."
            )
            continue

        # Find distinct documents that contain a near-duplicate
        matching_doc_ids: Set[str] = set()
        for j in range(num_chunks):
            if chunks[j].doc_id != chunk_i.doc_id and sim[i, j] >= threshold:
                matching_doc_ids.add(chunks[j].doc_id)

        # If >= min_doc_frequency total documents contain this near-duplicate chunk, it's boilerplate
        if (len(matching_doc_ids) + 1) >= min_doc_frequency:
            chunk_i.is_template = True
            discounted_per_doc[chunk_i.doc_id].append(
                f"[Common Template (found across {len(matching_doc_ids) + 1} submissions)]: {chunk_i.text[:120]}..."
            )

    # 2. Check explicit template files if provided
    for t_file in template_files:
        if os.path.exists(t_file):
            try:
                with open(t_file, "r", encoding="utf-8", errors="ignore") as f:
                    template_content = f.read().lower()
                for c in chunks:
                    if not c.is_template:
                        # Check substring overlap or high lexical match
                        if c.normalized_text and c.normalized_text in template_content:
                            c.is_template = True
                            discounted_per_doc[c.doc_id].append(
                                f"[Explicit Template File Match]: {c.text[:120]}..."
                            )
            except Exception:
                pass

    # Deduplicate audit trail messages per document
    for doc_id in discounted_per_doc:
        discounted_per_doc[doc_id] = list(dict.fromkeys(discounted_per_doc[doc_id]))

    return chunks, discounted_per_doc
