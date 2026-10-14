"""Automated unit and integration test suite for Multimodal Integrity Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.integrity.models import Chunk, Modality, ExtractionStatus
from src.integrity.extractors.pdf import extract_pdf
from src.integrity.extractors.docx import extract_docx
from src.integrity.extractors.video import extract_video
from src.integrity.chunking import chunk_segments, split_into_sentences
from src.integrity.index import VectorIndex
from src.integrity.boilerplate import detect_and_discount_templates
from src.integrity.cleaning import compute_word_jaccard, clean_text
from rapidfuzz import fuzz


def test_self_match_exclusion():
    """Verify that same-document chunk pairs are strictly masked to -inf."""
    c1 = Chunk(chunk_id="d1_c0", doc_id="doc_1", modality="pdf", text="Text A", normalized_text="text a")
    c2 = Chunk(chunk_id="d1_c1", doc_id="doc_1", modality="pdf", text="Text B", normalized_text="text b")
    c3 = Chunk(chunk_id="d2_c0", doc_id="doc_2", modality="pdf", text="Text C", normalized_text="text c")

    # Identical embeddings for all three to test masking
    emb = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float32)

    index = VectorIndex(chunks=[c1, c2, c3], embeddings=emb)
    sim = index.compute_masked_similarity_matrix()

    # Self-match (diagonal) must be -inf
    assert sim[0, 0] == -np.inf
    assert sim[1, 1] == -np.inf
    assert sim[2, 2] == -np.inf

    # Intra-document overlap (c1 vs c2) must be -inf
    assert sim[0, 1] == -np.inf
    assert sim[1, 0] == -np.inf

    # Cross-document pair (c1 vs c3) must be preserved
    assert pytest.approx(sim[0, 2], 0.01) == 1.0


def test_zero_byte_pdf_handling():
    """Verify zero-byte PDF returns FAILED without crashing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        temp_path = f.name

    try:
        res = extract_pdf(temp_path, "empty.pdf")
        assert res.status == ExtractionStatus.FAILED
        assert any("zero-byte" in w.lower() for w in res.warnings)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_silent_video_handling():
    """Verify silent video without audio track returns WARNING without crashing."""
    silent_path = os.path.join("sample_submissions", "mp4_03_silent.mp4")
    if os.path.exists(silent_path):
        res = extract_video(silent_path, "mp4_03_silent.mp4")
        assert res.status == ExtractionStatus.WARNING
        assert any("no audio" in w.lower() for w in res.warnings)


def test_template_frequency_discounting():
    """Verify that chunks repeating across >= 3 documents are marked as template."""
    c1 = Chunk(chunk_id="d1_c0", doc_id="doc_1", modality="pdf", text="Shared instructions", normalized_text="shared instructions", word_count=30)
    c2 = Chunk(chunk_id="d2_c0", doc_id="doc_2", modality="pdf", text="Shared instructions", normalized_text="shared instructions", word_count=30)
    c3 = Chunk(chunk_id="d3_c0", doc_id="doc_3", modality="pdf", text="Shared instructions", normalized_text="shared instructions", word_count=30)
    c4 = Chunk(chunk_id="d4_c0", doc_id="doc_4", modality="pdf", text="Unique original answer", normalized_text="unique original answer", word_count=30)

    # Identical embeddings for c1, c2, c3; orthogonal for c4
    emb = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    updated_chunks, discounted = detect_and_discount_templates([c1, c2, c3, c4], emb, min_doc_frequency=3)

    assert updated_chunks[0].is_template is True
    assert updated_chunks[1].is_template is True
    assert updated_chunks[2].is_template is True
    assert updated_chunks[3].is_template is False


def test_lexical_paraphrase_duality():
    """Verify Jaccard and Levenshtein distinguish exact copy from paraphrasing."""
    orig = "Operating margins contracted across the third quarter due to acute supply chain bottlenecks."
    exact = "Operating margins contracted across the third quarter due to acute supply chain bottlenecks."
    paraphrase = "During the third financial quarter profit margins decreased because of severe distribution logjams."

    # Exact copy: high lexical ratio and high Jaccard
    assert fuzz.ratio(orig, exact) > 90
    assert compute_word_jaccard(orig, exact) > 0.90

    # Paraphrase: low lexical ratio and low Jaccard
    assert fuzz.ratio(orig, paraphrase) < 65
    assert compute_word_jaccard(orig, paraphrase) < 0.20


def test_sentence_splitting_abbreviations():
    """Verify sentence splitter does not break on common abbreviations."""
    sample = "Incumbents misjudge disruptions e.g. low-end entrants. Dr. Porter noted this in the U.S. market."
    sentences = split_into_sentences(sample)
    assert len(sentences) == 2
    assert "e.g." in sentences[0]
    assert "U.S." in sentences[1]
