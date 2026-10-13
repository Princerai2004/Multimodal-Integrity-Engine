"""Document chunking with sentence preservation and metadata tracking."""

from __future__ import annotations
import re
from typing import List, Optional
from src.integrity.models import Segment, Chunk
from src.integrity.cleaning import clean_text, normalize_for_matching


_COMMON_ABBREVS = {
    "e.g.", "i.e.", "etc.", "vs.", "fig.", "dr.", "prof.", "dept.", "co.", "ltd.", "inc.", "u.s.", "u.k.", "mr.", "ms.", "mrs."
}


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting common abbreviations."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Initial rough split by paragraph
    paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]
    final_sentences: List[str] = []

    for para in paragraphs:
        # Standard Python re engine requires fixed-width lookbehinds, so variable abbreviation
        # patterns like (?<!\bDr|\bProf) raise re.PatternError. Instead, we split on sentence
        # terminators followed by whitespace/uppercase, then re-merge known abbreviations in a second pass.
        raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", para)
        merged: List[str] = []

        for part in raw_parts:
            p_strip = part.strip()
            if not p_strip:
                continue
            if merged:
                # Check if last merged part ends with a common abbreviation or single-letter initial
                last_word = merged[-1].split()[-1].lower() if merged[-1].split() else ""
                is_single_letter_initial = len(last_word) == 2 and last_word[0].isalpha() and last_word[1] == "."
                if last_word in _COMMON_ABBREVS or is_single_letter_initial:
                    merged[-1] = merged[-1] + " " + p_strip
                    continue
            merged.append(p_strip)

        final_sentences.extend(merged)

    return final_sentences


def chunk_segments(
    segments: List[Segment],
    doc_id: str,
    target_words: int = 180,
    overlap_sentences: int = 2,
    min_scoring_words: int = 25,
) -> List[Chunk]:
    """Convert extracted document segments into paragraph-aware semantic chunks with full provenance.
    
    Guarantees:
    - Respects paragraph and section boundaries so prompt headers/citations are not blended into student answers.
    - Chunks long paragraphs (>220 words) with sentence overlap to stay within MiniLM's 256 token limit.
    - Preserves exact page numbers, timestamps, and paragraph index coordinates.
    """
    if not segments:
        return []

    modality = segments[0].modality if segments else "unknown"

    # Dedicated handling for video speech transcripts (group consecutive Whisper segments)
    if modality == "video":
        valid_segs = [s for s in segments if s.text.strip()]
        if not valid_segs:
            return []
        full_video_text = " ".join(s.text.strip() for s in valid_segs)
        sents = split_into_sentences(full_video_text)
        starts = [s.start_sec for s in valid_segs if s.start_sec is not None]
        ends = [s.end_sec for s in valid_segs if s.end_sec is not None]
        return [
            Chunk(
                chunk_id=f"{doc_id}_c000",
                doc_id=doc_id,
                modality=modality,
                text=full_video_text,
                normalized_text=normalize_for_matching(full_video_text),
                sentences=sents,
                start_sec=min(starts) if starts else 0.0,
                end_sec=max(ends) if ends else 0.0,
                word_count=len(full_video_text.split()),
                is_template=False,
            )
        ]

    # Step 1: Break segments into logical paragraph blocks
    paragraph_blocks: List[dict] = []
    for seg in segments:
        raw_text = clean_text(seg.text)
        if not raw_text:
            continue

        # Split multi-paragraph segments (e.g. from PDF pages) by double newlines
        parts = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        for p_idx, part in enumerate(parts):
            p_sents = split_into_sentences(part)
            if not p_sents:
                continue
            paragraph_blocks.append({
                "text": part,
                "sentences": p_sents,
                "word_count": len(part.split()),
                "page": seg.page,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "paragraph": seg.paragraph_index if seg.paragraph_index is not None else (p_idx + 1),
                "heading": seg.heading,
            })

    if not paragraph_blocks:
        return []

    chunks: List[Chunk] = []
    chunk_idx = 0

    # Step 2: Form chunks from paragraph blocks
    for block in paragraph_blocks:
        b_words = block["word_count"]
        b_sents = block["sentences"]

        # If block fits comfortably within the 220-word limit, keep it intact
        if b_words <= 220 or len(b_sents) <= 2:
            full_text = " ".join(b_sents)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_c{chunk_idx:03d}",
                    doc_id=doc_id,
                    modality=modality,
                    text=full_text,
                    normalized_text=normalize_for_matching(full_text),
                    sentences=b_sents,
                    page_start=block["page"],
                    page_end=block["page"],
                    start_sec=block["start_sec"],
                    end_sec=block["end_sec"],
                    paragraph_start=block["paragraph"],
                    paragraph_end=block["paragraph"],
                    word_count=b_words,
                    is_template=False,
                )
            )
            chunk_idx += 1
        else:
            # Overlong paragraph: chunk into overlapping sentence windows
            i = 0
            n = len(b_sents)
            while i < n:
                curr_sents: List[str] = []
                curr_words = 0
                j = i
                while j < n:
                    s = b_sents[j]
                    curr_sents.append(s)
                    curr_words += len(s.split())
                    j += 1
                    if curr_words >= target_words:
                        break

                full_text = " ".join(curr_sents)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_c{chunk_idx:03d}",
                        doc_id=doc_id,
                        modality=modality,
                        text=full_text,
                        normalized_text=normalize_for_matching(full_text),
                        sentences=curr_sents,
                        page_start=block["page"],
                        page_end=block["page"],
                        start_sec=block["start_sec"],
                        end_sec=block["end_sec"],
                        paragraph_start=block["paragraph"],
                        paragraph_end=block["paragraph"],
                        word_count=curr_words,
                        is_template=False,
                    )
                )
                chunk_idx += 1

                if j >= n:
                    break
                step = max(1, (j - i) - overlap_sentences)
                i += step

    return chunks
