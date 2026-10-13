"""PDF text extractor using PyMuPDF (fitz) with header/footer stripping."""

from __future__ import annotations
import os
import re
from collections import Counter
from typing import List, Tuple
from src.integrity.models import ExtractionResult, ExtractionStatus, Segment, Modality


def extract_pdf(file_path: str, doc_id: str) -> ExtractionResult:
    """Extract text from a PDF document, preserving page numbers and stripping headers/footers.
    
    Robustness guarantees:
    - Gracefully handles zero-byte files (returns FAILED with descriptive warning).
    - Catches corrupted or password-protected files (returns FAILED).
    - Identifies scanned or empty PDFs (returns WARNING).
    - Never raises unhandled exceptions.
    """
    if not os.path.exists(file_path):
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            status=ExtractionStatus.FAILED,
            warnings=["File does not exist"],
        )

    if os.path.getsize(file_path) == 0:
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            status=ExtractionStatus.FAILED,
            warnings=["zero-byte file"],
        )

    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            return ExtractionResult(
                doc_id=doc_id,
                file_path=file_path,
                status=ExtractionStatus.FAILED,
                warnings=["PyMuPDF (pymupdf/fitz) is not installed"],
            )

    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            status=ExtractionStatus.FAILED,
            warnings=[f"Failed to open PDF: {str(exc)}"],
        )

    num_pages = len(doc)
    if num_pages == 0:
        doc.close()
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            status=ExtractionStatus.WARNING,
            warnings=["PDF contains zero pages"],
        )

    # First pass: collect margin candidate lines across all pages to detect repeating headers/footers
    margin_candidates: List[str] = []
    page_blocks: List[Tuple[int, List[dict]]] = []

    for page_idx in range(num_pages):
        page = doc[page_idx]
        page_height = page.rect.height
        top_margin = page_height * 0.08
        bottom_margin = page_height * 0.92

        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        page_content: List[dict] = []

        for b in blocks:
            if b[6] != 0:  # 0 is text block, 1 is image block
                continue
            x0, y0, x1, y1, text, _, _ = b
            text_str = text.strip()
            if not text_str:
                continue

            is_margin = (y1 <= top_margin) or (y0 >= bottom_margin)
            if is_margin:
                # Replace numbers with placeholder to match "Page 1 of 5", "Page 2 of 5"
                pattern = re.sub(r"\d+", "#", text_str.lower().strip())
                margin_candidates.append(pattern)

            page_content.append({
                "y0": y0,
                "y1": y1,
                "text": text_str,
                "is_margin": is_margin,
            })

        page_blocks.append((page_idx + 1, page_content))

    doc.close()

    # Determine which margin patterns repeat across at least half the pages
    pattern_counts = Counter(margin_candidates)
    threshold_pages = max(2, num_pages // 2) if num_pages > 1 else 999
    boilerplate_patterns = {pat for pat, count in pattern_counts.items() if count >= threshold_pages}

    # Second pass: assemble clean segments
    segments: List[Segment] = []
    total_words = 0

    for page_num, blocks in page_blocks:
        page_text_pieces: List[str] = []
        for b in blocks:
            raw_text = b["text"]
            pattern = re.sub(r"\d+", "#", raw_text.lower().strip())

            # Skip header/footer boilerplate pattern
            if b["is_margin"] and pattern in boilerplate_patterns:
                continue

            # Skip bare page number lines
            if re.match(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", raw_text.strip(), re.IGNORECASE):
                continue

            page_text_pieces.append(raw_text)
            total_words += len(raw_text.split())

        if page_text_pieces:
            combined_page_text = "\n\n".join(page_text_pieces)
            segments.append(
                Segment(
                    text=combined_page_text,
                    doc_id=doc_id,
                    modality=Modality.PDF.value,
                    page=page_num,
                )
            )

    if total_words < 15:
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            segments=segments,
            status=ExtractionStatus.WARNING,
            warnings=["no extractable text (scanned PDF? OCR out of scope)"],
        )

    return ExtractionResult(
        doc_id=doc_id,
        file_path=file_path,
        segments=segments,
        status=ExtractionStatus.OK,
    )
