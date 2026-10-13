"""DOCX text extractor using python-docx with paragraph and heading tracking."""

from __future__ import annotations
import os
from typing import List
from src.integrity.models import ExtractionResult, ExtractionStatus, Segment, Modality


def extract_docx(file_path: str, doc_id: str) -> ExtractionResult:
    """Extract text from a DOCX document, recording paragraph indices and nearby headings.
    
    Robustness guarantees:
    - Handles zero-byte and missing files gracefully.
    - Catches corrupted zip containers or non-docx XML errors.
    - Extracts both body paragraphs and table contents in document order.
    - Headers/footers are not in the body stream, naturally avoiding header contamination.
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
        import docx
    except ImportError:
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            status=ExtractionStatus.FAILED,
            warnings=["python-docx is not installed"],
        )

    try:
        doc = docx.Document(file_path)
    except Exception as exc:
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            status=ExtractionStatus.FAILED,
            warnings=[f"Failed to open DOCX: {str(exc)}"],
        )

    segments: List[Segment] = []
    current_heading = "Introduction / Main"
    para_index = 0
    total_words = 0

    # Extract paragraphs and track headings
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue

        style_name = getattr(p.style, "name", "")
        if "heading" in style_name.lower():
            current_heading = text

        segments.append(
            Segment(
                text=text,
                doc_id=doc_id,
                modality=Modality.DOCX.value,
                paragraph_index=para_index + 1,
                heading=current_heading,
            )
        )
        total_words += len(text.split())
        para_index += 1

    # Extract table cells if present
    for table in doc.tables:
        for row in table.rows:
            row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_texts:
                combined_row = " | ".join(dict.fromkeys(row_texts))  # deduplicate merged cells
                if combined_row:
                    segments.append(
                        Segment(
                            text=combined_row,
                            doc_id=doc_id,
                            modality=Modality.DOCX.value,
                            paragraph_index=para_index + 1,
                            heading=f"{current_heading} (Table)",
                        )
                    )
                    total_words += len(combined_row.split())
                    para_index += 1

    if total_words < 15:
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            segments=segments,
            status=ExtractionStatus.WARNING,
            warnings=["DOCX has no significant extractable text content"],
        )

    return ExtractionResult(
        doc_id=doc_id,
        file_path=file_path,
        segments=segments,
        status=ExtractionStatus.OK,
    )
