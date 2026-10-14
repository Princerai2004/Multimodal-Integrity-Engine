#!/usr/bin/env python3
"""Unified CLI Runner for Multimodal Submission Integrity & Plagiarism Detection Engine.

Usage:
    python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import yaml
from typing import Any, Dict, List

from src.integrity.models import (
    DocumentReport,
    ExtractionResult,
    ExtractionStatus,
    Modality,
    RiskLevel,
)
from src.integrity.extractors.pdf import extract_pdf
from src.integrity.extractors.docx import extract_docx
from src.integrity.extractors.video import extract_video
from src.integrity.chunking import chunk_segments
from src.integrity.boilerplate import detect_and_discount_templates
from src.integrity.embedder import Embedder
from src.integrity.index import VectorIndex
from src.integrity.peer import run_peer_matching
from src.integrity.external import ExternalEngine
from src.integrity.ai_detect import AIDetector
from src.integrity.scoring import calculate_risk_level, generate_document_explanation
from src.integrity.reporter import ReportGenerator

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("IntegrityEngine")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration YAML with safe fallbacks."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not parse {config_path}: {e}. Using defaults.")
    return {
        "matching": {"semantic_threshold": 0.80, "exact_threshold": 0.92},
        "template_handling": {"similarity_threshold": 0.90, "min_frequency_docs": 3},
        "models": {"whisper_model": "tiny", "embedding": "sentence-transformers/all-MiniLM-L6-v2"},
    }


def scan_input_directory(input_dir: str) -> List[str]:
    """Discover all supported candidate files (.pdf, .docx, .mp4), skipping hidden files."""
    supported_exts = {".pdf", ".docx", ".mp4"}
    valid_files: List[str] = []

    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return []

    for entry in sorted(os.listdir(input_dir)):
        if entry.startswith(".") or entry == "dataset_manifest.json":
            continue
        full_path = os.path.join(input_dir, entry)
        if os.path.isfile(full_path):
            _, ext = os.path.splitext(entry.lower())
            if ext in supported_exts:
                valid_files.append(full_path)
            else:
                logger.warning(f"Skipping unsupported file format: {entry}")

    return valid_files


def run_pipeline(
    input_dir: str,
    output_dir: str,
    config_path: str = "config.yaml",
    whisper_model_override: str = None,
) -> int:
    """Execute end-to-end multimodal integrity check."""
    config = load_config(config_path)
    logger.info("Initializing Multimodal Integrity Engine...")

    whisper_model = whisper_model_override or config.get("models", {}).get("whisper_model", "tiny")
    embedding_model = config.get("models", {}).get("embedding", "sentence-transformers/all-MiniLM-L6-v2")
    semantic_thresh = config.get("matching", {}).get("semantic_threshold", 0.80)
    exact_thresh = config.get("matching", {}).get("exact_threshold", 0.92)
    template_thresh = config.get("template_handling", {}).get("similarity_threshold", 0.90)
    min_tmpl_freq = config.get("template_handling", {}).get("min_frequency_docs", 3)
    ref_dir = config.get("external", {}).get("reference_dir", "reference_corpus")

    files = scan_input_directory(input_dir)
    logger.info(f"Discovered {len(files)} candidate submissions in '{input_dir}'")

    if not files:
        logger.warning("No candidate submissions found to evaluate.")
        return 0

    # 1. Extraction Phase
    extraction_results: Dict[str, ExtractionResult] = {}
    for fpath in files:
        doc_id = os.path.basename(fpath)
        _, ext = os.path.splitext(fpath.lower())

        try:
            if ext == ".pdf":
                res = extract_pdf(fpath, doc_id)
            elif ext == ".docx":
                res = extract_docx(fpath, doc_id)
            elif ext == ".mp4":
                res = extract_video(fpath, doc_id, whisper_model=whisper_model)
            else:
                continue
            extraction_results[doc_id] = res
            if res.status == ExtractionStatus.FAILED:
                logger.warning(f"Extraction failed for {doc_id}: {'; '.join(res.warnings)}")
            elif res.warnings:
                logger.info(f"Extraction warning for {doc_id}: {'; '.join(res.warnings)}")
        except Exception as exc:
            logger.error(f"Unexpected error parsing {doc_id}: {exc}")
            extraction_results[doc_id] = ExtractionResult(
                doc_id=doc_id,
                file_path=fpath,
                status=ExtractionStatus.FAILED,
                warnings=[str(exc)],
            )

    # 2. Chunking Phase
    all_chunks = []
    doc_chunks_map = {}
    for doc_id, res in extraction_results.items():
        if res.status == ExtractionStatus.FAILED or not res.segments:
            doc_chunks_map[doc_id] = []
            continue
        chunks = chunk_segments(res.segments, doc_id=doc_id)
        doc_chunks_map[doc_id] = chunks
        all_chunks.extend(chunks)

    logger.info(f"Constructed {len(all_chunks)} semantic chunks across {len(files)} submissions.")

    # 3. Embedding Phase
    embedder = Embedder(model_name=embedding_model)
    if all_chunks:
        texts = [c.normalized_text for c in all_chunks]
        embeddings = embedder.encode(texts, show_progress_bar=False)
    else:
        import numpy as np
        embeddings = np.empty((0, 384), dtype=np.float32)

    # 4. Template & Boilerplate Discounting Phase
    all_chunks, discounted_per_doc = detect_and_discount_templates(
        chunks=all_chunks,
        embeddings=embeddings,
        threshold=template_thresh,
        min_doc_frequency=min_tmpl_freq,
    )

    # 5. Peer & Cross-Modal Matching Phase
    vector_index = VectorIndex(chunks=all_chunks, embeddings=embeddings)
    peer_matches_map, peer_pcts, doc_matrix, doc_ids = run_peer_matching(
        chunks=all_chunks,
        vector_index=vector_index,
        embedder=embedder,
        semantic_threshold=semantic_thresh,
        exact_threshold=exact_thresh,
    )

    # 6. External Reference Engine
    external_engine = ExternalEngine(reference_dir=ref_dir, embedder=embedder)

    # 7. AI Content Estimation
    ai_detector = AIDetector()

    # 8. Report Compilation
    reports: List[DocumentReport] = []
    reporter = ReportGenerator(output_dir=output_dir)

    for doc_id, res in extraction_results.items():
        doc_chunks = doc_chunks_map.get(doc_id, [])
        is_failed = (res.status == ExtractionStatus.FAILED) or (len(doc_chunks) == 0 and res.status == ExtractionStatus.WARNING)

        if is_failed:
            risk_lvl, explanation = calculate_risk_level(0.0, 0.0, 0.0, status="FAILED")
            rep = DocumentReport(
                document_id=doc_id,
                total_integrity_risk=RiskLevel.UNPROCESSED,
                peer_similarity_percentage=0.0,
                external_similarity_percentage=0.0,
                ai_generated_probability=0.0,
                peer_matches=[],
                multimodal_matches=[],
                external_matches=[],
                template_text_discounted=[],
                explanations=f"Document could not be processed. Warnings: {'; '.join(res.warnings)}",
                warnings=res.warnings,
                status="UNPROCESSED",
            )
        else:
            peer_m = [m for m in peer_matches_map.get(doc_id, []) if not m.is_multimodal]
            multi_m = [m for m in peer_matches_map.get(doc_id, []) if m.is_multimodal]
            peer_pct = peer_pcts.get(doc_id, 0.0)

            # External check
            ext_m, ext_pct = external_engine.match_document(doc_chunks, threshold=semantic_thresh)

            # AI estimation
            combined_text = " ".join(c.text for c in doc_chunks if not c.is_template)
            is_oral = any(c.modality == Modality.VIDEO.value for c in doc_chunks)
            ai_prob, _ = ai_detector.estimate(combined_text, is_oral_transcript=is_oral)

            risk_lvl, exp_summary = calculate_risk_level(peer_pct, ext_pct, ai_prob, status="PROCESSED")
            detailed_exp = generate_document_explanation(
                doc_id=doc_id,
                risk_level=risk_lvl,
                peer_percentage=peer_pct,
                external_percentage=ext_pct,
                ai_probability=ai_prob,
                peer_matches=peer_m,
                multimodal_matches=multi_m,
                external_matches=ext_m,
            )

            rep = DocumentReport(
                document_id=doc_id,
                total_integrity_risk=risk_lvl,
                peer_similarity_percentage=peer_pct,
                external_similarity_percentage=ext_pct,
                ai_generated_probability=ai_prob,
                peer_matches=peer_m,
                multimodal_matches=multi_m,
                external_matches=ext_m,
                template_text_discounted=discounted_per_doc.get(doc_id, []),
                explanations=detailed_exp,
                warnings=res.warnings,
                status="PROCESSED",
            )

        reports.append(rep)
        reporter.write_json_report(rep)
        reporter.write_markdown_report(rep)

    # Write aggregate batch summaries
    reporter.write_batch_summary(reports, doc_matrix=doc_matrix, doc_ids=doc_ids)

    logger.info(f"Verification complete! Generated reports for {len(reports)} documents in '{output_dir}'.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Submission Integrity & Plagiarism Detection Engine"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./sample_submissions",
        help="Path to folder containing submissions (.pdf, .docx, .mp4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./reports",
        help="Path to output folder where JSON and Markdown reports will be written",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default=None,
        help="Whisper model size override (tiny, base, small)",
    )

    args = parser.parse_args()
    sys.exit(
        run_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            whisper_model_override=args.whisper_model,
        )
    )


if __name__ == "__main__":
    main()
