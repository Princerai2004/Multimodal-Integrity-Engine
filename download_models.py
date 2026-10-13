#!/usr/bin/env python3
"""Pre-download and cache all models required by the Multimodal Integrity Engine.

Ensures single-command setup as required by the assessment brief:
    pip install -r requirements.txt && python download_models.py
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelDownloader")


def download_all():
    logger.info("==================================================")
    logger.info("Downloading models for Multimodal Integrity Engine")
    logger.info("==================================================")

    # 1. Sentence-Transformers: all-MiniLM-L6-v2
    logger.info("[1/4] Pre-caching Sentence-Transformers (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("✓ all-MiniLM-L6-v2 successfully cached.")
    except Exception as e:
        logger.warning(f"Could not pre-cache all-MiniLM-L6-v2: {e}")

    # 2. Faster-Whisper: tiny
    logger.info("[2/4] Pre-caching Faster-Whisper (tiny)...")
    try:
        from faster_whisper import WhisperModel
        _ = WhisperModel("tiny", device="cpu", compute_type="int8", download_root=None)
        logger.info("✓ Faster-Whisper (tiny) successfully cached.")
    except Exception as e:
        logger.warning(f"Could not pre-cache Faster-Whisper tiny: {e}")

    # 3. AI Detector: roberta-base-openai-detector
    logger.info("[3/4] Pre-caching Hugging Face AI Detector (roberta-base-openai-detector)...")
    try:
        from transformers import pipeline
        _ = pipeline("text-classification", model="roberta-base-openai-detector")
        logger.info("✓ roberta-base-openai-detector successfully cached.")
    except Exception as e:
        logger.warning(f"Could not pre-cache roberta-base-openai-detector: {e}")

    # 4. Fallback Model: gpt2
    logger.info("[4/4] Pre-caching GPT-2 for Perplexity / Burstiness...")
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _ = GPT2TokenizerFast.from_pretrained("gpt2")
        _ = GPT2LMHeadModel.from_pretrained("gpt2")
        logger.info("✓ gpt2 successfully cached.")
    except Exception as e:
        logger.warning(f"Could not pre-cache gpt2: {e}")

    logger.info("==================================================")
    logger.info("All model pre-caching routines finished.")
    logger.info("==================================================")


if __name__ == "__main__":
    download_all()
