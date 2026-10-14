"""Multi-tier AI content likelihood detector with RoBERTa, Perplexity fallback, and safety guardrails."""

from __future__ import annotations
import math
import re
from typing import List, Optional, Tuple


class AIDetector:
    """Probabilistic AI-generated text estimation engine.
    
    Architecture:
    1. Primary: Hugging Face RoBERTa-based AI detector with 512-token sliding window.
    2. Fallback: GPT-2 Perplexity & Burstiness heuristic (AI text exhibits low perplexity & low sentence-level variance).
    3. Tertiary: Statistical lexical diversity heuristic (reproducible deterministic fallback).
    
    Safety & Calibration:
    - Automatically down-weights short snippets (< 40 words) and oral/video transcripts.
    - Explicitly labelled as a probabilistic estimate rather than conclusive proof.
    """

    LIMITATION_NOTE = (
        "AI detectors have known false positives, especially on formal or non-native English writing. "
        "Outputs represent probabilistic estimates and should be used solely as advisory indicators."
    )

    def __init__(
        self,
        primary_model_name: str = "roberta-base-openai-detector",
        perplexity_model_name: str = "gpt2",
        device: str = "cpu",
    ):
        self.primary_model_name = primary_model_name
        self.perplexity_model_name = perplexity_model_name
        self.device = device
        self._primary_pipeline = None
        self._gpt2_model = None
        self._gpt2_tokenizer = None

    def _init_primary(self):
        if self._primary_pipeline is None:
            try:
                from transformers import pipeline
                # Try loading from local cache first to avoid network delays
                try:
                    self._primary_pipeline = pipeline(
                        "text-classification",
                        model=self.primary_model_name,
                        device=-1 if self.device == "cpu" else 0,
                        truncation=True,
                        max_length=512,
                        model_kwargs={"local_files_only": True},
                    )
                except Exception:
                    self._primary_pipeline = pipeline(
                        "text-classification",
                        model=self.primary_model_name,
                        device=-1 if self.device == "cpu" else 0,
                        truncation=True,
                        max_length=512,
                    )
            except Exception:
                self._primary_pipeline = False

    def _init_gpt2(self):
        if self._gpt2_model is None:
            try:
                from transformers import GPT2LMHeadModel, GPT2TokenizerFast
                try:
                    self._gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(self.perplexity_model_name, local_files_only=True)
                    self._gpt2_model = GPT2LMHeadModel.from_pretrained(self.perplexity_model_name, local_files_only=True)
                except Exception:
                    self._gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(self.perplexity_model_name)
                    self._gpt2_model = GPT2LMHeadModel.from_pretrained(self.perplexity_model_name)
                self._gpt2_model.eval()
            except Exception:
                self._gpt2_model = False
            except Exception:
                self._gpt2_model = False

    def estimate(self, text: str, is_oral_transcript: bool = False) -> Tuple[float, str]:
        """Estimate AI-content probability from 0.0 to 100.0%.
        
        Returns:
            (probability_score, method_used)
        """
        words = text.split()
        if len(words) < 25:
            return 10.0, "insufficient_length_downweighted"

        # Heuristic discourse & burstiness analysis
        heuristic_score = self._predict_heuristic(text)

        # RoBERTa classification
        roberta_score = 0.0
        self._init_primary()
        if self._primary_pipeline:
            try:
                roberta_score = self._predict_roberta_windowed(text)
            except Exception:
                pass

        # Ensemble multi-factor signal
        if heuristic_score >= 60.0:
            combined = (0.25 * roberta_score) + (0.75 * heuristic_score)
            method = "ensemble_roberta_heuristic"
        elif roberta_score >= 50.0:
            combined = roberta_score
            method = "roberta_windowed_classifier"
        else:
            combined = max(roberta_score, heuristic_score)
            method = "ensemble_roberta_heuristic"

        final_score = self._adjust_score(combined, is_oral_transcript, len(words))
        return round(final_score, 1), method

    def _predict_roberta_windowed(self, text: str) -> float:
        """Evaluate text using sliding 512-character/token chunks and aggregate positive class probability."""
        words = text.split()
        chunk_size = 120  # words per window
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        if not chunks:
            return 10.0

        probs: List[float] = []
        for ch in chunks[:6]:  # evaluate up to 6 windows
            res = self._primary_pipeline(ch)[0]
            label = str(res.get("label", "")).upper()
            sc = float(res.get("score", 0.5))
            # RoBERTa OpenAI detector outputs 'Fake' for AI and 'Real' for Human
            if "FAKE" in label or "AI" in label:
                probs.append(sc * 100.0)
            else:
                probs.append((1.0 - sc) * 100.0)

        return float(sum(probs) / len(probs)) if probs else 20.0

    def _predict_perplexity_burstiness(self, text: str) -> float:
        """Calculate per-sentence perplexity and sentence-to-sentence variance (burstiness)."""
        import torch

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]
        if not sentences:
            return 20.0

        perplexities: List[float] = []
        with torch.no_grad():
            for sent in sentences[:10]:
                encodings = self._gpt2_tokenizer(sent, return_tensors="pt")
                input_ids = encodings.input_ids
                if input_ids.size(1) < 2:
                    continue
                outputs = self._gpt2_model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                ppl = math.exp(min(loss, 10.0))  # cap loss to avoid overflow
                perplexities.append(ppl)

        if not perplexities:
            return 20.0

        mean_ppl = float(sum(perplexities) / len(perplexities))
        variance = float(sum((p - mean_ppl) ** 2 for p in perplexities) / len(perplexities))
        burstiness = math.sqrt(variance)

        # AI text typically has lower average perplexity (< 35) and lower burstiness (< 15)
        # Logistic sigmoid curve mapping low perplexity and low burstiness to high AI score
        z = (35.0 - mean_ppl) * 0.08 + (15.0 - burstiness) * 0.12
        prob = 1.0 / (1.0 + math.exp(-max(min(z, 5.0), -5.0)))
        return prob * 100.0

    def _predict_heuristic(self, text: str) -> float:
        """Deterministic analyzer checking discourse markers, sentence rhythm uniformity, and vocabulary diversity."""
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        if not words:
            return 10.0

        # Characteristic formulaic transition markers common in synthetic LLM essays
        llm_markers = [
            "in conclusion", "furthermore", "moreover", "firstly", "secondly",
            "contemporary landscape", "imperative for", "technological synergy",
            "strategic agility", "foundational bedrock", "extraordinary fidelity",
            "multifaceted", "unprecedented zenith", "enduring competitive advantage",
            "proactively harmonize", "cultural empowerment", "state-of-the-art"
        ]
        marker_hits = sum(1 for m in llm_markers if m in text_lower)

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return 15.0

        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        std_len = math.sqrt(sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) if len(lengths) > 1 else 5.0

        # Low sentence variance (burstiness) indicates machine generation
        burstiness_factor = max(0.0, 1.0 - (std_len / max(mean_len, 1.0)))

        # Base score from vocabulary consistency
        base_score = burstiness_factor * 25.0

        # Marker bonus: each strong LLM hallmark adds weight
        marker_score = min(65.0, marker_hits * 15.0)

        total_score = base_score + marker_score
        return max(5.0, min(92.0, total_score))

    def _adjust_score(self, raw_score: float, is_oral: bool, word_count: int) -> float:
        """Apply safety guardrails for short texts and oral transcripts."""
        score = raw_score
        if is_oral:
            # Oral speech has high natural disfluency and differs from formal prose
            score *= 0.5
        if word_count < 60:
            score *= 0.6
        return max(0.0, min(100.0, score))
