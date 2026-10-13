"""Text cleaning, normalization, and lexical similarity utilities."""

from __future__ import annotations
import re
import unicodedata
from typing import Set


def clean_text(text: str) -> str:
    """Perform robust text cleaning and normalization.
    
    1. Normalizes unicode characters (NFKC).
    2. Reconnects hyphenated words split across line breaks (e.g. multi-\\nmodal -> multimodal).
    3. Removes non-printable and control characters while preserving standard spacing.
    4. Normalizes whitespace, carriage returns, and excessive newlines.
    """
    if not text:
        return ""

    # Normalize unicode to standard form
    text = unicodedata.normalize("NFKC", text)

    # Rejoin hyphenated line breaks (e.g., "intel-\\nligence" -> "intelligence")
    text = re.sub(r"(\b[a-zA-Z]+)-\s*\r?\n\s*([a-zA-Z]+\b)", r"\1\2", text)

    # Remove non-printable control characters (keep \n, \t)
    text = "".join(ch for ch in text if ch in ("\n", "\t") or (unicodedata.category(ch)[0] != "C" and ord(ch) >= 32))

    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple inline spaces/tabs to single space
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Generate a lightly normalized lowercase string for semantic & lexical matching."""
    cleaned = clean_text(text)
    # Lowercase and clean boundary whitespace
    lowered = cleaned.lower()
    # Normalize punctuation spacing
    normalized = re.sub(r"\s+", " ", lowered)
    return normalized.strip()


def get_word_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract word n-grams from text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def compute_word_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute word n-gram Jaccard similarity between two texts."""
    ngrams_a = get_word_ngrams(text_a, n=n)
    ngrams_b = get_word_ngrams(text_b, n=n)

    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = len(ngrams_a.intersection(ngrams_b))
    union = len(ngrams_a.union(ngrams_b))

    return float(intersection / union) if union > 0 else 0.0
