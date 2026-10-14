"""Scoring engine and multi-factor integrity risk classifier."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from src.integrity.models import RiskLevel, MatchRecord


def calculate_risk_level(
    peer_percentage: float,
    external_percentage: float,
    ai_probability: float,
    status: str = "PROCESSED",
    rules_config: Dict[str, Any] = None,
) -> Tuple[RiskLevel, str]:
    """Determine document risk level and generate an intuitive explanation.
    
    Risk Level Rules:
    - UNPROCESSED: Failed extraction or empty file.
    - HIGH: Peer >= 50%, or External >= 50%, or (AI >= 80% with peer/external >= 20%).
    - MEDIUM: Peer >= 25%, or External >= 25%, or AI >= 60%.
    - LOW: All indicators below medium thresholds.
    
    Returns:
        (risk_level, explanation_summary)
    """
    if status in ("FAILED", "UNPROCESSED"):
        return RiskLevel.UNPROCESSED, "File processing failed or file is corrupted/unreadable."

    peer = float(peer_percentage)
    ext = float(external_percentage)
    ai = float(ai_probability)

    reasons: List[str] = []

    # HIGH checks
    is_high = False
    if peer >= 50.0:
        is_high = True
        reasons.append(f"Critical internal peer similarity ({peer:.1f}%)")
    if ext >= 50.0:
        is_high = True
        reasons.append(f"Extensive external reference overlap ({ext:.1f}%)")
    if ai >= 80.0 and (peer >= 20.0 or ext >= 20.0):
        is_high = True
        reasons.append(f"High AI-generation probability ({ai:.1f}%) combined with copied sections")

    if is_high:
        explanation = f"Flagged as HIGH risk due to: {', '.join(reasons)}."
        return RiskLevel.HIGH, explanation

    # MEDIUM checks
    is_medium = False
    if peer >= 25.0:
        is_medium = True
        reasons.append(f"Moderate peer overlap ({peer:.1f}%)")
    if ext >= 25.0:
        is_medium = True
        reasons.append(f"Moderate external overlap ({ext:.1f}%)")
    if ai >= 60.0:
        is_medium = True
        reasons.append(f"Elevated AI-generation likelihood ({ai:.1f}%)")

    if is_medium:
        explanation = f"Flagged as MEDIUM risk due to: {', '.join(reasons)}."
        return RiskLevel.MEDIUM, explanation

    # LOW
    explanation = (
        f"Assessed as LOW risk. Independent originality verified. "
        f"Peer similarity: {peer:.1f}%, External match: {ext:.1f}%, AI likelihood: {ai:.1f}%."
    )
    return RiskLevel.LOW, explanation


def generate_document_explanation(
    doc_id: str,
    risk_level: RiskLevel,
    peer_percentage: float,
    external_percentage: float,
    ai_probability: float,
    peer_matches: List[MatchRecord],
    multimodal_matches: List[MatchRecord],
    external_matches: List[MatchRecord],
) -> str:
    """Build a comprehensive human-readable explanation matching the brief's sample output."""
    sentences: List[str] = []

    if peer_matches:
        top_m = peer_matches[0]
        sentences.append(
            f"High internal peer similarity detected against {top_m.matched_file} "
            f"(score: {top_m.similarity_score:.2f}, classification: {top_m.match_type})."
        )
    elif peer_percentage > 0:
        sentences.append(f"Internal peer overlap identified at {peer_percentage:.1f}%.")
    else:
        sentences.append("No significant internal peer similarity detected.")

    if multimodal_matches:
        mm = multimodal_matches[0]
        sentences.append(
            f"Cross-modal match identified with {mm.matched_file} "
            f"at timestamp {mm.location.get('matched_timestamp', mm.location.get('query_timestamp', 'N/A'))}."
        )

    if external_matches:
        em = external_matches[0]
        sentences.append(
            f"External reference match identified against {em.matched_file} ({external_percentage:.1f}% overlap)."
        )
    else:
        sentences.append("No external database matches found.")

    if ai_probability >= 70.0:
        sentences.append(f"Elevated AI-generated content probability ({ai_probability:.1f}%).")
    else:
        sentences.append("AI likelihood low to moderate.")

    return " ".join(sentences)
