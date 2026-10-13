"""Data models for the Multimodal Integrity Engine."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Modality(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    VIDEO = "video"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNPROCESSED = "UNPROCESSED"


class ExtractionStatus(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    FAILED = "FAILED"


@dataclass
class Segment:
    """An atomic text unit extracted from a source document."""
    text: str
    doc_id: str
    modality: str
    page: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    paragraph_index: Optional[int] = None
    heading: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of parsing and extracting text from a file."""
    doc_id: str
    file_path: str
    segments: List[Segment] = field(default_factory=list)
    status: ExtractionStatus = ExtractionStatus.OK
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A semantic chunk of sentences with overlapping boundaries."""
    chunk_id: str
    doc_id: str
    modality: str
    text: str
    normalized_text: str
    sentences: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    paragraph_start: Optional[int] = None
    paragraph_end: Optional[int] = None
    word_count: int = 0
    is_template: bool = False

    def formatted_location(self) -> Dict[str, Any]:
        """Convert chunk coordinates to location dictionary for reports."""
        loc: Dict[str, Any] = {}
        if self.page_start is not None:
            if self.page_start == self.page_end or self.page_end is None:
                loc["page"] = self.page_start
            else:
                loc["page_range"] = f"{self.page_start}-{self.page_end}"
        if self.start_sec is not None and self.end_sec is not None:
            loc["timestamp"] = f"{_format_ts(self.start_sec)} - {_format_ts(self.end_sec)}"
        if self.paragraph_start is not None:
            loc["paragraph"] = self.paragraph_start
        return loc


def _format_ts(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    total_sec = int(round(seconds))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class MatchRecord:
    """A detected similarity match between two chunks or sentences."""
    matched_file: str
    similarity_score: float
    query_passage: str
    matched_passage: str
    location: Dict[str, Any]
    match_type: str = "exact"  # "exact", "near-exact", "paraphrase"
    lexical_score: float = 0.0
    jaccard_score: float = 0.0
    is_multimodal: bool = False
    query_chunk_id: Optional[str] = None
    matched_chunk_id: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        """Serializes to the schema specified in the assessment brief."""
        return {
            "matched_file": self.matched_file,
            "similarity_score": round(float(self.similarity_score), 2),
            "query_passage": self.query_passage,
            "matched_passage": self.matched_passage,
            "location": self.location,
            "match_type": self.match_type,
            "lexical_score": round(float(self.lexical_score), 2),
            "jaccard_score": round(float(self.jaccard_score), 2),
        }


@dataclass
class DocumentReport:
    """Complete integrity verification report for a single submission."""
    document_id: str
    total_integrity_risk: RiskLevel
    peer_similarity_percentage: float
    external_similarity_percentage: float
    ai_generated_probability: float
    peer_matches: List[MatchRecord] = field(default_factory=list)
    multimodal_matches: List[MatchRecord] = field(default_factory=list)
    external_matches: List[MatchRecord] = field(default_factory=list)
    template_text_discounted: List[str] = field(default_factory=list)
    explanations: str = ""
    warnings: List[str] = field(default_factory=list)
    status: str = "PROCESSED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to schema required by the assessment."""
        return {
            "document_id": self.document_id,
            "total_integrity_risk": self.total_integrity_risk.value if isinstance(self.total_integrity_risk, RiskLevel) else str(self.total_integrity_risk),
            "scores": {
                "peer_similarity_percentage": round(self.peer_similarity_percentage, 1),
                "external_similarity_percentage": round(self.external_similarity_percentage, 1),
                "ai_generated_probability": round(self.ai_generated_probability, 1),
            },
            "peer_matches": [m.to_json_dict() for m in self.peer_matches],
            "multimodal_matches": [m.to_json_dict() for m in self.multimodal_matches],
            "external_matches": [m.to_json_dict() for m in self.external_matches],
            "template_text_discounted": self.template_text_discounted,
            "warnings": self.warnings,
            "status": self.status,
            "explanations": self.explanations,
        }
