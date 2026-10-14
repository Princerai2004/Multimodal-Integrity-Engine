"""Report generation for JSON schema compliance and 30-second human Markdown auditing."""

from __future__ import annotations
import difflib
import json
import os
import re
from typing import Any, Dict, List
import numpy as np

from src.integrity.models import DocumentReport, MatchRecord, RiskLevel


def highlight_word_overlap(query_text: str, matched_text: str) -> str:
    """Highlight overlapping words using difflib so a reviewer's eye jumps straight to the match."""
    words_q = re.findall(r"\S+|\s+", query_text)
    words_m = re.findall(r"\S+|\s+", matched_text)

    seq = difflib.SequenceMatcher(None, [w.lower() for w in words_q], [w.lower() for w in words_m])
    highlighted: List[str] = []

    for opcode, a0, a1, b0, b1 in seq.get_opcodes():
        piece = "".join(words_q[a0:a1])
        if opcode == "equal" and len(piece.strip()) > 2:
            highlighted.append(f"**{piece}**")
        else:
            highlighted.append(piece)

    return "".join(highlighted)


class ReportGenerator:
    """Generates per-document JSON reports, visual Markdown reports, and cross-batch summaries."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_json_report(self, report: DocumentReport) -> str:
        """Write JSON report strictly following the schema specified in the brief."""
        out_path = os.path.join(self.output_dir, f"{report.document_id}.json")
        data = report.to_dict()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return out_path

    def write_markdown_report(self, report: DocumentReport) -> str:
        """Write human-readable Markdown report optimized for 30-second audit verification."""
        out_path = os.path.join(self.output_dir, f"{report.document_id}.md")

        risk_emoji = {
            RiskLevel.HIGH: "🚨 **HIGH INTEGRITY RISK**",
            RiskLevel.MEDIUM: "⚠️ **MEDIUM INTEGRITY RISK**",
            RiskLevel.LOW: "✅ **LOW INTEGRITY RISK (ORIGINAL)**",
            RiskLevel.UNPROCESSED: "⛔ **UNPROCESSED / CORRUPTED**",
        }.get(report.total_integrity_risk, str(report.total_integrity_risk))

        md: List[str] = []
        md.append(f"# Integrity Verification Audit: `{report.document_id}`\n")
        md.append(f"**Risk Assessment:** {risk_emoji}\n")
        md.append(f"> **Executive Summary:** {report.explanations}\n")

        # Score Breakdown Table
        md.append("## 1. Quantitative Risk Breakdown\n")
        md.append("| Signal Category | Score | Severity Thresholds | Indicator |")
        md.append("| :--- | :---: | :---: | :---: |")

        peer_ind = "🔴 High" if report.peer_similarity_percentage >= 50 else ("🟡 Moderate" if report.peer_similarity_percentage >= 25 else "🟢 Clean")
        ext_ind = "🔴 High" if report.external_similarity_percentage >= 50 else ("🟡 Moderate" if report.external_similarity_percentage >= 25 else "🟢 Clean")
        ai_ind = "🔴 High" if report.ai_generated_probability >= 75 else ("🟡 Moderate" if report.ai_generated_probability >= 60 else "🟢 Natural")

        md.append(f"| **Internal Peer-to-Peer Similarity** | **{report.peer_similarity_percentage:.1f}%** | ≥50% High, ≥25% Med | {peer_ind} |")
        md.append(f"| **External Knowledge Reference Match** | **{report.external_similarity_percentage:.1f}%** | ≥50% High, ≥25% Med | {ext_ind} |")
        md.append(f"| **AI-Generated Likelihood Probability** | **{report.ai_generated_probability:.1f}%** | ≥75% High, ≥60% Med | {ai_ind} |\n")

        # Top Overlaps Section
        all_matches = report.peer_matches + report.multimodal_matches + report.external_matches
        all_matches.sort(key=lambda m: m.similarity_score, reverse=True)
        top_matches = all_matches[:10]

        md.append("## 2. Evidence & Explainable Overlap Matches (Top 10)\n")
        if not top_matches:
            md.append("*No significant suspicious passages detected above similarity threshold.*\n")
        else:
            for idx, m in enumerate(top_matches, start=1):
                m_category = "🎙️ Multimodal (Speech-to-Text)" if m.is_multimodal else ("🌐 External Source" if m in report.external_matches else "📄 Peer Document")
                md.append(f"### Match #{idx} — [{m.match_type.upper()}] {m_category}")
                md.append(f"- **Matched Target:** `{m.matched_file}`")
                md.append(f"- **Cosine Similarity:** `{m.similarity_score:.2f}` | **Lexical Match Ratio:** `{m.lexical_score:.2f}` | **3-Gram Jaccard:** `{m.jaccard_score:.2f}`")
                
                # Format location neatly
                loc_parts = []
                for k, v in m.location.items():
                    loc_parts.append(f"**{k.replace('_', ' ').title()}:** `{v}`")
                md.append(f"- **Coordinates:** {', '.join(loc_parts)}\n")

                # Passages with visual diff
                q_diff = highlight_word_overlap(m.query_passage, m.matched_passage)
                md.append(f"**Query Passage (`{report.document_id}`):**\n> {q_diff}\n")
                md.append(f"**Matched Source Passage (`{m.matched_file}`):**\n> {m.matched_passage}\n")
                md.append("---\n")

        # Template text discounted section
        md.append("## 3. Common Template & Boilerplate Discounted\n")
        if report.template_text_discounted:
            md.append("The following shared prompt headers or institutional boilerplate were isolated and excluded from plagiarism scoring:\n")
            for t_item in report.template_text_discounted:
                md.append(f"- {t_item}")
            md.append("")
        else:
            md.append("*No repetitive boilerplate identified.*\n")

        # Warnings section
        md.append("## 4. Processing Warnings & Diagnostics\n")
        if report.warnings:
            for w in report.warnings:
                md.append(f"- ⚠️ {w}")
            md.append("")
        else:
            md.append("✅ *Clean extraction with zero errors.*\n")

        # Limitations Footer
        md.append("## 5. Methodological Limitations & Model Notice\n")
        md.append(
            "> [!NOTE]\n"
            "> **AI Estimation Notice:** AI content classifiers output probabilistic estimates and can exhibit "
            "elevated false-positive rates on formal, academic, or non-native English prose. Scores should serve "
            "as advisory indicators for human review rather than definitive determinations.\n"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        return out_path

    def write_batch_summary(
        self,
        reports: List[DocumentReport],
        doc_matrix: np.ndarray,
        doc_ids: List[str],
    ) -> Tuple[str, str]:
        """Write aggregate summary.csv and summary.md spanning the entire evaluated batch."""
        csv_path = os.path.join(self.output_dir, "summary.csv")
        md_path = os.path.join(self.output_dir, "summary.md")

        # 1. Write CSV
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Document_ID,Status,Risk_Level,Peer_Similarity_%,External_Match_%,AI_Probability_%,Top_Matched_File,Warnings\n")
            for r in reports:
                top_hit = r.peer_matches[0].matched_file if r.peer_matches else (r.multimodal_matches[0].matched_file if r.multimodal_matches else "None")
                warn_str = "; ".join(r.warnings) if r.warnings else "None"
                risk_str = r.total_integrity_risk.value if isinstance(r.total_integrity_risk, RiskLevel) else str(r.total_integrity_risk)
                f.write(
                    f'"{r.document_id}","{r.status}","{risk_str}",{r.peer_similarity_percentage:.1f},'
                    f'{r.external_similarity_percentage:.1f},{r.ai_generated_probability:.1f},'
                    f'"{top_hit}","{warn_str}"\n'
                )

        # 2. Write Markdown Summary
        md: List[str] = []
        md.append("# Multimodal Integrity Engine: Batch Evaluation Summary\n")
        md.append(f"**Total Documents Evaluated:** {len(reports)}\n")

        high_count = sum(1 for r in reports if r.total_integrity_risk == RiskLevel.HIGH)
        med_count = sum(1 for r in reports if r.total_integrity_risk == RiskLevel.MEDIUM)
        low_count = sum(1 for r in reports if r.total_integrity_risk == RiskLevel.LOW)
        unproc_count = sum(1 for r in reports if r.total_integrity_risk == RiskLevel.UNPROCESSED)

        md.append("## Overview Metrics\n")
        md.append(f"- 🚨 **High Risk Flags:** {high_count}")
        md.append(f"- ⚠️ **Medium Risk Flags:** {med_count}")
        md.append(f"- 🟢 **Low Risk (Clean):** {low_count}")
        md.append(f"- ⛔ **Unprocessed / Warnings:** {unproc_count}\n")

        md.append("## Submission Status & Scores\n")
        md.append("| Document ID | Status | Risk Level | Peer Sim % | External % | AI Prob % | Top Match |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

        for r in reports:
            top_hit = r.peer_matches[0].matched_file if r.peer_matches else (r.multimodal_matches[0].matched_file if r.multimodal_matches else "-")
            risk_val = r.total_integrity_risk.value if isinstance(r.total_integrity_risk, RiskLevel) else str(r.total_integrity_risk)
            badge = f"**{risk_val}**" if risk_val == "HIGH" else risk_val
            md.append(
                f"| [`{r.document_id}`]({r.document_id}.md) | {r.status} | {badge} | "
                f"{r.peer_similarity_percentage:.1f}% | {r.external_similarity_percentage:.1f}% | "
                f"{r.ai_generated_probability:.1f}% | `{top_hit}` |"
            )

        # Document-by-Document Similarity Matrix (Scenario 5 requirement)
        md.append("\n## Cross-Document Peer Similarity Matrix (%)\n")
        if len(doc_ids) > 0 and doc_matrix.shape[0] == len(doc_ids):
            header = "| Document | " + " | ".join(f"`{d}`" for d in doc_ids) + " |"
            sep = "| :--- | " + " | ".join([":---:"] * len(doc_ids)) + " |"
            md.append(header)
            md.append(sep)

            for i, doc_row in enumerate(doc_ids):
                row_vals = []
                for j in range(len(doc_ids)):
                    val = doc_matrix[i, j]
                    val_str = f"{val:.0f}%"
                    if i != j and val >= 50.0:
                        val_str = f"**{val_str}**"
                    row_vals.append(val_str)
                md.append(f"| `{doc_row}` | " + " | ".join(row_vals) + " |")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        return csv_path, md_path
