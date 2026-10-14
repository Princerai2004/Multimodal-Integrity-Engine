# Multimodal Integrity Engine: Batch Evaluation Summary

**Total Documents Evaluated:** 15

## Overview Metrics

- 🚨 **High Risk Flags:** 5
- ⚠️ **Medium Risk Flags:** 2
- 🟢 **Low Risk (Clean):** 6
- ⛔ **Unprocessed / Warnings:** 2

## Submission Status & Scores

| Document ID | Status | Risk Level | Peer Sim % | External % | AI Prob % | Top Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| [`docx_01_original_energy.docx`](docx_01_original_energy.docx.md) | PROCESSED | **HIGH** | 77.3% | 0.0% | 17.9% | `docx_02_paraphrase_of_01.docx` |
| [`docx_02_paraphrase_of_01.docx`](docx_02_paraphrase_of_01.docx.md) | PROCESSED | **HIGH** | 77.5% | 0.0% | 18.5% | `docx_01_original_energy.docx` |
| [`docx_03_template_answer_b.docx`](docx_03_template_answer_b.docx.md) | PROCESSED | LOW | 0.0% | 0.0% | 13.8% | `-` |
| [`docx_04_template_answer_c.docx`](docx_04_template_answer_c.docx.md) | PROCESSED | LOW | 0.0% | 0.0% | 9.8% | `-` |
| [`docx_05_template_answer_d.docx`](docx_05_template_answer_d.docx.md) | PROCESSED | LOW | 0.0% | 0.0% | 15.0% | `-` |
| [`mp4_01_reads_pdf_03.mp4`](mp4_01_reads_pdf_03.mp4.md) | PROCESSED | **HIGH** | 100.0% | 0.0% | 7.3% | `pdf_03_original_retail.pdf` |
| [`mp4_02_original_pitch.mp4`](mp4_02_original_pitch.mp4.md) | PROCESSED | LOW | 0.0% | 0.0% | 3.0% | `-` |
| [`mp4_03_silent.mp4`](mp4_03_silent.mp4.md) | UNPROCESSED | UNPROCESSED | 0.0% | 0.0% | 0.0% | `-` |
| [`pdf_01_original_logistics.pdf`](pdf_01_original_logistics.pdf.md) | PROCESSED | **HIGH** | 100.0% | 0.0% | 18.5% | `pdf_02_exact_copy_of_01.pdf` |
| [`pdf_02_exact_copy_of_01.pdf`](pdf_02_exact_copy_of_01.pdf.md) | PROCESSED | **HIGH** | 100.0% | 0.0% | 18.5% | `pdf_01_original_logistics.pdf` |
| [`pdf_03_original_retail.pdf`](pdf_03_original_retail.pdf.md) | PROCESSED | MEDIUM | 26.1% | 0.0% | 20.9% | `mp4_01_reads_pdf_03.mp4` |
| [`pdf_04_llm_generated.pdf`](pdf_04_llm_generated.pdf.md) | PROCESSED | MEDIUM | 0.0% | 0.0% | 64.5% | `-` |
| [`pdf_05_template_answer_a.pdf`](pdf_05_template_answer_a.pdf.md) | PROCESSED | LOW | 0.0% | 0.0% | 13.3% | `-` |
| [`pdf_06_empty.pdf`](pdf_06_empty.pdf.md) | UNPROCESSED | UNPROCESSED | 0.0% | 0.0% | 0.0% | `-` |
| [`pdf_07_template_answer_e.pdf`](pdf_07_template_answer_e.pdf.md) | PROCESSED | LOW | 0.0% | 0.0% | 12.0% | `-` |

## Cross-Document Peer Similarity Matrix (%)

| Document | `docx_01_original_energy.docx` | `docx_02_paraphrase_of_01.docx` | `docx_03_template_answer_b.docx` | `docx_04_template_answer_c.docx` | `docx_05_template_answer_d.docx` | `mp4_01_reads_pdf_03.mp4` | `mp4_02_original_pitch.mp4` | `pdf_01_original_logistics.pdf` | `pdf_02_exact_copy_of_01.pdf` | `pdf_03_original_retail.pdf` | `pdf_04_llm_generated.pdf` | `pdf_05_template_answer_a.pdf` | `pdf_07_template_answer_e.pdf` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `docx_01_original_energy.docx` | 100% | **77%** | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `docx_02_paraphrase_of_01.docx` | **78%** | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `docx_03_template_answer_b.docx` | 0% | 0% | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `docx_04_template_answer_c.docx` | 0% | 0% | 0% | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `docx_05_template_answer_d.docx` | 0% | 0% | 0% | 0% | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `mp4_01_reads_pdf_03.mp4` | 0% | 0% | 0% | 0% | 0% | 100% | 0% | 0% | 0% | **100%** | 0% | 0% | 0% |
| `mp4_02_original_pitch.mp4` | 0% | 0% | 0% | 0% | 0% | 0% | 100% | 0% | 0% | 0% | 0% | 0% | 0% |
| `pdf_01_original_logistics.pdf` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 100% | **100%** | 0% | 0% | 0% | 0% |
| `pdf_02_exact_copy_of_01.pdf` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **100%** | 100% | 0% | 0% | 0% | 0% |
| `pdf_03_original_retail.pdf` | 0% | 0% | 0% | 0% | 0% | 26% | 0% | 0% | 0% | 100% | 0% | 0% | 0% |
| `pdf_04_llm_generated.pdf` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 100% | 0% | 0% |
| `pdf_05_template_answer_a.pdf` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 100% | 0% |
| `pdf_07_template_answer_e.pdf` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 100% |