#!/usr/bin/env python3
"""Synthetic Dataset & Reference Corpus Generator for Multimodal Integrity Engine.

Generates:
- 15 Diverse test files in sample_submissions/ (7 PDFs, 5 DOCX, 3 MP4s)
- Embedded ground-truth relationships covering Scenarios 1 to 5
- 12 Simulated external knowledge base articles in reference_corpus/
- sample_submissions/dataset_manifest.json with verified ground truth
"""

import json
import os
import shutil
import subprocess
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfgen import canvas
import docx


TEMPLATE_PROMPT = (
    "CASE PROMPT & EVALUATION DIRECTIVES: "
    "Please conduct a comprehensive operational, financial, and strategic evaluation of the proposed transformation plan. "
    "Your analysis must explicitly assess capital allocation constraints, supply chain bottlenecks, organizational change risks, "
    "and long-term customer lifetime value impact. All quantitative projections must be rigorously grounded in verified industry benchmarks. "
    "Submissions omitting regulatory compliance considerations or exceeding the designated scope will face structural score deductions. "
    "As articulated by Porter (1985), sustained competitive advantage demands coherent alignment between operational effectiveness "
    "and differentiated strategic positioning. In addition, Christensen (1997) demonstrates that incumbents routinely misjudge disruptive "
    "technologies until legacy profit centers are permanently compromised. Synthesize these frameworks into your final boardroom recommendations."
)

TEXT_LOGISTICS_ORIGINAL = [
    "Executive Overview: Modern supply chain networks are navigating unprecedented cost pressures driven by escalating fuel volatility, driver retention deficits, and legacy telematics fragmentation. Over the past four quarters, line-haul carrier expenses expanded by 14.8 percent while on-time delivery metrics contracted across Tier-1 distribution corridors.",
    "Root Cause Analysis: Fleet telemetry logs reveal that empty backhaul miles currently constitute 22.4 percent of aggregate route mileage. This deadhead inefficiency consumes approximately $4.2 million in unrecoverable operational overhead annually. In addition, preventative maintenance schedules adhere to arbitrary 90-day intervals rather than real-time engine telemetry, incurring repetitive mechanical failures on primary transit routes.",
    "Proposed Strategic Roadmap: We recommend deploying a decentralized automated dispatch engine integrated with dynamic load-matching exchanges. By synchronizing freight brokerage APIs with onboard electronic logging devices, the logistics fleet can compress deadhead transit to under 9.5 percent within two quarters of phased rollout.",
    "Financial Feasibility and ROI: Capital requirements for IoT telematics hardware and system retrofits total $1.85 million. Projected operating expense savings from fuel reduction and optimized routing reach $3.4 million in Year 1, yielding an estimated net present value of $5.2 million over a 36-month investment horizon."
]

TEXT_RETAIL_ORIGINAL = [
    "Market Context and Omnichannel Imperative: Urban retail operators are experiencing a structural shift in customer purchasing behavior, characterized by diminishing brick-and-mortar footfall and surging expectations for rapid fulfillment. Store replenishment cycles that previously functioned on weekly cadences now fail to sustain inventory velocity in dense metropolitan trade areas.",
    "Operating margins contracted by 4.2 percent across the third quarter due to acute supply chain bottlenecks, inventory holding surcharges, and escalating last-mile carrier fees. To reverse this margin compression, regional distribution centers must transition toward automated micro-fulfillment centers capable of dispatching same-day orders within a five-mile radius.",
    "Technology and Infrastructure Architecture: Deploying automated storage and retrieval systems within urban retail footprints reduces order picking latency from 45 minutes to under 7 minutes per basket. Real-time radio-frequency inventory tracking eliminates phantom stockouts, driving on-shelf availability to 98.7 percent across high-velocity product categories.",
    "Capital Budget and Five-Year Horizon: Upgrading seven pilot retail locations requires an initial capital expenditure of $3.6 million. Expected customer retention improvements and fulfillment cost reductions generate an anticipated payback period of 18 months, strengthening consolidated EBITDA by 280 basis points."
]

TEXT_LLM_ESSAY = [
    "In the contemporary landscape of global enterprise management, the imperative for strategic agility and technological synergy has reached an unprecedented zenith. Organizations across diverse industrial sectors are increasingly confronted with multifaceted disruptions that necessitate comprehensive paradigms of innovation and operational excellence.",
    "Firstly, it is essential to recognize that data-driven decision architectures represent the foundational bedrock of sustainable competitive differentiation. By systematically leveraging predictive analytics and machine learning methodologies, forward-looking enterprises can anticipate consumer demand shifts with extraordinary fidelity. Furthermore, cultivating cross-functional collaboration fosters organizational resilience and accelerates time-to-market across critical product cycles.",
    "Moreover, an examination of contemporary organizational dynamics reveals that digital transformation initiatives frequently falter due to inadequate change management protocols rather than intrinsic technological limitations. It is therefore crucial for senior leadership to champion empathetic communication and comprehensive upskilling programs. In conclusion, organizations that proactively harmonize cultural empowerment with state-of-the-art technological infrastructures will inevitably secure an enduring competitive advantage in an evolving global economy."
]

TEXT_ENERGY_ORIGINAL = [
    "Strategic Assessment of Decentralized Energy Systems: Commercial and industrial facilities face mounting grid reliability challenges, surging peak-demand tariff structures, and escalating carbon compliance mandates. Conventional utility supply frameworks no longer provide the resilience or price predictability required for mission-critical industrial manufacturing operations.",
    "Microgrid Integration and Levelized Cost of Energy: Deploying on-site hybrid solar photovoltaic arrays paired with lithium-iron-phosphate battery storage reduces levelized cost of energy from $0.142 per kilowatt-hour to $0.086 per kilowatt-hour over a 20-year operational lifecycle. Smart inverters enable instantaneous islanding during grid voltage fluctuations, eliminating production stoppages.",
    "Battery Degradation and Thermal Management: Cell degradation models demonstrate that active liquid cooling maintains lithium-iron-phosphate battery capacity above 82 percent after 6,000 equivalent full cycles. Implementing intelligent state-of-charge limits between 15 percent and 85 percent prevents premature electrolyte breakdown and extends operational asset longevity.",
    "Economic Valuation and Payback Analysis: The turnkey microgrid installation requires a gross capital investment of $2.4 million, offset by a 30 percent federal investment tax credit. Net annual energy cost avoidance of $480,000 yields an unsubsidized payback period of 3.5 years, generating a project internal rate of return of 21.4 percent."
]

TEXT_ENERGY_PARAPHRASE = [
    "Evaluation of Distributed Clean Energy Systems: Commercial and industrial manufacturing plants encounter power grid instability, rising peak power costs, and stringent carbon reduction regulations. Traditional utility contracts fail to guarantee the operational continuity and price predictability required for mission-critical manufacturing facilities.",
    "On-Site Solar Microgrids and Levelized Energy Pricing: Constructing distributed photovoltaic arrays coupled with lithium-iron-phosphate energy storage decreases levelized energy costs from 14.2 cents to 8.6 cents per kilowatt-hour across a twenty-year service life. Intelligent inverters facilitate automated microgrid islanding whenever voltage dips occur, preventing plant downtime.",
    "Battery Degradation Modeling and Liquid Cooling: Thermal management simulations reveal that active liquid circulation preserves lithium-iron-phosphate battery capacity beyond 82 percent following 6,000 deep discharge cycles. Regulating operating battery charge strictly within 15 to 85 percent averts accelerated cell aging and prolongs system working life.",
    "Capital Expenditures and Return on Investment: Building the decentralized microgrid requires a total initial capital expense of $2.4 million before a thirty percent clean energy tax credit. Yearly utility bill savings of $480,000 produce an unsubsidized capital payback in 3.5 years, producing an internal rate of return of 21.4 percent."
]

TEXT_ANSWERS = {
    "A": "Operational Analysis on Supply Chain Automation: The proposed micro-fulfillment network requires standardizing SKU packaging across regional vendors to enable robotic sortation. Fleet route telemetry confirms that urban transit corridors experience the highest fuel burn rates during morning congestion windows. Implementing off-peak deliveries to retail lockers reduces transport emissions by 18 percent and cuts line-haul delays.",
    "B": "Financial Structuring and Debt Modeling: Financing the transition requires blending senior secured equipment leases with municipal green development bonds. Maintaining an interest coverage ratio above 4.5x ensures credit rating stability while shielding cash flows from interest rate volatility. Sensitivity analysis shows the model remains cash-flow positive under a 20 percent decline in base volume.",
    "C": "Customer Segmentation and Lifetime Value: Customer cohort analysis reveals that delivery windows under two hours increase 90-day repeat purchasing rates by 34 percent. Net Promoter Scores among urban subscribers correlate directly with packaging integrity and delivery punctuality. Reallocating marketing budgets toward retention guarantees improved customer acquisition efficiency.",
    "D": "Regulatory Compliance and Workforce Safety: Automating regional sorting facilities introduces occupational safety compliance requirements under revised OSHA industrial automation guidelines. Establishing joint safety committees and providing certified cobot maintenance training programs mitigates union grievances and preserves labor harmony during rollout.",
    "E": "Competitive Moats and Vendor Bargaining Power: Securing proprietary long-term supply contracts with Tier-1 battery cell manufacturers insulates the business from international supply chain shocks. Dual-sourcing critical electronic components lowers component pricing by 12 percent while avoiding dependency on single-source suppliers."
}


class NumberedCanvas(canvas.Canvas):
    """Canvas supporting running headers and page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self.header_text = "Integrity Engine Document Assessment"
        self.footer_prefix = "Page"

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))
        # Header (top margin)
        self.drawString(54, 750, self.header_text)
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        # Footer (bottom margin)
        self.line(54, 45, 558, 45)
        page_str = f"{self.footer_prefix} {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 32, page_str)
        self.drawString(54, 32, "Confidential Examination Submission")
        self.restoreState()


def create_pdf(file_path: str, paragraphs: List[str], header_title: str, footer_prefix: str = "Page"):
    """Create a formatted PDF across 2 pages with headers and footers."""
    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=14,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=12,
    )

    story = [
        Paragraph(header_title, title_style),
        Spacer(1, 10),
    ]

    # Distribute paragraphs across at least 2 pages
    mid = len(paragraphs) // 2
    for p in paragraphs[:mid]:
        story.append(Paragraph(p, body_style))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    for p in paragraphs[mid:]:
        story.append(Paragraph(p, body_style))
        story.append(Spacer(1, 6))

    def make_canvas(*args, **kwargs):
        c = NumberedCanvas(*args, **kwargs)
        c.header_text = header_title
        c.footer_prefix = footer_prefix
        return c

    doc.build(story, canvasmaker=make_canvas)


def create_docx(file_path: str, title: str, paragraphs: List[str]):
    """Create a structured DOCX file with headings and body paragraphs."""
    doc = docx.Document()
    doc.add_heading(title, level=1)

    for p in paragraphs:
        doc.add_paragraph(p)

    doc.save(file_path)


def synthesize_speech_wav(text: str, out_wav_path: str) -> bool:
    """Use Windows SAPI via PowerShell to synthesize high-quality speech WAV."""
    clean_speech_text = text.replace('"', '""').replace("'", "''")
    ps_cmd = f"""
    Add-Type -AssemblyName System.Speech;
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
    $synth.Rate = 0;
    $synth.SetOutputToWaveFile('{out_wav_path}');
    $synth.Speak('{clean_speech_text}');
    $synth.Dispose();
    """
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=25,
        )
        return res.returncode == 0 and os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 0
    except Exception:
        return False


def _get_ffmpeg_bin() -> str:
    """Find ffmpeg executable in PATH or winget installation directories."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(local_app_data, "Microsoft", "WinGet", "Packages"),
        r"C:\Program Files\ffmpeg\bin",
        r"C:\ffmpeg\bin",
    ]
    for cdir in candidates:
        if os.path.exists(cdir):
            for root, _, files in os.walk(cdir):
                if "ffmpeg.exe" in [f.lower() for f in files]:
                    return os.path.join(root, "ffmpeg.exe")
    return "ffmpeg"


def create_video_from_audio(wav_path: str, out_mp4_path: str, title: str) -> bool:
    """Combine speech audio with a static presentation canvas using ffmpeg."""
    ffmpeg_bin = _get_ffmpeg_bin()
    cover_png = out_mp4_path.replace(".mp4", "_cover.png")
    # Generate cover canvas
    fallback_cover_cmd = [
        ffmpeg_bin, "-y",
        "-f", "lavfi",
        "-i", "color=c=navy:s=640x360:d=1",
        "-frames:v", "1",
        cover_png,
    ]

    try:
        subprocess.run(fallback_cover_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    except Exception as e:
        print(f"Error creating cover: {e}")

    # Encode MP4
    mp4_cmd = [
        ffmpeg_bin, "-y",
        "-loop", "1",
        "-i", cover_png,
        "-i", wav_path,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-pix_fmt", "yuv420p",
        out_mp4_path,
    ]
    try:
        subprocess.run(mp4_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return os.path.exists(out_mp4_path) and os.path.getsize(out_mp4_path) > 0
    finally:
        if os.path.exists(cover_png):
            try:
                os.remove(cover_png)
            except OSError:
                pass


def create_silent_video(out_mp4_path: str) -> bool:
    """Create a silent video without audio streams using ffmpeg lavfi."""
    ffmpeg_bin = _get_ffmpeg_bin()
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "lavfi",
        "-i", "color=c=black:s=640x360:d=10",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4_path,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        return res.returncode == 0 and os.path.exists(out_mp4_path)
    except Exception:
        return False


def build_reference_corpus(ref_dir: str):
    """Generate 12 external simulated reference articles in reference_corpus/."""
    os.makedirs(ref_dir, exist_ok=True)

    articles = {
        "ref_01_supply_chain_logistics.txt": (
            "Supply Chain Telematics and Logistics Management. "
            "Modern freight logistics relies heavily on telematics and route optimization engines. "
            "Fleet efficiency is benchmarked against fuel economy, driver fatigue cycles, and empty deadhead mileage. "
            "Integrating GPS routing with automated warehouse sortation substantially lowers transit latency."
        ),
        "ref_02_freight_telematics.txt": (
            "Freight Optimization Protocols in North American Transit. "
            "Long-haul fleet logistics suffers from high operational friction caused by manual scheduling. "
            "Dynamic dispatch software pairs empty outbound tractor-trailers with return freights to boost asset productivity."
        ),
        "ref_03_renewable_microgrids.txt": (
            "Microgrid Integration and Clean Energy Decentralization. "
            "Decentralized power networks leverage hybrid solar PV generation and battery energy storage. "
            "The levelized cost of energy (LCOE) has declined steadily over the past decade, making industrial microgrids economically compelling."
        ),
        "ref_04_battery_storage_chemistry.txt": (
            "Lithium-Iron-Phosphate (LFP) Battery Lifecycle Management. "
            "LFP chemistry delivers superior thermal stability and extended cycle life compared to nickel-manganese chemistries. "
            "Active thermal regulation minimizes degradation during heavy cycling in industrial energy storage."
        ),
        "ref_05_retail_omnichannel_fulfillment.txt": (
            "Omnichannel Retail Fulfillment and Micro-Distribution Centers. "
            "Operating margins contracted by 4.2 percent across the third quarter due to acute supply chain bottlenecks, "
            "inventory holding surcharges, and escalating last-mile carrier fees. "
            "Urban retailers deploy micro-fulfillment hubs to accelerate delivery velocity."
        ),
        "ref_06_customer_churn_analytics.txt": (
            "Customer Lifetime Value and Retention Modeling in Banking. "
            "Predictive churn models utilize customer transaction velocity, service interactions, and fee friction to detect attrition signals. "
            "Proactive interventions through digital channels consistently improve retention cohorts."
        ),
        "ref_07_disruptive_innovation_theory.txt": (
            "Disruptive Innovation in Incumbent Market Environments. "
            "Christensen observed that incumbent firms prioritize sustaining innovations tailored to their most profitable customers. "
            "Disruptive technologies originate in simpler, lower-margin market segments before migrating upmarket."
        ),
        "ref_08_competitive_strategy_frameworks.txt": (
            "Porter's Competitive Advantage and Value Chain Configurations. "
            "Competitive advantage arises from the value an organization creates for its buyers. "
            "Cost leadership and differentiation represent distinct strategic choices that demand deliberate organizational focus."
        ),
        "ref_09_cloud_computing_architectures.txt": (
            "Cloud Infrastructure and High-Availability Microservices. "
            "Distributed enterprise software systems prioritize horizontal scalability, fault-tolerant message queues, and resilient database partitioning."
        ),
        "ref_10_corporate_finance_valuation.txt": (
            "Capital Budgeting and Corporate Valuation Methodologies. "
            "Discounted cash flow modeling projects free cash flows against the weighted average cost of capital to evaluate investment hurdle rates."
        ),
        "ref_11_workforce_safety_ergonomics.txt": (
            "Industrial Automation and Occupational Safety Standards. "
            "Integrating collaborative robotics into fulfillment centers demands comprehensive safety certifications and worker ergonomic training."
        ),
        "ref_12_ecommerce_logistics_automation.txt": (
            "Autonomous Mobile Robots in E-commerce Warehouses. "
            "Robotic sortation systems drastically compress pick-to-pack latency while enhancing inventory count precision."
        ),
    }

    for fname, content in articles.items():
        with open(os.path.join(ref_dir, fname), "w", encoding="utf-8") as f:
            f.write(content.strip())


def generate_all_submissions(sample_dir: str):
    """Generate all 15 sample submissions and the dataset manifest."""
    os.makedirs(sample_dir, exist_ok=True)

    print("Generating PDFs...")
    # 1. pdf_01: original logistics analysis
    create_pdf(
        os.path.join(sample_dir, "pdf_01_original_logistics.pdf"),
        TEXT_LOGISTICS_ORIGINAL,
        header_title="Logistics Fleet Optimization & Telemetry Study",
        footer_prefix="Page",
    )

    # 2. pdf_02: exact copy of 01 with different header/footer
    create_pdf(
        os.path.join(sample_dir, "pdf_02_exact_copy_of_01.pdf"),
        TEXT_LOGISTICS_ORIGINAL,
        header_title="North American Freight Fleet Modernization",
        footer_prefix="Sheet",
    )

    # 3. pdf_03: original retail analysis (source for cross-modal test)
    create_pdf(
        os.path.join(sample_dir, "pdf_03_original_retail.pdf"),
        TEXT_RETAIL_ORIGINAL,
        header_title="Omnichannel Retail Transformation & Urban Fulfillment",
        footer_prefix="Page",
    )

    # 4. pdf_04: LLM-generated essay
    create_pdf(
        os.path.join(sample_dir, "pdf_04_llm_generated.pdf"),
        TEXT_LLM_ESSAY,
        header_title="Strategic Synthesis on Enterprise Agility",
        footer_prefix="Page",
    )

    # 5. pdf_05: template prompt + Answer A
    create_pdf(
        os.path.join(sample_dir, "pdf_05_template_answer_a.pdf"),
        [TEMPLATE_PROMPT, TEXT_ANSWERS["A"]],
        header_title="Case Study Response - Candidate 101",
        footer_prefix="Page",
    )

    # 6. pdf_06: empty zero-byte file
    open(os.path.join(sample_dir, "pdf_06_empty.pdf"), "w").close()

    # 7. pdf_07: template prompt + Answer E
    create_pdf(
        os.path.join(sample_dir, "pdf_07_template_answer_e.pdf"),
        [TEMPLATE_PROMPT, TEXT_ANSWERS["E"]],
        header_title="Case Study Response - Candidate 105",
        footer_prefix="Page",
    )

    print("Generating DOCX files...")
    # 8. docx_01: original renewable energy
    create_docx(
        os.path.join(sample_dir, "docx_01_original_energy.docx"),
        title="Decentralized Renewable Microgrids & Storage Economics",
        paragraphs=TEXT_ENERGY_ORIGINAL,
    )

    # 9. docx_02: paraphrase of docx_01
    create_docx(
        os.path.join(sample_dir, "docx_02_paraphrase_of_01.docx"),
        title="Evaluation of Distributed Clean Energy Architecture",
        paragraphs=TEXT_ENERGY_PARAPHRASE,
    )

    # 10. docx_03: template prompt + Answer B
    create_docx(
        os.path.join(sample_dir, "docx_03_template_answer_b.docx"),
        title="Strategic Analysis Report - Submission B",
        paragraphs=[TEMPLATE_PROMPT, TEXT_ANSWERS["B"]],
    )

    # 11. docx_04: template prompt + Answer C
    create_docx(
        os.path.join(sample_dir, "docx_04_template_answer_c.docx"),
        title="Strategic Analysis Report - Submission C",
        paragraphs=[TEMPLATE_PROMPT, TEXT_ANSWERS["C"]],
    )

    # 12. docx_05: template prompt + Answer D
    create_docx(
        os.path.join(sample_dir, "docx_05_template_answer_d.docx"),
        title="Strategic Analysis Report - Submission D",
        paragraphs=[TEMPLATE_PROMPT, TEXT_ANSWERS["D"]],
    )

    print("Generating MP4 Video submissions...")
    # 13. mp4_01: reads paragraph 2 of pdf_03 (Cross-modal duplication)
    speech_text_01 = (
        "Operating margins contracted by 4.2 percent across the third quarter due to acute supply chain bottlenecks, "
        "inventory holding surcharges, and escalating last-mile carrier fees. To reverse this margin compression, "
        "regional distribution centers must transition toward automated micro-fulfillment centers capable of dispatching same-day orders within a five-mile radius."
    )
    wav_01 = os.path.join(sample_dir, "temp_speech_01.wav")
    synthesize_speech_wav(speech_text_01, wav_01)
    create_video_from_audio(wav_01, os.path.join(sample_dir, "mp4_01_reads_pdf_03.mp4"), title="Retail Strategy Presentation")
    if os.path.exists(wav_01):
        os.remove(wav_01)

    # 14. mp4_02: original pitch on retail banking
    speech_text_02 = (
        "Good morning evaluation panel. Today I am presenting our innovative customer retention architecture for retail banking. "
        "By tracking transaction velocity and digital touchpoint friction, our automated churn predictor flags at-risk accounts "
        "with ninety-two percent precision before capital flight occurs."
    )
    wav_02 = os.path.join(sample_dir, "temp_speech_02.wav")
    synthesize_speech_wav(speech_text_02, wav_02)
    create_video_from_audio(wav_02, os.path.join(sample_dir, "mp4_02_original_pitch.mp4"), title="Banking Retention Pitch")
    if os.path.exists(wav_02):
        os.remove(wav_02)

    # 15. mp4_03: silent video with no audio track
    create_silent_video(os.path.join(sample_dir, "mp4_03_silent.mp4"))

    # Manifest file
    manifest_data = {
        "dataset_name": "Multimodal Integrity Engine Benchmark Dataset",
        "total_files": 15,
        "files": [
            {"filename": "pdf_01_original_logistics.pdf", "type": "pdf", "label": "original", "source_topic": "logistics"},
            {"filename": "pdf_02_exact_copy_of_01.pdf", "type": "pdf", "label": "exact_copy", "copied_from": "pdf_01_original_logistics.pdf"},
            {"filename": "pdf_03_original_retail.pdf", "type": "pdf", "label": "original", "source_topic": "retail"},
            {"filename": "pdf_04_llm_generated.pdf", "type": "pdf", "label": "llm_generated", "generator": "ChatGPT/Claude synthetic"},
            {"filename": "pdf_05_template_answer_a.pdf", "type": "pdf", "label": "template_user", "template_section": "150-word prompt"},
            {"filename": "pdf_06_empty.pdf", "type": "pdf", "label": "corrupted", "condition": "zero-byte file"},
            {"filename": "pdf_07_template_answer_e.pdf", "type": "pdf", "label": "template_user", "template_section": "150-word prompt"},
            {"filename": "docx_01_original_energy.docx", "type": "docx", "label": "original", "source_topic": "renewable_energy"},
            {"filename": "docx_02_paraphrase_of_01.docx", "type": "docx", "label": "paraphrase", "copied_from": "docx_01_original_energy.docx"},
            {"filename": "docx_03_template_answer_b.docx", "type": "docx", "label": "template_user", "template_section": "150-word prompt"},
            {"filename": "docx_04_template_answer_c.docx", "type": "docx", "label": "template_user", "template_section": "150-word prompt"},
            {"filename": "docx_05_template_answer_d.docx", "type": "docx", "label": "template_user", "template_section": "150-word prompt"},
            {"filename": "mp4_01_reads_pdf_03.mp4", "type": "mp4", "label": "cross_modal_copy", "copied_from": "pdf_03_original_retail.pdf"},
            {"filename": "mp4_02_original_pitch.mp4", "type": "mp4", "label": "original", "source_topic": "banking_pitch"},
            {"filename": "mp4_03_silent.mp4", "type": "mp4", "label": "silent", "condition": "no audio stream"},
        ],
        "ground_truth_relationships": [
            {"doc_a": "pdf_01_original_logistics.pdf", "doc_b": "pdf_02_exact_copy_of_01.pdf", "relation": "exact_copy", "expected_peer_score": ">=90%"},
            {"doc_a": "docx_01_original_energy.docx", "doc_b": "docx_02_paraphrase_of_01.docx", "relation": "paraphrase", "expected_peer_score": ">=80%", "expected_lexical": "<70%"},
            {"doc_a": "mp4_01_reads_pdf_03.mp4", "doc_b": "pdf_03_original_retail.pdf", "relation": "cross_modal_copy", "expected_score": ">=80%"},
            {"doc_a": "pdf_05_template_answer_a.pdf", "doc_b": "docx_03_template_answer_b.docx", "relation": "benign_template", "discounted": True},
            {"doc_a": "docx_04_template_answer_c.docx", "doc_b": "docx_05_template_answer_d.docx", "relation": "benign_template", "discounted": True},
            {"doc_a": "pdf_07_template_answer_e.pdf", "doc_b": "pdf_05_template_answer_a.pdf", "relation": "benign_template", "discounted": True},
        ],
    }

    manifest_path = os.path.join(sample_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Dataset successfully created in '{sample_dir}' with manifest '{manifest_path}'.")


def main():
    sample_dir = "sample_submissions"
    ref_dir = "reference_corpus"
    print("Generating simulated external knowledge base...")
    build_reference_corpus(ref_dir)
    print("Generating synthetic submission dataset...")
    generate_all_submissions(sample_dir)
    print("Done!")


if __name__ == "__main__":
    main()
