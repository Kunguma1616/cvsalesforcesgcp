from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pdfminer.high_level import extract_text
import re
import os
import uuid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from azure.storage.blob import BlobServiceClient
from simple_salesforce import Salesforce
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    HRFlowable, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from io import BytesIO
from typing import List
from pydantic import BaseModel, Field

routes = APIRouter(prefix="/cv", tags=["CV Upload"])

# ── Env ──────────────────────────────────────────────────────────────────────
api_key  = os.getenv("GROQ_API_KEY")
MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── Schemas ───────────────────────────────────────────────────────────────────
class CriterionScore(BaseModel):
    criterion_name: str  = Field(description="Name of the evaluation criterion")
    score:          float = Field(description="Score out of 5", ge=1.0, le=5.0)
    explanation:    str   = Field(description="Detailed explanation for the score")


class ResumeAnalysisOutput(BaseModel):
    job_category:             str              = Field(description="Classified job category")
    job_subcategory:          str              = Field(description="More specific role classification")
    criteria_scores:          List[CriterionScore] = Field(description="List of 5-7 key criteria")
    pros:                     List[str]        = Field(description="Strengths of the resume")
    cons:                     List[str]        = Field(description="Weaknesses or missing areas")
    overall_assessment:       str              = Field(description="Overall match summary")
    key_requirements_met:     List[str]        = Field(description="Requirements satisfied")
    key_requirements_missing: List[str]        = Field(description="Requirements missing")
    skills_identified:        List[str]        = Field(description="All skills extracted from resume")
    skills_relevant:          List[str]        = Field(description="Skills related to job description")
    skills_missing:           List[str]        = Field(description="Required job skills missing in resume")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _calculate_ats_score(resume_text: str, job_desc: str) -> float:
    try:
        jd_kw = set(re.findall(r'\b\w+\b', job_desc.lower()))
        if not jd_kw:
            return 0.0
        rv_kw = set(re.findall(r'\b\w+\b', resume_text.lower()))
        return len(jd_kw & rv_kw) / len(jd_kw)
    except Exception:
        return 0.0


def _get_structured_report(resume: str, job_desc: str) -> ResumeAnalysisOutput:
    system_prompt = """
You are an elite HR analyst and resume evaluator with expertise in recruitment,
competency-based evaluation, and job-market benchmarking.

Your responsibilities:
1. Classify the job category and subcategory
2. Score 5-7 evaluation criteria (1-5 scale) with detailed explanations
3. List 5-8 pros and cons
4. List requirements met and missing
5. Extract all skills, relevant skills, and missing skills
6. Write a comprehensive overall assessment (8-12 sentences)

Return structured JSON ONLY.
"""
    llm = ChatGroq(api_key=api_key, model=MODEL_ID, temperature=0.2)
    structured_llm = llm.with_structured_output(
        ResumeAnalysisOutput, method="function_calling", include_raw=False
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"JOB DESCRIPTION:\n{job_desc}\n\nRESUME:\n{resume}\n\nReturn valid JSON ONLY.")
    ]
    return structured_llm.invoke(messages)


def _clean_text(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'^[\s]*[-•*]\s+', '', text, flags=re.MULTILINE)
    return re.sub(r'\s+', ' ', text).strip()


def _score_chart(ats_score: float, ai_score: float) -> BytesIO:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        ['ATS Keyword\nMatch', 'AI Evaluation\nScore'],
        [ats_score * 100, ai_score * 100],
        color=['#27549D', '#7099DB'],
        width=0.5
    )
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f'{h:.1f}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Match Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('Resume Match Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


def _build_pdf(candidate_name, job_category, job_subcategory,
               ats_score, ai_score, analysis, chart_buffer) -> bytes:
    buff   = BytesIO()
    doc    = SimpleDocTemplate(buff, pagesize=A4,
                               leftMargin=50, rightMargin=50,
                               topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"], fontSize=26, alignment=TA_CENTER,
        textColor=colors.HexColor("#27549D"), spaceAfter=20, fontName='Helvetica-Bold')
    sec_hdr = ParagraphStyle(
        "SectionHeader", parent=styles["Heading1"], fontSize=16,
        textColor=colors.HexColor("#27549D"), spaceAfter=12, spaceBefore=20,
        fontName='Helvetica-Bold')
    sub_hdr = ParagraphStyle(
        "SubsectionHeader", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#0f1e33"), spaceAfter=8, spaceBefore=12,
        fontName='Helvetica-Bold')
    body = ParagraphStyle(
        "BodyText", parent=styles["BodyText"], fontSize=10, leading=16,
        alignment=TA_JUSTIFY, spaceAfter=10, textColor=colors.HexColor("#1a1a1a"),
        fontName='Helvetica')
    bullet = ParagraphStyle(
        "BulletStyle", parent=styles["BodyText"], fontSize=10, leading=16,
        leftIndent=20, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"),
        fontName='Helvetica')

    story = []

    # ── Header ──
    story.append(Paragraph("Resume Analysis Report", title_style))
    story.append(Spacer(1, 10))

    info_table = Table(
        [['Candidate Name:', candidate_name],
         ['Job Category:',   job_category],
         ['Specific Role:',  job_subcategory]],
        colWidths=[2*inch, 4*inch]
    )
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ('TEXTCOLOR',  (0, 0), (-1, -1), colors.HexColor("#1a1a1a")),
        ('ALIGN',      (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME',   (0, 0), (0, -1),  'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db"))
    ]))
    story += [info_table, Spacer(1, 20),
              HRFlowable(width="100%", thickness=2, color=colors.HexColor("#27549D")),
              Spacer(1, 20)]

    # ── Scores ──
    story.append(Paragraph("Score Summary", sec_hdr))
    score_table = Table(
        [['Metric', 'Score', 'Percentage'],
         ['ATS Keyword Match',   f'{ats_score:.2f}', f'{ats_score*100:.1f}%'],
         ['AI Evaluation Score', f'{ai_score:.2f}',  f'{ai_score*100:.1f}%']],
        colWidths=[2.5*inch, 1.5*inch, 1.5*inch]
    )
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0),  colors.HexColor("#27549D")),
        ('TEXTCOLOR',  (0, 0), (-1, 0),  colors.whitesmoke),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME',   (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, 0),  11),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f9fafb")),
        ('GRID',       (0, 0), (-1, -1), 1, colors.HexColor("#d1d5db")),
    ]))
    story += [score_table, Spacer(1, 20),
              Image(chart_buffer, width=5.5*inch, height=2.8*inch),
              Spacer(1, 20), PageBreak()]

    # ── Criteria ──
    story.append(Paragraph("Evaluation Criteria Analysis", sec_hdr))
    story.append(Spacer(1, 10))
    for c in analysis.criteria_scores[:7]:
        story += [
            Paragraph(c.criterion_name, sub_hdr),
            Paragraph(f"<b>Score:</b> {c.score}/5.0", body),
            Paragraph(_clean_text(c.explanation), body),
            Spacer(1, 10)
        ]

    # ── Pros / Cons ──
    story += [Spacer(1, 10), Paragraph("Strengths (Pros)", sec_hdr)]
    for i, p in enumerate(analysis.pros[:8], 1):
        story.append(Paragraph(f"{i}. {_clean_text(p)}", bullet))
    story += [Spacer(1, 15), Paragraph("Areas for Improvement (Cons)", sec_hdr)]
    for i, c in enumerate(analysis.cons[:8], 1):
        story.append(Paragraph(f"{i}. {_clean_text(c)}", bullet))

    story.append(PageBreak())

    # ── Requirements ──
    story += [Paragraph("Requirements Analysis", sec_hdr), Spacer(1, 10),
              Paragraph("Requirements Met", sub_hdr)]
    for i, r in enumerate(analysis.key_requirements_met[:8], 1):
        story.append(Paragraph(f"{i}. {_clean_text(r)}", bullet))
    story += [Spacer(1, 15), Paragraph("Requirements Missing", sub_hdr)]
    for i, r in enumerate(analysis.key_requirements_missing[:8], 1):
        story.append(Paragraph(f"{i}. {_clean_text(r)}", bullet))

    # ── Skills ──
    story += [Spacer(1, 20), Paragraph("Skills Analysis", sec_hdr), Spacer(1, 10)]
    skills_table = Table(
        [['Metric', 'Count'],
         ['Total Skills Identified',    str(len(analysis.skills_identified))],
         ['Relevant Skills Matched',    str(len(analysis.skills_relevant))],
         ['Required Skills Missing',    str(len(analysis.skills_missing))]],
        colWidths=[3*inch, 2*inch]
    )
    skills_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0),  colors.HexColor("#27549D")),
        ('TEXTCOLOR',  (0, 0), (-1, 0),  colors.whitesmoke),
        ('ALIGN',      (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME',   (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f9fafb")),
        ('GRID',       (0, 0), (-1, -1), 1, colors.HexColor("#d1d5db"))
    ]))
    story += [skills_table, Spacer(1, 20),
              Paragraph("Overall Summary", sec_hdr),
              Paragraph(_clean_text(analysis.overall_assessment), body)]

    # ── Footer ──
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#6b7280"), alignment=TA_CENTER)
    story += [Spacer(1, 30),
              HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d1d5db")),
              Spacer(1, 10),
              Paragraph("Generated by Aspect AI Resume Analyzer | Confidential Document",
                        footer_style)]

    doc.build(story)
    buff.seek(0)
    return buff.read()


def _upload_to_azure(pdf_bytes: bytes, filename: str):
    connect_str    = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_CONTAINER_NAME")
    sas_token      = os.getenv("AZURE_SAS_TOKEN")

    if not all([connect_str, container_name, sas_token]):
        return None, "Azure credentials missing from environment"
    if "sv=" not in sas_token:
        return None, "Invalid SAS token format (must contain 'sv=')"

    blob_service  = BlobServiceClient.from_connection_string(connect_str)
    container     = blob_service.get_container_client(container_name)
    blob_client   = container.get_blob_client(filename)
    blob_client.upload_blob(BytesIO(pdf_bytes), overwrite=True)
    return f"{blob_client.url}?{sas_token}", None


def _create_sf_record(first_name, last_name, email, azure_url, trade, ats_score=None, ai_score=None):
    try:
        sf = Salesforce(
            username=os.getenv("SF_USERNAME"),
            password=os.getenv("SF_PASSWORD"),
            security_token=os.getenv("SF_SECURITY_TOKEN"),
            domain=os.getenv("SF_DOMAIN", "login")
        )
        payload = {
            "First_Name__c":     first_name,
            "Last_Name__c":      last_name,
            "Email_Address__c":  email,
            "Your_CV__c":        azure_url,
            "Primary_Trade__c":  trade,
        }
        if ats_score is not None:
            payload["ATS_Score__c"] = ats_score
        if ai_score is not None:
            payload["AI_Score__c"] = ai_score

        # Try with scores first; fall back without if fields don't exist in org
        try:
            result = sf.Engineer_Application__c.create(payload)
            return {"success": True, "result": result}
        except Exception:
            payload.pop("ATS_Score__c", None)
            payload.pop("AI_Score__c", None)
            result = sf.Engineer_Application__c.create(payload)
            return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Route ─────────────────────────────────────────────────────────────────────
@routes.post("/upload-and-analyze")
async def upload_and_analyze(
    name:            str        = Form(...),
    email:           str        = Form(...),
    trade:           str        = Form(...),
    job_description: str        = Form(...),
    resume:          UploadFile = File(...)
):
    """
    Upload a CV (PDF), run AI analysis, generate a report PDF,
    store it in Azure Blob Storage, and create a Salesforce record.
    """
    if not api_key:
        raise HTTPException(500, "GROQ_API_KEY not configured on server")
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # 1 ── Extract text
    try:
        contents    = await resume.read()
        resume_text = extract_text(BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Failed to read PDF: {e}")

    if not resume_text or len(resume_text.strip()) < 50:
        raise HTTPException(400, "Could not extract text. Ensure the PDF is not scanned/image-only.")

    # 2 ── AI analysis
    try:
        ats_score = _calculate_ats_score(resume_text, job_description)
        analysis  = _get_structured_report(resume_text, job_description)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    avg_criteria = (
        sum(c.score for c in analysis.criteria_scores) / len(analysis.criteria_scores)
        if analysis.criteria_scores else 0
    )
    ai_score = avg_criteria / 5.0

    # 3 ── Generate PDF report
    try:
        chart_buf = _score_chart(ats_score, ai_score)
        pdf_bytes = _build_pdf(
            name, analysis.job_category, analysis.job_subcategory,
            ats_score, ai_score, analysis, chart_buf
        )
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")

    # 4 ── Upload to Azure
    azure_url   = None
    azure_error = None
    safe_name   = name.replace(' ', '_')
    filename    = f"{safe_name}_{uuid.uuid4().hex[:6]}.pdf"
    try:
        azure_url, azure_error = _upload_to_azure(pdf_bytes, filename)
    except Exception as e:
        azure_error = str(e)

    # 5 ── Salesforce record
    sf_result = None
    if azure_url:
        parts      = name.split()
        first_name = parts[0]
        last_name  = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
        try:
            sf_result = _create_sf_record(
            first_name, last_name, email, azure_url, trade,
            ats_score=round(ats_score * 100, 1),
            ai_score=round(ai_score * 100, 1),
        )
        except Exception:
            pass

    return {
        "candidate_name":  name,
        "candidate_email": email,
        "trade":           trade,
        "ats_score":       round(ats_score * 100, 1),
        "ai_score":        round(ai_score  * 100, 1),
        "analysis":        analysis.dict(),
        "azure_url":       azure_url,
        "azure_error":     azure_error,
        "salesforce_record": sf_result,
        "pdf_generated":   True
    }