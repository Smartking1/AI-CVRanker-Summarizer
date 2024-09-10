from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from preprocess import preprocess_text
from summarization import summarize_resume_with_job_desc
from ranking import rank_resume
from PyPDF2 import PdfReader

app = FastAPI()

class JobApplication(BaseModel):
    job_description: str
    resume: str

def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.post("/rank_resume")
async def rank_resume_endpoint(job_description_file: UploadFile = File(...), resume_file: UploadFile = File(...)):
    # Extract text from PDF files
    job_description_text = extract_text_from_pdf(job_description_file.file)
    resume_text = extract_text_from_pdf(resume_file.file)

    # Preprocess text
    job_description_text_preprocessed = preprocess_text(job_description_text)
    resume_text_preprocessed = preprocess_text(resume_text)

    # Rank resume
    score, category = rank_resume(job_description_text_preprocessed, resume_text_preprocessed)
    
    # Summarize resume
    summary = summarize_resume_with_job_desc(resume_text_preprocessed, job_description_text_preprocessed)

    return {
        "ranking_score": score,
        "category": category,
        "summary": summary
    }
