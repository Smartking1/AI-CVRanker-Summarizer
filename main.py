import streamlit as st
from preprocess import preprocess_text
from summarization import summarize_resume_with_job_desc
from ranking import rank_resume
from PyPDF2 import PdfReader

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("Resume Ranking and Summarization")

    st.sidebar.header("Upload Files")
    resume_file = st.sidebar.file_uploader("Upload Resume PDF", type="pdf")
    job_description_file = st.sidebar.file_uploader("Upload Job Description PDF", type="pdf")

    if resume_file and job_description_file:
        # Extract text from PDFs
        resume_text = extract_text_from_pdf(resume_file)
        job_description_text = extract_text_from_pdf(job_description_file)
        
        # Preprocess text
        resume_text_preprocessed = preprocess_text(resume_text)
        job_description_text_preprocessed = preprocess_text(job_description_text)
        
        # Rank resume
        score, category = rank_resume(job_description_text_preprocessed, resume_text_preprocessed)

        # Display ranking
        st.subheader("Resume Ranking")
        st.write(f"Ranking score: {score:.2f}")
        st.write(f"Category: {category}")

        # Summarize resume
        summary = summarize_resume_with_job_desc(resume_text, job_description_text)
        st.subheader("Relevance-Aware Resume Summary")
        st.write(summary)

if __name__ == "__main__":
    main()
