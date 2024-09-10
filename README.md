This project is a resume ranking and summarization system. It helps to evaluate resumes based on a given job description and provide a ranking score, category, and a concise summary of the resume. The system uses two primary functionalities: ranking resumes through semantic similarity and generating a summary of the resume.

### Functionalities:

1. **Resume Ranking:**
   - The project compares a job description and a resume using **cosine similarity** between their respective embeddings.
   - It uses **BERT ('bert-base-uncased')** for embedding both the job description and the resume text.
   - A normalized similarity score is calculated between the two embeddings using cosine similarity, and the resume is categorized into:
     - *Highly Recommended* (score ≥ 0.8)
     - *Average* (score ≥ 0.5)
     - *Not Recommended* (score < 0.5)

2. **Resume Summarization:**
   - The system generates a brief summary of the resume using the **BART model ('facebook/bart-large-cnn')**.
   - The summarization model condenses the content of the resume into a shorter, relevance-based summary that highlights key points, such as educational background, work experience, and skills.

3. **Preprocessing:**
   - Both the job description and resume go through text preprocessing steps (removal of non-ASCII characters, stop words, punctuation, etc.) using **NLTK** to ensure better input for embedding.

4. **Interaction:**
   - Users upload both the job description and resume in PDF format, which are processed and passed to the system.
   - The model outputs a similarity score, categorization, and a summary of the resume.

### Models:
- **Ranking Model:** `bert-base-uncased` (BERT) is used to create embeddings of the job description and resume text.
- **Summarization Model:** `facebook/bart-large-cnn` (BART) is employed for generating the resume summary.

The project is designed to streamline the recruitment process by quickly assessing how relevant a resume is to a specific job posting, thus assisting recruiters in shortlisting candidates efficiently.
