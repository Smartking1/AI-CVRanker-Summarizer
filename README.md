# CVRanker

### Project Overview: Resume Ranking and Summarization System

This project is designed to automatically rank resumes based on their relevance to a given job description using **cosine similarity** of textual embeddings. The primary goal is to streamline the process of matching job descriptions to candidate resumes and providing an efficient way to assess and categorize resumes into levels of relevance (e.g., "Highly Recommended," "Average," or "Not Recommended").

#### Key Functionalities:

1. **Resume and Job Description Input**:
   - The system accepts both a resume and a job description as inputs. 
   - These inputs can either be in **text** or **PDF** format, from which the text content is extracted and preprocessed.

2. **Text Preprocessing**:
   - Both the resume and job description undergo text preprocessing to improve the quality of the input data before embedding.
   - Preprocessing includes:
     - Lowercasing
     - Removing stopwords, punctuation, and special characters
     - Lemmatization (to normalize words)
     - Tokenization
   - This ensures that the model focuses on important, clean content rather than irrelevant noise.

3. **Text Embedding**:
   - The preprocessed resume and job description texts are embedded into a numerical vector representation using a **pre-trained language model**.
   - The current implementation uses **Sentence-BERT** (e.g., `paraphrase-MiniLM-L6-v2`) to generate these embeddings. This model is specifically designed for tasks requiring semantic textual similarity.

4. **Cosine Similarity for Ranking**:
   - After embedding, the system computes the **cosine similarity** between the job description and the resume embeddings.
   - Cosine similarity provides a score between `-1` and `1` where `1` means the texts are identical, and `-1` means they are completely dissimilar. This score is normalized to fall between `0` and `1` to make it easier to interpret.

5. **Relevance Categorization**:
   - Based on the normalized similarity score, the resume is categorized into three groups:
     - **Highly Recommended**: If the score is above `0.8`.
     - **Average**: If the score is between `0.5` and `0.8`.
     - **Not Recommended**: If the score is below `0.5`.
   - This allows for quick decision-making when filtering candidates based on job fit.

6. **Summary Generation (Optional)**:
   - The system can optionally generate a concise summary of the resume using a language model like **BART**. This feature helps in quickly understanding key points from lengthy resumes.
   - The summary is created by focusing on relevant details extracted from the resume, such as education, work experience, and key skills.

7. **Database Integration**:
   - The system can integrate with a **database** to store:
     - Uploaded resumes and job descriptions.
     - Similarity scores and categories.
     - Summaries of resumes for future use.
   - A database structure proposal involves:
     - **Tables for Resumes**: Storing resume text, embeddings, and metadata (e.g., candidate name, contact details).
     - **Tables for Job Descriptions**: Storing job descriptions, embeddings, and position metadata.
     - **Tables for Ranking Results**: Storing similarity scores and rankings for each resume-job description pair.
   - Indexing would be recommended for faster querying based on similarity scores or categories.

8. **Front-end Interface**:
   - The project includes a front-end interface (e.g., using **Streamlit**) where users can:
     - Upload resumes and job descriptions.
     - View the ranked results, which display the similarity score, category, and optional summary.
     - Download or export the results for further review.

9. **Deployment**:
   - The back-end of the system is built using **FastAPI**, allowing the ranking system to be deployed as an API for integration with other tools.
   - The system can also be deployed with Streamlit as a user-friendly interface for HR teams or recruiters.

#### Example Workflow:
1. **Step 1**: A recruiter uploads a job description and multiple resumes in text or PDF format.
2. **Step 2**: The system preprocesses the inputs to clean and standardize them.
3. **Step 3**: The preprocessed texts are embedded into numerical vectors using Sentence-BERT.
4. **Step 4**: Cosine similarity is calculated between the job description and each resume.
5. **Step 5**: Each resume is categorized based on its relevance to the job description and displayed in a ranked list.
6. **Step 6**: (Optional) Summaries of each resume are generated to provide a quick overview.

#### Advantages:
- **Efficiency**: Automates the tedious process of matching resumes to job descriptions.
- **Accuracy**: Uses advanced models like Sentence-BERT to capture the semantic meaning of the text for better matching.
- **Customizability**: The categorization thresholds and embedding models can be fine-tuned based on user needs.
- **Scalability**: Designed to handle multiple resumes and job descriptions efficiently.

By implementing this system, recruiters can dramatically reduce the time spent on initial resume screening and ensure that they focus on the most relevant candidates for the role.
