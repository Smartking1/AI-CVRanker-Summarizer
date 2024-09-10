import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer, util

# Load pre-trained models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('bert-base-nli-mean-tokens')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Function to calculate similarity between resume text and job description text
def calculate_similarity(resume_text, job_description_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_description_embedding = model.encode(job_description_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_description_embedding).item()
    return similarity_score

# Function to summarize resume based on relevance to job description
def summarize_resume_with_job_desc(resume_text, job_description_text):
    # Rank resume text relevance to the job description
    relevance_score = calculate_similarity(resume_text, job_description_text)

    # Use BART to summarize the resume
    inputs = bart_tokenizer([resume_text], max_length=512, return_tensors='pt', padding=True, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Generate a concise summary
    summary_text = f"Summary:\n{summary}"
    
    return summary_text
