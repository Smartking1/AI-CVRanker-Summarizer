# ranking.py

import os
from preprocess import preprocess_text
#from summarization import summarize_with_t5
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Define paths to save the models locally
RANKER_MODEL_DIR = "./ranker_model"

def load_ranking_model():
    model_name = "bert-base-uncased"
    if not os.path.exists(RANKER_MODEL_DIR) or not os.listdir(RANKER_MODEL_DIR):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokenizer.save_pretrained(RANKER_MODEL_DIR)
        model.save_pretrained(RANKER_MODEL_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(RANKER_MODEL_DIR)
        model = AutoModel.from_pretrained(RANKER_MODEL_DIR)
    return model, tokenizer

def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def calculate_similarity(job_desc_embedding, resume_embedding):
    job_desc_embedding = job_desc_embedding.detach().numpy().reshape(1, -1)
    resume_embedding = resume_embedding.detach().numpy().reshape(1, -1)
    similarity_score = cosine_similarity(job_desc_embedding, resume_embedding).item()
    return similarity_score

def categorize_score(normalized_score):
    if normalized_score >= 0.8:
        return "Highly Recommended"
    elif normalized_score >= 0.5:
        return "Average"
    else:
        return "Not Recommended"

def rank_resume(job_description, cv_text):
    ranker_model, ranker_tokenizer = load_ranking_model()
    job_desc_cleaned = preprocess_text(job_description)
    cv_text_cleaned = preprocess_text(cv_text)
    job_desc_embedding = embed_text(job_desc_cleaned, ranker_model, ranker_tokenizer)
    resume_embedding = embed_text(cv_text_cleaned, ranker_model, ranker_tokenizer)
    similarity_score = calculate_similarity(job_desc_embedding, resume_embedding)
    normalized_score = (similarity_score + 1) / 2  # Normalize the score
    category = categorize_score(normalized_score)

    # Generate summary using T5
    #summary = summarize_with_t5(cv_text_cleaned)

    return normalized_score, category
