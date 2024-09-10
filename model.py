import os
import re
import unicodedata
import nltk
import inflect
import spacy
from PyPDF2 import PdfReader 
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Define paths to save the models locally
RANKER_MODEL_DIR = "./ranker_model"
SUMMARIZER_MODEL_DIR = "./summarizer_model"

# Load spaCy model for preprocessing
nlp = spacy.load("en_core_web_sm")

# NLTK setup
def load_nltk():
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')
    print('nltk packages are ready')

# Preprocessing functions
def remove_non_ascii(words):
    new_words = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
    return new_words

def to_lowercase(words):
    return [word.lower() for word in words]

def remove_punctuation(words):
    return [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word) != '']

def replace_numbers(words):
    p = inflect.engine()
    return [p.number_to_words(word) if word.isdigit() else word for word in words]

def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

def stem_words(words):
    stemmer = LancasterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in words]

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    preprocessed = normalize(tokens)
    return ' '.join(preprocessed)

# Load BERT model for ranking
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

# Embed text using the ranking model
def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Calculate cosine similarity between job description and resume
def calculate_similarity(job_desc_embedding, resume_embedding):
    job_desc_embedding = job_desc_embedding.detach().numpy().reshape(1, -1)
    resume_embedding = resume_embedding.detach().numpy().reshape(1, -1)
    similarity_score = cosine_similarity(job_desc_embedding, resume_embedding).item()
    return similarity_score

# Categorize similarity score
def categorize_score(normalized_score):
    if normalized_score >= 0.8:
        return "High Match"
    elif normalized_score >= 0.5:
        return "Moderate Match"
    else:
        return "Low Match"

# Rank the resume based on job description
def rank_resume(job_description, cv_text, ranker_model, ranker_tokenizer):
    job_desc_cleaned = preprocess_text(job_description)
    cv_text_cleaned = preprocess_text(cv_text)
    job_desc_embedding = embed_text(job_desc_cleaned, ranker_model, ranker_tokenizer)
    resume_embedding = embed_text(cv_text_cleaned, ranker_model, ranker_tokenizer)
    similarity_score = calculate_similarity(job_desc_embedding, resume_embedding)
    normalized_score = (similarity_score + 1) / 2  # Normalize the score
    category = categorize_score(normalized_score)
    return normalized_score, category

# Load BART model for summarization
def load_summarizer_model():
    model_name = "facebook/bart-large-cnn"
    if not os.path.exists(SUMMARIZER_MODEL_DIR) or not os.listdir(SUMMARIZER_MODEL_DIR):
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer.save_pretrained(SUMMARIZER_MODEL_DIR)
        model.save_pretrained(SUMMARIZER_MODEL_DIR)
    else:
        tokenizer = BartTokenizer.from_pretrained(SUMMARIZER_MODEL_DIR)
        model = BartForConditionalGeneration.from_pretrained(SUMMARIZER_MODEL_DIR)
    return model, tokenizer

# Summarize resume text with BART
def summarize_resume_with_bart(resume_text, bart_model, bart_tokenizer):
    inputs = bart_tokenizer([resume_text], max_length=512, return_tensors='pt', padding=True, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Extract resume entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Generate summary based on entities
def generate_summary(resume_entities, job_description_entities):
    summary = "This candidate has the following key details:\n"
    
    summary += "Resume Entities:\n"
    for entity, label in resume_entities:
        summary += f"- {label}: {entity}\n"
    
    summary += "Job Description Entities:\n"
    for entity, label in job_description_entities:
        summary += f"- {label}: {entity}\n"
    
    return summary
