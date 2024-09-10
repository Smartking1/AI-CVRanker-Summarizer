# preprocess.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import T5Tokenizer

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Normalization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Preprocessing steps
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back to a single string
    return ' '.join(tokens)
