from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import os
from werkzeug.utils import secure_filename
import pymysql
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# MySQL database connection
def get_database_data():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='similaritydb'
    )
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM similaritydataset")
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    data = pd.DataFrame(rows, columns=columns)
    connection.close()
    return data

# Fetch and preprocess the dataset
data = get_database_data()
preprocessed_texts = tfidf_vectorizer.transform(data['plagiarized_text'].fillna(""))

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def get_snippets(source_text, input_text, ngram_size=5):
    source_text_clean = preprocess_text(source_text)
    input_text_clean = preprocess_text(input_text)

    source_words = source_text_clean.split()
    input_words = input_text_clean.split()

    def generate_ngrams(words, n):
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    input_ngrams = set(generate_ngrams(input_words, ngram_size))
    source_ngrams = generate_ngrams(source_words, ngram_size)

    matching_snippets = [ngram for ngram in source_ngrams if ngram in input_ngrams]
    combined_snippets = []
    current_snippet = []

    for i, ngram in enumerate(matching_snippets):
        if not current_snippet:
            current_snippet.append(ngram)
        else:
            prev_ngram = current_snippet[-1].split()
            current_words = ngram.split()
            if prev_ngram[-(ngram_size - 1):] == current_words[:ngram_size - 1]:
                current_snippet.append(current_words[-1])
            else:
                combined_snippets.append(' '.join(current_snippet))
                current_snippet = [ngram]

    if current_snippet:
        combined_snippets.append(' '.join(current_snippet))

    return sorted(set(combined_snippets), key=lambda snippet: source_text_clean.find(snippet))

def detect(input_text):
    if not input_text.strip():
        return "No text provided", []

    input_text = preprocess_text(input_text)
    vectorized_text = tfidf_vectorizer.transform([input_text])
    prediction = model.predict(vectorized_text)

    if prediction[0] == 0:
        return "No Plagiarism Detected", []

    cosine_similarities = cosine_similarity(vectorized_text, preprocessed_texts)[0]
    plagiarism_sources = []

    threshold = 0.35
    for i, similarity in enumerate(cosine_similarities):
        if similarity > threshold:
            plagiarism_percentage = round(similarity * 100, 2)
            source_title = data['source_text'].iloc[i]
            source_text = data['plagiarized_text'].iloc[i]
            matching_snippets = get_snippets(source_text, input_text)
            plagiarism_sources.append((source_title, plagiarism_percentage, matching_snippets))

    plagiarism_sources.sort(key=lambda x: x[1], reverse=True)
    detection_result = "Plagiarism Detected" if plagiarism_sources else "No Plagiarism Detected"
    return detection_result, plagiarism_sources

def extract_text_from_file(file):
    text = ""
    if file.filename.endswith('.pdf'):
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form.get('text', "").strip()
    files = request.files.getlist("files[]")

    for file in files:
        if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            input_text += "\n" + extract_text_from_file(file)

    detection_result, plagiarism_sources = detect(input_text)
    word_count = len(input_text.split())

    # Calculate plagiarism and unique percentages
    total_percentage = sum(source[1] for source in plagiarism_sources)
    plagiarized_percentage = min(total_percentage, 100)  # Capping at 100%
    unique_percentage = 100 - plagiarized_percentage

    return render_template(
        'index.html',
        result=detection_result,
        plagiarism_sources=plagiarism_sources,
        word_count=word_count,
        results_found=len(plagiarism_sources),
        plagiarized_percentage=plagiarized_percentage,
        unique_percentage=unique_percentage
    )

if __name__ == "__main__":
    app.run(debug=True)
