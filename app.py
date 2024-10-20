from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import os
from werkzeug.utils import secure_filename
import pymysql

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# MySQL database connection
def get_database_data():
    """Fetch data from MySQL database."""
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='similaritydb'
    )
    cursor = connection.cursor()

    # Execute the query to get data
    query = "SELECT * FROM similaritydataset"
    cursor.execute(query)
    rows = cursor.fetchall()

    # Fetch column names and create a DataFrame
    columns = [col[0] for col in cursor.description]
    data = pd.DataFrame(rows, columns=columns)

    # Close the connection
    connection.close()

    return data

# Fetch the dataset from MySQL
data = get_database_data()

# Preprocess the texts in the dataset for similarity calculations
preprocessed_texts = tfidf_vectorizer.transform(data['plagiarized_text'].fillna(""))

def detect(input_text):
    """Detect plagiarism in the input text."""
    if not input_text.strip():
        return "No text provided", []

    # Vectorize the input text
    vectorized_text = tfidf_vectorizer.transform([input_text])
    prediction = model.predict(vectorized_text)

    # Handle cases where the model predicts 'No Plagiarism'
    if prediction[0] == 0:
        return "No Plagiarism Detected", []

    # If plagiarism is detected, calculate similarity scores
    cosine_similarities = cosine_similarity(vectorized_text, preprocessed_texts)[0]
    plagiarism_sources = []

    # Adjusted threshold for more accurate results
    threshold = 0.35  
    for i, similarity in enumerate(cosine_similarities):
        if similarity > threshold:
            plagiarism_percentage = round(similarity * 100, 2)
            source_title = data['source_text'].iloc[i]
            plagiarism_sources.append((source_title, plagiarism_percentage))

    # Sort by similarity percentage in descending order
    plagiarism_sources.sort(key=lambda x: x[1], reverse=True)

    detection_result = "Plagiarism Detected" if plagiarism_sources else "No Plagiarism Detected"
    return detection_result, plagiarism_sources

def extract_text_from_file(file):
    """Extract text from uploaded PDF or TXT file."""
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
    """Render the home page."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    """Handle plagiarism detection requests."""
    input_text = request.form.get('text', "").strip()

    # Process uploaded files, if any
    files = request.files.getlist("files[]")
    for file in files:
        if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            file_text = extract_text_from_file(file)
            input_text += "\n" + file_text

    # Run the detection logic
    detection_result, plagiarism_sources = detect(input_text)

    return render_template(
        'index.html', 
        result=detection_result, 
        plagiarism_sources=plagiarism_sources
    )

if __name__ == "__main__":
    app.run(debug=True)
