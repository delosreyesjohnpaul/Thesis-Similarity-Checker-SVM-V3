from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import mysql.connector
import difflib

app = Flask(__name__)

# Set file upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Connect to MySQL database
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="plagiarism_db"
    )

# Function to preprocess text (remove punctuation, lowercase, etc.)
def preprocess_text(text):
    return ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text).strip()

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to read the file content
def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_extension.lower() == '.pdf':
        import PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() for page in pdf_reader.pages)
            return text

# Function to detect plagiarized snippets
def get_snippets(source_text, input_text, ngram_size=5):
    """Get matching phrases from source text that are present in the input text."""
    # Preprocess both source and input text
    source_text_clean = preprocess_text(source_text)
    input_text_clean = preprocess_text(input_text)

    # Tokenize source and input text into words
    source_words = source_text_clean.split()
    input_words = input_text_clean.split()

    # Function to generate n-grams from words list
    def generate_ngrams(words, n):
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    # Generate n-grams for both source and input text
    input_ngrams = set(generate_ngrams(input_words, ngram_size))
    source_ngrams = generate_ngrams(source_words, ngram_size)

    # Find matching n-grams between source and input
    matching_snippets = [ngram for ngram in source_ngrams if ngram in input_ngrams]

    # Combine consecutive n-grams into larger, coherent snippets
    combined_snippets = []
    current_snippet = []
    for i, ngram in enumerate(matching_snippets):
        # If this is the first n-gram, add it to current snippet
        if not current_snippet:
            current_snippet.append(ngram)
        else:
            # Check if this n-gram follows the previous one sequentially
            prev_ngram = current_snippet[-1]
            prev_words = prev_ngram.split()
            current_words = ngram.split()

            if prev_words[-(ngram_size - 1):] == current_words[:ngram_size - 1]:
                # This n-gram continues the previous one
                current_snippet.append(current_words[-1])
            else:
                # Save the current snippet and start a new one
                combined_snippets.append(' '.join(current_snippet))
                current_snippet = [ngram]

    # Append the last snippet if there is one
    if current_snippet:
        combined_snippets.append(' '.join(current_snippet))

    # Return unique matching snippets, sorted by their order in the source text
    return sorted(set(combined_snippets), key=lambda snippet: source_text_clean.find(snippet))

# Function to compare input file with the database records
def check_plagiarism(input_text):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content FROM documents")
    documents = cursor.fetchall()
    conn.close()

    plagiarized_results = []
    
    for doc_id, title, content in documents:
        match_ratio = difflib.SequenceMatcher(None, input_text, content).ratio()
        if match_ratio > 0.3:  # Set a threshold for plagiarism detection (e.g., 30%)
            matching_snippets = get_snippets(content, input_text)
            if matching_snippets:
                plagiarized_results.append({
                    'document_id': doc_id,
                    'title': title,
                    'match_percentage': round(match_ratio * 100, 2),
                    'matching_snippets': matching_snippets
                })
    
    return plagiarized_results

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Upload file route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read and process file content
        input_text = read_file(file_path)

        # Check for plagiarism
        plagiarism_results = check_plagiarism(input_text)

        return jsonify(plagiarism_results)

    return jsonify({"error": "Invalid file type"})

# Main entry point
if __name__ == "__main__":
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True)
