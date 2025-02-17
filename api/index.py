import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS
import boto3
import pandas as pd
import fitz  # PyMuPDF
import faiss
import numpy as np
import textwrap
from cachetools import TTLCache, cached
import requests
import traceback
from flask import Flask, request, jsonify, render_template
from docx import Document
from llama_index.embeddings.gemini import GeminiEmbedding

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.config['ALLOWED_HOSTS'] = ['*']
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

S3_BUCKET = "filereaderflask"
S3_REGION = "us-east-1"
AWS_ACCESS_KEY = "AKIAVPEYWEKOSMOV3SKU"
AWS_SECRET_KEY = "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"

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION
)
CORS(app, resources={r"/*": {"origins": "*"}})

API_KEY = "AIzaSyCVvXQwZtnJJqQHcq0vR-dC1QOdl4WjcRQ"  
# Initialize Gemini Embedding Model
embedding_model = GeminiEmbedding(model_name="models/embedding-001", api_key=API_KEY)

# Cache for quick retrieval
stored_books = {}  # {filename: {"text": str, "embeddings": np.array, "index": FAISS object}}

# FAISS Index
faiss_indexes = {}  # {filename: FAISS Index object}

# Cache extracted content
content_cache = TTLCache(maxsize=100, ttl=3600)

@cached(cache=content_cache)
def get_cached_or_extract_text(file_url):
    """Retrieve extracted text from cache or extract if not available."""
    file_name = file_url.split("/")[-1]

    if file_name in stored_books:
        return stored_books[file_name]["text"]

    temp_file_path = f"/tmp/{file_name}"

    # Read the file content directly from S3
    s3_object = s3.get_object(Bucket=S3_BUCKET, Key=file_name)
    file_content = s3_object["Body"].read()

    # Save the file locally
    with open(temp_file_path, "wb") as f:
        f.write(file_content)

    # Extract text from document
    extracted_text = extract_text_from_document(temp_file_path)
    os.remove(temp_file_path)

    # Store extracted text
    stored_books[file_name] = {"text": extracted_text}

    return extracted_text


def extract_text_from_document(file_path):
    """Extract text from different document types."""
    _, ext = os.path.splitext(file_path)
    text = ""

    if ext.lower() == ".pdf":
        pdf_document = fitz.open(file_path)
        text = "\n".join([pdf_document[page].get_text() for page in range(pdf_document.page_count)])
        pdf_document.close()
    elif ext.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    elif ext.lower() == ".csv":
        df = pd.read_csv(file_path)
        text = " ".join(df.astype(str).values.flatten())
    elif ext.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            text = extract_text_from_json(data)
    elif ext.lower() == ".docx":
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = "Unsupported file type."

    return text


def extract_text_from_json(data):
    """Extract text from JSON objects."""
    if isinstance(data, dict):
        return " ".join(extract_text_from_json(value) for value in data.values())
    elif isinstance(data, list):
        return " ".join(extract_text_from_json(item) for item in data)
    else:
        return str(data)


def split_document(doc, max_size=9000):
    """Split document into chunks."""
    return textwrap.wrap(doc, max_size)


def get_embeddings(text_chunks):
    """Generate embeddings efficiently in batches."""
    return np.array(embedding_model.get_text_embedding_batch(text_chunks), dtype=np.float32)

@app.route("/")
def index():
    """Render the index page."""
    return render_template("index.html")
import traceback

def create_faiss_index(embeddings):
    """Create FAISS index and return."""
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    return faiss_index


@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload file to S3 and return file URL."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_name = file.filename

    try:
        s3.upload_fileobj(file, S3_BUCKET, file_name)
        file_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file_name}"
        return jsonify({"message": "File uploaded successfully", "file_url": file_url}), 200
    except Exception as e:
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


@app.route("/process_file", methods=["POST"])
def process_file():
    """Process uploaded file, retrieve similar texts, and generate response."""
    file_url = request.json.get("file_url")
    query = request.json.get("query")

    if not file_url or not query:
        return jsonify({"error": "Missing file_url or query"}), 400

    try:
        file_name = file_url.split("/")[-1]

        # Check if book is already stored
        if file_name in stored_books:
            extracted_text = stored_books[file_name]["text"]
            faiss_index = stored_books[file_name]["index"]
            text_chunks = split_document(extracted_text)
        else:
            extracted_text = get_cached_or_extract_text(file_url)
            text_chunks = split_document(extracted_text)

            # Generate embeddings and create FAISS index
            embeddings = get_embeddings(text_chunks)
            faiss_index = create_faiss_index(embeddings)

            # Store results for fast future access
            stored_books[file_name] = {"text": extracted_text, "embeddings": embeddings, "index": faiss_index}

        # Retrieve similar texts
        retrieved_texts = retrieve_similar_texts_from_query(faiss_index, query, text_chunks)

        # Generate response
        response_text = generate_response(API_KEY, retrieved_texts, query)

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def retrieve_similar_texts_from_query(faiss_index, query, texts, top_k=1):
    """Retrieve similar texts using FAISS index."""
    query_embedding = np.array(embedding_model.get_text_embedding_batch([query]))
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [texts[i] for i in indices[0]]


def generate_response(api_key, retrieved_docs, query, max_retries=5, backoff_factor=1):
    """Efficient API calling with retries & exponential backoff."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

    if isinstance(retrieved_docs, list):
        retrieved_docs = "\n".join(retrieved_docs)

    doc_chunks = split_document(retrieved_docs)

    final_response = []

    def request_chunk(chunk):
        prompt = f"""
               You are an AI assistant with expertise in information retrieval and natural language understanding.
               Below are some relevant documents retrieved based on a user's query. Your task is to generate an informative, coherent, and contextually relevant response using the given information.

               Retrieved Documents:
               {chunk}

               User Query: {query}

               Instructions:
               - Use only the information in the retrieved documents to answer the query.
               - If the documents contain partial or indirect information, infer as much as possible while staying within the context.
               - If the retrieved documents do not contain enough information, politely acknowledge this and provide a general response based on related concepts.

               Response:
               """

        for attempt in range(max_retries):
            response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
            if response.status_code == 200:
                return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text",
                                                                                                               "")

            if response.status_code == 429:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                return f"Error: {response.status_code}, {response.text}"

    # Parallel API calls using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(request_chunk, doc_chunks))

    return "\n".join(results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
