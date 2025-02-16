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
from flask import Flask, request, jsonify, render_template
from docx import Document
# from sentence_transformers import SentenceTransformer
from llama_index.embeddings.gemini import GeminiEmbedding
import textwrap
# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.config['ALLOWED_HOSTS'] = ['*']
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
# S3 Configurations
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

# Gemini API Key
API_KEY = "AIzaSyCVvXQwZtnJJqQHcq0vR-dC1QOdl4WjcRQ"  # Move this to env variables in production
# API_KEY = "a2acc367-5005-4c0c-8193-65ad2a351877"

def initialize_embedding_model(api_key: str, model_name: str = "models/embedding-001"):
    """Initialize the GeminiEmbedding model"""
    return GeminiEmbedding(model_name=model_name, api_key=api_key, title="this is a document")


def extract_text_from_json(data):
    """Recursively extract text from JSON objects."""
    if isinstance(data, dict):
        return " ".join(extract_text_from_json(value) for value in data.values())
    elif isinstance(data, list):
        return " ".join(extract_text_from_json(item) for item in data)
    else:
        return str(data)


def extract_text_from_document(file_path):
    print("Extracting Text from Document::**********************************************")
    """Extract text from various document formats."""
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


def get_embeddings(model, texts, batch_size=10):
    """Generate embeddings in batches to handle large files efficiently."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.get_text_embedding_batch(batch)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings, dtype=np.float32)


def create_faiss_index(embeddings):
    """Create and return a FAISS index."""
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    return faiss_index


def retrieve_similar_texts_from_query(model, faiss_index, query, texts, top_k=1):
    """Retrieve similar texts using FAISS index."""
    print("Retrieving similar texts from the document:")
    query_embedding = model.get_text_embedding_batch([query])
    query_embedding = np.array(query_embedding)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [texts[i] for i in indices[0]]

content_cache = TTLCache(maxsize=100, ttl=3600)

@cached(cache=content_cache)
def get_cached_or_extract_text(file_url):
    """Check if the file content is already cached or extract it from the document."""
    file_name = file_url.split("/")[-1]
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

    return extracted_text


def split_document(doc, max_size=9000):
    """Split the document into smaller chunks."""
    if isinstance(doc, list):  # Ensure it's a string
        doc = "\n".join(doc)
    chunks = textwrap.wrap(doc, max_size)
    return chunks


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

@app.route("/")
def index():
    """Render the index page."""
    return render_template("index.html")
import traceback

@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload a file to S3."""
    print("Received request to upload file")

    # Check if file is part of the request
    if "file" not in request.files:
        error_message = "No file provided in the request"
        print(f"Error: {error_message}")
        return jsonify({"error": error_message}), 400

    file = request.files["file"]
    print(f"File received: {file.filename}")
    print(f"File size: {len(file.read())} bytes")
    file.seek(0)  # Reset the file pointer after reading its size

    try:
        print(f"Uploading {file.filename} to S3 bucket {S3_BUCKET} in region {S3_REGION}...")

        # Attempt to upload file to S3
        s3.upload_fileobj(file, S3_BUCKET, file.filename)

        # Build file URL for successful upload
        file_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file.filename}"
        print(f"File uploaded successfully: {file_url}")

        return jsonify({"message": "File uploaded successfully", "file_url": file_url}), 200

    except Exception as e:
        # Log detailed error information
        error_message = f"Error uploading file: {str(e)}"
        print(error_message)

        # Optionally print stack trace for debugging purposes
        print("Stack Trace:")
        traceback.print_exc()

        # Return a detailed error message in the response
        return jsonify({"error": error_message, "details": traceback.format_exc()}), 500



@app.route("/process_file", methods=["POST"])
def process_file():
    """Process uploaded file, retrieve similar texts, and generate response."""
    file_url = request.json.get("file_url")
    query = request.json.get("query")

    if not file_url or not query:
        return jsonify({"error": "Missing file_url or query"}), 400

    try:
        # Extract filename from S3 URL
        # file_name = file_url.split("/")[-1]
        # temp_file_path = f"/tmp/{file_name}"
        #
        # # Read the file content directly from S3
        # s3_object = s3.get_object(Bucket=S3_BUCKET, Key=file_name)
        # file_content = s3_object["Body"].read()
        #
        # # Ensure /tmp exists
        # os.makedirs("/tmp", exist_ok=True)
        #
        # # Save the file locally
        # with open(temp_file_path, "wb") as f:
        #     f.write(file_content)
        #
        # # Verify if the file exists before proceeding
        # if not os.path.exists(temp_file_path):
        #     return jsonify({"error": f"File {temp_file_path} was not created successfully."}), 500
        #
        # # Extract text from document
        # extracted_text = extract_text_from_document(temp_file_path)
        # print("Extracted text::",extracted_text)
        # # Clean up temporary file
        # os.remove(temp_file_path)
        extracted_text = get_cached_or_extract_text(file_url)
        texts = [extracted_text]
        print("Text are ::",texts)
        text_chunks = split_document(extracted_text, max_size=9000)
        # Initialize embedding model
        embedding_model = initialize_embedding_model(API_KEY)

        # Generate embeddings
        # embeddings = np.array(get_embeddings(embedding_model, texts), dtype=np.float32)
        embeddings = get_embeddings(embedding_model, text_chunks, batch_size=10)
        # Create FAISS index
        faiss_index = create_faiss_index(embeddings)

        # Retrieve similar texts
        retrieved_texts = retrieve_similar_texts_from_query(embedding_model, faiss_index, query, text_chunks)

        # Generate response
        response_text = generate_response(API_KEY, retrieved_texts, query)

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)


