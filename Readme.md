# Document Processing and Q&A System

This project is a web application built with Flask that allows users to upload various types of documents (PDF, text, CSV, JSON, DOCX) and query the content for relevant information. It leverages machine learning models and FAISS indexing to extract text from documents, generate embeddings, and perform similarity searches.

## Features

- **Document Upload**: Users can upload documents to the system.
- **Text Extraction**: Extracts text from different file formats like PDF, TXT, CSV, JSON, DOCX.
- **Embedding Generation**: Generates embeddings using the Gemini API.
- **FAISS Indexing**: Uses FAISS for fast retrieval of similar documents.
- **Question Answering**: Allows users to input queries and retrieve relevant responses based on the uploaded document.

## Technologies Used

- **Flask**: Web framework for building the web app.
- **Boto3**: AWS SDK for interacting with AWS services (S3).
- **FAISS**: Library for efficient similarity search and clustering.
- **GeminiEmbedding**: Embedding generation from Gemini API.
- **PyMuPDF**: PDF text extraction.
- **Pandas**: Data handling (for CSV files).
- **python-docx**: DOCX text extraction.
- **Requests**: For making API calls to Gemini API.

## Prerequisites

To run this project locally, ensure you have the following:

- Python 3.12
- Flask
- Boto3
- FAISS
- PyMuPDF
- pandas
- python-docx

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yash161/superchargeplus-ai-rag-yashshah.git
    ```

2. Navigate to the project directory:

    ```bash
    cd superchargeplus-ai-rag-yashshah
    ```

3. Create a virtual environment:

    - On Windows:
      ```bash
      python -m venv venv
      ```

    - On macOS/Linux:
      ```bash
      python3 -m venv venv
      ```

4. Activate the virtual environment:

    - On Windows:
      ```bash
      source venv/Scripts/activate
      ```

    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

5. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

6. Set up AWS credentials for S3 access:

    - Add your `AWS_ACCESS_KEY` and `AWS_SECRET_KEY` to the environment variables or use an AWS credentials file.

7. Run the Flask application:

    ```bash
    python app.py
    ```

8. The app will be running at `http://127.0.0.1:8000/`.

## Scalability & Optimization

One of the key focuses of this project is its scalability and optimization to handle large numbers of document uploads and queries effectively. Here's how these factors are addressed:

- **Scalable Cloud Hosting**: The project is hosted on an EC2 Linux instance, enabling easy scaling. While the instance has been stopped due to cost considerations, it can be restarted anytime based on user requirements. The project is designed to scale horizontally, making it capable of handling multiple concurrent users.
  
- **S3 Integration**: By leveraging AWS S3 for file storage, multiple documents can be uploaded, stored, and processed without impacting performance. This allows the system to manage large volumes of data efficiently, while ensuring easy retrieval and processing of files.

- **Efficient Similarity Search with FAISS**: FAISS is used for fast and optimized similarity search. Its high-performance indexing allows the system to handle large datasets and return relevant results quickly, even with multiple document uploads.

- **Optimized API Calls**: The Gemini API integration is optimized to minimize unnecessary calls and ensure that embedding generation is fast and reliable, supporting the overall system’s efficiency.

## Creativity & Problem-Solving

This project showcases creativity and problem-solving across several areas:

- **Multi-format Document Processing**: Handling different file formats (PDF, TXT, CSV, JSON, DOCX) in a seamless manner required creative approaches to text extraction, such as using PyMuPDF for PDFs and python-docx for DOCX files. The system adapts flexibly to various document types, making it versatile for real-world usage.

- **Embedding Generation & Similarity Search**: One of the main challenges was ensuring that the system could understand and retrieve relevant information from documents. By integrating the Gemini API for embedding generation and using FAISS for similarity searches, the system is capable of providing highly relevant answers to user queries, based on the content of uploaded documents.

- **API Integration with AWS**: Solving the problem of scalable file storage was achieved using AWS S3, allowing for secure and efficient document uploads. Coupled with EC2 hosting, this solution provides a robust cloud-based system that can be easily scaled up when needed.

- **Optimizing User Experience**: The interface has been designed to be user-friendly, with clear instructions for uploading and processing files. The focus on simplicity helps reduce friction for users, ensuring that they can quickly upload documents and get answers without facing technical hurdles.

## Hosting

The project is hosted on an EC2 Linux instance, but it has been stopped due to cost considerations. If you'd like to check it out, please let me know, and I will start the instance for you. The project is fully scalable, and the EC2 instance can be restarted as needed to accommodate more users. 

Additionally, the files are uploaded to an S3 bucket, allowing for multiple documents to be uploaded and processed concurrently without affecting performance.

## API Endpoints

### Upload File

- **URL**: `/upload`
- **Method**: `POST`
- **Description**: Uploads a file to the S3 bucket.
- **Parameters**: 
  - `file` (form data): The file to be uploaded.

### Process File

- **URL**: `/process_file`
- **Method**: `POST`
- **Description**: Processes an uploaded file, retrieves similar texts, and generates a response to a user query.
- **Parameters**:
  - `file_url` (JSON): URL of the file stored in S3.
  - `query` (JSON): User query to be answered based on the document content.

## Example Usage

1. Upload a file to the system by sending a POST request to `/upload` with the file attached.
2. Once uploaded, call the `/process_file` endpoint with the URL of the uploaded file and a query for the system to answer.

## Environment Variables

- `AWS_ACCESS_KEY`: Your AWS access key for S3.
- `AWS_SECRET_KEY`: Your AWS secret key for S3.
- `API_KEY`: Your Gemini API key for embedding generation.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
