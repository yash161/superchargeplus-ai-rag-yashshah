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

- Python 3.x
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

3. Create and activate a virtual environment:

    - On Windows:
      ```bash
      source venv/Scripts/activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Set up AWS credentials for S3 access:

    - Add your `AWS_ACCESS_KEY` and `AWS_SECRET_KEY` to the environment variables or use an AWS credentials file.

6. Run the Flask application:

    ```bash
    python app.py
    ```

7. The app will be running at `http://127.0.0.1:5000/`.

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
