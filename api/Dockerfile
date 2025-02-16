# Stage 1: Build
FROM python:3.12 AS builder

WORKDIR /app

# Create and activate virtual environment
RUN python3 -m venv venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Run
FROM python:3.12 AS runner

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/venv venv

# Ensure virtual environment is activated
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy application code
COPY . .

# Set environment variable for Flask app
ENV FLASK_APP=api/index.py

# Expose the application port
EXPOSE 8000

# Run Flask application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8000"]
