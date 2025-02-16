# Stage 1: Build
FROM python:3-alpine AS builder

WORKDIR /app

RUN python3 -m venv venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 2: Run
FROM python:3-alpine AS runner

WORKDIR /app

COPY --from=builder /app/venv venv
COPY . .

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV FLASK_APP=api/index.py

WORKDIR /app/api

EXPOSE 8000

CMD ["python", "index.py"]
