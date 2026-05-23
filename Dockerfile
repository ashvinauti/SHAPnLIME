FROM python:3.11-slim

LABEL maintainer="XAI-IDS Team"
LABEL description="XAI-IDS Pro — Explainable AI Intrusion Detection System"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow-cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create required directories
RUN mkdir -p logs models reports data

# Expose API and Dashboard ports
EXPOSE 8000 8501

# Default: start API
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
