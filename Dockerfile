FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
COPY *.py ./

# Set Python path to include src
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Create non-root user for security
RUN useradd -m -u 1000 llmuser && chown -R llmuser:llmuser /app
USER llmuser

# Default command to run tests
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Example usage:
# Build: docker build -t llm-fine-tune .
# Run tests: docker run --rm llm-fine-tune
# Run data prep: docker run --rm -v $(pwd)/data_inputs:/app/data_inputs llm-fine-tune python data_prep.py
# Run with HF fallback: docker run --rm llm-fine-tune python generate_with_olama.py --input data/sample_sft_small.jsonl --output /tmp/test.jsonl --use_hf_fallback --max_gen 2