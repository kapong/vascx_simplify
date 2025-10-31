# Dockerfile for testing simple-vascx library
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy package files
COPY pyproject.toml setup.py README.md ./
COPY src/ ./src/
COPY tests/ ./tests/

# Install the package with test dependencies
RUN pip install --no-cache-dir -e ".[test,dev]"

# Default command runs tests
CMD ["pytest", "-v", "--cov=simple_vascx", "--cov-report=term-missing"]
