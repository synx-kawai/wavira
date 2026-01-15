FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY *.py .

# Create data directory
RUN mkdir -p /app/data

# Default command (overridden in docker-compose)
CMD ["python", "--version"]
