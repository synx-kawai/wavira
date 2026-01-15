FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY *.py .

# Create data directory
RUN mkdir -p /app/data

# Default command (overridden in docker-compose)
CMD ["python", "--version"]
