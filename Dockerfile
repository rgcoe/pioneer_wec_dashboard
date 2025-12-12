FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fetch_wavss.py .
COPY webapp.py .

# Create output directory
RUN mkdir -p /app/output

# Expose Dash default port
EXPOSE 8050

# Run the Dash webapp continuously
CMD ["python", "webapp.py"]
