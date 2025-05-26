# Base image
FROM python:3.10-slim

# Install build dependencies for scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    gcc \
    g++ \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Streamlit config
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.enableCORS=false"]

