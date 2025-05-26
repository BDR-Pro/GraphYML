# Base image
FROM python:3.13-alpine

# Install build dependencies for scikit-learn
RUN apk update && apk add --no-cache \
    build-base \
    libffi-dev \
    gcc \
    g++ \
    python3-dev \
    py3-pip \
    curl \
    linux-headers \
    openblas-dev \
    && rm -rf /var/cache/apk/*

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
CMD ["streamlit", "run", "db.py", "--server.port=8501", "--server.enableCORS=false"]
