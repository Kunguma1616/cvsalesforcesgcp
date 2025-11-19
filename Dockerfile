# Use Python 3.12 slim
FROM python:3.12-slim

# Install system-level dependencies required by:
# - reportlab (fonts, rendering libs)
# - matplotlib (freetype, png)
# - pdfminer.six (libxml, libxslt)
# - numpy / pandas / scipy (build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libxml2 \
    libxml2-dev \
    libxslt1.1 \
    libxslt1-dev \
    libcairo2 \
    libfreetype6 \
    libfreetype6-dev \
    libpng16-16 \
    libpng-dev \
    libjpeg62-turbo \
    libjpeg62-turbo-dev \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Work directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file first
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code (this includes app.py)
COPY . .

# Cloud Run env for Streamlit
ENV PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_HEADLESS=true
ENV STREAMLIT_TELEMETRY=false
ENV PYTHONUNBUFFERED=1

# Start Streamlit on Cloud Run
CMD ["bash", "-lc", "streamlit run app.py \
  --server.port ${PORT} \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false"]
