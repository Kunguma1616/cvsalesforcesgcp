FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of app
COPY . .

# Expose Streamlit default port
EXPOSE 8080

# Cloud Run: Streamlit must run on correct port
ENV PORT=8080

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
