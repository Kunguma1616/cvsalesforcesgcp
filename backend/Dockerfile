                                                                                                                                                                                                                                                                    FROM node:22-alpine AS frontend-builder

                                                                                                                                                                                                                                                                    WORKDIR /frontend

                                                                                                                                                                                                                                                                    COPY frontend/package.json frontend/package-lock.json* ./
                                                                                                                                                                                                                                                                    RUN npm install --legacy-peer-deps

                                                                                                                                                                                                                                                                    COPY frontend/ ./
                                                                                                                                                                                                                                                                    RUN npm run build

                                                                                                                                                                                                                                                                    # Stage 2: Python backend
                                                                                                                                                                                                                                                                    FROM python:3.12-slim

                                                                                                                                                                                                                                                                    WORKDIR /app

                                                                                                                                                                                                                                                                    RUN apt-get update && apt-get install -y \
                                                                                                                                                                                                                                                                        gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in optimized order
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt
                                                                                                                                                                                                                                                                    RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
                                                                                                                                                                                                                                                                    USER appuser

                                                                                                                                                                                                                                                                    ENV PYTHONUNBUFFERED=1
                                                                                                                                                                                                                                                                    ENV PORT=8080

                                                                                                                                                                                                                                                                    EXPOSE 8080

                                                                                                                                                                                                                                                                    CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 75"]