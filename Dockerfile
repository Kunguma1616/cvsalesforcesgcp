FROM node:22-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json ./
RUN npm install

COPY frontend/ ./
RUN chmod -R +x node_modules/.bin/
RUN npm run build

# Stage 2: Python backend
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
&& rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

COPY --from=frontend-builder /frontend/dist ./static

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 75"]
