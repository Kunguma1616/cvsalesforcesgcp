# Docker Setup Guide

## Quick Start with Docker Compose (Recommended)

```bash
# Build and run the complete app
docker-compose up --build

# The app will be available at http://localhost:8080
# Frontend will be served from the built static files
```

## Manual Docker Build & Run

### Option 1: Build from Root Directory
```bash
# Build the Docker image
docker build -f backend/Dockerfile -t cv-app .

# Run the container
docker run -p 8080:8080 \
  --env-file .env \
  cv-app
```

### Option 2: Build from Backend Directory
```bash
cd backend

# Build
docker build -t cv-app .

# Run
docker run -p 8080:8080 \
  --env-file ../.env \
  -v $(pwd):/app \
  cv-app
```

## Environment Variables

The following variables must be set in `.env`:

```
GROQ_API_KEY=your_key_here
MICROSOFT_CLIENT_ID=your_client_id
MICROSOFT_CLIENT_SECRET=your_secret
MICROSOFT_TENANT_ID=your_tenant_id
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_CONTAINER_NAME=resourcecv
AZURE_SAS_TOKEN=your_sas_token
SF_USERNAME=your_salesforce_user
SF_PASSWORD=your_salesforce_password
SF_SECURITY_TOKEN=your_salesforce_token
SF_DOMAIN=login
ALLOWED_EMAIL_DOMAIN=aspect.co.uk
```

## Troubleshooting

### Error: "ModuleNotFoundError" or "ImportError"
- Ensure Python 3.12 is used (matches your local environment)
- Clear Docker cache: `docker system prune -a`
- Rebuild: `docker-compose up --build`

### Error: "StreamlitDeprecationWarning" or conflicting dependencies
- The `streamlit` and `torch` packages have been removed from requirements.txt
- They're not needed for FastAPI backend and caused build conflicts

### Container exits immediately
- Check logs: `docker logs <container_id>`
- Verify .env file exists and has all required variables
- Ensure PORT 8080 is not already in use

### Frontend not loading
- The frontend must be built during Docker build (`npm run build`)
- Static files are copied to `/app/static`
- API proxy points to `http://localhost:8080`

## Production Deployment

For production, use the optimized Dockerfile command:
```dockerfile
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 75"]
```

Do NOT use `--reload` flag in production!

## Development with Docker

For development with live reload:
```bash
docker-compose up --build
```

This mounts the backend volume, allowing code changes to trigger reloads.
