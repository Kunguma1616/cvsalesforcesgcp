#!/bin/bash
# start-backend.sh - Start the Python backend server

echo "🚀 Starting Fleet App Backend..."
cd backend

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found in backend directory"
    echo "Please create backend/.env with the required environment variables"
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Run FastAPI app with uvicorn
echo "🎬 Starting FastAPI server on http://localhost:8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
