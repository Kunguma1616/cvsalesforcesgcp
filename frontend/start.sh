#!/bin/bash
# start-frontend.sh - Start the React frontend

echo "🚀 Starting Fleet App Frontend..."
cd frontend

# Check if node_modules exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node dependencies..."
    npm install
fi

# Start development server
echo "🎬 Starting Vite development server on http://localhost:3000..."
npm run dev
