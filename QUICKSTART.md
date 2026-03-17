# Getting Started with Fleet App

## 📋 Quick Start Guide

### 1. **Backend Setup (Terminal 1)**

```bash
cd backend
pip install -r requirements.txt
streamlit run app.py --server.port=8501
```

**OR use the batch file (Windows):**
```bash
backend\start.bat
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### 2. **Frontend Setup (Terminal 2)**

```bash
cd frontend
npm install
npm run dev
```

**OR use the batch file (Windows):**
```bash
frontend\start.bat
```

**Expected output:**
```
  VITE v5.x.x  ready in xxx ms
  ➜  Local:   http://localhost:3000/
```

## 🎯 Access the Application

- **Frontend Dashboard:** http://localhost:3000
- **Backend API:** http://localhost:8501

## ⚙️ Environment Variables

Create `backend/.env`:

```env
# Required minimum configuration
GROQ_API_KEY=sk-grok-xxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile

# Optional: Salesforce integration
SF_USERNAME=your-email@example.com
SF_PASSWORD=your-password
SF_SECURITY_TOKEN=your-token
SF_DOMAIN=login

# Optional: Azure Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_CONTAINER_NAME=fleet-app
AZURE_SAS_TOKEN=sv=2021-xx-xx&...
```

## 🎨 Frontend Features

- ✅ Fleet Dashboard with vehicle statistics
- ✅ Live vehicle allocation tracking
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Custom Fleet App branding with brand colors
- ✅ Navigation with AI Assistant, Analytics, Upload options

## 🔧 Backend Features

- ✅ Streamlit-based web interface
- ✅ PDF resume parsing and analysis
- ✅ Groq LLaMA AI integration for intelligent analysis
- ✅ Salesforce CRM integration
- ✅ Azure Blob Storage for document storage

## 📱 Responsive Breakpoints

The frontend is fully responsive:
- **Mobile:** 320px+
- **Tablet:** 768px+
- **Desktop:** 1024px+

## 🎨 Theme Customization

Edit colors in `frontend/src/config/colors.ts`:

```typescript
export const colors = {
  brand: {
    blue: "#27549D",      // Primary brand color
    yellow: "#F1FF24",    // Accent color
  },
  // ... more colors
};
```

## 🚨 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.12+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Try clearing cache
rm -rf ~/.streamlit
streamlit cache clear
```

### Frontend won't start
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Start with verbose output
npm run dev -- --debug
```

### Port already in use
```bash
# Find process using port 8501 (backend)
netstat -ano | findstr :8501

# Or use different port
streamlit run app.py --server.port=8502
```

## 📦 Project Structure

```
cvsalesforcesgcp-main/
├── backend/                 # Python Backend (Streamlit)
│   ├── app.py              # Main application
│   ├── requirements.txt     # Python packages
│   ├── Dockerfile
│   ├── start.bat           # Windows start script
│   └── start.sh            # Linux/Mac start script
│
├── frontend/               # React Frontend (Vite)
│   ├── src/
│   │   ├── components/     # Header, UI components
│   │   ├── pages/          # HomePage, other pages
│   │   ├── config/         # Theme & colors
│   │   ├── App.tsx         # Main app
│   │   └── main.tsx        # Entry point
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── start.bat           # Windows start script
│   └── start.sh            # Linux/Mac start script
│
├── README.md               # Full documentation
├── STARTUP.bat             # Windows startup helper
├── .gitignore
└── images.png              # (Old logo - can be replaced)
```

## 🔄 Development Workflow

1. **Backend Development:**
   - Edit `backend/app.py`
   - Streamlit auto-reloads changes
   - Check terminal for errors

2. **Frontend Development:**
   - Edit files in `frontend/src/`
   - Vite provides hot module reloading (HMR)
   - Check browser console for errors

3. **Styling:**
   - Use Tailwind CSS classes
   - Or edit `frontend/src/index.css`
   - Color palette in `frontend/src/config/colors.ts`

## 📝 Building for Production

### Backend
```bash
cd backend
docker build -f Dockerfile -t fleet-app-backend:latest .
```

### Frontend
```bash
cd frontend
npm run build
# Output: dist/ folder ready for deployment
```

## 🆘 Need Help?

1. Check browser console (F12) for frontend errors
2. Check terminal output for backend errors
3. Verify `.env` file has correct values
4. Ensure ports 3000 and 8501 are not in use
5. Try stopping and restarting both servers

---

**Version:** 1.0.0  
**Last Updated:** March 2026
