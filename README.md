# CV Analysis System

AI-powered resume analysis with Groq LLM, Azure Storage integration, and Salesforce CRM connectivity.

## 🚀 Quick Start

### Option 1: Run Everything at Once (Recommended)
```bash
start_all.bat
```

This will:
- Start FastAPI backend on http://127.0.0.1:8000
- Start React frontend on http://localhost:3000
- Open browser automatically

### Option 2: Manual Start
```bash
# Terminal 1 - Backend
cd backend
start_backend.bat

# Terminal 2 - Frontend
cd frontend
start_frontend.bat
```

**Then open:** http://localhost:3000/upload

---

## 📚 Documentation

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[QUICK_START.md](QUICK_START.md)** | Quick reference card | First 5 minutes |
| **[STARTUP_GUIDE.md](STARTUP_GUIDE.md)** | Detailed startup steps | Setting up for first time |
| **[SETUP.md](SETUP.md)** | Complete project guide | Full understanding |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues & fixes | When something breaks |
| **[AZURE_SETUP.md](AZURE_SETUP.md)** | Azure configuration | Setting up cloud storage |
| **[AZURE_FIX_QUICK.md](AZURE_FIX_QUICK.md)** | Azure SAS token fix | If Azure upload fails |

**👉 New user? Start with [QUICK_START.md](QUICK_START.md)**

---

## 🏗️ Project Structure

```
cvsalesforcesgcp-main/
├── start_all.bat                 # ← Click this to start everything
├── QUICK_START.md                # ← Read this first
├── STARTUP_GUIDE.md              # Full startup instructions
├── TROUBLESHOOTING.md            # Common issues & fixes
├── diagnose.bat                  # Run this if something breaks
│
├── backend/                      # FastAPI Server (Python)
│   ├── main.py                   # Main application (501 lines)
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Template for credentials
│   ├── .env                      # Your credentials (create this!)
│   ├── start_backend.bat         # Start backend only
│   └── verify_dependencies.py    # Check if dependencies installed
│
└── frontend/                     # React Frontend (TypeScript/Vite)
    ├── src/
    │   ├── pages/
    │   │   ├── UploadCVPage.tsx           # Resume upload form
    │   │   ├── AnalysisResultPage.tsx    # Results display
    │   │   └── AnalysisReportsPage.tsx   # Reports dashboard
    │   ├── components/
    │   │   ├── Header.tsx
    │   │   ├── Sidebar.tsx
    │   │   └── MainLayout.tsx
    │   ├── config/
    │   │   ├── colors.ts          # Brand colors
    │   │   └── theme.ts           # Theme configuration
    │   └── App.tsx                # Main app
    ├── package.json               # Node dependencies
    ├── vite.config.ts             # Vite configuration
    └── start_frontend.bat         # Start frontend only
```

---

## ✨ Features

### Resume Analysis
- 📄 PDF upload and text extraction
- 🎯 ATS keyword scoring
- 🤖 LLM-powered job matching analysis (using Groq)
- 📊 7-10 evaluation criteria with detailed scoring
- 💼 Skills identification and matching

### Report Generation
- 📑 Professional 3+ page PDF reports
- 📈 Visual score charts and metrics
- 📋 Structured evaluation results
- ⬇️ Download button for reports

### Cloud Integration
- ☁️ Azure Blob Storage for file hosting
- 🔗 Public SAS-signed URLs for downloads
- 💾 Salesforce CRM record creation
- 🔔 Real-time status notifications

### User Interface
- 🎨 Responsive React UI with Tailwind CSS
- 📱 Mobile-friendly design
- 🎯 Real-time feedback
- 🟢 Status indicators for storage/CRM
- ⚡ Fast Vite dev server

---

## 🔧 Technology Stack

### Backend
- **Framework:** FastAPI (Python)
- **LLM:** Groq API (llama-3.3-70b-versatile)
- **PDF Processing:** pdfminer.six (extraction) + reportlab (generation)
- **Cloud Storage:** Azure Blob Storage
- **CRM:** Salesforce
- **Server:** Uvicorn
- **Port:** 8000

### Frontend
- **Framework:** React + TypeScript
- **Build Tool:** Vite
- **Styling:** Tailwind CSS + PostCSS
- **HTTP Client:** Axios
- **Icons:** Lucide React
- **Port:** 3000

---

## 📋 Requirements

- **Python:** 3.10 or higher
- **Node.js:** 16 or higher (includes npm)
- **API Keys:**
  - Groq API key (from console.groq.com)
  - Azure Storage credentials
  - Salesforce credentials (optional)

---

## ⚙️ Configuration

### 1. Create `.env` file

Create `backend/.env` with your credentials:

```
# Groq API (REQUIRED - system won't work without this)
GROQ_API_KEY=gsk_YOUR_KEY_HERE

# Azure Storage (OPTIONAL but recommended)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_CONTAINER_NAME=resumes
AZURE_SAS_TOKEN=sv=2023-11-09&ss=b&srt=sco&sp=rwdlacupitfx&...

# Salesforce (OPTIONAL)
SF_USERNAME=your_email@salesforce.com
SF_PASSWORD=your_password
SF_SECURITY_TOKEN=your_security_token
```

See `backend/.env.example` for full template.

### 2. Get Your Keys

**Groq API Key:** https://console.groq.com/
- Create account
- Generate API key
- Copy to `.env`

**Azure Storage:** See [AZURE_SETUP.md](AZURE_SETUP.md)

**Salesforce:** See [SETUP.md](SETUP.md)

---

## 🎯 Typical Workflow

1. **Start the system**
   ```bash
   start_all.bat
   ```

2. **Upload CV**
   - Go to http://localhost:3000/upload
   - Enter candidate name, email, trade, job description
   - Upload PDF resume

3. **View Results**
   - See ATS score (keyword matching)
   - See AI score (LLM analysis)
   - Download PDF report
   - View Azure storage status
   - Check Salesforce record creation

4. **Download & Integrate**
   - Download PDF report
   - View file in Azure (if configured)
   - Check Salesforce for created record

---

## 🐛 Troubleshooting

### Port Already In Use
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill it (replace <PID> with process number)
taskkill /PID <PID> /F
```

### Dependencies Not Installing
```bash
cd backend
pip install -r requirements.txt --upgrade
```

### Backend Won't Start
Run diagnostic:
```bash
diagnose.bat
```

Then check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for your specific error.

### Something Still Broken?
1. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - most issues covered there
2. See [STARTUP_GUIDE.md](STARTUP_GUIDE.md) - detailed setup walkthrough
3. See [SETUP.md](SETUP.md) - complete system documentation

---

## 🔍 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/analyze` | Upload CV and get analysis |
| GET | `/health` | Check if backend is running |
| GET `/docs` | Swagger UI | Test endpoints interactively |

**Try API:** http://127.0.0.1:8000/docs

---

## 📊 Project Status

✅ **Complete Features:**
- FastAPI backend with full resume analysis
- React frontend with upload form and results display
- PDF text extraction and ATS scoring
- LLM-based structured analysis (7-10 criteria)
- Professional PDF report generation
- Azure Blob Storage integration
- Salesforce record creation
- Windows startup scripts
- Comprehensive documentation

🔧 **Requires Configuration:**
- Groq API key (required for analysis)
- Azure credentials (optional, graceful failure)
- Salesforce credentials (optional, graceful failure)

📈 **Future Enhancements:**
- Analytics dashboard
- Batch CV processing
- Additional CRM integrations
- Custom LLM models

---

## 🚦 Getting Help

### If Something Breaks:

1. **Run diagnostic:**
   ```bash
   diagnose.bat
   ```

2. **Check error message** - it's usually specific

3. **See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - find your error type

4. **Still stuck?** Check full docs:
   - [STARTUP_GUIDE.md](STARTUP_GUIDE.md) - Step by step
   - [AZURE_SETUP.md](AZURE_SETUP.md) - Azure specific
   - [SETUP.md](SETUP.md) - Complete manual

---

## 📝 License

See LICENSE file for details.

---

**Ready to start?** 👉 [QUICK_START.md](QUICK_START.md)

Or just double-click: `start_all.bat`

Green: #2EB844
Red: #D15134
Gray: #848EA3
```

See [frontend/src/config/colors.ts](frontend/src/config/colors.ts) for complete color configuration.

## 🚀 Getting Started

### Prerequisites

- **Backend:** Python 3.12+, pip
- **Frontend:** Node.js 18+, npm
- **Environment:** Groq API Key, Azure credentials (optional for cloud storage)

### ⚙️ Configuration

#### Backend Setup

1. **Create `.env` file in `backend/` directory:**

```env
# Groq API
GROQ_API_KEY=your-api-key-here
GROQ_MODEL=llama-3.3-70b-versatile

# Salesforce
SF_USERNAME=your-sf-username
SF_PASSWORD=your-sf-password
SF_SECURITY_TOKEN=your-sf-token
SF_DOMAIN=login

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
AZURE_CONTAINER_NAME=your-container-name
AZURE_SAS_TOKEN=your-sas-token
```

2. **Install dependencies:**

```bash
cd backend
pip install -r requirements.txt
```

3. **Run the backend:**

```bash
cd backend
streamlit run app.py
```

Backend will be available at: **http://localhost:8501**

#### Frontend Setup

1. **Install dependencies:**

```bash
cd frontend
npm install
```

2. **Start development server:**

```bash
cd frontend
npm run dev
```

Frontend will be available at: **http://localhost:3000**

## 🎯 Running Both Servers

### Terminal 1 - Backend
```bash
cd backend
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

### Using Start Scripts (Linux/Mac)

```bash
# Terminal 1
./backend/start.sh

# Terminal 2
./frontend/start.sh
```

## 📝 Available Scripts

### Backend
```bash
cd backend
streamlit run app.py              # Run development server
streamlit run app.py --logger.level=debug  # Debug mode
```

### Frontend
```bash
cd frontend
npm run dev                       # Start development server
npm run build                     # Build for production
npm run preview                   # Preview production build
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build backend image
docker build -f backend/Dockerfile -t fleet-app-backend:latest backend/

# Run backend container
docker run -p 8501:8501 --env-file backend/.env fleet-app-backend:latest

# Build frontend image (create Dockerfile in frontend/ if needed)
# Run frontend container
docker run -p 3000:3000 fleet-app-frontend:latest
```

## 📦 Dependencies

### Backend
- **streamlit** - Web app framework
- **langchain-groq** - Groq integration
- **simple-salesforce** - Salesforce API
- **azure-storage-blob** - Azure cloud storage
- **reportlab** - PDF generation
- **pydantic** - Data validation

### Frontend
- **react** - UI framework
- **react-router-dom** - Routing
- **tailwindcss** - Styling
- **typescript** - Type safety
- **lucide-react** - Icons
- **vite** - Build tool

## 🎨 Customization

### Update Brand Colors

Edit [frontend/src/config/colors.ts](frontend/src/config/colors.ts):

```typescript
export const colors = {
  brand: {
    blue: "#27549D",     // Change primary blue
    yellow: "#F1FF24",   // Change accent yellow
  },
  // ... more colors
};
```

### Modify Homepage Layout

Edit [frontend/src/pages/HomePage.tsx](frontend/src/pages/HomePage.tsx)

### Add New Components

Place new components in `frontend/src/components/`

## 🔄 API Integration

Frontend connects to backend via proxy configured in [frontend/vite.config.ts](frontend/vite.config.ts):

```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8501',
    changeOrigin: true,
  }
}
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fleet App Frontend                        │
│  (React/TypeScript + Tailwind CSS + Vite)                   │
│  - Dashboard Component                                       │
│  - Fleet Statistics                                          │
│  - Analytics Views                                           │
│  - Responsive UI                                             │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Fleet App Backend                          │
│  (Python Streamlit + Groq LLaMA)                            │
│  - Resume Analysis Engine                                   │
│  - Fleet Data Processing                                    │
│  - Salesforce Integration                                   │
│  - Azure Storage Integration                                │
└──────────────────────────────────────────────────────────────┘
```

## 🔐 Security

- Use environment variables for sensitive data (.env files)
- Never commit .env files to version control
- Rotate API keys periodically
- Use CORS policies in production

## 📞 Support

For issues or questions:
1. Check the error messages in browser console
2. Review backend logs in terminal
3. Verify environment variables are set correctly
4. Ensure both servers are running

## 📄 License

Proprietary - Fleet App

---

**Last Updated:** March 2026
**Version:** 1.0.0
