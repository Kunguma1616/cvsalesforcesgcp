# Backend Startup Guide

## Quick Start (Recommended)

### **Option 1: Use the Startup Script** (Easiest)

Double-click this file:
```
start_all.bat
```

This will:
- ✓ Start Backend on http://127.0.0.1:8000
- ✓ Start Frontend on http://localhost:3000
- ✓ Open browser automatically

**That's it!** Both services run in separate windows.

---

## Manual Start (If Script Doesn't Work)

### **Terminal 1 - Start Backend:**

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### **Terminal 2 - Start Frontend:**

```bash
cd frontend
npm run dev
```

Expected output:
```
Local:   http://localhost:3000
Press q to quit
```

---

## Common Issues & Fixes

### **Issue: "transformers import error" / "torch not found"**

**Fix:** Reinstall dependencies
```bash
cd backend
pip install -r requirements.txt --upgrade
```

### **Issue: "No such option: --reload"**

**Cause:** Typo in command (make sure it's `--reload` not `--reloadain`)

**Fix:** Use the startup scripts - they have the correct command

### **Issue: Port 8000 already in use**

**Fix:** Kill the process using port 8000
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Then restart backend.

### **Issue: Frontend won't compile**

**Fix:** Clear cache and reinstall
```bash
cd frontend
rm -r node_modules package-lock.json
npm install
npm run dev
```

---

## File Structure

```
cvsalesforcesgcp-main/
├── start_all.bat          ← Click this to start everything
├── backend/
│   ├── start_backend.bat  ← Click to start backend only
│   ├── main.py            ← FastAPI app
│   └── .env               ← Your credentials (required!)
├── frontend/
│   ├── start_frontend.bat ← Click to start frontend only
│   └── package.json       ← React app
└── SETUP.md               ← Full setup guide
```

---

## How to Access

Once both services are running:

- **Upload CV:** http://localhost:3000/upload
- **View Results:** http://localhost:3000/analysis-result
- **API Health:** http://127.0.0.1:8000/health
- **API Docs:** http://127.0.0.1:8000/docs

---

## Stopping Services

### Using the Start Script
- Close the "Backend" window
- Close the "Frontend" window

### Manual Start
- Press `CTRL+C` in each terminal

---

## Example Workflow

1. Double-click **start_all.bat**
2. Wait for browser to open at http://localhost:3000
3. Go to **Upload CV** page
4. Fill in candidate info and upload PDF
5. View analysis results
6. Download PDF and verify Salesforce record

---

## Environment Setup

Before first run, create `backend/.env`:

```
GROQ_API_KEY=your_api_key
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_CONTAINER_NAME=resumes
AZURE_SAS_TOKEN=your_sas_token
SF_USERNAME=your_salesforce_email
SF_PASSWORD=your_salesforce_password
SF_SECURITY_TOKEN=your_security_token
```

See `.env.example` for template.

---

## Troubleshooting Commands

**Check Python version:**
```bash
python --version  # Should be 3.10+
```

**Check if dependencies are installed:**
```bash
pip list | findstr langchain
```

**Test backend imports:**
```bash
cd backend
python -c "from main import app; print('OK')"
```

**Check ports:**
```bash
netstat -ano | findstr :8000
netstat -ano | findstr :3000
```

---

## Getting Help

1. Check logs in terminal - look for red error messages
2. See SETUP.md for detailed configuration
3. See AZURE_SETUP.md for Azure issues
4. See AZURE_FIX_QUICK.md for quick Azure fixes
