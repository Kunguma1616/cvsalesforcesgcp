# Getting Started Checklist

Use this checklist to verify your system is ready to run the CV Analysis application.

---

## ✅ System Requirements

- [ ] **Windows 10 or higher** (or macOS/Linux)
- [ ] **Python 3.10+** installed
  ```bash
  python --version
  # Should show: Python 3.10.x or higher
  ```
- [ ] **Node.js 16+** installed
  ```bash
  node --version
  npm --version
  # Should show version numbers
  ```

**Not installed?**
- Download Python: https://www.python.org/downloads/
  - **Important:** Check "Add Python to PATH" during installation
- Download Node.js: https://nodejs.org/
  - Includes npm automatically

---

## ✅ Project Files

From the project root (`cvsalesforcesgcp-main/`), verify these files exist:

- [ ] `start_all.bat` (or `start_all.sh` on Linux/Mac)
- [ ] `README.md`
- [ ] `QUICK_START.md`
- [ ] `STARTUP_GUIDE.md`
- [ ] `TROUBLESHOOTING.md`

Check subdirectories:

- [ ] `backend/main.py` (FastAPI application)
- [ ] `backend/requirements.txt`
- [ ] `backend/.env.example`
- [ ] `backend/start_backend.bat`
- [ ] `backend/verify_dependencies.py`
- [ ] `frontend/package.json`
- [ ] `frontend/start_frontend.bat`
- [ ] `frontend/src/pages/UploadCVPage.tsx`
- [ ] `frontend/src/pages/AnalysisResultPage.tsx`

**Files missing?**
- Re-download project from GitHub
- Ensure all files were extracted/cloned

---

## ✅ API Credentials (Required)

### Groq API Key (REQUIRED)

The system **cannot work** without this:

- [ ] Created account at https://console.groq.com/
- [ ] Generated API key
- [ ] Key format: `gsk_...` (long string)

**Don't have a key?**
1. Go to https://console.groq.com/
2. Sign up (free account)
3. Go to API Keys section
4. Click "Create API Key"
5. Copy the key (starts with `gsk_`)
6. Save it temporarily (you'll add to `.env` in next step)

### Azure Storage (OPTIONAL but Recommended)

Needed if you want to store PDFs in cloud:

- [ ] Azure Storage Account created
- [ ] Storage container named `resumes` created
- [ ] Connection string copied
- [ ] SAS token generated (MUST contain `sp=rwdlacupitfx`)

**Don't have Azure?**
- See [AZURE_SETUP.md](AZURE_SETUP.md) for step-by-step guide
- Or skip for now, system works without it

### Salesforce Credentials (OPTIONAL)

Needed if you want to create Salesforce records:

- [ ] Salesforce org/sandbox access
- [ ] Username (usually email)
- [ ] Password
- [ ] Security token (or reset it)

**Don't use Salesforce?**
- Leave these blank in `.env`
- System will work, just log errors gracefully

---

## ✅ Environment Configuration

### Create `.env` File

Create file: `backend/.env`

**Minimum (REQUIRED):**
```
GROQ_API_KEY=gsk_YOUR_KEY_HERE
```

**Full Configuration (OPTIONAL):**
```
GROQ_API_KEY=gsk_YOUR_KEY_HERE
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_CONTAINER_NAME=resumes
AZURE_SAS_TOKEN=sv=2023-11-09&ss=b&srt=sco&sp=rwdlacupitfx&se=...
SF_USERNAME=your_email@salesforce.com
SF_PASSWORD=your_password
SF_SECURITY_TOKEN=your_token
```

**Steps:**
1. Right-click in `backend/` folder
2. Select "New File"
3. Name it `.env` (note the dot prefix)
4. Open it and paste your credentials
5. Save

---

## ✅ Test Prerequisites

Run these commands to verify everything is ready:

```bash
# Test Python
python --version
# Should show: Python 3.10.x or higher

# Test Node.js
node --version
npm --version
# Both should show version numbers

# Test pip
pip --version
# Should show pip version

# Change to backend directory
cd backend

# Test that main.py exists
python -m py_compile main.py
# Should show no errors (or "main.pyc" created)

# Test .env file
cat .env
# Should show your GROQ_API_KEY (and optionally other vars)
```

All should work without errors.

---

## ✅ Dependency Installation

Before starting the system, install Python dependencies once:

```bash
cd backend
pip install -r requirements.txt
```

**Takes:** 2-5 minutes first time
**After:** Fast startup next time

Monitor for errors - fix any that appear (see TROUBLESHOOTING.md).

---

## ✅ Port Availability

Make sure these ports are free:

```bash
# Check port 8000
netstat -ano | findstr :8000
# Should show NOTHING or TIME_WAIT (not LISTEN)

# Check port 3000
netstat -ano | findstr :3000
# Should show NOTHING or TIME_WAIT (not LISTEN)
```

**Ports already in use?**
- See TROUBLESHOOTING.md section "Port Already In Use"
- Or close applications using those ports

---

## ✅ Ready to Start!

If you've checked all boxes above, you're ready:

```bash
# Double-click this file:
start_all.bat

# Or run manually:
cd backend && start start_backend.bat
# In another terminal:
cd frontend && start start_frontend.bat
```

---

## 📧 First Run Expectations

### Backend Terminal (first run):
```
Installing/updating dependencies... (may take 1-2 min)
✓ Python found
✓ Dependencies ready
[3/4] Starting FastAPI backend...
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Frontend Terminal (first run):
```
npm installing...
Added XX packages...

  Local:   http://localhost:3000
  Press q to quit
```

### Browser:
- Automatically opens to http://localhost:3000
- Should show "Upload CV" form

---

## 🎯 First Test (Optional)

Once system is running, test the full flow:

1. Go to http://localhost:3000/upload
2. Enter:
   - Name: "Test User"
   - Email: test@example.com
   - Trade: "Electrical" (or any trade)
   - Job Description: "Looking for experienced electrician"
3. Upload any PDF (even a blank one works for testing)
4. Click "Analyze CV"
5. Should see results page with:
   - ✓ ATS Score (number)
   - ✓ AI Score (number)
   - ✓ Blue "PDF Generated" box with download
   - ✓ Azure/Salesforce status (if configured)

If this works, **your system is fully operational!**

---

## 🚨 Common Issues Before Starting

### Issue: "python: command not found"
- Python not in PATH
- Solution: Reinstall Python, check "Add to PATH" box
- Verify: Close terminal, open new one, try again

### Issue: "npm: command not found"
- Node.js not installed
- Solution: Install Node.js from nodejs.org
- Verify: Close terminal, open new one, try again

### Issue: "Permission denied" on .env
- File has wrong permissions
- Solution: Delete .env, recreate it in editor
- Windows: Use Notepad or VS Code to create file

### Issue: "main.py" or "requirements.txt" not found
- Wrong directory
- Solution: Make sure current directory is `cvsalesforcesgcp-main/backend/`
- Use: `ls` or `dir` to verify files present

### Issue: "GROQ_API_KEY=gsk_..." in terminal output
- .env file exposed in logs
- This is okay - key is already in .env, just don't share it
- Best practice: Don't paste full keys in public places

---

## ✅ Final Verification

Before declaring "ready":

- [ ] Can run `python --version` → Shows 3.10+
- [ ] Can run `npm --version` → Shows valid version
- [ ] File `backend/.env` exists and contains `GROQ_API_KEY=gsk_...`
- [ ] File `backend/main.py` exists
- [ ] File `frontend/package.json` exists
- [ ] Ports 3000 and 8000 are free (no service listening)
- [ ] All files listed in "Project Files" section exist

---

## 🚀 Ready!

You can now:

```bash
# Start everything
start_all.bat
```

System will start in 10-30 seconds and open http://localhost:3000 automatically.

---

## 📚 Next Steps

1. **First run successful?** → Go to http://localhost:3000/upload
2. **Upload a CV and test** → See results with scores
3. **Want to understand more?** → Read [SETUP.md](SETUP.md)
4. **Something broke?** → See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 💡 Tips

- **Keep terminal windows open** - Shows what's happening
- **First startup slower** - Dependencies loading
- **Each restart faster** - Dependencies cached
- **Check terminal for errors** - React/Python will show issues there
- **Browser cache** - Press F12 → Application → Clear Storage if stuck

---

**You're all set!** 🎉

Start the system: `start_all.bat`
