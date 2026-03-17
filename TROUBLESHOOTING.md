# Troubleshooting Guide

## Quick Diagnostics

Run this command from the project root:
```bash
diagnose.bat
```

This will check:
- ✓ Python installation
- ✓ pip availability
- ✓ Port availability (3000, 8000)
- ✓ Required files exist
- ✓ Backend imports working
- ✓ All dependencies installed

---

## Common Issues & Solutions

### 1. **Port Already In Use**

**Error:** 
```
Address already in use
Cannot bind to 127.0.0.1:8000
```

**Solution:**

Find process using port 8000:
```bash
netstat -ano | findstr :8000
```

Kill the process (replace `<PID>` with the process number):
```bash
taskkill /PID <PID> /F
```

---

### 2. **Python Not Found**

**Error:**
```
'python' is not recognized as an internal or external command
```

**Solution:**

1. Install Python 3.10+ from https://www.python.org/downloads/
2. **Important:** During installation, check "Add Python to PATH"
3. Restart your terminal/command prompt
4. Verify: `python --version`

---

### 3. **Dependencies Import Errors**

**Error:**
```
ModuleNotFoundError: No module named 'langchain_groq'
ImportError: cannot import name 'X' from 'Y'
```

**Solution:**

Reinstall all dependencies:
```bash
cd backend
pip install -r requirements.txt --upgrade
```

If still failing, clean and reinstall:
```bash
pip uninstall langchain langchain-core langchain-groq -y
pip install -r requirements.txt
```

---

### 4. **"sp is mandatory" Azure Error**

**Error:**
```
XML error: sp is mandatory. Cannot be empty
```

**This is NOT a startup error.** See [AZURE_FIX_QUICK.md](AZURE_FIX_QUICK.md) for fix.

Solution: Your SAS token is missing the `sp=` parameter. Regenerate token with proper permissions.

---

### 5. **Backend Starts but Returns 502 Bad Gateway**

**Error when accessing /api/analyze:**
```
502 Bad Gateway
```

**Solution:**

1. Check backend is running:
   ```bash
   curl http://127.0.0.1:8000/docs
   ```
   Should return API documentation page

2. Check Groq API key in `.env`:
   ```
   GROQ_API_KEY=gsk_***  (should NOT be empty)
   ```

3. Test Groq connection:
   ```bash
   python -c "from groq import Groq; print('OK')"
   ```

---

### 6. **Frontend Shows "Cannot GET /upload"**

**Error:**
```
Cannot GET /upload
```

**Solution:**

1. Make sure frontend is running:
   - Should see "Local: http://localhost:3000" in terminal

2. Check React dev server:
   ```bash
   cd frontend
   npm run dev
   ```

3. Clear browser cache:
   - Press `F12` → Application → Clear Storage
   - Refresh page

---

### 7. **"npm: command not found"**

**Error:**
```
'npm' is not recognized as an internal or external command
```

**Solution:**

1. Install Node.js from https://nodejs.org/ (includes npm)
2. Restart terminal/command prompt
3. Verify: `npm --version`

---

### 8. **Backend Slow to Start / "KeyboardInterrupt"**

**Issue:**
- Backend takes 30+ seconds to start
- Getting stuck on imports
- `KeyboardInterrupt` errors

**Solution:**

This is the dependency loading issue. Fix:
```bash
cd backend
pip install --upgrade langchain-groq langchain-core
```

Then start normally (the --reload flag has been removed):
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

---

### 9. **".env Not Found" Warning**

**Warning (not critical):**
```
WARNING: .env file not found!
```

**Solution:**

Create `backend/.env` with:
```
GROQ_API_KEY=your_key_here
AZURE_STORAGE_CONNECTION_STRING=your_string
AZURE_SAS_TOKEN=your_token
SF_USERNAME=your_email@salesforce.com
SF_PASSWORD=your_password
SF_SECURITY_TOKEN=your_token
```

See `backend/.env.example` for template.

**Without .env:**
- Groq API calls will fail (errors in /analyze)
- Azure uploads will fail gracefully
- Salesforce records won't be created

---

### 10. **"Backend imports successfully" but still crashes**

**Issue:**
- `python -c "from main import app"` works
- But `uvicorn main:app` fails

**Solution:**

1. Check uvicorn error output carefully - look for specific service errors
2. If Groq key is invalid: Update `backend/.env` with real API key
3. If Azure/SF credentials wrong: System will still start but integration fails
4. Run diagnostic: `diagnose.bat`

---

## Step-by-Step Debugging

### If Nothing Works:

**Step 1:** Verify Python works
```bash
python -c "import sys; print(sys.version)"
```

**Step 2:** Verify pip works
```bash
pip list
```

**Step 3:** Install dependencies fresh
```bash
cd backend
pip install -r requirements.txt --upgrade --no-cache-dir
```

**Step 4:** Test imports
```bash
python -c "from main import app; print('OK')"
```

**Step 5:** Try starting with verbose output
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --log-level debug
```

**Step 6:** If still failing, check specific errors
```bash
python main.py
```

---

## Getting Help

**Provide the following when reporting issues:**

1. Output from: `diagnose.bat`
2. Python version: `python --version`
3. Node version: `npm --version`
4. Full error message from terminal (copy entire stack trace)
5. Files you've created (`.env`, startup scripts)

---

## Recovery Checklist

If exploring many troubleshooting steps, reset with:

```bash
# Clean Python cache
cd backend
python -m py_compile .
find . -type d -name __pycache__ -exec rm -rf {} +

# Clean pip cache
pip cache purge

# Reinstall everything
pip install -r requirements.txt --upgrade --no-cache-dir

# Test import
python -c "from main import app; print('OK')"
```

Then try starting again:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

---

## Success Indicators

When system is working correctly, you should see:

1. **Backend Terminal:**
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000
   INFO:     Application startup complete
   ```

2. **Frontend Terminal:**
   ```
   Local: http://localhost:3000
   Press q quit
   ```

3. **Browser:**
   - http://localhost:3000 loads
   - Can see upload form
   - API proxy works (uploads succeed)

---

## Contact & Next Steps

Once system is running:
1. Create `.env` with your credentials
2. Visit http://localhost:3000/upload
3. Upload a test CV
4. Check results page for all notifications
