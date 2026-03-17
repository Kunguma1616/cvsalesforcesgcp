# Quick Start Reference Card

## 🚀 Start Everything (Easiest)

**Option 1: Double-click this file:**
```
start_all.bat
```

**Option 2: Or run from Terminal:**
```bash
# Terminal 1
cd backend && start_backend.bat

# Terminal 2  
cd frontend && start_frontend.bat
```

---

## 📋 Files & What They Do

| File | Purpose | When to Use |
|------|---------|------------|
| `start_all.bat` | Start backend + frontend together | 99% of the time |
| `backend/start_backend.bat` | Start backend only | Debugging frontend |
| `frontend/start_frontend.bat` | Start frontend only | Debugging backend |
| `diagnose.bat` | Check system health | If something's broken |
| `STARTUP_GUIDE.md` | Detailed startup instructions | First time setup |
| `TROUBLESHOOTING.md` | Common issues & fixes | When errors occur |
| `SETUP.md` | Complete project setup | Full documentation |

---

## 🕐 What Happens After You Click `start_all.bat`

```
1. Backend window opens
   ↓ Installing/updating dependencies...
   ↓ Checking Python version...
   ↓ Verifying imports...
   ↓ Starting FastAPI on http://127.0.0.1:8000
   
2. Frontend window opens  
   ↓ Installing node_modules if needed...
   ↓ Starting Vite on http://localhost:3000
   
3. Browser opens automatically
   ↓ Shows upload form at http://localhost:3000/upload
```

**Expected to take:** 15-30 seconds first time (slower if first install)

---

## ✅ Success Checklist

After starting, verify:

- [ ] Backend window shows: `Application startup complete`
- [ ] Frontend window shows: `Local: http://localhost:3000`
- [ ] Browser opened to http://localhost:3000
- [ ] Upload form is visible
- [ ] No red error messages in terminals

---

## 🔴 If Something Fails

**Quick fixes in order:**

1. **Close everything** (both windows)
   ```bash
   Close backend window (Ctrl+C)
   Close frontend window (Ctrl+C)
   ```

2. **Run diagnostic**
   ```bash
   diagnose.bat
   ```

3. **Read error message** - it usually tells you exactly what's wrong

4. **See TROUBLESHOOTING.md** - find your error and follow the fix

5. **Still stuck?** Check these files:
   - [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
   - [STARTUP_GUIDE.md](STARTUP_GUIDE.md) - Detailed walkthrough
   - [SETUP.md](SETUP.md) - Full system architecture

---

## 🔑 Environment Setup

Before first use, create `backend/.env`:

```
GROQ_API_KEY=gsk_YOUR_KEY_HERE
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_CONTAINER_NAME=resumes
AZURE_SAS_TOKEN=sv=2023-11-09&ss=b&srt=sco&sp=rwdlacupitfx&...
SF_USERNAME=your_email@salesforce.com  
SF_PASSWORD=yourpassword
SF_SECURITY_TOKEN=your_security_token
```

See `backend/.env.example` for template.

**Without these credentials:**
- System still works but integrations won't (graceful failures)
- At minimum, add `GROQ_API_KEY` for analysis to work

---

## 🌐 Access Points

Once running:

| Service | URL | PURPOSE |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Upload CVs, view results |
| **Backend API** | http://127.0.0.1:8000 | REST API endpoints |
| **API Docs** | http://127.0.0.1:8000/docs | Read API documentation |
| **Swagger UI** | http://127.0.0.1:8000/redoc | Alternative API docs |

---

## 🎯 Typical User Flow

```
1. Visit http://localhost:3000/upload
2. Enter candidate info (name, email, trade)
3. Enter job description
4. Upload PDF resume
5. Click "Analyze CV"
6. See results with:
   ✓ ATS Score (keyword matching)
   ✓ AI Score (LLM analysis)
   ✓ Download PDF Report button
   ☁️ Azure Storage status
   💾 Salesforce Record ID (if created)
```

---

## 🛑 Stop Services

**From Terminal:**
- Press `CTRL+C` in each window

**Or close the windows directly**

**Or if stuck:**
```bash
taskkill /F /IM python.exe      # Kill backend
taskkill /F /IM node.exe        # Kill frontend
```

---

## 📱 Ports Used

- **3000** - Frontend (React)
- **8000** - Backend (FastAPI)

If ports are already in use, see TROUBLESHOOTING.md section "Port Already In Use"

---

## 💡 Pro Tips

1. **First time slower?** 
   - Dependencies install on first run (~5 min)
   - Next time is fast (10-20 sec)

2. **Want backend without frontend?**
   ```bash
   cd backend
   start_backend.bat
   ```

3. **Want to see API requests?**
   - Open http://127.0.0.1:8000/docs
   - Try endpoints there directly

4. **Need to reset everything?**
   ```bash
   # Delete node_modules to force clean install
   rm -r frontend/node_modules
   # Then restart
   start_all.bat
   ```

---

## 📚 Documentation

- **NEW USER?** → Read [STARTUP_GUIDE.md](STARTUP_GUIDE.md)
- **SOMETHING BROKEN?** → Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **WANT FULL DETAILS?** → See [SETUP.md](SETUP.md)
- **AZURE ISSUES?** → See [AZURE_SETUP.md](AZURE_SETUP.md)
- **AZURE SAS TOKEN?** → See [AZURE_FIX_QUICK.md](AZURE_FIX_QUICK.md)

---

**Ready to go!** 🚀

Try: `start_all.bat`
