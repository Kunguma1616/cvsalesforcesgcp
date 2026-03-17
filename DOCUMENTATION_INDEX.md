# Documentation Index & Navigation

Need help? This guide points you to the right documentation.

---

## 🎯 Find Your Answer

### I want to...

#### 🚀 **Get the system running**
→ Start here: [QUICK_START.md](QUICK_START.md) (5 min read)
- How to start the system
- What files do what
- Expected output if working

#### 📋 **Set up everything from scratch**
→ Read: [GETTING_STARTED.md](GETTING_STARTED.md) (10 min checklist)
- Verify all requirements
- Create .env file
- Test prerequisites
- First-time setup

#### 🔧 **Understand configuration**
→ See: [SETUP.md](SETUP.md) (30 min read)
- Complete system architecture
- All API endpoints
- How components interact
- Detailed setup guide

#### ⚠️ **Something is broken**
→ Go to: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) (varies)
- Common issues
- Specific fixes
- Debugging steps
- When to use diagnostic tool

#### ☁️ **Set up Azure Storage**
→ Check: [AZURE_SETUP.md](AZURE_SETUP.md) (20 min)
- Step-by-step Azure configuration
- Storage account creation
- SAS token generation
- Container setup

#### 🔑 **Fix Azure SAS token error**
→ Quick fix: [AZURE_FIX_QUICK.md](AZURE_FIX_QUICK.md) (5 min)
- "sp is mandatory" error
- Generating valid token
- Permission requirements

#### 💻 **Start backend only**
→ Run: `backend/start_backend.bat`
- Or read: [STARTUP_GUIDE.md](STARTUP_GUIDE.md) for manual steps

#### 🎨 **Start frontend only**
→ Run: `frontend/start_frontend.bat`
- Or read: [STARTUP_GUIDE.md](STARTUP_GUIDE.md) for manual steps

#### 📊 **Check if system is healthy**
→ Run: `diagnose.bat`
- Tests Python/Node.js
- Checks ports
- Verifies dependencies
- Shows detailed status

#### 🤔 **Where do I upload the CV?**
→ **http://localhost:3000/upload** (after starting system)

#### 🔗 **How do I access the API?**
→ **http://127.0.0.1:8000/docs** (after starting backend)

---

## 📚 Documentation Map

### Quick References (5-10 minutes)
| Document | Content | When to Use |
|----------|---------|------------|
| [QUICK_START.md](QUICK_START.md) | Start everything, access points, common issues | First time running system |
| [START_BACKEND.bat](backend/start_backend.bat) | Click to start backend | Testing backend only |
| `diagnose.bat` | Check system health | Something's broken |

### Setup & Configuration (10-30 minutes)
| Document | Content | When to Use |
|----------|---------|------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Checklist, requirements, setup steps | Initial setup from scratch |
| [STARTUP_GUIDE.md](STARTUP_GUIDE.md) | Detailed startup instructions | Understanding startup process |
| [AZURE_SETUP.md](AZURE_SETUP.md) | Azure configuration walkthrough | Setting up cloud storage |
| [AZURE_FIX_QUICK.md](AZURE_FIX_QUICK.md) | Azure SAS token fix | "sp is mandatory" error |

### Detailed Documentation (30+ minutes)
| Document | Content | When to Use |
|----------|---------|------------|
| [SETUP.md](SETUP.md) | Complete system, architecture, all endpoints | Understanding full system |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues, debugging, fixes | Fixing problems |
| [README.md](README.md) | Project overview, features, tech stack | Overall understanding |

---

## 🔍 Filter by Problem Type

### 🔴 System Won't Start
1. Run: `diagnose.bat`
2. Read: [TROUBLESHOOTING.md](TROUBLESHOOTING.md#common-issues--solutions)
3. Look for your error message
4. Follow the fix

### 🟡 Can't Upload CV
1. Check backend is running: `python -c "from main import app"`
2. Check frontend is running: http://localhost:3000
3. Check .env has GROQ_API_KEY: `cat backend/.env | findstr GROQ`
4. See: [TROUBLESHOOTING.md](TROUBLESHOOTING.md#backend-starts-but-returns-502-bad-gateway)

### 🟠 Azure Upload Failing
1. See: [AZURE_FIX_QUICK.md](AZURE_FIX_QUICK.md)
2. Verify SAS token has `sp=rwdlacupitfx`
3. Read: [AZURE_SETUP.md](AZURE_SETUP.md)
4. If still failing: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### 🟢 Salesforce Record Not Creating
1. Check .env has SF credentials
2. Verify Salesforce account has Engineer_Application__c object
3. See: [SETUP.md](SETUP.md) - Salesforce configuration section

### ⚫ Port Already In Use
1. Find which process: `netstat -ano | findstr :8000`
2. Kill it: `taskkill /PID <PID> /F`
3. See: [TROUBLESHOOTING.md](TROUBLESHOOTING.md#1-port-already-in-use)

### ⚪ Slow Startup / Dependencies Won't Install
1. Run: `pip install -r backend/requirements.txt --upgrade`
2. See: [TROUBLESHOOTING.md](TROUBLESHOOTING.md#8-backend-slow-to-start--keyboardinterrupt)

### 💜 Still Stuck?
1. Run: `diagnose.bat` (shows detailed status)
2. Read: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) (covers 90% of issues)
3. Check: [SETUP.md](SETUP.md) (full technical details)
4. See: [STARTUP_GUIDE.md](STARTUP_GUIDE.md) (step-by-step walkthrough)

---

## 📖 Reading Order by Role

### I'm a New User
1. [QUICK_START.md](QUICK_START.md) - 5 min
2. [GETTING_STARTED.md](GETTING_STARTED.md) - 10 min
3. Run: `start_all.bat`
4. Upload a test CV
5. Done! ✓

### I Need to Configure Everything
1. [QUICK_START.md](QUICK_START.md) - Quick overview
2. [GETTING_STARTED.md](GETTING_STARTED.md) - Checklist
3. [SETUP.md](SETUP.md) - Full configuration
4. [AZURE_SETUP.md](AZURE_SETUP.md) - Cloud setup
5. [STARTUP_GUIDE.md](STARTUP_GUIDE.md) - Startup process

### I'm a System Administrator
1. [SETUP.md](SETUP.md) - Architecture & endpoints
2. [STARTUP_GUIDE.md](STARTUP_GUIDE.md) - Deployment
3. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Diagnostics
4. `diagnose.bat` - Health checks
5. Run: `start_all.bat`

### I'm Debugging Issues
1. Run: `diagnose.bat` - See health
2. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Find your error
3. Follow specific fix
4. If not there: [SETUP.md](SETUP.md) - Technical details

---

## 🗺️ File Navigation

```
cvsalesforcesgcp-main/
│
├── 📍 START HERE
│   ├── README.md ........................ Project overview
│   ├── QUICK_START.md .................. 5-min quick reference
│   ├── GETTING_STARTED.md .............. Setup checklist
│   └── start_all.bat ................... Click to run everything
│
├── 📚 GUIDES & DOCUMENTATION
│   ├── STARTUP_GUIDE.md ................ Detailed startup steps
│   ├── SETUP.md ........................ Complete system guide
│   ├── AZURE_SETUP.md .................. Azure configuration
│   ├── AZURE_FIX_QUICK.md .............. Azure quick fixes
│   ├── TROUBLESHOOTING.md .............. Common issues & fixes
│   └── DOCUMENTATION_INDEX.md .......... THIS FILE
│
├── 🔧 TOOLS & DIAGNOSTICS
│   ├── diagnose.bat .................... System health check
│   ├── backend/
│   │   ├── start_backend.bat ........... Start backend only
│   │   ├── verify_dependencies.py ...... Check dependencies
│   │   ├── main.py ..................... FastAPI application
│   │   └── .env.example ................ Template for secrets
│   │
│   └── frontend/
│       ├── start_frontend.bat .......... Start frontend only
│       ├── package.json ................ Node dependencies
│       └── src/
│           └── pages/
│               ├── UploadCVPage.tsx .... Resume upload
│               └── AnalysisResultPage.tsx .. Results display
```

---

## ⏱️ Time Estimates

| Task | Time | Document |
|------|------|----------|
| Quick start | 5 min | [QUICK_START.md](QUICK_START.md) |
| Initial setup | 15 min | [GETTING_STARTED.md](GETTING_STARTED.md) |
| First run | 2-5 min | [QUICK_START.md](QUICK_START.md) |
| Fix common issue | 5-10 min | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Full setup | 30-60 min | [SETUP.md](SETUP.md) |
| Azure config | 20 min | [AZURE_SETUP.md](AZURE_SETUP.md) |
| Debug system | 5-15 min | `diagnose.bat` + TROUBLESHOOTING |
| First test | 5 min | Upload CV after starting |

---

## 🆘 When in Doubt

1. **System won't start?**
   → Run `diagnose.bat` first
   → Then see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

2. **Don't know what to do?**
   → Start with [QUICK_START.md](QUICK_START.md)
   → Then run `start_all.bat`

3. **Want to understand everything?**
   → Read [SETUP.md](SETUP.md)
   → Full technical documentation

4. **Need to fix something specific?**
   → Search [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
   → Find your error, follow fix

5. **Azure issues?**
   → First: [AZURE_FIX_QUICK.md](AZURE_FIX_QUICK.md)
   → Then: [AZURE_SETUP.md](AZURE_SETUP.md)

---

## 🎯 Success Definition

You've successfully set up the system when:

✅ `start_all.bat` runs
✅ Backend window shows: "Application startup complete"
✅ Frontend window shows: "Local: http://localhost:3000"
✅ Browser opens to http://localhost:3000/upload
✅ Upload form is visible
✅ No red errors in either terminal

Next: Upload a CV and see results! 🎉

---

**Need help?** Check the right documentation above for your situation.

**First time?** Start with [QUICK_START.md](QUICK_START.md)
