# Microsoft OAuth Authentication - Quick Start

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Azure AD tenant (Microsoft 365 admin access)
- Backend: Python 3.8+, pip
- Frontend: Node.js 16+, npm
- Text editor for .env files

---

## Step 1: Register App in Azure AD (3-5 min)

### 1.1 Go to Azure Portal
```
https://portal.azure.com
→ Search "App registrations" → Click it
```

### 1.2 Register New App
```
Click "New registration"
Name: CV Analysis Portal
Account type: Any Azure AD directory (multitenant)
Click Register
```

### 1.3 Copy Credentials
On the app overview page:
- **Copy "Application (client) ID"** → Save as `MICROSOFT_CLIENT_ID`

Go to "Certificates & secrets":
- Click "New client secret"
- **Copy secret value** → Save as `MICROSOFT_CLIENT_SECRET`

### 1.4 Set Redirect URI
Go to "Authentication":
- Click "Add URI"
- Add: `http://localhost:8000/api/auth/callback/microsoft`
- Click Save

### 1.5 Add Permissions
Go to "API permissions":
- Click "Add a permission"
- Select "Microsoft Graph" → "Delegated"
- Search and add: `openid`, `profile`, `email`, `User.Read`
- Click "Add permissions"

---

## Step 2: Create .env Files (2 min)

### Backend: `/backend/.env`
```env
MICROSOFT_CLIENT_ID=paste_your_client_id_here
MICROSOFT_CLIENT_SECRET=paste_your_secret_here
MICROSOFT_TENANT_ID=common
FRONTEND_URL=http://localhost:5173
BACKEND_URL=http://localhost:8000
ALLOWED_EMAIL_DOMAIN=aspect.co.uk

# ... rest of your existing config ...
GROQ_API_KEY=...
AZURE_STORAGE_CONNECTION_STRING=...
# etc.
```

### Frontend: `/frontend/.env`
```env
VITE_API_URL=http://localhost:8000
```

---

## Step 3: Install & Run (2-3 min)

### Terminal 1 - Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
✅ Should show: `[LAUNCH] Starting server on http://127.0.0.1:8000`

### Terminal 2 - Frontend  
```bash
cd frontend
npm install
npm run dev
```
✅ Should show: `Local: http://localhost:5173`

---

## Step 4: Test (1 min)

1. Open `http://localhost:5173`
2. Should redirect to login page
3. Click "Sign in with Microsoft"
4. Sign in with your Microsoft account
5. ✅ Should redirect to dashboard with your name

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Microsoft OAuth not configured" | Check `.env` file has correct credentials |
| "Invalid redirect URI" | Register `http://localhost:8000/api/auth/callback/microsoft` in Azure AD |
| "Unauthorized domain" | Change `ALLOWED_EMAIL_DOMAIN` in `.env` to match your email |
| Page stuck on loading | Check browser console (F12) and terminal output |
| "Invalid or expired session" | Log out and log back in |

---

## Check Authentication Works

**Browser Console** (F12):
- Go to DevTools → Application → LocalStorage
- Should see: `session_id`, `user_email`, `user_name`

**Network Tab** (F12):
- Look for `/api/auth/verify/` call
- Should return 200 with user data

**App Header**:
- Should show your name/email
- Click to see logout option

---

## What's Protected Now?

✅ **ALL routes require login**:
- Home `/`
- Upload `/upload`
- Results `/analysis-result`
- Rankings `/engineer-ranking`
- Reports `/reports`, etc.

❌ **Only public route**: `/login`

---

## Environment Variables Reference

### Required (Authentication)
```
MICROSOFT_CLIENT_ID          - From Azure AD app registration
MICROSOFT_CLIENT_SECRET      - From Azure AD app registration
MICROSOFT_TENANT_ID          - Use 'common' or your tenant ID
ALLOWED_EMAIL_DOMAIN         - Email domain allowed to login (e.g., aspect.co.uk)
FRONTEND_URL                 - Where frontend runs (e.g., http://localhost:5173)
BACKEND_URL                  - Where backend runs (e.g., http://localhost:8000)
```

### Existing (Keep your current config)
- `GROQ_API_KEY` - AI model key
- `AZURE_STORAGE_CONNECTION_STRING` - Azure storage
- `AZURE_CONTAINER_NAME` - Blob storage
- `AZURE_SAS_TOKEN` - Storage access
- `SF_USERNAME`, `SF_PASSWORD`, etc. - Salesforce
- All other existing configs

---

## Full Documentation

For detailed setup, troubleshooting, and production deployment:
📖 **See**: `AUTHENTICATION.md`

For overview of all changes made:
📖 **See**: `IMPLEMENTATION_SUMMARY.md`

---

## Important Notes for Production

When deploying to production, update:

1. **Azure AD**: Register production URLs
   ```
   Redirect URI: https://yourdomain.com/api/auth/callback/microsoft
   ```

2. **Environment Variables**: Update URLs
   ```env
   FRONTEND_URL=https://yourdomain.com
   BACKEND_URL=https://api.yourdomain.com
   ```

3. **Use HTTPS**: Enforce in production
4. **Session Storage**: Switch from in-memory to database
5. **CORS**: Update to production domains

---

## API Endpoints

All auth endpoints start with `/api/auth/`:

| Endpoint | What it does |
|----------|--------------|
| `GET /microsoft` | Start login flow |
| `GET /callback/microsoft?code=...` | OAuth callback |
| `GET /verify/{session_id}` | Check if logged in |
| `GET /session/{session_id}` | Get user info |
| `POST /logout/{session_id}` | Log out |
| `GET /health` | Check auth system |

---

## Success Indicators ✅

Your authentication is working when you see:

1. ✅ Redirected to login page on first visit
2. ✅ Microsoft login button works
3. ✅ Sign in redirects you back to app
4. ✅ Your name shows in top-right header
5. ✅ Click name → see logout option
6. ✅ Logout clears session and redirects to login
7. ✅ Refreshing page keeps you logged in (session persistent)

---

## Still Need Help?

1. Check browser console (F12 → Console tab) for errors
2. Check terminal output where backend is running
3. View detailed guide: `AUTHENTICATION.md`
4. Verify `.env` file has no typos or missing values

---

**You're all set! 🎉 Your CV Analysis Portal now has enterprise authentication!**
