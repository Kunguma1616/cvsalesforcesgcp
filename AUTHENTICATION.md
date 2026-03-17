# Microsoft OAuth Authentication Setup Guide

This guide explains how to set up Microsoft OAuth authentication for the CV Analysis Portal.

## Overview

The application uses Microsoft Azure AD (Entra ID) for single sign-on (SSO) authentication. All users must have a valid company email (default: `@aspect.co.uk`) to access the platform.

## Architecture

1. **Frontend**: React-based login page that redirects to Microsoft
2. **Backend**: FastAPI with OAuth2 callbacks and session management
3. **Session Management**: In-memory session storage with 24-hour expiration

## Prerequisites

1. Microsoft Azure AD tenant access
2. Backend and frontend running on localhost
3. Python 3.8+ (backend)
4. Node.js 16+ (frontend)

## Step 1: Register Application in Azure AD

### 1.1 Go to Azure Portal
- Navigate to https://portal.azure.com
- Search for "App registrations" and open it

### 1.2 Create New Application
- Click "New registration"
- Enter application name: `CV Analysis Portal`
- Select account type: `Accounts in any organizational directory (Any Azure AD directory - Multitenant)`
- Click "Register"

### 1.3 Get Client ID and Secret
After registration, you'll see the application overview:

1. **Copy Client ID** (Application ID) - you'll need this for `MICROSOFT_CLIENT_ID`
2. Go to "Certificates & secrets"
3. Click "New client secret"
4. Set expiration to "24 months"
5. Click "Add"
6. **Copy the secret value** - you'll need this for `MICROSOFT_CLIENT_SECRET`

### 1.4 Configure Redirect URI
1. Go to "Authentication" tab
2. Under "Redirect URIs", click "Add URI"
3. Add: `http://localhost:8000/api/auth/callback/microsoft`
4. Save changes

### 1.5 Configure API Permissions
1. Go to "API permissions"
2. Click "Add a permission"
3. Select "Microsoft Graph"
4. Select "Delegated permissions"
5. Search and add these permissions:
   - `openid`
   - `profile`
   - `email`
   - `User.Read`
6. Click "Add permissions"
7. Click "Grant admin consent" (if available)

## Step 2: Environment Configuration

### 2.1 Backend Setup

Create or update `.env` file in the `backend/` directory:

```dotenv
# ========= MICROSOFT OAUTH ==========
MICROSOFT_CLIENT_ID=your_client_id_here
MICROSOFT_CLIENT_SECRET=your_client_secret_here
MICROSOFT_TENANT_ID=common

# ========= URLs ==========
FRONTEND_URL=http://localhost:5173
BACKEND_URL=http://localhost:8000

# ========= DOMAIN RESTRICTION ==========
ALLOWED_EMAIL_DOMAIN=aspect.co.uk

# ... other configuration variables ...
```

**Important Notes:**
- `MICROSOFT_TENANT_ID`: Use `common` for multi-tenant or your specific tenant ID (found in Azure AD > Overview)
- `ALLOWED_EMAIL_DOMAIN`: Change this to match your company's email domain
- `FRONTEND_URL`: Must match where your frontend is running
- `BACKEND_URL`: Must match where your backend is running

### 2.2 Frontend Setup

Create `.env` file in the `frontend/` directory:

```dotenv
VITE_API_URL=http://localhost:8000
VITE_APP_NAME=CV Analysis Portal
```

## Step 3: Install Dependencies

### Backend
```bash
cd backend
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## Step 4: Run the Application

### Terminal 1 - Backend
```bash
cd backend
python main.py
```

Backend will start at: `http://localhost:8000`

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

Frontend will start at: `http://localhost:5173`

## Step 5: Test the Authentication Flow

1. Navigate to `http://localhost:5173`
2. You should be redirected to the login page
3. Click "Sign in with Microsoft"
4. You'll be redirected to Microsoft login
5. Sign in with your company email
6. After successful login, you'll be redirected back to the dashboard

## Authentication Flow Diagram

```
User Browser          Frontend              Backend            Microsoft
    |                    |                    |                   |
    |                    |                    |                   |
    +--visit app-------->|                    |                   |
    |                    |                    |                   |
    |                    +--redirect to login-|                   |
    |<---show login------|                    |                   |
    |                    |                    |                   |
    +--click sign in---->|                    |                   |
    |                    +--/api/auth/microsoft                   |
    |                    |-------OAuth redirect URL-->|           |
    |                    |                    |--redirect to MS-->|
    |<---redirect to MS login -----------------------|           |
    |                    |                    |                   |
    +--user enters creds and grants consent----------->|
    |                    |                    |                   |
    |                    |<---auth code returned -------|
    |                    |                    |                   |
    |                    +--/api/auth/callback/microsoft
    |                    |<---exchange code for token--|
    |                    |<---get user info-------------|
    |                    |---validate domain & create session---+
    |                    |<---create session at session_id -----+
    |                    |                    |                   |
    |<---set session cookie & redirect to dashboard--|
    |                    |                    |                   |
```

## API Endpoints

### Authentication Endpoints

#### 1. Get Microsoft OAuth URL
```
GET /api/auth/microsoft
```
Initiates the Microsoft OAuth flow by redirecting to Microsoft login page.

#### 2. OAuth Callback
```
GET /api/auth/callback/microsoft?code=...&state=...
```
Handles the callback from Microsoft after user authentication.

**Response on success:**
```
Redirect to: http://localhost:5173/?user=Name&email=user@example.com&session=session_id
```

#### 3. Get Session
```
GET /api/auth/session/{session_id}
```
Retrieves user information for a given session.

**Response:**
```json
{
    "user": {
        "name": "John Doe",
        "email": "john@aspect.co.uk",
        "id": "microsoft_user_id"
    },
    "session": "active"
}
```

#### 4. Verify Session
```
GET /api/auth/verify/{session_id}
```
Verifies if a session is still valid and returns expiration time.

**Response:**
```json
{
    "valid": true,
    "user": {
        "name": "John Doe",
        "email": "john@aspect.co.uk",
        "id": "microsoft_user_id"
    },
    "expires_at": "2024-03-20T10:30:00"
}
```

#### 5. Logout
```
POST /api/auth/logout/{session_id}
```
Clears the session data.

**Response:**
```json
{
    "success": true
}
```

#### 6. Health Check
```
GET /api/auth/health
```
Returns authentication system status.

**Response:**
```json
{
    "status": "ok",
    "microsoft_oauth_configured": true,
    "tenant_id": "common",
    "allowed_domain": "aspect.co.uk",
    "active_sessions": 5
}
```

## Frontend Components

### LoginPage Component
Located at: `frontend/src/pages/LoginPage.tsx`

Displays the login page with Microsoft sign-in button. Handles OAuth callback parameters from the backend.

### AuthContext Hook
Located at: `frontend/src/hooks/useAuth.tsx`

Provides authentication state across the application:
- `isAuthenticated`: Current authentication status
- `user`: Current user object (name, email)
- `sessionId`: Current session token
- `logout()`: Function to log out the user
- `isLoading`: Loading state

### ProtectedRoute Component
Located at: `frontend/src/components/ProtectedRoute.tsx`

Wraps routes that require authentication. Redirects unauthenticated users to login page.

### MainLayout Updates
The header now displays:
- User name/email
- Logout button in a dropdown menu

## Troubleshooting

### "Microsoft OAuth not configured" Error
**Solution:** Ensure `MICROSOFT_CLIENT_ID` and `MICROSOFT_CLIENT_SECRET` are set in `.env` file.

### "Unauthorized domain" Error
**Solution:** Check that user's email domain matches `ALLOWED_EMAIL_DOMAIN`. Change the setting if needed.

### "Invalid or expired session" Error
**Causes:**
- Session has expired (24 hours)
- Server was restarted (sessions are in-memory)

**Solution:** Log in again.

### Redirect URI Mismatch Error
**Solution:** Ensure `BACKEND_URL` in `.env` matches the registered redirect URI in Azure AD:
- Must be: `http://localhost:8000/api/auth/callback/microsoft`

### "No access token received" Error
**Causes:**
- Microsoft OAuth flow was interrupted
- Invalid client credentials

**Solution:** Check Azure AD credentials and permissions.

## Security Considerations

1. **Session Storage**: Currently uses in-memory storage. For production, use:
   - Redis
   - Database (PostgreSQL, MongoDB)
   - Encrypted JWT tokens

2. **HTTPS**: Use HTTPS in production
   ```python
   # Add to main.py for production
   app.add_middleware(HTTPSRedirectMiddleware)
   ```

3. **CORS**: Update allowed origins for production domains

4. **Email Domain**: Configure `ALLOWED_EMAIL_DOMAIN` for your organization

5. **Session Expiration**: Currently 24 hours. Adjust in `auth.py`:
   ```python
   timedelta(hours=24)  # Change this value
   ```

## Production Deployment

### For Azure App Service:

1. Update environment variables in Azure Portal
2. Register production redirect URI:
   ```
   https://your-domain.com/api/auth/callback/microsoft
   ```
3. Enable HTTPS redirect
4. Switch session storage to database

### For Docker:

```dockerfile
# docker-compose.yml
services:
  api:
    environment:
      - MICROSOFT_CLIENT_ID=${MICROSOFT_CLIENT_ID}
      - MICROSOFT_CLIENT_SECRET=${MICROSOFT_CLIENT_SECRET}
      - FRONTEND_URL=https://your-domain.com
      - BACKEND_URL=https://api.your-domain.com
```

## Support

For issues or questions, contact the development team or check the application logs:
- Backend logs: Check terminal running `python main.py`
- Frontend logs: Check browser console (F12)

## References

- [Microsoft Identity Platform](https://docs.microsoft.com/en-us/azure/active-directory/develop/)
- [OAuth 2.0 Authorization Code Flow](https://docs.microsoft.com/en-us/azure/active-directory/develop/v2-oauth2-auth-code-flow)
- [Microsoft Graph API](https://docs.microsoft.com/en-us/graph/overview)
