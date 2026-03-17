from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from datetime import datetime, timedelta
import secrets
import requests
from dotenv import load_dotenv
from urllib.parse import urlencode

load_dotenv()

router = APIRouter(prefix="/api/auth", tags=["authentication"])

sessions: Dict[str, Dict[str, Any]] = {}

MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID", "")
MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET", "")
MICROSOFT_TENANT_ID = os.getenv("MICROSOFT_TENANT_ID", "common")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# 🔐 Company email domain restriction
ALLOWED_EMAIL_DOMAIN = os.getenv("ALLOWED_EMAIL_DOMAIN", "aspect.co.uk")


class AuthSession(BaseModel):
    user: Dict[str, str]
    session: str


# ------------------------------
# SESSION MANAGEMENT
# ------------------------------

def create_session(user_data: Dict[str, Any]) -> str:
    session_id = secrets.token_urlsafe(32)

    sessions[session_id] = {
        "user": user_data,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=24)
    }

    print(f"✅ Session created: {session_id}")
    return session_id


def get_session_user(session_id: str) -> Optional[Dict[str, Any]]:
    session = sessions.get(session_id)

    if session:
        if datetime.now() < session["expires_at"]:
            return session["user"]
        else:
            del sessions[session_id]

    return None


def clear_session(session_id: str) -> bool:
    if session_id in sessions:
        del sessions[session_id]
        print(f"✅ Session cleared: {session_id}")
        return True
    return False


# ------------------------------
# MICROSOFT LOGIN
# ------------------------------

@router.get("/microsoft")
async def microsoft_signin():

    if not MICROSOFT_CLIENT_ID or not MICROSOFT_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Microsoft OAuth not configured.")

    redirect_uri = f"{BACKEND_URL}/api/auth/callback/microsoft"

    params = {
        "client_id": MICROSOFT_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": "openid profile email User.Read",
        "prompt": "select_account",
    }

    microsoft_auth_url = f"https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}/oauth2/v2.0/authorize?{urlencode(params)}"

    print("🔐 Redirecting to Microsoft OAuth")

    return RedirectResponse(url=microsoft_auth_url)


# ------------------------------
# CALLBACK
# ------------------------------

@router.get("/callback/microsoft")
async def microsoft_callback(
        code: str = Query(None),
        error: str = Query(None),
        error_description: str = Query(None)
):

    print("\n================ Microsoft OAuth Callback ================")

    if error:
        return RedirectResponse(url=f"{FRONTEND_URL}/login?error=oauth_error")

    if not code:
        return RedirectResponse(url=f"{FRONTEND_URL}/login?error=no_code")

    try:

        token_url = f"https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}/oauth2/v2.0/token"
        redirect_uri = f"{BACKEND_URL}/api/auth/callback/microsoft"

        token_response = requests.post(
            token_url,
            data={
                "client_id": MICROSOFT_CLIENT_ID,
                "client_secret": MICROSOFT_CLIENT_SECRET,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        token_data = token_response.json()

        if "error" in token_data:
            print("❌ Token exchange failed")
            return RedirectResponse(url=f"{FRONTEND_URL}/login?error=token_exchange_failed")

        access_token = token_data.get("access_token")

        if not access_token:
            return RedirectResponse(url=f"{FRONTEND_URL}/login?error=no_token")

        # ------------------------------
        # GET USER INFO
        # ------------------------------

        user_response = requests.get(
            "https://graph.microsoft.com/v1.0/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        if user_response.status_code != 200:
            return RedirectResponse(url=f"{FRONTEND_URL}/login?error=user_info_failed")

        user_data = user_response.json()

        user_email = user_data.get("mail") or user_data.get("userPrincipalName")
        user_name = user_data.get("displayName")
        user_id = user_data.get("id")

        if not user_email:
            return RedirectResponse(url=f"{FRONTEND_URL}/login?error=no_email")

        # ------------------------------
        # DOMAIN CHECK
        # ------------------------------

        email_domain = user_email.split("@")[-1]

        if email_domain.lower() != ALLOWED_EMAIL_DOMAIN.lower():
            print(f"❌ Unauthorized domain: {user_email}")

            return RedirectResponse(
                url=f"{FRONTEND_URL}/?error=unauthorized_domain"
            )

        print(f"✅ Login allowed: {user_email}")

        # ------------------------------
        # CREATE SESSION
        # ------------------------------

        user_info = {
            "name": user_name,
            "email": user_email,
            "id": user_id
        }

        session_id = create_session(user_info)

        redirect_params = {
            "user": user_name,
            "email": user_email,
            "session": session_id
        }

        redirect_url = f"{FRONTEND_URL}/login?{urlencode(redirect_params)}"

        print("✅ Login successful")
        print("=========================================================\n")

        return RedirectResponse(url=redirect_url)

    except Exception as e:
        print("❌ Server error:", str(e))
        return RedirectResponse(url=f"{FRONTEND_URL}/login?error=server_error")


# ------------------------------
# SESSION API
# ------------------------------

@router.get("/session/{session_id}")
async def get_session_endpoint(session_id: str):

    user = get_session_user(session_id)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    return {"user": user, "session": "active"}


@router.post("/logout/{session_id}")
async def logout(session_id: str):

    success = clear_session(session_id)

    return {"success": success}


@router.get("/verify/{session_id}")
async def verify_session(session_id: str):

    user = get_session_user(session_id)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    session = sessions.get(session_id)

    expires_at = session.get("expires_at") if session else None

    return {
        "valid": True,
        "user": user,
        "expires_at": expires_at.isoformat() if expires_at else None
    }


# ------------------------------
# HEALTH CHECK
# ------------------------------

@router.get("/health")
async def health_check():

    return {
        "status": "ok",
        "microsoft_oauth_configured": bool(MICROSOFT_CLIENT_ID and MICROSOFT_CLIENT_SECRET),
        "tenant_id": MICROSOFT_TENANT_ID,
        "allowed_domain": ALLOWED_EMAIL_DOMAIN,
        "active_sessions": len(sessions)
    }