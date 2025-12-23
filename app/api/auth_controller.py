from __future__ import annotations

import secrets
import time
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import create_access_token
from app.db import crud
from app.db.database import get_db
from app.helper.logger import get_logger

auth_router = APIRouter(tags=["auth"])
settings = get_settings()
logger = get_logger(__name__)

# In-memory state storage with TTL (10 minutes)
# Key: state token, Value: timestamp when created
_oauth_states: dict[str, float] = {}
_STATE_TTL = 600  # 10 minutes in seconds


def _cleanup_expired_states():
    """Remove expired state tokens from memory."""
    current_time = time.time()
    expired = [state for state, timestamp in _oauth_states.items() if current_time - timestamp > _STATE_TTL]
    for state in expired:
        _oauth_states.pop(state, None)


def _store_oauth_state(state: str) -> None:
    """Store an OAuth state token with current timestamp."""
    _cleanup_expired_states()
    _oauth_states[state] = time.time()
    logger.debug(f"Stored OAuth state: {state[:20]}... (total states: {len(_oauth_states)})")


def _validate_and_consume_oauth_state(state: str) -> bool:
    """
    Validate and consume an OAuth state token.
    Returns True if valid, False otherwise.
    State is consumed (deleted) after validation to prevent replay attacks.
    """
    _cleanup_expired_states()
    
    if state not in _oauth_states:
        logger.warning(f"OAuth state not found in storage: {state[:20]}...")
        return False
    
    timestamp = _oauth_states[state]
    current_time = time.time()
    
    if current_time - timestamp > _STATE_TTL:
        logger.warning(f"OAuth state expired: {state[:20]}... (age: {current_time - timestamp:.1f}s)")
        _oauth_states.pop(state, None)
        return False
    
    # Consume the state (delete it) to prevent replay attacks
    _oauth_states.pop(state, None)
    logger.debug(f"Validated and consumed OAuth state: {state[:20]}...")
    return True


def _set_session_cookie(response: Response, token: str) -> None:
    """
    Set the JWT session cookie.

    Note: SameSite is set to 'lax' by default – adjust to 'none' and ensure HTTPS
    termination if you need cross-site cookies (e.g., frontend on a different domain).
    """
    response.set_cookie(
        key=settings.jwt_cookie_name,
        value=token,
        httponly=True,
        secure=False,  # Change to True when serving over HTTPS
        samesite="lax",
        max_age=settings.jwt_access_token_expires_minutes * 60,
        path="/",
    )


@auth_router.get("/google/login")
async def google_oauth_login(response: Response) -> RedirectResponse:
    """
    Start the Google OAuth flow.

    Generates a CSRF `state` token, stores it in a short-lived cookie, and
    redirects the user to Google's OAuth consent screen.
    """
    if not settings.google_client_id or not settings.google_redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth is not configured on the server",
        )

    # Log the redirect URI being used for debugging
    logger.info(f"Starting Google OAuth flow with redirect_uri: {settings.google_redirect_uri}")

    state = secrets.token_urlsafe(32)
    
    # Store state in server-side memory (more reliable than cookies for cross-site redirects)
    _store_oauth_state(state)
    logger.debug(f"Generated and stored OAuth state: {state[:20]}... (first 20 chars)")

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "include_granted_scopes": "true",
        "state": state,
        "prompt": "consent",
    }

    google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    url = httpx.URL(google_auth_url, params=params)

    logger.debug(f"Redirecting to Google OAuth with redirect_uri: {settings.google_redirect_uri}")
    return RedirectResponse(url=str(url))


@auth_router.get("/google/callback")
async def google_oauth_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """
    Handle Google's OAuth callback:

    - Validate the CSRF state
    - Exchange authorization code for tokens
    - Fetch user info and upsert into Postgres
    - Issue JWT session token in an HttpOnly cookie
    - Redirect user to the frontend /model-prompt page
    """
    logger.info(f"OAuth callback received - code present: {bool(code)}, state present: {bool(state)}, error: {error}")
    
    # Handle OAuth errors from Google
    if error:
        error_msg = f"Google OAuth error: {error}"
        if error_description:
            error_msg += f" - {error_description}"
        
        # Special handling for redirect_uri_mismatch
        if error == "redirect_uri_mismatch":
            error_msg += (
                f"\n\nConfiguration Issue: The redirect URI configured in your code "
                f"({settings.google_redirect_uri}) does not match what's configured in Google Cloud Console.\n\n"
                f"To fix this:\n"
                f"1. Go to Google Cloud Console → APIs & Services → Credentials\n"
                f"2. Open your OAuth 2.0 Client ID\n"
                f"3. Add this EXACT redirect URI to 'Authorized redirect URIs':\n"
                f"   {settings.google_redirect_uri}\n"
                f"4. Make sure it matches EXACTLY (including http/https, port, and path)\n"
                f"5. Save and wait a few minutes for changes to propagate"
            )
            logger.error(error_msg)
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )

    if not state:
        logger.error("OAuth state parameter is missing from callback URL")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth state parameter is missing from callback URL.",
        )
    
    # Validate state from server-side storage (more reliable than cookies)
    if not _validate_and_consume_oauth_state(state):
        logger.error(f"Invalid or expired OAuth state: {state[:20]}...")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state. Please try logging in again.",
        )
    
    logger.debug(f"OAuth state validated successfully: {state[:20]}...")

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code",
        )

    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth is not fully configured on the server",
        )

    # Exchange authorization code for tokens
    token_endpoint = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": settings.google_redirect_uri,
        "grant_type": "authorization_code",
    }

    logger.debug(f"Exchanging code for token with redirect_uri: {settings.google_redirect_uri}")
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(token_endpoint, data=data)

    if token_resp.status_code != 200:
        error_text = token_resp.text
        logger.error(f"Token exchange failed: {error_text}")
        
        # Check if it's a redirect_uri_mismatch error
        if "redirect_uri_mismatch" in error_text.lower():
            error_msg = (
                f"Failed to exchange code for token: {error_text}\n\n"
                f"Configuration Issue: The redirect URI ({settings.google_redirect_uri}) "
                f"does not match what's configured in Google Cloud Console.\n\n"
                f"To fix:\n"
                f"1. Go to Google Cloud Console → APIs & Services → Credentials\n"
                f"2. Open your OAuth 2.0 Client ID\n"
                f"3. Add this EXACT redirect URI: {settings.google_redirect_uri}\n"
                f"4. Ensure it matches EXACTLY (protocol, host, port, path)\n"
                f"5. Save and wait a few minutes"
            )
        else:
            error_msg = f"Failed to exchange code for token: {error_text}"
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )

    token_data = token_resp.json()
    access_token = token_data.get("access_token")

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Access token missing in Google response",
        )

    # Fetch user info from Google
    async with httpx.AsyncClient() as client:
        userinfo_resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if userinfo_resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch user info from Google: {userinfo_resp.text}",
        )

    userinfo = userinfo_resp.json()
    email = userinfo.get("email")
    full_name = userinfo.get("name")

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google user info missing email",
        )

    # Upsert user
    from app.core.security import hash_password  # local import to avoid cycles
    import uuid

    user = crud.get_user_by_email(db, email)
    if not user:
        random_secret = hash_password(uuid.uuid4().hex)
        user = crud.create_user(db, email=email, hashed_password=random_secret, full_name=full_name)

    # Issue JWT session
    jwt_token = create_access_token(subject=str(user.id), additional_claims={"email": user.email})

    # Redirect to frontend /model-prompt page
    frontend_redirect_url = f"{settings.frontend_url.rstrip('/')}/model-prompt"
    logger.info(f"OAuth login successful for {email}, redirecting to: {frontend_redirect_url}")
    
    redirect_response = RedirectResponse(url=frontend_redirect_url, status_code=status.HTTP_302_FOUND)
    _set_session_cookie(redirect_response, jwt_token)

    # State was already consumed during validation, no need to clear cookie

    return redirect_response


@auth_router.get("/config")
async def get_oauth_config() -> JSONResponse:
    """
    Debug endpoint to show the configured OAuth settings (without secrets).
    
    Useful for troubleshooting redirect_uri_mismatch errors.
    """
    return JSONResponse(
        content={
            "google_client_id": settings.google_client_id[:10] + "..." if settings.google_client_id else None,
            "google_redirect_uri": settings.google_redirect_uri,
            "frontend_url": settings.frontend_url,
            "frontend_redirect_url": f"{settings.frontend_url.rstrip('/')}/model-prompt",
            "configured": bool(settings.google_client_id and settings.google_client_secret and settings.google_redirect_uri),
            "instructions": (
                "To fix redirect_uri_mismatch:\n"
                "1. Copy the google_redirect_uri value above\n"
                "2. Go to Google Cloud Console → APIs & Services → Credentials\n"
                "3. Open your OAuth 2.0 Client ID\n"
                "4. Add the redirect_uri EXACTLY as shown above to 'Authorized redirect URIs'\n"
                "5. Save and wait a few minutes for changes to propagate"
            ),
        }
    )


