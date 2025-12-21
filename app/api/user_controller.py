from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from google.auth.transport import requests as google_requests  # type: ignore
from google.oauth2 import id_token  # type: ignore
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.helper.logger import get_logger

user_router = APIRouter(tags=["users"])
settings = get_settings()
logger = get_logger(__name__)


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8)
    full_name: Optional[str] = None
    is_active: Optional[bool] = None


class GoogleLogin(BaseModel):
    id_token: str = Field(..., description="Google ID token obtained on the client")


class UserRead(BaseModel):
    id: uuid.UUID
    email: EmailStr
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


def _set_session_cookie(response: Response, token: str) -> None:
    """Helper to set the JWT session cookie."""
    response.set_cookie(
        key=settings.jwt_cookie_name,
        value=token,
        httponly=True,
        secure=False,  # Switch to True when using HTTPS
        samesite="lax",
        max_age=settings.jwt_access_token_expires_minutes * 60,
        path="/",
    )


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """Resolve the current user from the JWT cookie."""
    token = request.cookies.get(settings.jwt_cookie_name)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    try:
        user_id = uuid.UUID(str(payload["sub"]))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        )

    user = crud.get_user(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return user


@user_router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserCreate, response: Response, db: Session = Depends(get_db)):
    existing = crud.get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    hashed = hash_password(payload.password)
    user = crud.create_user(db, email=payload.email, hashed_password=hashed, full_name=payload.full_name)

    # Issue a session on registration for convenience.
    jwt_token = create_access_token(subject=str(user.id), additional_claims={"email": user.email})
    _set_session_cookie(response, jwt_token)

    return user


@user_router.post("/login", response_model=UserRead)
def login_user(payload: UserLogin, response: Response, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    jwt_token = create_access_token(subject=str(user.id), additional_claims={"email": user.email})
    _set_session_cookie(response, jwt_token)

    return user


@user_router.post("/login/google", response_model=UserRead)
def login_with_google(payload: GoogleLogin, response: Response, db: Session = Depends(get_db)):
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GOOGLE_CLIENT_ID missing on server",
        )

    try:
        id_info = id_token.verify_oauth2_token(
            payload.id_token,
            google_requests.Request(),
            settings.google_client_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid Google token: {exc}") from exc

    email = id_info.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google token missing email claim")

    user = crud.get_user_by_email(db, email)
    if not user:
        random_secret = hash_password(uuid.uuid4().hex)
        full_name = id_info.get("name")
        user = crud.create_user(db, email=email, hashed_password=random_secret, full_name=full_name)

    jwt_token = create_access_token(subject=str(user.id), additional_claims={"email": user.email})
    _set_session_cookie(response, jwt_token)

    return user


@user_router.post("/logout")
def logout_user(response: Response) -> None:
    """Clear the session cookie."""
    response.delete_cookie(settings.jwt_cookie_name, path="/")


@user_router.get("/me", response_model=UserRead)
def read_current_user(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Return the currently authenticated user based on the JWT cookie.
    
    This endpoint is typically called by the frontend to check authentication status
    and retrieve user profile information.
    """
    # Log the request for debugging (you can remove this in production if too verbose)
    logger.debug(f"GET /me called for user: {current_user.email} (ID: {current_user.id})")
    
    return current_user


@user_router.get("/{user_id}", response_model=UserRead)
def get_user(user_id: uuid.UUID, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@user_router.put("/{user_id}", response_model=UserRead)
def update_user(user_id: uuid.UUID, payload: UserUpdate, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if payload.email and payload.email != user.email:
        existing = crud.get_user_by_email(db, payload.email)
        if existing and existing.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

    hashed_pw = hash_password(payload.password) if payload.password else None
    updated = crud.update_user(
        db,
        user_id=user_id,
        email=payload.email,
        hashed_password=hashed_pw,
        full_name=payload.full_name,
        is_active=payload.is_active,
    )

    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return updated

