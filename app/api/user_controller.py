from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from google.auth.transport import requests as google_requests  # type: ignore
from google.oauth2 import id_token  # type: ignore
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_password, verify_password
from app.db import crud
from app.db.database import get_db

user_router = APIRouter(tags=["users"])
settings = get_settings()


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


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


@user_router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    existing = crud.get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    hashed = hash_password(payload.password)
    user = crud.create_user(db, email=payload.email, hashed_password=hashed, full_name=payload.full_name)
    return user


@user_router.post("/login", response_model=UserRead)
def login_user(payload: UserLogin, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return user


@user_router.post("/login/google", response_model=UserRead)
def login_with_google(payload: GoogleLogin, db: Session = Depends(get_db)):
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

    return user


@user_router.get("/{user_id}", response_model=UserRead)
def get_user(user_id: uuid.UUID, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

