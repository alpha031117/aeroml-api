"""CRUD helpers for users and training sessions."""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models
from app.utils.json_utils import clean_json_for_storage


def get_user(db: Session, user_id: uuid.UUID) -> Optional[models.User]:
    return db.get(models.User, user_id)


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    stmt = select(models.User).where(models.User.email == email)
    return db.execute(stmt).scalar_one_or_none()


def create_user(db: Session, *, email: str, hashed_password: str, full_name: Optional[str] = None) -> models.User:
    user = models.User(email=email, hashed_password=hashed_password, full_name=full_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_training_session(
    db: Session,
    *,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    status: str,
    metadata: Optional[dict] = None,
) -> models.TrainingSession:
    training_session = models.TrainingSession(
        session_id=session_id,
        user_id=user_id,
        status=status,
        metadata_json=metadata,
    )
    db.add(training_session)
    db.commit()
    db.refresh(training_session)
    return training_session


def get_training_session(db: Session, session_id: uuid.UUID) -> Optional[models.TrainingSession]:
    stmt = select(models.TrainingSession).where(models.TrainingSession.session_id == session_id)
    return db.execute(stmt).scalar_one_or_none()


def update_training_session(
    db: Session,
    *,
    session_id: uuid.UUID,
    **updates,
) -> Optional[models.TrainingSession]:
    # Allow callers to continue using "metadata" keyword while storing in the renamed column.
    if "metadata" in updates and "metadata_json" not in updates:
        updates["metadata_json"] = updates.pop("metadata")

    training_session = get_training_session(db, session_id)
    if not training_session:
        return None

    # Clean JSON fields before setting them to avoid PostgreSQL NaN errors
    cleaned_updates = {}
    for key, value in updates.items():
        if hasattr(training_session, key) and value is not None:
            # Clean JSON fields to remove NaN/Infinity values that PostgreSQL doesn't accept
            if key in ("performance", "metadata_json") and isinstance(value, (dict, list)):
                cleaned_updates[key] = clean_json_for_storage(value)
            else:
                cleaned_updates[key] = value

    for key, value in cleaned_updates.items():
        setattr(training_session, key, value)

    try:
        db.add(training_session)
        db.commit()
        db.refresh(training_session)
        return training_session
    except Exception as e:
        db.rollback()
        raise


def get_sessions_for_user(db: Session, user_id: uuid.UUID) -> List[models.TrainingSession]:
    stmt = (
        select(models.TrainingSession)
        .where(models.TrainingSession.user_id == user_id)
        .order_by(models.TrainingSession.created_at.desc())
    )
    return db.execute(stmt).scalars().all()


def delete_training_session(db: Session, session_id: uuid.UUID) -> bool:
    training_session = get_training_session(db, session_id)
    if not training_session:
        return False

    db.delete(training_session)
    db.commit()
    return True


