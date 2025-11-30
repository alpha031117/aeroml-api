"""Security helpers for hashing and verifying user passwords."""

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

password_hasher = PasswordHasher()


def hash_password(raw_password: str) -> str:
    """Hash a password using Argon2."""
    return password_hasher.hash(raw_password)


def verify_password(raw_password: str, hashed_password: str) -> bool:
    """Verify a provided password against a stored hash."""
    try:
        return password_hasher.verify(hashed_password, raw_password)
    except VerifyMismatchError:
        return False
    except Exception:
        return False




