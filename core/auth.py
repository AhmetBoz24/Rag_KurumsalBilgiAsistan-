"""Admin auth: bcrypt sifre + JWT token.

.env'den okunan:
  ADMIN_USERNAME
  ADMIN_PASSWORD_HASH (bcrypt)
  JWT_SECRET
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import (
    ADMIN_USERNAME,
    ADMIN_PASSWORD_HASH,
    JWT_SECRET,
    JWT_ALGORITHM,
    JWT_EXPIRE_DAYS,
)


pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_ctx.verify(plain, hashed)
    except Exception:
        return False


def authenticate(username: str, password: str) -> bool:
    if not ADMIN_PASSWORD_HASH or not JWT_SECRET:
        return False
    if username != ADMIN_USERNAME:
        return False
    return verify_password(password, ADMIN_PASSWORD_HASH)


def create_token(username: str) -> str:
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET .env'de ayarli degil")
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
        "role": "admin",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    if not JWT_SECRET:
        return None
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None


async def require_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> str:
    """FastAPI dependency: admin JWT'si dogruysa username dondur, yoksa 401."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yetkisiz",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yetkisiz",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload.get("sub", "admin")
