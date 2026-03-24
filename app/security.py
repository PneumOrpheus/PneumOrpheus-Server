from fastapi import Header, HTTPException, status

from app.config import get_settings


async def verify_api_key(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.inference_api_key:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.inference_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")
