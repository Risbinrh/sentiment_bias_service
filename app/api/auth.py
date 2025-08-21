from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings


security = HTTPBearer()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verify API key from Authorization header
    """
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return credentials.credentials


async def get_api_key() -> str:
    """
    Dependency to get valid API key
    """
    return Depends(verify_api_key)