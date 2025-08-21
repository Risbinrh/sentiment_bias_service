import time
from typing import Dict
from fastapi import Request, HTTPException
from collections import defaultdict
from app.core.config import settings


class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
        self.limit_per_hour = settings.rate_limit_per_hour
        self.window = 3600  # 1 hour in seconds
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if client is within rate limit
        """
        now = time.time()
        
        # Clean old requests outside the window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window
        ]
        
        # Check if within limit
        if len(self.requests[client_id]) >= self.limit_per_hour:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """
        Get remaining requests for client
        """
        return max(0, self.limit_per_hour - len(self.requests[client_id]))
    
    def get_reset_time(self, client_id: str) -> int:
        """
        Get timestamp when rate limit resets for client
        """
        if not self.requests[client_id]:
            return int(time.time())
        
        oldest_request = min(self.requests[client_id])
        return int(oldest_request + self.window)


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(request: Request):
    """
    FastAPI dependency to check rate limit
    """
    client_ip = request.client.host
    api_key = request.headers.get("authorization", "").replace("Bearer ", "")
    
    # Use API key if available, otherwise use IP
    client_id = api_key if api_key else client_ip
    
    if not rate_limiter.is_allowed(client_id):
        remaining = rate_limiter.get_remaining(client_id)
        reset_time = rate_limiter.get_reset_time(client_id)
        
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(settings.rate_limit_per_hour),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time)
            }
        )
    
    return True