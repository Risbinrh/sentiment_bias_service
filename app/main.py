import logging
import json
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

from app.core.config import settings
from app.api.endpoints import router
from app.core.ollama_client import get_ollama_client


# Configure logging
def setup_logging():
    """Setup application logging"""
    log_level = getattr(logging, settings.log_level.upper())
    
    if settings.log_format == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []
    root_logger.addHandler(handler)
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("fastapi").setLevel(log_level)
    logging.getLogger("app").setLevel(log_level)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    
    # Verify Ollama connectivity
    try:
        ollama_client = await get_ollama_client()
        health = await ollama_client.check_health()
        
        if health["primary_url"] or health["fallback_url"]:
            logger.info("Successfully connected to Ollama service")
            if health["model_available"]:
                logger.info(f"Model {settings.ollama_model} is available")
            else:
                logger.warning(f"Model {settings.ollama_model} is not available")
        else:
            logger.error("Failed to connect to Ollama service")
    except Exception as e:
        logger.error(f"Error during startup health check: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    try:
        ollama_client = await get_ollama_client()
        await ollama_client.close()
    except:
        pass


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Enterprise-grade news analysis service powered by Ollama",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests"""
    start_time = datetime.now()
    
    # Get client info
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "Unknown")
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (datetime.now() - start_time).total_seconds()
    
    # Log request
    logger = logging.getLogger("app.middleware")
    log_data = {
        "method": request.method,
        "url": str(request.url),
        "client_ip": client_ip,
        "user_agent": user_agent,
        "status_code": response.status_code,
        "process_time": f"{process_time:.3f}s"
    }
    
    if response.status_code >= 400:
        logger.warning(f"Request failed: {log_data}")
    else:
        logger.info(f"Request completed: {log_data}")
    
    # Add processing time header
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger = logging.getLogger("app.exceptions")
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger = logging.getLogger("app.exceptions")
    logger.exception(f"Unhandled exception: {request.method} {request.url}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.version,
        description="""
## Enterprise News Analysis API

This API provides comprehensive analysis of news articles using Ollama-powered AI models.

### Features
- **Fast Analysis**: 18x faster than traditional multi-model approaches
- **Comprehensive**: Complete enterprise newsroom metadata schema
- **Reliable**: Built-in fallback mechanisms and error handling
- **Scalable**: Batch processing and rate limiting

### Authentication
All endpoints require an API key in the Authorization header:
```
Authorization: Bearer your-api-key-here
```

### Rate Limiting
- 100 requests per hour per API key
- Rate limit headers included in responses

### Endpoints
- `POST /api/v1/analyze-comprehensive` - Analyze single article
- `POST /api/v1/analyze-comprehensive/batch` - Batch analysis (max 10 URLs)
- `GET /api/v1/analyze-comprehensive/health` - Service health check
        """,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add security to all paths
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method != "get" or "health" not in path:
                openapi_schema["paths"][path][method]["security"] = [
                    {"BearerAuth": []}
                ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Add root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "running",
        "endpoints": {
            "analysis": "/api/v1/analyze-comprehensive",
            "batch": "/api/v1/analyze-comprehensive/batch", 
            "health": "/api/v1/analyze-comprehensive/health"
        },
        "documentation": "/docs",
        "redoc": "/redoc"
    }

# Include API routes
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )