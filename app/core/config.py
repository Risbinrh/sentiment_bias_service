import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama Configuration
    ollama_primary_url: str = os.getenv("OLLAMA_PRIMARY_URL", "http://20.64.243.4:11434")
    ollama_fallback_url: str = os.getenv("OLLAMA_FALLBACK_URL", "http://172.17.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))
    
    # API Configuration
    api_key: str = os.getenv("API_KEY", "prod-key-2025")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "3456"))
    
    # Rate Limiting
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))
    
    # Redis Cache
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "900"))
    
    # Monitoring
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "9090"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    
    # Application
    app_name: str = "Ollama Enterprise News Analysis Service"
    version: str = "2.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()