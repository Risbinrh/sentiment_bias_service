#!/usr/bin/env python3
"""
Ollama Enterprise News Analysis Service - Production Runner

This script provides a production-ready way to start the service with
proper configuration validation and startup checks.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from app.core.config import settings
from app.core.ollama_client import get_ollama_client


async def validate_configuration():
    """Validate configuration and dependencies before starting"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Configuration:")
    logger.info(f"  - Primary Ollama URL: {settings.ollama_primary_url}")
    logger.info(f"  - Fallback Ollama URL: {settings.ollama_fallback_url}")
    logger.info(f"  - Model: {settings.ollama_model}")
    logger.info(f"  - API Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"  - Rate Limit: {settings.rate_limit_per_hour}/hour")
    logger.info(f"  - Debug Mode: {settings.debug}")
    
    # Test Ollama connectivity
    logger.info("Testing Ollama connectivity...")
    try:
        ollama_client = await get_ollama_client()
        health = await ollama_client.check_health()
        
        if health["primary_url"]:
            logger.info("✓ Primary Ollama URL is accessible")
        elif health["fallback_url"]:
            logger.info("✓ Fallback Ollama URL is accessible")
            logger.warning("⚠ Primary URL not accessible, using fallback")
        else:
            logger.error("✗ Neither Ollama URL is accessible")
            return False
        
        if health["model_available"]:
            logger.info(f"✓ Model '{settings.ollama_model}' is available")
        else:
            logger.error(f"✗ Model '{settings.ollama_model}' is not available")
            logger.info(f"Available models: {health['models']}")
            return False
            
        await ollama_client.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to connect to Ollama: {e}")
        return False


def main():
    """Main entry point"""
    # Setup basic logging for startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate configuration
    try:
        validation_result = asyncio.run(validate_configuration())
        if not validation_result:
            logger.error("Configuration validation failed. Exiting.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Startup cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        sys.exit(1)
    
    logger.info("Configuration validation passed. Starting server...")
    
    # Start the server
    try:
        uvicorn.run(
            "app.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.debug,
            log_level=settings.log_level.lower(),
            access_log=True,
            server_header=False,
            date_header=False
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()