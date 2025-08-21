import httpx
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from app.core.config import settings


logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self):
        self.primary_url = settings.ollama_primary_url
        self.fallback_url = settings.ollama_fallback_url
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout
        self.client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate analysis using Ollama API
        """
        default_options = {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 2048
        }
        
        if options:
            default_options.update(options)
        
        request_payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": default_options
        }
        
        # Only add format if not disabled
        if options and options.get("format") is not None:
            if options.get("format") != "":
                request_payload["format"] = options.get("format")
        else:
            request_payload["format"] = "json"
        
        # Create fresh client for each request
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            # Try primary URL first
            try:
                response = await self._make_request_with_client(client, self.primary_url, request_payload)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Primary Ollama URL failed: {e}")
            
            # Fallback to secondary URL
            try:
                logger.info("Attempting fallback Ollama URL")
                response = await self._make_request_with_client(client, self.fallback_url, request_payload)
                if response:
                    return response
            except Exception as e:
                logger.error(f"Fallback Ollama URL also failed: {e}")
                raise Exception(f"Both Ollama endpoints failed: {e}")
    
    async def _make_request_with_client(self, client: httpx.AsyncClient, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API with provided client
        """
        endpoint = f"{url}/api/generate"
        
        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the generated response
            if "response" in result:
                response_text = result["response"]
                # Try to parse as JSON first (for backward compatibility)
                try:
                    generated_json = json.loads(response_text)
                    return generated_json
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the text response directly
                    return {"response": response_text}
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama: {e}")
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}")
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check if Ollama service is healthy and model is available
        """
        health_status = {
            "primary_url": False,
            "fallback_url": False,
            "model_available": False,
            "models": []
        }
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            # Check primary URL
            try:
                response = await client.get(f"{self.primary_url}/api/tags")
                if response.status_code == 200:
                    health_status["primary_url"] = True
                    models_data = response.json()
                    health_status["models"] = [m["name"] for m in models_data.get("models", [])]
                    health_status["model_available"] = self.model in health_status["models"]
            except Exception as e:
                logger.debug(f"Primary health check failed: {e}")
            
            # Check fallback URL if primary failed
            if not health_status["primary_url"]:
                try:
                    response = await client.get(f"{self.fallback_url}/api/tags")
                    if response.status_code == 200:
                        health_status["fallback_url"] = True
                        models_data = response.json()
                        health_status["models"] = [m["name"] for m in models_data.get("models", [])]
                        health_status["model_available"] = self.model in health_status["models"]
                except Exception as e:
                    logger.debug(f"Fallback health check failed: {e}")
        
        return health_status
    
    async def close(self):
        """
        Close the HTTP client (no-op since we use context managers)
        """
        pass


# Singleton instance
_ollama_client = None


async def get_ollama_client() -> OllamaClient:
    """
    Get or create Ollama client instance
    """
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client