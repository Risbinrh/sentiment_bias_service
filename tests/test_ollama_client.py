import pytest
from unittest.mock import patch, MagicMock
import httpx
from app.core.ollama_client import OllamaClient


class TestOllamaClient:
    @pytest.fixture
    def client(self):
        """Create Ollama client"""
        return OllamaClient()
    
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_generate_success(self, mock_client_class):
        """Test successful generation"""
        # Mock httpx client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '{"test": "data"}'
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        client = OllamaClient()
        client.client = mock_client
        
        result = await client.generate("test prompt")
        
        assert result == {"test": "data"}
        mock_client.post.assert_called_once()
    
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_generate_fallback(self, mock_client_class):
        """Test fallback to secondary URL"""
        mock_client = MagicMock()
        
        # First call fails, second succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=MagicMock()
        )
        
        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {
            "response": '{"fallback": "success"}'
        }
        mock_response_success.raise_for_status.return_value = None
        
        mock_client.post.side_effect = [mock_response_fail, mock_response_success]
        mock_client_class.return_value = mock_client
        
        client = OllamaClient()
        client.client = mock_client
        
        result = await client.generate("test prompt")
        
        assert result == {"fallback": "success"}
        assert mock_client.post.call_count == 2
    
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_generate_both_fail(self, mock_client_class):
        """Test when both URLs fail"""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=MagicMock()
        )
        mock_client_class.return_value = mock_client
        
        client = OllamaClient()
        client.client = mock_client
        
        with pytest.raises(Exception, match="Both Ollama endpoints failed"):
            await client.generate("test prompt")
    
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_check_health_success(self, mock_client_class):
        """Test successful health check"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:1b"},
                {"name": "other:model"}
            ]
        }
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        client = OllamaClient()
        client.client = mock_client
        
        result = await client.check_health()
        
        assert result["primary_url"] == True
        assert result["model_available"] == True
        assert "llama3.2:1b" in result["models"]
    
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_check_health_model_unavailable(self, mock_client_class):
        """Test health check when model is not available"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "other:model"}
            ]
        }
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        client = OllamaClient()
        client.client = mock_client
        
        result = await client.check_health()
        
        assert result["primary_url"] == True
        assert result["model_available"] == False
        assert "llama3.2:1b" not in result["models"]
    
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio  
    async def test_check_health_connection_failed(self, mock_client_class):
        """Test health check when connection fails"""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection failed")
        mock_client_class.return_value = mock_client
        
        client = OllamaClient()
        client.client = mock_client
        
        result = await client.check_health()
        
        assert result["primary_url"] == False
        assert result["fallback_url"] == False
        assert result["model_available"] == False