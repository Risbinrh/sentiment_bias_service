import pytest
import httpx
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings


class TestAPI:
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Authentication headers"""
        return {"Authorization": f"Bearer {settings.api_key}"}
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == settings.app_name
        assert data["version"] == settings.version
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        with patch("app.core.ollama_client.get_ollama_client") as mock_client:
            # Mock Ollama client
            mock_ollama = MagicMock()
            mock_ollama.check_health.return_value = {
                "primary_url": True,
                "fallback_url": False,
                "model_available": True,
                "models": ["llama3.2:1b"]
            }
            mock_client.return_value = mock_ollama
            
            response = client.get("/api/v1/analyze-comprehensive/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["ollama_connected"] == True
            assert data["model_available"] == True
    
    def test_analyze_endpoint_without_auth(self, client):
        """Test analyze endpoint without authentication"""
        response = client.post("/api/v1/analyze-comprehensive", json={
            "url": "https://example.com/article"
        })
        assert response.status_code == 403  # Forbidden without auth
    
    def test_analyze_endpoint_invalid_auth(self, client):
        """Test analyze endpoint with invalid authentication"""
        headers = {"Authorization": "Bearer invalid-key"}
        response = client.post("/api/v1/analyze-comprehensive", json={
            "url": "https://example.com/article"
        }, headers=headers)
        assert response.status_code == 401  # Unauthorized
    
    def test_analyze_endpoint_missing_data(self, client, auth_headers):
        """Test analyze endpoint with missing required data"""
        response = client.post("/api/v1/analyze-comprehensive", 
                             json={}, headers=auth_headers)
        assert response.status_code == 400
        assert "Either 'url' or 'text' must be provided" in response.json()["detail"]
    
    def test_analyze_endpoint_text_without_title(self, client, auth_headers):
        """Test analyze endpoint with text but no title"""
        response = client.post("/api/v1/analyze-comprehensive", json={
            "text": "Some article content"
        }, headers=auth_headers)
        assert response.status_code == 400
        assert "'title' is required when providing 'text'" in response.json()["detail"]
    
    @patch("app.services.analyzer.get_analyzer")
    def test_analyze_endpoint_success(self, mock_analyzer, client, auth_headers):
        """Test successful analysis"""
        # Mock analyzer
        mock_analyzer_instance = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.provenance.processing_time_ms = 2500
        mock_analyzer_instance.analyze_comprehensive.return_value = mock_metadata
        mock_analyzer.return_value = mock_analyzer_instance
        
        response = client.post("/api/v1/analyze-comprehensive", json={
            "url": "https://example.com/article"
        }, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "processing_time_ms" in data
        assert "metadata" in data
    
    @patch("app.services.analyzer.get_analyzer")  
    def test_batch_endpoint_success(self, mock_analyzer, client, auth_headers):
        """Test successful batch analysis"""
        # Mock analyzer
        mock_analyzer_instance = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.provenance.processing_time_ms = 2500
        mock_analyzer_instance.analyze_comprehensive.return_value = mock_metadata
        mock_analyzer.return_value = mock_analyzer_instance
        
        response = client.post("/api/v1/analyze-comprehensive/batch", json={
            "urls": [
                "https://example.com/article1",
                "https://example.com/article2"
            ]
        }, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert "summary_statistics" in data
    
    def test_batch_endpoint_too_many_urls(self, client, auth_headers):
        """Test batch endpoint with too many URLs"""
        urls = [f"https://example.com/article{i}" for i in range(11)]
        
        response = client.post("/api/v1/analyze-comprehensive/batch", json={
            "urls": urls
        }, headers=auth_headers)
        
        assert response.status_code == 400
        assert "Maximum 10 URLs allowed" in response.json()["detail"]
    
    @patch("app.api.rate_limit.rate_limiter")
    def test_rate_limiting(self, mock_rate_limiter, client, auth_headers):
        """Test rate limiting"""
        mock_rate_limiter.is_allowed.return_value = False
        mock_rate_limiter.get_remaining.return_value = 0
        mock_rate_limiter.get_reset_time.return_value = 1234567890
        
        response = client.post("/api/v1/analyze-comprehensive", json={
            "url": "https://example.com/article"
        }, headers=auth_headers)
        
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert "X-RateLimit-Limit" in response.headers