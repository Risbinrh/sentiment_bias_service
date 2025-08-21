import pytest
from unittest.mock import patch, MagicMock
import httpx
from app.services.scraper import WebScraper


class TestWebScraper:
    @pytest.fixture
    def scraper(self):
        """Create web scraper"""
        return WebScraper()
    
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_extract_article_success(self, mock_client_class):
        """Test successful article extraction"""
        # Mock HTML content
        html_content = """
        <html>
            <head>
                <title>Test Article</title>
                <meta property="og:site_name" content="Test Publisher">
                <meta name="author" content="Test Author">
                <meta property="article:published_time" content="2025-08-21T15:30:00Z">
            </head>
            <body>
                <article>
                    <h1>Test Article Title</h1>
                    <p>This is the first paragraph of the article.</p>
                    <p>This is the second paragraph with more content.</p>
                </article>
            </body>
        </html>
        """
        
        # Mock httpx response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        # Mock newspaper Article
        with patch("app.services.scraper.Article") as mock_article_class:
            mock_article = MagicMock()
            mock_article.text = "This is the first paragraph of the article. This is the second paragraph with more content."
            mock_article.title = "Test Article Title"
            mock_article.authors = ["Test Author"]
            mock_article.publish_date = None
            mock_article.meta_lang = "en"
            mock_article.top_image = "https://example.com/image.jpg"
            mock_article.images = {"https://example.com/image1.jpg", "https://example.com/image2.jpg"}
            mock_article.keywords = ["test", "article", "content"]
            mock_article_class.return_value = mock_article
            
            scraper = WebScraper()
            scraper.client = mock_client
            
            result = await scraper.extract_article("https://example.com/article")
            
            assert result["url"] == "https://example.com/article"
            assert result["title"] == "Test Article Title"
            assert result["publisher"] == "Test Publisher"
            assert result["author"] == "Test Author"
            assert result["language"] == "en"
            assert result["word_count"] > 0
            assert "content_hash" in result
    
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_extract_article_http_error(self, mock_client_class):
        """Test article extraction with HTTP error"""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )
        mock_client_class.return_value = mock_client
        
        scraper = WebScraper()
        scraper.client = mock_client
        
        with pytest.raises(Exception, match="Failed to fetch URL"):
            await scraper.extract_article("https://example.com/nonexistent")
    
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_fallback_extraction(self, mock_client_class):
        """Test fallback extraction when newspaper fails"""
        html_content = """
        <html>
            <head><title>Fallback Test</title></head>
            <body>
                <h1>Fallback Article</h1>
                <p>This is fallback content extraction.</p>
            </body>
        </html>
        """
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        scraper = WebScraper()
        scraper.client = mock_client
        
        result = await scraper._fallback_extraction("https://example.com/fallback")
        
        assert result["url"] == "https://example.com/fallback"
        assert result["title"] == "Fallback Article"
        assert "fallback content extraction" in result["text"].lower()
        assert result["publisher"] == "Example"  # Extracted from URL
    
    def test_extract_publisher_from_url(self, scraper):
        """Test publisher extraction from URL"""
        test_cases = [
            ("https://www.cnn.com/article", "Cnn"),
            ("https://techcrunch.com/news", "Techcrunch"),
            ("https://www.nytimes.com/section/article", "Nytimes"),
            ("https://reuters.com/business", "Reuters")
        ]
        
        for url, expected in test_cases:
            result = scraper._extract_publisher_from_url(url)
            assert result == expected
    
    def test_generate_hash(self, scraper):
        """Test content hash generation"""
        text1 = "This is test content"
        text2 = "This is different content"
        text3 = "This is test content"  # Same as text1
        
        hash1 = scraper._generate_hash(text1)
        hash2 = scraper._generate_hash(text2) 
        hash3 = scraper._generate_hash(text3)
        
        assert hash1 == hash3  # Same content should produce same hash
        assert hash1 != hash2  # Different content should produce different hash
        assert len(hash1) == 16  # Hash should be 16 characters
    
    @pytest.mark.asyncio
    async def test_scraper_context_manager(self):
        """Test scraper as async context manager"""
        async with WebScraper() as scraper:
            assert isinstance(scraper, WebScraper)
            assert scraper.client is not None