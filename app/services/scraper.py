import httpx
from bs4 import BeautifulSoup
from newspaper import Article
from datetime import datetime
from dateutil import parser
from typing import Dict, Any, Optional
import hashlib
import logging
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class WebScraper:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def extract_article(self, url: str) -> Dict[str, Any]:
        """
        Extract article content and metadata from URL
        """
        try:
            # Fetch the webpage
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Parse with BeautifulSoup for better extraction
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Use newspaper3k for article extraction
            article = Article(url)
            article.download(input_html=response.text)
            article.parse()
            
            # Extract metadata
            metadata = {
                "url": url,
                "title": self._extract_title(article, soup),
                "text": article.text or self._extract_text(soup),
                "publisher": self._extract_publisher(article, soup, url),
                "author": self._extract_author(article, soup),
                "published_date": self._extract_date(article, soup),
                "language": article.meta_lang or "en",
                "word_count": len(article.text.split()) if article.text else 0,
                "content_hash": self._generate_hash(article.text or ""),
                "top_image": article.top_image,
                "images": list(article.images)[:10],  # Limit to 10 images
                "keywords": article.keywords[:20],  # Limit to 20 keywords
                "summary": article.summary if hasattr(article, 'summary') else None,
                "html": response.text[:10000]  # Store first 10k chars of HTML for backup parsing
            }
            
            return metadata
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise Exception(f"Failed to fetch URL: {e}")
        except Exception as e:
            logger.error(f"Error extracting article from {url}: {e}")
            # Try fallback extraction
            return await self._fallback_extraction(url)
    
    async def _fallback_extraction(self, url: str) -> Dict[str, Any]:
        """
        Fallback extraction method using only BeautifulSoup
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {
                "url": url,
                "title": self._extract_title_from_soup(soup),
                "text": text[:50000],  # Limit to 50k chars
                "publisher": self._extract_publisher_from_url(url),
                "author": None,
                "published_date": None,
                "language": "en",
                "word_count": len(text.split()),
                "content_hash": self._generate_hash(text),
                "top_image": None,
                "images": [],
                "keywords": [],
                "summary": None,
                "html": response.text[:10000]
            }
        except Exception as e:
            logger.error(f"Fallback extraction failed for {url}: {e}")
            raise Exception(f"Unable to extract content from URL: {e}")
    
    def _extract_title(self, article: Article, soup: BeautifulSoup) -> str:
        """Extract title from article or soup"""
        if article.title:
            return article.title
        return self._extract_title_from_soup(soup)
    
    def _extract_title_from_soup(self, soup: BeautifulSoup) -> str:
        """Extract title from soup"""
        # Try multiple title selectors
        selectors = [
            'h1',
            'meta[property="og:title"]',
            'meta[name="twitter:title"]',
            'title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    return element.get('content', '')
                else:
                    return element.get_text(strip=True)
        
        return "Unknown Title"
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main text content from soup"""
        # Try to find article body
        article_selectors = [
            'article',
            '[role="main"]',
            '.article-body',
            '.entry-content',
            '.post-content',
            'main'
        ]
        
        for selector in article_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(separator=' ', strip=True)
        
        # Fallback to all paragraph text
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text(strip=True) for p in paragraphs)
    
    def _extract_publisher(self, article: Article, soup: BeautifulSoup, url: str) -> str:
        """Extract publisher name"""
        # Try article meta
        if article.meta_data.get('og', {}).get('site_name'):
            return article.meta_data['og']['site_name']
        
        # Try meta tags
        meta_publisher = soup.find('meta', {'property': 'og:site_name'})
        if meta_publisher:
            return meta_publisher.get('content', '')
        
        # Try schema.org data
        schema = soup.find('script', {'type': 'application/ld+json'})
        if schema:
            try:
                import json
                data = json.loads(schema.string)
                if isinstance(data, dict):
                    publisher = data.get('publisher', {})
                    if isinstance(publisher, dict):
                        return publisher.get('name', '')
            except:
                pass
        
        # Fallback to domain name
        return self._extract_publisher_from_url(url)
    
    def _extract_publisher_from_url(self, url: str) -> str:
        """Extract publisher from URL domain"""
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return domain.split('.')[0].title()
    
    def _extract_author(self, article: Article, soup: BeautifulSoup) -> Optional[str]:
        """Extract author name"""
        if article.authors:
            return ', '.join(article.authors)
        
        # Try meta tags
        author_selectors = [
            'meta[name="author"]',
            'meta[property="article:author"]',
            '[rel="author"]',
            '.author-name',
            '.by-author',
            '.byline'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    return element.get('content', '')
                else:
                    return element.get_text(strip=True)
        
        return None
    
    def _extract_date(self, article: Article, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publication date"""
        if article.publish_date:
            return article.publish_date
        
        # Try meta tags
        date_selectors = [
            ('meta[property="article:published_time"]', 'content'),
            ('meta[name="publish_date"]', 'content'),
            ('time[datetime]', 'datetime'),
            ('time', None)
        ]
        
        for selector, attr in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get(attr) if attr else element.get_text(strip=True)
                if date_str:
                    try:
                        return parser.parse(date_str)
                    except:
                        continue
        
        return None
    
    def _generate_hash(self, text: str) -> str:
        """Generate content hash"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Singleton instance
_scraper = None


async def get_scraper() -> WebScraper:
    """Get or create scraper instance"""
    global _scraper
    if _scraper is None:
        _scraper = WebScraper()
    return _scraper