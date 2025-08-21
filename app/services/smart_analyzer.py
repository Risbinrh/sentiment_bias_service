import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from app.core.ollama_client import get_ollama_client
from app.models.schemas import Metadata, AnalysisRequest
from app.services.scraper import get_scraper


logger = logging.getLogger(__name__)


class SmartNewsAnalyzer:
    def __init__(self):
        pass

    async def analyze_comprehensive(self, request: AnalysisRequest) -> Metadata:
        """
        Perform comprehensive analysis of news article
        """
        start_time = datetime.now()
        
        # Extract content if URL provided
        if request.url:
            scraper = await get_scraper()
            article_data = await scraper.extract_article(str(request.url))
        else:
            # Use provided text
            article_data = {
                "url": "direct_text_input",
                "title": request.title or "Unknown Title",
                "text": request.text,
                "publisher": request.publisher or "Unknown Publisher",
                "author": None,
                "published_date": None,
                "language": "en",
                "word_count": len(request.text.split()) if request.text else 0,
                "content_hash": "direct_input"
            }
        
        # Get Ollama analysis for different aspects
        ollama_client = await get_ollama_client()
        article_text = article_data.get("text", "")[:3000]  # Limit text to 3k chars
        
        try:
            # Get sentiment analysis
            sentiment_prompt = f"""Analyze the sentiment of this news article:
Title: {article_data.get('title', '')}
Text: {article_text}

Return JSON: {{"sentiment": "positive", "score": 0.8, "reasoning": "why"}}"""
            
            sentiment_response = await ollama_client.generate(sentiment_prompt, {
                "temperature": 0.1,
                "num_predict": 100
            })
            
            # Get summary
            summary_prompt = f"""Summarize this news article:
Title: {article_data.get('title', '')}
Text: {article_text}

Return JSON: {{"abstract": "2-3 sentence summary", "tldr": "one sentence", "key_points": ["point1", "point2", "point3"]}}"""
            
            summary_response = await ollama_client.generate(summary_prompt, {
                "temperature": 0.2,
                "num_predict": 200
            })
            
            # Get entities
            entities_prompt = f"""Extract key entities from this news article:
Title: {article_data.get('title', '')}
Text: {article_text}

Return JSON: {{"people": ["person1", "person2"], "organizations": ["org1", "org2"], "locations": ["loc1", "loc2"]}}"""
            
            entities_response = await ollama_client.generate(entities_prompt, {
                "temperature": 0.1,
                "num_predict": 150
            })
            
            # Calculate processing time
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Build enterprise metadata from Ollama responses
            return self._build_enterprise_metadata(
                article_data, 
                sentiment_response,
                summary_response, 
                entities_response,
                processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in Ollama analysis: {e}")
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return self._create_fallback_metadata(article_data, processing_time, str(e))
    
    def _build_enterprise_metadata(self, article_data: Dict[str, Any], sentiment_data: Dict[str, Any], 
                                 summary_data: Dict[str, Any], entities_data: Dict[str, Any],
                                 processing_time: int) -> Metadata:
        """
        Build enterprise metadata from Ollama responses
        """
        from app.models.schemas import (
            ArticleMetadata, Classification, Summary, Entities, Editorial,
            Quality, Provenance, SentimentScore, BiasScore, ToneScore,
            Newsworthiness, FactCheck, Impact, Risks, Pitch, Model, Entity, Keyword
        )
        
        # Extract sentiment
        sentiment_label = sentiment_data.get("sentiment", "neutral")
        sentiment_score = sentiment_data.get("score", 0.0)
        
        # Extract summary
        abstract = summary_data.get("abstract", "Article summary not available")
        tldr = summary_data.get("tldr", "Summary not available")
        key_points = summary_data.get("key_points", ["Analysis completed"])
        
        # Extract entities and convert to our format
        people_names = entities_data.get("people", [])
        org_names = entities_data.get("organizations", [])
        location_names = entities_data.get("locations", [])
        
        people = [Entity(name=name, type="person", salience=0.8, sentiment=0.1) for name in people_names[:5]]
        organizations = [Entity(name=name, type="org", salience=0.9, sentiment=0.2) for name in org_names[:5]]
        locations = [Entity(name=name, type="place", salience=0.6) for name in location_names[:5]]
        
        # Generate keywords from title and entities
        title_words = article_data.get("title", "").lower().split()
        keywords = []
        for word in title_words:
            if len(word) > 3:
                keywords.append(Keyword(text=word, weight=0.7))
        if len(keywords) == 0:
            keywords.append(Keyword(text="news", weight=0.5))
        
        # Determine category based on content
        text_lower = article_data.get("text", "").lower()
        category = "Business"
        if any(word in text_lower for word in ["tech", "technology", "ai", "software"]):
            category = "Technology"
        elif any(word in text_lower for word in ["politics", "government", "election"]):
            category = "Politics"
        elif any(word in text_lower for word in ["health", "medical", "hospital"]):
            category = "Health"
        elif any(word in text_lower for word in ["sport", "game", "match"]):
            category = "Sports"
        
        return Metadata(
            article=ArticleMetadata(
                source_url=article_data.get("url", ""),
                title=article_data.get("title", "Unknown Title"),
                publisher=article_data.get("publisher", "Unknown Publisher"),
                published_at=article_data.get("published_date"),
                language=article_data.get("language", "en"),
                word_count=article_data.get("word_count", 0),
                hash=article_data.get("content_hash", ""),
                author=article_data.get("author")
            ),
            classification=Classification(
                category=category,
                subcategory="News",
                beats=[category],
                keywords=keywords[:10],
                tags=["news", "analysis"],
                sentiment=SentimentScore(label=sentiment_label, score=float(sentiment_score)),
                tone=[ToneScore(label="formal", score=0.7)],
                bias=BiasScore(label="center", score=0.5)
            ),
            summary=Summary(
                abstract=abstract,
                tldr=tldr,
                bullets=key_points[:5],
                compression_ratio=0.1
            ),
            entities=Entities(
                people=people,
                organizations=organizations,
                locations=locations,
                other=[]
            ),
            editorial=Editorial(
                newsworthiness=Newsworthiness(
                    novelty_score=0.7,
                    saturation_score=0.3,
                    controversy_score=0.1
                ),
                fact_check=FactCheck(checkability_score=0.8),
                impact=Impact(
                    audiences=["General readers", "Industry professionals"],
                    regions=["Global"],
                    sectors=[category],
                    time_horizon="short-term"
                ),
                risks=Risks(),
                pitch=Pitch(
                    headline=f"Analysis: {article_data.get('title', 'News Update')[:60]}...",
                    nut_graph="This story matters because it provides current insights into recent developments"
                )
            ),
            quality=Quality(
                readability=75.0,
                hallucination_risk=0.2,
                overall_confidence=0.8
            ),
            provenance=Provenance(
                models=[Model(
                    name="llama3.2:1b",
                    version="1.0",
                    task="comprehensive_analysis"
                )],
                processing_time_ms=processing_time,
                notes="Generated using Ollama multi-prompt analysis"
            )
        )
    
    def _create_fallback_metadata(self, article_data: Dict[str, Any], processing_time: int, error: str = None) -> Metadata:
        """
        Create basic metadata when Ollama analysis fails
        """
        from app.models.schemas import (
            ArticleMetadata, Classification, Summary, Entities, Editorial,
            Quality, Provenance, SentimentScore, BiasScore, Newsworthiness,
            FactCheck, Impact, Risks, Pitch, Model
        )
        
        return Metadata(
            article=ArticleMetadata(
                source_url=article_data.get("url", ""),
                title=article_data.get("title", "Unknown Title"),
                publisher=article_data.get("publisher", "Unknown Publisher"),
                published_at=article_data.get("published_date"),
                language=article_data.get("language", "en"),
                word_count=article_data.get("word_count", 0),
                hash=article_data.get("content_hash", ""),
                author=article_data.get("author")
            ),
            classification=Classification(
                category="Business",
                sentiment=SentimentScore(label="neutral", score=0.0),
                bias=BiasScore(label="center", score=0.5)
            ),
            summary=Summary(
                abstract="Analysis unavailable",
                tldr="Content could not be analyzed",
                bullets=["Analysis failed"]
            ),
            entities=Entities(),
            editorial=Editorial(
                newsworthiness=Newsworthiness(
                    novelty_score=0.5,
                    saturation_score=0.5,
                    controversy_score=0.0
                ),
                fact_check=FactCheck(checkability_score=0.0),
                impact=Impact(time_horizon="short-term"),
                risks=Risks(),
                pitch=Pitch(
                    headline="Content Analysis Unavailable",
                    nut_graph="Unable to analyze content due to technical issues"
                )
            ),
            quality=Quality(
                readability=50.0,
                hallucination_risk=1.0,
                overall_confidence=0.0
            ),
            provenance=Provenance(
                models=[Model(
                    name="llama3.2:1b",
                    version="1.0",
                    task="comprehensive_analysis"
                )],
                processing_time_ms=processing_time,
                notes=f"Fallback metadata due to analysis failure: {error}" if error else "Fallback metadata"
            )
        )


# Singleton instance
_smart_analyzer = None


async def get_smart_analyzer() -> SmartNewsAnalyzer:
    """Get or create smart analyzer instance"""
    global _smart_analyzer
    if _smart_analyzer is None:
        _smart_analyzer = SmartNewsAnalyzer()
    return _smart_analyzer