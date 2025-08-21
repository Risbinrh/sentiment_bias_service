import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from app.core.ollama_client import get_ollama_client
from app.models.schemas import Metadata, AnalysisRequest
from app.services.scraper import get_scraper


logger = logging.getLogger(__name__)


class SimpleNewsAnalyzer:
    def __init__(self):
        self.analysis_prompt_template = """Analyze: "{title}" by {publisher}

Text: {text}

Return valid JSON:
{{"article":{{"source_url":"{url}","title":"{title}","publisher":"{publisher}","published_at":"{published_date}","language":"en","word_count":{word_count},"hash":"{content_hash}","author":"{author}","byline":null}},"classification":{{"category":"Business","subcategory":"News","beats":["Business"],"keywords":[{{"text":"news","weight":0.8}}],"tags":["news"],"sentiment":{{"label":"positive","score":0.5}},"tone":[{{"label":"formal","score":0.7}}],"bias":{{"label":"center","score":0.5,"method":"content_analysis"}}}},"summary":{{"abstract":"Summary here.","tldr":"TLDR here.","bullets":["Point 1","Point 2"],"compression_ratio":0.1}},"entities":{{"people":[],"organizations":[],"locations":[],"other":[]}},"editorial":{{"newsworthiness":{{"novelty_score":0.5,"saturation_score":0.5,"controversy_score":0.1}},"fact_check":{{"checkability_score":0.7,"claims":[]}},"angles":[],"impact":{{"audiences":["General"],"regions":["Global"],"sectors":["Business"],"time_horizon":"short-term"}},"risks":{{"legal":[],"ethical":[],"safety":[]}},"pitch":{{"headline":"Headline","subheading":null,"hook":null,"nut_graph":"Matters because","call_to_action":null,"next_steps":[]}}}},"quality":{{"readability":70.0,"hallucination_risk":0.2,"overall_confidence":0.7}},"provenance":{{"pipeline_version":"ollama-analyzer@2.0.0","models":[{{"name":"llama3.2:1b","vendor":"Ollama","version":"1.0","task":"comprehensive_analysis"}}],"processing_time_ms":0,"notes":"Ollama analysis"}}}}

Update sentiment (positive/negative/neutral), write real summary. JSON only."""

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
        
        # Prepare prompt
        prompt = self.analysis_prompt_template.format(
            url=article_data.get("url", ""),
            title=article_data.get("title", ""),
            publisher=article_data.get("publisher", ""),
            author=article_data.get("author", "null"),
            published_date=article_data.get("published_date", "null"),
            language=article_data.get("language", "en"),
            word_count=article_data.get("word_count", 0),
            content_hash=article_data.get("content_hash", ""),
            text=article_data.get("text", "")[:2000]  # Limit text to 2k chars for fast processing
        )
        
        # Call Ollama for analysis
        ollama_client = await get_ollama_client()
        
        try:
            response = await ollama_client.generate(prompt, {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1024  # Reduced for faster response
            })
            
            # Calculate processing time
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Handle response
            if isinstance(response, dict) and "raw_response" not in response:
                # Update processing time in provenance
                if "provenance" in response:
                    response["provenance"]["processing_time_ms"] = processing_time
                
                # Validate and create Metadata object
                return Metadata(**response)
            else:
                # Handle failed JSON parsing
                logger.error("Failed to get valid JSON from Ollama")
                return self._create_fallback_metadata(article_data, processing_time)
                
        except Exception as e:
            logger.error(f"Error in Ollama analysis: {e}")
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return self._create_fallback_metadata(article_data, processing_time, str(e))
    
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
                category="Unknown",
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
_simple_analyzer = None


async def get_simple_analyzer() -> SimpleNewsAnalyzer:
    """Get or create simple analyzer instance"""
    global _simple_analyzer
    if _simple_analyzer is None:
        _simple_analyzer = SimpleNewsAnalyzer()
    return _simple_analyzer