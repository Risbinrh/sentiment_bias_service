import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from app.core.ollama_client import get_ollama_client
from app.models.schemas import Metadata, AnalysisRequest
from app.services.scraper import get_scraper


logger = logging.getLogger(__name__)


class NewsAnalyzer:
    def __init__(self):
        self.analysis_prompt_template = self._create_analysis_prompt_template()
    
    def _create_analysis_prompt_template(self) -> str:
        """
        Create comprehensive analysis prompt template for Ollama
        """
        return """Analyze this news article and return JSON with enterprise schema. 

Article Information:
URL: {url}
Title: {title}
Publisher: {publisher}
Author: {author}
Published: {published_date}
Language: {language}
Word Count: {word_count}

Article Content:
{text}

INSTRUCTIONS:
Analyze this article and provide a comprehensive JSON response that covers all aspects below. Ensure ALL fields are populated with meaningful data. Use your expertise to provide accurate, detailed analysis.

REQUIRED JSON STRUCTURE:
{{
  "article": {{
    "source_url": "{url}",
    "title": "{title}",
    "publisher": "{publisher}",
    "published_at": "{published_date}",
    "language": "{language}",
    "word_count": {word_count},
    "hash": "{content_hash}",
    "author": "{author}",
    "byline": null
  }},
  "classification": {{
    "category": "Primary category (e.g., Business, Technology, Politics, Health, Sports, etc.)",
    "subcategory": "More specific classification",
    "beats": ["List", "of", "relevant", "beats"],
    "keywords": [
      {{"text": "keyword1", "weight": 0.95}},
      {{"text": "keyword2", "weight": 0.87}}
    ],
    "tags": ["list", "of", "content", "tags"],
    "sentiment": {{
      "label": "positive|negative|neutral|mixed",
      "score": 0.0
    }},
    "tone": [
      {{"label": "formal|casual|analytical|emotional|optimistic|pessimistic", "score": 0.0}}
    ],
    "bias": {{
      "label": "left|center-left|center|center-right|right",
      "score": 0.0,
      "method": "content_analysis"
    }}
  }},
  "summary": {{
    "abstract": "2-3 sentence comprehensive summary of the article's main points",
    "tldr": "Single sentence distillation of the core message",
    "bullets": [
      "Key takeaway 1",
      "Key takeaway 2", 
      "Key takeaway 3"
    ],
    "compression_ratio": 0.0
  }},
  "entities": {{
    "people": [
      {{"name": "Person Name", "type": "person", "salience": 0.0, "sentiment": 0.0}}
    ],
    "organizations": [
      {{"name": "Company/Org Name", "type": "org", "salience": 0.0, "sentiment": 0.0}}
    ],
    "locations": [
      {{"name": "Location Name", "type": "place", "salience": 0.0}}
    ],
    "other": [
      {{"name": "Other Entity", "type": "product|event|concept", "salience": 0.0}}
    ]
  }},
  "editorial": {{
    "newsworthiness": {{
      "novelty_score": 0.0,
      "saturation_score": 0.0,
      "controversy_score": 0.0
    }},
    "fact_check": {{
      "checkability_score": 0.0,
      "claims": [
        {{
          "text": "Verifiable claim from the article",
          "priority": 1,
          "suggested_sources": ["https://source1.com"]
        }}
      ]
    }},
    "angles": [
      {{
        "label": "Editorial angle",
        "rationale": "Why this angle is relevant"
      }}
    ],
    "impact": {{
      "audiences": ["Target audience 1", "Target audience 2"],
      "regions": ["Geographic regions affected"],
      "sectors": ["Industry sectors impacted"],
      "time_horizon": "short-term|medium-term|long-term"
    }},
    "risks": {{
      "legal": ["Legal considerations if any"],
      "ethical": ["Ethical considerations if any"],
      "safety": ["Safety considerations if any"]
    }},
    "pitch": {{
      "headline": "Compelling headline for republication",
      "subheading": "Supporting subheading",
      "hook": "Engaging opening hook",
      "nut_graph": "2-3 sentence explanation of why this matters now",
      "call_to_action": "What readers should do next",
      "next_steps": [
        {{
          "action": "Specific next step",
          "owner": "Team/Person responsible",
          "due": null
        }}
      ]
    }}
  }},
  "quality": {{
    "readability": 75.0,
    "hallucination_risk": 0.1,
    "overall_confidence": 0.9
  }},
  "provenance": {{
    "pipeline_version": "ollama-analyzer@2.0.0",
    "models": [
      {{
        "name": "llama3.2:1b",
        "vendor": "Ollama", 
        "version": "1.0",
        "task": "comprehensive_analysis"
      }}
    ],
    "processing_time_ms": 0,
    "notes": "Generated using Ollama local inference with enterprise schema"
  }}
}}

ANALYSIS GUIDELINES:
1. SENTIMENT: Score from -1.0 (very negative) to +1.0 (very positive)
2. SALIENCE: Entity importance from 0.0 to 1.0
3. BIAS: Political leaning analysis based on language and framing
4. NEWSWORTHINESS: Assess novelty, market saturation, controversy potential
5. FACT-CHECKING: Identify verifiable claims that could be checked
6. IMPACT: Consider who is affected and in what timeframe
7. QUALITY METRICS: Assess readability, potential for AI hallucination, confidence

Be thorough, accurate, and provide meaningful insights that would be valuable for enterprise newsroom decision-making.

Respond ONLY with the JSON structure above, no additional text or formatting.
"""

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
            text=article_data.get("text", "")[:10000]  # Limit text to 10k chars
        )
        
        # Call Ollama for analysis
        ollama_client = await get_ollama_client()
        
        try:
            response = await ollama_client.generate(prompt)
            
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
_analyzer = None


async def get_analyzer() -> NewsAnalyzer:
    """Get or create analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = NewsAnalyzer()
    return _analyzer