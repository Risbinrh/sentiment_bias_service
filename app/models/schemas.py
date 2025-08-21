from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class BiasLabel(str, Enum):
    LEFT = "left"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    RIGHT = "right"


class ToneLabel(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    ANALYTICAL = "analytical"
    EMOTIONAL = "emotional"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"


class TimeHorizon(str, Enum):
    SHORT_TERM = "short-term"
    MEDIUM_TERM = "medium-term"
    LONG_TERM = "long-term"


class Keyword(BaseModel):
    text: str
    weight: float = Field(ge=0, le=1)


class SentimentScore(BaseModel):
    label: SentimentLabel
    score: float = Field(ge=-1, le=1)


class ToneScore(BaseModel):
    label: ToneLabel
    score: float = Field(ge=0, le=1)


class BiasScore(BaseModel):
    label: BiasLabel
    score: float = Field(ge=0, le=1)
    method: str = "content_analysis"


class Entity(BaseModel):
    name: str
    type: str
    salience: float = Field(ge=0, le=1)
    sentiment: Optional[float] = Field(default=None, ge=-1, le=1)


class Claim(BaseModel):
    text: str
    priority: int = Field(ge=1)
    suggested_sources: List[str] = []


class Angle(BaseModel):
    label: str
    rationale: str


class NextStep(BaseModel):
    action: str
    owner: str
    due: Optional[datetime] = None


class ArticleMetadata(BaseModel):
    source_url: str
    title: str
    publisher: str
    published_at: Optional[datetime] = None
    language: str = "en"
    word_count: Optional[int] = None
    hash: Optional[str] = None
    author: Optional[str] = None
    byline: Optional[str] = None


class Classification(BaseModel):
    category: str
    subcategory: Optional[str] = None
    beats: List[str] = []
    keywords: List[Keyword] = []
    tags: List[str] = []
    sentiment: SentimentScore
    tone: List[ToneScore] = []
    bias: BiasScore


class Summary(BaseModel):
    abstract: str = Field(description="2-3 sentence comprehensive summary")
    tldr: str = Field(description="Single sentence distillation")
    bullets: List[str] = Field(description="3-5 key takeaways")
    compression_ratio: Optional[float] = None


class Entities(BaseModel):
    people: List[Entity] = []
    organizations: List[Entity] = []
    locations: List[Entity] = []
    other: List[Entity] = []


class Newsworthiness(BaseModel):
    novelty_score: float = Field(ge=0, le=1)
    saturation_score: float = Field(ge=0, le=1)
    controversy_score: float = Field(ge=0, le=1)


class FactCheck(BaseModel):
    checkability_score: float = Field(ge=0, le=1)
    claims: List[Claim] = []


class Impact(BaseModel):
    audiences: List[str] = []
    regions: List[str] = []
    sectors: List[str] = []
    time_horizon: TimeHorizon


class Risks(BaseModel):
    legal: List[str] = []
    ethical: List[str] = []
    safety: List[str] = []


class Pitch(BaseModel):
    headline: str
    subheading: Optional[str] = None
    hook: Optional[str] = None
    nut_graph: str
    call_to_action: Optional[str] = None
    next_steps: List[NextStep] = []


class Editorial(BaseModel):
    newsworthiness: Newsworthiness
    fact_check: FactCheck
    angles: List[Angle] = []
    impact: Impact
    risks: Risks
    pitch: Pitch


class Quality(BaseModel):
    readability: float = Field(ge=0, le=100)
    hallucination_risk: float = Field(ge=0, le=1)
    overall_confidence: float = Field(ge=0, le=1)


class Model(BaseModel):
    name: str
    vendor: str = "Ollama"
    version: str
    task: str


class Provenance(BaseModel):
    pipeline_version: str = "ollama-analyzer@2.0.0"
    models: List[Model] = []
    processing_time_ms: int
    notes: Optional[str] = None


class Context(BaseModel):
    timeline: Optional[List[Dict[str, Any]]] = None
    related_stories: Optional[List[Dict[str, Any]]] = None


class Multimedia(BaseModel):
    visualization_potential: Optional[float] = Field(default=None, ge=0, le=1)
    asset_suggestions: Optional[List[str]] = None


class Metadata(BaseModel):
    article: ArticleMetadata
    classification: Classification
    summary: Summary
    entities: Entities
    editorial: Editorial
    quality: Quality
    provenance: Provenance
    context: Optional[Context] = None
    multimedia: Optional[Multimedia] = None


class AnalysisRequest(BaseModel):
    url: Optional[HttpUrl] = None
    text: Optional[str] = None
    title: Optional[str] = None
    publisher: Optional[str] = None
    options: Optional[Dict[str, bool]] = Field(default_factory=lambda: {
        "include_timeline": False,
        "related_stories": False,
        "detailed_entities": True
    })


class AnalysisResponse(BaseModel):
    success: bool
    processing_time_ms: int
    article_url: Optional[str] = None
    metadata: Metadata
    error: Optional[str] = None


class BatchAnalysisRequest(BaseModel):
    urls: List[HttpUrl] = Field(max_length=10)
    options: Optional[Dict[str, bool]] = Field(default_factory=lambda: {
        "include_timeline": False,
        "related_stories": False,
        "detailed_entities": True
    })


class BatchAnalysisResponse(BaseModel):
    success: bool
    total_processing_time_ms: int
    results: List[AnalysisResponse]
    summary_statistics: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    status: str
    ollama_connected: bool
    model_available: bool
    version: str
    uptime_seconds: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None