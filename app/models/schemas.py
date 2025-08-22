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


class SEOAnalysis(BaseModel):
    search_engine_visibility: float = Field(ge=0, le=1, description="How likely this content is to rank well")
    keyword_density: float = Field(ge=0, le=1, description="Keyword optimization score")
    content_freshness: float = Field(ge=0, le=1, description="Timeliness and relevance score")
    readability_score: float = Field(ge=0, le=1, description="Content accessibility for search engines")
    trending_potential: float = Field(ge=0, le=1, description="Likelihood to go viral or trend")
    search_intent_match: str = Field(description="Primary search intent (informational/navigational/transactional)")
    target_keywords: List[str] = Field(description="Primary keywords for SEO targeting")
    content_gaps: List[str] = Field(description="Missing elements that could improve SEO")
    competitor_advantage: float = Field(ge=0, le=1, description="How this content compares to competitors")
    overall_seo_score: float = Field(ge=0, le=1, description="Overall SEO potential score")


class NewsroomPitchScore(BaseModel):
    newsworthiness: float = Field(ge=0, le=1, description="How newsworthy this story is")
    audience_appeal: float = Field(ge=0, le=1, description="Expected audience engagement")
    exclusivity_factor: float = Field(ge=0, le=1, description="How unique/exclusive this story is")
    social_media_potential: float = Field(ge=0, le=1, description="Likelihood to spread on social media")
    editorial_urgency: float = Field(ge=0, le=1, description="How time-sensitive this story is")
    resource_requirements: float = Field(ge=0, le=1, description="Editorial resources needed (inverse scale)")
    brand_alignment: float = Field(ge=0, le=1, description="How well this aligns with publication brand")
    controversy_risk: float = Field(ge=0, le=1, description="Potential for negative backlash")
    follow_up_potential: float = Field(ge=0, le=1, description="Likelihood of generating follow-up stories")
    overall_pitch_score: float = Field(ge=0, le=1, description="Overall newsroom recommendation score")
    recommendation: str = Field(description="Editorial recommendation (Pursue/Consider/Pass)")
    pitch_notes: List[str] = Field(description="Key points for the editorial pitch")


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
    seo_analysis: SEOAnalysis
    newsroom_pitch_score: NewsroomPitchScore
    provenance: Provenance
    context: Optional[Context] = None
    multimedia: Optional[Multimedia] = None


class AnalysisRequest(BaseModel):
    url: HttpUrl


class AnalysisResponse(BaseModel):
    success: bool
    processing_time_ms: int
    article_url: Optional[str] = None
    metadata: Metadata
    error: Optional[str] = None


class BatchAnalysisRequest(BaseModel):
    urls: List[HttpUrl] = Field(max_length=10)


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