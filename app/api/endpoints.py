import asyncio
from datetime import datetime, timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request
import logging

from app.models.schemas import (
    AnalysisRequest, AnalysisResponse, BatchAnalysisRequest, 
    BatchAnalysisResponse, HealthCheckResponse
)
from app.services.final_analyzer import get_final_analyzer
from app.core.ollama_client import get_ollama_client
from app.api.auth import verify_api_key
from app.api.rate_limit import check_rate_limit
from app.core.config import settings


logger = logging.getLogger(__name__)
router = APIRouter()

# Track service start time for uptime calculation
SERVICE_START_TIME = datetime.now()


@router.post("/analyze-comprehensive", response_model=AnalysisResponse)
async def analyze_comprehensive(
    request: AnalysisRequest,
    _: str = Depends(verify_api_key),
    __: bool = Depends(check_rate_limit)
):
    """
    Perform comprehensive analysis of a news article
    """
    start_time = datetime.now()
    
    try:
        # Perform analysis
        analyzer = await get_final_analyzer()
        metadata = await analyzer.analyze_comprehensive(request)
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return AnalysisResponse(
            success=True,
            processing_time_ms=processing_time,
            article_url=str(request.url),
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/analyze-comprehensive/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    _: str = Depends(verify_api_key),
    __: bool = Depends(check_rate_limit)
):
    """
    Perform batch analysis of multiple news articles
    """
    start_time = datetime.now()
    
    try:
        if len(request.urls) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 URLs allowed per batch request"
            )
        
        analyzer = await get_final_analyzer()
        
        # Process all URLs concurrently
        tasks = []
        for url in request.urls:
            analysis_request = AnalysisRequest(url=url)
            tasks.append(analyzer.analyze_comprehensive(analysis_request))
        
        # Execute all analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        analysis_responses = []
        success_count = 0
        error_count = 0
        total_articles_processing_time = 0
        
        for i, result in enumerate(results):
            url = str(request.urls[i])
            
            if isinstance(result, Exception):
                logger.error(f"Error analyzing {url}: {result}")
                analysis_responses.append(AnalysisResponse(
                    success=False,
                    processing_time_ms=0,
                    article_url=url,
                    metadata=None,
                    error=str(result)
                ))
                error_count += 1
            else:
                analysis_responses.append(AnalysisResponse(
                    success=True,
                    processing_time_ms=result.provenance.processing_time_ms,
                    article_url=url,
                    metadata=result
                ))
                success_count += 1
                total_articles_processing_time += result.provenance.processing_time_ms
        
        # Calculate total processing time
        total_processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Generate summary statistics
        summary_stats = {
            "total_urls": len(request.urls),
            "successful_analyses": success_count,
            "failed_analyses": error_count,
            "success_rate": success_count / len(request.urls) if request.urls else 0,
            "average_processing_time_ms": (
                total_articles_processing_time / success_count if success_count > 0 else 0
            )
        }
        
        return BatchAnalysisResponse(
            success=error_count == 0,
            total_processing_time_ms=total_processing_time,
            results=analysis_responses,
            summary_statistics=summary_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        total_processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.get("/analyze-comprehensive/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check service health and Ollama connectivity
    """
    try:
        # Check Ollama connectivity
        ollama_client = await get_ollama_client()
        health_status = await ollama_client.check_health()
        
        ollama_connected = health_status["primary_url"] or health_status["fallback_url"]
        model_available = health_status["model_available"]
        
        # Calculate uptime
        uptime = (datetime.now() - SERVICE_START_TIME).total_seconds()
        
        # Basic performance metrics
        performance_metrics = {
            "available_models": health_status["models"],
            "primary_url_status": health_status["primary_url"],
            "fallback_url_status": health_status["fallback_url"]
        }
        
        # Determine overall status
        if ollama_connected and model_available:
            status = "healthy"
        elif ollama_connected:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return HealthCheckResponse(
            status=status,
            ollama_connected=ollama_connected,
            model_available=model_available,
            version=settings.version,
            uptime_seconds=uptime,
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            ollama_connected=False,
            model_available=False,
            version=settings.version,
            uptime_seconds=(datetime.now() - SERVICE_START_TIME).total_seconds()
        )


@router.get("/")
async def root():
    """
    Root endpoint with service information
    """
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "running",
        "endpoints": {
            "analysis": "/api/v1/analyze-comprehensive",
            "batch": "/api/v1/analyze-comprehensive/batch", 
            "health": "/api/v1/analyze-comprehensive/health"
        },
        "documentation": "/docs"
    }