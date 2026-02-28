"""
SmartTender AI — FastAPI Backend
==================================
REST API wrapping the existing scraping, NLP extraction,
relevance filtering, and strategic evaluation modules.

Endpoints:
    GET  /health                  → liveness check
    GET  /sources                 → list available scraping sources
    POST /scrape                  → run scraping pipeline
    POST /analyze                 → NLP + relevance + strategic on tenders
    POST /pipeline                → full pipeline: scrape → analyze
    GET  /tenders/latest          → load latest scraped tenders from disk
    GET  /results/latest          → load latest analysis results from disk
    GET  /profile                 → company profile info

Author: SmartTender AI Team
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ================================================================
# PATH SETUP
# ================================================================
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"

sys.path.insert(0, str(BACKEND_DIR))

# ================================================================
# LAZY MODULE LOADING (heavy imports deferred to first call)
# ================================================================
_detector = None
_pipeline = None


def _get_detector():
    """Lazy-load TenderDetector (loads SBERT + spaCy on first call)."""
    global _detector
    if _detector is None:
        from module1_tender_detection import TenderDetector
        _detector = TenderDetector(threshold=0.65, low_threshold=0.40)
    return _detector


def _get_pipeline():
    """Lazy-load ScrapingPipeline."""
    global _pipeline
    if _pipeline is None:
        from scraping.pipeline import ScrapingPipeline
        _pipeline = ScrapingPipeline()
    return _pipeline


# ================================================================
# PYDANTIC MODELS
# ================================================================

class ScrapeRequest(BaseModel):
    """Request body for /scrape endpoint."""
    query: str = Field(default="IT services", description="Search keywords")
    sources: Optional[List[str]] = Field(
        default=None,
        description="Source names to scrape. None = all. Options: SAM.GOV, TED, UNGM, TUNEPS, CONTRACTS_FINDER"
    )


class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint."""
    tenders: List[Dict[str, Any]] = Field(
        ..., description="List of tender dicts to analyze"
    )


class PipelineRequest(BaseModel):
    """Request body for /pipeline (scrape + analyze)."""
    query: str = Field(default="IT services", description="Search keywords")
    sources: Optional[List[str]] = Field(
        default=None,
        description="Source names. None = all."
    )


# ================================================================
# FASTAPI APP
# ================================================================

app = FastAPI(
    title="SmartTender AI",
    description=(
        "Intelligent tender detection platform for Inetum Tunisie. "
        "Scrapes 5 public procurement sources, extracts keywords via NLP, "
        "scores relevance using SBERT cosine similarity, and provides "
        "strategic evaluation with win probability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/sources")
async def list_sources():
    """List available scraping sources with their tiers."""
    return {
        "sources": [
            {"name": "SAM.GOV", "tier": "API", "region": "United States"},
            {"name": "TED", "tier": "RSS/Feed", "region": "European Union"},
            {"name": "UNGM", "tier": "HTML", "region": "United Nations (Global)"},
            {"name": "TUNEPS", "tier": "Browser/JS", "region": "Tunisia"},
            {"name": "CONTRACTS_FINDER", "tier": "API", "region": "United Kingdom"},
        ]
    }


@app.get("/profile")
async def get_profile():
    """Return the company profile used for relevance scoring."""
    from relevance.company_profile import CompanyProfile, DEFAULT_PROFILE_DATA
    profile = CompanyProfile.default()
    return {
        "name": profile.name,
        "description": profile.description,
        "domains": {
            name: {"weight": info["weight"], "description": info["description"]}
            for name, info in profile.domains.items()
        },
        "skills": profile.skills,
        "certifications": profile.certifications,
        "total_domains": len(profile.domains),
        "total_skills": len(profile.skills),
        "total_certifications": len(profile.certifications),
    }


@app.post("/scrape")
async def scrape_tenders(req: ScrapeRequest):
    """
    Run the multi-tier scraping pipeline.

    Sources: SAM.GOV (API), TED (RSS), UNGM (HTML), TUNEPS (Browser), Contracts Finder (API)
    """
    try:
        pipeline = _get_pipeline()
        fallback = DATA_DIR / "sample_tenders.json"

        start = time.perf_counter()
        tenders = pipeline.run(
            query=req.query,
            sources=req.sources,
            fallback_file=fallback,
        )
        elapsed = time.perf_counter() - start

        results = [t.to_dict() for t in tenders]

        # Group by platform
        by_platform: Dict[str, int] = {}
        for t in results:
            p = t.get("platform", "Unknown")
            by_platform[p] = by_platform.get(p, 0) + 1

        return {
            "total": len(results),
            "by_platform": by_platform,
            "processing_time_seconds": round(elapsed, 2),
            "tenders": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_tenders(req: AnalyzeRequest):
    """
    Run NLP keyword extraction + relevance filtering + strategic evaluation
    on a list of tenders.
    """
    if not req.tenders:
        raise HTTPException(status_code=400, detail="No tenders provided")

    try:
        detector = _get_detector()

        start = time.perf_counter()
        results = detector.analyze_tenders(req.tenders)
        elapsed = time.perf_counter() - start

        # Compute summary stats
        relevant = sum(1 for r in results if r.get("decision") == "RELEVANT")
        low_rel = sum(1 for r in results if r.get("decision") == "LOW_RELEVANCE")
        irrelevant = sum(1 for r in results if r.get("decision") == "IRRELEVANT")

        avg_score = (
            sum(r.get("relevance_score", 0) for r in results) / len(results)
            if results else 0
        )
        avg_win = (
            sum(r.get("win_probability", 0) for r in results) / len(results)
            if results else 0
        )

        return {
            "summary": {
                "total": len(results),
                "relevant": relevant,
                "low_relevance": low_rel,
                "irrelevant": irrelevant,
                "avg_relevance_score": round(avg_score, 2),
                "avg_win_probability": round(avg_win, 2),
                "processing_time_seconds": round(elapsed, 2),
            },
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline")
async def full_pipeline(req: PipelineRequest):
    """
    Full pipeline: scrape → NLP extraction → relevance filtering → strategic evaluation.
    """
    try:
        # Step 1: Scrape
        pipeline = _get_pipeline()
        fallback = DATA_DIR / "sample_tenders.json"

        start = time.perf_counter()
        normalized = pipeline.run(
            query=req.query,
            sources=req.sources,
            fallback_file=fallback,
        )
        scrape_time = time.perf_counter() - start

        # Convert to legacy dicts for the detector
        tender_dicts = [t.to_legacy_dict() for t in normalized]

        # Step 2: Analyze
        detector = _get_detector()
        analyze_start = time.perf_counter()
        results = detector.analyze_tenders(tender_dicts)
        analyze_time = time.perf_counter() - analyze_start

        total_time = time.perf_counter() - start

        # Summary
        relevant = sum(1 for r in results if r.get("decision") == "RELEVANT")
        low_rel = sum(1 for r in results if r.get("decision") == "LOW_RELEVANCE")
        irrelevant = sum(1 for r in results if r.get("decision") == "IRRELEVANT")

        # Platform breakdown
        by_platform: Dict[str, int] = {}
        for t in normalized:
            by_platform[t.platform] = by_platform.get(t.platform, 0) + 1

        # Save results
        output_file = OUTPUT_DIR / "analysis_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        return {
            "summary": {
                "total_scraped": len(normalized),
                "total_analyzed": len(results),
                "relevant": relevant,
                "low_relevance": low_rel,
                "irrelevant": irrelevant,
                "by_platform": by_platform,
                "scrape_time_seconds": round(scrape_time, 2),
                "analyze_time_seconds": round(analyze_time, 2),
                "total_time_seconds": round(total_time, 2),
            },
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tenders/latest")
async def get_latest_tenders():
    """Load the latest scraped tenders from disk."""
    filepath = OUTPUT_DIR / "scraped_tenders_latest.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="No scraped tenders found. Run /scrape first.")

    with open(filepath, "r", encoding="utf-8") as f:
        tenders = json.load(f)

    by_platform: Dict[str, int] = {}
    for t in tenders:
        p = t.get("platform", "Unknown")
        by_platform[p] = by_platform.get(p, 0) + 1

    return {
        "total": len(tenders),
        "by_platform": by_platform,
        "tenders": tenders,
    }


@app.get("/results/latest")
async def get_latest_results():
    """Load the latest analysis results from disk."""
    filepath = OUTPUT_DIR / "analysis_results.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="No analysis results found. Run /pipeline first.")

    with open(filepath, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Normalize legacy format (old results without decision/strategic fields)
    for r in results:
        if "decision" not in r:
            score = r.get("relevance_score", 0)
            if score >= 65:
                r["decision"] = "RELEVANT"
            elif score >= 40:
                r["decision"] = "LOW_RELEVANCE"
            else:
                r["decision"] = "IRRELEVANT"
        r.setdefault("win_probability", 0)
        r.setdefault("strategic_relevance_score", 0)
        r.setdefault("deadline_risk", "N/A")
        r.setdefault("difficulty_level", "N/A")
        r.setdefault("competition_intensity", "N/A")
        r.setdefault("days_remaining", "N/A")
        r.setdefault("complexity_score", 0)
        r.setdefault("best_matching_domain", "General")
        r.setdefault("platform", r.get("platform", "Unknown"))
        r.setdefault("organization", "")
        r.setdefault("location", "")
        r.setdefault("budget", r.get("budget", "N/A"))
        r.setdefault("deadline", r.get("deadline", "N/A"))
        r.setdefault("semantic_similarity", 0)
        r.setdefault("skill_overlap", 0)
        r.setdefault("domain_similarity", 0)
        r.setdefault("matched_skills", [])
        r.setdefault("missing_skills", [])
        r.setdefault("top_keywords", r.get("extracted_keywords", []))
        r.setdefault("detected_certifications", [])

        # Flatten old nested keyword format into a flat list
        kw = r.get("top_keywords", [])
        if isinstance(kw, dict):
            flat = []
            for key in ("tfidf_keywords", "noun_chunks", "entities", "tech_skills"):
                flat.extend(kw.get(key, []))
            # Deduplicate, keep order, limit to 20
            seen = set()
            unique = []
            for k in flat:
                if k.lower() not in seen:
                    seen.add(k.lower())
                    unique.append(k)
            r["top_keywords"] = unique[:20]

    relevant = sum(1 for r in results if r.get("decision") == "RELEVANT")
    low_rel = sum(1 for r in results if r.get("decision") == "LOW_RELEVANCE")
    irrelevant = sum(1 for r in results if r.get("decision") == "IRRELEVANT")

    return {
        "summary": {
            "total": len(results),
            "relevant": relevant,
            "low_relevance": low_rel,
            "irrelevant": irrelevant,
        },
        "results": results,
    }
