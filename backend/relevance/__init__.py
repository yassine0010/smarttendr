"""
SmartTender AI — Relevance Filtering Package
===============================================
Production-grade relevance filtering using cosine similarity
between tender representations and company expertise profiles.

Modules:
    company_profile  – Structured company profile with domain vectors
    similarity       – Vector construction + cosine similarity engine
    filter_engine    – Decision logic, thresholding, batch filtering
    calibration      – Threshold tuning & precision/recall analysis
"""

from backend.relevance.filter_engine import (
    RelevanceFilter,
    FilterResult,
    FilterBatchResult,
    FilterDecision,
)
from backend.relevance.company_profile import CompanyProfile
from backend.relevance.similarity import SimilarityEngine
from backend.relevance.calibration import ThresholdCalibrator, CalibrationReport

__all__ = [
    "RelevanceFilter",
    "FilterResult",
    "FilterBatchResult",
    "FilterDecision",
    "CompanyProfile",
    "SimilarityEngine",
    "ThresholdCalibrator",
    "CalibrationReport",
]
