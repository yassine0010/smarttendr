"""
SmartTender AI — Relevance Filtering Engine
=============================================
Orchestrates similarity computation to produce a final
RELEVANT / LOW_RELEVANCE / IRRELEVANT decision per tender.

Decision thresholds (tuned for multilingual / cross-lingual scoring):
    score ≥ 0.55  →  RELEVANT          (pursue)
    0.30 ≤ score < 0.55  →  LOW_RELEVANCE  (review)
    score < 0.30  →  IRRELEVANT         (skip)

These defaults are configurable and can be tuned via the
calibration module.

Author: SmartTender AI Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from backend.relevance.company_profile import CompanyProfile
from backend.relevance.similarity import SimilarityEngine


# ================================================================
# DECISION ENUM
# ================================================================

class FilterDecision(str, Enum):
    """Three-tier relevance decision."""

    RELEVANT = "RELEVANT"
    LOW_RELEVANCE = "LOW_RELEVANCE"
    IRRELEVANT = "IRRELEVANT"

    def __str__(self) -> str:
        return self.value


# ================================================================
# FILTER RESULT
# ================================================================

@dataclass
class FilterResult:
    """
    Structured output for a single tender's relevance assessment.

    This is the final product of the filtering pipeline.
    Every field is JSON-serializable for downstream consumers.
    """

    tender_id: str
    tender_title: str

    # Score breakdown
    final_score: float
    semantic_similarity: float
    skill_overlap: float
    domain_similarity: float

    # Decision
    threshold: float
    decision: FilterDecision

    # Diagnostics
    best_matching_domain: str
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    computation_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize to plain dict for JSON output."""
        return {
            "tender_id": self.tender_id,
            "tender_title": self.tender_title,
            "final_score": self.final_score,
            "semantic_similarity": self.semantic_similarity,
            "skill_overlap": self.skill_overlap,
            "domain_similarity": self.domain_similarity,
            "threshold": self.threshold,
            "decision": str(self.decision),
            "best_matching_domain": self.best_matching_domain,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "computation_time_ms": round(self.computation_time_ms, 2),
        }


# ================================================================
# FILTER BATCH RESULT
# ================================================================

@dataclass
class FilterBatchResult:
    """Summary of a batch filtering run."""

    total_tenders: int
    relevant_count: int
    low_relevance_count: int
    irrelevant_count: int
    results: List[FilterResult]
    processing_time_ms: float

    # Score statistics
    mean_score: float = 0.0
    max_score: float = 0.0
    min_score: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize to plain dict."""
        return {
            "summary": {
                "total_tenders": self.total_tenders,
                "relevant": self.relevant_count,
                "low_relevance": self.low_relevance_count,
                "irrelevant": self.irrelevant_count,
                "processing_time_ms": round(self.processing_time_ms, 2),
                "mean_score": round(self.mean_score, 4),
                "max_score": round(self.max_score, 4),
                "min_score": round(self.min_score, 4),
            },
            "results": [r.to_dict() for r in self.results],
        }

    @property
    def relevant_tenders(self) -> List[FilterResult]:
        """Only RELEVANT results."""
        return [r for r in self.results if r.decision == FilterDecision.RELEVANT]

    @property
    def low_relevance_tenders(self) -> List[FilterResult]:
        """Only LOW_RELEVANCE results."""
        return [r for r in self.results if r.decision == FilterDecision.LOW_RELEVANCE]


# ================================================================
# RELEVANCE FILTER (main class)
# ================================================================

class RelevanceFilter:
    """
    Production relevance filtering engine.

    Workflow:
        1. Initialize with company profile + thresholds
        2. Precompute profile embeddings (once)
        3. For each tender: embed → score → decide → FilterResult
        4. Return sorted + classified results

    Thread-safe for read-only operations after init.
    """

    # Default thresholds (tuned for multilingual cross-lingual scoring)
    DEFAULT_RELEVANT_THRESHOLD = 0.55
    DEFAULT_LOW_RELEVANCE_THRESHOLD = 0.30

    def __init__(
        self,
        profile: Optional[CompanyProfile] = None,
        relevant_threshold: float = DEFAULT_RELEVANT_THRESHOLD,
        low_relevance_threshold: float = DEFAULT_LOW_RELEVANCE_THRESHOLD,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the relevance filter.

        Args:
            profile:                 Company profile (uses default if None).
            relevant_threshold:      Score >= this → RELEVANT.
            low_relevance_threshold: Score >= this (and < relevant) → LOW_RELEVANCE.
            model_name:              SBERT model identifier.
            weights:                 Custom hybrid scoring weights.
        """
        self.relevant_threshold = relevant_threshold
        self.low_relevance_threshold = low_relevance_threshold

        # Initialize similarity engine
        self.engine = SimilarityEngine(
            model_name=model_name,
            weights=weights,
        )

        # Load and embed company profile
        self.profile = profile or CompanyProfile.default()
        self.engine.load_profile(self.profile)

    # ============================================================
    # SINGLE TENDER FILTERING
    # ============================================================

    def filter_tender(self, tender: Dict) -> FilterResult:
        """
        Evaluate a single tender's relevance.

        Args:
            tender: Tender dict with at least "title", "description".
                    Optional: "detected_skills", "detected_domain",
                    "detected_certifications", "id" or "tender_id".

        Returns:
            FilterResult with scores, decision, and diagnostics.
        """
        start = time.perf_counter()

        # Embed the tender
        embedding = self.engine.embed_tender(tender)

        # Compute hybrid score
        scores = self.engine.compute_hybrid_score(
            tender, embedding, self.profile
        )

        # Make decision
        decision = self._decide(scores["final_score"])

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Build result
        tender_id = (
            tender.get("id")
            or tender.get("tender_id")
            or tender.get("notice_id", "unknown")
        )

        return FilterResult(
            tender_id=str(tender_id),
            tender_title=tender.get("title", "Untitled"),
            final_score=scores["final_score"],
            semantic_similarity=scores["semantic_similarity"],
            skill_overlap=scores["skill_overlap"],
            domain_similarity=scores["domain_similarity"],
            threshold=self.relevant_threshold,
            decision=decision,
            best_matching_domain=scores["best_matching_domain"],
            matched_skills=scores["matched_skills"],
            missing_skills=scores["missing_skills"],
            computation_time_ms=elapsed_ms,
        )

    # ============================================================
    # BATCH FILTERING
    # ============================================================

    def filter_tenders(
        self,
        tenders: List[Dict],
        sort_by_score: bool = True,
    ) -> FilterBatchResult:
        """
        Evaluate and classify a batch of tenders.

        Uses SBERT batch encoding for efficient embedding,
        then computes per-tender scores and decisions.

        Args:
            tenders:       List of tender dicts.
            sort_by_score: If True, results sorted descending by score.

        Returns:
            FilterBatchResult with summary statistics and per-tender results.
        """
        if not tenders:
            return FilterBatchResult(
                total_tenders=0,
                relevant_count=0,
                low_relevance_count=0,
                irrelevant_count=0,
                results=[],
                processing_time_ms=0.0,
            )

        start = time.perf_counter()

        # Batch-embed all tenders
        embeddings = self.engine.embed_tenders_batch(tenders)

        # Score each tender
        results: List[FilterResult] = []
        for i, tender in enumerate(tenders):
            t_start = time.perf_counter()

            scores = self.engine.compute_hybrid_score(
                tender, embeddings[i], self.profile
            )
            decision = self._decide(scores["final_score"])

            t_elapsed = (time.perf_counter() - t_start) * 1000

            tender_id = (
                tender.get("id")
                or tender.get("tender_id")
                or tender.get("notice_id", "unknown")
            )

            results.append(FilterResult(
                tender_id=str(tender_id),
                tender_title=tender.get("title", "Untitled"),
                final_score=scores["final_score"],
                semantic_similarity=scores["semantic_similarity"],
                skill_overlap=scores["skill_overlap"],
                domain_similarity=scores["domain_similarity"],
                threshold=self.relevant_threshold,
                decision=decision,
                best_matching_domain=scores["best_matching_domain"],
                matched_skills=scores["matched_skills"],
                missing_skills=scores["missing_skills"],
                computation_time_ms=t_elapsed,
            ))

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Sort by score descending
        if sort_by_score:
            results.sort(key=lambda r: r.final_score, reverse=True)

        # Compute statistics
        scores_list = [r.final_score for r in results]
        relevant = sum(1 for r in results if r.decision == FilterDecision.RELEVANT)
        low_rel = sum(1 for r in results if r.decision == FilterDecision.LOW_RELEVANCE)
        irrelevant = sum(1 for r in results if r.decision == FilterDecision.IRRELEVANT)

        return FilterBatchResult(
            total_tenders=len(results),
            relevant_count=relevant,
            low_relevance_count=low_rel,
            irrelevant_count=irrelevant,
            results=results,
            processing_time_ms=elapsed_ms,
            mean_score=round(sum(scores_list) / len(scores_list), 4),
            max_score=round(max(scores_list), 4),
            min_score=round(min(scores_list), 4),
        )

    # ============================================================
    # THRESHOLD DECISION
    # ============================================================

    def _decide(self, score: float) -> FilterDecision:
        """
        Apply threshold logic.

            score ≥ 0.55           → RELEVANT
            0.30 ≤ score < 0.55   → LOW_RELEVANCE
            score < 0.30           → IRRELEVANT
        """
        if score >= self.relevant_threshold:
            return FilterDecision.RELEVANT
        elif score >= self.low_relevance_threshold:
            return FilterDecision.LOW_RELEVANCE
        else:
            return FilterDecision.IRRELEVANT

    def update_thresholds(
        self,
        relevant: Optional[float] = None,
        low_relevance: Optional[float] = None,
    ) -> None:
        """Update thresholds at runtime (for calibration)."""
        if relevant is not None:
            self.relevant_threshold = relevant
        if low_relevance is not None:
            self.low_relevance_threshold = low_relevance
