"""
SmartTender AI — Threshold Calibration Utility
================================================
Provides tools to analyze score distributions and tune
the RELEVANT / LOW_RELEVANCE / IRRELEVANT thresholds.

Use cases:
    1. Run on a set of tenders, inspect score distribution
    2. Find optimal thresholds via precision/recall tradeoffs
    3. Validate thresholds on labeled data

Author: SmartTender AI Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.relevance.filter_engine import (
    FilterBatchResult,
    FilterDecision,
    FilterResult,
    RelevanceFilter,
)


@dataclass
class CalibrationReport:
    """Report from threshold calibration analysis."""

    total_tenders: int
    score_distribution: Dict[str, float]  # mean, std, p25, p50, p75
    current_thresholds: Dict[str, float]
    tier_counts: Dict[str, int]
    suggested_thresholds: Dict[str, float]
    analysis_notes: List[str]

    def to_dict(self) -> Dict:
        return {
            "total_tenders": self.total_tenders,
            "score_distribution": self.score_distribution,
            "current_thresholds": self.current_thresholds,
            "tier_counts": self.tier_counts,
            "suggested_thresholds": self.suggested_thresholds,
            "analysis_notes": self.analysis_notes,
        }


class ThresholdCalibrator:
    """
    Analyzes score distributions and suggests optimal thresholds.

    Strategy:
        - If too many tenders are RELEVANT (>50%), raise the threshold
        - If too few tenders are RELEVANT (<10%), lower the threshold
        - Use percentile-based suggestions as a starting point
    """

    def __init__(self, relevance_filter: RelevanceFilter):
        self.filter = relevance_filter

    def analyze_distribution(
        self,
        batch_result: FilterBatchResult,
    ) -> CalibrationReport:
        """
        Analyze the score distribution of a batch result
        and suggest threshold adjustments.

        Args:
            batch_result: Output from RelevanceFilter.filter_tenders().

        Returns:
            CalibrationReport with statistics and suggestions.
        """
        if not batch_result.results:
            return CalibrationReport(
                total_tenders=0,
                score_distribution={},
                current_thresholds={},
                tier_counts={},
                suggested_thresholds={},
                analysis_notes=["No tenders to analyze."],
            )

        scores = np.array([r.final_score for r in batch_result.results])

        # Distribution statistics
        distribution = {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "p25": round(float(np.percentile(scores, 25)), 4),
            "p50_median": round(float(np.percentile(scores, 50)), 4),
            "p75": round(float(np.percentile(scores, 75)), 4),
            "p90": round(float(np.percentile(scores, 90)), 4),
        }

        current = {
            "relevant": self.filter.relevant_threshold,
            "low_relevance": self.filter.low_relevance_threshold,
        }

        tier_counts = {
            "relevant": batch_result.relevant_count,
            "low_relevance": batch_result.low_relevance_count,
            "irrelevant": batch_result.irrelevant_count,
        }

        # Suggest thresholds
        suggested, notes = self._suggest_thresholds(
            scores, batch_result, distribution
        )

        return CalibrationReport(
            total_tenders=batch_result.total_tenders,
            score_distribution=distribution,
            current_thresholds=current,
            tier_counts=tier_counts,
            suggested_thresholds=suggested,
            analysis_notes=notes,
        )

    def _suggest_thresholds(
        self,
        scores: np.ndarray,
        batch: FilterBatchResult,
        dist: Dict[str, float],
    ) -> Tuple[Dict[str, float], List[str]]:
        """Compute suggested thresholds based on distribution."""
        notes = []
        total = batch.total_tenders
        relevant_pct = batch.relevant_count / total * 100

        # Start with current thresholds
        suggested_relevant = self.filter.relevant_threshold
        suggested_low = self.filter.low_relevance_threshold

        # Rule 1: Too many RELEVANT (>50%) → raise threshold
        if relevant_pct > 50:
            suggested_relevant = round(float(np.percentile(scores, 60)), 2)
            notes.append(
                f"Too many RELEVANT ({relevant_pct:.0f}%). "
                f"Suggest raising threshold to {suggested_relevant}"
            )

        # Rule 2: Too few RELEVANT (<10%) → lower threshold
        elif relevant_pct < 10 and total >= 5:
            suggested_relevant = round(float(np.percentile(scores, 80)), 2)
            notes.append(
                f"Too few RELEVANT ({relevant_pct:.0f}%). "
                f"Suggest lowering threshold to {suggested_relevant}"
            )
        else:
            notes.append(
                f"RELEVANT rate ({relevant_pct:.0f}%) looks reasonable."
            )

        # Low relevance threshold: target ~25th percentile
        suggested_low = round(float(np.percentile(scores, 25)), 2)

        # Sanity: low < relevant
        if suggested_low >= suggested_relevant:
            suggested_low = round(suggested_relevant - 0.15, 2)
            notes.append("Adjusted low_relevance threshold to maintain gap.")

        # Score spread check
        if dist["std"] < 0.05:
            notes.append(
                "⚠ Very low score variance (std={:.4f}). "
                "Scores may not differentiate well between tenders. "
                "Consider enriching the company profile.".format(dist["std"])
            )

        return {
            "relevant": max(0.1, suggested_relevant),
            "low_relevance": max(0.05, suggested_low),
        }, notes

    def calibrate_on_labeled(
        self,
        tenders: List[Dict],
        labels: List[bool],
        threshold_range: Tuple[float, float] = (0.30, 0.90),
        steps: int = 25,
    ) -> Dict:
        """
        Find the optimal threshold by sweeping over a range
        and computing precision/recall on labeled data.

        Args:
            tenders: List of tender dicts.
            labels:  True = relevant, False = irrelevant.
            threshold_range: (min_threshold, max_threshold).
            steps:   Number of threshold values to test.

        Returns:
            Dict with optimal threshold and metrics.
        """
        batch_result = self.filter.filter_tenders(tenders, sort_by_score=False)
        scores = [r.final_score for r in batch_result.results]

        thresholds = np.linspace(
            threshold_range[0], threshold_range[1], steps
        )

        best_f1 = 0.0
        best_threshold = 0.5
        metrics_history = []

        for t in thresholds:
            tp = sum(1 for s, l in zip(scores, labels) if s >= t and l)
            fp = sum(1 for s, l in zip(scores, labels) if s >= t and not l)
            fn = sum(1 for s, l in zip(scores, labels) if s < t and l)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics_history.append({
                "threshold": round(float(t), 3),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            })

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        return {
            "optimal_threshold": round(best_threshold, 3),
            "best_f1": round(best_f1, 4),
            "metrics_history": metrics_history,
        }
