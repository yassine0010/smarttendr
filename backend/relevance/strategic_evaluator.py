"""
SmartTender AI — Strategic Evaluation Layer
=============================================
Adds three computed outputs ON TOP of the existing relevance
filtering system (which remains untouched):

    1. strategic_relevance_score (0–100%)
    2. win_probability           (0–100%)
    3. deadline_risk             ("LOW" / "MEDIUM" / "HIGH")

Design principles:
    - Fully deterministic (no ML training, no randomness)
    - Pure Python (no external services)
    - Additive layer — does NOT modify existing hybrid scoring
    - Uses only fields already present in FilterResult + enriched tender

Author: SmartTender AI Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from backend.relevance.company_profile import CompanyProfile
from backend.relevance.filter_engine import FilterResult


# ================================================================
# ENUMS
# ================================================================

class DeadlineRisk(str, Enum):
    """Deadline urgency classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    def __str__(self) -> str:
        return self.value


class DifficultyLevel(str, Enum):
    """How hard this tender is for the company to deliver."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    def __str__(self) -> str:
        return self.value


class CompetitionIntensity(str, Enum):
    """Expected competition level on this tender."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    def __str__(self) -> str:
        return self.value


# ================================================================
# STRATEGIC RESULT (output dataclass)
# ================================================================

@dataclass
class StrategicResult:
    """
    Output of the strategic evaluation layer.

    All fields are deterministic and JSON-serializable.
    This is returned alongside (not replacing) FilterResult.
    """

    # Core computed outputs
    strategic_relevance_score: int       # 0–100
    win_probability: int                 # 0–100
    deadline_risk: DeadlineRisk

    # Supporting metrics
    days_remaining: int                  # calendar days until deadline
    complexity_score: int                # required + missing skills count
    difficulty_level: DifficultyLevel
    competition_intensity: CompetitionIntensity

    # Score breakdown (transparency)
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to plain dict for JSON output."""
        return {
            "strategic_relevance_score": self.strategic_relevance_score,
            "win_probability": self.win_probability,
            "deadline_risk": str(self.deadline_risk),
            "days_remaining": self.days_remaining,
            "complexity_score": self.complexity_score,
            "difficulty_level": str(self.difficulty_level),
            "competition_intensity": str(self.competition_intensity),
            "score_breakdown": {
                k: round(v, 4) for k, v in self.score_breakdown.items()
            },
        }


# ================================================================
# STRATEGIC EVALUATOR
# ================================================================

class StrategicEvaluator:
    """
    Computes strategic_relevance_score, win_probability, and
    deadline_risk on top of the existing FilterResult.

    Does NOT modify any existing scoring logic.

    Usage:
        evaluator = StrategicEvaluator(profile=company_profile)
        result = evaluator.evaluate(tender_dict, filter_result)
    """

    # Weight allocation for strategic relevance score
    WEIGHTS = {
        "final_score": 0.40,
        "skill_coverage": 0.20,
        "domain_weight": 0.15,
        "budget_compatibility": 0.15,
        "geographic_match": 0.10,
    }

    # Default budget range (in USD-equivalent) for the company
    DEFAULT_BUDGET_RANGE = {
        "min": 50_000,
        "max": 5_000_000,
    }

    # Company operating regions (ISO alpha-2 or region names)
    DEFAULT_REGIONS: Set[str] = {
        "TN", "Tunisia", "Tunisie", "تونس",
        "FR", "France",
        "DZ", "Algeria", "Algérie",
        "MA", "Morocco", "Maroc",
        "Europe", "EU", "Africa",
    }

    # Generic / highly competitive domains
    GENERIC_DOMAINS: Set[str] = {
        "IT Services",
        "Consulting",
        "General",
        "Digital Transformation",
    }

    # Niche / specialized domains
    NICHE_DOMAINS: Set[str] = {
        "AI/Machine Learning",
        "Cybersecurity",
        "Data Analytics",
        "ERP",
    }

    def __init__(
        self,
        profile: Optional[CompanyProfile] = None,
        budget_range: Optional[Dict[str, int]] = None,
        regions: Optional[Set[str]] = None,
        today: Optional[date] = None,
    ):
        """
        Initialize the strategic evaluator.

        Args:
            profile:      Company profile (used for domain weights).
            budget_range: {"min": int, "max": int} in USD-equivalent.
            regions:      Set of country/region strings the company operates in.
            today:        Override today's date (for testing). Defaults to date.today().
        """
        self.profile = profile or CompanyProfile.default()
        self.budget_range = budget_range or self.DEFAULT_BUDGET_RANGE.copy()
        self.regions = {r.lower().strip() for r in (regions or self.DEFAULT_REGIONS)}
        self._today = today  # None means use date.today() at call time

    @property
    def today(self) -> date:
        """Current date (allows override for deterministic tests)."""
        return self._today or date.today()

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    def evaluate(
        self,
        tender: Dict,
        filter_result: FilterResult,
    ) -> StrategicResult:
        """
        Compute all three strategic outputs for a single tender.

        Args:
            tender:         Enriched tender dict (same as passed to RelevanceFilter).
            filter_result:  The FilterResult already computed by the existing engine.

        Returns:
            StrategicResult with strategic_relevance_score, win_probability,
            deadline_risk, and supporting metrics.
        """
        # ----- Component scores -----
        skill_coverage = self._compute_skill_coverage(
            filter_result.matched_skills,
            tender.get("detected_skills", []),
        )
        domain_weight = self._compute_domain_weight(
            filter_result.best_matching_domain,
        )
        budget_compat = self._compute_budget_compatibility(
            tender.get("budget_amount"),
        )
        geo_match = self._compute_geographic_match(
            tender.get("location", ""),
            tender.get("title", ""),
            tender.get("description", ""),
        )

        # ----- 1) Strategic Relevance Score (0–100) -----
        weighted_sum = (
            self.WEIGHTS["final_score"] * filter_result.final_score
            + self.WEIGHTS["skill_coverage"] * skill_coverage
            + self.WEIGHTS["domain_weight"] * domain_weight
            + self.WEIGHTS["budget_compatibility"] * budget_compat
            + self.WEIGHTS["geographic_match"] * geo_match
        )
        strategic_relevance_score = max(0, min(100, round(weighted_sum * 100)))

        # ----- 3) Deadline Risk (compute before win_probability) -----
        deadline_risk, days_remaining, complexity_score = self._compute_deadline_risk(
            tender.get("deadline"),
            tender.get("detected_skills", []),
            filter_result.missing_skills,
        )

        # ----- 2) Win Probability (0–100) -----
        win_probability, difficulty_level, competition_intensity = (
            self._compute_win_probability(
                strategic_relevance_score=strategic_relevance_score,
                filter_result=filter_result,
                budget_compat=budget_compat,
                deadline_risk=deadline_risk,
                best_domain=filter_result.best_matching_domain,
                geo_match=geo_match,
            )
        )

        # Build breakdown for transparency
        score_breakdown = {
            "final_score_component": filter_result.final_score * self.WEIGHTS["final_score"],
            "skill_coverage_component": skill_coverage * self.WEIGHTS["skill_coverage"],
            "domain_weight_component": domain_weight * self.WEIGHTS["domain_weight"],
            "budget_compat_component": budget_compat * self.WEIGHTS["budget_compatibility"],
            "geographic_match_component": geo_match * self.WEIGHTS["geographic_match"],
            "skill_coverage_raw": skill_coverage,
            "domain_weight_raw": domain_weight,
            "budget_compatibility_raw": budget_compat,
            "geographic_match_raw": geo_match,
        }

        return StrategicResult(
            strategic_relevance_score=strategic_relevance_score,
            win_probability=win_probability,
            deadline_risk=deadline_risk,
            days_remaining=days_remaining,
            complexity_score=complexity_score,
            difficulty_level=difficulty_level,
            competition_intensity=competition_intensity,
            score_breakdown=score_breakdown,
        )

    # ============================================================
    # BATCH EVALUATION
    # ============================================================

    def evaluate_batch(
        self,
        tenders: List[Dict],
        filter_results: List[FilterResult],
    ) -> List[StrategicResult]:
        """
        Evaluate a batch of tenders.

        Args:
            tenders:        List of enriched tender dicts (same order).
            filter_results: List of FilterResults (same order).

        Returns:
            List of StrategicResult (same order).
        """
        if len(tenders) != len(filter_results):
            raise ValueError(
                f"Mismatched lengths: {len(tenders)} tenders vs "
                f"{len(filter_results)} filter results"
            )
        return [
            self.evaluate(tender, fr)
            for tender, fr in zip(tenders, filter_results)
        ]

    # ============================================================
    # COMPONENT: Skill Coverage Ratio
    # ============================================================

    @staticmethod
    def _compute_skill_coverage(
        matched_skills: List[str],
        detected_skills: List[str],
    ) -> float:
        """
        skill_coverage = |matched| / |detected|

        Returns 0.5 (neutral) if no skills detected.
        """
        if not detected_skills:
            return 0.5
        return len(matched_skills) / len(detected_skills)

    # ============================================================
    # COMPONENT: Domain Weight Alignment
    # ============================================================

    def _compute_domain_weight(self, best_domain: str) -> float:
        """
        Returns the company's internal priority weight for the
        best-matching domain. Higher weight = more strategic fit.

        Falls back to 0.5 if domain not in profile.
        """
        return self.profile.domain_weight(best_domain)

    # ============================================================
    # COMPONENT: Budget Compatibility
    # ============================================================

    def _compute_budget_compatibility(
        self,
        budget_amount: Optional[float],
    ) -> float:
        """
        Budget compatibility:
            unknown  → 0.5 (neutral)
            within range → 1.0
            outside range → 0.3
        """
        if budget_amount is None or budget_amount <= 0:
            return 0.5

        bmin = self.budget_range["min"]
        bmax = self.budget_range["max"]

        if bmin <= budget_amount <= bmax:
            return 1.0
        else:
            return 0.3

    # ============================================================
    # COMPONENT: Geographic Match
    # ============================================================

    def _compute_geographic_match(
        self,
        location: str,
        title: str,
        description: str,
    ) -> float:
        """
        Geographic alignment:
            country in company regions → 1.0
            international tender (no specific country) → 0.7
            specific country not in regions → 0.4
        """
        location_lower = location.lower().strip() if location else ""

        # Check if location matches any of the company's operating regions
        if location_lower:
            for region in self.regions:
                if region in location_lower or location_lower in region:
                    return 1.0

        # Check if this looks like an international / multi-country tender
        combined = f"{title} {description} {location}".lower()
        international_signals = [
            "international", "global", "worldwide", "multi-country",
            "multiple countries", "multinational", "multi-national",
            "mondial", "international",  # FR
            "دولي", "عالمي",  # AR
        ]
        for signal in international_signals:
            if signal in combined:
                return 0.7

        # Specific location but not in our regions
        if location_lower:
            return 0.4

        # No location info at all — treat as neutral-international
        return 0.7

    # ============================================================
    # COMPONENT: Deadline Risk
    # ============================================================

    def _compute_deadline_risk(
        self,
        deadline_raw: Optional[str],
        detected_skills: List[str],
        missing_skills: List[str],
    ) -> tuple[DeadlineRisk, int, int]:
        """
        Deadline risk based on days remaining + complexity.

        Returns:
            (risk_level, days_remaining, complexity_score)
        """
        # Parse deadline
        days_remaining = self._parse_days_remaining(deadline_raw)

        # Complexity = required skills + missing skills
        num_required = len(detected_skills)
        num_missing = len(missing_skills)
        complexity_score = num_required + num_missing

        # Classify risk
        if days_remaining < 7 and complexity_score > 8:
            risk = DeadlineRisk.HIGH
        elif days_remaining < 14 or complexity_score > 10:
            risk = DeadlineRisk.MEDIUM
        else:
            risk = DeadlineRisk.LOW

        return risk, days_remaining, complexity_score

    def _parse_days_remaining(self, deadline_raw: Optional[str]) -> int:
        """
        Parse a deadline string into days remaining from today.

        Supports multiple formats:
            - ISO date: "2026-03-15"
            - European: "15/03/2026", "15.03.2026"
            - Long: "March 15, 2026"
            - With time: "2026-03-15T23:59:00"

        Returns 999 if unparseable (= no deadline pressure).
        """
        if not deadline_raw or deadline_raw in ("N/A", "Unknown", "None", ""):
            return 999

        # Clean the string
        deadline_str = str(deadline_raw).strip()

        # Try multiple date formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M",
            "%d/%m/%Y",
            "%d.%m.%Y",
            "%d-%m-%Y",
            "%m/%d/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]

        for fmt in formats:
            try:
                deadline_date = datetime.strptime(deadline_str, fmt).date()
                delta = (deadline_date - self.today).days
                return max(0, delta)  # 0 if already passed
            except ValueError:
                continue

        # Last resort: try to extract a date-like pattern
        import re
        iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", deadline_str)
        if iso_match:
            try:
                deadline_date = datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
                delta = (deadline_date - self.today).days
                return max(0, delta)
            except ValueError:
                pass

        return 999  # Unknown deadline = no time pressure

    # ============================================================
    # COMPONENT: Win Probability + Difficulty + Competition
    # ============================================================

    def _compute_win_probability(
        self,
        strategic_relevance_score: int,
        filter_result: FilterResult,
        budget_compat: float,
        deadline_risk: DeadlineRisk,
        best_domain: str,
        geo_match: float,
    ) -> tuple[int, DifficultyLevel, CompetitionIntensity]:
        """
        Heuristic win probability with adjustments.

        Base = strategic_relevance_score / 100

        Adjustments:
            +0.05 if skill_overlap > 0.75
            +0.05 if domain_similarity > 0.75
            -0.10 if missing_skills > 40% of detected
            -0.10 if budget outside ideal range
            -0.10 if deadline_risk == HIGH

        Returns:
            (win_probability_0_100, difficulty_level, competition_intensity)
        """
        base = strategic_relevance_score / 100.0

        # Positive adjustments
        if filter_result.skill_overlap > 0.75:
            base += 0.05
        if filter_result.domain_similarity > 0.75:
            base += 0.05

        # Negative adjustments
        total_skills = len(filter_result.matched_skills) + len(filter_result.missing_skills)
        if total_skills > 0:
            missing_ratio = len(filter_result.missing_skills) / total_skills
            if missing_ratio > 0.40:
                base -= 0.10

        if budget_compat < 0.5:
            base -= 0.10

        if deadline_risk == DeadlineRisk.HIGH:
            base -= 0.10

        # Clamp [0, 1]
        base = max(0.0, min(1.0, base))
        win_probability = round(base * 100)

        # Difficulty level (based on final_score from existing engine)
        if filter_result.final_score > 0.75:
            difficulty = DifficultyLevel.LOW
        elif filter_result.final_score >= 0.55:
            difficulty = DifficultyLevel.MEDIUM
        else:
            difficulty = DifficultyLevel.HIGH

        # Competition intensity
        competition = self._assess_competition(best_domain, geo_match)

        return win_probability, difficulty, competition

    def _assess_competition(
        self,
        best_domain: str,
        geo_match: float,
    ) -> CompetitionIntensity:
        """
        Estimate competition intensity:
            HIGH   — international + generic domain
            MEDIUM — regional scope
            LOW    — niche domain or local market
        """
        is_international = geo_match >= 0.7 and geo_match < 1.0
        is_local = geo_match >= 1.0
        is_generic = best_domain in self.GENERIC_DOMAINS
        is_niche = best_domain in self.NICHE_DOMAINS

        if is_international and is_generic:
            return CompetitionIntensity.HIGH
        elif is_niche and is_local:
            return CompetitionIntensity.LOW
        elif is_niche:
            return CompetitionIntensity.MEDIUM
        elif is_local:
            return CompetitionIntensity.MEDIUM
        else:
            return CompetitionIntensity.HIGH
