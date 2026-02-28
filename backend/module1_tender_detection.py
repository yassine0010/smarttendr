"""
SmartTender AI - Module 1: Smart Tender Detection
==================================================
Features:
- Multi-tier Web Scraping pipeline (API → RSS → HTML → Browser)
  Platforms: SAM.gov, TED Europa, UNGM, TUNEPS, Contracts Finder UK
- Keyword extraction (spaCy + TF-IDF)
- Semantic embedding (Sentence-BERT)
- Relevance filtering (cosine similarity + skill overlap + domain match)
- Three-tier decision: RELEVANT / LOW_RELEVANCE / IRRELEVANT

Architecture: See docs/SCRAPING_ARCHITECTURE.md for full design.
              See docs/RELEVANCE_FILTERING_ARCHITECTURE.md for scoring.

Author: SmartTender AI Team
Date: 2025-2026
"""

import re
import sys
import spacy
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any, Optional

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).parent))

from scraping.pipeline import ScrapingPipeline
from scraping.base import NormalizedTender
from nlp.keyword_extraction import KeywordExtractor, ExtractionResult
from relevance.filter_engine import RelevanceFilter, FilterResult, FilterDecision, FilterBatchResult
from relevance.company_profile import CompanyProfile
from relevance.strategic_evaluator import StrategicEvaluator, StrategicResult

# ================================================================
# LOAD NLP MODELS
# ================================================================
print("[Module 1] Loading NLP models...")

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("  -> Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize keyword extraction pipeline (reuses the same spaCy model)
keyword_extractor = KeywordExtractor(model_name="en_core_web_sm", top_keywords=30)

# Initialize relevance filter (loads SBERT model + precomputes profile embeddings)
print("[Module 1] Loading relevance filter (multilingual SBERT + company profile)...")
relevance_filter = RelevanceFilter(
    profile=CompanyProfile.default(),
    relevant_threshold=0.55,
    low_relevance_threshold=0.30,
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)

# Initialize strategic evaluator (lightweight — no model loading)
strategic_evaluator = StrategicEvaluator(profile=CompanyProfile.default())
print("[Module 1] Models loaded successfully!")


# ================================================================
# TENDER DETECTOR CLASS
# ================================================================
class TenderDetector:
    """
    Smart Tender Detection Engine

    This class provides functionality to:
    1. Extract keywords from tender documents (spaCy NER + TF-IDF)
    2. Compute relevance using the production RelevanceFilter
       (SBERT cosine similarity + skill overlap + domain match)
    3. Classify tenders: RELEVANT / LOW_RELEVANCE / IRRELEVANT
    4. Rank and filter tenders by final hybrid score
    """

    def __init__(
        self,
        company_profile: Optional[CompanyProfile] = None,
        threshold: float = 0.55,
        low_threshold: float = 0.30,
    ):
        """
        Initialize the TenderDetector.

        Args:
            company_profile: CompanyProfile object (uses default if None)
            threshold:       Score >= this → RELEVANT
            low_threshold:   Score >= this (< threshold) → LOW_RELEVANCE
        """
        self.threshold = threshold
        self.low_threshold = low_threshold

        # Use the module-level relevance filter, or create a custom one
        if company_profile is not None:
            self._filter = RelevanceFilter(
                profile=company_profile,
                relevant_threshold=threshold,
                low_relevance_threshold=low_threshold,
            )
            self._evaluator = StrategicEvaluator(profile=company_profile)
        else:
            self._filter = relevance_filter
            self._filter.update_thresholds(
                relevant=threshold, low_relevance=low_threshold
            )
            self._evaluator = strategic_evaluator

    # ============================================================
    # KEYWORD EXTRACTION (powered by NLP pipeline)
    # ============================================================
    def extract_keywords(
        self,
        text: str,
        title: str = "",
        existing_metadata: Optional[Dict] = None,
    ) -> ExtractionResult:
        """
        Extract structured fields from tender text using the
        production NLP pipeline (spaCy NER + TF-IDF + taxonomy).

        Args:
            text: The tender description text
            title: Tender title (provides NER context)
            existing_metadata: Fallback fields from scraper

        Returns:
            ExtractionResult with domain, skills, budget, deadline,
            organization, location, and ranked keywords.
        """
        return keyword_extractor.extract(
            text,
            title=title,
            existing_metadata=existing_metadata,
        )

    # ============================================================
    # RELEVANCE SCORING (powered by RelevanceFilter)
    # ============================================================
    def compute_relevance(self, tender: Dict) -> FilterResult:
        """
        Compute relevance score between tender and company profile.

        Uses the production RelevanceFilter which computes:
        - Semantic similarity (45%): SBERT cosine similarity
        - Skill overlap (35%): Jaccard-style skill matching
        - Domain match (20%): Multi-domain vector alignment

        Args:
            tender: Dictionary containing tender information

        Returns:
            FilterResult with scores, decision, and diagnostics.
        """
        return self._filter.filter_tender(tender)

    # ============================================================
    # ANALYZE TENDERS
    # ============================================================
    def analyze_tenders(
        self,
        tenders: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Analyze a list of tenders and return ranked results.

        Pipeline per tender:
            1. NLP keyword extraction (spaCy NER + TF-IDF)
            2. Relevance filtering (SBERT cosine + skill overlap + domain)
            3. Merge extracted fields into structured result

        Args:
            tenders: List of tender dictionaries

        Returns:
            List of analysis results, sorted by relevance score
        """
        results = []

        for tender in tenders:
            # Run NLP keyword extraction pipeline
            existing_meta = {
                "budget": tender.get("budget"),
                "deadline": tender.get("deadline"),
                "organization": tender.get("organization"),
                "location": tender.get("location"),
            }
            extraction: ExtractionResult = self.extract_keywords(
                tender.get("description", ""),
                title=tender.get("title", ""),
                existing_metadata=existing_meta,
            )

            # Enrich tender with NLP-detected fields for the relevance filter
            enriched = {
                **tender,
                "detected_domain": extraction.domain,
                "detected_skills": [s["name"] for s in extraction.skills],
                "detected_certifications": extraction.certifications,
            }

            # Compute relevance using the production filter
            filter_result: FilterResult = self.compute_relevance(enriched)

            # Compute strategic evaluation (additive layer)
            strategic_result: StrategicResult = self._evaluator.evaluate(
                enriched, filter_result
            )

            # Build result object merging relevance + extraction + strategic
            score_pct = round(filter_result.final_score * 100, 2)
            decision_str = str(filter_result.decision)

            # Generate human-readable score explanation (text)
            score_explanation = self._explain_score(
                score_pct=score_pct,
                decision=decision_str,
                semantic=filter_result.semantic_similarity,
                skill=filter_result.skill_overlap,
                domain=filter_result.domain_similarity,
                best_domain=filter_result.best_matching_domain,
                matched_skills=filter_result.matched_skills,
                missing_skills=filter_result.missing_skills,
                strategic=strategic_result,
                tender=tender,
            )

            # Generate structured explanation dict for rich UI
            score_explanation_detail = self._explain_score_structured(
                score_pct=score_pct,
                decision=decision_str,
                semantic=filter_result.semantic_similarity,
                skill=filter_result.skill_overlap,
                domain=filter_result.domain_similarity,
                best_domain=filter_result.best_matching_domain,
                matched_skills=filter_result.matched_skills,
                missing_skills=filter_result.missing_skills,
                strategic=strategic_result,
                tender=tender,
            )

            result = {
                "id": tender.get("id", ""),
                "title": tender.get("title", ""),
                "url": tender.get("url", "") or tender.get("source_url", ""),
                "platform": tender.get("platform", "Unknown"),
                "deadline": extraction.deadline or tender.get("deadline", "N/A"),
                "budget": extraction.budget or tender.get("budget", "N/A"),
                "budget_amount": extraction.budget_amount,
                "budget_currency": extraction.budget_currency,
                "category": tender.get("category", "Unknown"),
                # Relevance filter results
                "relevance_score": score_pct,
                "is_relevant": filter_result.decision == FilterDecision.RELEVANT,
                "decision": decision_str,
                "score_explanation": score_explanation,
                "score_explanation_detail": score_explanation_detail,
                "semantic_similarity": filter_result.semantic_similarity,
                "skill_overlap": filter_result.skill_overlap,
                "domain_similarity": filter_result.domain_similarity,
                "best_matching_domain": filter_result.best_matching_domain,
                "matched_skills": filter_result.matched_skills,
                "missing_skills": filter_result.missing_skills,
                # Strategic evaluation results
                "strategic_relevance_score": strategic_result.strategic_relevance_score,
                "win_probability": strategic_result.win_probability,
                "difficulty_level": str(strategic_result.difficulty_level),
                "competition_intensity": str(strategic_result.competition_intensity),
                "deadline_risk": str(strategic_result.deadline_risk),
                "days_remaining": strategic_result.days_remaining,
                "complexity_score": strategic_result.complexity_score,
                "score_breakdown": strategic_result.score_breakdown,
                # NLP extraction results
                "nlp_extraction": extraction.to_dict(),
                "detected_domain": extraction.domain,
                "detected_skills": [s["name"] for s in extraction.skills],
                "detected_certifications": extraction.certifications,
                "organization": extraction.organization or tender.get("organization", ""),
                "location": extraction.location or tender.get("location", ""),
                "top_keywords": [kw["term"] for kw in extraction.top_keywords[:10]],
                # Legacy fields for backward compat
                "required_skills": tender.get("required_skills", []),
                "analysis_timestamp": str(np.datetime64('now'))
            }
            results.append(result)

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return results

    # ============================================================
    # SCORE EXPLANATION GENERATOR
    # ============================================================

    @staticmethod
    def _explain_score(
        score_pct: float,
        decision: str,
        semantic: float,
        skill: float,
        domain: float,
        best_domain: str,
        matched_skills: list,
        missing_skills: list,
        strategic,
        tender: dict,
    ) -> str:
        """
        Generate a human-readable explanation of why a tender
        received its relevance score and decision.
        """
        parts = []

        # ── Overall verdict ──
        if decision == "RELEVANT":
            parts.append(
                f"✅ This tender scores {score_pct:.0f}% and is classified as RELEVANT — "
                f"it closely matches Inetum's expertise."
            )
        elif decision == "LOW_RELEVANCE":
            parts.append(
                f"⚠️ This tender scores {score_pct:.0f}% and is classified as LOW RELEVANCE — "
                f"there is partial alignment with Inetum's capabilities but gaps remain."
            )
        else:
            parts.append(
                f"❌ This tender scores {score_pct:.0f}% and is classified as IRRELEVANT — "
                f"it falls outside Inetum's core service areas."
            )

        # ── Domain alignment ──
        if domain > 0.50:
            parts.append(
                f"📌 Strong domain match: the tender aligns well with our "
                f"'{best_domain}' practice (domain similarity {domain:.0%})."
            )
        elif domain > 0.35:
            parts.append(
                f"📌 Moderate domain match: the tender partially aligns with our "
                f"'{best_domain}' practice (domain similarity {domain:.0%})."
            )
        else:
            parts.append(
                f"📌 Weak domain match: the tender's subject area does not clearly "
                f"match any of our core domains (best match: '{best_domain}' at {domain:.0%})."
            )

        # ── Semantic similarity ──
        if semantic > 0.45:
            parts.append(
                f"🔤 The tender description is semantically close to our company profile "
                f"(similarity {semantic:.0%})."
            )
        elif semantic > 0.30:
            parts.append(
                f"🔤 Moderate semantic similarity ({semantic:.0%}) — the tender uses "
                f"some terminology related to our services."
            )
        else:
            parts.append(
                f"🔤 Low semantic similarity ({semantic:.0%}) — the tender's language "
                f"is distant from our service descriptions. This may be due to a "
                f"non-English tender text or a different industry focus."
            )

        # ── Skills ──
        if matched_skills:
            parts.append(
                f"🛠️ Matched skills: {', '.join(matched_skills[:8])}. "
                f"We have direct expertise in {len(matched_skills)} "
                f"of the required technologies."
            )
        elif skill >= 0.6:
            parts.append(
                "🛠️ No specific skills were extracted from the tender text, "
                "but IT-related keywords suggest this is within our domain."
            )
        else:
            parts.append(
                "🛠️ No matching skills detected. The tender either requires "
                "skills outside our portfolio or lacks detailed technical requirements."
            )

        if missing_skills:
            parts.append(
                f"⚠️ Missing skills: {', '.join(missing_skills[:5])}. "
                f"These would need to be addressed through partnerships or hiring."
            )

        # ── Strategic factors ──
        risk = str(strategic.deadline_risk)
        win = strategic.win_probability
        difficulty = str(strategic.difficulty_level)

        if win >= 70:
            parts.append(f"🏆 High win probability ({win}%) — strong competitive position.")
        elif win >= 50:
            parts.append(f"🏆 Moderate win probability ({win}%) — competitive but not guaranteed.")
        else:
            parts.append(f"🏆 Low win probability ({win}%) — significant competitive challenges.")

        if risk == "HIGH":
            days = strategic.days_remaining
            parts.append(
                f"⏰ HIGH deadline risk — only {days} days remaining. "
                f"Immediate action required if pursuing."
            )
        elif risk == "MEDIUM":
            parts.append(
                f"⏰ Medium deadline risk — sufficient time to prepare a quality bid."
            )

        if difficulty in ("HIGH", "VERY_HIGH"):
            parts.append(
                f"📊 {difficulty.replace('_', ' ').title()} difficulty level — "
                f"consider resource allocation carefully."
            )

        # ── Recommendation ──
        if decision == "RELEVANT" and win >= 60:
            parts.append("💡 Recommendation: PURSUE — strong fit and good win probability.")
        elif decision == "RELEVANT":
            parts.append("💡 Recommendation: REVIEW — good fit but assess competitive landscape.")
        elif decision == "LOW_RELEVANCE" and score_pct >= 50:
            parts.append(
                "💡 Recommendation: REVIEW with caution — partial fit, evaluate "
                "if gaps can be filled through partnerships."
            )
        elif decision == "LOW_RELEVANCE":
            parts.append("💡 Recommendation: LOW PRIORITY — limited alignment with our strengths.")
        else:
            parts.append("💡 Recommendation: SKIP — this tender is not a good match.")

        return " ".join(parts)

    # ============================================================
    # STRUCTURED SCORE EXPLANATION (for rich UI)
    # ============================================================

    @staticmethod
    def _explain_score_structured(
        score_pct: float,
        decision: str,
        semantic: float,
        skill: float,
        domain: float,
        best_domain: str,
        matched_skills: list,
        missing_skills: list,
        strategic,
        tender: dict,
    ) -> dict:
        """
        Return a structured dict for the UI to render a rich,
        section-by-section explanation card.
        """
        # ── Verdict ──
        if decision == "RELEVANT":
            verdict = {
                "icon": "✅",
                "label": "RELEVANT",
                "color": "#28a745",
                "summary": f"This tender scores {score_pct:.0f}% and closely matches Inetum's expertise.",
            }
        elif decision == "LOW_RELEVANCE":
            verdict = {
                "icon": "⚠️",
                "label": "LOW RELEVANCE",
                "color": "#ffc107",
                "summary": f"This tender scores {score_pct:.0f}% — partial alignment with Inetum's capabilities but gaps remain.",
            }
        else:
            verdict = {
                "icon": "❌",
                "label": "IRRELEVANT",
                "color": "#dc3545",
                "summary": f"This tender scores {score_pct:.0f}% — it falls outside Inetum's core service areas.",
            }

        # ── Domain alignment ──
        if domain > 0.50:
            domain_info = {
                "level": "strong",
                "icon": "🟢",
                "text": f"Strong domain match with our '{best_domain}' practice.",
                "value": round(domain, 3),
                "best_domain": best_domain,
            }
        elif domain > 0.35:
            domain_info = {
                "level": "moderate",
                "icon": "🟡",
                "text": f"Moderate domain match with our '{best_domain}' practice.",
                "value": round(domain, 3),
                "best_domain": best_domain,
            }
        else:
            domain_info = {
                "level": "weak",
                "icon": "🔴",
                "text": f"Weak domain match. Best match: '{best_domain}'.",
                "value": round(domain, 3),
                "best_domain": best_domain,
            }

        # ── Semantic similarity ──
        if semantic > 0.45:
            semantic_info = {
                "level": "high",
                "icon": "🟢",
                "text": "Tender description is semantically close to our company profile.",
                "value": round(semantic, 3),
            }
        elif semantic > 0.30:
            semantic_info = {
                "level": "moderate",
                "icon": "🟡",
                "text": "Moderate semantic similarity — some terminology related to our services.",
                "value": round(semantic, 3),
            }
        else:
            semantic_info = {
                "level": "low",
                "icon": "🔴",
                "text": "Low semantic similarity — may be non-English or different industry.",
                "value": round(semantic, 3),
            }

        # ── Skills ──
        if matched_skills:
            skill_info = {
                "level": "matched",
                "icon": "🟢",
                "text": f"Direct expertise in {len(matched_skills)} required technologies.",
                "matched": list(matched_skills[:10]),
                "missing": list(missing_skills[:8]) if missing_skills else [],
                "value": round(skill, 3),
            }
        elif skill >= 0.6:
            skill_info = {
                "level": "inferred",
                "icon": "🟡",
                "text": "No specific skills extracted, but IT-related keywords suggest our domain.",
                "matched": [],
                "missing": list(missing_skills[:8]) if missing_skills else [],
                "value": round(skill, 3),
            }
        else:
            skill_info = {
                "level": "none",
                "icon": "🔴",
                "text": "No matching skills detected — outside our portfolio or lacks requirements.",
                "matched": [],
                "missing": list(missing_skills[:8]) if missing_skills else [],
                "value": round(skill, 3),
            }

        # ── Strategic factors ──
        win = strategic.win_probability
        risk = str(strategic.deadline_risk)
        difficulty = str(strategic.difficulty_level)
        competition = str(strategic.competition_intensity)
        days = strategic.days_remaining

        if win >= 70:
            win_info = {"level": "high", "icon": "🟢", "text": f"Strong competitive position ({win}%)."}
        elif win >= 50:
            win_info = {"level": "moderate", "icon": "🟡", "text": f"Competitive but not guaranteed ({win}%)."}
        else:
            win_info = {"level": "low", "icon": "🔴", "text": f"Significant competitive challenges ({win}%)."}

        strategic_info = {
            "win_probability": win,
            "win_info": win_info,
            "deadline_risk": risk,
            "days_remaining": days,
            "difficulty": difficulty,
            "competition": competition,
        }

        # ── Recommendation ──
        if decision == "RELEVANT" and win >= 60:
            recommendation = {
                "action": "PURSUE",
                "icon": "🚀",
                "color": "#28a745",
                "text": "Strong fit and good win probability. Recommended to pursue.",
            }
        elif decision == "RELEVANT":
            recommendation = {
                "action": "REVIEW",
                "icon": "👀",
                "color": "#17a2b8",
                "text": "Good fit — assess competitive landscape before committing.",
            }
        elif decision == "LOW_RELEVANCE" and score_pct >= 50:
            recommendation = {
                "action": "REVIEW WITH CAUTION",
                "icon": "⚖️",
                "color": "#ffc107",
                "text": "Partial fit — evaluate if gaps can be filled through partnerships.",
            }
        elif decision == "LOW_RELEVANCE":
            recommendation = {
                "action": "LOW PRIORITY",
                "icon": "📋",
                "color": "#6c757d",
                "text": "Limited alignment with our strengths.",
            }
        else:
            recommendation = {
                "action": "SKIP",
                "icon": "⏭️",
                "color": "#dc3545",
                "text": "This tender is not a good match for Inetum.",
            }

        return {
            "verdict": verdict,
            "domain": domain_info,
            "semantic": semantic_info,
            "skills": skill_info,
            "strategic": strategic_info,
            "recommendation": recommendation,
        }

    # ============================================================
    # WEB SCRAPING (production pipeline)
    # ============================================================
    @staticmethod
    def scrape(
        platform: str = "ALL",
        query: str = "IT services",
        sources: List[str] | None = None,
    ) -> List[Dict]:
        """
        Scrape real tenders using the multi-tier scraping pipeline.

        The pipeline automatically selects the best strategy per source:
        - Tier 1 (API): SAM.gov, Contracts Finder UK
        - Tier 2 (RSS): TED Europa RSS feeds
        - Tier 3 (HTML): UNGM, static government portals
        - Tier 4 (Browser): TUNEPS (JS-rendered)

        Args:
            platform: "ALL", or specific source name (e.g. "SAM.GOV", "TED")
            query: Search keyword for platforms that support it
            sources: Explicit list of source names to scrape

        Returns:
            List of tender dictionaries (legacy format)
        """
        pipeline = ScrapingPipeline()

        # Map platform shorthand to source names
        if platform.upper() != "ALL" and not sources:
            platform_map = {
                "TUNEPS": ["TUNEPS"],
                "TED": ["TED"],
                "TED EUROPA": ["TED"],
                "SAM": ["SAM.GOV"],
                "SAM.GOV": ["SAM.GOV"],
                "UNGM": ["UNGM"],
                "CONTRACTS_FINDER": ["CONTRACTS_FINDER"],
                "CF": ["CONTRACTS_FINDER"],
            }
            sources = platform_map.get(platform.upper(), None)

        # Run the pipeline
        fallback_file = Path(__file__).parent.parent / "data" / "sample_tenders.json"
        normalized_tenders = pipeline.run(
            query=query,
            sources=sources,
            fallback_file=fallback_file,
        )

        # Convert to legacy dict format for backward compatibility
        return [t.to_legacy_dict() for t in normalized_tenders]

    # ============================================================
    # MOCK SCRAPER (fallback for demonstration)
    # ============================================================
    @staticmethod
    def mock_scrape(platform: str = "TUNEPS") -> List[Dict]:
        """
        Simulate scraping a tender platform.
        In production, this would use actual web scraping.

        Args:
            platform: Name of the platform to scrape

        Returns:
            List of tender dictionaries
        """
        # Import sample data
        from utils import load_json
        all_tenders = load_json("sample_tenders.json")

        # Filter by platform if specified
        if platform.upper() != "ALL":
            filtered = [
                t for t in all_tenders
                if t.get("platform", "").upper() == platform.upper()
            ]
            return filtered if filtered else all_tenders

        return all_tenders


# ================================================================
# STANDALONE EXECUTION
# ================================================================
if __name__ == "__main__":
    import json
    from pathlib import Path

    # Initialize detector with production thresholds
    detector = TenderDetector(threshold=0.65, low_threshold=0.40)

    # ----------------------------------------------------------
    # Step 1: Run the multi-tier scraping pipeline
    # ----------------------------------------------------------
    print("\n[Step 1] Running multi-tier scraping pipeline...")
    print("         Sources: SAM.gov (API) | TED (RSS) | UNGM (HTML) | TUNEPS (Browser) | Contracts Finder (API)")
    scraped_tenders = detector.scrape(platform="ALL", query="IT services")

    # ----------------------------------------------------------
    # Step 2: Use scraped data or fallback to sample data
    # ----------------------------------------------------------
    if scraped_tenders:
        tenders = scraped_tenders
        print(f"\n[OK] Using {len(tenders)} tenders from scraping pipeline")
    else:
        print("\n[Info] No tenders scraped — using sample data as fallback")
        data_dir = Path(__file__).parent.parent / "data"
        sample_file = data_dir / "sample_tenders.json"

        if sample_file.exists():
            with open(sample_file, "r", encoding="utf-8") as f:
                tenders = json.load(f)
        else:
            tenders = [{
                "id": "T001",
                "title": "Cloud ERP System Development",
                "description": "Development of cloud-based ERP using Python, AWS, Docker",
                "required_skills": ["Python", "AWS", "Docker", "PostgreSQL"],
                "platform": "TUNEPS",
                "deadline": "2025-03-01",
                "category": "IT Services"
            }]

    # ----------------------------------------------------------
    # Step 3: Analyze & rank tenders with NLP + relevance engine
    # ----------------------------------------------------------
    print(f"\n[Step 2] Analyzing {len(tenders)} tenders (NLP + SBERT cosine similarity)...")
    results = detector.analyze_tenders(tenders)

    # ----------------------------------------------------------
    # Step 4: Display results with three-tier classification
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(" SMARTTENDER AI - TENDER DETECTION RESULTS")
    print("=" * 70)

    decision_icons = {
        "RELEVANT": "✅",
        "LOW_RELEVANCE": "🔶",
        "IRRELEVANT": "⬜",
    }

    for r in results:
        decision = r.get("decision", "IRRELEVANT")
        icon = decision_icons.get(decision, "⬜")

        print(f"\n{icon} {r['title'][:65]}")
        print(f"   Score: {r['relevance_score']}% | Decision: {decision}")
        print(f"   Semantic: {r.get('semantic_similarity', 0):.3f} | "
              f"Skills: {r.get('skill_overlap', 0):.3f} | "
              f"Domain: {r.get('domain_similarity', 0):.3f}")
        print(f"   Strategic: {r.get('strategic_relevance_score', 0)}% | "
              f"Win Prob: {r.get('win_probability', 0)}% | "
              f"Difficulty: {r.get('difficulty_level', 'N/A')}")
        print(f"   Deadline Risk: {r.get('deadline_risk', 'N/A')} | "
              f"Days Left: {r.get('days_remaining', 'N/A')} | "
              f"Competition: {r.get('competition_intensity', 'N/A')}")
        print(f"   Platform: {r['platform']} | Deadline: {r['deadline']}")
        print(f"   Best Domain: {r.get('best_matching_domain', 'N/A')} | Budget: {r.get('budget', 'N/A')}")
        if r.get('organization'):
            print(f"   Organization: {r['organization']}")
        if r.get('matched_skills'):
            print(f"   ✓ Matched Skills: {', '.join(r['matched_skills'][:8])}")
        if r.get('missing_skills'):
            print(f"   ✗ Missing Skills: {', '.join(r['missing_skills'][:5])}")
        if r.get('detected_certifications'):
            print(f"   Certifications: {', '.join(r['detected_certifications'][:5])}")
        if r.get('top_keywords'):
            print(f"   Keywords: {', '.join(r['top_keywords'][:6])}")

    print("\n" + "=" * 70)
    relevant_count = sum(1 for r in results if r.get("decision") == "RELEVANT")
    low_count = sum(1 for r in results if r.get("decision") == "LOW_RELEVANCE")
    irrelevant_count = sum(1 for r in results if r.get("decision") == "IRRELEVANT")
    print(f" Summary: {relevant_count} RELEVANT | {low_count} LOW_RELEVANCE | "
          f"{irrelevant_count} IRRELEVANT  (total: {len(results)})")
    print(f" Thresholds: RELEVANT ≥ {detector.threshold:.0%} | "
          f"LOW_RELEVANCE ≥ {detector.low_threshold:.0%}")
    print("=" * 70)

    # ----------------------------------------------------------
    # Step 5: Save analysis results
    # ----------------------------------------------------------
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    analysis_file = output_dir / "analysis_results.json"

    # Remove non-serializable fields for JSON output
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items()}
        serializable_results.append(sr)

    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[Saved] Analysis results → {analysis_file}")
