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
    relevant_threshold=0.65,
    low_relevance_threshold=0.40,
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
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
        threshold: float = 0.65,
        low_threshold: float = 0.40,
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
        else:
            self._filter = relevance_filter
            self._filter.update_thresholds(
                relevant=threshold, low_relevance=low_threshold
            )

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

            # Build result object merging relevance + extraction
            result = {
                "id": tender.get("id", ""),
                "title": tender.get("title", ""),
                "platform": tender.get("platform", "Unknown"),
                "deadline": extraction.deadline or tender.get("deadline", "N/A"),
                "budget": extraction.budget or tender.get("budget", "N/A"),
                "budget_amount": extraction.budget_amount,
                "budget_currency": extraction.budget_currency,
                "category": tender.get("category", "Unknown"),
                # Relevance filter results
                "relevance_score": round(filter_result.final_score * 100, 2),
                "is_relevant": filter_result.decision == FilterDecision.RELEVANT,
                "decision": str(filter_result.decision),
                "semantic_similarity": filter_result.semantic_similarity,
                "skill_overlap": filter_result.skill_overlap,
                "domain_similarity": filter_result.domain_similarity,
                "best_matching_domain": filter_result.best_matching_domain,
                "matched_skills": filter_result.matched_skills,
                "missing_skills": filter_result.missing_skills,
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
