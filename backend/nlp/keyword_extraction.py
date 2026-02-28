"""
SmartTender AI — Keyword Extraction Pipeline (Orchestrator)
=============================================================
The main public API for the NLP keyword extraction module.

Combines all sub-modules into a single pipeline:

    Raw Text
      │
      ▼
    ┌────────────────────┐
    │  1. Preprocessing  │  Clean, normalize, fix OCR
    └────────┬───────────┘
             │
      ▼ spaCy Doc
    ┌────────────────────┐
    │  2. NER Extractor  │  spaCy NER + rule-based budget/deadline
    └────────┬───────────┘
             │
    ┌────────────────────┐
    │  3. TF-IDF Engine  │  Keywords, domains, skills
    └────────┬───────────┘
             │
    ┌────────────────────┐
    │  4. Merge & Rank   │  Combine NER + TF-IDF → structured output
    └────────┬───────────┘
             │
      ▼ ExtractionResult (JSON-serializable)

Usage:
    from backend.nlp.keyword_extraction import KeywordExtractor

    extractor = KeywordExtractor()            # loads spaCy model once
    result = extractor.extract(tender_text)   # returns ExtractionResult
    print(result.to_dict())                   # JSON output

Batch usage:
    results = extractor.extract_batch(list_of_texts)

Author: SmartTender AI Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import spacy
from spacy.tokens import Doc

try:
    from langdetect import detect as _langdetect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

from backend.nlp.preprocessing import (
    preprocess_text,
    lemmatize,
    segment_sentences,
)
from backend.nlp.ner_extractor import (
    NERResult,
    extract_entities,
)
from backend.nlp.tfidf_extractor import (
    TFIDFResult,
    extract_tfidf_keywords,
)


# ================================================================
# EXTRACTION RESULT (Structured Output)
# ================================================================

@dataclass
class ExtractionResult:
    """
    Final structured output of the keyword extraction pipeline.

    This is the JSON contract between the NLP module and downstream
    consumers (relevance scoring, alert system, database).

    Matches the required output format:
    {
        "domain": "...",
        "skills": [...],
        "budget": "...",
        "deadline": "...",
        "organization": "...",
        "location": "..."
    }
    """
    # ── Primary fields ──
    domain: str = ""                            # Primary domain/sector
    domains: List[str] = field(default_factory=list)  # All detected domains (multi-domain)
    skills: List[Dict[str, Any]] = field(default_factory=list)  # Ranked skills
    budget: Optional[str] = None                # Human-readable budget string
    budget_amount: Optional[float] = None       # Parsed numeric amount
    budget_currency: Optional[str] = None       # ISO currency code
    deadline: Optional[str] = None              # ISO-8601 date string
    deadline_raw: Optional[str] = None          # Original text of deadline
    organization: Optional[str] = None          # Primary issuing organization
    organizations: List[str] = field(default_factory=list)  # All organizations
    location: Optional[str] = None              # Primary location
    locations: List[str] = field(default_factory=list)  # All locations

    # ── Extended fields ──
    top_keywords: List[Dict[str, Any]] = field(default_factory=list)
    noun_chunks: List[str] = field(default_factory=list)
    all_dates: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)

    # ── Metadata ──
    detected_language: str = "en"            # ISO 639-1 code (en, fr, ar)
    processing_time_ms: float = 0.0
    text_length: int = 0
    sentence_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the standard JSON output format."""
        return {
            "domain": self.domain,
            "domains": self.domains,
            "skills": self.skills,
            "budget": self.budget,
            "budget_amount": self.budget_amount,
            "budget_currency": self.budget_currency,
            "deadline": self.deadline,
            "deadline_raw": self.deadline_raw,
            "organization": self.organization,
            "organizations": self.organizations,
            "location": self.location,
            "locations": self.locations,
            "top_keywords": self.top_keywords,
            "noun_chunks": self.noun_chunks,
            "all_dates": self.all_dates,
            "certifications": self.certifications,
            "meta": {
                "detected_language": self.detected_language,
                "processing_time_ms": round(self.processing_time_ms, 1),
                "text_length": self.text_length,
                "sentence_count": self.sentence_count,
            },
        }

    def to_compact_dict(self) -> Dict[str, Any]:
        """Minimal output matching the required JSON format."""
        return {
            "domain": self.domain,
            "skills": [s["name"] for s in self.skills],
            "budget": self.budget or "Not specified",
            "deadline": self.deadline or "Not specified",
            "organization": self.organization or "Not specified",
            "location": self.location or "Not specified",
        }


# ================================================================
# KEYWORD EXTRACTOR (Main Pipeline)
# ================================================================

class KeywordExtractor:
    """
    Production-grade keyword extraction pipeline.

    Loads the spaCy model once and reuses it across calls.
    Thread-safe for read-only operations (spaCy Doc creation).

    Args:
        model_name:   spaCy model to load (default: en_core_web_sm).
        top_keywords: Number of TF-IDF keywords to extract.
        fix_ocr:      Apply OCR noise repair during preprocessing.
    """

    # Supported languages → spaCy model mapping
    LANG_MODELS = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
    }

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        top_keywords: int = 30,
        fix_ocr: bool = True,
    ):
        self.top_keywords = top_keywords
        self.fix_ocr = fix_ocr

        # Load spaCy models: English (always) + French (if available)
        self._models: Dict[str, spacy.language.Language] = {}
        for lang, mname in self.LANG_MODELS.items():
            try:
                nlp_model = spacy.load(mname)
                nlp_model.max_length = 200_000
                self._models[lang] = nlp_model
            except OSError:
                if lang == "en":
                    # English is mandatory — try to install
                    import subprocess
                    subprocess.run(
                        ["python", "-m", "spacy", "download", mname],
                        check=True,
                    )
                    nlp_model = spacy.load(mname)
                    nlp_model.max_length = 200_000
                    self._models[lang] = nlp_model
                # French/others: silently skip if not installed

        # Default model reference (backward compat)
        self.nlp = self._models.get("en", list(self._models.values())[0])

    def _detect_language(self, text: str) -> str:
        """
        Detect text language using langdetect.

        Returns ISO 639-1 code: 'en', 'fr', 'ar', etc.
        Falls back to 'en' on any error.
        """
        if not HAS_LANGDETECT or not text or len(text.strip()) < 20:
            return "en"
        try:
            lang = _langdetect(text[:1000])  # sample first 1000 chars
            return lang
        except Exception:
            return "en"

    def _get_nlp_for_lang(self, lang: str) -> spacy.language.Language:
        """
        Get the best spaCy model for the detected language.

        - 'en' → en_core_web_sm
        - 'fr' → fr_core_news_sm (if installed)
        - 'ar' or others → fallback to English (regex still works)
        """
        if lang in self._models:
            return self._models[lang]
        # Arabic and other unsupported languages: use English model
        # (skill regex patterns still match, TF-IDF still works on tokens)
        return self.nlp

    # ============================================================
    # PUBLIC: Single document extraction
    # ============================================================
    def extract(
        self,
        text: str,
        *,
        title: str = "",
        existing_metadata: Optional[Dict] = None,
    ) -> ExtractionResult:
        """
        Extract structured fields from a single tender text.

        Args:
            text:              Raw tender description/body text.
            title:             Tender title (prepended for better NER context).
            existing_metadata: Pre-existing fields (budget, deadline from scraper)
                               to use as fallback if NLP extraction fails.

        Returns:
            ExtractionResult with all structured fields.
        """
        start = time.perf_counter()

        # ── Stage 1: Preprocessing ──
        full_text = f"{title}. {text}" if title else text
        cleaned = preprocess_text(full_text, fix_ocr=self.fix_ocr)

        if not cleaned:
            return ExtractionResult(processing_time_ms=0, text_length=0)

        # ── Stage 1.5: Language Detection ──
        detected_lang = self._detect_language(cleaned)
        nlp_model = self._get_nlp_for_lang(detected_lang)

        # ── Stage 2: spaCy processing (language-appropriate model) ──
        doc = nlp_model(cleaned)

        # ── Stage 3: NER extraction ──
        ner_result: NERResult = extract_entities(doc, raw_text=cleaned)

        # ── Stage 4: Sentence segmentation & lemmatization ──
        sentences = segment_sentences(doc, min_length=15)
        lemmas = lemmatize(doc)

        # Extract noun chunks for TF-IDF context
        noun_chunks = [
            chunk.text.strip().lower()
            for chunk in doc.noun_chunks
            if 2 < len(chunk.text.strip()) and len(chunk.text.split()) <= 4
        ]
        noun_chunks = list(dict.fromkeys(noun_chunks))[:20]  # deduplicate

        # ── Stage 5: TF-IDF keyword extraction ──
        tfidf_result: TFIDFResult = extract_tfidf_keywords(
            sentences=sentences,
            raw_text=cleaned,
            noun_chunks=noun_chunks,
            top_n=self.top_keywords,
        )

        # ── Stage 6: Merge NER + TF-IDF into final result ──
        result = self._merge_results(
            ner_result=ner_result,
            tfidf_result=tfidf_result,
            noun_chunks=noun_chunks,
            sentences=sentences,
            cleaned_text=cleaned,
            existing_metadata=existing_metadata,
        )

        # Metadata
        elapsed = (time.perf_counter() - start) * 1000
        result.detected_language = detected_lang
        result.processing_time_ms = elapsed
        result.text_length = len(cleaned)
        result.sentence_count = len(sentences)

        return result

    # ============================================================
    # PUBLIC: Batch extraction
    # ============================================================
    def extract_batch(
        self,
        tenders: List[Dict[str, Any]],
        *,
        text_field: str = "description",
        title_field: str = "title",
    ) -> List[ExtractionResult]:
        """
        Extract keywords from a list of tender dictionaries.

        Args:
            tenders:     List of tender dicts (from scraping pipeline).
            text_field:  Key for the tender body text.
            title_field: Key for the tender title.

        Returns:
            List of ExtractionResult objects (same order as input).
        """
        results: List[ExtractionResult] = []

        for tender in tenders:
            text = tender.get(text_field, "")
            title = tender.get(title_field, "")

            # Pass existing metadata as fallback
            existing = {
                "budget": tender.get("budget"),
                "deadline": tender.get("deadline"),
                "organization": tender.get("organization"),
                "location": tender.get("location"),
            }

            result = self.extract(
                text,
                title=title,
                existing_metadata=existing,
            )
            results.append(result)

        return results

    # ============================================================
    # INTERNAL: Merge NER + TF-IDF results
    # ============================================================
    def _merge_results(
        self,
        ner_result: NERResult,
        tfidf_result: TFIDFResult,
        noun_chunks: List[str],
        sentences: List[str],
        cleaned_text: str,
        existing_metadata: Optional[Dict] = None,
    ) -> ExtractionResult:
        """
        Combine NER and TF-IDF outputs into a single ExtractionResult.

        Merge strategy:
            - Domain:       TF-IDF taxonomy mapping (primary)
            - Skills:       Pattern matching ranked by TF-IDF score
            - Budget:       NER MONEY entities → rule-based regex → scraper fallback
            - Deadline:     Rule-based with signal words → NER DATE → scraper fallback
            - Organization: NER ORG entities → scraper fallback
            - Location:     NER GPE entities → scraper fallback
        """
        meta = existing_metadata or {}
        result = ExtractionResult()

        # ── Domain ──
        result.domains = tfidf_result.detected_domains or ["General"]
        result.domain = result.domains[0] if result.domains else "General"

        # ── Skills ──
        # Separate certifications from regular skills
        skills_list = []
        certs_list = []
        for skill in tfidf_result.detected_skills:
            entry = skill.to_dict()
            if skill.category == "Certification":
                certs_list.append(entry["name"])
            else:
                skills_list.append(entry)
        result.skills = skills_list
        result.certifications = certs_list

        # ── Budget ──
        if ner_result.budget and ner_result.budget.amount:
            b = ner_result.budget
            result.budget = b.raw_text
            result.budget_amount = b.amount
            result.budget_currency = b.currency
        elif meta.get("budget"):
            result.budget = meta["budget"]

        # ── Deadline ──
        if ner_result.deadline and ner_result.deadline.iso_date:
            result.deadline = ner_result.deadline.iso_date
            result.deadline_raw = ner_result.deadline.raw_text
        elif meta.get("deadline"):
            result.deadline = meta["deadline"]

        # ── Organization ──
        result.organizations = ner_result.organizations
        if ner_result.organizations:
            result.organization = ner_result.organizations[0]
        elif meta.get("organization"):
            result.organization = meta["organization"]

        # ── Location ──
        result.locations = ner_result.locations
        if ner_result.locations:
            result.location = ner_result.locations[0]
        elif meta.get("location"):
            result.location = meta["location"]

        # ── Keywords & Chunks ──
        result.top_keywords = [kw.to_dict() for kw in tfidf_result.top_keywords]
        result.noun_chunks = noun_chunks
        result.all_dates = ner_result.all_dates

        return result
