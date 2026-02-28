"""
SmartTender AI — TF-IDF Keyword Extraction Engine
====================================================
Extracts domain-specific keywords from tender text using TF-IDF,
then maps them to the domain taxonomy and skill dictionary.

Pipeline:
    1. TF-IDF vectorization on sentence-level corpus
    2. Top-K keyword extraction with scores
    3. Domain classification via taxonomy lookup
    4. Technology/skill detection via compiled regex patterns
    5. Skill ranking by (TF-IDF score × category weight)

Why TF-IDF over raw word frequency?
    Tender text is full of generic procurement language ("shall provide",
    "in accordance with"). TF-IDF naturally down-weights these high-DF
    terms and surfaces the *distinguishing* technical terms.

When to upgrade to embeddings:
    TF-IDF excels at keyword extraction from single documents. Switch
    to SBERT embeddings when you need:
    - Cross-document semantic similarity
    - Zero-shot domain classification
    - Handling synonyms (TF-IDF misses "ML" ≈ "Machine Learning")
    The current pipeline uses SBERT in the relevance scoring stage
    (module1) while TF-IDF handles keyword extraction here.

Author: SmartTender AI Team
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.nlp.taxonomy import (
    DOMAIN_TAXONOMY,
    SKILL_CATEGORIES_WEIGHT,
    SKILL_PATTERNS,
    TENDER_STOPWORDS,
)


# ================================================================
# DATA CLASSES
# ================================================================

@dataclass
class ScoredKeyword:
    """A keyword with its TF-IDF importance score."""
    term: str
    score: float           # TF-IDF weight (0-1)
    category: str = ""     # Optional: matched taxonomy category

    def to_dict(self) -> Dict:
        d = {"term": self.term, "score": round(self.score, 4)}
        if self.category:
            d["category"] = self.category
        return d


@dataclass
class DetectedSkill:
    """A technology/skill detected via pattern matching + TF-IDF."""
    name: str              # Canonical name (e.g., "Python")
    category: str          # Category (e.g., "Programming Language")
    weight: float          # Category weight (from taxonomy)
    tfidf_score: float     # TF-IDF score if found in keywords, else 0
    combined_score: float  # weight × (1 + tfidf_score) for ranking

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category,
            "importance_score": round(self.combined_score, 4),
        }


@dataclass
class TFIDFResult:
    """Aggregated output of the TF-IDF extraction pass."""
    top_keywords: List[ScoredKeyword] = field(default_factory=list)
    detected_domains: List[str] = field(default_factory=list)
    detected_skills: List[DetectedSkill] = field(default_factory=list)
    noun_chunks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "top_keywords": [k.to_dict() for k in self.top_keywords],
            "detected_domains": self.detected_domains,
            "detected_skills": [s.to_dict() for s in self.detected_skills],
            "noun_chunks": self.noun_chunks,
        }


# ================================================================
# COMPILE SKILL REGEX ONCE
# ================================================================

_COMPILED_SKILLS: List[Tuple[str, str, re.Pattern]] = [
    (canonical, category, re.compile(pattern, re.IGNORECASE))
    for canonical, category, pattern in SKILL_PATTERNS
]


# ================================================================
# PUBLIC API
# ================================================================

def extract_tfidf_keywords(
    sentences: List[str],
    raw_text: str,
    noun_chunks: Optional[List[str]] = None,
    *,
    top_n: int = 30,
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 1,
    max_df: float = 0.85,
) -> TFIDFResult:
    """
    Full TF-IDF extraction pipeline.

    Args:
        sentences:    List of sentence strings (from preprocessing).
        raw_text:     Full raw text for pattern matching.
        noun_chunks:  Optional pre-extracted noun chunks from spaCy.
        top_n:        Number of top TF-IDF keywords to extract.
        ngram_range:  N-gram range for TF-IDF vectorizer.
        min_df:       Minimum document frequency.
        max_df:       Maximum document frequency (fraction).

    Returns:
        TFIDFResult with keywords, domains, and skills.
    """
    result = TFIDFResult()

    if noun_chunks:
        result.noun_chunks = noun_chunks[:20]

    # ── Step 1: TF-IDF keyword extraction ──
    if sentences and len(sentences) >= 1:
        keywords = _compute_tfidf_keywords(
            sentences,
            top_n=top_n,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        result.top_keywords = keywords

    # ── Step 2: Domain classification from taxonomy ──
    result.detected_domains = _classify_domains(
        result.top_keywords,
        result.noun_chunks,
        raw_text,
    )

    # ── Step 3: Skill/technology detection ──
    tfidf_term_scores = {kw.term.lower(): kw.score for kw in result.top_keywords}
    result.detected_skills = _detect_skills(raw_text, tfidf_term_scores)

    return result


# ================================================================
# STEP 1: TF-IDF KEYWORD COMPUTATION
# ================================================================

def _compute_tfidf_keywords(
    sentences: List[str],
    *,
    top_n: int = 30,
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 1,
    max_df: float = 0.85,
) -> List[ScoredKeyword]:
    """
    Compute TF-IDF on sentence corpus and extract top keywords.

    Uses a custom stopword list (English + tender-specific) and
    ngram_range=(1,3) to capture multi-word technical terms like
    "machine learning", "penetration testing", "supply chain management".
    """
    # Combine sklearn's English stopwords with our tender-specific ones
    custom_stop = list(TENDER_STOPWORDS)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",       # Built-in English stopwords
            max_features=top_n * 3,     # Over-fetch then re-rank
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,          # log(1 + tf) for better scaling
            strip_accents="unicode",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]{1,}\b",  # Min 2 chars, allow C++, C#
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Empty vocabulary (all stopwords)
        return []

    feature_names = vectorizer.get_feature_names_out()

    # Aggregate TF-IDF scores across all sentences (mean)
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    # Sort by score descending
    top_indices = mean_scores.argsort()[::-1]

    keywords: List[ScoredKeyword] = []
    seen_lower: Set[str] = set()

    for idx in top_indices:
        term = feature_names[idx]
        score = float(mean_scores[idx])

        if score <= 0:
            break

        # Skip tender-specific stopwords that sklearn missed
        term_lower = term.lower()
        if term_lower in TENDER_STOPWORDS:
            continue

        # Deduplicate (e.g., "cloud" and "cloud computing")
        if term_lower in seen_lower:
            continue
        seen_lower.add(term_lower)

        keywords.append(ScoredKeyword(term=term, score=score))

        if len(keywords) >= top_n:
            break

    return keywords


# ================================================================
# STEP 2: DOMAIN CLASSIFICATION
# ================================================================

def _classify_domains(
    keywords: List[ScoredKeyword],
    noun_chunks: List[str],
    raw_text: str,
) -> List[str]:
    """
    Map extracted keywords to the domain taxonomy.

    Strategy:
        1. Check each TF-IDF keyword against DOMAIN_TAXONOMY keys
        2. Check noun chunks against taxonomy
        3. Score each domain by (number of matches × max TF-IDF score)
        4. Return domains sorted by score

    Handles multi-domain tenders by returning all matching domains
    (not just the top one).
    """
    domain_scores: Dict[str, float] = {}
    text_lower = raw_text.lower()

    # Check TF-IDF keywords
    for kw in keywords:
        term_lower = kw.term.lower()
        for trigger, domain in DOMAIN_TAXONOMY.items():
            if trigger in term_lower or term_lower in trigger:
                current = domain_scores.get(domain, 0)
                domain_scores[domain] = current + kw.score

    # Check noun chunks
    for chunk in (noun_chunks or []):
        chunk_lower = chunk.lower()
        for trigger, domain in DOMAIN_TAXONOMY.items():
            if trigger in chunk_lower or chunk_lower in trigger:
                domain_scores[domain] = domain_scores.get(domain, 0) + 0.1

    # Also scan raw text for domain keywords (catch terms TF-IDF filtered out)
    for trigger, domain in DOMAIN_TAXONOMY.items():
        if trigger in text_lower:
            domain_scores[domain] = domain_scores.get(domain, 0) + 0.05

    if not domain_scores:
        return ["General"]

    # Sort by score, return all domains above a minimum threshold
    sorted_domains = sorted(
        domain_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Include domains with score >= 20% of the top domain
    top_score = sorted_domains[0][1]
    threshold = top_score * 0.2

    return [
        domain for domain, score in sorted_domains
        if score >= threshold
    ]


# ================================================================
# STEP 3: SKILL / TECHNOLOGY DETECTION
# ================================================================

def _detect_skills(
    raw_text: str,
    tfidf_scores: Dict[str, float],
) -> List[DetectedSkill]:
    """
    Detect skills/technologies using compiled regex patterns,
    then rank by (category_weight × (1 + tfidf_score)).

    This combines:
        - Pattern matching: catches exact technology names
        - TF-IDF scoring: boosts skills that are topically important
        - Category weighting: ranks Programming Languages > Methodologies
    """
    detected: List[DetectedSkill] = []
    seen: Set[str] = set()

    for canonical, category, pattern in _COMPILED_SKILLS:
        if canonical in seen:
            continue

        if pattern.search(raw_text):
            seen.add(canonical)

            # Look up TF-IDF score for this term
            canonical_lower = canonical.lower()
            tfidf_score = tfidf_scores.get(canonical_lower, 0.0)

            # Also check partial matches in TF-IDF terms
            if tfidf_score == 0:
                for term, score in tfidf_scores.items():
                    if canonical_lower in term or term in canonical_lower:
                        tfidf_score = max(tfidf_score, score)

            # Category weight
            cat_weight = SKILL_CATEGORIES_WEIGHT.get(category, 0.5)

            # Combined ranking score
            combined = cat_weight * (1.0 + tfidf_score)

            detected.append(DetectedSkill(
                name=canonical,
                category=category,
                weight=cat_weight,
                tfidf_score=tfidf_score,
                combined_score=combined,
            ))

    # Sort by combined score descending
    detected.sort(key=lambda s: s.combined_score, reverse=True)

    return detected
