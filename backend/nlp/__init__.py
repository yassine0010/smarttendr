"""
SmartTender AI — NLP Package
==============================
Production-grade keyword extraction pipeline using spaCy NER + TF-IDF.

Modules:
    preprocessing   – Text cleaning, normalization, lemmatization
    ner_extractor   – spaCy NER + rule-based entity extraction
    tfidf_extractor – TF-IDF domain keyword extraction with scoring
    taxonomy        – Domain/sector taxonomy & technology skill dictionary
    keyword_extraction – Main pipeline orchestrator (public API)
"""

from backend.nlp.keyword_extraction import KeywordExtractor, ExtractionResult

__all__ = ["KeywordExtractor", "ExtractionResult"]
