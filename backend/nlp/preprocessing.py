"""
SmartTender AI — Text Preprocessing Pipeline
===============================================
Converts raw tender text (HTML-cleaned or OCR output) into
normalized, lemmatized tokens ready for NER + TF-IDF analysis.

Pipeline stages:
    1. Unicode normalization (NFKC)
    2. HTML entity & residual tag removal
    3. Whitespace & line-break normalization
    4. OCR noise repair (common mis-scans)
    5. spaCy tokenization → lemmatization → stopword removal
    6. Sentence segmentation for TF-IDF

Edge Cases Handled:
    - Scanned PDFs with noisy OCR text
    - Mixed-language tenders (EN/FR/AR)
    - Bullet-list formatting artifacts
    - Currency symbols and number formatting

Author: SmartTender AI Team
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import spacy
from spacy.tokens import Doc

from backend.nlp.taxonomy import TENDER_STOPWORDS


# ================================================================
# COMPILED REGEX PATTERNS (compiled once at module load)
# ================================================================

# Residual HTML tags and entities
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_HTML_ENTITIES = re.compile(r"&[a-zA-Z]+;|&#\d+;")

# Bullet / list markers: •, -, *, ▪, ►, ○, ■, etc.
_RE_BULLETS = re.compile(r"^[\s]*[•\-\*▪►○■◆➤➢→⇒]\s*", re.MULTILINE)

# Multiple whitespace / newlines → single space
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

# OCR noise patterns
_RE_OCR_PIPE = re.compile(r"[|l]{2,}")           # || or ll mis-scans
_RE_OCR_BROKEN = re.compile(r"(?<=[a-z])\s(?=[a-z]{1,2}\s)")  # broken words
_RE_OCR_GARBAGE = re.compile(r"[^\x20-\x7E\u00C0-\u024F\u0600-\u06FF\n]+")

# Page headers/footers from PDFs
_RE_PAGE_NUM = re.compile(
    r"(?:^|\n)\s*(?:page|p\.?)\s*\d+\s*(?:of\s*\d+)?\s*(?:\n|$)",
    re.IGNORECASE,
)

# Repeated separator lines: ===, ---, ***, ___
_RE_SEPARATORS = re.compile(r"[-=*_]{4,}")


# ================================================================
# PUBLIC API
# ================================================================

def preprocess_text(
    raw_text: str,
    *,
    max_length: int = 50_000,
    fix_ocr: bool = True,
) -> str:
    """
    Full preprocessing pipeline: raw text → cleaned text.

    Args:
        raw_text:    Raw tender text (may contain HTML residue, OCR noise).
        max_length:  Truncate input beyond this character count.
        fix_ocr:     Apply OCR-specific noise repair heuristics.

    Returns:
        Cleaned, normalized text string ready for NLP.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text[:max_length]

    # Step 1: Unicode normalize (NFKC collapses compatibility chars)
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Strip residual HTML
    text = _RE_HTML_TAGS.sub(" ", text)
    text = _RE_HTML_ENTITIES.sub(" ", text)

    # Step 3: Remove page numbers & separator lines
    text = _RE_PAGE_NUM.sub("\n", text)
    text = _RE_SEPARATORS.sub(" ", text)

    # Step 4: Normalize bullets into sentence-friendly format
    text = _RE_BULLETS.sub("", text)

    # Step 5: OCR noise repair
    if fix_ocr:
        text = _repair_ocr_noise(text)

    # Step 6: Whitespace normalization
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    text = _RE_MULTI_SPACE.sub(" ", text)

    # Step 7: Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    return text.strip()


def lemmatize(
    doc: Doc,
    *,
    min_token_len: int = 2,
    keep_entities: bool = True,
) -> List[str]:
    """
    Lemmatize a spaCy Doc, filtering stopwords and noise.

    Args:
        doc:            Pre-processed spaCy Doc object.
        min_token_len:  Minimum character length for a lemma.
        keep_entities:  If True, preserve named entities as-is (no lemma).

    Returns:
        List of lemmatized tokens.
    """
    entity_spans = set()
    if keep_entities:
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                entity_spans.add(i)

    lemmas: List[str] = []
    for token in doc:
        # Skip punctuation, spaces, and numbers-only
        if token.is_punct or token.is_space or token.like_num:
            continue

        # Keep named entity tokens verbatim
        if token.i in entity_spans:
            lemmas.append(token.text)
            continue

        lemma = token.lemma_.lower().strip()

        # Filter
        if len(lemma) < min_token_len:
            continue
        if token.is_stop:
            continue
        if lemma in TENDER_STOPWORDS:
            continue

        lemmas.append(lemma)

    return lemmas


def segment_sentences(doc: Doc, min_length: int = 15) -> List[str]:
    """
    Extract meaningful sentences from a spaCy Doc.

    Filters out very short fragments (headers, table cells)
    that would add noise to TF-IDF.

    Args:
        doc:        spaCy Doc object.
        min_length: Minimum character length for a sentence.

    Returns:
        List of sentence strings.
    """
    sentences = []
    for sent in doc.sents:
        text = sent.text.strip()
        if len(text) >= min_length:
            sentences.append(text)
    return sentences


# ================================================================
# INTERNAL HELPERS
# ================================================================

def _repair_ocr_noise(text: str) -> str:
    """
    Heuristic repair of common OCR scanning artifacts.

    Handles:
        - Garbage Unicode characters from bad scans
        - Broken words (s p a c e d  o u t)
        - Double-pipe artifacts (|| → ll)
    """
    # Remove non-printable garbage (keep Latin, Arabic, basic punctuation)
    text = _RE_OCR_GARBAGE.sub(" ", text)

    # Fix common OCR substitutions
    ocr_fixes = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "'": "'",
        "'": "'",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",     # Non-breaking space
    }
    for bad, good in ocr_fixes.items():
        text = text.replace(bad, good)

    return text


def normalize_for_matching(text: str) -> str:
    """
    Aggressive normalization for fuzzy matching / dedup.
    Strips everything to lowercase alphanumeric + spaces.
    """
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()
