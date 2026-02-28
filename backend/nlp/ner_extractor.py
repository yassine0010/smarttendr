"""
SmartTender AI — Named Entity Recognition Extractor
======================================================
Combines spaCy's statistical NER with rule-based patterns
to extract structured entities from tender text.

Extracted entity types:
    - ORG    → organization / issuer
    - GPE    → location (country, city)
    - MONEY  → budget / estimated value
    - DATE   → deadlines, publication dates
    - NORP   → nationalities, groups
    - FAC    → facilities

Rule-based augmentation:
    - Regex patterns for budget amounts with currency symbols
    - Deadline detection using signal-word proximity
    - Date normalization to ISO-8601

Why both NER + rules?
    spaCy's statistical model catches ~80% of entities but misses
    domain-specific patterns (e.g., "500,000 TND", "DT 1.2M").
    Rule-based patterns raise recall for budget/deadline to ~95%.

Author: SmartTender AI Team
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from spacy.tokens import Doc

from backend.nlp.taxonomy import CURRENCY_MAP, DEADLINE_SIGNALS


# ================================================================
# DATA CLASSES FOR EXTRACTED ENTITIES
# ================================================================

@dataclass
class ExtractedEntity:
    """A single extracted entity with metadata."""
    text: str               # Original text span
    label: str              # Entity type (ORG, MONEY, DATE, GPE, ...)
    start_char: int = 0     # Character offset in source text
    end_char: int = 0
    confidence: float = 1.0 # 1.0 for spaCy NER, 0.8 for rule-based
    source: str = "ner"     # "ner" or "rule"


@dataclass
class BudgetInfo:
    """Structured budget extraction result."""
    raw_text: str = ""
    amount: Optional[float] = None
    currency: Optional[str] = None
    is_estimated: bool = False

    def to_dict(self) -> Dict:
        return {
            "raw_text": self.raw_text,
            "amount": self.amount,
            "currency": self.currency,
            "is_estimated": self.is_estimated,
        }


@dataclass
class DeadlineInfo:
    """Structured deadline extraction result."""
    raw_text: str = ""
    iso_date: Optional[str] = None      # ISO-8601 string
    signal_word: str = ""               # Which signal triggered it
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "raw_text": self.raw_text,
            "iso_date": self.iso_date,
            "signal_word": self.signal_word,
            "confidence": self.confidence,
        }


@dataclass
class NERResult:
    """Aggregated output of the NER extraction pass."""
    organizations: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    budget: Optional[BudgetInfo] = None
    deadline: Optional[DeadlineInfo] = None
    all_dates: List[str] = field(default_factory=list)
    all_money: List[str] = field(default_factory=list)
    raw_entities: List[ExtractedEntity] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "organizations": self.organizations,
            "locations": self.locations,
            "budget": self.budget.to_dict() if self.budget else None,
            "deadline": self.deadline.to_dict() if self.deadline else None,
            "all_dates": self.all_dates,
            "all_money": self.all_money,
        }


# ================================================================
# COMPILED PATTERNS
# ================================================================

# Budget: matches amounts with currency symbols/codes
# Examples: "$500,000", "€1.2M", "500 000 TND", "1,200,000.00 EUR"
_RE_BUDGET = re.compile(
    r"(?:"
    r"(?P<symbol>[$€£¥₹])\s*(?P<amount1>[\d,.\s]+)"
    r"|"
    r"(?P<amount2>[\d,.\s]+)\s*(?P<code>[A-Z]{3}|DT|dinars?)"
    r")"
    r"\s*(?P<suffix>[MmKkBb](?:illion|illion)?)?",
    re.IGNORECASE,
)

# Budget signal words (to boost confidence)
_RE_BUDGET_SIGNAL = re.compile(
    r"\b(?:budget|estimated\s+value|contract\s+value|ceiling|"
    r"maximum\s+value|total\s+amount|worth|valued?\s+at|"
    r"montant|valeur\s+estimée)\b",
    re.IGNORECASE,
)

# Date patterns: various formats
_DATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # ISO: 2025-03-15
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "%Y-%m-%d"),
    # European: 15/03/2025 or 15.03.2025
    (re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b"), "%d/%m/%Y"),
    # Written: March 15, 2025 or 15 March 2025
    (re.compile(
        r"\b(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+"
        r"(\d{4})\b",
        re.IGNORECASE,
    ), "%d %B %Y"),
    (re.compile(
        r"\b(January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+"
        r"(\d{1,2}),?\s+(\d{4})\b",
        re.IGNORECASE,
    ), "%B %d %Y"),
    # French dates: 15 mars 2025
    (re.compile(
        r"\b(\d{1,2})\s+"
        r"(janvier|février|mars|avril|mai|juin|juillet|"
        r"août|septembre|octobre|novembre|décembre)\s+"
        r"(\d{4})\b",
        re.IGNORECASE,
    ), "FR"),
]

_FRENCH_MONTHS = {
    "janvier": 1, "février": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
}


# ================================================================
# PUBLIC API
# ================================================================

def extract_entities(doc: Doc, raw_text: str = "") -> NERResult:
    """
    Run full NER extraction: spaCy statistical + rule-based augmentation.

    Args:
        doc:      spaCy Doc (already processed through pipeline).
        raw_text: Original raw text for regex-based extraction.

    Returns:
        NERResult with structured organizations, locations, budget, deadline.
    """
    if not raw_text:
        raw_text = doc.text

    result = NERResult()

    # ── Pass 1: spaCy statistical NER ──
    _extract_spacy_entities(doc, result)

    # ── Pass 2: Rule-based budget extraction ──
    rule_budgets = _extract_budget_regex(raw_text)
    if rule_budgets:
        # Pick the best budget (highest amount near a budget signal word)
        best = _select_best_budget(rule_budgets, raw_text)
        if best:
            # Prefer rule-based budget if spaCy didn't find one, or rule has higher amount
            if result.budget is None or (
                best.amount and result.budget.amount
                and best.amount > result.budget.amount
            ):
                result.budget = best

    # ── Pass 3: Rule-based deadline extraction ──
    rule_dates = _extract_dates_regex(raw_text)
    deadline = _select_submission_deadline(rule_dates, raw_text)
    if deadline:
        # Prefer rule-based if higher confidence or spaCy missed it
        if result.deadline is None or deadline.confidence > result.deadline.confidence:
            result.deadline = deadline

    # Deduplicate
    result.organizations = _deduplicate(result.organizations)
    result.locations = _deduplicate(result.locations)
    result.all_dates = _deduplicate(result.all_dates)
    result.all_money = _deduplicate(result.all_money)

    return result


# ================================================================
# PASS 1: SPACY STATISTICAL NER
# ================================================================

def _extract_spacy_entities(doc: Doc, result: NERResult) -> None:
    """Populate NERResult from spaCy's built-in NER predictions."""
    for ent in doc.ents:
        text = ent.text.strip()
        if len(text) < 2:
            continue

        entity = ExtractedEntity(
            text=text,
            label=ent.label_,
            start_char=ent.start_char,
            end_char=ent.end_char,
            source="ner",
        )
        result.raw_entities.append(entity)

        if ent.label_ == "ORG":
            result.organizations.append(text)

        elif ent.label_ in ("GPE", "LOC"):
            result.locations.append(text)

        elif ent.label_ == "MONEY":
            result.all_money.append(text)
            # Try to parse as budget
            budget = _parse_money_entity(text)
            if budget and (
                result.budget is None
                or (budget.amount and result.budget.amount
                    and budget.amount > result.budget.amount)
            ):
                result.budget = budget

        elif ent.label_ == "DATE":
            result.all_dates.append(text)


# ================================================================
# PASS 2: RULE-BASED BUDGET EXTRACTION
# ================================================================

def _extract_budget_regex(text: str) -> List[BudgetInfo]:
    """Extract all budget-like amounts using regex patterns."""
    budgets: List[BudgetInfo] = []

    for match in _RE_BUDGET.finditer(text):
        raw = match.group(0).strip()

        # Determine amount
        amount_str = match.group("amount1") or match.group("amount2") or ""
        amount_str = amount_str.replace(" ", "").replace(",", "")

        try:
            amount = float(amount_str) if amount_str else None
        except ValueError:
            amount = None

        # Apply suffix multiplier
        suffix = match.group("suffix") or ""
        if suffix:
            suffix_lower = suffix[0].lower()
            if suffix_lower == "m":
                amount = (amount or 0) * 1_000_000
            elif suffix_lower == "k":
                amount = (amount or 0) * 1_000
            elif suffix_lower == "b":
                amount = (amount or 0) * 1_000_000_000

        # Determine currency
        symbol = match.group("symbol") or ""
        code = match.group("code") or ""
        currency = None
        for key, val in CURRENCY_MAP.items():
            if symbol and key in symbol:
                currency = val
                break
            if code and key == code.lower():
                currency = val
                break

        # Check if "estimated" is nearby
        context_start = max(0, match.start() - 50)
        context = text[context_start:match.end() + 20].lower()
        is_estimated = "estimat" in context or "approximat" in context

        if amount and amount > 100:  # Filter out noise (page numbers, etc.)
            budgets.append(BudgetInfo(
                raw_text=raw,
                amount=amount,
                currency=currency,
                is_estimated=is_estimated,
            ))

    return budgets


def _select_best_budget(
    budgets: List[BudgetInfo],
    full_text: str,
) -> Optional[BudgetInfo]:
    """
    From multiple budget candidates, select the most likely contract value.

    Heuristic: prefer amounts near budget signal words, then largest amount.
    """
    if not budgets:
        return None

    scored: List[Tuple[float, BudgetInfo]] = []
    text_lower = full_text.lower()

    for b in budgets:
        score = b.amount or 0

        # Boost if near a budget signal word
        raw_lower = b.raw_text.lower()
        idx = text_lower.find(raw_lower)
        if idx >= 0:
            context = text_lower[max(0, idx - 100):idx + len(raw_lower) + 50]
            if _RE_BUDGET_SIGNAL.search(context):
                score *= 10  # Strong boost

        scored.append((score, b))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _parse_money_entity(text: str) -> Optional[BudgetInfo]:
    """Parse a spaCy MONEY entity into BudgetInfo."""
    # Try to extract number
    num_match = re.search(r"[\d,.\s]+", text)
    if not num_match:
        return None

    amount_str = num_match.group().replace(",", "").replace(" ", "").strip()
    try:
        amount = float(amount_str)
    except ValueError:
        return None

    # Detect currency
    currency = None
    text_lower = text.lower()
    for key, val in CURRENCY_MAP.items():
        if key in text_lower:
            currency = val
            break

    return BudgetInfo(raw_text=text, amount=amount, currency=currency)


# ================================================================
# PASS 3: RULE-BASED DATE / DEADLINE EXTRACTION
# ================================================================

def _extract_dates_regex(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Extract all date-like strings from text using regex.

    Returns:
        List of (raw_text, iso_date_or_None) tuples.
    """
    found: List[Tuple[str, Optional[str]]] = []

    for pattern, fmt in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            iso = _parse_date_to_iso(match, fmt)
            found.append((raw, iso))

    return found


def _parse_date_to_iso(match: re.Match, fmt: str) -> Optional[str]:
    """Convert a regex match to ISO-8601 date string."""
    try:
        if fmt == "FR":
            day = int(match.group(1))
            month_name = match.group(2).lower()
            year = int(match.group(3))
            month = _FRENCH_MONTHS.get(month_name, 0)
            if month:
                return date(year, month, day).isoformat()
            return None

        if fmt == "%d/%m/%Y":
            d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return date(y, m, d).isoformat()

        if fmt == "%Y-%m-%d":
            y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return date(y, m, d).isoformat()

        if fmt == "%d %B %Y":
            raw = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            return datetime.strptime(raw, fmt).date().isoformat()

        if fmt == "%B %d %Y":
            raw = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            return datetime.strptime(raw.replace(",", ""), fmt).date().isoformat()

    except (ValueError, IndexError):
        pass

    return None


def _select_submission_deadline(
    dates: List[Tuple[str, Optional[str]]],
    full_text: str,
) -> Optional[DeadlineInfo]:
    """
    From all extracted dates, identify the submission deadline.

    Strategy:
        1. Score each date by proximity to deadline signal words
        2. Among high-scoring dates, pick the latest one (deadlines are usually
           the furthest future date mentioned near a signal)
        3. If no signal words found, pick the latest future date

    Handles edge case of multiple deadlines by preferring submission-related ones.
    """
    if not dates:
        return None

    text_lower = full_text.lower()
    today = date.today()

    candidates: List[Tuple[float, DeadlineInfo]] = []

    for raw, iso in dates:
        # Find position of this date in text
        raw_lower = raw.lower()
        idx = text_lower.find(raw_lower)
        if idx < 0:
            continue

        # Check proximity to each deadline signal word
        best_signal = ""
        best_signal_score = 0.0

        # Look in a window of 200 chars before the date
        context_start = max(0, idx - 200)
        context = text_lower[context_start:idx + len(raw) + 50]

        for signal in DEADLINE_SIGNALS:
            if signal.lower() in context:
                # Closer signal → higher score
                sig_idx = context.find(signal.lower())
                distance = abs(sig_idx - (idx - context_start))
                # Score: 1.0 for adjacent, decays with distance
                score = max(0.1, 1.0 - distance / 200)
                if score > best_signal_score:
                    best_signal_score = score
                    best_signal = signal

        # Parse ISO date to check if it's in the future
        future_bonus = 0.0
        if iso:
            try:
                d = date.fromisoformat(iso)
                if d >= today:
                    future_bonus = 0.3
            except ValueError:
                pass

        total_score = best_signal_score + future_bonus

        candidates.append((total_score, DeadlineInfo(
            raw_text=raw,
            iso_date=iso,
            signal_word=best_signal,
            confidence=round(min(1.0, total_score), 2),
        )))

    if not candidates:
        return None

    # Sort by score descending, then by date (latest first) for ties
    candidates.sort(
        key=lambda x: (x[0], x[1].iso_date or ""),
        reverse=True,
    )

    return candidates[0][1]


# ================================================================
# UTILITIES
# ================================================================

def _deduplicate(items: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen = set()
    result = []
    for item in items:
        normalized = item.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(item.strip())
    return result
