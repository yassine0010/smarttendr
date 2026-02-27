"""
SmartTender AI — Base Scraper & Data Models
=============================================
Defines the abstract base class every scraper must implement,
plus the unified data models for normalized tender output.
"""

from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


# ================================================================
# ENUMS
# ================================================================

class ScraperTier(str, Enum):
    """Classification of scraping method complexity."""
    T1_API = "api"
    T2_FEED = "feed"
    T3_HTML = "html"
    T4_BROWSER = "browser"


class ScrapeStatus(str, Enum):
    """Result status of a scrape operation."""
    SUCCESS = "success"
    PARTIAL = "partial"          # some pages failed
    EMPTY = "empty"              # 0 results (possible breakage)
    ERROR = "error"              # unrecoverable error
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"


# ================================================================
# CONFIGURATION
# ================================================================

@dataclass
class ScraperConfig:
    """Per-source scraper configuration."""
    source_name: str
    tier: ScraperTier
    base_url: str
    enabled: bool = True
    max_results: int = 50
    timeout: int = 30
    delay: float = 1.5              # seconds between requests
    max_retries: int = 3
    requests_per_minute: int = 20
    concurrent: int = 2
    api_key: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    proxy: Optional[str] = None


# ================================================================
# NORMALIZED TENDER MODEL
# ================================================================

@dataclass
class NormalizedTender:
    """
    Universal tender representation.
    Every scraper must produce data in this format.
    """
    # Identity
    source_id: str = ""
    platform: str = ""
    source_url: str = ""

    # Core fields
    title: str = ""
    description: str = ""
    organization: str = ""
    category: str = ""

    # Dates
    published_date: str = ""
    deadline: str = ""

    # Financial
    budget: str = ""
    budget_amount: Optional[float] = None
    budget_currency: Optional[str] = None

    # Location
    country: str = ""
    region: str = ""
    location: str = ""

    # Requirements
    required_skills: List[str] = field(default_factory=list)
    required_certifications: List[str] = field(default_factory=list)

    # Metadata
    raw_html: str = ""
    scrape_timestamp: str = ""
    content_hash: str = ""

    def __post_init__(self):
        if not self.scrape_timestamp:
            self.scrape_timestamp = datetime.now(timezone.utc).isoformat()
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA-256 of title + description for deduplication."""
        content = f"{self.title}|{self.description}".strip().lower()
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        # Don't include raw_html in normal output (too large)
        d.pop("raw_html", None)
        return d

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to the legacy format expected by TenderDetector.
        Ensures backward compatibility with Module 1.
        """
        return {
            "id": self.source_id,
            "title": self.title,
            "platform": self.platform,
            "published_date": self.published_date,
            "deadline": self.deadline,
            "budget": self.budget,
            "description": self.description,
            "required_skills": self.required_skills,
            "category": self.category,
            "location": self.location or self.country,
            "source_url": self.source_url,
        }


# ================================================================
# SCRAPER RESULT
# ================================================================

@dataclass
class ScraperResult:
    """Encapsulates the output of a single scraper run."""
    source_name: str
    tier: ScraperTier
    status: ScrapeStatus
    tenders: List[NormalizedTender] = field(default_factory=list)
    error_message: str = ""
    duration_seconds: float = 0.0
    pages_fetched: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def count(self) -> int:
        return len(self.tenders)

    def summary(self) -> str:
        return (
            f"[{self.source_name}] {self.status.value}: "
            f"{self.count} tenders in {self.duration_seconds:.1f}s"
        )


# ================================================================
# ABSTRACT BASE SCRAPER
# ================================================================

class BaseScraper(ABC):
    """
    Abstract base class for all scrapers.
    
    Every platform scraper must:
    1. Inherit from BaseScraper
    2. Implement `scrape()` → ScraperResult
    3. Register itself via @ScraperRegistry.register()
    """

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.logger = structlog.get_logger(
            scraper=config.source_name,
            tier=config.tier.value,
        )

    @abstractmethod
    def scrape(self, query: str = "") -> ScraperResult:
        """
        Execute the scraping operation.

        Args:
            query: Optional search query to filter results

        Returns:
            ScraperResult with normalized tenders
        """
        ...

    def _timed_scrape(self, query: str = "") -> ScraperResult:
        """Wrapper that times the scrape and handles top-level errors."""
        start = time.monotonic()
        try:
            result = self.scrape(query)
            result.duration_seconds = round(time.monotonic() - start, 2)
            self.logger.info(
                "scrape_complete",
                status=result.status.value,
                count=result.count,
                duration=result.duration_seconds,
            )
            return result
        except Exception as e:
            duration = round(time.monotonic() - start, 2)
            self.logger.error("scrape_failed", error=str(e), duration=duration)
            return ScraperResult(
                source_name=self.config.source_name,
                tier=self.config.tier,
                status=ScrapeStatus.ERROR,
                error_message=str(e),
                duration_seconds=duration,
            )

    @staticmethod
    def clean_text(text: str, max_length: int = 5000) -> str:
        """Normalize whitespace and truncate text."""
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_length] if len(text) > max_length else text

    @staticmethod
    def parse_budget(text: str) -> tuple[Optional[float], Optional[str]]:
        """
        Extract numeric budget and currency from text.
        Examples:
            "500,000 TND" → (500000.0, "TND")
            "€200,000"    → (200000.0, "EUR")
            "$1.5M"       → (1500000.0, "USD")
        """
        if not text:
            return None, None

        # Currency mapping
        currency_map = {
            "$": "USD", "€": "EUR", "£": "GBP",
            "TND": "TND", "DT": "TND", "USD": "USD",
            "EUR": "EUR", "GBP": "GBP", "CHF": "CHF",
        }

        currency = None
        for symbol, code in currency_map.items():
            if symbol in text.upper():
                currency = code
                break

        # Extract number
        match = re.search(r'[\d,]+\.?\d*', text.replace(' ', ''))
        if match:
            num_str = match.group().replace(',', '')
            try:
                amount = float(num_str)
                # Handle M/K suffixes
                if re.search(r'[Mm]', text):
                    amount *= 1_000_000
                elif re.search(r'[Kk]', text):
                    amount *= 1_000
                return amount, currency
            except ValueError:
                pass

        return None, currency
