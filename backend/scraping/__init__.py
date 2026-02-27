"""
SmartTender AI — Scraping Package
==================================
Production-grade multi-tier scraping pipeline.

Tier 1: API clients (SAM.gov, TED Open Data, Contracts Finder)
Tier 2: RSS / Atom feed parsers
Tier 3: Static HTML scrapers (httpx + BeautifulSoup)
Tier 4: Browser automation (Scrapling / Playwright)
"""

from .base import BaseScraper, ScraperResult, NormalizedTender, ScraperConfig
from .registry import ScraperRegistry
from .pipeline import ScrapingPipeline

__all__ = [
    "BaseScraper",
    "ScraperResult",
    "NormalizedTender",
    "ScraperConfig",
    "ScraperRegistry",
    "ScrapingPipeline",
]
