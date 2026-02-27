"""
SmartTender AI — Scraper Registry
===================================
Auto-discovery registry for scraper classes.
Allows adding new sources without modifying the pipeline.
"""

from __future__ import annotations

from typing import Dict, Type

import structlog

from .base import BaseScraper, ScraperConfig

logger = structlog.get_logger(__name__)


class ScraperRegistry:
    """
    Central registry of all available scrapers.

    Usage:
        @ScraperRegistry.register("SAM.gov")
        class SamGovScraper(BaseScraper):
            ...

        # Later:
        scraper_cls = ScraperRegistry.get("SAM.gov")
        scraper = scraper_cls(config)
        result = scraper.scrape()
    """

    _registry: Dict[str, Type[BaseScraper]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a scraper class."""
        def decorator(scraper_cls: Type[BaseScraper]):
            cls._registry[name.upper()] = scraper_cls
            logger.debug("scraper_registered", name=name, cls=scraper_cls.__name__)
            return scraper_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseScraper] | None:
        """Look up a scraper class by source name."""
        return cls._registry.get(name.upper())

    @classmethod
    def list_sources(cls) -> list[str]:
        """Return all registered source names."""
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, config: ScraperConfig) -> BaseScraper | None:
        """Instantiate a scraper by name."""
        scraper_cls = cls.get(name)
        if scraper_cls is None:
            logger.warning("scraper_not_found", name=name)
            return None
        return scraper_cls(config)

    @classmethod
    def create_all(cls, configs: Dict[str, ScraperConfig]) -> list[BaseScraper]:
        """Instantiate all enabled scrapers from config dict."""
        scrapers = []
        for name, config in configs.items():
            if not config.enabled:
                logger.info("scraper_disabled", name=name)
                continue
            scraper = cls.create(name, config)
            if scraper:
                scrapers.append(scraper)
        return scrapers
