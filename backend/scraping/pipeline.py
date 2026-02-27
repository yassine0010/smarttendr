"""
SmartTender AI — Scraping Pipeline Orchestrator
=================================================
Orchestrates all scrapers, handles fallbacks, deduplication,
and produces a unified list of normalized tenders.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import structlog

from .base import (
    BaseScraper, NormalizedTender, ScraperConfig, ScraperResult,
    ScraperTier, ScrapeStatus,
)
from .registry import ScraperRegistry

# Import all sources to trigger registration
from . import sources  # noqa: F401

logger = structlog.get_logger(__name__)


# ================================================================
# DEFAULT SOURCE CONFIGURATIONS
# ================================================================

DEFAULT_CONFIGS: Dict[str, ScraperConfig] = {
    "SAM.GOV": ScraperConfig(
        source_name="SAM.gov",
        tier=ScraperTier.T1_API,
        base_url="https://api.sam.gov/opportunities/v2/search",
        max_results=25,
        timeout=30,
        requests_per_minute=60,
        api_key="DEMO_KEY",
        enabled=True,
    ),
    "TED": ScraperConfig(
        source_name="TED Europa",
        tier=ScraperTier.T1_API,
        base_url="https://ted.europa.eu",
        max_results=25,
        timeout=30,
        requests_per_minute=30,
        enabled=True,
    ),
    "UNGM": ScraperConfig(
        source_name="UNGM",
        tier=ScraperTier.T3_HTML,
        base_url="https://www.ungm.org/Public/Notice",
        max_results=25,
        timeout=30,
        delay=2.0,
        requests_per_minute=20,
        enabled=True,
    ),
    "TUNEPS": ScraperConfig(
        source_name="TUNEPS",
        tier=ScraperTier.T4_BROWSER,
        base_url="https://www.tuneps.tn",
        max_results=20,
        timeout=20,
        delay=2.0,
        requests_per_minute=10,
        enabled=True,
    ),
    "CONTRACTS_FINDER": ScraperConfig(
        source_name="Contracts Finder UK",
        tier=ScraperTier.T1_API,
        base_url="https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search",
        max_results=25,
        timeout=30,
        requests_per_minute=30,
        enabled=True,
    ),
}


# ================================================================
# PIPELINE CLASS
# ================================================================

class ScrapingPipeline:
    """
    Orchestrates the full scraping pipeline:
    1. Instantiate scrapers from registry
    2. Execute each scraper sequentially (async upgrade planned)
    3. Normalize and deduplicate results
    4. Cache results and produce output
    5. Report metrics
    """

    def __init__(
        self,
        configs: Dict[str, ScraperConfig] | None = None,
        output_dir: str | Path | None = None,
        enable_cache: bool = True,
    ):
        self.configs = configs or DEFAULT_CONFIGS
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent.parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = enable_cache
        self.logger = structlog.get_logger("pipeline")

    def run(
        self,
        query: str = "IT services",
        sources: List[str] | None = None,
        fallback_file: str | Path | None = None,
    ) -> List[NormalizedTender]:
        """
        Execute the full scraping pipeline.

        Args:
            query: Search keyword for platforms that support it
            sources: List of source names to scrape (None = all enabled)
            fallback_file: Path to fallback JSON if all scrapers fail

        Returns:
            Deduplicated list of NormalizedTender objects
        """
        start = time.monotonic()
        self.logger.info("pipeline_start", query=query, sources=sources)

        # Determine which sources to scrape
        source_names = sources or list(self.configs.keys())

        all_tenders: List[NormalizedTender] = []
        all_results: List[ScraperResult] = []

        # Execute each scraper
        for name in source_names:
            config = self.configs.get(name.upper())
            if not config:
                self.logger.warning("source_not_configured", name=name)
                continue
            if not config.enabled:
                self.logger.info("source_disabled", name=name)
                continue

            scraper = ScraperRegistry.create(name, config)
            if not scraper:
                self.logger.warning("scraper_not_registered", name=name)
                continue

            # Execute with timing
            self.logger.info("scraper_execute", source=name, tier=config.tier.value)
            result = scraper._timed_scrape(query)
            all_results.append(result)

            if result.tenders:
                all_tenders.extend(result.tenders)
                self.logger.info(
                    "scraper_success",
                    source=name,
                    count=result.count,
                    duration=result.duration_seconds,
                )
            else:
                self.logger.warning(
                    "scraper_empty",
                    source=name,
                    status=result.status.value,
                    error=result.error_message,
                )

        # Deduplicate
        before_dedup = len(all_tenders)
        all_tenders = self._deduplicate(all_tenders)
        dedup_removed = before_dedup - len(all_tenders)

        # Fallback to sample data if nothing scraped
        if not all_tenders and fallback_file:
            all_tenders = self._load_fallback(fallback_file)
            self.logger.info("fallback_loaded", count=len(all_tenders))

        # Save results
        self._save_results(all_tenders, all_results)

        # Final metrics
        duration = round(time.monotonic() - start, 2)
        self.logger.info(
            "pipeline_complete",
            total_tenders=len(all_tenders),
            sources_scraped=len(all_results),
            successful_sources=sum(1 for r in all_results if r.count > 0),
            dedup_removed=dedup_removed,
            duration=duration,
        )

        # Print summary
        self._print_summary(all_results, all_tenders, duration)

        return all_tenders

    def _deduplicate(self, tenders: List[NormalizedTender]) -> List[NormalizedTender]:
        """
        Remove duplicate tenders using content hash.
        Step 1: Exact hash match (title + description SHA-256)
        Step 2: Title similarity (exact lowercase match)
        """
        seen_hashes = set()
        seen_titles = set()
        unique = []

        for tender in tenders:
            # Hash-based dedup
            if tender.content_hash in seen_hashes:
                continue

            # Title-based dedup (normalized)
            norm_title = tender.title.strip().lower()
            if norm_title in seen_titles:
                continue

            seen_hashes.add(tender.content_hash)
            seen_titles.add(norm_title)
            unique.append(tender)

        return unique

    def _load_fallback(self, filepath: str | Path) -> List[NormalizedTender]:
        """Load tenders from a fallback JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            tenders = []
            for item in data:
                tenders.append(NormalizedTender(
                    source_id=item.get("id", ""),
                    platform=item.get("platform", "Sample"),
                    source_url=item.get("source_url", ""),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    organization=item.get("organization", ""),
                    category=item.get("category", ""),
                    published_date=item.get("published_date", ""),
                    deadline=item.get("deadline", ""),
                    budget=item.get("budget", ""),
                    country=item.get("location", ""),
                    required_skills=item.get("required_skills", []),
                    required_certifications=item.get("required_certifications", []),
                ))
            return tenders
        except Exception as e:
            self.logger.error("fallback_load_error", error=str(e))
            return []

    def _save_results(
        self,
        tenders: List[NormalizedTender],
        results: List[ScraperResult],
    ):
        """Save scraped tenders and run report to output directory."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Save tenders as JSON
        tenders_file = self.output_dir / f"scraped_tenders_{timestamp}.json"
        tender_dicts = [t.to_dict() for t in tenders]
        with open(tenders_file, "w", encoding="utf-8") as f:
            json.dump(tender_dicts, f, indent=2, ensure_ascii=False)

        # Also save as latest (overwrite)
        latest_file = self.output_dir / "scraped_tenders_latest.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(tender_dicts, f, indent=2, ensure_ascii=False)

        # Save run report
        report = {
            "timestamp": timestamp,
            "total_tenders": len(tenders),
            "sources": [
                {
                    "name": r.source_name,
                    "tier": r.tier.value,
                    "status": r.status.value,
                    "count": r.count,
                    "duration": r.duration_seconds,
                    "error": r.error_message,
                }
                for r in results
            ],
        }
        report_file = self.output_dir / f"scrape_report_{timestamp}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(
            "results_saved",
            tenders_file=str(tenders_file),
            report_file=str(report_file),
        )

    def _print_summary(
        self,
        results: List[ScraperResult],
        tenders: List[NormalizedTender],
        duration: float,
    ):
        """Print a human-readable summary of the pipeline run."""
        print("\n" + "=" * 70)
        print(" SMARTTENDER AI — SCRAPING PIPELINE REPORT")
        print("=" * 70)

        for r in results:
            icon = "✅" if r.count > 0 else ("⚠️ " if r.status == ScrapeStatus.EMPTY else "❌")
            print(f"  {icon} {r.source_name:<25} │ {r.tier.value:<8} │ "
                  f"{r.count:>3} tenders │ {r.duration_seconds:>5.1f}s │ {r.status.value}")
            if r.error_message and r.count == 0:
                # Truncate long error messages
                err = r.error_message[:80] + "..." if len(r.error_message) > 80 else r.error_message
                print(f"     └─ {err}")

        print("─" * 70)

        # Platform breakdown
        platform_counts = {}
        for t in tenders:
            platform_counts[t.platform] = platform_counts.get(t.platform, 0) + 1

        print(f"  Total unique tenders: {len(tenders)}")
        for platform, count in sorted(platform_counts.items(), key=lambda x: -x[1]):
            print(f"    • {platform}: {count}")
        print(f"  Pipeline duration: {duration:.1f}s")
        print("=" * 70)
