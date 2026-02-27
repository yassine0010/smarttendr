"""
SmartTender AI — Contracts Finder UK Scraper (Tier 1: API)
============================================================
Scrapes the UK Government Contracts Finder using their public API.

API Docs: https://www.contractsfinder.service.gov.uk/apidocumentation
Endpoint: https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search
Auth:     No API key required (fully open data)

This is an exemplary Tier 1 source — clean REST API, JSON responses,
no authentication required. Included as a reliable fallback source.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from ..base import (
    BaseScraper, NormalizedTender, ScraperConfig, ScraperResult,
    ScraperTier, ScrapeStatus,
)
from ..registry import ScraperRegistry


@ScraperRegistry.register("CONTRACTS_FINDER")
class ContractsFinderScraper(BaseScraper):
    """
    UK Contracts Finder API client.
    
    Uses the OCDS (Open Contracting Data Standard) search endpoint.
    Returns well-structured JSON with releases following the OCDS schema.
    """

    API_URL = "https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search"

    def __init__(self, config: ScraperConfig | None = None):
        if config is None:
            config = ScraperConfig(
                source_name="Contracts Finder UK",
                tier=ScraperTier.T1_API,
                base_url=self.API_URL,
                max_results=25,
                timeout=30,
                requests_per_minute=30,
            )
        super().__init__(config)

    def scrape(self, query: str = "IT services") -> ScraperResult:
        """
        Search UK Contracts Finder for opportunities.

        Args:
            query: Keyword to search for

        Returns:
            ScraperResult with normalized tenders
        """
        self.logger.info("scrape_start", query=query)

        # Date range: last 90 days
        now = datetime.now(timezone.utc)
        published_from = (now - timedelta(days=90)).strftime("%Y-%m-%dT00:00:00Z")
        published_to = now.strftime("%Y-%m-%dT23:59:59Z")

        params = {
            "searchCriteria.keyword": query,
            "searchCriteria.publishedFrom": published_from,
            "searchCriteria.publishedTo": published_to,
            "searchCriteria.stages": "tender",
            "size": min(self.config.max_results, 100),
            "page": 1,
        }

        tenders = []
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.get(self.API_URL, params=params)
                response.raise_for_status()
                data = response.json()

                releases = data.get("releases", [])
                self.logger.info("api_response", releases_count=len(releases))

                for release in releases:
                    tender = self._normalize(release)
                    if tender.title:
                        tenders.append(tender)

            status = ScrapeStatus.SUCCESS if tenders else ScrapeStatus.EMPTY
            return ScraperResult(
                source_name="Contracts Finder UK",
                tier=ScraperTier.T1_API,
                status=status,
                tenders=tenders,
                pages_fetched=1,
            )

        except httpx.HTTPStatusError as e:
            self.logger.error("api_error", status=e.response.status_code)
            return ScraperResult(
                source_name="Contracts Finder UK",
                tier=ScraperTier.T1_API,
                status=ScrapeStatus.ERROR,
                error_message=f"HTTP {e.response.status_code}",
            )
        except Exception as e:
            self.logger.error("scrape_error", error=str(e))
            return ScraperResult(
                source_name="Contracts Finder UK",
                tier=ScraperTier.T1_API,
                status=ScrapeStatus.ERROR,
                error_message=str(e),
            )

    def _normalize(self, release: dict) -> NormalizedTender:
        """Convert a Contracts Finder OCDS release to NormalizedTender."""
        # OCDS structure: release > tender
        tender_data = release.get("tender", {})
        
        # Title and description
        title = tender_data.get("title", release.get("tag", [""])[0] if release.get("tag") else "")
        description = tender_data.get("description", "")

        # Organization (buyer)
        buyer = release.get("buyer", {})
        org_name = buyer.get("name", "")

        # Dates
        published = release.get("date", "")
        deadline = ""
        tender_period = tender_data.get("tenderPeriod", {})
        if tender_period:
            deadline = tender_period.get("endDate", "")

        # Budget
        budget_str = ""
        budget_amount = None
        budget_currency = None
        value = tender_data.get("value", {})
        if value:
            budget_amount = value.get("amount")
            budget_currency = value.get("currency", "GBP")
            if budget_amount:
                budget_str = f"{budget_amount:,.0f} {budget_currency}"

        # OCDS ID
        ocid = release.get("ocid", "")
        notice_id = ocid.split("-")[-1] if ocid else str(hash(title) % 100000)

        # Classification
        category = "Government Services"
        classifications = tender_data.get("classification", {})
        if classifications:
            scheme = classifications.get("description", "")
            if any(kw in scheme.lower() for kw in ["it", "software", "digital", "computer"]):
                category = "IT Services"

        return NormalizedTender(
            source_id=f"CF-{notice_id}",
            platform="Contracts Finder UK",
            source_url=f"https://www.contractsfinder.service.gov.uk/Notice/{notice_id}",
            title=title,
            description=self.clean_text(description) if description else title,
            organization=org_name,
            category=category,
            published_date=published,
            deadline=deadline,
            budget=budget_str,
            budget_amount=budget_amount,
            budget_currency=budget_currency,
            country="United Kingdom",
            location="United Kingdom",
        )
