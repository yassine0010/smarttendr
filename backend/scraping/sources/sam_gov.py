"""
SmartTender AI — SAM.gov Scraper (Tier 1: API)
================================================
Uses the official SAM.gov Opportunities Public API v2.

API Docs: https://open.gsa.gov/api/get-opportunities-public-api/
Endpoint: https://api.sam.gov/opportunities/v2/search
Auth:     Free API key (DEMO_KEY for testing, register at SAM.gov for production)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from ..base import (
    BaseScraper, NormalizedTender, ScraperConfig, ScraperResult,
    ScraperTier, ScrapeStatus,
)
from ..registry import ScraperRegistry


@ScraperRegistry.register("SAM.GOV")
class SamGovScraper(BaseScraper):
    """
    SAM.gov API client.
    
    The API requires:
    - api_key (DEMO_KEY works for testing, limited to 1000 calls/day)
    - postedFrom and postedTo (mandatory, max 1 year range)
    - Returns JSON with opportunitiesData array
    """

    API_URL = "https://api.sam.gov/opportunities/v2/search"

    def __init__(self, config: ScraperConfig | None = None):
        if config is None:
            config = ScraperConfig(
                source_name="SAM.gov",
                tier=ScraperTier.T1_API,
                base_url=self.API_URL,
                max_results=25,
                timeout=30,
                requests_per_minute=60,
                api_key="DEMO_KEY",
            )
        super().__init__(config)

    def scrape(self, query: str = "information technology") -> ScraperResult:
        """
        Fetch opportunities from SAM.gov API.

        Args:
            query: Search keyword (searches title and description)

        Returns:
            ScraperResult with normalized tenders
        """
        self.logger.info("scrape_start", query=query)

        # Build date range (last 6 months to now)
        now = datetime.now(timezone.utc)
        posted_from = (now - timedelta(days=180)).strftime("%m/%d/%Y")
        posted_to = now.strftime("%m/%d/%Y")

        params = {
            "api_key": self.config.api_key or "DEMO_KEY",
            "postedFrom": posted_from,
            "postedTo": posted_to,
            "limit": min(self.config.max_results, 1000),
            "offset": 0,
        }

        # Add keyword search if query provided
        # Note: 'title' is a keyword filter, 'ptype' filters by procurement type
        if query:
            params["title"] = query

        tenders = []
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                self.logger.debug("api_request", url=self.API_URL, params=params)
                response = client.get(self.API_URL, params=params)
                response.raise_for_status()
                data = response.json()

                total_records = data.get("totalRecords", 0)
                opportunities = data.get("opportunitiesData", [])

                self.logger.info(
                    "api_response",
                    total_records=total_records,
                    returned=len(opportunities),
                )

                for opp in opportunities:
                    tender = self._normalize(opp)
                    tenders.append(tender)

            status = ScrapeStatus.SUCCESS if tenders else ScrapeStatus.EMPTY
            return ScraperResult(
                source_name="SAM.gov",
                tier=ScraperTier.T1_API,
                status=status,
                tenders=tenders,
                pages_fetched=1,
            )

        except httpx.HTTPStatusError as e:
            self.logger.error("api_http_error", status=e.response.status_code)
            return ScraperResult(
                source_name="SAM.gov",
                tier=ScraperTier.T1_API,
                status=ScrapeStatus.RATE_LIMITED if e.response.status_code == 429 else ScrapeStatus.ERROR,
                error_message=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            self.logger.error("api_error", error=str(e))
            return ScraperResult(
                source_name="SAM.gov",
                tier=ScraperTier.T1_API,
                status=ScrapeStatus.ERROR,
                error_message=str(e),
            )

    def _normalize(self, opp: dict) -> NormalizedTender:
        """Convert a SAM.gov opportunity JSON to NormalizedTender."""
        # Extract organization name
        org = opp.get("fullParentPathName", "")
        if not org:
            org = opp.get("organizationName", "US Government")

        # Extract point of contact
        poc = ""
        contacts = opp.get("pointOfContact", [])
        if contacts and isinstance(contacts, list):
            poc = contacts[0].get("fullname", "")

        # Build description from available fields
        description = opp.get("title", "")
        desc_link = opp.get("description", "")
        if desc_link and isinstance(desc_link, str) and not desc_link.startswith("http"):
            description = desc_link

        # Classify category based on NAICS code
        naics = opp.get("naicsCode", "")
        category = self._naics_to_category(naics)

        # Build source URL
        ui_link = opp.get("uiLink", "")
        if not ui_link:
            notice_id = opp.get("noticeId", "")
            if notice_id:
                ui_link = f"https://sam.gov/opp/{notice_id}/view"

        return NormalizedTender(
            source_id=f"SAM-{opp.get('noticeId', 'unknown')}",
            platform="SAM.gov",
            source_url=ui_link,
            title=opp.get("title", "Untitled"),
            description=self.clean_text(description),
            organization=org,
            category=category,
            published_date=opp.get("postedDate", ""),
            deadline=opp.get("responseDeadLine", ""),
            country="United States",
            location=self._extract_location(opp),
            required_skills=[],
            required_certifications=[],
        )

    @staticmethod
    def _naics_to_category(naics: str) -> str:
        """Map NAICS code prefix to human-readable category."""
        if not naics:
            return "Government Services"
        prefix = naics[:2]
        mapping = {
            "51": "IT Services",
            "54": "Professional Services",
            "56": "Administrative Services",
            "33": "Manufacturing",
            "23": "Construction",
            "48": "Transportation",
            "62": "Healthcare",
            "61": "Education",
            "52": "Finance",
            "92": "Public Administration",
        }
        return mapping.get(prefix, "Government Services")

    @staticmethod
    def _extract_location(opp: dict) -> str:
        """Extract location from place of performance."""
        pop = opp.get("placeOfPerformance", {})
        if isinstance(pop, dict):
            city = pop.get("city", {})
            state = pop.get("state", {})
            city_name = city.get("name", "") if isinstance(city, dict) else ""
            state_name = state.get("name", "") if isinstance(state, dict) else ""
            parts = [p for p in [city_name, state_name] if p]
            if parts:
                return ", ".join(parts) + ", USA"
        return "United States"
