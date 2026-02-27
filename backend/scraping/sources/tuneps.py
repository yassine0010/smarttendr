"""
SmartTender AI — TUNEPS Scraper (Tier 3: HTML + DataTables API)
================================================================
Scrapes Tunisian public procurement tenders from the official
marchespublics.gov.tn portal.

The original TUNEPS site (www.tuneps.tn) is a JavaScript SPA that
requires a headless browser.  However, the related government portal
at marchespublics.gov.tn exposes a server-side DataTables JSON
endpoint that we can query directly with HTTP.

Strategy: Send DataTables-style GET parameters to the appels-doffres
endpoint with ``X-Requested-With: XMLHttpRequest`` to receive JSON.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from ..base import (
    BaseScraper, NormalizedTender, ScraperConfig, ScraperResult,
    ScraperTier, ScrapeStatus,
)
from ..registry import ScraperRegistry


@ScraperRegistry.register("TUNEPS")
class TunepsScraper(BaseScraper):
    """
    Tunisian public procurement scraper.

    Uses the marchespublics.gov.tn DataTables server-side JSON API
    which returns structured tender data without requiring JavaScript.
    """

    BASE_URL = "https://www.marchespublics.gov.tn"
    TENDERS_URL = "https://www.marchespublics.gov.tn/fr/appels-doffres"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7,ar;q=0.6",
        "Accept-Encoding": "gzip, deflate, br",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
    }

    # DataTables column definitions (must match the server's expectations)
    _DT_COLUMNS = [
        {"data": "id", "name": "", "searchable": "true", "orderable": "true"},
        {"data": "organization.name_fr", "name": "organization.name_fr",
         "searchable": "true", "orderable": "true"},
        {"data": "title_fr", "name": "title_fr",
         "searchable": "true", "orderable": "true"},
        {"data": "tenderPeriod_endDate", "name": "tenderPeriod_endDate",
         "searchable": "true", "orderable": "true"},
        {"data": "publication_date", "name": "publication_date",
         "searchable": "true", "orderable": "true"},
        {"data": "reservedSME", "name": "reservedSME",
         "searchable": "true", "orderable": "true"},
    ]

    def __init__(self, config: ScraperConfig | None = None):
        if config is None:
            config = ScraperConfig(
                source_name="TUNEPS",
                tier=ScraperTier.T3_HTML,
                base_url=self.BASE_URL,
                max_results=20,
                timeout=25,
                delay=1.5,
                requests_per_minute=15,
                concurrent=1,
            )
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self, query: str = "") -> ScraperResult:
        """Scrape Tunisian tenders from marchespublics.gov.tn."""
        self.logger.info("scrape_start", query=query)
        try:
            return self._scrape_datatables(query)
        except httpx.ConnectError:
            self.logger.warning("portal_unreachable", url=self.TENDERS_URL)
            return ScraperResult(
                source_name="TUNEPS",
                tier=ScraperTier.T3_HTML,
                status=ScrapeStatus.ERROR,
                error_message="marchespublics.gov.tn unreachable",
            )
        except Exception as e:
            self.logger.error("scrape_error", error=str(e))
            return ScraperResult(
                source_name="TUNEPS",
                tier=ScraperTier.T3_HTML,
                status=ScrapeStatus.ERROR,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # Internal — DataTables server-side endpoint
    # ------------------------------------------------------------------

    def _build_dt_params(
        self,
        start: int = 0,
        length: int = 15,
        draw: int = 1,
        search: str = "",
    ) -> dict[str, str]:
        """Build DataTables-compatible query parameters."""
        params: dict[str, str] = {
            "draw": str(draw),
            "start": str(start),
            "length": str(length),
            "search[value]": search,
            "search[regex]": "false",
        }
        for i, col in enumerate(self._DT_COLUMNS):
            prefix = f"columns[{i}]"
            params[f"{prefix}[data]"] = col["data"]
            params[f"{prefix}[name]"] = col.get("name", "")
            params[f"{prefix}[searchable]"] = col.get("searchable", "true")
            params[f"{prefix}[orderable]"] = col.get("orderable", "true")
            params[f"{prefix}[search][value]"] = ""
            params[f"{prefix}[search][regex]"] = "false"
        return params

    def _scrape_datatables(self, query: str) -> ScraperResult:
        """Query the DataTables endpoint and parse JSON."""
        page_size = min(self.config.max_results, 25)
        all_tenders: list[NormalizedTender] = []
        pages_fetched = 0
        draw = 1

        with httpx.Client(
            timeout=self.config.timeout,
            follow_redirects=True,
            headers=self.HEADERS,
        ) as client:
            while len(all_tenders) < self.config.max_results:
                params = self._build_dt_params(
                    start=len(all_tenders),
                    length=page_size,
                    draw=draw,
                    search=query,
                )
                resp = client.get(self.TENDERS_URL, params=params)
                resp.raise_for_status()
                pages_fetched += 1

                data = resp.json()
                records = data.get("data", [])
                total = data.get("recordsFiltered", 0)

                self.logger.debug(
                    "page_fetched",
                    draw=draw,
                    records=len(records),
                    total=total,
                )

                if not records:
                    break

                for record in records:
                    tender = self._parse_record(record)
                    if tender:
                        all_tenders.append(tender)

                # Stop if we have all available records
                if len(all_tenders) >= total or len(all_tenders) >= self.config.max_results:
                    break

                draw += 1
                time.sleep(self.config.delay)

        all_tenders = all_tenders[: self.config.max_results]
        status = ScrapeStatus.SUCCESS if all_tenders else ScrapeStatus.EMPTY

        return ScraperResult(
            source_name="TUNEPS",
            tier=ScraperTier.T3_HTML,
            status=status,
            tenders=all_tenders,
            pages_fetched=pages_fetched,
            error_message="" if all_tenders else "No tenders found",
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_record(self, record: dict[str, Any]) -> NormalizedTender | None:
        """Convert a single DataTables JSON record to NormalizedTender."""
        tender_id = record.get("id", "")
        title = record.get("title_fr", "")

        if not title or len(title) < 3:
            return None

        # Organization
        org_data = record.get("organization", {})
        org_name = ""
        if isinstance(org_data, dict):
            org_name = org_data.get("name_fr", "")
        elif isinstance(org_data, str):
            org_name = org_data

        # Dates
        deadline = record.get("tenderPeriod_endDate", "")
        published = record.get("publication_date", "")

        # Clean dates (format: "2026-02-27 12:24:07" -> "2026-02-27")
        if deadline:
            deadline = deadline.split(" ")[0] if " " in deadline else deadline
        if published:
            published = published.split(" ")[0] if " " in published else published

        # SME reserved
        plan_data = record.get("plan", {})
        reserved_sme = False
        if isinstance(plan_data, dict):
            reserved_sme = plan_data.get("reservedSME", False)

        # Build source URL
        source_url = f"{self.TENDERS_URL}/{tender_id}" if tender_id else ""

        return NormalizedTender(
            source_id=f"TUNEPS-{tender_id}" if tender_id else f"TUNEPS-{hash(title) % 100000}",
            platform="TUNEPS",
            source_url=source_url,
            title=self.clean_text(title),
            description=self.clean_text(title),
            deadline=deadline,
            published_date=published,
            category="PME" if reserved_sme else "Public Procurement",
            country="Tunisia",
            location="Tunis, Tunisia",
            organization=org_name or "République Tunisienne",
        )
