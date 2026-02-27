"""
SmartTender AI — UNGM Scraper (Tier 3: HTML + AJAX)
=====================================================
Scrapes the United Nations Global Marketplace (UNGM).

UNGM renders a search page and loads results via a JSON POST
to /Public/Notice/Search which returns HTML table rows.

Strategy: POST JSON search criteria to the UNGM Notice Search
endpoint, then parse the returned HTML fragments.
"""

from __future__ import annotations

import re
import time
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from ..base import (
    BaseScraper, NormalizedTender, ScraperConfig, ScraperResult,
    ScraperTier, ScrapeStatus,
)
from ..registry import ScraperRegistry


@ScraperRegistry.register("UNGM")
class UngmScraper(BaseScraper):
    """
    UNGM scraper using the internal Notice Search AJAX endpoint.

    UNGM's notice page at /Public/Notice loads results via a POST
    to /Public/Notice/Search with a JSON body.  The response is an
    HTML fragment containing ``div.dataRow`` elements.
    """

    BASE_URL = "https://www.ungm.org"
    NOTICE_URL = "https://www.ungm.org/Public/Notice"
    SEARCH_URL = "https://www.ungm.org/Public/Notice/Search"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    AJAX_HEADERS = {
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }

    def __init__(self, config: ScraperConfig | None = None):
        if config is None:
            config = ScraperConfig(
                source_name="UNGM",
                tier=ScraperTier.T3_HTML,
                base_url=self.NOTICE_URL,
                max_results=25,
                timeout=30,
                delay=2.0,
                requests_per_minute=20,
            )
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self, query: str = "") -> ScraperResult:
        """Scrape UNGM for open procurement notices."""
        self.logger.info("scrape_start", query=query)
        try:
            return self._scrape_via_search_api(query)
        except Exception as e:
            self.logger.error("scrape_error", error=str(e))
            return ScraperResult(
                source_name="UNGM",
                tier=ScraperTier.T3_HTML,
                status=ScrapeStatus.ERROR,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # Internal — AJAX search endpoint
    # ------------------------------------------------------------------

    def _build_search_payload(
        self, query: str = "", page: int = 0, page_size: int = 15,
    ) -> dict:
        """Build the JSON payload expected by /Public/Notice/Search."""
        today = datetime.now().strftime("%d-%b-%Y")
        return {
            "PageIndex": page,
            "PageSize": page_size,
            "Title": query,
            "Description": "",
            "Reference": "",
            "PublishedFrom": "",
            "PublishedTo": today,
            "DeadlineFrom": today,
            "DeadlineTo": "",
            "Countries": [],
            "Agencies": [],
            "UNSPSCs": [],
            "NoticeTypes": [],
            "SortField": "DatePublished",
            "SortAscending": False,
            "isPicker": False,
            "IsSustainable": False,
            "IsActive": True,
            "NoticeDisplayType": None,
            "NoticeSearchTotalLabelId": "noticeSearchTotal",
            "TypeOfCompetitions": [],
        }

    def _scrape_via_search_api(self, query: str) -> ScraperResult:
        """POST to /Public/Notice/Search and parse the HTML response."""
        page_size = min(self.config.max_results, 15)
        pages_needed = max(1, (self.config.max_results + page_size - 1) // page_size)
        all_tenders: list[NormalizedTender] = []
        pages_fetched = 0

        with httpx.Client(
            timeout=self.config.timeout,
            follow_redirects=True,
            headers=self.HEADERS,
        ) as client:
            # Establish session cookies
            client.get(self.NOTICE_URL)
            time.sleep(0.5)

            for page_idx in range(pages_needed):
                if len(all_tenders) >= self.config.max_results:
                    break

                payload = self._build_search_payload(
                    query=query,
                    page=page_idx,
                    page_size=page_size,
                )

                resp = client.post(
                    self.SEARCH_URL,
                    json=payload,
                    headers={
                        **self.HEADERS,
                        **self.AJAX_HEADERS,
                        "Referer": self.NOTICE_URL,
                    },
                )
                resp.raise_for_status()
                pages_fetched += 1

                tenders = self._parse_search_response(resp.text)
                self.logger.debug(
                    "page_parsed",
                    page=page_idx,
                    found=len(tenders),
                )
                if not tenders:
                    break

                all_tenders.extend(tenders)

                if page_idx < pages_needed - 1:
                    time.sleep(self.config.delay)

        all_tenders = all_tenders[: self.config.max_results]
        status = ScrapeStatus.SUCCESS if all_tenders else ScrapeStatus.EMPTY

        return ScraperResult(
            source_name="UNGM",
            tier=ScraperTier.T3_HTML,
            status=status,
            tenders=all_tenders,
            pages_fetched=pages_fetched,
            error_message="" if all_tenders else "No open notices found",
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_search_response(self, html: str) -> list[NormalizedTender]:
        """Parse the HTML fragment returned by /Public/Notice/Search."""
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select("div.dataRow, div.tableRow.dataRow")
        tenders: list[NormalizedTender] = []

        for row in rows:
            tender = self._parse_row(row)
            if tender:
                tenders.append(tender)

        return tenders

    def _parse_row(self, row) -> NormalizedTender | None:
        """Parse a single dataRow div from the UNGM search results."""
        notice_id = row.get("data-noticeid", "")

        # --- title ---
        title_cell = row.select_one("div.resultTitle")
        title = ""
        if title_cell:
            # The visible title text (not "Open in a new window")
            texts = [
                t.strip()
                for t in title_cell.stripped_strings
                if t.strip() and t.strip() != "Open in a new window"
            ]
            title = texts[0] if texts else ""

        if not title or len(title) < 5:
            return None

        # --- deadline ---
        deadline_cell = row.select_one("div.resultInfo1.deadline, div.deadline")
        deadline = ""
        if deadline_cell:
            raw = deadline_cell.get_text(strip=True)
            m = re.search(r"(\d{1,2}-\w{3}-\d{4})", raw)
            if m:
                deadline = m.group(1)

        # --- published date ---
        cells = row.select("div.tableCell")
        published = ""
        if len(cells) >= 4:
            raw = cells[3].get_text(strip=True)
            m = re.search(r"(\d{1,2}-\w{3}-\d{4})", raw)
            if m:
                published = m.group(1)

        # --- agency ---
        agency_cell = row.select_one("div.resultAgency")
        agency = agency_cell.get_text(strip=True) if agency_cell else "United Nations"

        # --- notice type ---
        notice_type = ""
        if len(cells) >= 6:
            notice_type = cells[5].get_text(strip=True)

        # --- source URL ---
        source_url = f"{self.BASE_URL}/Public/Notice/{notice_id}" if notice_id else ""

        return NormalizedTender(
            source_id=f"UNGM-{notice_id}" if notice_id else f"UNGM-{hash(title) % 100000}",
            platform="UNGM",
            source_url=source_url,
            title=self.clean_text(title),
            description=self.clean_text(title),
            deadline=deadline,
            published_date=published,
            category=notice_type or "International Procurement",
            country="International",
            organization=agency,
        )
