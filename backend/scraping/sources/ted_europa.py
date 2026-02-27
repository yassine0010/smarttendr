"""
SmartTender AI — TED Europa Scraper (Tier 1: API + Tier 2: RSS)
================================================================
Multi-strategy scraper for TED (Tenders Electronic Daily).

Strategy 1 (Primary):  TED API v3 — POST-based search API
Strategy 2 (Fallback): TED RSS feeds — structured XML feeds
Strategy 3 (Fallback): TED Open Data — CSV bulk download

TED API v3 requires POST method with JSON body.
RSS feeds at: https://ted.europa.eu/rss/search.rss
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import feedparser
import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

from ..base import (
    BaseScraper, NormalizedTender, ScraperConfig, ScraperResult,
    ScraperTier, ScrapeStatus,
)
from ..registry import ScraperRegistry


@ScraperRegistry.register("TED")
class TedEuropaScraper(BaseScraper):
    """
    TED Europa scraper using multiple fallback strategies.
    
    1. Try TED API v3 (POST search)
    2. Fall back to RSS feeds
    3. Fall back to HTML search results
    """

    API_URL = "https://api.ted.europa.eu/v3/notices/search"
    RSS_URL = "https://ted.europa.eu/rss/search.rss"
    SEARCH_URL = "https://ted.europa.eu/en/search/result"

    # TED API v3 fields for notice metadata
    API_FIELDS = [
        "title-proc",
        "title-lot",
        "description-glo",
        "organisation-name-buyer",
        "organisation-country-buyer",
        "deadline-receipt-tender-date-lot",
        "publication-date",
        "contract-nature-main-proc",
    ]

    def __init__(self, config: ScraperConfig | None = None):
        if config is None:
            config = ScraperConfig(
                source_name="TED Europa",
                tier=ScraperTier.T1_API,
                base_url=self.API_URL,
                max_results=25,
                timeout=30,
                requests_per_minute=30,
            )
        super().__init__(config)

    def scrape(self, query: str = "IT services") -> ScraperResult:
        """
        Scrape TED Europa with cascading fallback strategies.
        """
        self.logger.info("scrape_start", query=query)

        # Strategy 1: TED API v3
        result = self._scrape_via_api(query)
        if result.status == ScrapeStatus.SUCCESS and result.count > 0:
            return result

        self.logger.info("api_fallback", reason=result.error_message or "empty results")

        # Strategy 2: RSS Feed
        result = self._scrape_via_rss(query)
        if result.status == ScrapeStatus.SUCCESS and result.count > 0:
            return result

        self.logger.info("rss_fallback", reason=result.error_message or "empty results")

        # Strategy 3: HTML search page
        result = self._scrape_via_html(query)
        return result

    # ----------------------------------------------------------------
    # Strategy 1: TED API v3 (POST)
    # ----------------------------------------------------------------
    def _scrape_via_api(self, query: str) -> ScraperResult:
        """
        Use TED API v3 with POST request and expert query syntax.
        
        The TED API uses expert query language:
        - FT = "keyword" for full-text search
        - PD > YYYYMMDD for publication date filter
        - AND/OR/NOT for combining conditions
        
        Returns structured notice data with title, description, org, deadline.
        """
        try:
            # Build expert query: full-text search + recent publications
            from datetime import datetime, timedelta, timezone
            cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y%m%d")
            expert_query = f'FT = "{query}" AND PD > {cutoff}'

            payload = {
                "query": expert_query,
                "fields": self.API_FIELDS,
                "page": 1,
                "limit": min(self.config.max_results, 100),
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            with httpx.Client(timeout=self.config.timeout) as client:
                self.logger.debug("api_request", query=expert_query)
                response = client.post(
                    self.API_URL,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                total = data.get("totalNoticeCount", 0)
                notices = data.get("notices", [])

                self.logger.info(
                    "api_response",
                    total_available=total,
                    returned=len(notices),
                )

                tenders = []
                for notice in notices:
                    tender = self._normalize_api_notice(notice)
                    if tender.title:
                        tenders.append(tender)

                return ScraperResult(
                    source_name="TED Europa",
                    tier=ScraperTier.T1_API,
                    status=ScrapeStatus.SUCCESS if tenders else ScrapeStatus.EMPTY,
                    tenders=tenders,
                    pages_fetched=1,
                )

        except httpx.HTTPStatusError as e:
            msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            self.logger.error("api_http_error", error=msg)
            return ScraperResult(
                source_name="TED Europa",
                tier=ScraperTier.T1_API,
                status=ScrapeStatus.ERROR,
                error_message=f"API error: {msg}",
            )
        except Exception as e:
            return ScraperResult(
                source_name="TED Europa",
                tier=ScraperTier.T1_API,
                status=ScrapeStatus.ERROR,
                error_message=f"API error: {e}",
            )

    # ----------------------------------------------------------------
    # Strategy 2: RSS Feeds
    # ----------------------------------------------------------------
    def _scrape_via_rss(self, query: str) -> ScraperResult:
        """Parse TED RSS feed for latest notices."""
        try:
            # TED RSS feed supports query parameters — URL-encode the query
            encoded_query = quote_plus(query)
            rss_url = f"{self.RSS_URL}?q={encoded_query}&sortField=PD&sortOrder=desc"
            
            self.logger.debug("rss_fetch", url=rss_url)

            # feedparser can fetch the URL directly
            feed = feedparser.parse(rss_url)

            if feed.bozo and not feed.entries:
                return ScraperResult(
                    source_name="TED Europa",
                    tier=ScraperTier.T2_FEED,
                    status=ScrapeStatus.ERROR,
                    error_message=f"RSS parse error: {feed.bozo_exception}",
                )

            tenders = []
            for entry in feed.entries[:self.config.max_results]:
                tender = self._normalize_rss_entry(entry)
                tenders.append(tender)

            return ScraperResult(
                source_name="TED Europa",
                tier=ScraperTier.T2_FEED,
                status=ScrapeStatus.SUCCESS if tenders else ScrapeStatus.EMPTY,
                tenders=tenders,
                pages_fetched=1,
            )

        except Exception as e:
            return ScraperResult(
                source_name="TED Europa",
                tier=ScraperTier.T2_FEED,
                status=ScrapeStatus.ERROR,
                error_message=f"RSS error: {e}",
            )

    # ----------------------------------------------------------------
    # Strategy 3: HTML Search Results
    # ----------------------------------------------------------------
    def _scrape_via_html(self, query: str) -> ScraperResult:
        """Scrape TED search results HTML page."""
        try:
            params = {
                "q": query,
                "sortField": "PD",
                "sortOrder": "desc",
            }

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }

            with httpx.Client(timeout=self.config.timeout, follow_redirects=True) as client:
                response = client.get(
                    self.SEARCH_URL,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                html = response.text

            soup = BeautifulSoup(html, "html.parser")

            # TED search results may use various selectors depending on version
            tenders = []
            
            # Try multiple selector strategies
            cards = (
                soup.select("div.notice-search-result") or
                soup.select("div[class*='result']") or
                soup.select("article") or
                soup.select("tr[data-notice-id]") or
                []
            )

            for card in cards[:self.config.max_results]:
                tender = self._normalize_html_card(card, html)
                if tender.title:
                    tenders.append(tender)

            # If no structured results found, TED is likely JS-rendered
            if not tenders:
                self.logger.warning(
                    "html_no_results",
                    hint="TED search page is JavaScript-rendered, "
                         "content not available via static HTML",
                )

            return ScraperResult(
                source_name="TED Europa",
                tier=ScraperTier.T3_HTML,
                status=ScrapeStatus.SUCCESS if tenders else ScrapeStatus.EMPTY,
                tenders=tenders,
                pages_fetched=1,
                error_message="" if tenders else "JS-rendered page, no static content",
            )

        except Exception as e:
            return ScraperResult(
                source_name="TED Europa",
                tier=ScraperTier.T3_HTML,
                status=ScrapeStatus.ERROR,
                error_message=f"HTML error: {e}",
            )

    # ----------------------------------------------------------------
    # Normalizers
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Helpers for TED API v3 multilingual field extraction
    # ----------------------------------------------------------------
    @staticmethod
    def _extract_text(field_value) -> str:
        """
        Extract the best text from a TED API v3 field value.

        Field values are multilingual dicts like:
            {"eng": "IT services", "fra": "Services informatiques"}
        or sometimes a plain string, a list, or nested structure.
        Prefer English, then French, then any first value.
        """
        if field_value is None:
            return ""
        if isinstance(field_value, str):
            return field_value
        if isinstance(field_value, list):
            # Some fields return a list of multilingual dicts or strings
            parts = [TedEuropaScraper._extract_text(item) for item in field_value]
            # For single-element lists, return the element directly
            non_empty = [p for p in parts if p]
            if len(non_empty) == 1:
                return non_empty[0]
            return " | ".join(non_empty)
        if isinstance(field_value, dict):
            # Prefer English → French → first available key
            for lang in ("eng", "en", "fra", "fr"):
                if lang in field_value:
                    val = field_value[lang]
                    return TedEuropaScraper._extract_text(val)
            # Return first value if any
            vals = list(field_value.values())
            return TedEuropaScraper._extract_text(vals[0]) if vals else ""
        return str(field_value)

    def _normalize_api_notice(self, notice: dict) -> NormalizedTender:
        """
        Normalize a TED API v3 notice JSON.

        Each notice looks like:
        {
            "publication-number": "123456-2025",
            "title-proc": {"eng": "...", "fra": "..."},
            "description-glo": {"eng": "...", ...},
            "organisation-name-buyer": {"eng": "Some Org"},
            "organisation-country-buyer": "FRA",
            "deadline-receipt-tender-date-lot": "2025-06-30+02:00",
            "publication-date": "2025-04-01",
            "contract-nature-main-proc": "services",
            "links": {"xml": {...}, "pdf": {...}, "html": {...}}
        }
        """
        pub_number = notice.get("publication-number", "")
        
        # Build source URL from links or publication number
        links = notice.get("links", {})
        source_url = ""
        if isinstance(links, dict):
            for fmt in ("html", "pdf", "xml"):
                link_obj = links.get(fmt, {})
                if isinstance(link_obj, dict):
                    # TED uses 3-letter ISO codes: ENG, FRA, DEU, etc.
                    for lang in ("ENG", "eng", "en", "FRA", "fra", "fr", "MUL"):
                        if lang in link_obj:
                            source_url = link_obj[lang]
                            break
                    if not source_url and link_obj:
                        # Just take the first available language
                        source_url = next(iter(link_obj.values()))
                elif isinstance(link_obj, str):
                    source_url = link_obj
                if source_url:
                    break
        if not source_url and pub_number:
            source_url = f"https://ted.europa.eu/en/notice/-/detail/{pub_number}"

        # Extract fields
        title = self._extract_text(notice.get("title-proc"))
        if not title:
            title = self._extract_text(notice.get("title-lot"))
        
        description = self._extract_text(notice.get("description-glo"))
        organization = self._extract_text(notice.get("organisation-name-buyer"))
        country = self._extract_text(notice.get("organisation-country-buyer")) or "EU"
        deadline = self._extract_text(notice.get("deadline-receipt-tender-date-lot"))
        pub_date = self._extract_text(notice.get("publication-date"))
        nature = self._extract_text(notice.get("contract-nature-main-proc"))

        # Clean deadline (remove timezone offset like +02:00)
        if deadline and "+" in deadline:
            deadline = deadline.split("+")[0]
        # Clean publication date similarly
        if pub_date and "+" in pub_date:
            pub_date = pub_date.split("+")[0]

        return NormalizedTender(
            source_id=f"TED-{pub_number}" if pub_number else f"TED-{hash(title) % 100000}",
            platform="TED Europa",
            source_url=source_url,
            title=title,
            description=self.clean_text(description) if description else title,
            organization=organization,
            category=f"Public Procurement — {nature}" if nature else "Public Procurement",
            published_date=pub_date,
            deadline=deadline,
            country=country,
        )

    def _normalize_rss_entry(self, entry) -> NormalizedTender:
        """Normalize an RSS feed entry."""
        # Extract document ID from link
        link = entry.get("link", "")
        doc_id = ""
        id_match = re.search(r'/(\d{6}-\d{4})', link)
        if id_match:
            doc_id = id_match.group(1)

        # Parse description HTML
        desc_html = entry.get("summary", entry.get("description", ""))
        desc_soup = BeautifulSoup(desc_html, "html.parser")
        description = desc_soup.get_text(separator=" ", strip=True)

        # Extract country from categories
        country = "EU"
        categories = entry.get("tags", [])
        if categories:
            for tag in categories:
                term = tag.get("term", "")
                if len(term) == 2 and term.isupper():
                    country = term
                    break

        return NormalizedTender(
            source_id=f"TED-{doc_id}" if doc_id else f"TED-{hash(link) % 100000}",
            platform="TED Europa",
            source_url=link,
            title=entry.get("title", "Untitled TED Notice"),
            description=self.clean_text(description) if description else entry.get("title", ""),
            organization=entry.get("author", ""),
            category="Public Procurement",
            published_date=entry.get("published", ""),
            deadline="",
            country=country,
        )

    def _normalize_html_card(self, card, raw_html: str) -> NormalizedTender:
        """Normalize an HTML search result card."""
        # Try various selectors for title
        title_el = (
            card.select_one("h3 a") or
            card.select_one(".title a") or
            card.select_one("a[href*='notice']") or
            card.select_one("a")
        )
        title = title_el.get_text(strip=True) if title_el else ""
        link = title_el.get("href", "") if title_el else ""
        if link and not link.startswith("http"):
            link = f"https://ted.europa.eu{link}"

        # Description
        desc_el = card.select_one(".description, .summary, p")
        description = desc_el.get_text(strip=True) if desc_el else title

        return NormalizedTender(
            source_id=f"TED-HTML-{hash(title) % 100000}",
            platform="TED Europa",
            source_url=link,
            title=title,
            description=self.clean_text(description),
            category="Public Procurement",
            country="EU",
            raw_html=raw_html[:5000] if raw_html else "",
        )
