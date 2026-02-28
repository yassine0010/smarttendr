# Scraping Module — Architecture

> **Describes only what is implemented in code.** No planned features.

---

## Overview

The scraping module collects tender notices from 5 government procurement platforms using HTTP requests. It produces a list of `NormalizedTender` objects saved as JSON files. There is no database — output goes to `output/` as flat JSON.

**Total: 1,691 lines across 10 files.**

---

## File Map

```
backend/scraping/
├── base.py              (285 lines)  Data models + abstract base class
├── pipeline.py          (338 lines)  Orchestrates all scrapers sequentially
├── registry.py          (75 lines)   Plugin registry with @register decorator
├── __init__.py
└── sources/
    ├── sam_gov.py           (205 lines)  SAM.gov REST API
    ├── ted_europa.py        (471 lines)  TED Europa API + RSS + HTML fallback
    ├── ungm.py              (265 lines)  UNGM AJAX POST + HTML parsing
    ├── tuneps.py            (252 lines)  Tunisia DataTables JSON API
    └── contracts_finder.py  (175 lines)  UK Contracts Finder REST API
```

---

## Data Models (`base.py`)

### `NormalizedTender`
Every scraper converts platform-specific data into this dataclass:
```
source_id, platform, source_url
title, description, organization, category
published_date, deadline
budget, budget_amount, budget_currency
country, region, location
required_skills[], required_certifications[]
raw_html, scrape_timestamp, content_hash (SHA-256 of title+description)
```

### `ScraperConfig`
Per-source settings: `source_name`, `tier`, `base_url`, `max_results`, `timeout`, `delay`, `max_retries`, `api_key`, `custom_headers`.

### `ScraperResult`
Output of one scraper run: `tenders[]`, `status` (success/partial/empty/error/rate_limited/blocked), `duration_seconds`, `error_message`.

### `ScraperTier` (enum)
Classification only — does not change behavior:
- `T1_API` — SAM.gov, Contracts Finder, TED
- `T3_HTML` — UNGM
- `T4_BROWSER` — TUNEPS (label only; actual implementation uses HTTP, not a browser)

---

## Registry (`registry.py`)

A class-level dictionary `_registry: Dict[str, Type[BaseScraper]]`.

Each scraper self-registers via decorator:
```python
@ScraperRegistry.register("SAM.GOV")
class SamGovScraper(BaseScraper):
    ...
```

Pipeline looks up scrapers by name:
```python
scraper = ScraperRegistry.create("SAM.GOV", config)
```

---

## Pipeline (`pipeline.py`)

### What `pipeline.run()` does (sequential, not async):

1. **Loop** over enabled source configs (`SAM.GOV`, `TED`, `UNGM`, `TUNEPS`, `CONTRACTS_FINDER`)
2. **Create** scraper instance from registry
3. **Call** `scraper._timed_scrape(query)` — wraps `scrape()` with timing + exception handling
4. **Collect** all `NormalizedTender` objects
5. **Deduplicate** — two-pass: SHA-256 content hash, then lowercase title match
6. **Fallback** — if zero tenders scraped, load from `data/sample_tenders.json`
7. **Save** to `output/`:
   - `scraped_tenders_{timestamp}.json` — all tenders
   - `scraped_tenders_latest.json` — overwritten each run
   - `scrape_report_{timestamp}.json` — per-source status/count/duration
8. **Print** summary table to console

### Default configs (hardcoded in pipeline.py):
| Source | max_results | timeout | delay | api_key |
|--------|------------|---------|-------|---------|
| SAM.GOV | 25 | 30s | 1.5s | `DEMO_KEY` |
| TED | 25 | 30s | 1.5s | none |
| UNGM | 25 | 30s | 2.0s | none |
| TUNEPS | 20 | 20s | 2.0s | none |
| CONTRACTS_FINDER | 25 | 30s | 1.5s | none |

---

## Scrapers (sources/)

### SAM.gov (`sam_gov.py`, 205 lines)

**Method:** GET to `https://api.sam.gov/opportunities/v2/search`

**Auth:** `api_key=DEMO_KEY` (query param). Free, limited to 1000 calls/day.

**Parameters sent:**
- `api_key`, `limit` (max_results)
- `postedFrom` (30 days ago), `postedTo` (today), format `MM/dd/yyyy`
- `ptype` (procurement type filter)
- `keyword` (search query)

**Response parsing:** JSON → `opportunitiesData[]` array. Each item maps:
- `title` → title
- `fullParentPathName` → organization
- `archiveDate` → deadline
- `postedDate` → published_date
- `naicsCode` → category
- `placeOfPerformance.state.name` → location
- `uiLink` → source_url

**Error handling:** Returns `ScrapeStatus.ERROR` on HTTP failure, `ScrapeStatus.EMPTY` on 0 results.

---

### TED Europa (`ted_europa.py`, 471 lines)

**Three fallback strategies executed in order:**

**Strategy 1 — TED API v3:**
- POST to `https://api.ted.europa.eu/v3/notices/search`
- JSON body with `query` (free text), `fields` list, `pageSize`, `page`
- Parses response JSON `notices[]` array
- Extracts: title (from `title-proc` or `title-lot`), description (`description-glo`), organization (`buyer-name`), deadline (`deadline-receipt-tender`), country (`addr-country`), budget (`val-total`)

**Strategy 2 — RSS feeds:**
- GET `https://ted.europa.eu/rss/search.rss?q={query}`
- Parses with `feedparser` library
- Extracts from RSS items: title, link, published date, description (HTML stripped)

**Strategy 3 — HTML search:**
- GET `https://ted.europa.eu/en/search/result?q={query}`
- Parses HTML with BeautifulSoup
- Extracts from search result cards

**Each strategy catches exceptions and falls through to the next.**

---

### UNGM (`ungm.py`, 265 lines)

**Method:** POST JSON to `https://www.ungm.org/Public/Notice/Search`

**Two-step process:**
1. GET the notice page (`/Public/Notice`) to obtain session cookies
2. POST search request with cookies + AJAX headers

**POST payload:**
```json
{
  "PageIndex": 0,
  "PageSize": 15,
  "Title": "",
  "Description": "",
  "DeadlineFrom": "/Date({today_ms})/",
  "PublishedTo": "/Date({today_ms})/",
  "SortField": "DatePublished",
  "SortAscending": false,
  "IsActive": true,
  "IsCancelled": false,
  "UNSPSCs": [],
  "Countries": [],
  "Agencies": []
}
```

**Required headers:** `X-Requested-With: XMLHttpRequest`, `Content-Type: application/json`

**Response:** HTML fragment (not JSON). Contains `div.dataRow` elements. Each row parsed:
- `div.resultTitle a` → title + notice ID (from href `/Public/Notice/{id}`)
- `div.resultInfo1` with class `deadline` → deadline text
- `div.resultAgency` → UN agency name
- Notice type from row metadata

**Pagination:** Loops `PageIndex` from 0 until no more `dataRow` elements found or `max_results` reached.

---

### TUNEPS (`tuneps.py`, 252 lines)

**Method:** GET with DataTables server-side parameters to `https://www.marchespublics.gov.tn/fr/appels-doffres`

**Key detail:** The same URL serves HTML (normal request) and JSON (when `X-Requested-With: XMLHttpRequest` header is sent). The JSON response uses jQuery DataTables server-side protocol.

**Parameters sent:** Standard DataTables format:
- `draw` (request counter), `start` (offset), `length` (page size)
- `columns[i][data]`, `columns[i][name]`, `columns[i][searchable]`, `columns[i][orderable]` for 10 columns
- `search[value]` (search query), `search[regex]` = false
- `order[0][column]`, `order[0][dir]`

**Response JSON:**
```json
{
  "recordsTotal": 93123,
  "recordsFiltered": 93123,
  "data": [
    {
      "id": "...",
      "title_fr": "...",
      "organization": { "name_fr": "..." },
      "tenderPeriod_endDate": "...",
      "publication_date": "...",
      "plan": { "reservedSME": true/false }
    }
  ]
}
```

**Each record mapped to:** title (`title_fr`), organization (`organization.name_fr`), deadline (`tenderPeriod_endDate`), published date, source URL (`/fr/appels-doffres/{id}`), country = "Tunisia".

**Pagination:** Increments `start` by `length` until `start >= recordsFiltered` or `max_results` reached.

---

### Contracts Finder UK (`contracts_finder.py`, 175 lines)

**Method:** GET to `https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search`

**Auth:** None required (open data).

**Parameters:** `publishedFrom` (30 days ago, ISO format), `publishedTo` (today), `size` (max_results), `keyword`.

**Response:** OCDS (Open Contracting Data Standard) JSON. Structure:
```json
{
  "releases": [
    {
      "id": "...",
      "tender": {
        "title": "...",
        "description": "...",
        "tenderPeriod": { "endDate": "..." },
        "value": { "amount": 0, "currency": "GBP" }
      },
      "buyer": { "name": "..." },
      "tag": ["tender"]
    }
  ]
}
```

**Mapping:** title, description, deadline (`tenderPeriod.endDate`), budget (`value.amount` + `value.currency`), organization (`buyer.name`), country = "United Kingdom".

---

## What Does NOT Exist in Code

| Feature | Status |
|---------|--------|
| PostgreSQL / any database | ❌ Output is JSON files only |
| Redis / SQLite cache | ❌ No caching layer |
| Circuit breaker | ❌ Just try/except |
| Rate limiter | ❌ Config field exists, not enforced |
| Alert system | ❌ Errors print to console only |
| Async execution | ❌ Scrapers run sequentially |
| Docker / docker-compose | ❌ No Dockerfile |
| Proxy rotation | ❌ Config field exists, never used |

---

## Dependencies

| Library | Used for |
|---------|----------|
| `httpx` | All HTTP requests (GET/POST) |
| `beautifulsoup4` | HTML parsing (UNGM, TED fallback) |
| `feedparser` | RSS parsing (TED fallback) |
| `structlog` | Structured logging to console |
