# SmartTender AI — Scraping Architecture & Strategy

> **Version**: 2.0 | **Date**: 2026-02-27 | **Author**: SmartTender AI Team
> **Status**: Production-Ready Design

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Source Classification Strategy](#2-source-classification-strategy)
3. [Scraping Technology Stack](#3-scraping-technology-stack)
4. [Anti-Detection Strategy](#4-anti-detection-strategy)
5. [Scalability Architecture](#5-scalability-architecture)
6. [Data Normalization Layer](#6-data-normalization-layer)
7. [Monitoring & Maintenance](#7-monitoring--maintenance)
8. [Security & Compliance](#8-security--compliance)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Final Recommended Stack](#10-final-recommended-stack)

---

## 1. Executive Summary

SmartTender AI monitors **100+ tender sources** daily across government portals,
EU platforms, international organizations, and private procurement sites. Each
source falls into one of four tiers based on how data can be extracted. The
architecture is designed around a **tiered extraction pipeline** that
automatically selects the lightest, fastest, most reliable method available for
each source before escalating to heavier approaches.

**Core Principle**: *API first → RSS second → Static HTML third → Browser
automation last.*

---

## 2. Source Classification Strategy

### 2.1 Source Tiers

| Tier | Type | Extraction Method | Latency | Reliability | Examples |
|------|------|-------------------|---------|-------------|----------|
| **T1** | Structured API | REST/GraphQL client | ~200ms | ★★★★★ | SAM.gov API, TED Open Data, Contracts Finder UK |
| **T2** | RSS / Atom / XML | Feed parser | ~500ms | ★★★★☆ | EU TED RSS, Government feeds |
| **T3** | Static HTML | Requests + BeautifulSoup | ~1-3s | ★★★☆☆ | Small government portals, UNGM listing |
| **T4** | Dynamic JS / SPA | Headless browser (Scrapling/Playwright) | ~5-15s | ★★☆☆☆ | TUNEPS, TED search UI, dynamic platforms |

### 2.2 Decision Tree

```
Source → Has public API?
  ├─ YES → T1: Use API client (requests + JSON parsing)
  │         └─ Rate-limited? → Add exponential backoff + queue
  ├─ NO  → Has RSS/Atom feed?
  │   ├─ YES → T2: Use feedparser
  │   ├─ NO  → Is content in static HTML?
  │   │   ├─ YES → T3: Use Requests + BeautifulSoup
  │   │   ├─ NO  → JavaScript-rendered?
  │   │   │   ├─ YES → T4: Use Scrapling (stealth browser)
  │   │   │   └─ NO  → Protected by Cloudflare/CAPTCHA?
  │   │   │       ├─ YES → T4 + Proxy rotation + CAPTCHA solver
  │   │   │       └─ NO  → T3 with custom headers
```

### 2.3 Current Platform Classification

| Platform | Tier | Method | Notes |
|----------|------|--------|-------|
| **SAM.gov** | T1 | REST API v2 (`api.sam.gov/opportunities/v2/search`) | Requires API key (free DEMO_KEY available), date range mandatory |
| **TED Europa** | T1/T2 | TED Open Data API (POST `api.ted.europa.eu`) + RSS feeds | API requires POST with JSON body; also has CSV bulk data |
| **UNGM** | T3/T4 | AJAX POST to internal API + HTML parsing | Uses ASP.NET server-side; data loaded via `__doPostBack` |
| **TUNEPS** | T4 | Headless browser (Scrapling/Playwright) | Fully JS-rendered, requires session management |
| **Contracts Finder (UK)** | T1 | REST API | Open data, JSON responses |
| **DGMARKET** | T3 | Static HTML scraping | Paginated listings |
| **Private portals** | T3/T4 | Mixed | Login-gated, variable structure |

---

## 3. Scraping Technology Stack

### 3.1 Technology Selection Matrix

| Tool | Use Case | Pros | Cons |
|------|----------|------|------|
| **`requests`** | T1 APIs, T3 static pages | Fast, lightweight, no overhead | No JS rendering |
| **`httpx`** | Async API calls at scale | HTTP/2 support, async-native | Slightly more complex |
| **`BeautifulSoup4`** | T3 HTML parsing | Simple, tolerant of malformed HTML | Slow on large DOMs |
| **`lxml`** | High-performance T3 parsing | 10x faster than BS4, XPath support | Stricter parsing |
| **`Scrapy`** | Large-scale T3 crawling | Built-in concurrency, middleware, pipelines | Heavy for simple tasks |
| **`Scrapling`** | T4 JS-rendered sites | Auto-adapts to selector changes, stealth | Newer library, browser overhead |
| **`Playwright`** | T4 complex SPAs, fallback | Full browser control, multi-browser | Resource-heavy |
| **`feedparser`** | T2 RSS/Atom feeds | Universal feed parsing | Limited to feeds |

### 3.2 Recommended Stack per Tier

```
T1 (APIs):     httpx (async) + pydantic (response validation)
T2 (Feeds):    feedparser + httpx
T3 (Static):   httpx + BeautifulSoup4 (simple) / Scrapy (at scale)
T4 (Dynamic):  Scrapling (primary) / Playwright (fallback)
```

### 3.3 Proxy Strategy

**Architecture**: Rotating residential proxy pool

| Layer | Provider Type | Use Case |
|-------|--------------|----------|
| **Layer 0** | Direct (no proxy) | T1 APIs with legitimate API keys |
| **Layer 1** | Datacenter proxies | T3 static sites with no anti-bot |
| **Layer 2** | Residential proxies | T4 protected sites, rate-limited portals |
| **Layer 3** | ISP proxies | High-value targets with strict fingerprinting |

**Rotation Logic**: Round-robin per domain, sticky sessions for multi-page flows.

### 3.4 Rate Limiting Strategy

```python
RATE_LIMITS = {
    "sam.gov":      {"requests_per_minute": 60,  "concurrent": 5},
    "ted.europa.eu": {"requests_per_minute": 30,  "concurrent": 3},
    "ungm.org":     {"requests_per_minute": 20,  "concurrent": 2},
    "tuneps.tn":    {"requests_per_minute": 10,  "concurrent": 1},
    "default":      {"requests_per_minute": 15,  "concurrent": 2},
}
```

---

## 4. Anti-Detection Strategy

### 4.1 Ethical Framework

1. **Always check `robots.txt`** before scraping any new source
2. **Prefer APIs and open data** over scraping wherever available
3. **Respect rate limits** — never exceed what a human user would generate
4. **Identify yourself** — set a real `User-Agent` with contact info for T1/T2
5. **Cache aggressively** — never re-scrape data that hasn't changed
6. **Comply with ToS** — if a site explicitly prohibits scraping, use their API or skip

### 4.2 Technical Anti-Detection

| Technique | Implementation |
|-----------|---------------|
| **User-Agent rotation** | Pool of 50+ real browser UAs, rotated per request |
| **Header fingerprint** | Full realistic header set (Accept, Accept-Language, Accept-Encoding, Connection) |
| **Request timing** | Gaussian-distributed delays (μ=2s, σ=0.5s) between requests |
| **Session management** | Persistent cookies per domain, session reuse |
| **TLS fingerprint** | Use `curl_cffi` or `tls_client` for JA3 fingerprint rotation |
| **Referer chain** | Build natural navigation paths (homepage → search → result) |

### 4.3 Caching Strategy

```
Layer 1: In-memory LRU cache (TTL: 5 min) — de-bounce duplicate requests
Layer 2: Redis cache (TTL: 1 hour) — API responses, parsed pages
Layer 3: SQLite/PostgreSQL (TTL: 24 hours) — full tender records
Layer 4: Raw HTML archive (permanent) — for audit, re-parsing, legal compliance
```

### 4.4 Retry & Backoff

```python
RETRY_CONFIG = {
    "max_retries": 3,
    "backoff_base": 2,          # seconds
    "backoff_multiplier": 2,     # exponential: 2, 4, 8
    "backoff_max": 60,           # cap at 60s
    "retry_on": [429, 500, 502, 503, 504],
    "circuit_breaker_threshold": 5,  # consecutive failures to trip
    "circuit_breaker_reset": 300,    # seconds before retry after trip
}
```

---

## 5. Scalability Architecture

### 5.1 Async vs Sync

| Component | Model | Reasoning |
|-----------|-------|-----------|
| T1 API calls | **Async** (httpx + asyncio) | I/O-bound, massively parallelizable |
| T2 Feed parsing | **Async** (httpx + feedparser) | Many feeds, minimal compute |
| T3 HTML scraping | **Async** (Scrapy or httpx) | Network-bound, concurrent pages |
| T4 Browser automation | **Sync with pool** (ProcessPoolExecutor) | Browser instances are CPU-heavy |
| NLP processing | **Sync with threads** | CPU-bound, benefit from process parallelism |

### 5.2 Queue System

**Choice**: Redis (via `rq` or `celery[redis]`)

**Why Redis over RabbitMQ**: Simpler deployment, acts as both queue and cache,
sufficient durability for our use case (tenders are re-scraped daily anyway).

```
┌─────────────┐     ┌──────────┐     ┌────────────────┐
│  Scheduler   │────▶│  Redis    │────▶│  Workers (N)   │
│  (cron/beat) │     │  Queue    │     │  T1/T2/T3/T4   │
└─────────────┘     └──────────┘     └────────────────┘
                         │                    │
                         ▼                    ▼
                    ┌──────────┐     ┌────────────────┐
                    │  Cache    │     │  PostgreSQL DB  │
                    │  (Redis)  │     │  (normalized)   │
                    └──────────┘     └────────────────┘
```

### 5.3 Distributed Crawling

- **Worker scaling**: Each worker handles one domain at a time (domain-locked)
- **Concurrency**: `max_workers = min(cpu_count * 2, 16)` for T1-T3
- **Browser pool**: Pre-warmed Scrapling/Playwright instances (max 4 per node)
- **Scheduling**: Staggered cron jobs to avoid burst load

### 5.4 Containerization

```yaml
# Docker Compose topology
services:
  scheduler:     # Celery Beat — triggers scraping tasks on schedule
  worker-api:    # T1/T2 workers (lightweight, high concurrency)
  worker-html:   # T3 workers (medium, Scrapy-based)
  worker-browser: # T4 workers (heavy, Playwright + Scrapling)
  redis:         # Queue + cache
  postgres:      # Persistent storage
  monitoring:    # Prometheus + Grafana
```

---

## 6. Data Normalization Layer

### 6.1 Unified Tender Schema

Every tender, regardless of source, is normalized to:

```python
@dataclass
class NormalizedTender:
    # Identity
    source_id: str          # Original ID from platform
    platform: str           # Source platform name
    source_url: str         # Direct URL to tender

    # Core fields
    title: str              # Tender title
    description: str        # Full description (max 10K chars)
    organization: str       # Issuing organization
    category: str           # Mapped to internal taxonomy

    # Dates
    published_date: datetime
    deadline: datetime
    last_updated: datetime

    # Financial
    budget_amount: float | None
    budget_currency: str | None

    # Location
    country: str
    region: str | None

    # Skills & Requirements
    required_skills: list[str]
    required_certifications: list[str]

    # Metadata
    raw_html: str           # Original HTML for audit
    scrape_timestamp: datetime
    content_hash: str       # SHA-256 for deduplication
```

### 6.2 PDF Handling

Many tenders attach detailed specs as PDFs:

1. **Detection**: Check for `.pdf` links in tender pages
2. **Download**: Fetch PDF to local storage / S3
3. **Extraction**: `PyMuPDF` (fitz) for text, `pdfplumber` for tables
4. **Fallback**: `Tesseract OCR` for scanned documents
5. **Indexing**: Extracted text appended to `description` field

### 6.3 Deduplication Strategy

```
Step 1: Exact match on (platform + source_id)
Step 2: Content hash match (SHA-256 of title + description)
Step 3: Fuzzy match (Levenshtein distance on title, threshold=0.85)
Step 4: Semantic dedup (SBERT embedding cosine similarity > 0.95)
```

---

## 7. Monitoring & Maintenance

### 7.1 Selector Breakage Detection

```python
# After each scrape run, validate:
assert len(results) > 0,              "Zero results — possible selector breakage"
assert len(results) >= min_expected,  "Below threshold — partial breakage"
assert all(r['title'] for r in results), "Empty titles — field selector broken"
assert all(len(r['description']) > 20 for r in results), "Descriptions too short"
```

### 7.2 Alerting Rules

| Condition | Severity | Action |
|-----------|----------|--------|
| 0 results from any platform | 🔴 Critical | Alert + fallback to cached data |
| Results drop > 50% vs 7-day avg | 🟡 Warning | Alert + log for review |
| Response time > 3× baseline | 🟡 Warning | Check rate limits / proxy |
| 3 consecutive failures | 🔴 Critical | Circuit breaker trips, alert |
| New fields detected in API | 🔵 Info | Log for schema review |

### 7.3 Logging

```
Structured JSON logs → stdout → collected by Loki/ELK
Fields: timestamp, source, tier, status_code, items_count, duration_ms, error
```

### 7.4 Performance Metrics (Prometheus)

- `scraper_requests_total{source, status}`
- `scraper_items_scraped{source}`
- `scraper_duration_seconds{source, tier}`
- `scraper_errors_total{source, error_type}`
- `scraper_cache_hit_ratio{source}`

---

## 8. Security & Compliance

### 8.1 Legal Framework

| Source Type | Legal Basis | Risk Level |
|-------------|-------------|------------|
| Public APIs with keys | Authorized use | ✅ Low |
| Open government data | FOI / Open Data directives | ✅ Low |
| Public HTML (no login) | Fair use (public procurement) | 🟡 Medium |
| Login-required portals | Terms of Service apply | 🔴 High — review ToS |
| Private corporate pages | May require partnership | 🔴 High |

### 8.2 GDPR Considerations

- **Public procurement data is generally not personal data**
- **Contact persons** named in tenders: store only name + role (public officials exception under GDPR Art. 6(1)(e))
- **Do not scrape**: bidder personal data, employee information
- **Retention**: Raw HTML archived max 2 years, then anonymized

### 8.3 Robots.txt Policy

```python
# ALWAYS check before first scrape of any domain
from urllib.robotparser import RobotFileParser

def can_fetch(url: str) -> bool:
    rp = RobotFileParser()
    rp.set_url(f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt")
    rp.read()
    return rp.can_fetch("SmartTenderBot/1.0", url)
```

---

## 9. Architecture Diagram

```
                          ┌─────────────────────────────────────┐
                          │         SCHEDULER (Celery Beat)      │
                          │   Cron: every 6h / daily / weekly    │
                          └──────────────┬──────────────────────┘
                                         │ dispatch tasks
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          REDIS (Queue + Cache)                         │
│   Queues: q:api, q:feed, q:html, q:browser, q:nlp                    │
│   Cache:  tender:{hash}, page:{url_hash}, robots:{domain}            │
└───┬──────────┬──────────────┬───────────────┬────────────────┬────────┘
    │          │              │               │                │
    ▼          ▼              ▼               ▼                ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────────┐
│Worker  │ │Worker  │ │Worker    │ │Worker       │ │Worker        │
│T1: API │ │T2: RSS │ │T3: HTML  │ │T4: Browser  │ │NLP Pipeline  │
│(httpx) │ │(feed-  │ │(httpx+   │ │(Scrapling/  │ │(spaCy+SBERT) │
│        │ │parser) │ │ BS4)     │ │ Playwright) │ │              │
└───┬────┘ └───┬────┘ └────┬─────┘ └──────┬──────┘ └──────┬───────┘
    │          │           │              │                │
    └──────────┴───────────┴──────┬───────┘                │
                                  ▼                        │
                    ┌─────────────────────────┐            │
                    │  NORMALIZER             │            │
                    │  Raw → NormalizedTender  │────────────┘
                    │  + PDF extraction        │
                    │  + Deduplication          │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  PostgreSQL              │
                    │  ├─ tenders (normalized) │
                    │  ├─ raw_pages (audit)    │
                    │  ├─ scrape_logs          │
                    │  └─ source_config        │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Module 1 (Detection)    │
                    │  NLP scoring + ranking   │
                    │  → Alert system          │
                    └─────────────────────────┘
```

---

## 10. Final Recommended Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **HTTP Client** | `httpx` (async) | HTTP/2, async-native, connection pooling |
| **HTML Parser** | `BeautifulSoup4` + `lxml` | Tolerant + fast combo |
| **Browser Automation** | `Scrapling` (primary), `Playwright` (fallback) | Scrapling auto-adapts selectors; Playwright for complex SPAs |
| **Feed Parser** | `feedparser` | Universal RSS/Atom support |
| **Queue** | `Redis` + `celery` | Simple, proven, dual-use as cache |
| **Database** | `PostgreSQL` | JSONB for flexible tender storage, full-text search |
| **NLP** | `spaCy` + `sentence-transformers` | NER + semantic embeddings |
| **PDF** | `PyMuPDF` + `pdfplumber` | Fast text + table extraction |
| **Monitoring** | `structlog` + `prometheus_client` | Structured logs + metrics |
| **Containerization** | `Docker Compose` (dev), `Kubernetes` (prod) | Standard orchestration |

### Cost-Reliability-Maintainability Trade-offs

| Dimension | Choice | Trade-off |
|-----------|--------|-----------|
| **Reliability** | Multi-tier fallback (API → HTML → Browser) | Higher complexity, but near-zero downtime |
| **Cost** | Self-hosted workers + free API keys | No SaaS scraping fees; requires DevOps effort |
| **Maintainability** | Scrapling auto-adapting selectors | Reduces manual selector maintenance by ~80% |
| **Scale** | Celery workers + Redis | Horizontal scaling; Redis is single-point (use Sentinel for HA) |

---

*This architecture is implemented in `backend/scraping/` as a modular pipeline.*
