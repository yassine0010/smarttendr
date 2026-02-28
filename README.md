# SmartTender AI 🎯

An AI-powered Tender Intelligence Platform that **scrapes tenders** from 5 government platforms, **extracts keywords** using NLP (spaCy + TF-IDF), and **ranks them** by relevance to your company profile.

---

## What It Does (Big Picture)

```
Government Websites          NLP Engine                 AI Scoring
┌──────────────┐        ┌──────────────────┐       ┌──────────────┐
│ SAM.gov      │        │ 1. Clean text    │       │ Sentence-BERT│
│ TED Europa   │───────▶│ 2. Extract NER   │──────▶│ + Skill match│──▶ Ranked Results
│ UNGM         │ scrape │ 3. TF-IDF keys   │ JSON  │ + Domain fit │
│ TUNEPS       │        │ 4. Merge & rank  │       │              │
│ ContractsFinder│       └──────────────────┘       └──────────────┘
└──────────────┘
```

**Input:** Government tender websites  
**Output:** Ranked list of tenders scored by relevance (0–100%) to your company

---

## Project Structure — What Each File Does

```
smart tender/
│
├── backend/
│   ├── module1_tender_detection.py   ← 🚀 MAIN ENTRY POINT (run this)
│   ├── utils.py                      ← Helper functions (JSON loading)
│   │
│   ├── scraping/                     ← MODULE 0: Web Scraping
│   │   ├── base.py                   ← Base scraper class + data models
│   │   ├── pipeline.py               ← Orchestrates all scrapers
│   │   ├── registry.py               ← Auto-discovers scraper plugins
│   │   └── sources/
│   │       ├── sam_gov.py            ← SAM.gov API scraper (US tenders)
│   │       ├── ted_europa.py         ← TED Europa API scraper (EU tenders)
│   │       ├── ungm.py              ← UNGM AJAX scraper (UN tenders)
│   │       ├── tuneps.py            ← Tunisia DataTables scraper
│   │       └── contracts_finder.py  ← UK Contracts Finder API
│   │
│   └── nlp/                          ← MODULE 1: Keyword Extraction (NLP)
│       ├── keyword_extraction.py     ← 🧠 Main pipeline orchestrator
│       ├── preprocessing.py          ← Text cleaning & normalization
│       ├── ner_extractor.py          ← spaCy NER + rule-based entities
│       ├── tfidf_extractor.py        ← TF-IDF keyword & domain detection
│       └── taxonomy.py               ← Domain dictionary & skill patterns
│
├── data/
│   ├── sample_tenders.json           ← Test data (3 sample tenders)
│   └── sample_cvs.json              ← Test CV profiles
│
├── docs/
│   ├── SCRAPING_ARCHITECTURE.md      ← Scraping module design doc
│   └── KEYWORD_EXTRACTION_ARCHITECTURE.md ← NLP module design doc
│
└── output/                           ← Generated results (auto-created)
    ├── scraped_tenders_latest.json
    └── analysis_results.json
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/yassine0010/smarttendr.git
cd smarttendr

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install spacy scikit-learn sentence-transformers numpy httpx beautifulsoup4 structlog

# 4. Download spaCy English model
python -m spacy download en_core_web_sm
```

---

## Quick Start — Run Everything

```bash
source venv/bin/activate
cd "smart tender"
python backend/module1_tender_detection.py
```

This will:
1. **Scrape** tenders from all 5 platforms
2. **Extract** keywords, budget, deadline, skills from each tender using NLP
3. **Score** each tender's relevance to the company profile (Inetum by default)
4. **Print** ranked results and save to `output/analysis_results.json`

---

## Module-by-Module Explanation with Examples

### 📦 Module 0: Web Scraping (`backend/scraping/`)

**What it does:** Collects raw tenders from 5 government websites.

Each platform has a different strategy:

| Platform | Country | Method | File |
|----------|---------|--------|------|
| SAM.gov | 🇺🇸 USA | REST API | `sam_gov.py` |
| TED Europa | 🇪🇺 EU | REST API | `ted_europa.py` |
| UNGM | 🇺🇳 UN | AJAX POST + HTML parsing | `ungm.py` |
| TUNEPS | 🇹🇳 Tunisia | DataTables JSON API | `tuneps.py` |
| Contracts Finder | 🇬🇧 UK | REST API | `contracts_finder.py` |

**Example — scrape only UNGM:**
```python
from backend.scraping.pipeline import ScrapingPipeline

pipeline = ScrapingPipeline()
tenders = pipeline.run(query="IT services", sources=["UNGM"])

for t in tenders[:3]:
    print(f"  {t.title} | Deadline: {t.deadline}")
```

**Example output:**
```
  Supply of ICT Equipment for UNDP | Deadline: 2026-03-20
  Web Development Services for WHO  | Deadline: 2026-04-01
  Cloud Migration Support - UNICEF  | Deadline: 2026-03-15
```

---

### 🧠 Module 1: Keyword Extraction (`backend/nlp/`)

This is the NLP brain. It takes raw tender text and extracts structured information.

#### How the Pipeline Works (4 stages):

```
Raw Tender Text
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: PREPROCESSING (preprocessing.py)               │
│   • Unicode normalize    • Remove HTML tags              │
│   • Fix OCR noise        • Remove bullet markers         │
│   • Normalize whitespace • Truncate to 50K chars         │
└────────────────────────┬────────────────────────────────┘
                         │ cleaned text
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: NER EXTRACTION (ner_extractor.py)               │
│   • spaCy NER → ORG, GPE, MONEY, DATE entities          │
│   • Regex patterns → budget amounts (€1.2M, 500K TND)   │
│   • Signal-word matching → find submission deadline      │
└────────────────────────┬────────────────────────────────┘
                         │ organizations, budget, deadline, locations
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: TF-IDF KEYWORDS (tfidf_extractor.py)            │
│   • Sentence-level TF-IDF → top 30 keywords             │
│   • Taxonomy lookup → domain classification              │
│   • Regex skill patterns → technology detection          │
└────────────────────────┬────────────────────────────────┘
                         │ keywords, domain, skills
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: MERGE (keyword_extraction.py)                   │
│   • Combine NER + TF-IDF results                         │
│   • Fallback to scraper metadata if NLP misses           │
│   • Output → ExtractionResult (JSON)                     │
└─────────────────────────────────────────────────────────┘
```

---

#### Example — Full Extraction

```python
from backend.nlp.keyword_extraction import KeywordExtractor

extractor = KeywordExtractor()

text = """
The Ministry of Digital Economy in Tunis is seeking proposals for
a cloud-based ERP system. Budget: 500,000 TND. Deadline for
submission: March 15, 2026. The system must be built with Python,
Django, deployed on AWS using Docker and Kubernetes. Agile
methodology required. ISO 27001 compliance mandatory.
"""

result = extractor.extract(text, title="Cloud ERP Development")
```

**What comes out:**
```json
{
  "domain": "ERP",
  "domains": ["ERP", "Cloud Computing", "IT Services"],
  "skills": [
    {"name": "Python",     "category": "Programming Language", "importance_score": 1.0},
    {"name": "Django",     "category": "Framework",            "importance_score": 0.95},
    {"name": "AWS",        "category": "Cloud Platform",       "importance_score": 0.95},
    {"name": "Docker",     "category": "DevOps",               "importance_score": 0.90},
    {"name": "Kubernetes", "category": "DevOps",               "importance_score": 0.90}
  ],
  "budget": "500,000 TND",
  "deadline": "2026-03-15",
  "organization": "Ministry of Digital Economy",
  "location": "Tunis",
  "certifications": ["ISO 27001"],
  "top_keywords": [
    {"term": "erp", "score": 0.12},
    {"term": "cloud", "score": 0.11},
    {"term": "cloud based", "score": 0.10}
  ]
}
```

---

#### Each Sub-Module Explained:

##### 1. `preprocessing.py` — Text Cleaning

**What it does:** Takes dirty text (HTML, OCR noise, weird formatting) → clean text.

```python
from backend.nlp.preprocessing import preprocess_text

dirty = """
<p>  The  Ministry &amp; Office  is seeking •  Python developers
===================================
Page 3 of 12
for a  ﬁnancial  system  </p>
"""

clean = preprocess_text(dirty)
# → "The Ministry & Office is seeking Python developers for a financial system"
```

**What it handles:**
| Problem | Example | Fix |
|---------|---------|-----|
| HTML tags | `<p>text</p>` | Stripped |
| HTML entities | `&amp;` | Replaced with `&` |
| OCR ligatures | `ﬁ`, `ﬂ` | → `fi`, `fl` |
| Bullet markers | `• item`, `► item` | Stripped |
| Page numbers | `Page 3 of 12` | Removed |
| Separator lines | `==========` | Removed |
| Multi-spaces | `the    ministry` | → `the ministry` |

---

##### 2. `ner_extractor.py` — Named Entity Recognition

**What it does:** Uses spaCy + regex to find organizations, locations, money, dates.

**spaCy NER** detects entities statistically:
```
"The Ministry of Finance in Tunis needs a $500,000 system by March 2026"
         ORG ──────────      GPE     MONEY ─────────    DATE ──────────
```

**Regex rules** catch what spaCy misses (especially budget formats):
```
Budget patterns matched:
  "$500,000"     → amount=500000, currency=USD
  "€1.2M"        → amount=1200000, currency=EUR
  "500 000 TND"  → amount=500000, currency=TND
  "1,200,000 DT" → amount=1200000, currency=TND
```

**Deadline selection** — when there are multiple dates, it picks the one near signal words:
```
"Published: January 10, 2026. Deadline for submission: March 15, 2026.
 Project starts: June 1, 2026."

  → 3 dates found
  → "March 15, 2026" selected (near "Deadline for submission")
```

---

##### 3. `tfidf_extractor.py` — TF-IDF Keyword Engine

**What it does:** Finds the most important/distinguishing words in a tender.

**Why TF-IDF, not just word counting?**
```
Word "shall"   → appears in EVERY tender    → TF-IDF score: 0.001 (filtered out)
Word "python"  → appears in THIS tender     → TF-IDF score: 0.15  (important!)
Word "docker"  → appears in THIS tender     → TF-IDF score: 0.12  (important!)
```

TF-IDF = Term Frequency × Inverse Document Frequency. High score = term is **distinctive** to this document.

**Domain Classification** — maps keywords to sectors using `taxonomy.py`:
```
Keywords: ["cloud", "erp", "microservices", "aws"]
                ↓ taxonomy lookup
Domains:  ["Cloud Computing", "ERP", "IT Services"]
```

**Skill Detection** — regex patterns find technology names:
```
Text: "Experience with Python, Django, and AWS required"
                          ↓ pattern matching
Skills: [
  Python     (Programming Language, score=1.0)
  Django     (Framework, score=0.95)
  AWS        (Cloud Platform, score=0.95)
]
```

Skills are ranked by: `category_weight × (1 + tfidf_score)`

---

##### 4. `taxonomy.py` — Knowledge Dictionaries

**What it contains:**

| Dictionary | Count | Purpose |
|-----------|-------|---------|
| `DOMAIN_TAXONOMY` | 100+ entries | Maps keywords → domain (e.g. "docker" → "DevOps") |
| `SKILL_PATTERNS` | 90+ patterns | Regex patterns for each technology |
| `SKILL_CATEGORIES_WEIGHT` | 13 categories | Priority weights (Programming=1.0, OS=0.6) |
| `CURRENCY_MAP` | 25 currencies | Symbol/code → ISO (€→EUR, TND, DT→TND) |
| `DEADLINE_SIGNALS` | 20+ phrases | "deadline for submission", "due date", "date limite"... |
| `TENDER_STOPWORDS` | 50+ words | Words to ignore: "shall", "pursuant", "herein"... |

---

### 📊 Module 2: Relevance Scoring (in `module1_tender_detection.py`)

**What it does:** Scores how relevant each tender is to your company.

**Three scoring components:**

```
┌──────────────────────────────────────────┐
│  RELEVANCE SCORE (weighted combination)  │
├──────────────────────────────────────────┤
│  45%  Semantic Similarity                │ ← Sentence-BERT embeddings
│       "How similar is tender text        │    cosine similarity
│        to company description?"          │
├──────────────────────────────────────────┤
│  35%  Skill Overlap                      │ ← Jaccard matching
│       "How many required skills          │    matched_skills / total_skills
│        does the company have?"           │
├──────────────────────────────────────────┤
│  20%  Domain Match                       │ ← Category alignment
│       "Is the tender in our domain?"     │    1.0 if match, 0.3 if not
└──────────────────────────────────────────┘
```

**Example:**
```
Tender: "Cloud ERP System with Python, AWS, Docker"
Company: Inetum (IT services, Python, AWS, Docker expertise)

  Semantic similarity: 0.72  (high — IT topics align)
  Skill overlap:       3/4   (Python, AWS, Docker match)
  Domain match:        1.0   (IT Services = IT Services)

  Score = 0.45×0.72 + 0.35×0.75 + 0.20×1.0 = 0.324 + 0.263 + 0.200 = 78.6%
  → ✅ RELEVANT (above 30% threshold)
```

---

## Edge Cases Handled

| Edge Case | How It's Handled |
|-----------|-----------------|
| **Missing budget** | Falls back to scraper metadata, then shows "Not specified" |
| **Multiple deadlines** | Signal-word proximity scoring picks the submission date |
| **Multi-domain tenders** | Returns all matching domains ranked by score |
| **Scanned PDF (OCR noise)** | `preprocessing.py` fixes ligatures, garbage chars, broken words |
| **Multi-language (EN/FR/AR)** | French month names, Arabic deadline signals, TND/DT currency |
| **Empty/short text** | Returns empty `ExtractionResult` gracefully |
| **Very long documents** | Truncated to 50K chars, spaCy max_length set to 200K |

---

## Full Run Example

```bash
$ python backend/module1_tender_detection.py

[Module 1] Loading NLP models...
[Module 1] Models loaded successfully!

[Step 1] Running multi-tier scraping pipeline...
  ✓ SAM.GOV       — 32 tenders
  ✓ TED            — 50 tenders
  ✓ UNGM           — 25 tenders
  ✓ TUNEPS         — 20 tenders
  ✓ CONTRACTS_FINDER — 19 tenders

[OK] Using 146 tenders from scraping pipeline

[Step 2] Analyzing 146 tenders with NLP engine...

======================================================================
 SMARTTENDER AI - TENDER DETECTION RESULTS
======================================================================

✅ Cloud ERP System Development
   Score: 78.6% | Status: RELEVANT
   Platform: TUNEPS | Deadline: 2026-03-15
   Domain: ERP | Budget: 500,000 TND
   Organization: Ministry of Digital Economy
   Skills: Python, AWS, Docker, Kubernetes, SAP

✅ AI-Powered Chatbot for Banking
   Score: 65.2% | Status: RELEVANT
   Platform: TED Europa | Deadline: 2026-04-01
   Domain: AI/Machine Learning | Budget: €200,000
   Skills: Python, TensorFlow, NLP, Salesforce

⬜ Bridge Construction - Highway A7
   Score: 12.1% | Status: LOW MATCH
   Platform: TUNEPS | Deadline: 2026-05-20
   Domain: Construction | Budget: 2,000,000 TND

======================================================================
 Summary: 83/146 tenders are relevant (threshold: 30%)
======================================================================
```

---

## When to Upgrade (Performance Notes)

| Current | When to Upgrade | Upgrade To |
|---------|----------------|------------|
| **spaCy `en_core_web_sm`** (15MB) | Extraction accuracy <85% | `en_core_web_trf` (transformer, 400MB) or fine-tune on tender data |
| **TF-IDF** for keywords | Need synonym matching ("ML" = "Machine Learning") | SBERT embeddings for zero-shot classification |
| **Regex skill patterns** | >150 skills to maintain | Train a custom spaCy NER model for SKILL entities |
| **Flat taxonomy dict** | Need hierarchical domains | Ontology (OWL) or knowledge graph |
| **Sequential processing** | >1000 tenders/batch | `spacy.pipe()` batch mode + multiprocessing |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Scraping | `httpx`, `beautifulsoup4`, `structlog` |
| NER | `spaCy` (en_core_web_sm) |
| Keyword Extraction | `scikit-learn` TF-IDF |
| Semantic Scoring | `sentence-transformers` (all-MiniLM-L6-v2) |
| Language | Python 3.13 |

---

## License

SmartTender AI Team © 2025-2026
