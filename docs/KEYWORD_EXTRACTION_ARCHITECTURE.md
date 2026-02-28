# Keyword Extraction Module — Architecture

> **Describes only what is implemented in code.** No planned features.

---

## Overview

The NLP module takes raw tender text and extracts structured fields using spaCy NER + TF-IDF. It supports **multilingual tenders** (French, Arabic, English, 50+ languages) via automatic language detection and language-specific spaCy models. It runs in-process (no external services, no database). Output is an `ExtractionResult` dataclass.

**Total: ~2,100 lines across 5 files.**

---

## File Map

```
backend/nlp/
├── keyword_extraction.py  (389 lines)  Main pipeline — public API
├── preprocessing.py       (239 lines)  Text cleaning & normalization
├── ner_extractor.py       (530 lines)  spaCy NER + regex patterns
├── tfidf_extractor.py     (361 lines)  TF-IDF keywords + domain/skill detection
└── taxonomy.py            (374 lines)  Dictionaries: domains, skills, currencies, stopwords
```

---

## Pipeline Flow

```
extractor.extract(text, title="...")
    │
    ▼
[1] preprocess_text(text)           ← preprocessing.py
    Unicode normalize, strip HTML, fix OCR, normalize whitespace
    │
    ▼  cleaned string
[1.5] _detect_language(cleaned)     ← langdetect
    Auto-detect language from first 1000 chars
    fr → fr_core_news_sm | en → en_core_web_sm | ar/other → en fallback
    │
    ▼  language code ("en", "fr", "ar", ...)
[2] nlp_model(cleaned)              ← spaCy (language-specific model)
    Tokenization, POS tagging, NER, dependency parsing
    │
    ▼  spaCy Doc object
[3] extract_entities(doc)           ← ner_extractor.py
    spaCy NER → ORG, GPE, MONEY, DATE
    Regex → budget amounts, date patterns
    Signal words → pick submission deadline from multiple dates
    │
    ▼  NERResult (orgs, locations, budget, deadline, dates)
[4] segment_sentences(doc) + lemmatize(doc)  ← preprocessing.py
    │
    ▼  sentence list + lemma list
[5] extract_tfidf_keywords(sentences, text)  ← tfidf_extractor.py
    TF-IDF on sentences → top 30 keywords
    Taxonomy lookup → domain classification
    Regex skill patterns → technology detection
    │
    ▼  TFIDFResult (keywords, domains, skills)
[6] _merge_results(ner, tfidf, metadata)     ← keyword_extraction.py
    Combine NER + TF-IDF → ExtractionResult
    Fall back to scraper metadata for missing fields
```

---

## Output Format

`ExtractionResult.to_dict()` returns:
```json
{
  "domain": "Cloud Computing",
  "domains": ["Cloud Computing", "IT Services"],
  "skills": [
    {"name": "Python", "category": "Programming Language", "importance_score": 1.0},
    {"name": "AWS", "category": "Cloud Platform", "importance_score": 0.95}
  ],
  "budget": "500,000 TND",
  "budget_amount": 500000.0,
  "budget_currency": "TND",
  "deadline": "2026-03-15",
  "deadline_raw": "March 15, 2026",
  "organization": "Ministry of Digital Economy",
  "organizations": ["Ministry of Digital Economy"],
  "location": "Tunis",
  "locations": ["Tunis", "Tunisia"],
  "top_keywords": [
    {"term": "cloud", "score": 0.12},
    {"term": "erp", "score": 0.10}
  ],
  "noun_chunks": ["cloud migration", "erp system"],
  "all_dates": ["March 15, 2026"],
  "certifications": ["ISO 27001", "PMP"],
  "meta": {
    "detected_language": "fr",
    "processing_time_ms": 27.7,
    "text_length": 450,
    "sentence_count": 5
  }
}
```

`ExtractionResult.to_compact_dict()` returns the minimal version:
```json
{
  "domain": "Cloud Computing",
  "skills": ["Python", "AWS"],
  "budget": "500,000 TND",
  "deadline": "2026-03-15",
  "organization": "Ministry of Digital Economy",
  "location": "Tunis"
}
```

---

## Stage Details

### Stage 1: Preprocessing (`preprocessing.py`)

`preprocess_text(raw_text, max_length=50000, fix_ocr=True)` → cleaned string

**Steps executed in order:**
1. Truncate to 50,000 chars
2. Unicode normalize (NFKC)
3. Strip residual HTML tags (`<p>`, `<div>`, etc.) via regex
4. Strip HTML entities (`&amp;`, `&#123;`, etc.)
5. Remove page numbers ("Page 3 of 12")
6. Remove separator lines (`====`, `----`, `****`)
7. Remove bullet markers (`•`, `►`, `▪`, `■`, `→`)
8. OCR noise repair (if `fix_ocr=True`):
   - Replace ligatures: `ﬁ`→`fi`, `ﬂ`→`fl`, `ﬀ`→`ff`
   - Replace smart quotes: `'`→`'`, `"`→`"`
   - Remove non-printable garbage (keeps Latin + Arabic + basic punctuation)
9. Collapse multiple newlines (3+ → 2)
10. Collapse multiple spaces to single space
11. Strip whitespace per line

`lemmatize(doc)` → list of lemma strings. Filters out: punctuation, spaces, numbers, stopwords, tokens < 2 chars, tender-specific stopwords from `taxonomy.py`. Preserves named entity tokens verbatim (no lemmatization).

`segment_sentences(doc, min_length=15)` → list of sentence strings. Filters out sentences shorter than 15 chars.

---

### Stage 2: NER Extraction (`ner_extractor.py`)

`extract_entities(doc, raw_text)` → `NERResult`

**Pass 1 — spaCy statistical NER:**
Iterates `doc.ents` and buckets by label:
- `ORG` → `organizations[]`
- `GPE` / `LOC` → `locations[]`
- `MONEY` → `all_money[]` + tries to parse as `BudgetInfo`
- `DATE` → `all_dates[]`

**Pass 2 — Rule-based budget regex:**
Pattern matches amounts with currency symbols/codes:
```
$500,000  |  €1.2M  |  500 000 TND  |  1,200,000 DT
```
Regex captures: symbol ($€£¥₹), amount (with commas/spaces), currency code (3-letter or "DT"/"dinars"), suffix multiplier (M/K/B).

Selection logic: if multiple budget candidates found, scores each by:
- Proximity to signal words ("budget", "estimated value", "contract value", etc.)
- Largest amount gets preference
- Amounts near signal words get 10× score boost

**Pass 3 — Rule-based deadline regex:**
Matches date formats:
- ISO: `2025-03-15`
- European: `15/03/2025`, `15.03.2025`
- Written English: `March 15, 2025` / `15 March 2025`
- Written French: `15 mars 2025`

Selection logic for submission deadline:
1. Find all dates in text
2. For each date, check 200 chars before it for deadline signal words (20 phrases in English + French + Arabic from `taxonomy.py`)
3. Score = signal proximity (closer = higher) + future date bonus (+0.3)
4. Pick highest-scoring date

**Data classes returned:**
- `BudgetInfo`: `raw_text`, `amount` (float), `currency` (ISO code), `is_estimated` (bool)
- `DeadlineInfo`: `raw_text`, `iso_date` (ISO-8601), `signal_word`, `confidence` (0-1)
- `NERResult`: `organizations[]`, `locations[]`, `budget`, `deadline`, `all_dates[]`, `all_money[]`

---

### Stage 3: TF-IDF Keywords (`tfidf_extractor.py`)

`extract_tfidf_keywords(sentences, raw_text, noun_chunks, top_n=30)` → `TFIDFResult`

**Step 1 — TF-IDF computation:**
- Uses `sklearn.TfidfVectorizer` on sentence-level corpus
- Settings: `ngram_range=(1,3)`, `sublinear_tf=True` (log(1+tf)), `max_df=0.85`, `stop_words="english"`
- Token pattern: `[a-zA-Z][a-zA-Z0-9+#.]{1,}` (min 2 chars, allows C++, C#)
- Aggregates scores across sentences (mean)
- Filters out tender-specific stopwords from `taxonomy.py`
- Deduplicates, returns top 30 as `ScoredKeyword(term, score)`

**Step 2 — Domain classification:**
- Checks each TF-IDF keyword against `DOMAIN_TAXONOMY` dictionary (100+ entries)
- Also checks noun chunks and raw text against taxonomy
- Scores domains by sum of matching keyword TF-IDF scores
- Returns all domains above 20% of top domain's score (handles multi-domain tenders)
- Falls back to `"General"` if nothing matches

**Step 3 — Skill detection:**
- 90+ compiled regex patterns from `SKILL_PATTERNS` in `taxonomy.py`
- Each pattern searched against raw text
- If found, combined score = `category_weight × (1 + tfidf_score)`
  - Category weights: Programming Language (1.0), Framework (0.95), Cloud (0.95), Database (0.90), DevOps (0.90), etc.
  - TF-IDF score: looked up from Step 1 keywords (exact or partial match)
- Skills sorted by combined score descending
- Certifications (PMP, CISSP, ISO 27001, GDPR, etc.) separated into their own list

---

### Stage 4: Merge (`keyword_extraction.py`)

`_merge_results()` combines NER + TF-IDF with fallback to scraper metadata:

| Field | Primary source | Fallback |
|-------|---------------|----------|
| `domain` | TF-IDF taxonomy mapping | `"General"` |
| `skills` | Regex pattern matching ranked by TF-IDF | empty list |
| `budget` | NER MONEY → regex rule-based | scraper metadata |
| `deadline` | Rule-based with signal words | scraper metadata |
| `organization` | NER ORG entities (first one) | scraper metadata |
| `location` | NER GPE entities (first one) | scraper metadata |
| `certifications` | Skill patterns with category="Certification" | empty list |

---

## Taxonomy (`taxonomy.py`)

All dictionaries are flat (O(1) lookup). Lowercase keys.

| Dictionary | Type | Count | Purpose |
|-----------|------|-------|---------|
| `DOMAIN_TAXONOMY` | `Dict[str, str]` | 175+ entries | keyword → domain label (EN + FR + AR) |
| `SKILL_PATTERNS` | `List[Tuple[str,str,str]]` | 90+ entries | (canonical_name, category, regex) |
| `SKILL_CATEGORIES_WEIGHT` | `Dict[str, float]` | 13 categories | category → weight (0.5–1.0) |
| `CURRENCY_MAP` | `Dict[str, str]` | 35+ entries | symbol/code/word → ISO currency |
| `DEADLINE_SIGNALS` | `List[str]` | 25+ phrases | EN + FR + AR signal words |
| `TENDER_STOPWORDS` | `FrozenSet[str]` | 80+ words | EN + FR + AR procurement noise words |

### Skill categories and weights:
```
Programming Language  1.00    Framework       0.95
Cloud Platform        0.95    Database        0.90
DevOps               0.90    ERP System      0.85
AI/ML Tool           0.85    Security Tool   0.85
Certification        0.80    Protocol        0.75
Methodology          0.70    Operating System 0.60
Other                0.50
```

### Domain taxonomy coverage (15 domains × 3 languages):
IT Services, Cloud Computing, ERP, AI/Machine Learning, Data Analytics, Cybersecurity, Construction, Healthcare, Energy, Telecommunications, Education & Training, Consulting, Supply & Logistics, Finance, Environment.

**French keywords (60+):** informatique, logiciel, cybersécurité, transformation numérique, progiciel de gestion, intelligence artificielle, énergie, bâtiment, santé, formation, etc.

**Arabic keywords (15+):** تكنولوجيا المعلومات, الذكاء الاصطناعي, أمن المعلومات, أشغال, بناء, صحة, طاقة, تعليم, تزويد, مياه, etc.

---

## Integration with Module 1 (`module1_tender_detection.py`)

`TenderDetector.extract_keywords()` calls `keyword_extractor.extract()` which runs the full pipeline above. The returned `ExtractionResult` feeds into:

1. **Domain** → used in domain match scoring (20% weight)
2. **Skills** → used in skill overlap scoring (35% weight)
3. **Budget/deadline/org/location** → merged into final analysis result JSON
4. **Top keywords** → included in output for display

---

## Performance (measured)

| Metric | Value |
|--------|-------|
| Average per tender | ~16ms |
| spaCy models loaded | en_core_web_sm (15MB) + fr_core_news_sm (15MB) |
| Language detection | langdetect (first 1000 chars, <1ms) |
| Max input length | 50,000 chars |
| spaCy max_length | 200,000 tokens |
| Languages detected in real data | fr=17, de=6, ar=4, en=4, + 14 others (45 tenders) |

---

## What Does NOT Exist in Code

| Feature | Status |
|---------|--------|
| Multilingual NER (French model) | ✅ fr_core_news_sm loaded alongside English |
| Language detection (50+ languages) | ✅ langdetect auto-detects per tender |
| French/Arabic domain taxonomy | ✅ 75+ multilingual keywords added |
| Database storage of extracted fields | ❌ Results stay in memory / JSON |
| Alert notifications (email, Slack) | ❌ Not implemented |
| Fine-tuned spaCy model | ❌ Uses stock models (en/fr) |
| Custom NER training for SKILL entities | ❌ Uses regex patterns only |
| Dedicated Arabic spaCy model | ❌ Uses English fallback + taxonomy matching |
| SBERT for keyword extraction | ❌ SBERT is only in relevance scoring |
| Async / batch spaCy.pipe() | ❌ Sequential processing |

---

## Dependencies

| Library | Used for |
|---------|----------|
| `spacy` (en_core_web_sm) | English tokenization, NER, lemmatization, sentence splitting |
| `spacy` (fr_core_news_sm) | French tokenization, NER, lemmatization |
| `langdetect` | Automatic language detection (50+ languages) |
| `scikit-learn` | `TfidfVectorizer` for keyword extraction |
| `numpy` | TF-IDF score aggregation (`mean`, `argsort`) |
