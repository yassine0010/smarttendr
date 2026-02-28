# Relevance Filtering Architecture

> **This document describes ONLY what the code actually implements.**
> No aspirational features. No planned features. Code or it didn't happen.

## Overview

The relevance filtering module (`backend/relevance/`) computes how well each
scraped tender matches a company's expertise profile. It produces a three-tier
decision (**RELEVANT** / **LOW_RELEVANCE** / **IRRELEVANT**) using a hybrid
scoring formula that combines:

1. **Semantic similarity** (SBERT cosine) — 45% weight
2. **Skill overlap** (Jaccard-style) — 35% weight
3. **Domain match** (multi-domain vector alignment) — 20% weight

---

## Package Structure

```
backend/relevance/
├── __init__.py           # Package exports
├── company_profile.py    # CompanyProfile dataclass + default profile
├── similarity.py         # SimilarityEngine (SBERT + cosine computation)
├── filter_engine.py      # RelevanceFilter, FilterResult, FilterDecision
└── calibration.py        # ThresholdCalibrator + CalibrationReport
```

---

## Mathematical Foundation

### Cosine Similarity

The core metric. Measures directional alignment between two vectors:

```
                    A · B
    cos(θ) = ─────────────────
              ‖A‖ × ‖B‖
```

- Range: [0, 1] for SBERT embeddings (positive space)
- Why cosine over Euclidean: measures semantic *orientation*, not magnitude.
  A short tender and a long tender about the same topic produce similar
  cosine scores but very different Euclidean distances.

### Hybrid Score Formula

```
final_score = 0.45 × semantic_sim + 0.35 × skill_overlap + 0.20 × domain_sim
```

Where:
- `semantic_sim`: Cosine similarity between tender SBERT embedding and company
  description embedding. Range [0, 1].
- `skill_overlap`: |tender_skills ∩ company_skills| / |tender_skills|.
  Returns 0.5 if tender has no detected skills (neutral). Range [0, 1].
- `domain_sim`: Max cosine similarity between tender embedding and each company
  domain embedding, weighted by domain priority. Range [0, 1].

### Decision Thresholds

```
score ≥ 0.65           → RELEVANT          (pursue)
0.40 ≤ score < 0.65   → LOW_RELEVANCE     (review)
score < 0.40           → IRRELEVANT        (skip)
```

Thresholds are configurable at runtime and can be tuned via the calibration module.

---

## Module Details

### `company_profile.py` — CompanyProfile

**What it does:** Stores a structured company expertise profile as the reference
point for all similarity comparisons.

**Key class:** `CompanyProfile` (dataclass)

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Company name |
| `description` | str | Free-text company description |
| `domains` | dict | Domain → {weight, description}. Weight [0, 1] |
| `skills` | list[str] | Skills the company has |
| `certifications` | list[str] | Certifications held |
| `description_embedding` | np.ndarray | Precomputed 384-dim SBERT vector |
| `domain_embeddings` | dict | Domain name → 384-dim SBERT vector |
| `skill_set` | set | Lowercase skill names for fast lookup |
| `cert_set` | set | Lowercase certification names for fast lookup |

**Default profile:** Inetum Tunisie with 7 weighted domains:
- IT Services (1.0), Cloud Computing (0.95), ERP (0.90),
  AI/ML (0.85), Cybersecurity (0.80), Data Analytics (0.80),
  Digital Transformation (0.75)

**Methods:**
- `CompanyProfile.default()` → loads the built-in default profile
- `CompanyProfile.from_dict(data)` → creates from a plain dict
- `profile.full_text()` → concatenates all text for a single SBERT embedding
- `profile.domain_weight(name)` → returns the priority weight for a domain
- `profile.domain_description(name)` → returns the description text for a domain

---

### `similarity.py` — SimilarityEngine

**What it does:** Loads the SBERT model, computes embeddings, and calculates
all three similarity components.

**Key class:** `SimilarityEngine`

**Initialization:**
1. Loads `all-MiniLM-L6-v2` (384-dim sentence transformer)
2. Stores hybrid scoring weights

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `load_profile(profile)` | CompanyProfile | CompanyProfile (mutated) | Precomputes global + per-domain embeddings |
| `embed_tender(tender)` | dict | np.ndarray (384,) | Single tender → SBERT vector |
| `embed_tenders_batch(tenders)` | list[dict] | np.ndarray (n, 384) | Batch encoding (efficient) |
| `compute_semantic_similarity(emb, profile)` | ndarray, CompanyProfile | float [0,1] | Cosine sim: tender vs company description |
| `compute_domain_similarity(emb, profile)` | ndarray, CompanyProfile | (float, str) | Max cosine sim across domains + best domain name |
| `compute_skill_overlap(tender, profile)` | dict, CompanyProfile | (float, list, list) | Jaccard overlap + matched/missing lists |
| `compute_hybrid_score(tender, emb, profile)` | dict, ndarray, CompanyProfile | dict | Full score breakdown with all components |
| `compute_batch_scores(tenders, profile)` | list[dict], CompanyProfile | list[dict] | Batch hybrid scores |

**Multi-domain matching logic:**
For each company domain, compute cosine(tender_embedding, domain_embedding),
then multiply by the domain's weight. Take the maximum. This means a cybersecurity
tender can match the cybersecurity domain even if IT Services is the primary domain.

---

### `filter_engine.py` — RelevanceFilter

**What it does:** Orchestrates similarity computation, applies thresholds,
and produces structured `FilterResult` objects.

**Key classes:**

1. **`FilterDecision`** (enum): `RELEVANT`, `LOW_RELEVANCE`, `IRRELEVANT`

2. **`FilterResult`** (dataclass):
   ```
   tender_id, tender_title,
   final_score, semantic_similarity, skill_overlap, domain_similarity,
   threshold, decision,
   best_matching_domain, matched_skills, missing_skills,
   computation_time_ms
   ```
   Has `.to_dict()` for JSON serialization.

3. **`FilterBatchResult`** (dataclass):
   ```
   total_tenders, relevant_count, low_relevance_count, irrelevant_count,
   results (list[FilterResult]), processing_time_ms,
   mean_score, max_score, min_score
   ```
   Has `.relevant_tenders` and `.low_relevance_tenders` properties.

4. **`RelevanceFilter`** (main class):

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `__init__(profile, thresholds, model, weights)` | config | — | Loads SBERT, embeds profile |
| `filter_tender(tender)` | dict | FilterResult | Score + classify one tender |
| `filter_tenders(tenders, sort_by_score)` | list[dict] | FilterBatchResult | Batch filter with statistics |
| `update_thresholds(relevant, low_relevance)` | floats | — | Change thresholds at runtime |

---

### `calibration.py` — ThresholdCalibrator

**What it does:** Analyzes score distributions and suggests optimal thresholds.

**Key classes:**

1. **`CalibrationReport`** (dataclass):
   Score statistics (mean, std, percentiles), current thresholds,
   tier counts, suggested thresholds, analysis notes.

2. **`ThresholdCalibrator`**:

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `analyze_distribution(batch_result)` | FilterBatchResult | CalibrationReport | Stats + auto-suggest thresholds |
| `calibrate_on_labeled(tenders, labels, range, steps)` | data + labels | dict | Sweep thresholds, find best F1 |

**Auto-suggestion rules:**
- If >50% RELEVANT → raise threshold (use 60th percentile)
- If <10% RELEVANT → lower threshold (use 80th percentile)
- LOW_RELEVANCE threshold targets 25th percentile
- Warns if score variance < 0.05 (poor differentiation)

---

## Integration with module1_tender_detection.py

The `TenderDetector` class in `module1_tender_detection.py` uses `RelevanceFilter`
as follows:

```
Module Load:
    relevance_filter = RelevanceFilter(
        profile=CompanyProfile.default(),
        relevant_threshold=0.65,
        low_relevance_threshold=0.40,
    )

Per Tender (in analyze_tenders):
    1. Extract keywords with NLP pipeline → ExtractionResult
    2. Enrich tender dict with detected_skills, detected_domain, detected_certifications
    3. filter_result = relevance_filter.filter_tender(enriched_tender)
    4. Merge filter_result scores + decision into output dict
```

The SBERT model is loaded **once** inside `RelevanceFilter.__init__()` →
`SimilarityEngine.__init__()`. Profile embeddings are precomputed once.
Per-tender cost is one SBERT encode + three cosine similarity computations.

---

## Data Flow

```
Scraped Tender (dict)
    │
    ├── title, description, required_skills, category
    │
    ▼
NLP Extraction (keyword_extraction.py)
    │
    ├── detected_domain, detected_skills, detected_certifications
    │
    ▼
Enriched Tender (dict with NLP fields)
    │
    ▼
SimilarityEngine
    │
    ├── embed_tender() → 384-dim vector
    │
    ├── compute_semantic_similarity() → float [0,1]
    ├── compute_skill_overlap() → float [0,1] + matched/missing lists
    ├── compute_domain_similarity() → float [0,1] + best domain name
    │
    ├── hybrid formula: 0.45×semantic + 0.35×skill + 0.20×domain
    │
    ▼
RelevanceFilter._decide(score)
    │
    ├── score ≥ 0.65 → RELEVANT
    ├── score ≥ 0.40 → LOW_RELEVANCE
    └── score < 0.40 → IRRELEVANT
    │
    ▼
FilterResult (structured output)
    │
    ├── tender_id, final_score, decision
    ├── semantic_similarity, skill_overlap, domain_similarity
    ├── best_matching_domain, matched_skills, missing_skills
    └── computation_time_ms
```

---

## Output Format

### Single Tender (FilterResult.to_dict())
```json
{
    "tender_id": "T001",
    "tender_title": "Cloud ERP System Development",
    "final_score": 0.7234,
    "semantic_similarity": 0.6812,
    "skill_overlap": 0.8000,
    "domain_similarity": 0.6500,
    "threshold": 0.65,
    "decision": "RELEVANT",
    "best_matching_domain": "Cloud Computing",
    "matched_skills": ["python", "aws", "docker"],
    "missing_skills": ["sap"],
    "computation_time_ms": 12.34
}
```

### Batch Result (FilterBatchResult.to_dict())
```json
{
    "summary": {
        "total_tenders": 146,
        "relevant": 23,
        "low_relevance": 58,
        "irrelevant": 65,
        "processing_time_ms": 1842.5,
        "mean_score": 0.4812,
        "max_score": 0.8934,
        "min_score": 0.1203
    },
    "results": [ ... ]
}
```

---

## What Does NOT Exist

| Feature | Status |
|---------|--------|
| PostgreSQL storage | ❌ Not implemented |
| Alert / notification system | ❌ Not implemented |
| Real-time streaming | ❌ Not implemented |
| GPU acceleration | ❌ Not implemented (CPU only) |
| Fine-tuned SBERT model | ❌ Uses pretrained all-MiniLM-L6-v2 |
| User feedback loop | ❌ Not implemented |
| REST API endpoint | ❌ Not implemented |
| A/B testing of thresholds | ❌ Not implemented |
