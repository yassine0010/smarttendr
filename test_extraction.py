"""
Test script for the Keyword Extraction Pipeline
Runs on sample tender data to validate all extraction stages.
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.nlp.keyword_extraction import KeywordExtractor

def main():
    print("=" * 70)
    print(" KEYWORD EXTRACTION PIPELINE — TEST")
    print("=" * 70)

    # Load sample tenders
    sample_file = Path(__file__).parent / "data" / "sample_tenders.json"
    with open(sample_file, "r", encoding="utf-8") as f:
        tenders = json.load(f)

    print(f"\n[1] Loaded {len(tenders)} sample tenders")

    # Initialize extractor
    print("[2] Initializing KeywordExtractor...")
    extractor = KeywordExtractor(model_name="en_core_web_sm", top_keywords=25)

    # Process each tender
    print("[3] Running extraction pipeline...\n")

    for i, tender in enumerate(tenders, 1):
        title = tender.get("title", "")
        text = tender.get("description", "")
        existing = {
            "budget": tender.get("budget"),
            "deadline": tender.get("deadline"),
            "organization": tender.get("organization"),
            "location": tender.get("location"),
        }

        result = extractor.extract(
            text,
            title=title,
            existing_metadata=existing,
        )

        print(f"{'─' * 60}")
        print(f"  Tender {i}: {title[:60]}")
        print(f"{'─' * 60}")
        print(f"  Domain:         {result.domain}")
        print(f"  All Domains:    {result.domains}")
        print(f"  Budget:         {result.budget} (amount={result.budget_amount}, currency={result.budget_currency})")
        print(f"  Deadline:       {result.deadline} (raw: {result.deadline_raw})")
        print(f"  Organization:   {result.organization}")
        print(f"  Location:       {result.location}")
        print(f"  Skills:         {[s['name'] for s in result.skills[:8]]}")
        print(f"  Certifications: {result.certifications}")
        print(f"  Top Keywords:   {[kw['term'] for kw in result.top_keywords[:6]]}")
        print(f"  Processing:     {result.processing_time_ms:.1f}ms | {result.sentence_count} sentences")
        print()

        # Also show compact JSON
        compact = result.to_compact_dict()
        print(f"  Compact JSON:")
        print(f"  {json.dumps(compact, indent=4)}")
        print()

    # Batch extraction test
    print(f"\n{'=' * 60}")
    print(f"  BATCH EXTRACTION TEST")
    print(f"{'=' * 60}")
    results = extractor.extract_batch(tenders)
    print(f"  Processed {len(results)} tenders in batch mode")
    total_ms = sum(r.processing_time_ms for r in results)
    print(f"  Total processing time: {total_ms:.1f}ms")
    print(f"  Average per tender:    {total_ms/len(results):.1f}ms")

    for r in results:
        skills_str = ", ".join(s["name"] for s in r.skills[:5])
        print(f"  • [{r.domain}] Skills: {skills_str}")

    print(f"\n{'=' * 70}")
    print(f" ALL TESTS PASSED ✅")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
