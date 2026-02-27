"""
SmartTender AI - Module 1: Smart Tender Detection
==================================================
Features:
- Multi-tier Web Scraping pipeline (API → RSS → HTML → Browser)
  Platforms: SAM.gov, TED Europa, UNGM, TUNEPS, Contracts Finder UK
- Keyword extraction (spaCy + TF-IDF)
- Semantic embedding (Sentence-BERT)
- Relevance scoring (cosine similarity)
- Alert system (threshold-based notification)

Architecture: See docs/SCRAPING_ARCHITECTURE.md for full design.

Author: SmartTender AI Team
Date: 2025-2026
"""

import re
import sys
import spacy
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any, Optional

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).parent))

from scraping.pipeline import ScrapingPipeline
from scraping.base import NormalizedTender

# ================================================================
# LOAD NLP MODELS
# ================================================================
print("[Module 1] Loading NLP models...")

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("  -> Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load Sentence-BERT for semantic embeddings
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Module 1] Models loaded successfully!")


# ================================================================
# DEFAULT COMPANY PROFILE
# ================================================================
DEFAULT_COMPANY_PROFILE = {
    "name": "Inetum Tunisie",
    "description": """
    Inetum is a leading IT services company specializing in
    digital transformation, cloud computing, ERP implementation,
    AI/ML solutions, custom software development, cybersecurity,
    and IT consulting. We operate across Europe, Africa, and the
    Middle East with expertise in Python, Java, AWS, Azure, SAP,
    Odoo, data analytics, and agile project management.
    """,
    "domains": [
        "IT Services", "Cloud Computing", "ERP",
        "AI/Machine Learning", "Cybersecurity",
        "Digital Transformation", "Software Development"
    ],
    "key_skills": [
        "Python", "Java", "AWS", "Azure", "Docker", "Kubernetes",
        "SAP", "Odoo", "NLP", "Machine Learning", "Agile/Scrum",
        "REST API", "Microservices", "PostgreSQL", "React"
    ]
}


# ================================================================
# TENDER DETECTOR CLASS
# ================================================================
class TenderDetector:
    """
    Smart Tender Detection Engine

    This class provides functionality to:
    1. Extract keywords from tender documents
    2. Compute relevance scores between tenders and company profile
    3. Rank and filter tenders by relevance
    """

    def __init__(
        self,
        company_profile: Optional[Dict] = None,
        threshold: float = 0.3
    ):
        """
        Initialize the TenderDetector.

        Args:
            company_profile: Dictionary containing company information
            threshold: Minimum relevance score (0-1) to consider relevant
        """
        self.company = company_profile or DEFAULT_COMPANY_PROFILE
        self.threshold = threshold

        # Pre-compute company profile embedding for efficiency
        self.company_embedding = sbert_model.encode(
            [self.company["description"]]
        )

    # ============================================================
    # KEYWORD EXTRACTION
    # ============================================================
    def extract_keywords(
        self,
        text: str,
        top_n: int = 15
    ) -> Dict[str, List[str]]:
        """
        Extract keywords from tender text using multiple NLP techniques.

        Args:
            text: The tender description text
            top_n: Number of top keywords to extract

        Returns:
            Dictionary containing:
            - entities: Named entities (organizations, dates, etc.)
            - noun_chunks: Noun phrases
            - tfidf_keywords: TF-IDF extracted terms
            - tech_skills: Technical skills (pattern-matched)
        """
        doc = nlp(text)

        # 1. Named Entity Recognition
        entity_labels = ("ORG", "GPE", "MONEY", "DATE",
                         "PRODUCT", "NORP", "FAC")
        entities = list(set([
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in entity_labels and len(ent.text.strip()) > 2
        ]))

        # 2. Noun Chunks (noun phrases)
        chunks = list(set([
            chunk.text.strip().lower()
            for chunk in doc.noun_chunks
            if len(chunk.text.strip()) > 2 and len(chunk.text.split()) <= 4
        ]))[:20]

        # 3. TF-IDF Keywords
        sentences = [
            sent.text for sent in doc.sents
            if len(sent.text.strip()) > 10
        ]
        tfidf_keywords = []
        if sentences:
            try:
                tfidf = TfidfVectorizer(
                    stop_words="english",
                    max_features=top_n,
                    ngram_range=(1, 2),
                    min_df=1
                )
                tfidf.fit_transform(sentences)
                tfidf_keywords = tfidf.get_feature_names_out().tolist()
            except ValueError:
                pass  # Handle empty vocabulary

        # 4. Technical Skills (Pattern-Based)
        tech_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP)\b',
            r'\b(React|Angular|Vue\.js|Node\.js|Django|Flask|Spring|Laravel)\b',
            r'\b(AWS|Azure|GCP|Google Cloud|Docker|Kubernetes|K8s|Terraform)\b',
            r'\b(PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Oracle|SQL Server)\b',
            r'\b(SAP|Odoo|Salesforce|ServiceNow|Dynamics)\b',
            r'\b(NLP|ML|AI|TensorFlow|PyTorch|Scikit-learn|Deep Learning)\b',
            r'\b(REST\s*API|GraphQL|Microservices|CI/CD|DevOps|MLOps)\b',
            r'\b(Agile|Scrum|Kanban|PMP|ITIL|PRINCE2|Jira)\b',
            r'\b(ISO\s*27001|GDPR|OWASP|CISSP|CEH|SOC2|HIPAA)\b',
            r'\b(Linux|Windows Server|Unix|Shell|Bash|PowerShell)\b',
        ]
        tech_skills = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            tech_skills.extend([m.strip() for m in matches])
        tech_skills = list(set(tech_skills))

        return {
            "entities": entities,
            "noun_chunks": chunks[:15],
            "tfidf_keywords": tfidf_keywords,
            "tech_skills": tech_skills
        }

    # ============================================================
    # RELEVANCE SCORING
    # ============================================================
    def compute_relevance(self, tender: Dict) -> float:
        """
        Compute relevance score between tender and company profile.

        The score is a weighted combination of:
        - Semantic similarity (45%): SBERT embeddings cosine similarity
        - Skill overlap (35%): Jaccard-like skill matching
        - Domain match (20%): Category alignment

        Args:
            tender: Dictionary containing tender information

        Returns:
            Relevance score as percentage (0-100)
        """
        # 1. Semantic Similarity (Sentence-BERT)
        tender_text = f"{tender.get('title', '')}. {tender.get('description', '')}"
        tender_embedding = sbert_model.encode([tender_text])
        semantic_sim = cosine_similarity(
            tender_embedding,
            self.company_embedding
        )[0][0]

        # 2. Skill Overlap Score
        tender_skills = set([
            s.lower().strip()
            for s in tender.get("required_skills", [])
        ])
        company_skills = set([
            s.lower().strip()
            for s in self.company["key_skills"]
        ])

        if tender_skills:
            skill_overlap = len(tender_skills & company_skills) / len(tender_skills)
        else:
            skill_overlap = 0.5  # Neutral if no skills specified

        # 3. Domain Match Score
        tender_category = tender.get("category", "").lower()
        domain_match = any(
            domain.lower() in tender_category or
            tender_category in domain.lower()
            for domain in self.company["domains"]
        )
        domain_score = 1.0 if domain_match else 0.3

        # 4. Weighted Final Score
        weights = {
            "semantic": 0.45,
            "skills": 0.35,
            "domain": 0.20
        }

        final_score = (
            weights["semantic"] * semantic_sim +
            weights["skills"] * skill_overlap +
            weights["domain"] * domain_score
        )

        return round(float(final_score * 100), 2)

    # ============================================================
    # ANALYZE TENDERS
    # ============================================================
    def analyze_tenders(
        self,
        tenders: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Analyze a list of tenders and return ranked results.

        Args:
            tenders: List of tender dictionaries

        Returns:
            List of analysis results, sorted by relevance score
        """
        results = []

        for tender in tenders:
            # Compute relevance score
            score = self.compute_relevance(tender)

            # Extract keywords
            keywords = self.extract_keywords(
                tender.get("description", "")
            )

            # Build result object
            result = {
                "id": tender.get("id", ""),
                "title": tender.get("title", ""),
                "platform": tender.get("platform", "Unknown"),
                "deadline": tender.get("deadline", "N/A"),
                "budget": tender.get("budget", "N/A"),
                "category": tender.get("category", "Unknown"),
                "relevance_score": score,
                "is_relevant": score >= (self.threshold * 100),
                "extracted_keywords": keywords,
                "required_skills": tender.get("required_skills", []),
                "analysis_timestamp": str(np.datetime64('now'))
            }
            results.append(result)

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return results

    # ============================================================
    # WEB SCRAPING (production pipeline)
    # ============================================================
    @staticmethod
    def scrape(
        platform: str = "ALL",
        query: str = "IT services",
        sources: List[str] | None = None,
    ) -> List[Dict]:
        """
        Scrape real tenders using the multi-tier scraping pipeline.

        The pipeline automatically selects the best strategy per source:
        - Tier 1 (API): SAM.gov, Contracts Finder UK
        - Tier 2 (RSS): TED Europa RSS feeds
        - Tier 3 (HTML): UNGM, static government portals
        - Tier 4 (Browser): TUNEPS (JS-rendered)

        Args:
            platform: "ALL", or specific source name (e.g. "SAM.GOV", "TED")
            query: Search keyword for platforms that support it
            sources: Explicit list of source names to scrape

        Returns:
            List of tender dictionaries (legacy format)
        """
        pipeline = ScrapingPipeline()

        # Map platform shorthand to source names
        if platform.upper() != "ALL" and not sources:
            platform_map = {
                "TUNEPS": ["TUNEPS"],
                "TED": ["TED"],
                "TED EUROPA": ["TED"],
                "SAM": ["SAM.GOV"],
                "SAM.GOV": ["SAM.GOV"],
                "UNGM": ["UNGM"],
                "CONTRACTS_FINDER": ["CONTRACTS_FINDER"],
                "CF": ["CONTRACTS_FINDER"],
            }
            sources = platform_map.get(platform.upper(), None)

        # Run the pipeline
        fallback_file = Path(__file__).parent.parent / "data" / "sample_tenders.json"
        normalized_tenders = pipeline.run(
            query=query,
            sources=sources,
            fallback_file=fallback_file,
        )

        # Convert to legacy dict format for backward compatibility
        return [t.to_legacy_dict() for t in normalized_tenders]

    # ============================================================
    # MOCK SCRAPER (fallback for demonstration)
    # ============================================================
    @staticmethod
    def mock_scrape(platform: str = "TUNEPS") -> List[Dict]:
        """
        Simulate scraping a tender platform.
        In production, this would use actual web scraping.

        Args:
            platform: Name of the platform to scrape

        Returns:
            List of tender dictionaries
        """
        # Import sample data
        from utils import load_json
        all_tenders = load_json("sample_tenders.json")

        # Filter by platform if specified
        if platform.upper() != "ALL":
            filtered = [
                t for t in all_tenders
                if t.get("platform", "").upper() == platform.upper()
            ]
            return filtered if filtered else all_tenders

        return all_tenders


# ================================================================
# STANDALONE EXECUTION
# ================================================================
if __name__ == "__main__":
    import json
    from pathlib import Path

    # Initialize detector
    detector = TenderDetector(threshold=0.3)

    # ----------------------------------------------------------
    # Step 1: Run the multi-tier scraping pipeline
    # ----------------------------------------------------------
    print("\n[Step 1] Running multi-tier scraping pipeline...")
    print("         Sources: SAM.gov (API) | TED (RSS) | UNGM (HTML) | TUNEPS (Browser) | Contracts Finder (API)")
    scraped_tenders = detector.scrape(platform="ALL", query="IT services")

    # ----------------------------------------------------------
    # Step 2: Use scraped data or fallback to sample data
    # ----------------------------------------------------------
    if scraped_tenders:
        tenders = scraped_tenders
        print(f"\n[OK] Using {len(tenders)} tenders from scraping pipeline")
    else:
        print("\n[Info] No tenders scraped — using sample data as fallback")
        data_dir = Path(__file__).parent.parent / "data"
        sample_file = data_dir / "sample_tenders.json"

        if sample_file.exists():
            with open(sample_file, "r", encoding="utf-8") as f:
                tenders = json.load(f)
        else:
            tenders = [{
                "id": "T001",
                "title": "Cloud ERP System Development",
                "description": "Development of cloud-based ERP using Python, AWS, Docker",
                "required_skills": ["Python", "AWS", "Docker", "PostgreSQL"],
                "platform": "TUNEPS",
                "deadline": "2025-03-01",
                "category": "IT Services"
            }]

    # ----------------------------------------------------------
    # Step 3: Analyze & rank tenders with NLP engine
    # ----------------------------------------------------------
    print(f"\n[Step 2] Analyzing {len(tenders)} tenders with NLP engine...")
    results = detector.analyze_tenders(tenders)

    # ----------------------------------------------------------
    # Step 4: Display results
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(" SMARTTENDER AI - TENDER DETECTION RESULTS")
    print("=" * 70)

    for r in results:
        status = "RELEVANT" if r["is_relevant"] else "LOW MATCH"
        icon = "✅" if r["is_relevant"] else "⬜"

        print(f"\n{icon} {r['title'][:65]}")
        print(f"   Score: {r['relevance_score']}% | Status: {status}")
        print(f"   Platform: {r['platform']} | Deadline: {r['deadline']}")
        if r['extracted_keywords']['tech_skills']:
            print(f"   Tech Skills Found: {', '.join(r['extracted_keywords']['tech_skills'][:5])}")

    print("\n" + "=" * 70)
    relevant_count = sum(1 for r in results if r["is_relevant"])
    print(f" Summary: {relevant_count}/{len(results)} tenders are relevant "
          f"(threshold: {detector.threshold * 100:.0f}%)")
    print("=" * 70)

    # ----------------------------------------------------------
    # Step 5: Save analysis results
    # ----------------------------------------------------------
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    analysis_file = output_dir / "analysis_results.json"

    # Remove non-serializable fields for JSON output
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items()}
        serializable_results.append(sr)

    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[Saved] Analysis results → {analysis_file}")
