"""
SmartTender AI — Similarity Computation Engine
=================================================
Builds vector representations and computes cosine similarity
between tenders and company profiles.

Two embedding spaces used:
    1. SBERT (all-MiniLM-L6-v2) — 384-dim sentence embeddings
       for semantic similarity between description texts.
    2. Skill sets — Jaccard-style overlap for discrete skill matching.

Both are combined into a single hybrid score.

Mathematical background:

    Cosine Similarity:
                       A · B
        cos(θ) = ─────────────────
                  ‖A‖ × ‖B‖

    Range: [-1, 1] for general vectors, [0, 1] for SBERT (positive space).

    Why cosine over Euclidean?
    - Cosine measures *direction* (semantic orientation), not magnitude.
    - A short tender and a long tender about the same topic will have
      similar direction but very different magnitudes.
    - Euclidean distance penalizes length differences, producing
      misleading scores for documents of different sizes.
    - SBERT embeddings are L2-normalized, so cosine = dot product
      (extremely fast computation).

Author: SmartTender AI Team
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from backend.relevance.company_profile import CompanyProfile


# ================================================================
# SIMILARITY ENGINE
# ================================================================

class SimilarityEngine:
    """
    Vector construction and similarity computation.

    Handles:
        - Loading and caching the SBERT model
        - Precomputing company profile embeddings (once)
        - Computing per-tender similarity scores (fast)
        - Multi-domain matching (max similarity across domains)
        - Hybrid scoring (semantic + skill overlap + domain match)

    Thread-safe for read-only operations after init.
    """

    # Default weight allocation for hybrid score components
    DEFAULT_WEIGHTS = {
        "semantic": 0.30,       # SBERT cosine similarity
        "skill_overlap": 0.25,  # Skill matching
        "domain_match": 0.20,   # Domain alignment
        "keyword_signal": 0.25, # IT keyword detection bonus
    }

    # Universal IT keywords that appear across languages
    # (technology names are not translated)
    IT_KEYWORDS = {
        # Technologies
        "python", "java", "javascript", ".net", "c#", "c++", "php", "ruby",
        "typescript", "golang", "rust", "scala", "kotlin", "swift",
        # Frameworks
        "react", "angular", "vue", "django", "spring", "node.js", "fastapi",
        "flask", "laravel", "express", "nextjs", "next.js",
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "ci/cd", "devops", "cloud", "saas", "paas", "iaas",
        # Data
        "sql", "nosql", "postgresql", "mysql", "mongodb", "oracle", "redis",
        "elasticsearch", "hadoop", "spark", "kafka", "etl", "power bi", "tableau",
        # ERP
        "sap", "odoo", "dynamics 365", "salesforce", "erp", "crm",
        # AI/ML
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "nlp", "chatbot", "computer vision", "data science",
        # Security
        "cybersecurity", "penetration testing", "siem", "soc", "iso 27001",
        "firewall", "encryption", "gdpr",
        # General IT terms (appear in all languages)
        "api", "rest", "microservices", "agile", "scrum",
        "software", "it", "digital", "ict",
        # French IT terms
        "informatique", "logiciel", "numérique", "progiciel",
        "cybersécurité", "développement",
        # German IT terms
        "softwareentwicklung", "it-dienstleistungen", "digitalisierung",
        "informationstechnologie", "datenbank",
        # Dutch IT terms
        "software", "informatica", "digitale", "ict-diensten",
        # Polish IT terms
        "oprogramowanie", "informatyczny", "cyfrowy",
        # Romanian IT terms
        "informatică", "software", "digitală",
    }

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the similarity engine.

        Args:
            model_name: Sentence-transformer model identifier.
            weights:    Custom weights for hybrid score components.
                        Keys: "semantic", "skill_overlap", "domain_match".
                        Values must sum to 1.0.
        """
        self.model_name = model_name
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Validate weights sum to 1.0 (4 components now)
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"Weights must sum to ~1.0, got {total}: {self.weights}"
            )

        # Load SBERT model (expensive — done once)
        self._model = SentenceTransformer(model_name)

    # ============================================================
    # PROFILE EMBEDDING (precomputed once)
    # ============================================================

    def load_profile(self, profile: CompanyProfile) -> CompanyProfile:
        """
        Precompute all embeddings for a company profile.

        Computes:
            1. Global description embedding (384-dim)
            2. Per-domain embeddings (one 384-dim vector per domain)

        These are cached on the profile object and reused
        for every tender comparison.

        Args:
            profile: CompanyProfile to embed.

        Returns:
            Same profile object, now with embeddings populated.
        """
        # Global embedding from full company text
        profile.description_embedding = self._encode(profile.full_text())

        # Per-domain embeddings
        for domain_name in profile.domain_names():
            domain_text = (
                f"{domain_name}. "
                f"{profile.domain_description(domain_name)}"
            )
            profile.domain_embeddings[domain_name] = self._encode(domain_text)

        return profile

    # ============================================================
    # TENDER EMBEDDING (computed per tender)
    # ============================================================

    def embed_tender(self, tender: Dict) -> np.ndarray:
        """
        Build a single SBERT embedding for a tender.

        Combines title + description + detected skills/domain
        into one text, then encodes.

        Args:
            tender: Tender dict with at minimum "title" and "description".

        Returns:
            384-dim numpy array.
        """
        parts = []
        if tender.get("title"):
            parts.append(tender["title"])
        if tender.get("description"):
            parts.append(tender["description"])
        if tender.get("detected_domain"):
            parts.append(f"Domain: {tender['detected_domain']}")
        if tender.get("detected_skills"):
            parts.append("Skills: " + ", ".join(tender["detected_skills"][:10]))
        elif tender.get("required_skills"):
            parts.append("Skills: " + ", ".join(tender["required_skills"][:10]))

        text = ". ".join(parts) if parts else ""
        return self._encode(text)

    def embed_tenders_batch(self, tenders: List[Dict]) -> np.ndarray:
        """
        Batch-encode multiple tenders efficiently.

        Uses SBERT's internal batching for GPU/CPU parallelism.

        Args:
            tenders: List of tender dicts.

        Returns:
            np.ndarray of shape (n_tenders, 384).
        """
        texts = []
        for t in tenders:
            parts = []
            if t.get("title"):
                parts.append(t["title"])
            if t.get("description"):
                parts.append(t["description"])
            if t.get("detected_domain"):
                parts.append(f"Domain: {t['detected_domain']}")
            if t.get("detected_skills"):
                parts.append("Skills: " + ", ".join(t["detected_skills"][:10]))
            elif t.get("required_skills"):
                parts.append("Skills: " + ", ".join(t["required_skills"][:10]))
            texts.append(". ".join(parts) if parts else "")

        return self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    # ============================================================
    # SIMILARITY COMPUTATION
    # ============================================================

    def compute_semantic_similarity(
        self,
        tender_embedding: np.ndarray,
        profile: CompanyProfile,
    ) -> float:
        """
        Cosine similarity between tender and company description.

                           tender · company
            cos(θ) = ─────────────────────────
                      ‖tender‖ × ‖company‖

        Args:
            tender_embedding: 384-dim tender vector.
            profile:          CompanyProfile with precomputed embedding.

        Returns:
            Similarity score in [0, 1].
        """
        if profile.description_embedding is None:
            return 0.0

        t = tender_embedding.reshape(1, -1)
        c = profile.description_embedding.reshape(1, -1)
        sim = float(sk_cosine(t, c)[0][0])

        # Clamp to [0, 1] — SBERT rarely produces negatives
        return max(0.0, min(1.0, sim))

    def compute_domain_similarity(
        self,
        tender_embedding: np.ndarray,
        profile: CompanyProfile,
    ) -> Tuple[float, str]:
        """
        Multi-domain matching: compare tender against each company
        domain vector and take the MAX similarity.

        This handles multi-domain companies — a cybersecurity tender
        matches the cybersecurity domain even if the company's
        primary domain is IT Services.

        Args:
            tender_embedding: 384-dim tender vector.
            profile:          CompanyProfile with per-domain embeddings.

        Returns:
            (max_similarity, best_matching_domain_name)
        """
        if not profile.domain_embeddings:
            return 0.0, "General"

        t = tender_embedding.reshape(1, -1)
        best_sim = 0.0
        best_domain = "General"

        for domain_name, domain_emb in profile.domain_embeddings.items():
            d = domain_emb.reshape(1, -1)
            sim = float(sk_cosine(t, d)[0][0])

            # Apply domain weight: higher-priority domains get a boost
            weight = profile.domain_weight(domain_name)
            weighted_sim = sim * weight

            if weighted_sim > best_sim:
                best_sim = weighted_sim
                best_domain = domain_name

        return max(0.0, min(1.0, best_sim)), best_domain

    def compute_skill_overlap(
        self,
        tender: Dict,
        profile: CompanyProfile,
    ) -> Tuple[float, List[str], List[str]]:
        """
        Jaccard-style skill overlap score.

                         |tender_skills ∩ company_skills|
            overlap = ────────────────────────────────────
                              |tender_skills|

        If tender has no skills detected, returns 0.5 (neutral).
        Also returns the matched and missing skill lists.

        Args:
            tender:  Tender dict with "detected_skills" or "required_skills".
            profile: CompanyProfile with skill_set.

        Returns:
            (overlap_score, matched_skills, missing_skills)
        """
        import re

        # Collect tender skills from multiple possible fields
        tender_skills_raw = (
            tender.get("detected_skills", [])
            or tender.get("required_skills", [])
        )
        tender_skills = {s.lower().strip() for s in tender_skills_raw if s}

        # Mine skills directly from title+description text
        # Tech terms like Python, SAP, AWS are universal across languages
        text_blob = (
            (tender.get("title", "") + " " + tender.get("description", ""))
            .lower()
        )

        # Short skills (<=3 chars) need word-boundary matching to avoid false positives
        # Longer skills can use substring matching safely
        for skill in profile.skill_set:
            if len(skill) <= 3:
                # Word boundary for short terms (sql, nlp, aws, etc.)
                if re.search(r'\b' + re.escape(skill) + r'\b', text_blob):
                    tender_skills.add(skill)
            elif len(skill) <= 5:
                # Word boundary for medium terms (react, redis, etc.)
                if re.search(r'\b' + re.escape(skill) + r'\b', text_blob):
                    tender_skills.add(skill)
            else:
                # Substring OK for longer terms (kubernetes, postgresql, etc.)
                if skill in text_blob:
                    tender_skills.add(skill)

        if not tender_skills:
            # If no skills detectable, check if this looks like an IT tender
            # via keyword signal. If so, give a positive score instead of neutral.
            text_blob_check = (
                (tender.get("title", "") + " " + tender.get("description", ""))
                .lower()
            )
            it_hits = sum(1 for kw in self.IT_KEYWORDS if kw in text_blob_check)
            if it_hits >= 3:
                return 0.70, [], []  # Likely IT tender, positive signal
            elif it_hits >= 1:
                return 0.60, [], []  # Some IT signal
            return 0.5, [], []  # Truly neutral

        matched = tender_skills & profile.skill_set
        missing = tender_skills - profile.skill_set

        # Score: how many of the company's skills are mentioned
        # Normalized by total skills found (favors tenders where we match most)
        overlap = len(matched) / max(len(tender_skills), 1)

        return (
            round(overlap, 4),
            sorted(matched),
            sorted(missing),
        )

    def compute_cert_overlap(
        self,
        tender: Dict,
        profile: CompanyProfile,
    ) -> float:
        """
        Certification overlap — binary bonus.

        Returns 1.0 if any required certifications match, else 0.0.
        This is used as a tie-breaker, not a primary score.
        """
        tender_certs = {
            c.lower().strip()
            for c in tender.get("detected_certifications", [])
        }
        if not tender_certs:
            return 0.5  # Neutral

        return 1.0 if (tender_certs & profile.cert_set) else 0.0

    # ============================================================
    # HYBRID SCORE (final combined score)
    # ============================================================

    def compute_hybrid_score(
        self,
        tender: Dict,
        tender_embedding: np.ndarray,
        profile: CompanyProfile,
    ) -> Dict:
        """
        Compute the final hybrid relevance score.

        Formula:
            score = w_s × semantic_sim + w_k × skill_overlap + w_d × domain_sim

        Where:
            w_s = 0.45 (semantic weight)
            w_k = 0.35 (skill overlap weight)
            w_d = 0.20 (domain match weight)

        Args:
            tender:           Tender dict.
            tender_embedding: Precomputed 384-dim vector.
            profile:          CompanyProfile with embeddings.

        Returns:
            Dict with score breakdown:
            {
                "final_score": 0.72,
                "semantic_similarity": 0.68,
                "skill_overlap": 0.80,
                "domain_similarity": 0.65,
                "best_matching_domain": "Cloud Computing",
                "matched_skills": ["python", "aws"],
                "missing_skills": ["sap"],
            }
        """
        # Component 1: Semantic similarity (SBERT cosine)
        semantic_sim = self.compute_semantic_similarity(
            tender_embedding, profile
        )

        # Component 2: Skill overlap (Jaccard)
        skill_score, matched, missing = self.compute_skill_overlap(
            tender, profile
        )

        # Component 3: Domain similarity (max across company domains)
        domain_sim, best_domain = self.compute_domain_similarity(
            tender_embedding, profile
        )

        # Component 4: IT keyword signal (language-agnostic)
        keyword_score = self._compute_keyword_signal(tender)

        # Weighted combination
        w = self.weights
        final = (
            w.get("semantic", 0.35) * semantic_sim
            + w.get("skill_overlap", 0.30) * skill_score
            + w.get("domain_match", 0.15) * domain_sim
            + w.get("keyword_signal", 0.20) * keyword_score
        )

        # Domain confidence boost: if the best matching domain is one of
        # the company's core domains AND domain_sim > threshold, apply a boost
        # This rewards tenders that clearly fall in our expertise
        if best_domain in ("IT Services", "Cloud Computing", "ERP",
                           "AI/Machine Learning", "Cybersecurity",
                           "Data Analytics", "Digital Transformation"):
            if domain_sim > 0.45:
                boost = 0.10  # +10% for strong domain alignment
                final = min(1.0, final + boost)
            elif domain_sim > 0.35:
                boost = 0.06  # +6% for moderate domain alignment
                final = min(1.0, final + boost)
            elif domain_sim > 0.25:
                boost = 0.03  # +3% for weak domain alignment
                final = min(1.0, final + boost)

        # Keyword density boost: if many IT keywords found, tender is
        # strongly IT-related even if semantic similarity is moderate
        if keyword_score > 0.6:
            final = min(1.0, final + 0.05)  # +5% for keyword-rich IT tenders

        return {
            "final_score": round(float(final), 4),
            "semantic_similarity": round(float(semantic_sim), 4),
            "skill_overlap": round(float(skill_score), 4),
            "domain_similarity": round(float(domain_sim), 4),
            "keyword_signal": round(float(keyword_score), 4),
            "best_matching_domain": best_domain,
            "matched_skills": matched,
            "missing_skills": missing,
        }

    # ============================================================
    # BATCH HYBRID SCORES
    # ============================================================

    def compute_batch_scores(
        self,
        tenders: List[Dict],
        profile: CompanyProfile,
    ) -> List[Dict]:
        """
        Compute hybrid scores for a batch of tenders efficiently.

        Uses SBERT batch encoding for the embedding step,
        then computes per-tender scores sequentially.

        Args:
            tenders: List of tender dicts.
            profile: CompanyProfile with precomputed embeddings.

        Returns:
            List of score dicts (same order as input).
        """
        # Batch encode all tenders
        embeddings = self.embed_tenders_batch(tenders)

        # Compute scores
        results = []
        for i, tender in enumerate(tenders):
            score = self.compute_hybrid_score(
                tender, embeddings[i], profile
            )
            results.append(score)

        return results

    # ============================================================
    # KEYWORD SIGNAL
    # ============================================================

    def _compute_keyword_signal(
        self,
        tender: Dict,
    ) -> float:
        """
        Language-agnostic IT keyword detection.

        Scans title + description for universal IT terms (technology
        names, frameworks, cloud providers) that are NOT translated
        across languages. This compensates for cross-lingual
        embedding degradation.

        Returns:
            Score in [0, 1]. 0 = no IT keywords, 1 = many IT keywords.
        """
        text = (
            (tender.get("title", "") + " " + tender.get("description", ""))
            .lower()
        )
        if not text.strip():
            return 0.0

        hits = 0
        for kw in self.IT_KEYWORDS:
            if kw in text:
                hits += 1

        # Sigmoid-like scaling: diminishing returns after 5 hits
        # 1 hit → 0.30, 2 → 0.50, 3 → 0.65, 5 → 0.80, 8+ → 0.95
        if hits == 0:
            return 0.0
        score = 1.0 - (1.0 / (1.0 + 0.4 * hits))
        return min(1.0, round(score, 4))

    # ============================================================
    # INTERNAL
    # ============================================================

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single text string to a normalized embedding."""
        return self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
