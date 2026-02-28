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
        "semantic": 0.45,       # SBERT cosine similarity
        "skill_overlap": 0.35,  # Jaccard-style skill matching
        "domain_match": 0.20,   # Domain alignment bonus
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

        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total}: {self.weights}"
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
        # Collect tender skills from multiple possible fields
        tender_skills_raw = (
            tender.get("detected_skills", [])
            or tender.get("required_skills", [])
        )
        tender_skills = {s.lower().strip() for s in tender_skills_raw if s}

        if not tender_skills:
            return 0.5, [], []  # Neutral if no skills specified

        matched = tender_skills & profile.skill_set
        missing = tender_skills - profile.skill_set

        overlap = len(matched) / len(tender_skills)

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

        # Weighted combination
        final = (
            self.weights["semantic"] * semantic_sim
            + self.weights["skill_overlap"] * skill_score
            + self.weights["domain_match"] * domain_sim
        )

        return {
            "final_score": round(float(final), 4),
            "semantic_similarity": round(float(semantic_sim), 4),
            "skill_overlap": round(float(skill_score), 4),
            "domain_similarity": round(float(domain_sim), 4),
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
    # INTERNAL
    # ============================================================

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single text string to a normalized embedding."""
        return self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
