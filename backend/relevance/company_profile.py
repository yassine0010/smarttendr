"""
SmartTender AI — Company Profile Representation
==================================================
Structured company expertise profile used as the reference
vector for relevance filtering.

The profile contains:
    - Free-text description (→ SBERT embedding)
    - Multiple domain labels (→ per-domain text for multi-domain matching)
    - Skill list (→ set for overlap computation)
    - Domain weights (optional — prioritize certain expertise areas)

All embeddings are precomputed at init time and cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


# ================================================================
# DEFAULT COMPANY PROFILE
# ================================================================

DEFAULT_PROFILE_DATA: Dict = {
    "name": "Inetum Tunisie",
    "description": (
        # English
        "Inetum is a leading IT services company specializing in "
        "digital transformation, cloud computing, ERP implementation, "
        "AI/ML solutions, custom software development, cybersecurity, "
        "and IT consulting. We operate across Europe, Africa, and the "
        "Middle East with expertise in Python, Java, AWS, Azure, SAP, "
        "Odoo, data analytics, and agile project management. "
        # French (for cross-lingual SBERT matching with FR/BE/LU tenders)
        "Inetum est une entreprise leader en services informatiques, "
        "spécialisée dans la transformation numérique, le cloud computing, "
        "l'implémentation ERP, les solutions d'intelligence artificielle, "
        "le développement logiciel sur mesure, la cybersécurité et le conseil IT. "
        "Nous intervenons en Europe, Afrique et Moyen-Orient avec des compétences "
        "en Python, Java, AWS, Azure, SAP, Odoo, analyse de données et gestion de projet agile. "
        # German (for DE/AT/CH tenders on TED Europa)
        "Inetum ist ein führendes IT-Dienstleistungsunternehmen mit Schwerpunkt "
        "auf digitale Transformation, Cloud Computing, ERP-Implementierung, "
        "KI/ML-Lösungen, Softwareentwicklung, Cybersicherheit und IT-Beratung. "
        # Spanish (for ES/LATAM tenders)
        "Inetum es una empresa líder en servicios de TI especializada en "
        "transformación digital, computación en la nube, implementación ERP, "
        "soluciones de IA, desarrollo de software, ciberseguridad y consultoría TI."
    ),
    "domains": {
        "IT Services": {
            "weight": 1.0,
            "description": (
                "Custom software development, system integration, "
                "web and mobile application development, IT consulting, "
                "technical support and managed services. "
                "Développement logiciel, intégration de systèmes, "
                "applications web et mobiles, conseil informatique. "
                "Softwareentwicklung, Systemintegration, IT-Beratung, IT-Dienstleistungen."
            ),
        },
        "Cloud Computing": {
            "weight": 0.95,
            "description": (
                "Cloud migration, cloud-native development, AWS, Azure, "
                "GCP, infrastructure as code, DevOps, Docker, Kubernetes, "
                "serverless architectures, SaaS platform development. "
                "Migration cloud, développement cloud-native, infrastructure as code. "
                "Cloud-Migration, Cloud-Infrastruktur, DevOps."
            ),
        },
        "ERP": {
            "weight": 0.90,
            "description": (
                "Enterprise resource planning implementation, SAP, Odoo, "
                "Dynamics 365, ERP customization, ERP migration, "
                "financial and HR modules, procurement workflows. "
                "Mise en œuvre ERP, personnalisation SAP, progiciel de gestion intégrée. "
                "ERP-Implementierung, SAP-Beratung, Unternehmenssoftware."
            ),
        },
        "AI/Machine Learning": {
            "weight": 0.85,
            "description": (
                "Artificial intelligence solutions, machine learning models, "
                "natural language processing, chatbots, computer vision, "
                "data science, predictive analytics, deep learning. "
                "Intelligence artificielle, apprentissage automatique, "
                "traitement du langage naturel, science des données. "
                "Künstliche Intelligenz, maschinelles Lernen, Datenwissenschaft."
            ),
        },
        "Cybersecurity": {
            "weight": 0.80,
            "description": (
                "Security audits, penetration testing, vulnerability assessment, "
                "SIEM, SOC, ISO 27001 compliance, GDPR, information security "
                "consulting, incident response. "
                "Audit de sécurité, test d'intrusion, cybersécurité, conformité RGPD. "
                "Sicherheitsaudit, Penetrationstest, Cybersicherheit, IT-Sicherheit."
            ),
        },
        "Data Analytics": {
            "weight": 0.80,
            "description": (
                "Business intelligence, data warehousing, reporting dashboards, "
                "ETL pipelines, big data processing, data visualization, "
                "Power BI, Tableau, data governance. "
                "Informatique décisionnelle, entrepôt de données, tableaux de bord. "
                "Business Intelligence, Datenanalyse, Berichtswesen."
            ),
        },
        "Digital Transformation": {
            "weight": 0.75,
            "description": (
                "Digital strategy consulting, process digitization, "
                "legacy modernization, e-government solutions, "
                "change management, agile transformation. "
                "Conseil en stratégie numérique, modernisation des systèmes, "
                "administration électronique. "
                "Digitale Strategie, Prozessdigitalisierung, E-Government."
            ),
        },
    },
    "skills": [
        "Python", "Java", "JavaScript", "TypeScript", "SQL",
        "AWS", "Azure", "Docker", "Kubernetes", "Terraform",
        "SAP", "Odoo", "Salesforce",
        "React", "Angular", "Node.js", "Django", "FastAPI", ".NET", "Spring",
        "PostgreSQL", "MySQL", "MongoDB", "Redis",
        "TensorFlow", "PyTorch", "NLP", "Machine Learning",
        "CI/CD", "DevOps", "Agile/Scrum",
        "REST API", "Microservices", "GraphQL",
        "Cybersecurity", "Penetration Testing", "ISO 27001",
    ],
    "certifications": [
        "AWS Certified", "Azure Certified", "PMP", "ITIL",
        "ISO 27001", "ISO 9001", "CISSP", "GDPR",
    ],
}


# ================================================================
# COMPANY PROFILE DATACLASS
# ================================================================

@dataclass
class CompanyProfile:
    """
    Structured representation of a company's expertise.

    Attributes:
        name:           Company display name.
        description:    Free-text description (used for SBERT embedding).
        domains:        Dict of domain_name → {weight, description}.
        skills:         List of technology/skill names.
        certifications: List of certification names.

    Precomputed (set by SimilarityEngine.load_profile):
        description_embedding: np.ndarray of shape (384,) for all-MiniLM-L6-v2
        domain_embeddings:     Dict[str, np.ndarray] — one embedding per domain
        skill_set:             Lowercased set for O(1) overlap lookup
        cert_set:              Lowercased set for certification matching
    """

    name: str = ""
    description: str = ""
    domains: Dict[str, Dict] = field(default_factory=dict)
    skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)

    # Precomputed vectors (populated by SimilarityEngine)
    description_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    domain_embeddings: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    skill_set: Set[str] = field(default_factory=set, repr=False)
    cert_set: Set[str] = field(default_factory=set, repr=False)

    def __post_init__(self):
        self.skill_set = {s.lower().strip() for s in self.skills}
        self.cert_set = {c.lower().strip() for c in self.certifications}

    @classmethod
    def from_dict(cls, data: Dict) -> "CompanyProfile":
        """Build profile from a dictionary (e.g. DEFAULT_PROFILE_DATA)."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            domains=data.get("domains", {}),
            skills=data.get("skills", []),
            certifications=data.get("certifications", []),
        )

    @classmethod
    def default(cls) -> "CompanyProfile":
        """Return the default Inetum profile."""
        return cls.from_dict(DEFAULT_PROFILE_DATA)

    def domain_names(self) -> List[str]:
        """List of domain labels."""
        return list(self.domains.keys())

    def domain_weight(self, domain: str) -> float:
        """Get the priority weight for a domain (default 0.5)."""
        info = self.domains.get(domain, {})
        return info.get("weight", 0.5)

    def domain_description(self, domain: str) -> str:
        """Get the text description for a domain."""
        info = self.domains.get(domain, {})
        return info.get("description", domain)

    def full_text(self) -> str:
        """
        Concatenated text of description + all domain descriptions.
        Used as the single SBERT input for the global company vector.
        """
        parts = [self.description]
        for domain, info in self.domains.items():
            parts.append(f"{domain}: {info.get('description', '')}")
        parts.append("Skills: " + ", ".join(self.skills))
        return " ".join(parts)
