"""
SmartTender AI — Domain Taxonomy & Skill Dictionary
=====================================================
Curated knowledge bases for classifying tender domains, mapping
technologies/skills, and detecting certifications.

Design Decisions:
    - Flat dictionaries for O(1) lookup, not nested hierarchies
    - Lowercase keys for case-insensitive matching
    - Weighted skill categories for importance ranking
    - Extensible: add new entries without changing extraction logic

Author: SmartTender AI Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple


# ================================================================
# DOMAIN / SECTOR TAXONOMY
# ================================================================

# Maps keyword triggers → canonical domain label.
# Each key is a lowercased term; if it appears in the TF-IDF top-K
# or noun chunks, we assign the corresponding domain.
DOMAIN_TAXONOMY: Dict[str, str] = {
    # IT & Software
    "software": "IT Services",
    "software development": "IT Services",
    "information technology": "IT Services",
    "it services": "IT Services",
    "web development": "IT Services",
    "mobile application": "IT Services",
    "application development": "IT Services",
    "digital transformation": "IT Services",
    "system integration": "IT Services",
    "database": "IT Services",
    "data management": "IT Services",
    "it infrastructure": "IT Services",
    "cloud": "Cloud Computing",
    "cloud computing": "Cloud Computing",
    "cloud migration": "Cloud Computing",
    "saas": "Cloud Computing",
    "paas": "Cloud Computing",
    "iaas": "Cloud Computing",
    "erp": "ERP",
    "enterprise resource planning": "ERP",
    "sap": "ERP",
    "odoo": "ERP",
    "dynamics": "ERP",
    # AI & Data
    "artificial intelligence": "AI/Machine Learning",
    "machine learning": "AI/Machine Learning",
    "deep learning": "AI/Machine Learning",
    "nlp": "AI/Machine Learning",
    "natural language processing": "AI/Machine Learning",
    "computer vision": "AI/Machine Learning",
    "data science": "AI/Machine Learning",
    "data analytics": "Data Analytics",
    "business intelligence": "Data Analytics",
    "big data": "Data Analytics",
    "data warehouse": "Data Analytics",
    "reporting": "Data Analytics",
    # Cybersecurity
    "cybersecurity": "Cybersecurity",
    "penetration testing": "Cybersecurity",
    "security audit": "Cybersecurity",
    "vulnerability": "Cybersecurity",
    "information security": "Cybersecurity",
    "siem": "Cybersecurity",
    "soc": "Cybersecurity",
    "firewall": "Cybersecurity",
    "encryption": "Cybersecurity",
    # Construction & Engineering
    "construction": "Construction",
    "building": "Construction",
    "civil engineering": "Construction",
    "road construction": "Construction",
    "infrastructure": "Construction",
    "bridge": "Construction",
    "renovation": "Construction",
    "architecture": "Construction",
    # Healthcare
    "healthcare": "Healthcare",
    "medical": "Healthcare",
    "hospital": "Healthcare",
    "pharmaceutical": "Healthcare",
    "health information": "Healthcare",
    "telemedicine": "Healthcare",
    "clinical": "Healthcare",
    "laboratory": "Healthcare",
    # Energy
    "energy": "Energy",
    "solar": "Energy",
    "wind energy": "Energy",
    "renewable": "Energy",
    "electricity": "Energy",
    "power plant": "Energy",
    "oil and gas": "Energy",
    "petroleum": "Energy",
    # Telecommunications
    "telecommunications": "Telecommunications",
    "telecom": "Telecommunications",
    "5g": "Telecommunications",
    "fiber optic": "Telecommunications",
    "network infrastructure": "Telecommunications",
    # Education & Training
    "education": "Education & Training",
    "training": "Education & Training",
    "e-learning": "Education & Training",
    "curriculum": "Education & Training",
    "university": "Education & Training",
    # Consulting
    "consulting": "Consulting",
    "advisory": "Consulting",
    "management consulting": "Consulting",
    "strategy": "Consulting",
    "feasibility study": "Consulting",
    # Supply & Logistics
    "logistics": "Supply & Logistics",
    "supply chain": "Supply & Logistics",
    "procurement": "Supply & Logistics",
    "warehousing": "Supply & Logistics",
    "transportation": "Supply & Logistics",
    # Finance
    "financial": "Finance",
    "banking": "Finance",
    "accounting": "Finance",
    "audit": "Finance",
    "insurance": "Finance",
    # Environment
    "environment": "Environment",
    "waste management": "Environment",
    "water treatment": "Environment",
    "pollution": "Environment",
    "sustainability": "Environment",

    # ── FRENCH DOMAIN KEYWORDS ──
    # IT & Software
    "logiciel": "IT Services",
    "développement logiciel": "IT Services",
    "informatique": "IT Services",
    "services informatiques": "IT Services",
    "développement web": "IT Services",
    "application mobile": "IT Services",
    "transformation numérique": "IT Services",
    "transformation digitale": "IT Services",
    "intégration système": "IT Services",
    "système d'information": "IT Services",
    "base de données": "IT Services",
    "infrastructure informatique": "IT Services",
    "numérique": "IT Services",
    # Cloud
    "cloud": "Cloud Computing",
    "infonuagique": "Cloud Computing",
    "hébergement cloud": "Cloud Computing",
    # ERP
    "progiciel de gestion": "ERP",
    "progiciel": "ERP",
    "gestion intégrée": "ERP",
    # AI & Data
    "intelligence artificielle": "AI/Machine Learning",
    "apprentissage automatique": "AI/Machine Learning",
    "traitement du langage naturel": "AI/Machine Learning",
    "science des données": "AI/Machine Learning",
    "analyse de données": "Data Analytics",
    "tableau de bord": "Data Analytics",
    "entrepôt de données": "Data Analytics",
    # Cybersecurity
    "cybersécurité": "Cybersecurity",
    "sécurité informatique": "Cybersecurity",
    "audit de sécurité": "Cybersecurity",
    "test d'intrusion": "Cybersecurity",
    "test de pénétration": "Cybersecurity",
    "sécurité des systèmes": "Cybersecurity",
    # Construction
    "construction": "Construction",
    "bâtiment": "Construction",
    "génie civil": "Construction",
    "travaux publics": "Construction",
    "rénovation": "Construction",
    "étanchéité": "Construction",
    "clôture": "Construction",
    # Healthcare
    "santé": "Healthcare",
    "médical": "Healthcare",
    "hôpital": "Healthcare",
    "pharmaceutique": "Healthcare",
    "laboratoire": "Healthcare",
    # Energy
    "énergie": "Energy",
    "énergie solaire": "Energy",
    "énergie renouvelable": "Energy",
    "électricité": "Energy",
    "pétrole": "Energy",
    # Telecommunications
    "télécommunications": "Telecommunications",
    "télécom": "Telecommunications",
    "fibre optique": "Telecommunications",
    "réseau": "Telecommunications",
    # Education
    "éducation": "Education & Training",
    "formation": "Education & Training",
    "enseignement": "Education & Training",
    "e-learning": "Education & Training",
    "université": "Education & Training",
    "examen": "Education & Training",
    "diplôme": "Education & Training",
    # Consulting
    "conseil": "Consulting",
    "étude de faisabilité": "Consulting",
    "assistance technique": "Consulting",
    "maîtrise d'ouvrage": "Consulting",
    # Supply & Logistics
    "logistique": "Supply & Logistics",
    "approvisionnement": "Supply & Logistics",
    "chaîne d'approvisionnement": "Supply & Logistics",
    "transport": "Supply & Logistics",
    "acquisition": "Supply & Logistics",
    "fourniture": "Supply & Logistics",
    "matériel roulant": "Supply & Logistics",
    # Finance
    "financier": "Finance",
    "comptabilité": "Finance",
    "audit": "Finance",
    "assurance": "Finance",
    # Environment
    "environnement": "Environment",
    "gestion des déchets": "Environment",
    "traitement des eaux": "Environment",
    "eau potable": "Environment",
    "développement durable": "Environment",

    # ── ARABIC DOMAIN KEYWORDS ──
    "تكنولوجيا المعلومات": "IT Services",
    "خدمات معلوماتية": "IT Services",
    "تطوير البرمجيات": "IT Services",
    "التحول الرقمي": "IT Services",
    "حوسبة سحابية": "Cloud Computing",
    "الذكاء الاصطناعي": "AI/Machine Learning",
    "أمن المعلومات": "Cybersecurity",
    "الأمن السيبراني": "Cybersecurity",
    "أشغال": "Construction",
    "بناء": "Construction",
    "صحة": "Healthcare",
    "طاقة": "Energy",
    "تعليم": "Education & Training",
    "تكوين": "Education & Training",
    "نقل": "Supply & Logistics",
    "تزويد": "Supply & Logistics",
    "ماء": "Environment",
    "الماء الصالح للشرب": "Environment",
    "بيئة": "Environment",
}


# ================================================================
# TECHNOLOGY & SKILL PATTERNS
# ================================================================

@dataclass(frozen=True)
class SkillEntry:
    """A single skill/technology with metadata for ranking."""
    canonical: str          # Display name (e.g. "Python")
    category: str           # Category bucket (e.g. "Programming Language")
    weight: float = 1.0     # Importance multiplier for ranking
    aliases: Tuple[str, ...] = ()  # Alternative names / abbreviations


# Categories ordered by relevance weight for IT tenders
SKILL_CATEGORIES_WEIGHT: Dict[str, float] = {
    "Programming Language": 1.0,
    "Framework": 0.95,
    "Cloud Platform": 0.95,
    "Database": 0.90,
    "DevOps": 0.90,
    "ERP System": 0.85,
    "AI/ML Tool": 0.85,
    "Methodology": 0.70,
    "Certification": 0.80,
    "Security Tool": 0.85,
    "Operating System": 0.60,
    "Protocol/Standard": 0.75,
    "Other": 0.50,
}

# Master skill dictionary — regex patterns as keys for matching
# Organized by category. Each tuple: (canonical_name, regex_pattern)
SKILL_PATTERNS: List[Tuple[str, str, str]] = [
    # ── Programming Languages ──
    ("Python",          "Programming Language",  r"\bPython\b"),
    ("Java",            "Programming Language",  r"\bJava\b(?!\s*Script)"),
    ("JavaScript",      "Programming Language",  r"\bJavaScript\b"),
    ("TypeScript",      "Programming Language",  r"\bTypeScript\b"),
    ("C++",             "Programming Language",  r"\bC\+\+\b"),
    ("C#",              "Programming Language",  r"\bC#\b"),
    ("Go",              "Programming Language",  r"\bGolang\b|\bGo\s+language\b"),
    ("Rust",            "Programming Language",  r"\bRust\b"),
    ("Ruby",            "Programming Language",  r"\bRuby\b"),
    ("PHP",             "Programming Language",  r"\bPHP\b"),
    ("R",               "Programming Language",  r"\bR\s+programming\b|\bR\s+language\b"),
    ("Scala",           "Programming Language",  r"\bScala\b"),
    ("Swift",           "Programming Language",  r"\bSwift\b"),
    ("Kotlin",          "Programming Language",  r"\bKotlin\b"),
    ("SQL",             "Programming Language",  r"\bSQL\b"),
    ("MATLAB",          "Programming Language",  r"\bMATLAB\b"),
    # ── Frameworks ──
    ("React",           "Framework",  r"\bReact(?:\.js|JS)?\b"),
    ("Angular",         "Framework",  r"\bAngular(?:JS)?\b"),
    ("Vue.js",          "Framework",  r"\bVue(?:\.js)?\b"),
    ("Node.js",         "Framework",  r"\bNode(?:\.js|JS)\b"),
    ("Django",          "Framework",  r"\bDjango\b"),
    ("Flask",           "Framework",  r"\bFlask\b"),
    ("FastAPI",         "Framework",  r"\bFastAPI\b"),
    ("Spring",          "Framework",  r"\bSpring(?:\s+Boot)?\b"),
    ("Laravel",         "Framework",  r"\bLaravel\b"),
    (".NET",            "Framework",  r"\b\.NET\b|ASP\.NET"),
    ("Next.js",         "Framework",  r"\bNext(?:\.js|JS)\b"),
    ("Express.js",      "Framework",  r"\bExpress(?:\.js)?\b"),
    # ── Cloud Platforms ──
    ("AWS",             "Cloud Platform",  r"\bAWS\b|Amazon\s+Web\s+Services"),
    ("Azure",           "Cloud Platform",  r"\bAzure\b"),
    ("GCP",             "Cloud Platform",  r"\bGCP\b|Google\s+Cloud"),
    ("Oracle Cloud",    "Cloud Platform",  r"\bOracle\s+Cloud\b|OCI\b"),
    # ── Databases ──
    ("PostgreSQL",      "Database",  r"\bPostgreSQL\b|\bPostgres\b"),
    ("MySQL",           "Database",  r"\bMySQL\b"),
    ("MongoDB",         "Database",  r"\bMongoDB\b"),
    ("Redis",           "Database",  r"\bRedis\b"),
    ("Elasticsearch",   "Database",  r"\bElasticsearch\b|\bElastic\b"),
    ("Oracle DB",       "Database",  r"\bOracle\s+(?:DB|Database)\b"),
    ("SQL Server",      "Database",  r"\bSQL\s+Server\b|MSSQL"),
    ("Cassandra",       "Database",  r"\bCassandra\b"),
    ("DynamoDB",        "Database",  r"\bDynamoDB\b"),
    # ── DevOps ──
    ("Docker",          "DevOps",  r"\bDocker\b"),
    ("Kubernetes",      "DevOps",  r"\bKubernetes\b|\bK8s\b"),
    ("Terraform",       "DevOps",  r"\bTerraform\b"),
    ("Ansible",         "DevOps",  r"\bAnsible\b"),
    ("Jenkins",         "DevOps",  r"\bJenkins\b"),
    ("GitLab CI",       "DevOps",  r"\bGitLab\s*CI\b"),
    ("GitHub Actions",  "DevOps",  r"\bGitHub\s+Actions\b"),
    ("CI/CD",           "DevOps",  r"\bCI\s*/?\s*CD\b"),
    ("DevOps",          "DevOps",  r"\bDevOps\b"),
    ("MLOps",           "DevOps",  r"\bMLOps\b"),
    # ── ERP Systems ──
    ("SAP",             "ERP System",  r"\bSAP\b"),
    ("Odoo",            "ERP System",  r"\bOdoo\b"),
    ("Salesforce",      "ERP System",  r"\bSalesforce\b"),
    ("ServiceNow",      "ERP System",  r"\bServiceNow\b"),
    ("Dynamics 365",    "ERP System",  r"\bDynamics\s*(?:365)?\b"),
    # ── AI/ML Tools ──
    ("TensorFlow",      "AI/ML Tool",  r"\bTensorFlow\b"),
    ("PyTorch",         "AI/ML Tool",  r"\bPyTorch\b"),
    ("Scikit-learn",    "AI/ML Tool",  r"\bScikit[\s-]*learn\b|sklearn"),
    ("Keras",           "AI/ML Tool",  r"\bKeras\b"),
    ("OpenCV",          "AI/ML Tool",  r"\bOpenCV\b"),
    ("Hugging Face",    "AI/ML Tool",  r"\bHugging\s*Face\b"),
    ("LangChain",       "AI/ML Tool",  r"\bLangChain\b"),
    ("spaCy",           "AI/ML Tool",  r"\bspaCy\b"),
    ("NLTK",            "AI/ML Tool",  r"\bNLTK\b"),
    ("Rasa",            "AI/ML Tool",  r"\bRasa\b"),
    ("Dialogflow",      "AI/ML Tool",  r"\bDialogflow\b"),
    # ── Methodologies ──
    ("Agile",           "Methodology",  r"\bAgile\b"),
    ("Scrum",           "Methodology",  r"\bScrum\b"),
    ("Kanban",          "Methodology",  r"\bKanban\b"),
    ("Waterfall",       "Methodology",  r"\bWaterfall\b"),
    ("ITIL",            "Methodology",  r"\bITIL\b"),
    ("PRINCE2",         "Methodology",  r"\bPRINCE2\b"),
    ("Six Sigma",       "Methodology",  r"\bSix\s+Sigma\b"),
    ("Lean",            "Methodology",  r"\bLean\s+(?:IT|Management|Development)\b"),
    # ── Certifications ──
    ("PMP",             "Certification",  r"\bPMP\b"),
    ("CISSP",           "Certification",  r"\bCISSP\b"),
    ("CEH",             "Certification",  r"\bCEH\b"),
    ("OSCP",            "Certification",  r"\bOSCP\b"),
    ("AWS Certified",   "Certification",  r"\bAWS\s+Certified\b"),
    ("Azure Certified", "Certification",  r"\bAzure\s+Certified\b"),
    ("ISO 27001",       "Certification",  r"\bISO\s*27001\b"),
    ("ISO 9001",        "Certification",  r"\bISO\s*9001\b"),
    ("SOC 2",           "Certification",  r"\bSOC\s*2\b"),
    ("GDPR",            "Certification",  r"\bGDPR\b"),
    ("HIPAA",           "Certification",  r"\bHIPAA\b"),
    ("OWASP",           "Certification",  r"\bOWASP\b"),
    # ── Security Tools ──
    ("Kali Linux",      "Security Tool",  r"\bKali\s+Linux\b"),
    ("Nmap",            "Security Tool",  r"\bNmap\b"),
    ("Metasploit",      "Security Tool",  r"\bMetasploit\b"),
    ("Burp Suite",      "Security Tool",  r"\bBurp\s+Suite\b"),
    ("Wireshark",       "Security Tool",  r"\bWireshark\b"),
    ("Splunk",          "Security Tool",  r"\bSplunk\b"),
    # ── Protocols / Standards ──
    ("REST API",        "Protocol/Standard",  r"\bREST(?:ful)?\s*API\b"),
    ("GraphQL",         "Protocol/Standard",  r"\bGraphQL\b"),
    ("gRPC",            "Protocol/Standard",  r"\bgRPC\b"),
    ("Microservices",   "Protocol/Standard",  r"\bMicroservices?\b"),
    ("SOAP",            "Protocol/Standard",  r"\bSOAP\b"),
    ("WebSocket",       "Protocol/Standard",  r"\bWebSocket\b"),
    # ── Operating Systems ──
    ("Linux",           "Operating System",  r"\bLinux\b"),
    ("Windows Server",  "Operating System",  r"\bWindows\s+Server\b"),
    ("Unix",            "Operating System",  r"\bUnix\b"),
]


# ================================================================
# CURRENCY PATTERNS
# ================================================================

CURRENCY_MAP: Dict[str, str] = {
    "$": "USD",
    "usd": "USD",
    "us$": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "tnd": "TND",
    "dt": "TND",
    "dinars": "TND",
    "dinar": "TND",
    "chf": "CHF",
    "cad": "CAD",
    "aud": "AUD",
    "sek": "SEK",
    "nok": "NOK",
    "dkk": "DKK",
    "jpy": "JPY",
    "¥": "JPY",
    "inr": "INR",
    "₹": "INR",
    "zar": "ZAR",
    "mad": "MAD",
    "dzd": "DZD",
    "egp": "EGP",
    "xof": "XOF",
    "xaf": "XAF",
    # French terms
    "euros": "EUR",
    "euro": "EUR",
    "dinars tunisiens": "TND",
    "dinar tunisien": "TND",
    "millimes": "TND",
    "dirhams": "MAD",
    "dirham": "MAD",
    "francs": "XOF",
    "franc cfa": "XOF",
}


# ================================================================
# DEADLINE SIGNAL WORDS
# ================================================================
# Words/phrases that typically precede the *submission* deadline
# (as opposed to project-start or project-end dates).
DEADLINE_SIGNALS: List[str] = [
    "submission deadline",
    "closing date",
    "deadline for submission",
    "proposals must be received by",
    "proposals due",
    "bids due",
    "offer deadline",
    "last date for submission",
    "response deadline",
    "due date",
    "submit by",
    "submit before",
    "no later than",
    "applications close",
    "tender closing",
    "expiry date",
    # French
    "date limite",
    "date de clôture",
    "délai de soumission",
    "date limite de réception",
    "date limite de remise",
    "ouverture des plis",
    "avant le",
    "au plus tard le",
    # Arabic
    "تاريخ الإغلاق",
    "آخر أجل",
    "تاريخ الانتهاء",
    "الموعد النهائي",
]


# ================================================================
# NOISE / STOPWORD EXTENSIONS
# ================================================================
# Domain-specific stopwords that TF-IDF should ignore beyond
# sklearn's built-in English stopwords.
TENDER_STOPWORDS: FrozenSet[str] = frozenset({
    "tender", "proposal", "bid", "rfp", "rfi", "rfq", "eoi",
    "request", "quotation", "procurement", "contract", "notice",
    "invitation", "solicitation", "amendment", "addendum",
    "annex", "attachment", "appendix", "section", "article",
    "clause", "paragraph", "herein", "thereof", "hereby",
    "undersigned", "signatory", "offeror", "bidder", "contractor",
    "vendor", "supplier", "applicant", "respondent",
    "shall", "must", "may", "should", "will", "required",
    "provide", "submit", "ensure", "include", "comply",
    "accordance", "pursuant", "reference", "regard",
    # French procurement stopwords
    "appel", "offre", "marché", "soumission", "fournisseur",
    "cahier", "charges", "cahier des charges", "lot", "lots",
    "prestation", "prestations", "objet", "titulaire",
    "montant", "doit", "peut", "sera", "sont", "dans", "pour",
    "avec", "des", "les", "une", "par", "sur", "aux", "cette",
    "tout", "tous", "toute", "toutes", "entre", "selon",
    "conformément", "relatif", "relative", "concernant",
    "acquisition", "fourniture", "travaux",
    # Arabic procurement stopwords
    "عرض", "مناقصة", "عقد", "شراء", "توريد",
    "من", "في", "على", "إلى", "عن", "مع",
    "هذا", "هذه", "التي", "الذي", "التى",
})
