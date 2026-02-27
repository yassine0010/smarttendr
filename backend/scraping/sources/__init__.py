"""
SmartTender AI — Scraper Sources Package
==========================================
Auto-imports all platform scrapers so they register with ScraperRegistry.
"""

from . import sam_gov
from . import ted_europa
from . import ungm
from . import tuneps
from . import contracts_finder

__all__ = [
    "sam_gov",
    "ted_europa",
    "ungm",
    "tuneps",
    "contracts_finder",
]
