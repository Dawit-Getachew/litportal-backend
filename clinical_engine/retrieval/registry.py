"""Guideline issuer registry (W2.3) — PROPOSED, pending owner review.

Each entry names an organisation whose guidelines LitPortal recognises, how its
guidelines reach LitPortal today, and what that permits. No official feed is
connected yet: every guideline is found through PubMed / Europe PMC indexing,
and is labelled that way ("indexed fallback"). Feeds that need a licence or
whose terms forbid automated use are marked blocked rather than scraped.

Indexing grants no processing permission by itself: AI use of a guideline's
abstract follows the source policy of the index it came from (PubMed /
Europe PMC: title + abstract), never the guideline's own full text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

REGISTRY_VERSION = "issuers.v1-proposed"
REGISTRY_STATUS = "proposed — pending owner review"


@dataclass(frozen=True)
class Issuer:
    id: str
    name: str
    aliases: tuple[str, ...]  # regexes, matched case-insensitively
    official_feed: str        # not_connected | blocked_licence | blocked_terms
    note: str = ""


ISSUERS: tuple[Issuer, ...] = (
    Issuer("ssc", "Surviving Sepsis Campaign (SCCM and ESICM)", (r"surviving sepsis campaign",), "not_connected"),
    Issuer("sccm", "Society of Critical Care Medicine", (r"society of critical care medicine", r"\bsccm\b",
                                                         r"american college of critical care medicine"),
           "not_connected"),
    Issuer("esicm", "European Society of Intensive Care Medicine", (r"european society of intensive care medicine",
                                                                    r"\besicm\b"), "not_connected"),
    Issuer("idsa", "Infectious Diseases Society of America", (r"infectious diseases society of america", r"\bidsa\b"),
           "not_connected"),
    Issuer("ats", "American Thoracic Society", (r"american thoracic society", r"\bats\b(?= (?:clinical|official))"),
           "not_connected"),
    Issuer("ers", "European Respiratory Society", (r"european respiratory society",), "not_connected"),
    Issuer("chest", "American College of Chest Physicians (CHEST)", (r"american college of chest physicians",
                                                                     r"\bchest guideline"), "not_connected"),
    Issuer("aha_acc", "American Heart Association / American College of Cardiology",
           (r"american heart association", r"american college of cardiology", r"\baha/acc\b", r"\bacc/aha\b"),
           "not_connected"),
    Issuer("esc", "European Society of Cardiology", (r"european society of cardiology", r"\besc guidelines?\b"),
           "not_connected"),
    Issuer("acep", "American College of Emergency Physicians", (r"american college of emergency physicians",),
           "not_connected"),
    Issuer("kdigo", "Kidney Disease: Improving Global Outcomes", (r"\bkdigo\b",), "not_connected"),
    Issuer("jssc", "Japanese Society of Intensive Care Medicine / J-SSCG", (r"japanese clinical practice guidelines? "
                                                                            r"for (?:the )?management of sepsis",
                                                                            r"\bj-sscg\b",
                                                                            r"japanese society of intensive care"),
           "not_connected"),
    Issuer("awmf", "German AWMF (S3 guidelines)", (r"\bs3[- ]guideline", r"\bs3[- ]leitlinie", r"\bawmf\b"),
           "not_connected"),
    Issuer("nice", "National Institute for Health and Care Excellence (NICE)",
           (r"national institute for health and care excellence", r"\bnice guideline", r"\bnice\b(?= (?:guidance|ng\d))"),
           "blocked_licence", "NICE's syndication API needs a licence; NICE guidance is linked, not ingested."),
    Issuer("who", "World Health Organization", (r"world health organi[sz]ation", r"\bwho guideline"), "not_connected",
           "WHO IRIS is not connected; WHO guidelines are found only when PubMed / Europe PMC index them."),
    Issuer("cdc", "Centers for Disease Control and Prevention", (r"centers for disease control", r"\bcdc\b"),
           "not_connected"),
    Issuer("uspstf", "US Preventive Services Task Force", (r"preventive services task force", r"\buspstf\b"),
           "not_connected"),
)

FEED_LABELS = {
    "not_connected": "Official feed not connected — found through PubMed / Europe PMC indexing",
    "blocked_licence": "Official feed blocked (licence needed) — found through PubMed / Europe PMC indexing",
    "blocked_terms": "Official feed blocked by its terms — found through PubMed / Europe PMC indexing",
}


# A national society whose name contains a registry issuer's name is a different
# organisation ("Chinese Society of Critical Care Medicine" is not SCCM).
_NATIONAL = frozenset("""
    chinese japanese korean taiwanese thai indonesian malaysian singaporean vietnamese filipino indian pakistani
    iranian saudi arab turkish israeli egyptian nigerian african asian german french italian spanish portuguese
    dutch belgian swiss austrian polish czech hungarian romanian greek russian ukrainian danish swedish norwegian
    finnish nordic scandinavian british irish scottish canadian mexican brazilian argentine argentinian chilean
    colombian peruvian latin australian zealand
""".split())


def _named(alias: str, low: str) -> bool:
    for m in re.finditer(alias, low):
        before = low[:m.start()].split()
        if before and before[-1].strip("(\"'") in _NATIONAL:
            continue
        return True
    return False


def match_issuers(text: str) -> list[Issuer]:
    low = (text or "").lower()
    return [i for i in ISSUERS if any(_named(a, low) for a in i.aliases)]


def issuer(issuer_id: str) -> Optional[Issuer]:
    return next((i for i in ISSUERS if i.id == issuer_id), None)


def public() -> dict:
    return {
        "version": REGISTRY_VERSION, "status": REGISTRY_STATUS,
        "issuers": [{"id": i.id, "name": i.name, "official_feed": i.official_feed,
                     "feed_label": FEED_LABELS[i.official_feed], "note": i.note} for i in ISSUERS],
    }
