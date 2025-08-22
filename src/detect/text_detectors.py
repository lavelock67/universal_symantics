"""Lightweight text detectors for mapping sentences to primitive names.

These are conservative, regex-based detectors intended to provide non-theatrical
signals for early MDL/Δ experiments. They should be replaced or augmented with
SRL/AMR/UD-based detectors for research-grade results.
"""

import re
from typing import List


_PATTERNS = [
    ("IsA", [
        re.compile(r"\b(is|are|was|were)\s+(an?|the)\s+\w+\b", re.IGNORECASE),
    ]),
    ("PartOf", [
        re.compile(r"\bpart\s+of\b", re.IGNORECASE),
        re.compile(r"\bcontains?\b", re.IGNORECASE),
        re.compile(r"\bhas\s+(an?|the)?\s*part\b", re.IGNORECASE),
    ]),
    ("Before", [
        re.compile(r"\bbefore\b", re.IGNORECASE),
    ]),
    ("After", [
        re.compile(r"\bafter\b", re.IGNORECASE),
    ]),
    ("AtLocation", [
        re.compile(r"\b(at|in|on)\b\s+(the|a|an)\b", re.IGNORECASE),
    ]),
    # NSM-style additions
    ("In", [re.compile(r"\bin\b\s+(the|a|an)?\s*\w+", re.IGNORECASE)]),
    ("On", [re.compile(r"\bon\b\s+(the|a|an)?\s*\w+", re.IGNORECASE)]),
    ("At", [re.compile(r"\bat\b\s+(the|a|an)?\s*\w+", re.IGNORECASE)]),
    ("Not", [re.compile(r"\bnot\b", re.IGNORECASE), re.compile(r"\bno\b", re.IGNORECASE)]),
    ("MoreThan", [re.compile(r"\bmore\s+than\b", re.IGNORECASE)]),
    ("LessThan", [re.compile(r"\bless\s+than\b", re.IGNORECASE)]),
    ("EqualTo", [re.compile(r"\bequal\s+to\b", re.IGNORECASE), re.compile(r"\bas\s+.*\s+as\b", re.IGNORECASE)]),
    ("All", [re.compile(r"\ball\b", re.IGNORECASE)]),
    ("Some", [re.compile(r"\bsome\b", re.IGNORECASE)]),
    ("Many", [re.compile(r"\bmany\b", re.IGNORECASE)]),
    ("Few", [re.compile(r"\bfew\b", re.IGNORECASE)]),
    ("Most", [re.compile(r"\bmost\b", re.IGNORECASE)]),
    ("None", [re.compile(r"\bno(ne)?\b", re.IGNORECASE)]),
    ("Can", [re.compile(r"\bcan\b", re.IGNORECASE), re.compile(r"\bcould\b", re.IGNORECASE)]),
    ("Must", [re.compile(r"\bmust\b", re.IGNORECASE), re.compile(r"\bhave to\b", re.IGNORECASE)]),
    ("Want", [re.compile(r"\bwant(s|ed)?\b", re.IGNORECASE)]),
    ("Know", [re.compile(r"\bknow(s|n|ing)?\b", re.IGNORECASE)]),
    ("Say", [re.compile(r"\bsay(s|ing|id)?\b", re.IGNORECASE)]),
    ("Because", [re.compile(r"\bbecause\b", re.IGNORECASE)]),
    ("Therefore", [re.compile(r"\btherefore\b|\bso\b", re.IGNORECASE)]),
    ("If", [re.compile(r"\bif\b", re.IGNORECASE)]),
    ("Unless", [re.compile(r"\bunless\b", re.IGNORECASE)]),
    ("InOrderTo", [re.compile(r"\bin\s+order\s+to\b|\bto\s+\w+", re.IGNORECASE)]),
]


def detect_primitives_in_text(text: str) -> List[str]:
    detected: List[str] = []
    for name, regexes in _PATTERNS:
        for rgx in regexes:
            if rgx.search(text):
                detected.append(name)
                break
    return detected


