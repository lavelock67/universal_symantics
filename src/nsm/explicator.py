#!/usr/bin/env python3
"""
Minimal NSM Explicator: prime-only templates and legality scoring.

This provides a tiny scaffold to move beyond relation hits toward
NSM-style explications and objective legality metrics.
"""
from __future__ import annotations

from typing import Dict, List, Set
import json
from pathlib import Path


class NSMExplicator:
    def __init__(self, exponents_path: str = "data/nsm_exponents_en_es_fr.json"):
        self.exponents: Dict[str, Dict[str, List[str]]] = {}
        self.lang_tokens: Dict[str, Set[str]] = {}
        self._load_exponents(exponents_path)

    def _load_exponents(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            # Minimal fallback if file missing
            self.exponents = {
                "primes": {
                    "BE": {"en": ["be", "is", "are"], "es": ["ser", "estar", "es"], "fr": ["être", "est", "sont"]},
                    "NOT": {"en": ["not", "no"], "es": ["no"], "fr": ["ne", "pas"]},
                    "BECAUSE": {"en": ["because"], "es": ["porque"], "fr": ["parce", "car"]},
                    "LIKE": {"en": ["like"], "es": ["como"], "fr": ["comme"]},
                    "THERE_IS": {"en": ["there", "is"], "es": ["hay", "existe"], "fr": ["il", "y", "a", "existe"]},
                    "IN": {"en": ["in", "on", "at"], "es": ["en", "a", "sobre"], "fr": ["à", "dans", "sur", "en"]},
                    "PART": {"en": ["part", "of"], "es": ["parte", "de"], "fr": ["partie", "de"]},
                    "DO": {"en": ["do", "does", "did"], "es": ["hacer", "hace"], "fr": ["faire", "fait"]},
                    "HAPPEN": {"en": ["happen", "happens"], "es": ["sucede", "pasa"], "fr": ["arrive", "se", "passe"]},
                }
            }
        else:
            with open(p, "r", encoding="utf-8") as f:
                self.exponents = json.load(f)

        # Flatten per language token sets for legality checks
        langs = ["en", "es", "fr"]
        for lang in langs:
            tokens: Set[str] = set()
            for _, data in self.exponents.get("primes", {}).items():
                for t in data.get(lang, []):
                    tokens.add(t.lower())
            # allow a few glue words/puncts
            tokens.update({"and", "y", "et", "de", "la", "le", "el", "un", "une", "a", "to"})
            self.lang_tokens[lang] = tokens

    def legality_score(self, text: str, lang: str) -> float:
        words = [w.strip(".,;:!?'") for w in text.lower().split()]
        if not words:
            return 0.0
        allowed = self.lang_tokens.get(lang, set())
        token_cov = sum(1 for w in words if w in allowed) / len(words)
        # Micro-grammar structural bonus: look for simple cues expected per primitive families
        structural_ok = 0.0
        txt = " ".join(words)
        # presence of a be/estar/être token
        be_ok = any(t in txt for t in self.exponents["primes"].get("BE", {}).get(lang, [])) or any(t in txt for t in self.exponents["primes"].get("BE_SOMEWHERE", {}).get(lang, []))
        # location cue
        loc_ok = any(t in txt for t in self.exponents["primes"].get("IN", {}).get(lang, []))
        # negation cue
        neg_ok = any(t in txt for t in self.exponents["primes"].get("NOT", {}).get(lang, []))
        # causal cue
        bec_ok = any(t in txt for t in self.exponents["primes"].get("BECAUSE", {}).get(lang, []))
        # existence cue
        ex_ok = any(t in txt for t in self.exponents["primes"].get("THERE_IS", {}).get(lang, []))
        simple_hits = sum([be_ok, loc_ok, neg_ok, bec_ok, ex_ok])
        structural_ok = 1.0 if simple_hits >= 1 else 0.0
        # Blend token coverage with structural check
        return 0.7 * token_cov + 0.3 * structural_ok

    def validate_legality(self, text: str, lang: str) -> bool:
        """Validate if text follows NSM legality rules.
        
        Args:
            text: Text to validate
            lang: Language code (en, es, fr)
            
        Returns:
            True if text is NSM-legal, False otherwise
        """
        legality_score = self.legality_score(text, lang)
        # Consider text legal if score is above threshold
        return legality_score >= 0.5

    def detect_primes(self, text: str, lang: str) -> List[str]:
        """Detect NSM primes in text.
        
        Args:
            text: Text to analyze
            lang: Language code (en, es, fr)
            
        Returns:
            List of detected NSM primes
        """
        words = [w.strip(".,;:!?'") for w in text.lower().split()]
        detected_primes = []
        
        # Check for NSM primes in the text
        for word in words:
            for prime, data in self.exponents.get("primes", {}).items():
                if word in data.get(lang, []):
                    detected_primes.append(prime)
        
        return detected_primes

    def template_for_primitive(self, primitive: str, lang: str = "en") -> str:
        # Very small prime-like templates
        if primitive == "AtLocation":
            return {"en": "something is in a place", "es": "algo está en un lugar", "fr": "quelque chose est dans un lieu"}[lang]
        if primitive == "PartOf":
            return {"en": "one part is part of another thing", "es": "una parte es parte de otra cosa", "fr": "une partie est partie d'une autre chose"}[lang]
        if primitive == "Causes":
            return {"en": "something happens because something else happens", "es": "algo pasa porque otra cosa pasa", "fr": "quelque chose arrive parce qu'une autre chose arrive"}[lang]
        if primitive == "Not":
            return {"en": "someone does not do something", "es": "alguien no hace algo", "fr": "quelqu'un ne fait pas quelque chose"}[lang]
        if primitive == "UsedFor":
            return {"en": "someone can do something with this thing", "es": "alguien puede hacer algo con esta cosa", "fr": "quelqu'un peut faire quelque chose avec cette chose"}[lang]
        if primitive == "SimilarTo":
            return {"en": "this thing is like another thing", "es": "esta cosa es como otra cosa", "fr": "cette chose est comme une autre chose"}[lang]
        if primitive == "DifferentFrom":
            return {"en": "this thing is not like another thing", "es": "esta cosa no es como otra cosa", "fr": "cette chose n'est pas comme une autre chose"}[lang]
        if primitive == "HasProperty":
            return {"en": "this thing is like this", "es": "esta cosa es así", "fr": "cette chose est comme cela"}[lang]
        if primitive == "Exist":
            return {"en": "there is something", "es": "hay algo", "fr": "il y a quelque chose"}[lang]
        # Default
        return {"en": "something is like this", "es": "algo es así", "fr": "quelque chose est comme cela"}[lang]

# TODO: Complete NSM translation system integration with UMR
# TODO: Add substitutability and cross-translatability NSM metrics
# TODO: Implement NSM legality micro-grammar for prime-only clauses
# TODO: Refine NSM substitutability using multilingual NLI or thresholds
# TODO: Implement NSM cross-translatability structural check across EN/ES/FR
# TODO: Implement NSM-based translation via explications + exponent surfacing
# TODO: Add NLI-based substitutability (multilingual XNLI) for explications
# TODO: Expand exponent tables to all 65 NSM primes (EN/ES/FR)
# TODO: Tighten NSM legality micro-grammar; curated molecule registry


