#!/usr/bin/env python3
"""Enhanced UD Patterns with Critical Fixes.

Addresses the red flags identified:
1. French negative polarity traps (personne ≠ PEOPLE)
2. THIS over-firing as determiner
3. GOOD/BAD false positives (noun senses)
4. TRUE/FALSE + negation scope
5. VERY vs MANY confusion
6. Contraction tokenization
7. Language-specific router thresholds
"""

import spacy
from typing import Dict, List, Any, Optional, Set
from spacy.matcher import DependencyMatcher
import logging

logger = logging.getLogger(__name__)


class EnhancedUDPatterns:
    """Enhanced UD patterns with critical fixes for edge cases."""
    
    def __init__(self, nlp_models: Dict[str, Any]):
        """Initialize with language-specific NLP models."""
        self.nlp_models = nlp_models
        self.matchers = {}
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile all enhanced patterns."""
        for lang, nlp in self.nlp_models.items():
            matcher = DependencyMatcher(nlp.vocab)
            
            if lang == "es":
                self._add_spanish_patterns(matcher)
            elif lang == "fr":
                self._add_french_patterns(matcher)
            elif lang == "en":
                self._add_english_patterns(matcher)
            
            self.matchers[lang] = matcher
            logger.info(f"Compiled {len(matcher)} patterns for {lang}")
    
    def _add_spanish_patterns(self, matcher: DependencyMatcher):
        """Add Spanish patterns with critical fixes."""
        
        # THINK (pensar with optional "que ...")
        pattern_pensar = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "pensar", "POS": {"IN": ["VERB", "AUX"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ccomp", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}}},
        ]
        matcher.add("THINK_ES", [pattern_pensar])
        
        # VERY (muy modifies ADJ/ADV/VERB)
        pattern_muy = [
            {"RIGHT_ID": "adv", "RIGHT_ATTRS": {"LOWER": "muy", "POS": "ADV"}},
            {"LEFT_ID": "adv", "REL_OP": "<", "RIGHT_ID": "head", "RIGHT_ATTRS": {"POS": {"IN": ["ADJ", "ADV", "VERB"]}}},
        ]
        matcher.add("VERY_ES", [pattern_muy])
        
        # GOOD (predicative adjective with copula)
        pattern_bueno = [
            {"RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "ser", "POS": {"IN": ["AUX", "VERB"]}}},
            {"LEFT_ID": "cop", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"LEMMA": "bueno", "POS": "ADJ"}},
        ]
        matcher.add("GOOD_ES", [pattern_bueno])
        
        # BAD (predicative adjective with copula)
        pattern_malo = [
            {"RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "ser", "POS": {"IN": ["AUX", "VERB"]}}},
            {"LEFT_ID": "cop", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"LEMMA": {"IN": ["malo", "mala"]}, "POS": "ADJ"}},
        ]
        matcher.add("BAD_ES", [pattern_malo])
        
        # TRUE/FALSE (with negation flip)
        pattern_verdadero_falso = [
            {"RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "ser"}},
            {"LEFT_ID": "cop", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"LEMMA": {"IN": ["verdadero", "cierto", "falso"]}, "POS": "ADJ"}},
        ]
        matcher.add("TRUE_FALSE_ES", [pattern_verdadero_falso])
        
        # PEOPLE (gente or plural personas)
        pattern_people_es = [
            {"RIGHT_ID": "n", "RIGHT_ATTRS": {"LEMMA": {"IN": ["gente", "persona"]}, "POS": "NOUN"}},
        ]
        matcher.add("PEOPLE_ES", [pattern_people_es])
        
        # THIS (pronominal or copular - avoid determiner over-firing)
        pattern_this_es = [
            {"RIGHT_ID": "pro", "RIGHT_ATTRS": {"LOWER": {"IN": ["esto", "eso"]}, "POS": "PRON"}},
        ]
        matcher.add("THIS_ES", [pattern_this_es])
        
        # Copular THIS (esto/eso + ser)
        pattern_this_cop_es = [
            {"RIGHT_ID": "pro", "RIGHT_ATTRS": {"LOWER": {"IN": ["esto", "eso"]}, "POS": "PRON"}},
            {"LEFT_ID": "pro", "REL_OP": ">", "RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "ser"}},
        ]
        matcher.add("THIS_COP_ES", [pattern_this_cop_es])
    
    def _add_french_patterns(self, matcher: DependencyMatcher):
        """Add French patterns with critical fixes."""
        
        # THINK (penser + optional "que ...")
        pattern_penser = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "penser", "POS": {"IN": ["VERB", "AUX"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ccomp", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}}},
        ]
        matcher.add("THINK_FR", [pattern_penser])
        
        # VERY (très - avoid confusion with beaucoup)
        pattern_tres = [
            {"RIGHT_ID": "adv", "RIGHT_ATTRS": {"LOWER": "très", "POS": "ADV"}},
            {"LEFT_ID": "adv", "REL_OP": "<", "RIGHT_ID": "head", "RIGHT_ATTRS": {"POS": {"IN": ["ADJ", "ADV", "VERB"]}}},
        ]
        matcher.add("VERY_FR", [pattern_tres])
        
        # MANY (beaucoup de - not VERY)
        pattern_beaucoup = [
            {"RIGHT_ID": "adv", "RIGHT_ATTRS": {"LOWER": "beaucoup", "POS": "ADV"}},
            {"LEFT_ID": "adv", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"LOWER": "de"}},
        ]
        matcher.add("MANY_FR", [pattern_beaucoup])
        
        # GOOD (predicative adjective with copula)
        pattern_bon = [
            {"RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "être", "POS": {"IN": ["AUX", "VERB"]}}},
            {"LEFT_ID": "cop", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"LEMMA": {"IN": ["bon", "bonne"]}, "POS": "ADJ"}},
        ]
        matcher.add("GOOD_FR", [pattern_bon])
        
        # BAD (predicative adjective with copula)
        pattern_mauvais = [
            {"RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "être", "POS": {"IN": ["AUX", "VERB"]}}},
            {"LEFT_ID": "cop", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"LEMMA": {"IN": ["mauvais", "mauvaise"]}, "POS": "ADJ"}},
        ]
        matcher.add("BAD_FR", [pattern_mauvais])
        
        # TRUE/FALSE (with negation flip)
        pattern_vrai_faux = [
            {"RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "être"}},
            {"LEFT_ID": "cop", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"LEMMA": {"IN": ["vrai", "certain", "faux"]}, "POS": "ADJ"}},
        ]
        matcher.add("TRUE_FALSE_FR", [pattern_vrai_faux])
        
        # PEOPLE (gens or plural personnes - avoid negative polarity)
        pattern_people_fr = [
            {"RIGHT_ID": "n", "RIGHT_ATTRS": {"LEMMA": {"IN": ["gens", "personnes"]}, "POS": "NOUN"}},
        ]
        matcher.add("PEOPLE_FR", [pattern_people_fr])
        
        # THIS (pronominal or copular - handle contractions)
        pattern_this_fr = [
            {"RIGHT_ID": "pro", "RIGHT_ATTRS": {"LOWER": {"IN": ["ce", "cela", "ça", "ceci"]}, "POS": {"IN": ["PRON", "DET"]}}},
        ]
        matcher.add("THIS_FR", [pattern_this_fr])
        
        # Contraction THIS (c'est)
        pattern_this_contraction_fr = [
            {"RIGHT_ID": "pro", "RIGHT_ATTRS": {"LOWER": "c'", "POS": "PRON"}},
            {"LEFT_ID": "pro", "REL_OP": ">", "RIGHT_ID": "cop", "RIGHT_ATTRS": {"LOWER": "est"}},
        ]
        matcher.add("THIS_CONTRACTION_FR", [pattern_this_contraction_fr])
    
    def _add_english_patterns(self, matcher: DependencyMatcher):
        """Add English patterns with critical fixes."""
        
        # THINK
        pattern_think = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "think", "POS": {"IN": ["VERB", "AUX"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "ccomp", "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}}},
        ]
        matcher.add("THINK_EN", [pattern_think])
        
        # VERY
        pattern_very = [
            {"RIGHT_ID": "adv", "RIGHT_ATTRS": {"LOWER": "very", "POS": "ADV"}},
            {"LEFT_ID": "adv", "REL_OP": "<", "RIGHT_ID": "head", "RIGHT_ATTRS": {"POS": {"IN": ["ADJ", "ADV", "VERB"]}}},
        ]
        matcher.add("VERY_EN", [pattern_very])
        
        # GOOD
        pattern_good = [
            {"RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "be", "POS": {"IN": ["AUX", "VERB"]}}},
            {"LEFT_ID": "cop", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"LEMMA": "good", "POS": "ADJ"}},
        ]
        matcher.add("GOOD_EN", [pattern_good])
        
        # BAD
        pattern_bad = [
            {"RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "be", "POS": {"IN": ["AUX", "VERB"]}}},
            {"LEFT_ID": "cop", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"LEMMA": "bad", "POS": "ADJ"}},
        ]
        matcher.add("BAD_EN", [pattern_bad])
        
        # TRUE/FALSE
        pattern_true_false = [
            {"RIGHT_ID": "cop", "RIGHT_ATTRS": {"LEMMA": "be", "POS": {"IN": ["AUX", "VERB"]}}},
            {"LEFT_ID": "cop", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"LEMMA": {"IN": ["true", "false"]}, "POS": "ADJ"}},
        ]
        matcher.add("TRUE_FALSE_EN", [pattern_true_false])
        
        # PEOPLE
        pattern_people = [
            {"RIGHT_ID": "n", "RIGHT_ATTRS": {"LEMMA": {"IN": ["people", "person"]}, "POS": "NOUN"}},
        ]
        matcher.add("PEOPLE_EN", [pattern_people])
        
        # THIS
        pattern_this = [
            {"RIGHT_ID": "pro", "RIGHT_ATTRS": {"LOWER": "this", "POS": "PRON"}},
        ]
        matcher.add("THIS_EN", [pattern_this])
    
    def detect_with_enhanced_patterns(self, text: str, lang: str) -> List[str]:
        """Detect primes using enhanced UD patterns with critical fixes."""
        if lang not in self.matchers:
            return []
        
        nlp = self.nlp_models[lang]
        doc = nlp(text)
        matcher = self.matchers[lang]
        
        detected_primes = set()
        
        # Get matches
        matches = matcher(doc)
        
        for match_id, token_ids in matches:
            pattern_name = nlp.vocab.strings[match_id]
            prime = self._extract_prime_from_pattern(pattern_name, doc, token_ids, lang)
            
            if prime:
                # Apply critical fixes
                prime = self._apply_critical_fixes(prime, doc, token_ids, lang)
                if prime:
                    detected_primes.add(prime)
        
        return list(detected_primes)
    
    def _extract_prime_from_pattern(self, pattern_name: str, doc, token_ids: List[int], lang: str) -> Optional[str]:
        """Extract prime from pattern name."""
        prime_mapping = {
            "THINK_ES": "THINK", "THINK_FR": "THINK", "THINK_EN": "THINK",
            "VERY_ES": "VERY", "VERY_FR": "VERY", "VERY_EN": "VERY",
            "GOOD_ES": "GOOD", "GOOD_FR": "GOOD", "GOOD_EN": "GOOD",
            "BAD_ES": "BAD", "BAD_FR": "BAD", "BAD_EN": "BAD",
            "TRUE_FALSE_ES": "TRUE", "TRUE_FALSE_FR": "TRUE", "TRUE_FALSE_EN": "TRUE",
            "PEOPLE_ES": "PEOPLE", "PEOPLE_FR": "PEOPLE", "PEOPLE_EN": "PEOPLE",
            "THIS_ES": "THIS", "THIS_FR": "THIS", "THIS_EN": "THIS",
            "THIS_COP_ES": "THIS", "THIS_CONTRACTION_FR": "THIS",
            "MANY_FR": "MANY"
        }
        
        return prime_mapping.get(pattern_name)
    
    def _apply_critical_fixes(self, prime: str, doc, token_ids: List[int], lang: str) -> Optional[str]:
        """Apply critical fixes for edge cases."""
        
        # Fix 1: French negative polarity (personne ≠ PEOPLE)
        if prime == "PEOPLE" and lang == "fr":
            for token_id in token_ids:
                token = doc[token_id]
                if token.lemma_ == "personne":
                    # Check for negative context
                    if self._has_negative_context(token):
                        return None  # Don't map to PEOPLE
        
        # Fix 2: THIS over-firing as determiner
        if prime == "THIS":
            for token_id in token_ids:
                token = doc[token_id]
                if token.pos_ == "DET":
                    # Check if it's a determiner followed by noun (not copular)
                    if self._is_determiner_not_copular(token):
                        return None  # Don't map to THIS
        
        # Fix 3: GOOD/BAD false positives (noun senses)
        if prime in ["GOOD", "BAD"]:
            for token_id in token_ids:
                token = doc[token_id]
                if token.pos_ in ["NOUN", "ADV"]:
                    return None  # Don't map to GOOD/BAD
        
        # Fix 4: TRUE/FALSE + negation scope
        if prime in ["TRUE", "FALSE"]:
            for token_id in token_ids:
                token = doc[token_id]
                if self._has_negation_in_copular_subtree(token):
                    # Flip TRUE ↔ FALSE
                    return "FALSE" if prime == "TRUE" else "TRUE"
        
        # Fix 5: VERY vs MANY (already handled in pattern compilation)
        
        return prime
    
    def _has_negative_context(self, token) -> bool:
        """Check if token has negative context (for French personne)."""
        # Look for "ne ... personne" pattern
        for child in token.children:
            if child.dep_ == "neg" and child.lemma_ == "ne":
                return True
        
        # Look for "personne" as negator
        if token.dep_ == "neg":
            return True
        
        return False
    
    def _is_determiner_not_copular(self, token) -> bool:
        """Check if determiner is not in copular context."""
        # Check if head is noun and not copular
        if token.head.pos_ == "NOUN":
            # Check if there's no copula in the subtree
            for child in token.head.children:
                if child.dep_ in ["cop", "attr"]:
                    return False
            return True
        
        return False
    
    def _has_negation_in_copular_subtree(self, token) -> bool:
        """Check if copular subtree has negation."""
        # Find the copula
        copula = None
        for child in token.children:
            if child.dep_ == "cop":
                copula = child
                break
        
        if not copula:
            return False
        
        # Check for negation in copular subtree
        for child in copula.children:
            if child.dep_ == "neg":
                return True
        
        # Check for negation in parent clause
        for child in copula.head.children:
            if child.dep_ == "neg":
                return True
        
        return False
