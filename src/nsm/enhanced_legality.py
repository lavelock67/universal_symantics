#!/usr/bin/env python3
"""
Enhanced NSM Legality Validation System.

This module provides comprehensive NSM legality validation based on
proper NSM grammar rules and semantic constraints.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

@dataclass
class NSMLegalityResult:
    """Result of NSM legality validation."""
    is_legal: bool
    score: float
    violations: List[str]
    detected_primes: List[str]
    structural_score: float
    semantic_score: float
    grammar_score: float

class EnhancedNSMLegalityValidator:
    """Enhanced NSM legality validator with proper grammar rules."""
    
    def __init__(self, exponents_path: str = "data/nsm_exponents_en_es_fr.json"):
        """Initialize the validator with NSM exponents."""
        self.exponents = self._load_exponents(exponents_path)
        self.lang_tokens = self._build_token_sets()
        self.grammar_rules = self._build_grammar_rules()
        
    def _load_exponents(self, path: str) -> Dict[str, Dict[str, List[str]]]:
        """Load NSM exponents from file."""
        p = Path(path)
        if not p.exists():
            # Fallback to basic exponents
            return {
                "primes": {
                    "I": {"en": ["i"], "es": ["yo"], "fr": ["je"]},
                    "YOU": {"en": ["you"], "es": ["tú", "usted"], "fr": ["tu", "vous"]},
                    "SOMETHING": {"en": ["something", "thing"], "es": ["algo", "cosa"], "fr": ["quelque chose", "chose"]},
                    "DO": {"en": ["do", "does", "did"], "es": ["hacer", "hace"], "fr": ["faire", "fait"]},
                    "HAPPEN": {"en": ["happen", "happens"], "es": ["suceder", "pasa"], "fr": ["arriver", "se passe"]},
                    "THINK": {"en": ["think", "thinks"], "es": ["pensar", "piensa"], "fr": ["penser", "pense"]},
                    "WANT": {"en": ["want", "wants"], "es": ["querer", "quiere"], "fr": ["vouloir", "veut"]},
                    "FEEL": {"en": ["feel", "feels"], "es": ["sentir", "siente"], "fr": ["sentir", "ressent"]},
                    "SEE": {"en": ["see", "sees"], "es": ["ver", "ve"], "fr": ["voir", "voit"]},
                    "SAY": {"en": ["say", "says"], "es": ["decir", "dice"], "fr": ["dire", "dit"]},
                    "GOOD": {"en": ["good"], "es": ["bueno", "buena"], "fr": ["bon", "bonne"]},
                    "BAD": {"en": ["bad"], "es": ["malo", "mala"], "fr": ["mauvais", "mauvaise"]},
                    "BIG": {"en": ["big", "large"], "es": ["grande"], "fr": ["grand", "grande"]},
                    "SMALL": {"en": ["small", "little"], "es": ["pequeño", "pequeña"], "fr": ["petit", "petite"]},
                    "NOT": {"en": ["not", "no"], "es": ["no"], "fr": ["ne", "pas"]},
                    "BECAUSE": {"en": ["because"], "es": ["porque"], "fr": ["parce que"]},
                    "IF": {"en": ["if"], "es": ["si"], "fr": ["si"]},
                    "CAN": {"en": ["can", "cannot"], "es": ["poder", "puede"], "fr": ["pouvoir", "peut"]},
                    "THERE_IS": {"en": ["there is", "there are"], "es": ["hay", "existe"], "fr": ["il y a", "existe"]},
                    "BE_SOMEWHERE": {"en": ["be in", "be at", "be on"], "es": ["estar en", "estar a"], "fr": ["être à", "être dans"]},
                    "PART": {"en": ["part of"], "es": ["parte de"], "fr": ["partie de"]},
                    "LIKE": {"en": ["like"], "es": ["como"], "fr": ["comme"]},
                    "SAME": {"en": ["same"], "es": ["mismo", "misma"], "fr": ["même"]},
                    "OTHER": {"en": ["other", "else"], "es": ["otro", "otra"], "fr": ["autre"]},
                    "THIS": {"en": ["this"], "es": ["este", "esta"], "fr": ["ce", "cette"]},
                    "SOME": {"en": ["some"], "es": ["algunos", "algunas"], "fr": ["quelques"]},
                    "ALL": {"en": ["all"], "es": ["todos", "todas"], "fr": ["tous", "toutes"]},
                    "MANY": {"en": ["many", "much"], "es": ["muchos", "muchas"], "fr": ["beaucoup"]},
                    "FEW": {"en": ["few", "little"], "es": ["pocos", "pocas"], "fr": ["peu"]},
                    "TIME": {"en": ["time"], "es": ["tiempo"], "fr": ["temps"]},
                    "NOW": {"en": ["now"], "es": ["ahora"], "fr": ["maintenant"]},
                    "BEFORE": {"en": ["before"], "es": ["antes"], "fr": ["avant"]},
                    "AFTER": {"en": ["after"], "es": ["después"], "fr": ["après"]},
                    "WHERE": {"en": ["where", "place"], "es": ["donde", "lugar"], "fr": ["où", "lieu"]},
                    "HERE": {"en": ["here"], "es": ["aquí"], "fr": ["ici"]},
                    "VERY": {"en": ["very"], "es": ["muy"], "fr": ["très"]},
                    "MORE": {"en": ["more"], "es": ["más"], "fr": ["plus"]},
                }
            }
        
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_token_sets(self) -> Dict[str, Set[str]]:
        """Build language-specific token sets for legality checks."""
        langs = ["en", "es", "fr"]
        token_sets = {}
        
        for lang in langs:
            tokens = set()
            # Add all NSM primes
            for prime_data in self.exponents.get("primes", {}).values():
                for token in prime_data.get(lang, []):
                    tokens.add(token.lower())
            
            # Add essential function words
            function_words = {
                "en": {"and", "or", "the", "a", "an", "to", "of", "with", "in", "on", "at", "for", "by", "from", "up", "down", "out", "off", "over", "under", "through", "between", "among", "during", "before", "after", "since", "until", "while", "when", "where", "why", "how", "what", "who", "which", "that", "this", "these", "those", "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "his", "hers", "ours", "theirs"},
                "es": {"y", "o", "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "con", "en", "sobre", "a", "para", "por", "desde", "hasta", "entre", "durante", "antes", "después", "cuando", "donde", "por qué", "cómo", "qué", "quién", "cuál", "que", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "mi", "tu", "su", "nuestro", "vuestro", "mío", "tuyo", "suyo"},
                "fr": {"et", "ou", "le", "la", "les", "un", "une", "des", "de", "du", "avec", "dans", "sur", "à", "pour", "par", "depuis", "jusqu'à", "entre", "pendant", "avant", "après", "quand", "où", "pourquoi", "comment", "quoi", "qui", "quel", "que", "ce", "cette", "ces", "mon", "ton", "son", "notre", "votre", "leur", "mien", "tien", "sien"}
            }
            tokens.update(function_words.get(lang, set()))
            
            token_sets[lang] = tokens
        
        return token_sets
    
    def _build_grammar_rules(self) -> Dict[str, List[Dict]]:
        """Build NSM grammar rules for validation."""
        return {
            "en": [
                # Basic NSM sentence patterns
                {"pattern": r"^(I|you|someone|something)\s+(think|want|feel|see|hear|say|do|happen)", "weight": 1.0},
                {"pattern": r"^(this|that|something)\s+(is|are)\s+(good|bad|big|small|same|other)", "weight": 1.0},
                {"pattern": r"^(there)\s+(is|are)\s+(something|someone)", "weight": 1.0},
                {"pattern": r"^(I|you|someone)\s+(can|cannot)\s+(do|happen)", "weight": 1.0},
                {"pattern": r"^(something|someone)\s+(is|are)\s+(in|at|on)\s+(somewhere|here|there)", "weight": 0.8},
                {"pattern": r"^(this|that)\s+(is|are)\s+(like|same|other)\s+(this|that)", "weight": 0.8},
                {"pattern": r"^(if)\s+(something)\s+(happen)\s+(then)\s+(something)\s+(happen)", "weight": 0.9},
                {"pattern": r"^(something)\s+(happen)\s+(because)\s+(something)\s+(happen)", "weight": 0.9},
                {"pattern": r"^(part)\s+(of)\s+(something)", "weight": 0.7},
                {"pattern": r"^(some|many|few|all)\s+(something)", "weight": 0.7},
                {"pattern": r"^(very)\s+(good|bad|big|small)", "weight": 0.6},
                {"pattern": r"^(more|less)\s+(than)", "weight": 0.6},
                {"pattern": r"^(before|after|when)\s+(something)\s+(happen)", "weight": 0.8},
                {"pattern": r"^(where)\s+(something)\s+(is|are)", "weight": 0.7},
                {"pattern": r"^(now)\s+(something)\s+(happen)", "weight": 0.7},
            ],
            "es": [
                # Spanish NSM patterns
                {"pattern": r"^(yo|tú|alguien|algo)\s+(piensa|quiere|siente|ve|oye|dice|hace|pasa)", "weight": 1.0},
                {"pattern": r"^(esto|eso|algo)\s+(es|son)\s+(bueno|malo|grande|pequeño|mismo|otro)", "weight": 1.0},
                {"pattern": r"^(hay)\s+(algo|alguien)", "weight": 1.0},
                {"pattern": r"^(yo|tú|alguien)\s+(puede|no puede)\s+(hacer|pasar)", "weight": 1.0},
                {"pattern": r"^(algo|alguien)\s+(está|están)\s+(en|a|sobre)\s+(algún lugar|aquí|allí)", "weight": 0.8},
                {"pattern": r"^(esto|eso)\s+(es|son)\s+(como|mismo|otro)\s+(esto|eso)", "weight": 0.8},
                {"pattern": r"^(si)\s+(algo)\s+(pasa)\s+(entonces)\s+(algo)\s+(pasa)", "weight": 0.9},
                {"pattern": r"^(algo)\s+(pasa)\s+(porque)\s+(algo)\s+(pasa)", "weight": 0.9},
                {"pattern": r"^(parte)\s+(de)\s+(algo)", "weight": 0.7},
                {"pattern": r"^(algunos|muchos|pocos|todos)\s+(algo)", "weight": 0.7},
                {"pattern": r"^(muy)\s+(bueno|malo|grande|pequeño)", "weight": 0.6},
                {"pattern": r"^(más|menos)\s+(que)", "weight": 0.6},
                {"pattern": r"^(antes|después|cuando)\s+(algo)\s+(pasa)", "weight": 0.8},
                {"pattern": r"^(dónde)\s+(algo)\s+(está|están)", "weight": 0.7},
                {"pattern": r"^(ahora)\s+(algo)\s+(pasa)", "weight": 0.7},
            ],
            "fr": [
                # French NSM patterns
                {"pattern": r"^(je|tu|quelqu'un|quelque chose)\s+(pense|veut|ressent|voit|entend|dit|fait|arrive)", "weight": 1.0},
                {"pattern": r"^(ceci|cela|quelque chose)\s+(est|sont)\s+(bon|mauvais|grand|petit|même|autre)", "weight": 1.0},
                {"pattern": r"^(il y a)\s+(quelque chose|quelqu'un)", "weight": 1.0},
                {"pattern": r"^(je|tu|quelqu'un)\s+(peut|ne peut pas)\s+(faire|arriver)", "weight": 1.0},
                {"pattern": r"^(quelque chose|quelqu'un)\s+(est|sont)\s+(à|dans|sur)\s+(quelque part|ici|là)", "weight": 0.8},
                {"pattern": r"^(ceci|cela)\s+(est|sont)\s+(comme|même|autre)\s+(ceci|cela)", "weight": 0.8},
                {"pattern": r"^(si)\s+(quelque chose)\s+(arrive)\s+(alors)\s+(quelque chose)\s+(arrive)", "weight": 0.9},
                {"pattern": r"^(quelque chose)\s+(arrive)\s+(parce que)\s+(quelque chose)\s+(arrive)", "weight": 0.9},
                {"pattern": r"^(partie)\s+(de)\s+(quelque chose)", "weight": 0.7},
                {"pattern": r"^(quelques|beaucoup|peu|tous)\s+(quelque chose)", "weight": 0.7},
                {"pattern": r"^(très)\s+(bon|mauvais|grand|petit)", "weight": 0.6},
                {"pattern": r"^(plus|moins)\s+(que)", "weight": 0.6},
                {"pattern": r"^(avant|après|quand)\s+(quelque chose)\s+(arrive)", "weight": 0.8},
                {"pattern": r"^(où)\s+(quelque chose)\s+(est|sont)", "weight": 0.7},
                {"pattern": r"^(maintenant)\s+(quelque chose)\s+(arrive)", "weight": 0.7},
            ]
        }
    
    def validate_legality(self, text: str, lang: str) -> NSMLegalityResult:
        """Comprehensive NSM legality validation."""
        if not text or not text.strip():
            return NSMLegalityResult(False, 0.0, ["Empty text"], [], 0.0, 0.0, 0.0)
        
        # Clean and normalize text
        clean_text = self._normalize_text(text)
        words = clean_text.lower().split()
        
        # Initialize result
        violations = []
        detected_primes = []
        
        # 1. Token coverage check
        token_score = self._check_token_coverage(words, lang)
        
        # 2. Grammar structure check
        grammar_score, grammar_violations = self._check_grammar_structure(clean_text, lang)
        violations.extend(grammar_violations)
        
        # 3. Semantic coherence check
        semantic_score, semantic_violations = self._check_semantic_coherence(words, lang)
        violations.extend(semantic_violations)
        
        # 4. Prime detection
        detected_primes = self._detect_primes(words, lang)
        
        # 5. Structural validation
        structural_score, structural_violations = self._check_structural_validity(clean_text, lang)
        violations.extend(structural_violations)
        
        # Calculate overall score
        overall_score = (0.3 * token_score + 
                        0.3 * grammar_score + 
                        0.2 * semantic_score + 
                        0.2 * structural_score)
        
        # Determine if legal (threshold can be adjusted)
        is_legal = overall_score >= 0.6
        
        return NSMLegalityResult(
            is_legal=is_legal,
            score=overall_score,
            violations=violations,
            detected_primes=detected_primes,
            structural_score=structural_score,
            semantic_score=semantic_score,
            grammar_score=grammar_score
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for analysis."""
        # Remove extra whitespace and punctuation
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s]', ' ', text)
        return text
    
    def _check_token_coverage(self, words: List[str], lang: str) -> float:
        """Check what percentage of words are NSM tokens."""
        if not words:
            return 0.0
        
        allowed_tokens = self.lang_tokens.get(lang, set())
        nsm_words = sum(1 for word in words if word in allowed_tokens)
        return nsm_words / len(words)
    
    def _check_grammar_structure(self, text: str, lang: str) -> Tuple[float, List[str]]:
        """Check if text follows NSM grammar patterns."""
        violations = []
        max_score = 0.0
        
        rules = self.grammar_rules.get(lang, [])
        for rule in rules:
            pattern = rule["pattern"]
            weight = rule["weight"]
            
            if re.search(pattern, text, re.IGNORECASE):
                max_score = max(max_score, weight)
            else:
                # Check for partial matches
                partial_match = self._check_partial_pattern_match(text, pattern)
                if partial_match > 0:
                    max_score = max(max_score, partial_match * weight)
        
        if max_score < 0.3:
            violations.append("Text does not follow NSM grammar patterns")
        
        return max_score, violations
    
    def _check_partial_pattern_match(self, text: str, pattern: str) -> float:
        """Check for partial pattern matches."""
        # Extract key components from pattern
        components = re.findall(r'\([^)]+\)', pattern)
        if not components:
            return 0.0
        
        # Count how many components are present
        present_components = 0
        for component in components:
            # Remove parentheses and split alternatives
            component_text = component[1:-1]
            alternatives = component_text.split('|')
            if any(alt.lower() in text.lower() for alt in alternatives):
                present_components += 1
        
        return present_components / len(components)
    
    def _check_semantic_coherence(self, words: List[str], lang: str) -> Tuple[float, List[str]]:
        """Check semantic coherence of the text."""
        violations = []
        score = 1.0
        
        # Check for semantic contradictions
        contradictions = self._check_contradictions(words, lang)
        if contradictions:
            violations.extend(contradictions)
            score -= 0.3
        
        # Check for logical consistency
        logical_issues = self._check_logical_consistency(words, lang)
        if logical_issues:
            violations.extend(logical_issues)
            score -= 0.2
        
        return max(0.0, score), violations
    
    def _check_contradictions(self, words: List[str], lang: str) -> List[str]:
        """Check for semantic contradictions."""
        violations = []
        
        # Check for contradictory pairs
        contradictions = {
            "en": [("good", "bad"), ("big", "small"), ("same", "other"), ("all", "none")],
            "es": [("bueno", "malo"), ("grande", "pequeño"), ("mismo", "otro"), ("todos", "ninguno")],
            "fr": [("bon", "mauvais"), ("grand", "petit"), ("même", "autre"), ("tous", "aucun")]
        }
        
        lang_contradictions = contradictions.get(lang, [])
        for word1, word2 in lang_contradictions:
            if word1 in words and word2 in words:
                violations.append(f"Contradictory terms: {word1} and {word2}")
        
        return violations
    
    def _check_logical_consistency(self, words: List[str], lang: str) -> List[str]:
        """Check logical consistency."""
        violations = []
        
        # Check for impossible combinations
        impossible_combinations = {
            "en": [("never", "always"), ("impossible", "possible")],
            "es": [("nunca", "siempre"), ("imposible", "posible")],
            "fr": [("jamais", "toujours"), ("impossible", "possible")]
        }
        
        lang_impossible = impossible_combinations.get(lang, [])
        for word1, word2 in lang_impossible:
            if word1 in words and word2 in words:
                violations.append(f"Impossible combination: {word1} and {word2}")
        
        return violations
    
    def _detect_primes(self, words: List[str], lang: str) -> List[str]:
        """Detect NSM primes in the text."""
        detected = []
        
        for word in words:
            for prime, data in self.exponents.get("primes", {}).items():
                if word in data.get(lang, []):
                    detected.append(prime)
        
        return list(set(detected))  # Remove duplicates
    
    def _check_structural_validity(self, text: str, lang: str) -> Tuple[float, List[str]]:
        """Check structural validity of the text."""
        violations = []
        score = 1.0
        
        # Check sentence length (NSM sentences should be relatively short)
        word_count = len(text.split())
        if word_count > 15:
            violations.append("Sentence too long for NSM")
            score -= 0.2
        elif word_count < 2:
            violations.append("Sentence too short")
            score -= 0.3
        
        # Check for complex structures that aren't NSM-like
        complex_patterns = [
            r'\b(however|nevertheless|furthermore|moreover|consequently)\b',
            r'\b(although|despite|in spite of|whereas|while)\b',
            r'\b(consequently|therefore|thus|hence|as a result)\b'
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append("Contains complex discourse markers")
                score -= 0.1
        
        return max(0.0, score), violations
    
    def legality_score(self, text: str, lang: str) -> float:
        """Get legality score (backward compatibility)."""
        result = self.validate_legality(text, lang)
        return result.score
    
    def is_legal(self, text: str, lang: str) -> bool:
        """Check if text is NSM legal (backward compatibility)."""
        result = self.validate_legality(text, lang)
        return result.is_legal

