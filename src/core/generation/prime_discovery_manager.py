#!/usr/bin/env python3
"""
Prime Discovery Manager

This module automatically manages the discovery of new NSM primes through UD analysis
and ensures they are consistently added to all supported languages.
"""

import json
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

from ..domain.models import Language, NSMPrime
from .language_expansion import LanguageExpansion

logger = logging.getLogger(__name__)

@dataclass
class PrimeDiscovery:
    """Represents a newly discovered prime."""
    prime_name: str
    source_corpus: str
    discovery_method: str  # "UD", "MWE", "SEMANTIC", "MANUAL"
    confidence_score: float
    semantic_category: str
    discovered_at: str
    evidence: List[str]  # Example sentences where it was found
    cross_lingual_validation: Dict[str, bool]  # Language -> validation result

@dataclass
class PrimeMapping:
    """Represents a prime mapping for a specific language."""
    prime_name: str
    language: Language
    word_form: str
    confidence: float
    source: str  # "manual", "auto", "validated"

class PrimeDiscoveryManager:
    """Manages automatic discovery and integration of new NSM primes."""
    
    def __init__(self, language_expansion: LanguageExpansion):
        """Initialize the prime discovery manager."""
        self.language_expansion = language_expansion
        self.discoveries_file = Path("data/prime_discoveries.json")
        self.mappings_file = Path("data/prime_mappings.json")
        self.backup_dir = Path("data/backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load existing discoveries and mappings
        self.discoveries = self._load_discoveries()
        self.mappings = self._load_mappings()
        
        # Track all supported languages
        self.supported_languages = list(Language)
        
        logger.info(f"PrimeDiscoveryManager initialized with {len(self.discoveries)} discoveries")
    
    def _load_discoveries(self) -> List[PrimeDiscovery]:
        """Load existing prime discoveries from file."""
        if self.discoveries_file.exists():
            try:
                with open(self.discoveries_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [PrimeDiscovery(**discovery) for discovery in data]
            except Exception as e:
                logger.warning(f"Failed to load discoveries: {e}")
        return []
    
    def _load_mappings(self) -> List[PrimeMapping]:
        """Load existing prime mappings from file."""
        if self.mappings_file.exists():
            try:
                with open(self.mappings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [PrimeMapping(**mapping) for mapping in data]
            except Exception as e:
                logger.warning(f"Failed to load mappings: {e}")
        return []
    
    def _save_discoveries(self):
        """Save discoveries to file."""
        try:
            with open(self.discoveries_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(d) for d in self.discoveries], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save discoveries: {e}")
    
    def _save_mappings(self):
        """Save mappings to file."""
        try:
            with open(self.mappings_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(m) for m in self.mappings], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save mappings: {e}")
    
    def discover_new_primes(self, corpus_data: Dict[str, List[str]], 
                          discovery_method: str = "UD") -> List[PrimeDiscovery]:
        """Discover new primes from corpus data."""
        logger.info(f"Starting prime discovery using {discovery_method} method")
        
        new_discoveries = []
        
        for corpus_name, texts in corpus_data.items():
            logger.info(f"Analyzing corpus: {corpus_name} ({len(texts)} texts)")
            
            # Analyze texts for potential new primes
            candidates = self._analyze_corpus_for_primes(texts, discovery_method)
            
            for candidate in candidates:
                # Validate if it's truly a new prime
                if self._validate_prime_candidate(candidate, texts):
                    discovery = PrimeDiscovery(
                        prime_name=candidate['name'],
                        source_corpus=corpus_name,
                        discovery_method=discovery_method,
                        confidence_score=candidate['confidence'],
                        semantic_category=candidate['category'],
                        discovered_at=datetime.now().isoformat(),
                        evidence=candidate['evidence'],
                        cross_lingual_validation={}
                    )
                    
                    # Check if this is actually new
                    if not self._is_existing_prime(discovery.prime_name):
                        new_discoveries.append(discovery)
                        logger.info(f"Discovered new prime: {discovery.prime_name}")
        
        return new_discoveries
    
    def _analyze_corpus_for_primes(self, texts: List[str], method: str) -> List[Dict]:
        """Analyze corpus texts for potential new primes."""
        candidates = []
        
        if method == "UD":
            candidates = self._ud_prime_analysis(texts)
        elif method == "MWE":
            candidates = self._mwe_prime_analysis(texts)
        elif method == "SEMANTIC":
            candidates = self._semantic_prime_analysis(texts)
        
        return candidates
    
    def _ud_prime_analysis(self, texts: List[str]) -> List[Dict]:
        """Analyze texts using Universal Dependencies for new primes."""
        # This would integrate with the existing UD detection system
        # For now, return a placeholder implementation
        candidates = []
        
        # Example UD patterns that might reveal new primes
        ud_patterns = {
            "ABILITY": {"pattern": "ability to", "category": "capability"},
            "OBLIGATION": {"pattern": "must", "category": "modality"},
            "AGAIN": {"pattern": "again", "category": "repetition"},
            "FINISH": {"pattern": "finish", "category": "completion"}
        }
        
        for text in texts:
            for prime_name, pattern_info in ud_patterns.items():
                if pattern_info["pattern"] in text.lower():
                    candidates.append({
                        "name": prime_name,
                        "confidence": 0.8,
                        "category": pattern_info["category"],
                        "evidence": [text]
                    })
        
        return candidates
    
    def _mwe_prime_analysis(self, texts: List[str]) -> List[Dict]:
        """Analyze texts for multi-word expressions that might be primes."""
        # This would integrate with the existing MWE detection system
        return []
    
    def _semantic_prime_analysis(self, texts: List[str]) -> List[Dict]:
        """Analyze texts using semantic similarity for new primes."""
        # This would use semantic embeddings to find new concepts
        return []
    
    def _validate_prime_candidate(self, candidate: Dict, texts: List[str]) -> bool:
        """Validate if a candidate is truly a new NSM prime."""
        # Check frequency across corpus
        frequency = sum(1 for text in texts if candidate['name'].lower() in text.lower())
        
        # Check semantic properties (universality, atomicity, etc.)
        # This is a simplified validation - in practice, this would be more sophisticated
        
        return frequency >= 3 and candidate['confidence'] >= 0.7
    
    def _is_existing_prime(self, prime_name: str) -> bool:
        """Check if a prime already exists in our system."""
        # Check against existing discoveries
        for discovery in self.discoveries:
            if discovery.prime_name == prime_name:
                return True
        
        # Check against standard NSM primes
        standard_primes = self._get_standard_nsm_primes()
        return prime_name in standard_primes
    
    def _get_standard_nsm_primes(self) -> Set[str]:
        """Get the set of standard NSM primes."""
        return {
            "I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY",
            "KIND", "PART", "THIS", "THE_SAME", "OTHER", "ONE", "TWO", 
            "SOME", "ALL", "MUCH", "MANY", "GOOD", "BAD", "BIG", "SMALL",
            "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR", "SAY", "WORDS",
            "TRUE", "FALSE", "DO", "HAPPEN", "MOVE", "TOUCH", "BE_SOMEWHERE",
            "THERE_IS", "HAVE", "BE_SOMEONE", "LIVE", "DIE", "WHEN", "NOW",
            "BEFORE", "AFTER", "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME",
            "MOMENT", "WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR",
            "INSIDE", "NOT", "MAYBE", "CAN", "BECAUSE", "IF", "VERY", "MORE", "LIKE"
        }
    
    def generate_cross_lingual_mappings(self, discovery: PrimeDiscovery) -> List[PrimeMapping]:
        """Generate mappings for a new prime across all languages."""
        logger.info(f"Generating cross-lingual mappings for {discovery.prime_name}")
        
        mappings = []
        
        # Define language-specific mappings for the new prime
        language_mappings = self._get_language_mappings_for_prime(discovery.prime_name)
        
        for language in self.supported_languages:
            if language.value in language_mappings:
                mapping = PrimeMapping(
                    prime_name=discovery.prime_name,
                    language=language,
                    word_form=language_mappings[language.value],
                    confidence=0.8,
                    source="auto"
                )
                mappings.append(mapping)
                
                # Update cross-lingual validation
                discovery.cross_lingual_validation[language.value] = True
        
        return mappings
    
    def _get_language_mappings_for_prime(self, prime_name: str) -> Dict[str, str]:
        """Get language-specific mappings for a new prime."""
        # This would use translation APIs, dictionaries, or manual mappings
        # For now, return placeholder mappings
        mappings = {
            "en": prime_name.lower(),
            "es": self._translate_to_spanish(prime_name),
            "fr": self._translate_to_french(prime_name),
            "de": self._translate_to_german(prime_name),
            "it": self._translate_to_italian(prime_name),
            "pt": self._translate_to_portuguese(prime_name),
            "ru": self._translate_to_russian(prime_name),
            "zh": self._translate_to_chinese(prime_name),
            "ja": self._translate_to_japanese(prime_name),
            "ko": self._translate_to_korean(prime_name)
        }
        
        return mappings
    
    def _translate_to_spanish(self, prime_name: str) -> str:
        """Translate prime name to Spanish."""
        translations = {
            "ABILITY": "habilidad", "OBLIGATION": "obligación",
            "AGAIN": "otra vez", "FINISH": "terminar"
        }
        return translations.get(prime_name, prime_name.lower())
    
    def _translate_to_french(self, prime_name: str) -> str:
        """Translate prime name to French."""
        translations = {
            "ABILITY": "capacité", "OBLIGATION": "obligation",
            "AGAIN": "encore", "FINISH": "finir"
        }
        return translations.get(prime_name, prime_name.lower())
    
    def _translate_to_german(self, prime_name: str) -> str:
        """Translate prime name to German."""
        translations = {
            "ABILITY": "fähigkeit", "OBLIGATION": "verpflichtung",
            "AGAIN": "wieder", "FINISH": "beenden"
        }
        return translations.get(prime_name, prime_name.lower())
    
    def _translate_to_italian(self, prime_name: str) -> str:
        """Translate prime name to Italian."""
        translations = {
            "ABILITY": "abilità", "OBLIGATION": "obbligo",
            "AGAIN": "di nuovo", "FINISH": "finire"
        }
        return translations.get(prime_name, prime_name.lower())
    
    def _translate_to_portuguese(self, prime_name: str) -> str:
        """Translate prime name to Portuguese."""
        translations = {
            "ABILITY": "habilidade", "OBLIGATION": "obrigação",
            "AGAIN": "novamente", "FINISH": "terminar"
        }
        return translations.get(prime_name, prime_name.lower())
    
    def _translate_to_russian(self, prime_name: str) -> str:
        """Translate prime name to Russian."""
        translations = {
            "ABILITY": "способность", "OBLIGATION": "обязанность",
            "AGAIN": "снова", "FINISH": "закончить"
        }
        return translations.get(prime_name, prime_name.lower())
    
    def _translate_to_chinese(self, prime_name: str) -> str:
        """Translate prime name to Chinese."""
        translations = {
            "ABILITY": "能力", "OBLIGATION": "义务",
            "AGAIN": "再次", "FINISH": "完成"
        }
        return translations.get(prime_name, prime_name.lower())
    
    def _translate_to_japanese(self, prime_name: str) -> str:
        """Translate prime name to Japanese."""
        translations = {
            "ABILITY": "能力", "OBLIGATION": "義務",
            "AGAIN": "再び", "FINISH": "終わる"
        }
        return translations.get(prime_name, prime_name.lower())
    
    def _translate_to_korean(self, prime_name: str) -> str:
        """Translate prime name to Korean."""
        translations = {
            "ABILITY": "능력", "OBLIGATION": "의무",
            "AGAIN": "다시", "FINISH": "끝내다"
        }
        return translations.get(prime_name, prime_name.lower())
    
    def integrate_new_primes(self, discoveries: List[PrimeDiscovery]) -> bool:
        """Integrate new prime discoveries into the language expansion system."""
        logger.info(f"Integrating {len(discoveries)} new prime discoveries")
        
        # Create backup before making changes
        self._create_backup()
        
        try:
            for discovery in discoveries:
                # Generate mappings for all languages
                mappings = self.generate_cross_lingual_mappings(discovery)
                
                # Add to discoveries and mappings
                self.discoveries.append(discovery)
                self.mappings.extend(mappings)
                
                # Update language expansion
                self._update_language_expansion(discovery, mappings)
                
                logger.info(f"Successfully integrated prime: {discovery.prime_name}")
            
            # Save updated data
            self._save_discoveries()
            self._save_mappings()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate new primes: {e}")
            self._restore_backup()
            return False
    
    def _update_language_expansion(self, discovery: PrimeDiscovery, mappings: List[PrimeMapping]):
        """Update the language expansion with new prime mappings."""
        # This would modify the language_expansion.py file or reload it
        # For now, we'll create a method to regenerate the file
        
        # Get current mappings for each language
        for mapping in mappings:
            language = mapping.language
            prime_name = mapping.prime_name
            word_form = mapping.word_form
            
            # Add to the appropriate language mapping in language_expansion
            if language in self.language_expansion.extended_mappings:
                self.language_expansion.extended_mappings[language][prime_name] = word_form
                logger.info(f"Added {prime_name} -> {word_form} to {language.value}")
    
    def _create_backup(self):
        """Create backup of current state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup discoveries
        if self.discoveries_file.exists():
            backup_file = self.backup_dir / f"discoveries_backup_{timestamp}.json"
            with open(self.discoveries_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
        
        # Backup mappings
        if self.mappings_file.exists():
            backup_file = self.backup_dir / f"mappings_backup_{timestamp}.json"
            with open(self.mappings_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
        
        logger.info(f"Created backup at {timestamp}")
    
    def _restore_backup(self):
        """Restore from latest backup."""
        # Find latest backup files
        discovery_backups = list(self.backup_dir.glob("discoveries_backup_*.json"))
        mapping_backups = list(self.backup_dir.glob("mappings_backup_*.json"))
        
        if discovery_backups:
            latest_discovery = max(discovery_backups, key=lambda x: x.stat().st_mtime)
            with open(latest_discovery, 'r') as src, open(self.discoveries_file, 'w') as dst:
                dst.write(src.read())
        
        if mapping_backups:
            latest_mapping = max(mapping_backups, key=lambda x: x.stat().st_mtime)
            with open(latest_mapping, 'r') as src, open(self.mappings_file, 'w') as dst:
                dst.write(src.read())
        
        logger.info("Restored from backup")
    
    def get_discovery_summary(self) -> Dict:
        """Get a summary of all prime discoveries."""
        return {
            "total_discoveries": len(self.discoveries),
            "total_mappings": len(self.mappings),
            "languages_supported": len(self.supported_languages),
            "discoveries_by_method": self._count_discoveries_by_method(),
            "discoveries_by_category": self._count_discoveries_by_category(),
            "recent_discoveries": self._get_recent_discoveries(5)
        }
    
    def _count_discoveries_by_method(self) -> Dict[str, int]:
        """Count discoveries by discovery method."""
        counts = {}
        for discovery in self.discoveries:
            method = discovery.discovery_method
            counts[method] = counts.get(method, 0) + 1
        return counts
    
    def _count_discoveries_by_category(self) -> Dict[str, int]:
        """Count discoveries by semantic category."""
        counts = {}
        for discovery in self.discoveries:
            category = discovery.semantic_category
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    def _get_recent_discoveries(self, count: int) -> List[Dict]:
        """Get the most recent discoveries."""
        sorted_discoveries = sorted(self.discoveries, 
                                  key=lambda x: x.discovered_at, 
                                  reverse=True)
        return [asdict(d) for d in sorted_discoveries[:count]]
