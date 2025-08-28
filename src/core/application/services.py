#!/usr/bin/env python3
"""
Application Services

This module implements the business logic for the NSM system using clean
architecture principles and proper dependency injection.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time

from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger, PerformanceContext
from src.shared.exceptions.exceptions import (
    NSMBaseException, DiscoveryError, CorpusProcessingError, 
    PerformanceError, create_error_context
)
from src.core.domain.models import (
    NSMPrime, MWE, PrimeCandidate, Corpus, DetectionResult, 
    DiscoveryResult, GenerationResult, Language, PrimeType, MWEType
)
from src.core.infrastructure.model_manager import get_model_manager


class DetectionService(ABC):
    """Abstract base class for prime detection services."""
    
    @abstractmethod
    def detect_primes(self, text: str, language: Language) -> DetectionResult:
        """Detect NSM primes in the given text."""
        pass
    
    @abstractmethod
    def detect_mwes(self, text: str, language: Language) -> List[MWE]:
        """Detect multi-word expressions in the given text."""
        pass


class DiscoveryService(ABC):
    """Abstract base class for prime discovery services."""
    
    @abstractmethod
    def discover_primes(self, corpus: List[str], language: Language, 
                       max_candidates: int, acceptance_threshold: float) -> DiscoveryResult:
        """Discover new prime candidates from the given corpus."""
        pass
    
    @abstractmethod
    def validate_candidate(self, candidate: PrimeCandidate) -> bool:
        """Validate a prime candidate."""
        pass


class GenerationService(ABC):
    """Abstract base class for text generation services."""
    
    @abstractmethod
    def generate_text(self, primes: List[str], target_language: Language, 
                     style: Optional[str] = None) -> GenerationResult:
        """Generate text from NSM primes."""
        pass


class NSMDetectionService(DetectionService):
    """Concrete implementation of prime detection service."""
    
    def __init__(self):
        """Initialize the detection service."""
        self.settings = get_settings()
        self.logger = get_logger("detection_service")
        self.model_manager = get_model_manager()
        
        # Initialize detection components
        self._init_detection_components()
    
    def _init_detection_components(self):
        """Initialize detection components."""
        try:
            # Load SpaCy models for supported languages
            self.spacy_models = {}
            for lang in ["en", "es", "fr"]:
                try:
                    self.spacy_models[lang] = self.model_manager.get_spacy_model(lang)
                    self.logger.info(f"Loaded SpaCy model for {lang}")
                except Exception as e:
                    self.logger.warning(f"Failed to load SpaCy model for {lang}: {str(e)}")
            
            # Load SBERT model for semantic analysis
            self.sbert_model = self.model_manager.get_sbert_model()
            self.logger.info("Loaded SBERT model for semantic analysis")
            
            # Initialize UD detector (from srl_ud_detectors.py)
            try:
                from src.detect.srl_ud_detectors import detect_primitives_dep, detect_primitives_lexical
                self.ud_detector = {
                    'dependency': detect_primitives_dep,
                    'lexical': detect_primitives_lexical
                }
                self.logger.info("Loaded UD detector for dependency-based detection")
            except Exception as e:
                self.logger.warning(f"Failed to load UD detector: {str(e)}")
                self.ud_detector = None
            
            # Initialize MWE detector (from mwe_tagger.py)
            try:
                from src.detect.mwe_tagger import MWETagger
                self.mwe_detector = MWETagger()
                self.logger.info("Loaded MWE detector for multi-word expressions")
            except Exception as e:
                self.logger.warning(f"Failed to load MWE detector: {str(e)}")
                self.mwe_detector = None
            
            # Initialize Missing Prime Detector (from implement_missing_primes.py)
            try:
                from implement_missing_primes import MissingPrimeDetector
                self.missing_prime_detector = MissingPrimeDetector()
                self.logger.info("Loaded Missing Prime Detector for ABOVE/INSIDE/NEAR/ONE/WORDS")
            except Exception as e:
                self.logger.warning(f"Failed to load Missing Prime Detector: {str(e)}")
                self.missing_prime_detector = None
            
        except Exception as e:
            self.logger.error(f"Failed to initialize detection components: {str(e)}")
            raise
    
    def detect_primes(self, text: str, language: Language) -> DetectionResult:
        """Detect NSM primes in the given text."""
        start_time = time.time()
        with PerformanceContext("detect_primes", self.logger):
            try:
                # Validate input
                if not text or not text.strip():
                    raise ValueError("Text cannot be empty")
                
                # Check corpus size limits
                if len(text) > self.settings.performance.max_corpus_size:
                    raise PerformanceError(
                        metric="corpus_size",
                        current_value=len(text),
                        limit=self.settings.performance.max_corpus_size,
                        context=create_error_context("detect_primes", language=language.value)
                    )
                
                # Get SpaCy model for the language
                spacy_model = self.spacy_models.get(language.value)
                if not spacy_model:
                    raise ValueError(f"No SpaCy model available for language: {language.value}")
                
                # Process text with SpaCy
                doc = spacy_model(text)
                
                # Detect primes using multiple methods
                primes = []
                
                # 1. Enhanced Semantic detection (SBERT-based) - PRIMARY METHOD
                semantic_primes = self._detect_primes_semantic_enhanced(doc, language)
                primes.extend(semantic_primes)
                self.logger.info(f"Enhanced semantic detection found: {[p.text for p in semantic_primes]}")
                
                # 2. Enhanced UD-based detection (dependency parsing)
                ud_primes = self._detect_primes_ud_enhanced(doc, language)
                primes.extend(ud_primes)
                self.logger.info(f"Enhanced UD detection found: {[p.text for p in ud_primes]}")
                
                # 3. MWE detection (multi-word expressions)
                
                # 4. MWE detection
                if self.mwe_detector:
                    try:
                        self.logger.info(f"Calling MWE detector for text: '{text}'")
                        mwes = self.mwe_detector.detect_mwes(text, language)
                        self.logger.info(f"MWE detector returned: {len(mwes)} MWEs")
                        for mwe in mwes:
                            self.logger.info(f"Processing MWE: {mwe.text} with primes: {mwe.primes}")
                            for prime_text in mwe.primes:
                                prime_obj = NSMPrime(
                                    text=prime_text.upper(),  # Normalize to uppercase
                                    type=self._get_prime_type(prime_text),
                                    language=language,
                                    confidence=mwe.confidence,
                                    frequency=1
                                )
                                primes.append(prime_obj)
                        self.logger.info(f"MWE detection found: {[mwe.text for mwe in mwes]}")
                        self.logger.info(f"MWE primes found: {[prime for mwe in mwes for prime in mwe.primes]}")
                    except Exception as e:
                        self.logger.warning(f"MWE detection failed: {str(e)}")
                        import traceback
                        self.logger.warning(f"MWE detection traceback: {traceback.format_exc()}")
                else:
                    self.logger.warning("MWE detector not available")
                
                # 5. Missing Prime detection (ABOVE, INSIDE, NEAR, ONE, WORDS)
                if self.missing_prime_detector:
                    try:
                        self.logger.info(f"Calling Missing Prime detector for text: '{text}'")
                        missing_primes = self.missing_prime_detector.detect_all_missing_primes(text, language)
                        self.logger.info(f"Missing Prime detector returned: {len(missing_primes)} primes")
                        for prime_name, confidence in missing_primes.items():
                            prime_obj = NSMPrime(
                                text=prime_name.upper(),  # Normalize to uppercase
                                type=self._get_prime_type(prime_name),
                                language=language,
                                confidence=confidence,
                                frequency=1
                            )
                            primes.append(prime_obj)
                        self.logger.info(f"Missing Prime detection found: {list(missing_primes.keys())}")
                    except Exception as e:
                        self.logger.warning(f"Missing Prime detection failed: {str(e)}")
                        import traceback
                        self.logger.warning(f"Missing Prime detection traceback: {traceback.format_exc()}")
                else:
                    self.logger.warning("Missing Prime detector not available")
                
                # Remove duplicates and sort by confidence
                unique_primes = self._deduplicate_primes(primes)
                
                # Use MWEs from the MWE detector (already detected above)
                mwes = []
                if self.mwe_detector:
                    try:
                        mwes = self.mwe_detector.detect_mwes(text, language)
                    except Exception as e:
                        self.logger.warning(f"MWE detection failed: {str(e)}")
                
                # Calculate overall confidence
                confidence = self._calculate_confidence(unique_primes, mwes)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                return DetectionResult(
                    primes=unique_primes,
                    mwes=mwes,
                    confidence=confidence,
                    processing_time=processing_time,
                    language=language,
                    source_text=text
                )
                
            except Exception as e:
                self.logger.exception(f"Prime detection failed: {str(e)}")
                raise
    
    def detect_mwes(self, text: str, language: Language) -> List[MWE]:
        """Detect multi-word expressions in the given text."""
        with PerformanceContext("detect_mwes", self.logger):
            try:
                # Get SpaCy model
                spacy_model = self.spacy_models.get(language.value)
                if not spacy_model:
                    return []
                
                # Process text
                doc = spacy_model(text)
                
                # Detect MWEs using lexical patterns
                mwes = self._detect_mwes_lexical(doc, language)
                
                # Skip dependency patterns for now - use only lexical patterns
                # mwes.extend(self._detect_mwes_dependency(doc, language))
                
                # Remove overlapping MWEs
                mwes = self._remove_overlapping_mwes(mwes)
                
                return mwes
                
            except Exception as e:
                self.logger.exception(f"MWE detection failed: {str(e)}")
                return []
    
    def _detect_primes_lexical(self, doc, language: Language) -> List[NSMPrime]:
        """Detect primes using lexical patterns."""
        primes = []
        
        # Define lexical patterns for each language
        patterns = self._get_lexical_patterns(language)
        
        for token in doc:
            for prime, pattern in patterns.items():
                if self._matches_pattern(token, pattern):
                    prime_obj = NSMPrime(
                        text=prime,  # Use the pattern name (e.g., "ABOVE") not token text
                        type=self._get_prime_type(prime),
                        language=language,
                        confidence=0.8,
                        frequency=1
                    )
                    primes.append(prime_obj)
        
        return primes
    
    def _detect_primes_semantic(self, doc, language: Language) -> List[NSMPrime]:
        """Detect primes using semantic similarity."""
        primes = []
        
        # Get embeddings for the document
        doc_embedding = self.sbert_model.encode(doc.text)
        
        # Compare with known prime embeddings
        known_primes = self._get_known_primes(language)
        
        for prime_text, prime_type in known_primes.items():
            prime_embedding = self.sbert_model.encode(prime_text)
            similarity = self._cosine_similarity(doc_embedding, prime_embedding)
            
            if similarity > 0.7:  # Threshold for semantic similarity
                prime_obj = NSMPrime(
                    text=prime_text.upper(),  # Normalize to uppercase
                    type=prime_type,
                    language=language,
                    confidence=similarity,
                    frequency=1
                )
                primes.append(prime_obj)
        
        return primes
    
    def _detect_primes_ud(self, doc, language: Language) -> List[NSMPrime]:
        """Detect primes using Universal Dependencies."""
        primes = []
        
        # Define UD patterns for prime detection
        ud_patterns = self._get_ud_patterns(language)
        
        for pattern_name, pattern in ud_patterns.items():
            matches = self._find_ud_matches(doc, pattern)
            for match in matches:
                prime_obj = NSMPrime(
                    text=match.text.upper(),  # Normalize to uppercase
                    type=self._get_prime_type(pattern_name),
                    language=language,
                    confidence=0.9,
                    frequency=1
                )
                primes.append(prime_obj)
        
        return primes
    
    def _detect_mwes_lexical(self, doc, language: Language) -> List[MWE]:
        """Detect MWEs using lexical patterns."""
        mwes = []
        
        # Define MWE patterns
        mwe_patterns = self._get_mwe_patterns(language)
        
        for pattern_name, pattern in mwe_patterns.items():
            matches = self._find_mwe_matches(doc, pattern)
            for match in matches:
                # Find the specific pattern that matched
                matched_primes = []
                if "patterns" in pattern:
                    for p in pattern["patterns"]:
                        if "text" in p and p["text"].lower() == match.text.lower():
                            matched_primes = p.get("primes", [])
                            break
                
                mwe_obj = MWE(
                    text=match.text,
                    type=self._get_mwe_type(pattern_name),
                    primes=matched_primes,
                    confidence=0.8,
                    start=match.start_char,
                    end=match.end_char,
                    language=language
                )
                mwes.append(mwe_obj)
        
        return mwes
    
    def _detect_mwes_dependency(self, doc, language: Language) -> List[MWE]:
        """Detect MWEs using dependency patterns."""
        # Skip dependency patterns for now - use only lexical patterns
        # This method is kept for future implementation
        return []
    
    def _deduplicate_primes(self, primes: List[NSMPrime]) -> List[NSMPrime]:
        """Remove duplicate primes and merge confidence scores."""
        prime_dict = {}
        
        for prime in primes:
            key = (prime.text.lower(), prime.type, prime.language)
            if key in prime_dict:
                # Merge confidence scores
                existing = prime_dict[key]
                existing.confidence = max(existing.confidence, prime.confidence)
                existing.frequency += prime.frequency
            else:
                prime_dict[key] = prime
        
        return list(prime_dict.values())
    
    def _remove_overlapping_mwes(self, mwes: List[MWE]) -> List[MWE]:
        """Remove overlapping MWEs, keeping the ones with higher confidence."""
        if not mwes:
            return mwes
        
        # Sort by confidence (descending) and length (descending)
        mwes.sort(key=lambda x: (x.confidence, x.end - x.start), reverse=True)
        
        non_overlapping = []
        for mwe in mwes:
            overlaps = False
            for existing in non_overlapping:
                if self._mwes_overlap(mwe, existing):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(mwe)
        
        return non_overlapping
    
    def _mwes_overlap(self, mwe1: MWE, mwe2: MWE) -> bool:
        """Check if two MWEs overlap."""
        return not (mwe1.end <= mwe2.start or mwe2.end <= mwe1.start)
    
    def _calculate_confidence(self, primes: List[NSMPrime], mwes: List[MWE]) -> float:
        """Calculate overall confidence score."""
        if not primes and not mwes:
            return 0.0
        
        total_confidence = 0.0
        total_items = 0
        
        for prime in primes:
            total_confidence += prime.confidence
            total_items += 1
        
        for mwe in mwes:
            total_confidence += mwe.confidence
            total_items += 1
        
        return total_confidence / total_items if total_items > 0 else 0.0
    
    def _get_lexical_patterns(self, language: Language) -> Dict[str, Dict[str, Any]]:
        """Get lexical patterns for prime detection."""
        # This would be loaded from configuration or database
        patterns = {
            # Mental predicates
            "THINK": {"lemma": "think", "pos": "VERB"},
            "SAY": {"lemma": "say", "pos": "VERB"},
            "WANT": {"lemma": "want", "pos": "VERB"},
            "KNOW": {"lemma": "know", "pos": "VERB"},
            "SEE": {"lemma": "see", "pos": "VERB"},
            "HEAR": {"lemma": "hear", "pos": "VERB"},
            "FEEL": {"lemma": "feel", "pos": "VERB"},
            
            # Evaluators
            "GOOD": {"lemma": "good", "pos": "ADJ"},
            "BAD": {"lemma": "bad", "pos": "ADJ"},
            "RIGHT": {"lemma": "right", "pos": "ADJ"},
            "WRONG": {"lemma": "wrong", "pos": "ADJ"},
            "TRUE": {"lemma": "true", "pos": "ADJ"},
            "FALSE": {"lemma": "false", "pos": "ADJ"},
            
            # Descriptors
            "BIG": {"lemma": "big", "pos": "ADJ"},
            "SMALL": {"lemma": "small", "pos": "ADJ"},
            "LONG": {"lemma": "long", "pos": "ADJ"},
            "SHORT": {"lemma": "short", "pos": "ADJ"},
            "WIDE": {"lemma": "wide", "pos": "ADJ"},
            "NARROW": {"lemma": "narrow", "pos": "ADJ"},
            "THICK": {"lemma": "thick", "pos": "ADJ"},
            "THIN": {"lemma": "thin", "pos": "ADJ"},
            "HEAVY": {"lemma": "heavy", "pos": "ADJ"},
            "LIGHT": {"lemma": "light", "pos": "ADJ"},
            "STRONG": {"lemma": "strong", "pos": "ADJ"},
            "WEAK": {"lemma": "weak", "pos": "ADJ"},
            "HARD": {"lemma": "hard", "pos": "ADJ"},
            "SOFT": {"lemma": "soft", "pos": "ADJ"},
            "WARM": {"lemma": "warm", "pos": "ADJ"},
            "COLD": {"lemma": "cold", "pos": "ADJ"},
            "NEW": {"lemma": "new", "pos": "ADJ"},
            "OLD": {"lemma": "old", "pos": "ADJ"},
            
            # Substantives
            "I": {"lemma": "i", "pos": "PRON"},
            "YOU": {"lemma": "you", "pos": "PRON"},
            "SOMEONE": {"lemma": "someone", "pos": "PRON"},
            "PEOPLE": {"lemma": "people", "pos": "NOUN"},
            "SOMETHING": {"lemma": "something", "pos": "PRON"},
            "THIS": {"lemma": "this", "pos": "DET"},
            "THING": {"lemma": "thing", "pos": "NOUN"},
            "BODY": {"lemma": "body", "pos": "NOUN"},
            "WORLD": {"lemma": "world", "pos": "NOUN"},
            "WATER": {"lemma": "water", "pos": "NOUN"},
            "FIRE": {"lemma": "fire", "pos": "NOUN"},
            "EARTH": {"lemma": "earth", "pos": "NOUN"},
            "SKY": {"lemma": "sky", "pos": "NOUN"},
            "DAY": {"lemma": "day", "pos": "NOUN"},
            "NIGHT": {"lemma": "night", "pos": "NOUN"},
            "YEAR": {"lemma": "year", "pos": "NOUN"},
            "MONTH": {"lemma": "month", "pos": "NOUN"},
            "WEEK": {"lemma": "week", "pos": "NOUN"},
            "TIME": {"lemma": "time", "pos": "NOUN"},
            "PLACE": {"lemma": "place", "pos": "NOUN"},
            "WAY": {"lemma": "way", "pos": "NOUN"},
            "PART": {"lemma": "part", "pos": "NOUN"},
            "KIND": {"lemma": "kind", "pos": "NOUN"},
            
            # Quantifiers
            "MORE": {"lemma": "more", "pos": "ADJ"},
            "MANY": {"lemma": "many", "pos": "ADJ"},
            "MUCH": {"lemma": "much", "pos": "ADJ"},
            "ALL": {"lemma": "all", "pos": "DET"},
            "SOME": {"lemma": "some", "pos": "DET"},
            "NO": {"lemma": "no", "pos": "DET"},
            "ONE": {"lemma": "one", "pos": "NUM"},
            "TWO": {"lemma": "two", "pos": "NUM"},
            
            # Actions
            "READ": {"lemma": "read", "pos": "VERB"},
            "DO": {"lemma": "do", "pos": "VERB"},
            "LIVE": {"lemma": "live", "pos": "VERB"},
            "DIE": {"lemma": "die", "pos": "VERB"},
            "COME": {"lemma": "come", "pos": "VERB"},
            "GO": {"lemma": "go", "pos": "VERB"},
            "GIVE": {"lemma": "give", "pos": "VERB"},
            "TAKE": {"lemma": "take", "pos": "VERB"},
            "MAKE": {"lemma": "make", "pos": "VERB"},
            "BECOME": {"lemma": "become", "pos": "VERB"},
            
            # Auxiliaries
            "BE": {"lemma": "be", "pos": "AUX"},
            "HAVE": {"lemma": "have", "pos": "AUX"},
            "CAN": {"lemma": "can", "pos": "AUX"},
            "MAY": {"lemma": "may", "pos": "AUX"},
            "WILL": {"lemma": "will", "pos": "AUX"},
            "SHOULD": {"lemma": "should", "pos": "AUX"},
            
            # Logical operators
            "NOT": {"lemma": "not", "pos": "PART"},
            "BECAUSE": {"lemma": "because", "pos": "SCONJ"},
            "IF": {"lemma": "if", "pos": "SCONJ"},
            
            # Spatiotemporal
            "WHEN": {"lemma": "when", "pos": "SCONJ"},
            "WHERE": {"lemma": "where", "pos": "ADV"},
            "WHERE": {"lemma": "where", "pos": "SCONJ"},
            "ABOVE": {"lemma": "above", "pos": "ADV"},
            "BELOW": {"lemma": "below", "pos": "ADP"},
            "INSIDE": {"lemma": "inside", "pos": "ADV"},
            "OUTSIDE": {"lemma": "outside", "pos": "ADP"},
            "NEAR": {"lemma": "near", "pos": "ADP"},
            "FAR": {"lemma": "far", "pos": "ADJ"},
            "WORDS": {"lemma": "word", "pos": "NOUN"},
            "NOW": {"lemma": "now", "pos": "ADV"},
            "BEFORE": {"lemma": "before", "pos": "ADP"},
            "AFTER": {"lemma": "after", "pos": "ADP"},
            "TODAY": {"lemma": "today", "pos": "ADV"},
            "TOMORROW": {"lemma": "tomorrow", "pos": "ADV"},
            "YESTERDAY": {"lemma": "yesterday", "pos": "ADV"},
            "HERE": {"lemma": "here", "pos": "ADV"},
            "THERE": {"lemma": "there", "pos": "ADV"},
            "SIDE": {"lemma": "side", "pos": "NOUN"},
            "MOMENT": {"lemma": "moment", "pos": "NOUN"},
            "A_LONG_TIME": {"lemma": "long", "pos": "ADJ"},
            "A_SHORT_TIME": {"lemma": "short", "pos": "ADJ"},
            "FOR_SOME_TIME": {"lemma": "time", "pos": "NOUN"},
            "BE_SOMEWHERE": {"lemma": "somewhere", "pos": "ADV"},
            "BE_SOMEONE": {"lemma": "someone", "pos": "PRON"},
            "THERE_IS": {"lemma": "there", "pos": "ADV"},
            
            # Additional missing primes
            "HAPPEN": {"lemma": "happen", "pos": "VERB"},
            "ONE": {"lemma": "one", "pos": "NUM"},
            "THE_SAME": {"lemma": "same", "pos": "ADJ"},
            "WORDS": {"lemma": "word", "pos": "NOUN"},
            "MAYBE": {"lemma": "maybe", "pos": "ADV"},
            "TOUCH": {"lemma": "touch", "pos": "VERB"},
            "MOVE": {"lemma": "move", "pos": "VERB"},
            "LIKE": {"lemma": "like", "pos": "ADP"},
            "MORE": {"lemma": "more", "pos": "ADJ"},
            "VERY": {"lemma": "very", "pos": "ADV"},
            "SOME": {"lemma": "some", "pos": "DET"},
            "THERE": {"lemma": "there", "pos": "ADV"},
            "ABOVE": {"lemma": "above", "pos": "ADV"},
            "FAR": {"lemma": "far", "pos": "ADJ"},
            "INSIDE": {"lemma": "inside", "pos": "ADV"},
            
            # Descriptors (continued)
            "OTHER": {"lemma": "other", "pos": "ADJ"},
            "SAME": {"lemma": "same", "pos": "ADJ"},
            "DIFFERENT": {"lemma": "different", "pos": "ADJ"},
            "LEFT": {"lemma": "left", "pos": "ADJ"},
            "RIGHT_SIDE": {"lemma": "right", "pos": "ADJ"},
            "LIKE": {"lemma": "like", "pos": "ADP"},
            
            # Intensifiers
            "VERY": {"lemma": "very", "pos": "ADV"},
        }
        
        # Add language-specific patterns
        if language == Language.SPANISH:
            patterns.update({
                "THINK": {"lemma": "pensar", "pos": "VERB"},
                "SAY": {"lemma": "decir", "pos": "VERB"},
                "WANT": {"lemma": "querer", "pos": "VERB"},
                "GOOD": {"lemma": "bueno", "pos": "ADJ"},
                "BAD": {"lemma": "malo", "pos": "ADJ"},
                "PEOPLE": {"lemma": "gente", "pos": "NOUN"},
                "THIS": {"lemma": "esto", "pos": "DET"},
                "VERY": {"lemma": "muy", "pos": "ADV"},
                "NOT": {"lemma": "no", "pos": "PART"},
                "MORE": {"lemma": "más", "pos": "ADJ"},
                "MANY": {"lemma": "muchos", "pos": "ADJ"},
                "READ": {"lemma": "leer", "pos": "VERB"},
                "FALSE": {"lemma": "falso", "pos": "ADJ"},
                "DO": {"lemma": "hacer", "pos": "VERB"},
            })
        elif language == Language.FRENCH:
            patterns.update({
                "THINK": {"lemma": "penser", "pos": "VERB"},
                "SAY": {"lemma": "dire", "pos": "VERB"},
                "WANT": {"lemma": "vouloir", "pos": "VERB"},
                "GOOD": {"lemma": "bon", "pos": "ADJ"},
                "BAD": {"lemma": "mauvais", "pos": "ADJ"},
                "PEOPLE": {"lemma": "gens", "pos": "NOUN"},
                "THIS": {"lemma": "ce", "pos": "DET"},
                "VERY": {"lemma": "très", "pos": "ADV"},
                "NOT": {"lemma": "ne", "pos": "PART"},
                "MORE": {"lemma": "plus", "pos": "ADJ"},
                "MANY": {"lemma": "beaucoup", "pos": "ADJ"},
                "READ": {"lemma": "lire", "pos": "VERB"},
                "FALSE": {"lemma": "faux", "pos": "ADJ"},
                "DO": {"lemma": "faire", "pos": "VERB"},
            })
        
        return patterns
    
    def _get_prime_type(self, prime_name: str) -> PrimeType:
        """Get prime type from prime name."""
        # Normalize to uppercase for consistent mapping
        prime_name = prime_name.upper()
        
        type_mapping = {
            # Mental predicates
            "THINK": PrimeType.MENTAL_PREDICATE,
            "SAY": PrimeType.MENTAL_PREDICATE,
            "WANT": PrimeType.MENTAL_PREDICATE,
            "KNOW": PrimeType.MENTAL_PREDICATE,
            "SEE": PrimeType.MENTAL_PREDICATE,
            "HEAR": PrimeType.MENTAL_PREDICATE,
            "FEEL": PrimeType.MENTAL_PREDICATE,
            
            # Evaluators
            "GOOD": PrimeType.EVALUATOR,
            "BAD": PrimeType.EVALUATOR,
            "RIGHT": PrimeType.EVALUATOR,
            "WRONG": PrimeType.EVALUATOR,
            "TRUE": PrimeType.EVALUATOR,
            "FALSE": PrimeType.EVALUATOR,
            
            # Descriptors
            "BIG": PrimeType.DESCRIPTOR,
            "SMALL": PrimeType.DESCRIPTOR,
            "LONG": PrimeType.DESCRIPTOR,
            "SHORT": PrimeType.DESCRIPTOR,
            "WIDE": PrimeType.DESCRIPTOR,
            "NARROW": PrimeType.DESCRIPTOR,
            "THICK": PrimeType.DESCRIPTOR,
            "THIN": PrimeType.DESCRIPTOR,
            "HEAVY": PrimeType.DESCRIPTOR,
            "LIGHT": PrimeType.DESCRIPTOR,
            "STRONG": PrimeType.DESCRIPTOR,
            "WEAK": PrimeType.DESCRIPTOR,
            "HARD": PrimeType.DESCRIPTOR,
            "SOFT": PrimeType.DESCRIPTOR,
            "WARM": PrimeType.DESCRIPTOR,
            "COLD": PrimeType.DESCRIPTOR,
            "NEW": PrimeType.DESCRIPTOR,
            "OLD": PrimeType.DESCRIPTOR,
            "VERY": PrimeType.DESCRIPTOR,
            
            # Substantives
            "I": PrimeType.SUBSTANTIVE,
            "YOU": PrimeType.SUBSTANTIVE,
            "SOMEONE": PrimeType.SUBSTANTIVE,
            "PEOPLE": PrimeType.SUBSTANTIVE,
            "SOMETHING": PrimeType.SUBSTANTIVE,
            "THIS": PrimeType.SUBSTANTIVE,
            "THING": PrimeType.SUBSTANTIVE,
            "BODY": PrimeType.SUBSTANTIVE,
            "WORLD": PrimeType.SUBSTANTIVE,
            "WATER": PrimeType.SUBSTANTIVE,
            "FIRE": PrimeType.SUBSTANTIVE,
            "EARTH": PrimeType.SUBSTANTIVE,
            "SKY": PrimeType.SUBSTANTIVE,
            "DAY": PrimeType.SUBSTANTIVE,
            "NIGHT": PrimeType.SUBSTANTIVE,
            "YEAR": PrimeType.SUBSTANTIVE,
            "MONTH": PrimeType.SUBSTANTIVE,
            "WEEK": PrimeType.SUBSTANTIVE,
            "TIME": PrimeType.SUBSTANTIVE,
            "PLACE": PrimeType.SUBSTANTIVE,
            "WAY": PrimeType.SUBSTANTIVE,
            "PART": PrimeType.SUBSTANTIVE,
            "KIND": PrimeType.SUBSTANTIVE,
            
            # Quantifiers
            "MORE": PrimeType.QUANTIFIER,
            "MANY": PrimeType.QUANTIFIER,
            "MUCH": PrimeType.QUANTIFIER,
            "ALL": PrimeType.QUANTIFIER,
            "SOME": PrimeType.QUANTIFIER,
            "NO": PrimeType.QUANTIFIER,
            "ONE": PrimeType.QUANTIFIER,
            "TWO": PrimeType.QUANTIFIER,
            
            # Actions
            "READ": PrimeType.ACTION,
            "DO": PrimeType.ACTION,
            "LIVE": PrimeType.ACTION,
            "DIE": PrimeType.ACTION,
            "COME": PrimeType.ACTION,
            "GO": PrimeType.ACTION,
            "GIVE": PrimeType.ACTION,
            "TAKE": PrimeType.ACTION,
            "MAKE": PrimeType.ACTION,
            "BECOME": PrimeType.ACTION,
            
            # Auxiliaries
            "BE": PrimeType.MODAL,
            "HAVE": PrimeType.MODAL,
            "CAN": PrimeType.MODAL,
            "MAY": PrimeType.MODAL,
            "WILL": PrimeType.MODAL,
            "SHOULD": PrimeType.MODAL,
            
            # Logical operators
            "NOT": PrimeType.LOGICAL_OPERATOR,
            "BECAUSE": PrimeType.LOGICAL_OPERATOR,
            "IF": PrimeType.LOGICAL_OPERATOR,
            "MAYBE": PrimeType.LOGICAL_OPERATOR,
            
            # Actions
            "HAPPEN": PrimeType.ACTION,
            "TOUCH": PrimeType.ACTION,
            "MOVE": PrimeType.ACTION,
            
            # Quantifiers
            "ONE": PrimeType.QUANTIFIER,
            "THE_SAME": PrimeType.QUANTIFIER,
            
            # Speech
            "WORDS": PrimeType.SPEECH,
            "TRUE": PrimeType.SPEECH,
            "FALSE": PrimeType.SPEECH,
            
            # Intensifiers
            "LIKE": PrimeType.INTENSIFIER,
            "MORE": PrimeType.INTENSIFIER,
            "VERY": PrimeType.INTENSIFIER,
            
            # Spatiotemporal
            "WHEN": PrimeType.TEMPORAL,
            "WHERE": PrimeType.SPATIAL,
            "ABOVE": PrimeType.SPATIAL,
            "BELOW": PrimeType.SPATIAL,
            "INSIDE": PrimeType.SPATIAL,
            "OUTSIDE": PrimeType.SPATIAL,
            "NEAR": PrimeType.SPATIAL,
            "FAR": PrimeType.SPATIAL,
            "NOW": PrimeType.TEMPORAL,
            "BEFORE": PrimeType.TEMPORAL,
            "AFTER": PrimeType.TEMPORAL,
            "TODAY": PrimeType.TEMPORAL,
            "TOMORROW": PrimeType.TEMPORAL,
            "YESTERDAY": PrimeType.TEMPORAL,
            "HERE": PrimeType.SPATIAL,
            "THERE": PrimeType.SPATIAL,
            "SIDE": PrimeType.SPATIAL,
            "MOMENT": PrimeType.TEMPORAL,
            "A_LONG_TIME": PrimeType.TEMPORAL,
            "A_SHORT_TIME": PrimeType.TEMPORAL,
            "FOR_SOME_TIME": PrimeType.TEMPORAL,
            "BE_SOMEWHERE": PrimeType.SPATIAL,
            "BE_SOMEONE": PrimeType.SUBSTANTIVE,
            "THERE_IS": PrimeType.SPATIAL,
            
            # Additional descriptors
            "OTHER": PrimeType.DESCRIPTOR,
            "SAME": PrimeType.DESCRIPTOR,
            "DIFFERENT": PrimeType.DESCRIPTOR,
            "LEFT": PrimeType.DESCRIPTOR,
            "RIGHT_SIDE": PrimeType.DESCRIPTOR,
            "LIKE": PrimeType.DESCRIPTOR,
        }
        return type_mapping.get(prime_name, PrimeType.SUBSTANTIVE)
    
    def _get_mwe_type(self, pattern_name: str) -> MWEType:
        """Get MWE type from pattern name."""
        type_mapping = {
            "quantifier": MWEType.QUANTIFIER,
            "intensifier": MWEType.INTENSIFIER,
            "negation": MWEType.NEGATION,
            "modality": MWEType.MODALITY,
            "time_expressions": MWEType.QUANTIFIER,
            "existence": MWEType.QUANTIFIER,
            "similarity": MWEType.QUANTIFIER,
        }
        return type_mapping.get(pattern_name, MWEType.IDIOM)
    
    def _matches_pattern(self, token, pattern: Dict[str, Any]) -> bool:
        """Check if token matches the given pattern."""
        for key, value in pattern.items():
            if key == "lemma" and token.lemma_.lower() != value.lower():
                return False
            elif key == "pos" and token.pos_ != value:
                return False
        return True
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _get_known_primes(self, language: Language) -> Dict[str, PrimeType]:
        """Get known primes for the given language."""
        # This would be loaded from a database or configuration
        return {
            "think": PrimeType.MENTAL_PREDICATE,
            "say": PrimeType.MENTAL_PREDICATE,
            "want": PrimeType.MENTAL_PREDICATE,
            "good": PrimeType.EVALUATOR,
            "bad": PrimeType.EVALUATOR,
        }
    
    def _get_ud_patterns(self, language: Language) -> Dict[str, Any]:
        """Get Universal Dependency patterns."""
        # This would be loaded from configuration
        return {}
    
    def _get_mwe_patterns(self, language: Language) -> Dict[str, Any]:
        """Get MWE patterns."""
        patterns = {
            "quantifier": {
                "patterns": [
                    {"text": "at least", "primes": ["MORE"]},
                    {"text": "no more than", "primes": ["NOT", "MORE"]},
                    {"text": "a lot of", "primes": ["MANY"]},
                    {"text": "at most", "primes": ["NOT", "MORE"]},
                    {"text": "half of", "primes": ["HALF"]},
                    {"text": "some of", "primes": ["SOME"]},
                    {"text": "all of", "primes": ["ALL"]},
                    {"text": "none of", "primes": ["NOT", "SOME"]},
                ]
            },
            "intensifier": {
                "patterns": [
                    {"text": "very good", "primes": ["VERY", "GOOD"]},
                    {"text": "very bad", "primes": ["VERY", "BAD"]},
                    {"text": "really good", "primes": ["VERY", "GOOD"]},
                    {"text": "really bad", "primes": ["VERY", "BAD"]},
                ]
            },
            "negation": {
                "patterns": [
                    {"text": "do not", "primes": ["NOT"]},
                    {"text": "cannot", "primes": ["NOT"]},
                    {"text": "won't", "primes": ["NOT"]},
                    {"text": "don't", "primes": ["NOT"]},
                    {"text": "isn't", "primes": ["NOT"]},
                    {"text": "aren't", "primes": ["NOT"]},
                    {"text": "hasn't", "primes": ["NOT"]},
                    {"text": "haven't", "primes": ["NOT"]},
                    {"text": "didn't", "primes": ["NOT"]},
                    {"text": "doesn't", "primes": ["NOT"]},
                    {"text": "not true", "primes": ["NOT", "TRUE"]},
                    {"text": "not good", "primes": ["NOT", "GOOD"]},
                    {"text": "not bad", "primes": ["NOT", "BAD"]},
                    {"text": "not right", "primes": ["NOT", "RIGHT"]},
                    {"text": "not wrong", "primes": ["NOT", "WRONG"]},
                ]
            },
            "time_expressions": {
                "patterns": [
                    {"text": "a long time", "primes": ["A_LONG_TIME"]},
                    {"text": "a short time", "primes": ["A_SHORT_TIME"]},
                    {"text": "for some time", "primes": ["FOR_SOME_TIME"]},
                ]
            },
            "existence": {
                "patterns": [
                    {"text": "there is", "primes": ["THERE_IS"]},
                    {"text": "there are", "primes": ["THERE_IS"]},
                    {"text": "be someone", "primes": ["BE_SOMEONE"]},
                    {"text": "be somewhere", "primes": ["BE_SOMEWHERE"]},
                ]
            },
            "similarity": {
                "patterns": [
                    {"text": "the same", "primes": ["THE_SAME"]},
                    {"text": "exactly the same", "primes": ["THE_SAME"]},
                    {"text": "identical", "primes": ["THE_SAME"]},
                ]
            }
        }
        
        # Add language-specific patterns
        if language == Language.SPANISH:
            patterns["quantifier"]["patterns"].extend([
                {"text": "muy bueno", "primes": ["VERY", "GOOD"]},
                {"text": "muy malo", "primes": ["VERY", "BAD"]},
                {"text": "muchos", "primes": ["MANY"]},
                {"text": "al menos", "primes": ["MORE"]},
                {"text": "no más de", "primes": ["NOT", "MORE"]},
                {"text": "mucho de", "primes": ["MANY"]},
            ])
        elif language == Language.FRENCH:
            patterns["quantifier"]["patterns"].extend([
                {"text": "très bon", "primes": ["VERY", "GOOD"]},
                {"text": "très mauvais", "primes": ["VERY", "BAD"]},
                {"text": "beaucoup de", "primes": ["MANY"]},
                {"text": "au moins", "primes": ["MORE"]},
                {"text": "pas plus de", "primes": ["NOT", "MORE"]},
            ])
        
        return patterns
    
    def _get_dependency_patterns(self, language: Language) -> Dict[str, Any]:
        """Get dependency patterns."""
        # This would be loaded from configuration
        return {}
    
    def _find_ud_matches(self, doc, pattern) -> List[Any]:
        """Find UD pattern matches."""
        # Implementation would depend on the specific UD library used
        return []
    
    def _find_mwe_matches(self, doc, pattern) -> List[Any]:
        """Find MWE pattern matches."""
        matches = []
        text = doc.text.lower()
        
        if "patterns" in pattern:
            for p in pattern["patterns"]:
                if "text" in p:
                    pattern_text = p["text"].lower()
                    start = 0
                    while True:
                        pos = text.find(pattern_text, start)
                        if pos == -1:
                            break
                        
                        # Create a simple match object
                        class Match:
                            def __init__(self, text, start_char, end_char):
                                self.text = text
                                self.start_char = start_char
                                self.end_char = end_char
                        
                        match = Match(p["text"], pos, pos + len(pattern_text))
                        matches.append(match)
                        start = pos + 1
        
        return matches

    def _detect_primes_semantic_enhanced(self, doc, language: Language) -> List[NSMPrime]:
        """Enhanced semantic detection using comprehensive embeddings - CANONICAL PRIMES ONLY."""
        primes = []
        
        # Get ONLY canonical NSM prime embeddings (no language-specific variations)
        canonical_primes = self._get_canonical_prime_embeddings()
        
        # Process each token for semantic similarity
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
            
            # Skip very short tokens (likely noise)
            if len(token.text) < 2:
                continue
            
            # Get token embedding
            token_embedding = self.sbert_model.encode(token.text)
            
            # Compare with canonical primes only
            for prime_text, prime_type in canonical_primes.items():
                prime_embedding = self.sbert_model.encode(prime_text)
                similarity = self._cosine_similarity(token_embedding, prime_embedding)
                
                # Higher threshold to reduce false positives
                if similarity > 0.75:  # Increased from 0.5
                    # Ensure confidence is between 0.0 and 1.0
                    confidence = max(0.0, min(1.0, similarity))
                    prime_obj = NSMPrime(
                        text=prime_text.upper(),  # Use canonical English prime name
                        type=prime_type,
                        language=language,
                        confidence=confidence,
                        frequency=1
                    )
                    primes.append(prime_obj)
        
        # Document-level semantic similarity (more restrictive)
        doc_embedding = self.sbert_model.encode(doc.text)
        for prime_text, prime_type in canonical_primes.items():
            prime_embedding = self.sbert_model.encode(prime_text)
            similarity = self._cosine_similarity(doc_embedding, prime_embedding)
            
            if similarity > 0.8:  # Higher document-level threshold
                # Ensure confidence is between 0.0 and 1.0
                confidence = max(0.0, min(1.0, similarity))
                prime_obj = NSMPrime(
                    text=prime_text.upper(),  # Use canonical English prime name
                    type=prime_type,
                    language=language,
                    confidence=confidence,
                    frequency=1
                )
                primes.append(prime_obj)
        
        return primes

    def _detect_primes_ud_enhanced(self, doc, language: Language) -> List[NSMPrime]:
        """Enhanced UD-based detection using dependency analysis."""
        primes = []
        
        # Analyze dependency structure for semantic roles
        for token in doc:
            # Subject of action
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                primes.extend(self._analyze_semantic_role(token, "SUBJECT", language))
            
            # Object of action
            elif token.dep_ == "dobj" and token.head.pos_ == "VERB":
                primes.extend(self._analyze_semantic_role(token, "OBJECT", language))
            
            # Main action (root verb)
            elif token.dep_ == "ROOT" and token.pos_ == "VERB":
                primes.extend(self._analyze_action(token, language))
            
            # Adjective modifier
            elif token.dep_ == "amod" and token.head.pos_ == "NOUN":
                primes.extend(self._analyze_descriptor(token, language))
            
            # Adverb modifier
            elif token.dep_ == "advmod" and token.head.pos_ in ["VERB", "ADJ"]:
                primes.extend(self._analyze_intensifier(token, language))
            
            # Negation
            elif token.dep_ == "neg" or token.text.lower() in ["not", "no", "ne", "non"]:
                primes.append(NSMPrime(
                    text="NOT",
                    type=PrimeType.LOGICAL_OPERATOR,
                    language=language,
                    confidence=0.9,
                    frequency=1
                ))
        
        return primes

    def _analyze_semantic_role(self, token, role: str, language: Language) -> List[NSMPrime]:
        """Analyze semantic role and extract relevant primes."""
        primes = []
        
        if role == "SUBJECT":
            if token.text.lower() in ["i", "yo", "je"]:
                primes.append(NSMPrime(
                    text="I", 
                    type=PrimeType.SUBSTANTIVE, 
                    language=language, 
                    confidence=0.9
                ))
            elif token.text.lower() in ["you", "tu", "vous"]:
                primes.append(NSMPrime(
                    text="YOU", 
                    type=PrimeType.SUBSTANTIVE, 
                    language=language, 
                    confidence=0.9
                ))
            elif token.text.lower() in ["people", "gente", "gens"]:
                primes.append(NSMPrime(
                    text="PEOPLE", 
                    type=PrimeType.SUBSTANTIVE, 
                    language=language, 
                    confidence=0.9
                ))
            elif token.text.lower() in ["someone", "alguien", "quelqu'un"]:
                primes.append(NSMPrime(
                    text="SOMEONE", 
                    type=PrimeType.SUBSTANTIVE, 
                    language=language, 
                    confidence=0.9
                ))
        
        elif role == "OBJECT":
            if token.text.lower() in ["this", "esto", "ceci"]:
                primes.append(NSMPrime(
                    text="THIS", 
                    type=PrimeType.SUBSTANTIVE, 
                    language=language, 
                    confidence=0.9
                ))
            elif token.text.lower() in ["thing", "cosa", "chose"]:
                primes.append(NSMPrime(
                    text="THING", 
                    type=PrimeType.SUBSTANTIVE, 
                    language=language, 
                    confidence=0.9
                ))
            elif token.text.lower() in ["something", "algo", "quelque chose"]:
                primes.append(NSMPrime(
                    text="SOMETHING", 
                    type=PrimeType.SUBSTANTIVE, 
                    language=language, 
                    confidence=0.9
                ))
        
        return primes

    def _analyze_action(self, token, language: Language) -> List[NSMPrime]:
        """Analyze action verbs and extract relevant primes."""
        primes = []
        
        action_mapping = {
            "think": "THINK",
            "pensar": "THINK",
            "penser": "THINK",
            "say": "SAY",
            "decir": "SAY",
            "dire": "SAY",
            "want": "WANT",
            "querer": "WANT",
            "vouloir": "WANT",
            "know": "KNOW",
            "saber": "KNOW",
            "savoir": "KNOW",
            "see": "SEE",
            "ver": "SEE",
            "voir": "SEE",
            "hear": "HEAR",
            "oír": "HEAR",
            "entendre": "HEAR",
            "feel": "FEEL",
            "sentir": "FEEL",
            "do": "DO",
            "hacer": "DO",
            "faire": "DO",
            "make": "MAKE",
            "hacer": "MAKE",
            "faire": "MAKE",
            "go": "GO",
            "ir": "GO",
            "aller": "GO",
            "come": "COME",
            "venir": "COME",
            "give": "GIVE",
            "dar": "GIVE",
            "donner": "GIVE",
            "take": "TAKE",
            "tomar": "TAKE",
            "prendre": "TAKE",
            "live": "LIVE",
            "vivir": "LIVE",
            "vivre": "LIVE",
            "die": "DIE",
            "morir": "DIE",
            "mourir": "DIE",
            "become": "BECOME",
            "llegar a ser": "BECOME",
            "devenir": "BECOME",
            "happen": "HAPPEN",
            "pasar": "HAPPEN",
            "arriver": "HAPPEN",
            "touch": "TOUCH",
            "tocar": "TOUCH",
            "toucher": "TOUCH",
            "move": "MOVE",
            "mover": "MOVE",
            "bouger": "MOVE",
            "read": "READ",
            "leer": "READ",
            "lire": "READ",
            "finish": "FINISH",
            "terminar": "FINISH",
            "finir": "FINISH",
        }
        
        if token.lemma_.lower() in action_mapping:
            prime_text = action_mapping[token.lemma_.lower()]
            prime_type = self._get_prime_type(prime_text)
            primes.append(NSMPrime(
                text=prime_text,
                type=prime_type,
                language=language,
                confidence=0.9,
                frequency=1
            ))
        
        return primes

    def _analyze_descriptor(self, token, language: Language) -> List[NSMPrime]:
        """Analyze descriptors and extract relevant primes."""
        primes = []
        
        descriptor_mapping = {
            "good": "GOOD",
            "bueno": "GOOD",
            "bon": "GOOD",
            "bad": "BAD",
            "malo": "BAD",
            "mauvais": "BAD",
            "right": "RIGHT",
            "derecho": "RIGHT",
            "droit": "RIGHT",
            "wrong": "WRONG",
            "incorrecto": "WRONG",
            "incorrect": "WRONG",
            "true": "TRUE",
            "verdadero": "TRUE",
            "vrai": "TRUE",
            "false": "FALSE",
            "falso": "FALSE",
            "faux": "FALSE",
            "big": "BIG",
            "grande": "BIG",
            "grand": "BIG",
            "small": "SMALL",
            "pequeño": "SMALL",
            "petit": "SMALL",
            "long": "LONG",
            "largo": "LONG",
            "long": "LONG",
            "short": "SHORT",
            "corto": "SHORT",
            "court": "SHORT",
            "wide": "WIDE",
            "ancho": "WIDE",
            "large": "WIDE",
            "narrow": "NARROW",
            "estrecho": "NARROW",
            "étroit": "NARROW",
            "thick": "THICK",
            "grueso": "THICK",
            "épais": "THICK",
            "thin": "THIN",
            "delgado": "THIN",
            "mince": "THIN",
            "heavy": "HEAVY",
            "pesado": "HEAVY",
            "lourd": "HEAVY",
            "light": "LIGHT",
            "ligero": "LIGHT",
            "léger": "LIGHT",
            "strong": "STRONG",
            "fuerte": "STRONG",
            "fort": "STRONG",
            "weak": "WEAK",
            "débil": "WEAK",
            "faible": "WEAK",
            "hard": "HARD",
            "duro": "HARD",
            "dur": "HARD",
            "soft": "SOFT",
            "suave": "SOFT",
            "doux": "SOFT",
            "warm": "WARM",
            "cálido": "WARM",
            "chaud": "WARM",
            "cold": "COLD",
            "frío": "COLD",
            "froid": "COLD",
            "new": "NEW",
            "nuevo": "NEW",
            "nouveau": "NEW",
            "old": "OLD",
            "viejo": "OLD",
            "vieux": "OLD",
            "same": "SAME",
            "mismo": "SAME",
            "même": "SAME",
            "different": "DIFFERENT",
            "diferente": "DIFFERENT",
            "différent": "DIFFERENT",
            "other": "OTHER",
            "otro": "OTHER",
            "autre": "OTHER",
        }
        
        if token.lemma_.lower() in descriptor_mapping:
            prime_text = descriptor_mapping[token.lemma_.lower()]
            prime_type = self._get_prime_type(prime_text)
            primes.append(NSMPrime(
                text=prime_text,
                type=prime_type,
                language=language,
                confidence=0.9,
                frequency=1
            ))
        
        return primes

    def _analyze_intensifier(self, token, language: Language) -> List[NSMPrime]:
        """Analyze intensifiers and extract relevant primes."""
        primes = []
        
        intensifier_mapping = {
            "very": "VERY",
            "muy": "VERY",
            "très": "VERY",
            "really": "VERY",
            "realmente": "VERY",
            "vraiment": "VERY",
            "much": "MUCH",
            "mucho": "MUCH",
            "beaucoup": "MUCH",
            "more": "MORE",
            "más": "MORE",
            "plus": "MORE",
        }
        
        if token.lemma_.lower() in intensifier_mapping:
            prime_text = intensifier_mapping[token.lemma_.lower()]
            prime_type = self._get_prime_type(prime_text)
            primes.append(NSMPrime(
                text=prime_text,
                type=prime_type,
                language=language,
                confidence=0.9,
                frequency=1
            ))
        
        return primes

    def _get_canonical_prime_embeddings(self) -> Dict[str, PrimeType]:
        """Get ONLY the 65 canonical NSM primes from Anna Wierzbicka's theory - no language variations."""
        # ONLY the 65 canonical NSM primes - no language-specific variations
        canonical_primes = {
            # Phase 1: Substantives (7 primes)
            "i": PrimeType.SUBSTANTIVE,
            "you": PrimeType.SUBSTANTIVE,
            "someone": PrimeType.SUBSTANTIVE,
            "people": PrimeType.SUBSTANTIVE,
            "something": PrimeType.SUBSTANTIVE,
            "thing": PrimeType.SUBSTANTIVE,
            "body": PrimeType.SUBSTANTIVE,
            
            # Phase 2: Relational substantives (2 primes)
            "kind": PrimeType.SUBSTANTIVE,
            "part": PrimeType.SUBSTANTIVE,
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "this": PrimeType.SUBSTANTIVE,
            "the_same": PrimeType.SUBSTANTIVE,
            "other": PrimeType.SUBSTANTIVE,
            "one": PrimeType.QUANTIFIER,
            "two": PrimeType.QUANTIFIER,
            "some": PrimeType.QUANTIFIER,
            "all": PrimeType.QUANTIFIER,
            "much": PrimeType.QUANTIFIER,
            "many": PrimeType.QUANTIFIER,
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "good": PrimeType.EVALUATOR,
            "bad": PrimeType.EVALUATOR,
            "big": PrimeType.DESCRIPTOR,
            "small": PrimeType.DESCRIPTOR,
            
            # Phase 5: Mental predicates (6 primes)
            "think": PrimeType.MENTAL_PREDICATE,
            "know": PrimeType.MENTAL_PREDICATE,
            "want": PrimeType.MENTAL_PREDICATE,
            "feel": PrimeType.MENTAL_PREDICATE,
            "see": PrimeType.MENTAL_PREDICATE,
            "hear": PrimeType.MENTAL_PREDICATE,
            
            # Phase 6: Speech (4 primes)
            "say": PrimeType.MENTAL_PREDICATE,
            "words": PrimeType.SUBSTANTIVE,
            "true": PrimeType.EVALUATOR,
            "false": PrimeType.EVALUATOR,
            
            # Phase 7: Actions and events (4 primes)
            "do": PrimeType.ACTION,
            "happen": PrimeType.ACTION,
            "move": PrimeType.ACTION,
            "touch": PrimeType.ACTION,
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "be_somewhere": PrimeType.SPATIAL,
            "there_is": PrimeType.SPATIAL,
            "have": PrimeType.MODAL,
            "be_someone": PrimeType.SUBSTANTIVE,
            
            # Phase 9: Life and death (2 primes)
            "live": PrimeType.ACTION,
            "die": PrimeType.ACTION,
            
            # Phase 10: Time (8 primes)
            "when": PrimeType.TEMPORAL,
            "now": PrimeType.TEMPORAL,
            "before": PrimeType.TEMPORAL,
            "after": PrimeType.TEMPORAL,
            "a_long_time": PrimeType.TEMPORAL,
            "a_short_time": PrimeType.TEMPORAL,
            "for_some_time": PrimeType.TEMPORAL,
            "moment": PrimeType.TEMPORAL,
            
            # Phase 11: Space (7 primes)
            "where": PrimeType.SPATIAL,
            "here": PrimeType.SPATIAL,
            "above": PrimeType.SPATIAL,
            "below": PrimeType.SPATIAL,
            "far": PrimeType.SPATIAL,
            "near": PrimeType.SPATIAL,
            "inside": PrimeType.SPATIAL,
            
            # Logical concepts (5 primes)
            "not": PrimeType.LOGICAL_OPERATOR,
            "maybe": PrimeType.LOGICAL_OPERATOR,
            "can": PrimeType.MODAL,
            "because": PrimeType.LOGICAL_OPERATOR,
            "if": PrimeType.LOGICAL_OPERATOR,
            
            # Intensifier and augmentor (3 primes)
            "very": PrimeType.INTENSIFIER,
            "more": PrimeType.QUANTIFIER,
            "like": PrimeType.INTENSIFIER,
        }
        
        return canonical_primes

    def _get_comprehensive_prime_embeddings(self, language: Language) -> Dict[str, PrimeType]:
        """Get comprehensive prime embeddings for semantic detection - CANONICAL NSM PRIMES ONLY."""
        # Base primes - ONLY the 65 canonical NSM primes from Anna Wierzbicka's theory
        base_primes = {
            # Phase 1: Substantives (7 primes)
            "i": PrimeType.SUBSTANTIVE,
            "you": PrimeType.SUBSTANTIVE,
            "someone": PrimeType.SUBSTANTIVE,
            "people": PrimeType.SUBSTANTIVE,
            "something": PrimeType.SUBSTANTIVE,
            "thing": PrimeType.SUBSTANTIVE,
            "body": PrimeType.SUBSTANTIVE,
            
            # Phase 2: Relational substantives (2 primes)
            "kind": PrimeType.SUBSTANTIVE,
            "part": PrimeType.SUBSTANTIVE,
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "this": PrimeType.SUBSTANTIVE,
            "the_same": PrimeType.SUBSTANTIVE,
            "other": PrimeType.SUBSTANTIVE,
            "one": PrimeType.QUANTIFIER,
            "two": PrimeType.QUANTIFIER,
            "some": PrimeType.QUANTIFIER,
            "all": PrimeType.QUANTIFIER,
            "much": PrimeType.QUANTIFIER,
            "many": PrimeType.QUANTIFIER,
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "good": PrimeType.EVALUATOR,
            "bad": PrimeType.EVALUATOR,
            "big": PrimeType.DESCRIPTOR,
            "small": PrimeType.DESCRIPTOR,
            
            # Phase 5: Mental predicates (6 primes)
            "think": PrimeType.MENTAL_PREDICATE,
            "know": PrimeType.MENTAL_PREDICATE,
            "want": PrimeType.MENTAL_PREDICATE,
            "feel": PrimeType.MENTAL_PREDICATE,
            "see": PrimeType.MENTAL_PREDICATE,
            "hear": PrimeType.MENTAL_PREDICATE,
            
            # Phase 6: Speech (4 primes)
            "say": PrimeType.MENTAL_PREDICATE,
            "words": PrimeType.SUBSTANTIVE,
            "true": PrimeType.EVALUATOR,
            "false": PrimeType.EVALUATOR,
            
            # Phase 7: Actions and events (4 primes)
            "do": PrimeType.ACTION,
            "happen": PrimeType.ACTION,
            "move": PrimeType.ACTION,
            "touch": PrimeType.ACTION,
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "be_somewhere": PrimeType.SPATIAL,
            "there_is": PrimeType.SPATIAL,
            "have": PrimeType.MODAL,
            "be_someone": PrimeType.SUBSTANTIVE,
            
            # Phase 9: Life and death (2 primes)
            "live": PrimeType.ACTION,
            "die": PrimeType.ACTION,
            
            # Phase 10: Time (8 primes)
            "when": PrimeType.TEMPORAL,
            "now": PrimeType.TEMPORAL,
            "before": PrimeType.TEMPORAL,
            "after": PrimeType.TEMPORAL,
            "a_long_time": PrimeType.TEMPORAL,
            "a_short_time": PrimeType.TEMPORAL,
            "for_some_time": PrimeType.TEMPORAL,
            "moment": PrimeType.TEMPORAL,
            
            # Phase 11: Space (7 primes)
            "where": PrimeType.SPATIAL,
            "here": PrimeType.SPATIAL,
            "above": PrimeType.SPATIAL,
            "below": PrimeType.SPATIAL,
            "far": PrimeType.SPATIAL,
            "near": PrimeType.SPATIAL,
            "inside": PrimeType.SPATIAL,
            
            # Logical concepts (5 primes)
            "not": PrimeType.LOGICAL_OPERATOR,
            "maybe": PrimeType.LOGICAL_OPERATOR,
            "can": PrimeType.MODAL,
            "because": PrimeType.LOGICAL_OPERATOR,
            "if": PrimeType.LOGICAL_OPERATOR,
            
            # Intensifier and augmentor (3 primes)
            "very": PrimeType.INTENSIFIER,
            "more": PrimeType.QUANTIFIER,
            "like": PrimeType.INTENSIFIER,
        }
        
        # Add language-specific variations
        if language == Language.SPANISH:
            spanish_primes = {
                "bueno": PrimeType.EVALUATOR,
                "malo": PrimeType.EVALUATOR,
                "pensar": PrimeType.MENTAL_PREDICATE,
                "saber": PrimeType.MENTAL_PREDICATE,
                "querer": PrimeType.MENTAL_PREDICATE,
                "decir": PrimeType.MENTAL_PREDICATE,
                "hacer": PrimeType.ACTION,
                "ser": PrimeType.MODAL,
                "gente": PrimeType.SUBSTANTIVE,
                "cosa": PrimeType.SUBSTANTIVE,
                "esto": PrimeType.SUBSTANTIVE,
                "muy": PrimeType.INTENSIFIER,
                "no": PrimeType.LOGICAL_OPERATOR,
                "porque": PrimeType.LOGICAL_OPERATOR,
                "si": PrimeType.LOGICAL_OPERATOR,
                "cuando": PrimeType.TEMPORAL,
                "donde": PrimeType.SPATIAL,
                "arriba": PrimeType.SPATIAL,
                "abajo": PrimeType.SPATIAL,
                "dentro": PrimeType.SPATIAL,
                "fuera": PrimeType.SPATIAL,
                "cerca": PrimeType.SPATIAL,
                "lejos": PrimeType.SPATIAL,
                "ahora": PrimeType.TEMPORAL,
                "antes": PrimeType.TEMPORAL,
                "después": PrimeType.TEMPORAL,
                "hoy": PrimeType.TEMPORAL,
                "mañana": PrimeType.TEMPORAL,
                "ayer": PrimeType.TEMPORAL,
                "aquí": PrimeType.SPATIAL,
                "allí": PrimeType.SPATIAL,
                "lado": PrimeType.SPATIAL,
                "momento": PrimeType.TEMPORAL,
                "largo": PrimeType.DESCRIPTOR,
                "corto": PrimeType.DESCRIPTOR,
                "tiempo": PrimeType.TEMPORAL,
                "algún lugar": PrimeType.SPATIAL,
                "alguien": PrimeType.SUBSTANTIVE,
                "pasar": PrimeType.ACTION,
                "uno": PrimeType.QUANTIFIER,
                "mismo": PrimeType.DESCRIPTOR,
                "palabra": PrimeType.SUBSTANTIVE,
                "tal vez": PrimeType.MODAL,
                "tocar": PrimeType.ACTION,
                "mover": PrimeType.ACTION,
                "gustar": PrimeType.EVALUATOR,
                "más": PrimeType.QUANTIFIER,
                "algunos": PrimeType.QUANTIFIER,
                "otro": PrimeType.DESCRIPTOR,
                "diferente": PrimeType.DESCRIPTOR,
                "izquierda": PrimeType.SPATIAL,
                "derecha": PrimeType.SPATIAL,
            }
            base_primes.update(spanish_primes)
        
        elif language == Language.FRENCH:
            french_primes = {
                "bon": PrimeType.EVALUATOR,
                "mauvais": PrimeType.EVALUATOR,
                "penser": PrimeType.MENTAL_PREDICATE,
                "savoir": PrimeType.MENTAL_PREDICATE,
                "vouloir": PrimeType.MENTAL_PREDICATE,
                "dire": PrimeType.MENTAL_PREDICATE,
                "faire": PrimeType.ACTION,
                "être": PrimeType.MODAL,
                "gens": PrimeType.SUBSTANTIVE,
                "chose": PrimeType.SUBSTANTIVE,
                "ceci": PrimeType.SUBSTANTIVE,
                "très": PrimeType.INTENSIFIER,
                "ne pas": PrimeType.LOGICAL_OPERATOR,
                "parce que": PrimeType.LOGICAL_OPERATOR,
                "si": PrimeType.LOGICAL_OPERATOR,
                "quand": PrimeType.TEMPORAL,
                "où": PrimeType.SPATIAL,
                "au-dessus": PrimeType.SPATIAL,
                "en-dessous": PrimeType.SPATIAL,
                "dedans": PrimeType.SPATIAL,
                "dehors": PrimeType.SPATIAL,
                "près": PrimeType.SPATIAL,
                "loin": PrimeType.SPATIAL,
                "maintenant": PrimeType.TEMPORAL,
                "avant": PrimeType.TEMPORAL,
                "après": PrimeType.TEMPORAL,
                "aujourd'hui": PrimeType.TEMPORAL,
                "demain": PrimeType.TEMPORAL,
                "hier": PrimeType.TEMPORAL,
                "ici": PrimeType.SPATIAL,
                "là": PrimeType.SPATIAL,
                "côté": PrimeType.SPATIAL,
                "moment": PrimeType.TEMPORAL,
                "long": PrimeType.DESCRIPTOR,
                "court": PrimeType.DESCRIPTOR,
                "temps": PrimeType.TEMPORAL,
                "quelque part": PrimeType.SPATIAL,
                "quelqu'un": PrimeType.SUBSTANTIVE,
                "arriver": PrimeType.ACTION,
                "un": PrimeType.QUANTIFIER,
                "même": PrimeType.DESCRIPTOR,
                "mot": PrimeType.SUBSTANTIVE,
                "peut-être": PrimeType.MODAL,
                "toucher": PrimeType.ACTION,
                "bouger": PrimeType.ACTION,
                "aimer": PrimeType.EVALUATOR,
                "plus": PrimeType.QUANTIFIER,
                "quelques": PrimeType.QUANTIFIER,
                "autre": PrimeType.DESCRIPTOR,
                "différent": PrimeType.DESCRIPTOR,
                "gauche": PrimeType.SPATIAL,
                "droite": PrimeType.SPATIAL,
            }
            base_primes.update(french_primes)
        
        return base_primes

    def _get_canonical_prime_mapping(self, language: Language) -> Dict[str, str]:
        """Get mapping from language-specific words to canonical English prime names - CANONICAL NSM PRIMES ONLY."""
        if language == Language.ENGLISH:
            return {}
        
        # Spanish to English canonical mapping - ONLY canonical NSM primes
        if language == Language.SPANISH:
            return {
                # Mental predicates
                "pensar": "THINK",
                "saber": "KNOW", 
                "querer": "WANT",
                "decir": "SAY",
                "ver": "SEE",
                "oír": "HEAR",
                "sentir": "FEEL",
                
                # Evaluators
                "bueno": "GOOD",
                "malo": "BAD",
                "verdadero": "TRUE",
                "falso": "FALSE",
                
                # Descriptors
                "grande": "BIG",
                "pequeño": "SMALL",
                
                # Substantives
                "yo": "I",
                "tú": "YOU",
                "alguien": "SOMEONE",
                "gente": "PEOPLE",
                "algo": "SOMETHING",
                "esto": "THIS",
                "este": "THIS",
                "cosa": "THING",
                "cuerpo": "BODY",
                "parte": "PART",
                "tipo": "KIND",
                "palabra": "WORDS",
                
                # Quantifiers
                "más": "MORE",
                "muchos": "MANY",
                "mucho": "MUCH",
                "todo": "ALL",
                "algunos": "SOME",
                "uno": "ONE",
                "dos": "TWO",
                
                # Actions
                "hacer": "DO",
                "vivir": "LIVE",
                "morir": "DIE",
                "pasar": "HAPPEN",
                "tocar": "TOUCH",
                "mover": "MOVE",
                
                # Modals
                "tener": "HAVE",
                "poder": "CAN",
                
                # Logical operators
                "no": "NOT",
                "porque": "BECAUSE",
                "si": "IF",
                "tal vez": "MAYBE",
                
                # Intensifiers
                "muy": "VERY",
                "gustar": "LIKE",
                
                # Temporal
                "cuando": "WHEN",
                "ahora": "NOW",
                "antes": "BEFORE",
                "después": "AFTER",
                "momento": "MOMENT",
                
                # Spatial
                "donde": "WHERE",
                "aquí": "HERE",
                "arriba": "ABOVE",
                "abajo": "BELOW",
                "dentro": "INSIDE",
                "cerca": "NEAR",
                "lejos": "FAR",
            }
        
        # French to English canonical mapping - ONLY canonical NSM primes
        elif language == Language.FRENCH:
            return {
                # Mental predicates
                "penser": "THINK",
                "savoir": "KNOW",
                "vouloir": "WANT", 
                "dire": "SAY",
                "voir": "SEE",
                "entendre": "HEAR",
                "sentir": "FEEL",
                
                # Evaluators
                "bon": "GOOD",
                "mauvais": "BAD",
                "vrai": "TRUE",
                "faux": "FALSE",
                
                # Descriptors
                "grand": "BIG",
                "petit": "SMALL",
                
                # Substantives
                "je": "I",
                "tu": "YOU",
                "quelqu'un": "SOMEONE",
                "gens": "PEOPLE",
                "quelque chose": "SOMETHING",
                "ceci": "THIS",
                "ce": "THIS",
                "chose": "THING",
                "corps": "BODY",
                "partie": "PART",
                "genre": "KIND",
                "mot": "WORDS",
                
                # Quantifiers
                "plus": "MORE",
                "beaucoup": "MUCH",
                "tout": "ALL",
                "quelques": "SOME",
                "un": "ONE",
                "deux": "TWO",
                
                # Actions
                "faire": "DO",
                "vivre": "LIVE",
                "mourir": "DIE",
                "arriver": "HAPPEN",
                "toucher": "TOUCH",
                "bouger": "MOVE",
                
                # Modals
                "avoir": "HAVE",
                "pouvoir": "CAN",
                
                # Logical operators
                "ne pas": "NOT",
                "parce que": "BECAUSE",
                "si": "IF",
                "peut-être": "MAYBE",
                
                # Intensifiers
                "très": "VERY",
                "aimer": "LIKE",
                
                # Temporal
                "quand": "WHEN",
                "maintenant": "NOW",
                "avant": "BEFORE",
                "après": "AFTER",
                "moment": "MOMENT",
                
                # Spatial
                "où": "WHERE",
                "ici": "HERE",
                "au-dessus": "ABOVE",
                "en-dessous": "BELOW",
                "dedans": "INSIDE",
                "près": "NEAR",
                "loin": "FAR",
            }
        
        return {}

    def _get_canonical_prime_name(self, language_specific_word: str, language: Language) -> str:
        """Get the canonical English prime name for a language-specific word."""
        if language == Language.ENGLISH:
            return language_specific_word.upper()
        
        canonical_mapping = self._get_canonical_prime_mapping(language)
        return canonical_mapping.get(language_specific_word.lower(), language_specific_word.upper())

    def add_language_support(self, language: Language, prime_mappings: Dict[str, PrimeType]) -> None:
        """Add support for a new language with comprehensive prime mappings."""
        # Store the mappings for future use
        if not hasattr(self, '_language_mappings'):
            self._language_mappings = {}
        
        self._language_mappings[language] = prime_mappings
        self.logger.info(f"Added language support for {language.value} with {len(prime_mappings)} prime mappings")

    def get_language_coverage_report(self, language: Language) -> Dict[str, Any]:
        """Generate a coverage report for a specific language."""
        # Get all prime mappings for the language
        all_mappings = self._get_comprehensive_prime_embeddings(language)
        
        # Count primes by type
        prime_counts = {}
        required_types = {
            PrimeType.MENTAL_PREDICATE,
            PrimeType.EVALUATOR,
            PrimeType.DESCRIPTOR,
            PrimeType.SUBSTANTIVE,
            PrimeType.QUANTIFIER,
            PrimeType.ACTION,
            PrimeType.MODAL,
            PrimeType.LOGICAL_OPERATOR,
            PrimeType.INTENSIFIER,
            PrimeType.TEMPORAL,
            PrimeType.SPATIAL,
            PrimeType.SPEECH,
        }
        
        for prime_type in required_types:
            count = sum(1 for _, pt in all_mappings.items() if pt == prime_type)
            prime_counts[prime_type.value] = count
        
        # Identify missing prime types
        covered_types = set(all_mappings.values())
        missing_types = required_types - covered_types
        
        # Calculate coverage percentage
        total_required = len(required_types)
        total_covered = len(covered_types)
        coverage_percentage = (total_covered / total_required) * 100 if total_required > 0 else 0
        
        return {
            "language": language.value,
            "total_mappings": len(all_mappings),
            "coverage_percentage": coverage_percentage,
            "prime_counts_by_type": prime_counts,
            "missing_prime_types": [pt.value for pt in missing_types],
            "total_required_types": total_required,
            "total_covered_types": total_covered,
        }


# Service factory
def create_detection_service() -> DetectionService:
    """Create a detection service instance."""
    return NSMDetectionService()
