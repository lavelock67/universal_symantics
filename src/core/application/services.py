#!/usr/bin/env python3
"""
Application Services

This module implements the business logic for the NSM system using clean
architecture principles and proper dependency injection.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time

from ...shared.config.settings import get_settings
from ...shared.logging.logger import get_logger, PerformanceContext
from ...shared.exceptions.exceptions import (
    NSMBaseException, DiscoveryError, CorpusProcessingError, 
    PerformanceError, create_error_context
)
from ...core.domain.models import (
    NSMPrime, MWE, PrimeCandidate, Corpus, DetectionResult, 
    DiscoveryResult, GenerationResult, Language, PrimeType, MWEType
)
from ...core.infrastructure.model_manager import get_model_manager


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
            
        except Exception as e:
            self.logger.error(f"Failed to initialize detection components: {str(e)}")
            raise
    
    def detect_primes(self, text: str, language: Language) -> DetectionResult:
        """Detect NSM primes in the given text."""
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
                primes = self._detect_primes_lexical(doc, language)
                primes.extend(self._detect_primes_semantic(doc, language))
                primes.extend(self._detect_primes_ud(doc, language))
                
                # Remove duplicates and sort by confidence
                unique_primes = self._deduplicate_primes(primes)
                
                # Detect MWEs
                mwes = self.detect_mwes(text, language)
                
                # Calculate overall confidence
                confidence = self._calculate_confidence(unique_primes, mwes)
                
                return DetectionResult(
                    primes=unique_primes,
                    mwes=mwes,
                    confidence=confidence,
                    processing_time=time.time(),
                    language=language
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
                
                # Detect MWEs using dependency patterns
                mwes.extend(self._detect_mwes_dependency(doc, language))
                
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
                        text=token.text,
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
                    text=prime_text,
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
                    text=match.text,
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
                mwe_obj = MWE(
                    text=match.text,
                    type=self._get_mwe_type(pattern_name),
                    primes=pattern.get("primes", []),
                    confidence=0.8,
                    start=match.start_char,
                    end=match.end_char,
                    language=language
                )
                mwes.append(mwe_obj)
        
        return mwes
    
    def _detect_mwes_dependency(self, doc, language: Language) -> List[MWE]:
        """Detect MWEs using dependency patterns."""
        mwes = []
        
        # Use SpaCy's dependency matcher
        matcher = self.spacy_models[language.value].get_pipe("dependency_matcher")
        
        # Define dependency patterns for MWEs
        dep_patterns = self._get_dependency_patterns(language)
        
        for pattern_name, pattern in dep_patterns.items():
            matches = matcher(doc)
            for match_id, token_ids in matches:
                if match_id == pattern_name:
                    start_token = doc[token_ids[0]]
                    end_token = doc[token_ids[-1]]
                    
                    mwe_obj = MWE(
                        text=doc[start_token.i:end_token.i + 1].text,
                        type=self._get_mwe_type(pattern_name),
                        primes=pattern.get("primes", []),
                        confidence=0.9,
                        start=start_token.idx,
                        end=end_token.idx + len(end_token.text),
                        language=language
                    )
                    mwes.append(mwe_obj)
        
        return mwes
    
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
            "THINK": {"lemma": "think", "pos": "VERB"},
            "SAY": {"lemma": "say", "pos": "VERB"},
            "WANT": {"lemma": "want", "pos": "VERB"},
            "GOOD": {"lemma": "good", "pos": "ADJ"},
            "BAD": {"lemma": "bad", "pos": "ADJ"},
            "BIG": {"lemma": "big", "pos": "ADJ"},
            "SMALL": {"lemma": "small", "pos": "ADJ"},
        }
        
        # Add language-specific patterns
        if language == Language.SPANISH:
            patterns.update({
                "THINK": {"lemma": "pensar", "pos": "VERB"},
                "SAY": {"lemma": "decir", "pos": "VERB"},
                "WANT": {"lemma": "querer", "pos": "VERB"},
                "GOOD": {"lemma": "bueno", "pos": "ADJ"},
                "BAD": {"lemma": "malo", "pos": "ADJ"},
            })
        elif language == Language.FRENCH:
            patterns.update({
                "THINK": {"lemma": "penser", "pos": "VERB"},
                "SAY": {"lemma": "dire", "pos": "VERB"},
                "WANT": {"lemma": "vouloir", "pos": "VERB"},
                "GOOD": {"lemma": "bon", "pos": "ADJ"},
                "BAD": {"lemma": "mauvais", "pos": "ADJ"},
            })
        
        return patterns
    
    def _get_prime_type(self, prime_name: str) -> PrimeType:
        """Get prime type from prime name."""
        type_mapping = {
            "THINK": PrimeType.MENTAL_PREDICATE,
            "SAY": PrimeType.MENTAL_PREDICATE,
            "WANT": PrimeType.MENTAL_PREDICATE,
            "GOOD": PrimeType.EVALUATOR,
            "BAD": PrimeType.EVALUATOR,
            "BIG": PrimeType.DESCRIPTOR,
            "SMALL": PrimeType.DESCRIPTOR,
        }
        return type_mapping.get(prime_name, PrimeType.SUBSTANTIVE)
    
    def _get_mwe_type(self, pattern_name: str) -> MWEType:
        """Get MWE type from pattern name."""
        type_mapping = {
            "quantifier": MWEType.QUANTIFIER,
            "intensifier": MWEType.INTENSIFIER,
            "negation": MWEType.NEGATION,
            "modality": MWEType.MODALITY,
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
        # This would be loaded from configuration
        return {}
    
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
        # Implementation would depend on the specific matching strategy
        return []


# Service factory
def create_detection_service() -> DetectionService:
    """Create a detection service instance."""
    return NSMDetectionService()
