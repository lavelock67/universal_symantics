#!/usr/bin/env python3
"""
Domain Models

This module defines the core domain models for the NSM system with proper
data structures, validation, and business logic.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


class PrimeType(str, Enum):
    """Types of NSM primes."""
    SUBSTANTIVE = "substantive"
    MENTAL_PREDICATE = "mental_predicate"
    LOGICAL_OPERATOR = "logical_operator"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    QUANTIFIER = "quantifier"
    EVALUATOR = "evaluator"
    ACTION = "action"
    DESCRIPTOR = "descriptor"
    MODAL = "modal"


class MWEType(str, Enum):
    """Types of multi-word expressions."""
    QUANTIFIER = "quantifier"
    INTENSIFIER = "intensifier"
    NEGATION = "negation"
    MODALITY = "modality"
    IDIOM = "idiom"
    PHRASAL_VERB = "phrasal_verb"


class DiscoveryStatus(str, Enum):
    """Status of discovery operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NSMPrime:
    """Represents an NSM prime with its properties."""
    
    text: str
    type: PrimeType
    language: Language
    universality_score: float = field(default=1.0)
    frequency: int = field(default=0)
    related_primes: List[str] = field(default_factory=list)
    semantic_cluster: Optional[str] = None
    confidence: float = field(default=1.0)
    
    def __post_init__(self):
        """Validate prime after initialization."""
        if not 0.0 <= self.universality_score <= 1.0:
            raise ValueError("Universality score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.frequency < 0:
            raise ValueError("Frequency must be non-negative")


@dataclass
class MWE:
    """Represents a multi-word expression."""
    
    text: str
    type: MWEType
    primes: List[str]
    confidence: float
    start: int
    end: int
    language: Language
    frequency: int = field(default=0)
    
    def __post_init__(self):
        """Validate MWE after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.start < 0 or self.end < 0:
            raise ValueError("Start and end positions must be non-negative")
        if self.start >= self.end:
            raise ValueError("Start position must be less than end position")


@dataclass
class PrimeCandidate:
    """Represents a candidate for a new NSM prime."""
    
    text: str
    mdl_delta: float
    confidence: float
    frequency: int
    semantic_cluster: str
    universality_score: float
    related_primes: List[str]
    context_examples: List[str]
    linguistic_features: Dict[str, Any]
    language: Language
    discovery_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate candidate after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.universality_score <= 1.0:
            raise ValueError("Universality score must be between 0.0 and 1.0")
        if self.frequency < 0:
            raise ValueError("Frequency must be non-negative")


@dataclass
class Corpus:
    """Represents a text corpus for analysis."""
    
    text: str
    language: Language
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def word_count(self) -> int:
        """Get word count of the corpus."""
        return len(self.text.split())
    
    @property
    def character_count(self) -> int:
        """Get character count of the corpus."""
        return len(self.text)
    
    @property
    def sentence_count(self) -> int:
        """Get approximate sentence count."""
        import re
        sentences = re.split(r'[.!?]+', self.text)
        return len([s for s in sentences if s.strip()])


@dataclass
class DetectionResult:
    """Result of prime detection operation."""
    
    primes: List[NSMPrime]
    mwes: List[MWE]
    confidence: float
    processing_time: float
    language: Language
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")


@dataclass
class DiscoveryResult:
    """Result of prime discovery operation."""
    
    candidates: List[PrimeCandidate]
    accepted: List[PrimeCandidate]
    rejected: List[PrimeCandidate]
    processing_time: float
    corpus_stats: Dict[str, Any]
    discovery_metrics: Dict[str, Any]
    status: DiscoveryStatus = DiscoveryStatus.COMPLETED
    
    def __post_init__(self):
        """Validate result after initialization."""
        if self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")


@dataclass
class GenerationResult:
    """Result of text generation operation."""
    
    generated_text: str
    source_primes: List[str]
    confidence: float
    processing_time: float
    target_language: Language
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")


# Pydantic models for API serialization
class PrimeDetectionRequest(BaseModel):
    """Request model for prime detection."""
    
    text: str = Field(..., min_length=1, description="Text to analyze")
    language: Language = Field(..., description="Language of the text")
    
    @validator('text')
    def validate_text(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        return v.strip()


class PrimeDiscoveryRequest(BaseModel):
    """Request model for prime discovery."""
    
    corpus: List[str] = Field(..., min_items=1, description="Corpus texts to analyze")
    max_candidates: int = Field(default=20, ge=1, le=100, description="Maximum candidates to generate")
    acceptance_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="MDL acceptance threshold")
    language: Language = Field(default=Language.ENGLISH, description="Primary language of the corpus")
    
    @validator('corpus')
    def validate_corpus(cls, v):
        if not any(text.strip() for text in v):
            raise ValueError("Corpus must contain at least one non-empty text")
        return [text.strip() for text in v if text.strip()]


class MWEDetectionRequest(BaseModel):
    """Request model for MWE detection."""
    
    text: str = Field(..., min_length=1, description="Text to analyze")
    language: Language = Field(..., description="Language of the text")
    
    @validator('text')
    def validate_text(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        return v.strip()


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    
    primes: List[str] = Field(..., min_items=1, description="NSM primes to generate from")
    target_language: Language = Field(..., description="Target language for generation")
    style: Optional[str] = Field(default="neutral", description="Generation style")
    
    @validator('primes')
    def validate_primes(cls, v):
        if not all(prime.strip() for prime in v):
            raise ValueError("All primes must be non-empty")
        return [prime.strip() for prime in v]


# Response models
class PrimeDetectionResponse(BaseModel):
    """Response model for prime detection."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[DetectionResult] = Field(None, description="Detection result")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., description="Total processing time")


class PrimeDiscoveryResponse(BaseModel):
    """Response model for prime discovery."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[DiscoveryResult] = Field(None, description="Discovery result")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., description="Total processing time")


class MWEDetectionResponse(BaseModel):
    """Response model for MWE detection."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    mwes: List[MWE] = Field(default_factory=list, description="Detected MWEs")
    primes: List[str] = Field(default_factory=list, description="Extracted primes")
    coverage: Dict[str, float] = Field(default_factory=dict, description="Coverage statistics")
    processing_time: float = Field(..., description="Processing time")
    error: Optional[str] = Field(None, description="Error message if failed")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[GenerationResult] = Field(None, description="Generation result")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., description="Total processing time")


# Utility functions
def create_prime(text: str, prime_type: PrimeType, language: Language, **kwargs) -> NSMPrime:
    """Create an NSM prime with validation."""
    return NSMPrime(text=text, type=prime_type, language=language, **kwargs)


def create_mwe(text: str, mwe_type: MWEType, primes: List[str], start: int, end: int, 
               language: Language, confidence: float = 0.8, **kwargs) -> MWE:
    """Create an MWE with validation."""
    return MWE(text=text, type=mwe_type, primes=primes, confidence=confidence, start=start, end=end, 
               language=language, **kwargs)


def create_candidate(text: str, mdl_delta: float, frequency: int, semantic_cluster: str,
                    language: Language, **kwargs) -> PrimeCandidate:
    """Create a prime candidate with validation."""
    return PrimeCandidate(text=text, mdl_delta=mdl_delta, frequency=frequency,
                         semantic_cluster=semantic_cluster, language=language, **kwargs)


def calculate_coverage(mwes: List[MWE], text: str) -> Dict[str, float]:
    """Calculate coverage statistics for MWEs."""
    if not text:
        return {"total": 0.0}
    
    total_chars = len(text)
    coverage_by_type = {}
    
    for mwe in mwes:
        mwe_type = mwe.type.value
        if mwe_type not in coverage_by_type:
            coverage_by_type[mwe_type] = 0
        
        mwe_length = mwe.end - mwe.start
        coverage_by_type[mwe_type] += mwe_length / total_chars
    
    total_coverage = sum(coverage_by_type.values())
    coverage_by_type["total"] = total_coverage
    
    return coverage_by_type
