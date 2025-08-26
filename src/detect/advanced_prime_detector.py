#!/usr/bin/env python3
"""
Advanced Prime Detection and Discovery System

This module implements sophisticated prime detection using:
- Neural semantic similarity for candidate identification
- Distributional semantics for prime discovery
- Cross-lingual validation for universal primitives
- Semantic clustering for prime categorization
"""

import logging
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import AutoTokenizer, AutoModel
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class PrimeCandidate:
    """Candidate for a new NSM prime."""
    surface_form: str
    language: str
    semantic_cluster: str
    frequency: int
    cross_lingual_equivalents: List[str]
    semantic_similarity: float
    universality_score: float
    confidence: float
    contexts: List[str]
    proposed_prime: str

@dataclass
class PrimeDiscoveryResult:
    """Result of prime discovery process."""
    candidates: List[PrimeCandidate]
    clusters: Dict[str, List[str]]
    universality_analysis: Dict[str, float]
    cross_lingual_mappings: Dict[str, Dict[str, str]]
    discovery_metrics: Dict[str, Any]

class AdvancedPrimeDetector:
    """Advanced prime detection and discovery system."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the advanced prime detector."""
        self.model_name = model_name
        self.sbert_model = None
        self.tokenizer = None
        self.encoder_model = None
        self.nlp_models = {}
        
        # Load models
        self._load_models()
        
        # Known NSM primes for validation
        self.known_primes = {
            "I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY",
            "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR",
            "SAY", "WORDS", "TRUE", "FALSE",
            "DO", "HAPPEN", "MOVE", "TOUCH",
            "LIVE", "DIE",
            "GOOD", "BAD", "BIG", "SMALL",
            "THIS", "THE_SAME", "OTHER", "ONE", "TWO", "SOME", "ALL", "MUCH", "MANY",
            "NOT", "MAYBE", "CAN", "BECAUSE", "IF",
            "VERY", "MORE", "LIKE",
            "WHEN", "NOW", "BEFORE", "AFTER",
            "WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "INSIDE"
        }
        
        # Semantic clusters for prime categorization
        self.semantic_clusters = {
            "person": ["I", "YOU", "SOMEONE", "PEOPLE"],
            "thing": ["SOMETHING", "THING", "BODY"],
            "mental": ["THINK", "KNOW", "WANT", "FEEL"],
            "perception": ["SEE", "HEAR"],
            "communication": ["SAY", "WORDS"],
            "truth": ["TRUE", "FALSE"],
            "action": ["DO", "HAPPEN", "MOVE", "TOUCH"],
            "life": ["LIVE", "DIE"],
            "evaluation": ["GOOD", "BAD", "BIG", "SMALL"],
            "determiner": ["THIS", "THE_SAME", "OTHER"],
            "quantity": ["ONE", "TWO", "SOME", "ALL", "MUCH", "MANY"],
            "logic": ["NOT", "MAYBE", "CAN", "BECAUSE", "IF"],
            "intensifier": ["VERY", "MORE", "LIKE"],
            "time": ["WHEN", "NOW", "BEFORE", "AFTER"],
            "space": ["WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "INSIDE"]
        }
    
    def _load_models(self):
        """Load neural models for semantic analysis."""
        try:
            # Load SBERT for semantic similarity
            logger.info(f"Loading SBERT model: {self.model_name}")
            self.sbert_model = SentenceTransformer(self.model_name)
            
            # Load BERT for contextual embeddings
            logger.info("Loading BERT model for contextual analysis")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            self.encoder_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
            
            # Load spaCy models for multiple languages
            languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
            for lang in languages:
                try:
                    if lang == "en":
                        self.nlp_models[lang] = spacy.load("en_core_web_lg")
                    elif lang == "es":
                        self.nlp_models[lang] = spacy.load("es_core_news_lg")
                    elif lang == "fr":
                        self.nlp_models[lang] = spacy.load("fr_core_news_lg")
                    else:
                        # Use smaller models for other languages
                        self.nlp_models[lang] = spacy.load(f"{lang}_core_web_sm")
                except Exception as e:
                    logger.warning(f"Could not load spaCy model for {lang}: {e}")
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def extract_candidates_from_corpus(self, corpus: List[str], language: str = "en") -> List[PrimeCandidate]:
        """Extract prime candidates from a large corpus."""
        candidates = []
        
        # Extract potential candidates using multiple methods
        lexical_candidates = self._extract_lexical_candidates(corpus, language)
        semantic_candidates = self._extract_semantic_candidates(corpus, language)
        distributional_candidates = self._extract_distributional_candidates(corpus, language)
        
        # Combine and rank candidates
        all_candidates = lexical_candidates + semantic_candidates + distributional_candidates
        
        # Remove duplicates and filter
        unique_candidates = self._deduplicate_candidates(all_candidates)
        filtered_candidates = self._filter_candidates(unique_candidates)
        
        # Score candidates
        for candidate in filtered_candidates:
            candidate.universality_score = self._calculate_universality_score(candidate)
            candidate.confidence = self._calculate_confidence_score(candidate)
            candidates.append(candidate)
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates
    
    def _extract_lexical_candidates(self, corpus: List[str], language: str) -> List[PrimeCandidate]:
        """Extract candidates using lexical patterns."""
        candidates = []
        nlp = self.nlp_models.get(language)
        if not nlp:
            return candidates
        
        # Extract high-frequency words and phrases
        word_freq = Counter()
        phrase_freq = Counter()
        
        for text in corpus:
            doc = nlp(text)
            
            # Single words
            for token in doc:
                if token.is_alpha and len(token.text) > 2:
                    word_freq[token.lemma_.lower()] += 1
            
            # Noun phrases
            for chunk in doc.noun_chunks:
                phrase_freq[chunk.text.lower()] += 1
        
        # Create candidates from high-frequency items
        for word, freq in word_freq.most_common(100):
            if freq > 10:  # Minimum frequency threshold
                candidate = PrimeCandidate(
                    surface_form=word,
                    language=language,
                    semantic_cluster="",
                    frequency=freq,
                    cross_lingual_equivalents=[],
                    semantic_similarity=0.0,
                    universality_score=0.0,
                    confidence=0.0,
                    contexts=self._extract_contexts(word, corpus),
                    proposed_prime=word.upper()
                )
                candidates.append(candidate)
        
        return candidates
    
    def _extract_semantic_candidates(self, corpus: List[str], language: str) -> List[PrimeCandidate]:
        """Extract candidates using semantic similarity to known primes."""
        candidates = []
        
        # Get embeddings for known primes
        prime_embeddings = {}
        for prime in self.known_primes:
            prime_embeddings[prime] = self.sbert_model.encode(prime)
        
        # Extract potential candidates
        candidate_embeddings = {}
        for text in corpus[:1000]:  # Sample for efficiency
            doc = self.nlp_models.get(language, self.nlp_models.get("en"))
            if not doc:
                continue
            
            for token in doc:
                if token.is_alpha and len(token.text) > 2:
                    candidate_embeddings[token.lemma_.lower()] = self.sbert_model.encode(token.text)
        
        # Find semantically similar candidates
        for candidate, embedding in candidate_embeddings.items():
            max_similarity = 0.0
            most_similar_prime = ""
            
            for prime, prime_embedding in prime_embeddings.items():
                similarity = cosine_similarity([embedding], [prime_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_prime = prime
            
            if max_similarity > 0.7:  # High similarity threshold
                candidate_obj = PrimeCandidate(
                    surface_form=candidate,
                    language=language,
                    semantic_cluster=self._get_cluster_for_prime(most_similar_prime),
                    frequency=0,
                    cross_lingual_equivalents=[],
                    semantic_similarity=max_similarity,
                    universality_score=0.0,
                    confidence=0.0,
                    contexts=[],
                    proposed_prime=candidate.upper()
                )
                candidates.append(candidate_obj)
        
        return candidates
    
    def _extract_distributional_candidates(self, corpus: List[str], language: str) -> List[PrimeCandidate]:
        """Extract candidates using distributional semantics."""
        candidates = []
        
        # Build co-occurrence matrix
        cooccurrence = defaultdict(Counter)
        vocab = set()
        
        for text in corpus[:5000]:  # Sample for efficiency
            doc = self.nlp_models.get(language, self.nlp_models.get("en"))
            if not doc:
                continue
            
            words = [token.lemma_.lower() for token in doc if token.is_alpha and len(token.text) > 2]
            vocab.update(words)
            
            # Build co-occurrence within window
            window_size = 5
            for i, word in enumerate(words):
                for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
                    if i != j:
                        cooccurrence[word][words[j]] += 1
        
        # Find distributionally similar words
        for word1 in list(vocab)[:100]:  # Sample for efficiency
            for word2 in list(vocab)[:100]:
                if word1 < word2:  # Avoid duplicates
                    similarity = self._calculate_distributional_similarity(
                        cooccurrence[word1], cooccurrence[word2]
                    )
                    
                    if similarity > 0.8:  # High similarity threshold
                        candidate = PrimeCandidate(
                            surface_form=word1,
                            language=language,
                            semantic_cluster="",
                            frequency=0,
                            cross_lingual_equivalents=[word2],
                            semantic_similarity=similarity,
                            universality_score=0.0,
                            confidence=0.0,
                            contexts=[],
                            proposed_prime=word1.upper()
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _calculate_distributional_similarity(self, cooc1: Counter, cooc2: Counter) -> float:
        """Calculate distributional similarity between two word contexts."""
        if not cooc1 or not cooc2:
            return 0.0
        
        # Get common vocabulary
        vocab = set(cooc1.keys()) | set(cooc2.keys())
        
        # Calculate cosine similarity
        vec1 = [cooc1.get(word, 0) for word in vocab]
        vec2 = [cooc2.get(word, 0) for word in vocab]
        
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def _deduplicate_candidates(self, candidates: List[PrimeCandidate]) -> List[PrimeCandidate]:
        """Remove duplicate candidates."""
        seen = set()
        unique = []
        
        for candidate in candidates:
            key = (candidate.surface_form, candidate.language)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        
        return unique
    
    def _filter_candidates(self, candidates: List[PrimeCandidate]) -> List[PrimeCandidate]:
        """Filter candidates based on quality criteria."""
        filtered = []
        
        for candidate in candidates:
            # Skip if already a known prime
            if candidate.surface_form.upper() in self.known_primes:
                continue
            
            # Skip if too short or too long
            if len(candidate.surface_form) < 2 or len(candidate.surface_form) > 20:
                continue
            
            # Skip if contains numbers or special characters
            if re.search(r'[0-9!@#$%^&*()]', candidate.surface_form):
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def _calculate_universality_score(self, candidate: PrimeCandidate) -> float:
        """Calculate universality score for a candidate."""
        score = 0.0
        
        # Frequency component
        if candidate.frequency > 0:
            score += min(candidate.frequency / 1000, 0.3)  # Max 0.3 for frequency
        
        # Semantic similarity component
        score += candidate.semantic_similarity * 0.4
        
        # Cross-lingual component
        if candidate.cross_lingual_equivalents:
            score += min(len(candidate.cross_lingual_equivalents) / 10, 0.3)
        
        return min(score, 1.0)
    
    def _calculate_confidence_score(self, candidate: PrimeCandidate) -> float:
        """Calculate confidence score for a candidate."""
        confidence = 0.0
        
        # Base confidence from universality
        confidence += candidate.universality_score * 0.6
        
        # Context diversity
        if len(candidate.contexts) > 5:
            confidence += 0.2
        
        # Semantic cluster assignment
        if candidate.semantic_cluster:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _extract_contexts(self, word: str, corpus: List[str]) -> List[str]:
        """Extract contexts for a word."""
        contexts = []
        
        for text in corpus:
            if word.lower() in text.lower():
                # Extract sentence containing the word
                sentences = text.split('.')
                for sentence in sentences:
                    if word.lower() in sentence.lower():
                        contexts.append(sentence.strip())
                        if len(contexts) >= 10:  # Limit contexts
                            break
                if len(contexts) >= 10:
                    break
        
        return contexts
    
    def _get_cluster_for_prime(self, prime: str) -> str:
        """Get semantic cluster for a prime."""
        for cluster, primes in self.semantic_clusters.items():
            if prime in primes:
                return cluster
        return ""

class PrimeDiscoveryPipeline:
    """Pipeline for discovering new NSM primes from large corpora."""
    
    def __init__(self):
        """Initialize the discovery pipeline."""
        self.detector = AdvancedPrimeDetector()
        self.discovery_results = []
    
    def discover_primes_from_corpora(self, corpora: Dict[str, List[str]]) -> PrimeDiscoveryResult:
        """Discover new primes from multiple language corpora."""
        all_candidates = []
        cross_lingual_mappings = {}
        
        # Extract candidates from each corpus
        for language, corpus in corpora.items():
            logger.info(f"Processing corpus for {language} ({len(corpus)} texts)")
            candidates = self.detector.extract_candidates_from_corpus(corpus, language)
            all_candidates.extend(candidates)
            
            # Build cross-lingual mappings
            for candidate in candidates:
                if candidate.surface_form not in cross_lingual_mappings:
                    cross_lingual_mappings[candidate.surface_form] = {}
                cross_lingual_mappings[candidate.surface_form][language] = candidate
        
        # Cluster candidates by semantic similarity
        clusters = self._cluster_candidates(all_candidates)
        
        # Analyze universality
        universality_analysis = self._analyze_universality(all_candidates, cross_lingual_mappings)
        
        # Calculate discovery metrics
        discovery_metrics = self._calculate_discovery_metrics(all_candidates, clusters)
        
        return PrimeDiscoveryResult(
            candidates=all_candidates,
            clusters=clusters,
            universality_analysis=universality_analysis,
            cross_lingual_mappings=cross_lingual_mappings,
            discovery_metrics=discovery_metrics
        )
    
    def _cluster_candidates(self, candidates: List[PrimeCandidate]) -> Dict[str, List[str]]:
        """Cluster candidates by semantic similarity."""
        if not candidates:
            return {}
        
        # Get embeddings for clustering
        embeddings = []
        candidate_texts = []
        
        for candidate in candidates:
            embedding = self.detector.sbert_model.encode(candidate.surface_form)
            embeddings.append(embedding)
            candidate_texts.append(candidate.surface_form)
        
        # Perform clustering
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
        
        # Group by clusters
        clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            if label >= 0:  # Skip noise points
                clusters[f"cluster_{label}"].append(candidate_texts[i])
        
        return dict(clusters)
    
    def _analyze_universality(self, candidates: List[PrimeCandidate], 
                            cross_lingual_mappings: Dict[str, Dict[str, PrimeCandidate]]) -> Dict[str, float]:
        """Analyze universality of candidates across languages."""
        universality_scores = {}
        
        for candidate in candidates:
            surface_form = candidate.surface_form
            if surface_form in cross_lingual_mappings:
                # Count languages where this candidate appears
                language_count = len(cross_lingual_mappings[surface_form])
                universality_scores[surface_form] = language_count / 10.0  # Normalize by expected max
        
        return universality_scores
    
    def _calculate_discovery_metrics(self, candidates: List[PrimeCandidate], 
                                   clusters: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate discovery metrics."""
        return {
            "total_candidates": len(candidates),
            "high_confidence_candidates": len([c for c in candidates if c.confidence > 0.8]),
            "universal_candidates": len([c for c in candidates if c.universality_score > 0.7]),
            "semantic_clusters": len(clusters),
            "average_confidence": np.mean([c.confidence for c in candidates]) if candidates else 0.0,
            "average_universality": np.mean([c.universality_score for c in candidates]) if candidates else 0.0
        }
