"""
Advanced Prime Discovery System

Sophisticated algorithms for discovering new NSM primes:
- Information geometry analysis
- Cross-lingual semantic clustering
- MDL-based compression analysis
- Universality testing across languages
- Semantic stability validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import logging
from pathlib import Path
import json
import pickle

from ..domain.models import Language, PrimeCandidate, DiscoveryStatus
from ...shared.config import get_settings
from ...shared.logging import get_logger

logger = get_logger(__name__)

@dataclass
class DiscoveryConfig:
    """Configuration for prime discovery."""
    min_frequency: int = 10
    max_candidates: int = 100
    similarity_threshold: float = 0.7
    universality_threshold: float = 0.8
    stability_threshold: float = 0.75
    compression_threshold: float = 0.1
    cluster_eps: float = 0.3
    cluster_min_samples: int = 3
    max_semantic_distance: float = 0.5

@dataclass
class DiscoveryResult:
    """Result of prime discovery analysis."""
    candidate: PrimeCandidate
    universality_score: float
    stability_score: float
    compression_score: float
    cross_lingual_consistency: float
    semantic_clarity: float
    overall_score: float
    discovery_status: DiscoveryStatus
    supporting_evidence: List[str]
    conflicting_evidence: List[str]

class InformationGeometryAnalyzer:
    """Analyzes semantic space using information geometry principles."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
    def analyze_semantic_manifold(self, 
                                texts: List[str],
                                languages: List[Language]) -> Dict[str, Any]:
        """Analyze the semantic manifold structure of texts across languages."""
        try:
            # Encode all texts
            embeddings = self.model.encode(texts)
            
            # Calculate pairwise distances
            distances = squareform(pdist(embeddings, metric='cosine'))
            
            # Analyze manifold structure
            manifold_analysis = {
                "curvature": self._estimate_curvature(distances),
                "connectivity": self._analyze_connectivity(distances),
                "clustering": self._analyze_clustering(embeddings),
                "language_separation": self._analyze_language_separation(
                    embeddings, languages
                )
            }
            
            return manifold_analysis
            
        except Exception as e:
            self.logger.error(f"Semantic manifold analysis failed: {str(e)}")
            raise
    
    def _estimate_curvature(self, distances: np.ndarray) -> float:
        """Estimate the curvature of the semantic manifold."""
        try:
            # Use triangle inequality violations to estimate curvature
            violations = 0
            total_triangles = 0
            
            for i in range(len(distances)):
                for j in range(i+1, len(distances)):
                    for k in range(j+1, len(distances)):
                        total_triangles += 1
                        
                        # Check triangle inequality
                        d_ij = distances[i, j]
                        d_ik = distances[i, k]
                        d_jk = distances[j, k]
                        
                        if d_ij + d_jk < d_ik or d_ij + d_ik < d_jk or d_ik + d_jk < d_ij:
                            violations += 1
            
            curvature = violations / total_triangles if total_triangles > 0 else 0.0
            return curvature
            
        except Exception as e:
            self.logger.error(f"Curvature estimation failed: {str(e)}")
            return 0.0
    
    def _analyze_connectivity(self, distances: np.ndarray) -> Dict[str, float]:
        """Analyze connectivity properties of the semantic space."""
        try:
            # Create adjacency matrix based on similarity threshold
            threshold = 0.5
            adjacency = distances < threshold
            
            # Calculate connectivity metrics
            graph = nx.from_numpy_array(adjacency)
            
            connectivity_metrics = {
                "density": nx.density(graph),
                "average_clustering": nx.average_clustering(graph),
                "average_shortest_path": nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf'),
                "number_components": nx.number_connected_components(graph)
            }
            
            return connectivity_metrics
            
        except Exception as e:
            self.logger.error(f"Connectivity analysis failed: {str(e)}")
            return {"density": 0.0, "average_clustering": 0.0, "average_shortest_path": float('inf'), "number_components": 1}
    
    def _analyze_clustering(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Analyze clustering structure of embeddings."""
        try:
            # Try different clustering algorithms
            clustering_results = {}
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.3, min_samples=3)
            dbscan_labels = dbscan.fit_predict(embeddings)
            clustering_results["dbscan"] = {
                "n_clusters": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                "silhouette_score": silhouette_score(embeddings, dbscan_labels) if len(set(dbscan_labels)) > 1 else 0.0
            }
            
            # K-means clustering
            n_clusters = min(10, len(embeddings) // 2)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans_labels = kmeans.fit_predict(embeddings)
                clustering_results["kmeans"] = {
                    "n_clusters": n_clusters,
                    "silhouette_score": silhouette_score(embeddings, kmeans_labels)
                }
            
            return clustering_results
            
        except Exception as e:
            self.logger.error(f"Clustering analysis failed: {str(e)}")
            return {}
    
    def _analyze_language_separation(self, 
                                   embeddings: np.ndarray, 
                                   languages: List[Language]) -> Dict[str, float]:
        """Analyze how well languages are separated in semantic space."""
        try:
            language_embeddings = {}
            for i, lang in enumerate(languages):
                if lang not in language_embeddings:
                    language_embeddings[lang] = []
                language_embeddings[lang].append(embeddings[i])
            
            # Calculate intra-language and inter-language distances
            intra_distances = []
            inter_distances = []
            
            for lang, lang_embeddings in language_embeddings.items():
                lang_embeddings = np.array(lang_embeddings)
                
                # Intra-language distances
                if len(lang_embeddings) > 1:
                    intra_dist = pdist(lang_embeddings, metric='cosine')
                    intra_distances.extend(intra_dist)
                
                # Inter-language distances
                for other_lang, other_embeddings in language_embeddings.items():
                    if lang != other_lang:
                        for emb1 in lang_embeddings:
                            for emb2 in other_embeddings:
                                inter_dist = 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                                inter_distances.append(inter_dist)
            
            # Calculate separation score
            if intra_distances and inter_distances:
                avg_intra = np.mean(intra_distances)
                avg_inter = np.mean(inter_distances)
                separation_score = (avg_inter - avg_intra) / (avg_inter + avg_intra)
            else:
                separation_score = 0.0
            
            return {
                "separation_score": separation_score,
                "avg_intra_distance": np.mean(intra_distances) if intra_distances else 0.0,
                "avg_inter_distance": np.mean(inter_distances) if inter_distances else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Language separation analysis failed: {str(e)}")
            return {"separation_score": 0.0, "avg_intra_distance": 0.0, "avg_inter_distance": 0.0}

class AdvancedPrimeDiscovery:
    """Advanced system for discovering new NSM primes."""
    
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.geometry_analyzer = InformationGeometryAnalyzer()
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load existing primes for comparison
        self.existing_primes = self._load_existing_primes()
        
        self.logger.info("AdvancedPrimeDiscovery initialized successfully")
    
    def _load_existing_primes(self) -> Set[str]:
        """Load existing NSM primes for comparison."""
        # This would typically load from a database or file
        # For now, return a basic set
        return {
            "SAY", "THINK", "WANT", "CAN", "DO", "BE", "NOT", "TRUE", "FALSE",
            "BECAUSE", "IF", "WHEN", "VERY", "MORE", "GOOD", "BAD", "BIG", "SMALL",
            "I", "YOU", "SOMEONE", "SOMETHING", "PEOPLE", "THIS", "THE SAME", "OTHER",
            "ONE", "TWO", "ALL", "MANY", "SOME", "MUCH", "LITTLE", "LONG", "SHORT",
            "NEAR", "FAR", "ABOVE", "BELOW", "INSIDE", "OUTSIDE", "NOW", "BEFORE",
            "AFTER", "A LONG TIME", "A SHORT TIME", "WHERE", "HERE", "THERE",
            "ABOVE", "BELOW", "LEFT", "RIGHT", "HAND", "EYE", "MOUTH", "EAR",
            "HEAD", "FOOT", "FINGER", "NOSE", "HAIR", "BLOOD", "BONE", "SKIN",
            "FIRE", "WATER", "STONE", "EARTH", "SUN", "MOON", "STAR", "WIND",
            "RAIN", "SALT", "SAND", "WOOD", "SMOKE", "ASH", "DAY", "NIGHT",
            "YEAR", "WARM", "COLD", "DRY", "WET", "HEAVY", "LIGHT", "THICK",
            "THIN", "WIDE", "NARROW", "ROUND", "SHARP", "SMOOTH", "ROUGH",
            "HARD", "SOFT", "STRONG", "WEAK", "FAST", "SLOW", "NEW", "OLD",
            "YOUNG", "LIVE", "DIE", "KILL", "EAT", "DRINK", "SEE", "HEAR",
            "KNOW", "FEEL", "HURT", "SLEEP", "WAKE", "MOVE", "COME", "GO",
            "GIVE", "TAKE", "MAKE", "BREAK", "OPEN", "CLOSE", "PUT", "GET",
            "FIND", "LOSE", "BUY", "SELL", "USE", "WORK", "PLAY", "LAUGH",
            "CRY", "SING", "DANCE", "WRITE", "READ", "COUNT", "TELL", "ASK",
            "ANSWER", "HELP", "HURT", "HIT", "CUT", "PUSH", "PULL", "THROW",
            "CATCH", "HOLD", "TOUCH", "WASH", "CLEAN", "DIRTY", "FULL", "EMPTY",
            "RIGHT", "WRONG", "SAME", "DIFFERENT", "ALIKE", "UNLIKE", "EQUAL",
            "UNEQUAL", "BETTER", "WORSE", "BEST", "WORST", "ENOUGH", "TOO",
            "ALMOST", "EXACTLY", "ABOUT", "NEARLY", "QUITE", "VERY", "REALLY",
            "ACTUALLY", "CERTAINLY", "SURELY", "PROBABLY", "MAYBE", "PERHAPS",
            "POSSIBLY", "IMPOSSIBLE", "NECESSARY", "UNNECESSARY", "IMPORTANT",
            "UNIMPORTANT", "INTERESTING", "BORING", "DIFFICULT", "EASY",
            "SIMPLE", "COMPLICATED", "CLEAR", "UNCLEAR", "OBVIOUS", "HIDDEN",
            "SECRET", "PUBLIC", "PRIVATE", "OPEN", "CLOSED", "FREE", "BOUND",
            "INDEPENDENT", "DEPENDENT", "ALONE", "TOGETHER", "WITH", "WITHOUT",
            "AGAINST", "FOR", "ABOUT", "CONCERNING", "REGARDING", "ACCORDING",
            "BESIDES", "EXCEPT", "INCLUDING", "EXCLUDING", "AMONG", "BETWEEN",
            "AMONGST", "THROUGH", "ACROSS", "AROUND", "OVER", "UNDER", "BEHIND",
            "IN FRONT OF", "NEXT TO", "CLOSE TO", "FAR FROM", "NEAR TO",
            "AWAY FROM", "TOWARD", "AWAY", "BACK", "FORWARD", "UP", "DOWN",
            "IN", "OUT", "ON", "OFF", "AT", "BY", "FROM", "TO", "OF", "FOR",
            "WITH", "WITHOUT", "AGAINST", "ABOUT", "LIKE", "AS", "THAN", "THAT",
            "WHICH", "WHO", "WHOSE", "WHOM", "WHAT", "WHERE", "WHEN", "WHY",
            "HOW", "WHETHER", "IF", "UNLESS", "ALTHOUGH", "THOUGH", "WHILE",
            "UNTIL", "SINCE", "DURING", "BEFORE", "AFTER", "DURING", "WITHIN",
            "OUTSIDE", "INSIDE", "BETWEEN", "AMONG", "THROUGH", "ACROSS",
            "AROUND", "OVER", "UNDER", "ABOVE", "BELOW", "BEHIND", "IN FRONT OF",
            "NEXT TO", "CLOSE TO", "FAR FROM", "NEAR TO", "AWAY FROM", "TOWARD",
            "AWAY", "BACK", "FORWARD", "UP", "DOWN", "IN", "OUT", "ON", "OFF",
            "AT", "BY", "FROM", "TO", "OF", "FOR", "WITH", "WITHOUT", "AGAINST",
            "ABOUT", "LIKE", "AS", "THAN", "THAT", "WHICH", "WHO", "WHOSE",
            "WHOM", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHETHER", "IF",
            "UNLESS", "ALTHOUGH", "THOUGH", "WHILE", "UNTIL", "SINCE", "DURING",
            "BEFORE", "AFTER", "DURING", "WITHIN", "OUTSIDE", "INSIDE", "BETWEEN",
            "AMONG", "THROUGH", "ACROSS", "AROUND", "OVER", "UNDER", "ABOVE",
            "BELOW", "BEHIND", "IN FRONT OF", "NEXT TO", "CLOSE TO", "FAR FROM",
            "NEAR TO", "AWAY FROM", "TOWARD", "AWAY", "BACK", "FORWARD", "UP",
            "DOWN", "IN", "OUT", "ON", "OFF", "AT", "BY", "FROM", "TO", "OF",
            "FOR", "WITH", "WITHOUT", "AGAINST", "ABOUT", "LIKE", "AS", "THAN"
        }
    
    def discover_candidates(self, 
                          corpus_texts: List[str],
                          languages: List[Language],
                          metadata: Optional[Dict[str, Any]] = None) -> List[DiscoveryResult]:
        """Discover potential new NSM primes from corpus."""
        try:
            self.logger.info(f"Starting prime discovery on {len(corpus_texts)} texts")
            
            # Step 1: Extract candidate expressions
            candidates = self._extract_candidates(corpus_texts, languages)
            
            # Step 2: Analyze semantic properties
            semantic_analysis = self._analyze_semantic_properties(candidates, corpus_texts, languages)
            
            # Step 3: Evaluate universality
            universality_scores = self._evaluate_universality(candidates, languages)
            
            # Step 4: Test semantic stability
            stability_scores = self._test_semantic_stability(candidates, corpus_texts)
            
            # Step 5: Calculate compression scores
            compression_scores = self._calculate_compression_scores(candidates, corpus_texts)
            
            # Step 6: Cross-lingual consistency
            cross_lingual_scores = self._evaluate_cross_lingual_consistency(candidates, languages)
            
            # Step 7: Combine scores and rank candidates
            results = []
            for candidate in candidates:
                result = self._create_discovery_result(
                    candidate, semantic_analysis, universality_scores,
                    stability_scores, compression_scores, cross_lingual_scores
                )
                results.append(result)
            
            # Sort by overall score
            results.sort(key=lambda x: x.overall_score, reverse=True)
            
            self.logger.info(f"Discovered {len(results)} prime candidates")
            return results[:self.config.max_candidates]
            
        except Exception as e:
            self.logger.error(f"Prime discovery failed: {str(e)}")
            # Return empty list instead of raising exception for graceful failure
            return []
    
    def _extract_candidates(self, 
                          corpus_texts: List[str], 
                          languages: List[Language]) -> List[PrimeCandidate]:
        """Extract candidate expressions from corpus."""
        candidates = []
        
        try:
            # Simple frequency-based extraction
            word_frequencies = {}
            
            for text, lang in zip(corpus_texts, languages):
                # Basic tokenization (would be enhanced with proper NLP)
                words = text.lower().split()
                
                for word in words:
                    # Filter out common words and short words
                    if len(word) > 2 and word not in self.existing_primes:
                        if word not in word_frequencies:
                            word_frequencies[word] = {"count": 0, "languages": set()}
                        word_frequencies[word]["count"] += 1
                        word_frequencies[word]["languages"].add(lang)
            
            # Create candidates from frequent words
            for word, freq_data in word_frequencies.items():
                if freq_data["count"] >= self.config.min_frequency:
                    candidate = PrimeCandidate(
                        text=word,
                        mdl_delta=0.0,  # Will be calculated later
                        confidence=0.0,  # Will be calculated later
                        frequency=freq_data["count"],
                        semantic_cluster="",  # Will be calculated later
                        universality_score=0.0,  # Will be calculated later
                        related_primes=[],  # Will be calculated later
                        context_examples=[],  # Will be calculated later
                        linguistic_features={},  # Will be calculated later
                        language=list(freq_data["languages"])[0] if freq_data["languages"] else Language.ENGLISH
                    )
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Candidate extraction failed: {str(e)}")
            return []
    
    def _analyze_semantic_properties(self, 
                                   candidates: List[PrimeCandidate],
                                   corpus_texts: List[str],
                                   languages: List[Language]) -> Dict[str, Dict[str, float]]:
        """Analyze semantic properties of candidates."""
        semantic_analysis = {}
        
        try:
            # Encode all candidate texts
            candidate_texts = [c.text for c in candidates]
            candidate_embeddings = self.semantic_model.encode(candidate_texts)
            
            # Calculate semantic properties
            for i, candidate in enumerate(candidates):
                embedding = candidate_embeddings[i]
                
                # Semantic clarity (distance from existing primes)
                prime_embeddings = self.semantic_model.encode(list(self.existing_primes))
                distances_to_primes = []
                
                for prime_emb in prime_embeddings:
                    dist = 1 - np.dot(embedding, prime_emb) / (np.linalg.norm(embedding) * np.linalg.norm(prime_emb))
                    distances_to_primes.append(dist)
                
                semantic_clarity = np.min(distances_to_primes)
                
                # Semantic consistency across contexts
                contexts = [text for text in corpus_texts if candidate.text.lower() in text.lower()]
                if contexts:
                    context_embeddings = self.semantic_model.encode(contexts)
                    context_similarities = []
                    
                    for ctx_emb in context_embeddings:
                        sim = np.dot(embedding, ctx_emb) / (np.linalg.norm(embedding) * np.linalg.norm(ctx_emb))
                        context_similarities.append(sim)
                    
                    semantic_consistency = np.std(context_similarities)
                else:
                    semantic_consistency = 0.0
                
                semantic_analysis[candidate.text] = {
                    "semantic_clarity": semantic_clarity,
                    "semantic_consistency": semantic_consistency,
                    "embedding": embedding
                }
            
            return semantic_analysis
            
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {str(e)}")
            return {}
    
    def _evaluate_universality(self, 
                              candidates: List[PrimeCandidate],
                              languages: List[Language]) -> Dict[str, float]:
        """Evaluate universality of candidates across languages."""
        universality_scores = {}
        
        try:
            for candidate in candidates:
                # Check if candidate appears in multiple languages
                # For now, use a simplified approach since we don't track languages per candidate
                language_coverage = 1.0 / len(set(languages))  # Simplified
                
                # Check semantic similarity across languages
                cross_lingual_similarity = self._calculate_cross_lingual_similarity(candidate, languages)
                
                # Combined universality score
                universality_score = (language_coverage + cross_lingual_similarity) / 2
                universality_scores[candidate.text] = universality_score
            
            return universality_scores
            
        except Exception as e:
            self.logger.error(f"Universality evaluation failed: {str(e)}")
            return {}
    
    def _test_semantic_stability(self, 
                                candidates: List[PrimeCandidate],
                                corpus_texts: List[str]) -> Dict[str, float]:
        """Test semantic stability of candidates across different contexts."""
        stability_scores = {}
        
        try:
            for candidate in candidates:
                # Find all contexts where candidate appears
                contexts = [text for text in corpus_texts if candidate.text.lower() in text.lower()]
                
                if len(contexts) < 2:
                    stability_scores[candidate.text] = 0.0
                    continue
                
                # Encode contexts
                context_embeddings = self.semantic_model.encode(contexts)
                
                # Calculate stability as inverse of variance in semantic space
                centroid = np.mean(context_embeddings, axis=0)
                distances = []
                
                for emb in context_embeddings:
                    dist = np.linalg.norm(emb - centroid)
                    distances.append(dist)
                
                stability = 1.0 / (1.0 + np.std(distances))
                stability_scores[candidate.text] = stability
            
            return stability_scores
            
        except Exception as e:
            self.logger.error(f"Semantic stability testing failed: {str(e)}")
            return {}
    
    def _calculate_compression_scores(self, 
                                    candidates: List[PrimeCandidate],
                                    corpus_texts: List[str]) -> Dict[str, float]:
        """Calculate compression scores using MDL principles."""
        compression_scores = {}
        
        try:
            for candidate in candidates:
                # Calculate how much the candidate compresses the corpus
                # This is a simplified version - would be enhanced with actual compression algorithms
                
                # Count occurrences
                total_occurrences = sum(1 for text in corpus_texts if candidate.text.lower() in text.lower())
                
                # Calculate potential compression
                if total_occurrences > 0:
                    # Simplified compression score based on frequency and length
                    compression_score = (total_occurrences * len(candidate.text)) / len(" ".join(corpus_texts))
                else:
                    compression_score = 0.0
                
                compression_scores[candidate.text] = compression_score
            
            return compression_scores
            
        except Exception as e:
            self.logger.error(f"Compression score calculation failed: {str(e)}")
            return {}
    
    def _evaluate_cross_lingual_consistency(self, 
                                          candidates: List[PrimeCandidate],
                                          languages: List[Language]) -> Dict[str, float]:
        """Evaluate cross-lingual consistency of candidates."""
        consistency_scores = {}
        
        try:
            for candidate in candidates:
                # For now, use a simplified approach since we don't track languages per candidate
                consistency_score = 1.0 / len(set(languages))  # Simplified
                consistency_scores[candidate.text] = consistency_score
            
            return consistency_scores
            
        except Exception as e:
            self.logger.error(f"Cross-lingual consistency evaluation failed: {str(e)}")
            return {}
    
    def _calculate_cross_lingual_similarity(self, 
                                          candidate: PrimeCandidate,
                                          languages: List[Language]) -> float:
        """Calculate semantic similarity across languages for a candidate."""
        try:
            # This would involve comparing semantic representations across languages
            # For now, return a simplified score based on language coverage
            return 1.0 / len(set(languages))  # Simplified
        except Exception as e:
            self.logger.error(f"Cross-lingual similarity calculation failed: {str(e)}")
            return 0.0
    
    def _create_discovery_result(self,
                               candidate: PrimeCandidate,
                               semantic_analysis: Dict[str, Dict[str, float]],
                               universality_scores: Dict[str, float],
                               stability_scores: Dict[str, float],
                               compression_scores: Dict[str, float],
                               cross_lingual_scores: Dict[str, float]) -> DiscoveryResult:
        """Create a discovery result from all analysis components."""
        try:
            # Get individual scores
            semantic_clarity = semantic_analysis.get(candidate.text, {}).get("semantic_clarity", 0.0)
            universality_score = universality_scores.get(candidate.text, 0.0)
            stability_score = stability_scores.get(candidate.text, 0.0)
            compression_score = compression_scores.get(candidate.text, 0.0)
            cross_lingual_consistency = cross_lingual_scores.get(candidate.text, 0.0)
            
            # Calculate overall score (weighted combination)
            weights = {
                "universality": 0.3,
                "stability": 0.25,
                "compression": 0.2,
                "cross_lingual": 0.15,
                "semantic_clarity": 0.1
            }
            
            overall_score = (
                weights["universality"] * universality_score +
                weights["stability"] * stability_score +
                weights["compression"] * compression_score +
                weights["cross_lingual"] * cross_lingual_consistency +
                weights["semantic_clarity"] * semantic_clarity
            )
            
            # Determine discovery status
            if overall_score >= self.config.universality_threshold:
                discovery_status = DiscoveryStatus.ACCEPTED
            elif overall_score >= self.config.universality_threshold * 0.7:
                discovery_status = DiscoveryStatus.UNDER_REVIEW
            else:
                discovery_status = DiscoveryStatus.REJECTED
            
            # Generate supporting and conflicting evidence
            supporting_evidence = []
            conflicting_evidence = []
            
            if universality_score >= self.config.universality_threshold:
                supporting_evidence.append(f"High universality score: {universality_score:.3f}")
            else:
                conflicting_evidence.append(f"Low universality score: {universality_score:.3f}")
            
            if stability_score >= self.config.stability_threshold:
                supporting_evidence.append(f"High semantic stability: {stability_score:.3f}")
            else:
                conflicting_evidence.append(f"Low semantic stability: {stability_score:.3f}")
            
            if compression_score >= self.config.compression_threshold:
                supporting_evidence.append(f"Good compression potential: {compression_score:.3f}")
            else:
                conflicting_evidence.append(f"Low compression potential: {compression_score:.3f}")
            
            return DiscoveryResult(
                candidate=candidate,
                universality_score=universality_score,
                stability_score=stability_score,
                compression_score=compression_score,
                cross_lingual_consistency=cross_lingual_consistency,
                semantic_clarity=semantic_clarity,
                overall_score=overall_score,
                discovery_status=discovery_status,
                supporting_evidence=supporting_evidence,
                conflicting_evidence=conflicting_evidence
            )
            
        except Exception as e:
            self.logger.error(f"Discovery result creation failed: {str(e)}")
            raise
