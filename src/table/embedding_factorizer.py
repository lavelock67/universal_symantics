#!/usr/bin/env python3
"""
Real Embedding Factorization Module

This module implements learned embedding-based factorization for information primitives,
replacing heuristic approaches with sophisticated vector representations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
import pickle
import json
import os

logger = logging.getLogger(__name__)

class EmbeddingFactorizer:
    """
    Real embedding-based factorization for information primitives.
    
    Uses learned embeddings to identify primitive patterns in text,
    replacing heuristic rule-based approaches with sophisticated
    vector similarity and clustering techniques.
    """
    
    def __init__(self, primitive_table, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding factorizer.
        
        Args:
            primitive_table: The periodic table of primitives
            embedding_model_name: Name of the sentence transformer model to use
        """
        self.primitive_table = primitive_table
        self.embedding_model_name = embedding_model_name
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        self.primitive_embeddings = {}
        self.primitive_examples = {}
        self._load_models()
        self._prepare_primitive_embeddings()
    
    def _load_models(self):
        """Load embedding models."""
        try:
            # Load sentence transformer for semantic embeddings
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer(self.embedding_model_name)
            logger.info(f"✓ Loaded sentence transformer: {self.embedding_model_name}")
        except ImportError:
            logger.warning("Sentence transformers not available, using TF-IDF only")
            self.sentence_transformer = None
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        logger.info("✓ Initialized TF-IDF vectorizer")
    
    def _prepare_primitive_embeddings(self):
        """Prepare embeddings for all primitives based on their examples and descriptions."""
        logger.info("Preparing primitive embeddings...")
        
        for primitive_name, primitive_data in self.primitive_table.primitives.items():
            # Collect examples and descriptions for this primitive
            examples = []
            
            # Add primitive name as example
            examples.append(primitive_name)
            
            # Add primitive description if available
            if isinstance(primitive_data, dict) and 'description' in primitive_data:
                examples.append(primitive_data['description'])
            
            # Add semantic examples based on primitive type
            semantic_examples = self._get_semantic_examples(primitive_name)
            examples.extend(semantic_examples)
            
            # Create embeddings for examples
            if self.sentence_transformer and examples:
                try:
                    embeddings = self.sentence_transformer.encode(examples)
                    # Use mean embedding as representative
                    self.primitive_embeddings[primitive_name] = np.mean(embeddings, axis=0)
                    self.primitive_examples[primitive_name] = examples
                except Exception as e:
                    logger.warning(f"Failed to create embedding for {primitive_name}: {e}")
                    self.primitive_embeddings[primitive_name] = None
            else:
                self.primitive_embeddings[primitive_name] = None
        
        logger.info(f"✓ Prepared embeddings for {len(self.primitive_embeddings)} primitives")
    
    def _get_semantic_examples(self, primitive_name: str) -> List[str]:
        """Get semantic examples for a primitive based on its type."""
        examples = []
        
        if "IsA" in primitive_name:
            examples.extend([
                "is a type of", "is an instance of", "belongs to category",
                "is classified as", "is a kind of", "is a form of"
            ])
        elif "PartOf" in primitive_name:
            examples.extend([
                "is part of", "belongs to", "is a component of",
                "is a member of", "is contained in", "is included in"
            ])
        elif "AtLocation" in primitive_name:
            examples.extend([
                "is located at", "is in", "is at", "is found in",
                "is situated at", "is positioned at"
            ])
        elif "Before" in primitive_name or "After" in primitive_name:
            examples.extend([
                "happens before", "occurs before", "precedes",
                "happens after", "occurs after", "follows"
            ])
        elif "Causes" in primitive_name or "Because" in primitive_name:
            examples.extend([
                "causes", "leads to", "results in", "brings about",
                "because of", "due to", "as a result of"
            ])
        elif "HasProperty" in primitive_name:
            examples.extend([
                "has property", "has characteristic", "has feature",
                "has attribute", "has quality", "has trait"
            ])
        elif "UsedFor" in primitive_name:
            examples.extend([
                "used for", "used to", "serves to", "purpose is",
                "function is", "designed for"
            ])
        elif "SimilarTo" in primitive_name or "Like" in primitive_name:
            examples.extend([
                "similar to", "like", "resembles", "is comparable to",
                "is analogous to", "is equivalent to"
            ])
        elif "DifferentFrom" in primitive_name:
            examples.extend([
                "different from", "differs from", "is distinct from",
                "is unlike", "contrasts with", "is dissimilar to"
            ])
        elif "Not" in primitive_name:
            examples.extend([
                "is not", "does not", "is not a", "is not an",
                "lacks", "is without", "is devoid of"
            ])
        elif "Exist" in primitive_name:
            examples.extend([
                "exists", "is present", "is there", "occurs",
                "can be found", "is available", "is extant"
            ])
        else:
            # Generic examples for unknown primitives
            examples.extend([
                f"involves {primitive_name.lower()}",
                f"relates to {primitive_name.lower()}",
                f"has {primitive_name.lower()} relationship"
            ])
        
        return examples
    
    def factorize_text(self, text: str, top_k: int = 5, similarity_threshold: float | None = None) -> List[Tuple[str, float]]:
        """Factorize text into primitive directions based on semantic similarity.

        Args:
            text: Input text
            top_k: Number of top primitives to return
            similarity_threshold: Minimum similarity score to include. If None, uses env PERIODIC_SIM_THRESHOLD or default 0.3
        """
        # Determine effective threshold
        if similarity_threshold is None:
            try:
                env_thr = float(os.getenv("PERIODIC_SIM_THRESHOLD", "0.3"))
            except ValueError:
                env_thr = 0.3
            similarity_threshold = env_thr
        logger.debug(f"EmbeddingFactorizer using similarity_threshold={similarity_threshold}")
        
        if not text.strip():
            return []
        
        # Get text embedding
        if self.sentence_transformer:
            try:
                text_embedding = self.sentence_transformer.encode([text])[0]
                similarities = {}
                
                # Calculate similarities with all primitives
                for primitive_name, primitive_embedding in self.primitive_embeddings.items():
                    if primitive_embedding is not None:
                        similarity = cosine_similarity(
                            [text_embedding], [primitive_embedding]
                        )[0][0]
                        similarities[primitive_name] = similarity
                
                # Sort by similarity and filter by threshold
                results = [
                    (primitive_name, float(score))
                    for primitive_name, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                    if score >= similarity_threshold
                ][:top_k]
                
                return results
                
            except Exception as e:
                logger.warning(f"Embedding factorization failed: {e}")
                return self._fallback_factorization(text, top_k)
        else:
            return self._fallback_factorization(text, top_k)
    
    def _fallback_factorization(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """Fallback factorization using TF-IDF similarity."""
        try:
            # Create TF-IDF vectors for text and primitive examples
            all_texts = [text]
            primitive_texts = []
            primitive_names = []
            
            for primitive_name, examples in self.primitive_examples.items():
                if examples:
                    # Combine examples into a single text
                    combined_examples = " ".join(examples)
                    all_texts.append(combined_examples)
                    primitive_texts.append(combined_examples)
                    primitive_names.append(primitive_name)
            
            if len(all_texts) < 2:
                return []
            
            # Fit and transform TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            text_vector = tfidf_matrix[0:1]
            primitive_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(text_vector, primitive_vectors)[0]
            
            # Create results
            results = []
            # Lower threshold for TF-IDF can be configured separately
            try:
                tfidf_thr = float(os.getenv("PERIODIC_TFIDF_THRESHOLD", "0.1"))
            except ValueError:
                tfidf_thr = 0.1
            for i, similarity in enumerate(similarities):
                if similarity > tfidf_thr:  # Lower threshold for TF-IDF
                    results.append((primitive_names[i], float(similarity)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.warning(f"Fallback factorization failed: {e}")
            return []
    
    def get_primitive_embedding(self, primitive_name: str) -> Optional[np.ndarray]:
        """Get the embedding for a specific primitive."""
        return self.primitive_embeddings.get(primitive_name)
    
    def get_primitive_examples(self, primitive_name: str) -> List[str]:
        """Get examples for a specific primitive."""
        return self.primitive_examples.get(primitive_name, [])
    
    def save_embeddings(self, filepath: str):
        """Save primitive embeddings to file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            embeddings_dict = {}
            for name, embedding in self.primitive_embeddings.items():
                if embedding is not None:
                    embeddings_dict[name] = embedding.tolist()
                else:
                    embeddings_dict[name] = None
            
            data = {
                "embeddings": embeddings_dict,
                "examples": self.primitive_examples,
                "model_name": self.embedding_model_name
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✓ Saved embeddings to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def load_embeddings(self, filepath: str):
        """Load primitive embeddings from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            embeddings_dict = {}
            for name, embedding_list in data["embeddings"].items():
                if embedding_list is not None:
                    embeddings_dict[name] = np.array(embedding_list)
                else:
                    embeddings_dict[name] = None
            
            self.primitive_embeddings = embeddings_dict
            self.primitive_examples = data["examples"]
            self.embedding_model_name = data.get("model_name", self.embedding_model_name)
            
            logger.info(f"✓ Loaded embeddings from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
    
    def analyze_embedding_quality(self) -> Dict[str, Any]:
        """Analyze the quality of primitive embeddings."""
        analysis = {
            "total_primitives": len(self.primitive_embeddings),
            "primitives_with_embeddings": 0,
            "primitives_without_embeddings": 0,
            "average_examples_per_primitive": 0,
            "embedding_dimensions": 0
        }
        
        total_examples = 0
        embedding_dims = set()
        
        for name, embedding in self.primitive_embeddings.items():
            if embedding is not None:
                analysis["primitives_with_embeddings"] += 1
                embedding_dims.add(embedding.shape[0])
            else:
                analysis["primitives_without_embeddings"] += 1
            
            examples = self.primitive_examples.get(name, [])
            total_examples += len(examples)
        
        if analysis["total_primitives"] > 0:
            analysis["average_examples_per_primitive"] = total_examples / analysis["total_primitives"]
        
        if embedding_dims:
            analysis["embedding_dimensions"] = list(embedding_dims)[0]
        
        return analysis
