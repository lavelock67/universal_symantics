"""Algebra for composing and manipulating information primitives.

This module implements the mathematical operations on primitives including
composition, factorization, simplification, and the difference operator (Δ)
that captures "differences that make a difference."
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import os
import json

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

from .schema import CompositionRule, PeriodicTable, Primitive, PrimitiveSignature, PrimitiveCategory

# Lazy spaCy pipelines for dynamic UD detection
_SPACY_PIPES: Dict[str, Any] = {}

def _load_spacy(code: str):
	try:
		import spacy  # type: ignore
		models = {
			"en": "en_core_web_sm",
			"es": "es_core_news_sm",
			"fr": "fr_core_news_sm",
		}
		model = models.get(code)
		if model is None:
			return None
		if code not in _SPACY_PIPES:
			try:
				_SPACY_PIPES[code] = spacy.load(model)
			except Exception:
				_SPACY_PIPES[code] = None
		return _SPACY_PIPES.get(code)
	except Exception:
		return None


@dataclass
class PrimitiveComposition:
    """Result of composing two or more primitives.
    
    Attributes:
        primitives: List of primitives in the composition
        result: The resulting primitive or composition pattern
        confidence: Confidence score for the composition
        metadata: Additional information about the composition
    """
    
    primitives: List[Primitive]
    result: Union[Primitive, List[Primitive]]
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class PrimitiveAlgebra:
    """Algebra for manipulating information primitives.
    
    This class provides methods for composing primitives, factoring patterns
    into primitives, and computing differences between states.
    """
    
    def __init__(self, periodic_table: PeriodicTable):
        """Initialize the algebra with a periodic table.
        
        Args:
            periodic_table: The periodic table containing primitives and rules
        """
        self.table = periodic_table
        self._composition_cache: Dict[Tuple[str, str], PrimitiveComposition] = {}
        self._factorization_cache: Dict[str, List[Primitive]] = {}
        # Lazy-loaded distant supervision classifiers (if available)
        self._ds_cls_loaded: bool = False
        self._ds_classifiers: Dict[str, Dict[str, Any]] = {}
        # Vec2Vec backoff cache (TF-IDF based)
        self._vec_ready: bool = False
        self._vec_names: List[str] = []
        self._vec_vectorizer: Any = None
        self._vec_matrix: Any = None
        # SBERT backoff cache
        self._sbert_ready: bool = False
        self._sbert_names: List[str] = []
        self._sbert_model: Any = None
        self._sbert_matrix: Any = None

    def _load_ds_classifiers(self) -> None:
        """Lazy-load distant supervision classifiers if present on disk."""
        if self._ds_cls_loaded:
            return
        self._ds_cls_loaded = True
        try:
            import os
            from pickle import load  # noqa: S403 (trusted local file)
            model_path = os.environ.get("PRIMITIVE_DS_MODEL", "models/primitive_detectors.pkl")
            with open(model_path, "rb") as fh:
                payload = load(fh)
            # Expect mapping name -> {name, vectorizer, model}
            if isinstance(payload, dict):
                self._ds_classifiers = payload
        except Exception:
            self._ds_classifiers = {}

    def _predict_primitives_ds(self, text: str, threshold: float = None) -> List[str]:
        """Predict primitives using a downstream classifier score threshold.

        Returns a list of primitive names with probability >= threshold.
        """
        if threshold is None:
            try:
                threshold = float(os.getenv('PERIODIC_DS_THRESHOLD', '0.6'))
            except ValueError:
                threshold = 0.6
        self._load_ds_classifiers()
        if not self._ds_classifiers:
            return []
        detected: List[str] = []
        for name, obj in self._ds_classifiers.items():
            vec = obj.get("vectorizer")
            mdl = obj.get("model")
            if vec is None or mdl is None:
                continue
            try:
                X = vec.transform([text])
                if hasattr(mdl, "predict_proba"):
                    proba = mdl.predict_proba(X)[0, 1]
                    if float(proba) >= threshold:
                        detected.append(name)
                else:
                    pred = mdl.predict(X)[0]
                    if int(pred) == 1:
                        detected.append(name)
            except Exception:
                continue
        return detected
    
    def compose(self, a: Primitive, b: Primitive) -> PrimitiveComposition:
        """Compose two primitives according to the algebra rules.
        
        Args:
            a: First primitive
            b: Second primitive
            
        Returns:
            Composition result with confidence score
            
        Raises:
            ValueError: If composition is not possible
        """
        # Check cache first
        cache_key = (a.name, b.name)
        if cache_key in self._composition_cache:
            return self._composition_cache[cache_key]
        
        # Check if composition is possible
        if not a.can_compose_with(b):
            raise ValueError(f"Cannot compose {a.name} with {b.name}")
        
        # Look for explicit composition rule
        rule = self.table.find_composition_rule(a.name, b.name)
        if rule:
            result = self._apply_composition_rule(rule, a, b)
            composition = PrimitiveComposition(
                primitives=[a, b],
                result=result,
                confidence=1.0,
                metadata={"rule": rule.description},
            )
        else:
            # Try to infer composition based on signatures
            result = self._infer_composition(a, b)
            composition = PrimitiveComposition(
                primitives=[a, b],
                result=result,
                confidence=0.7,  # Lower confidence for inferred compositions
                metadata={"inferred": True},
            )
        
        # Cache the result
        self._composition_cache[cache_key] = composition
        return composition
    
    def _apply_composition_rule(
        self, rule: CompositionRule, a: Primitive, b: Primitive
    ) -> Union[Primitive, List[Primitive]]:
        """Apply a composition rule to two primitives.
        
        Args:
            rule: The composition rule to apply
            a: First primitive
            b: Second primitive
            
        Returns:
            Result of applying the rule
        """
        if isinstance(rule.result, str):
            # Result is a single primitive name
            result_primitive = self.table.get_primitive(rule.result)
            if result_primitive:
                return result_primitive
            else:
                # Create a placeholder primitive
                return Primitive(
                    name=rule.result,
                    category=a.category,  # Default to first primitive's category
                    signature=PrimitiveSignature(arity=2),
                    description=f"Composition of {a.name} and {b.name}",
                )
        else:
            # Result is a list of primitive names
            result_primitives = []
            for name in rule.result:
                primitive = self.table.get_primitive(name)
                if primitive:
                    result_primitives.append(primitive)
                else:
                    # Create placeholder
                    result_primitives.append(
                        Primitive(
                            name=name,
                            category=a.category,
                            signature=PrimitiveSignature(arity=1),
                            description=f"Component of {a.name} + {b.name}",
                        )
                    )
            return result_primitives
    
    def _infer_composition(self, a: Primitive, b: Primitive) -> Union[Primitive, List[Primitive]]:
        """Infer composition when no explicit rule exists.
        
        Args:
            a: First primitive
            b: Second primitive
            
        Returns:
            Inferred composition result
        """
        # Simple heuristics for composition inference
        
        # If both have same category, try to find a more general primitive
        if a.category == b.category:
            category_primitives = self.table.get_primitives_by_category(a.category)
            # Look for a primitive that might represent the combination
            for p in category_primitives:
                if p.arity >= max(a.arity, b.arity):
                    return p
        
        # If one is neutral for the other, return the non-neutral one
        if a.neutral == b.name:
            return a
        if b.neutral == a.name:
            return b
        
        # Default: return both as a composition
        return [a, b]
    
    def factor(self, pattern: Any) -> List[Primitive]:
        """Factor a pattern into constituent primitives.
        
        Args:
            pattern: The pattern to factor (could be text, embedding, etc.)
            
        Returns:
            List of primitives that compose to form the pattern
        """
        # Check cache first
        pattern_hash = self._hash_pattern(pattern)
        if pattern_hash in self._factorization_cache:
            return self._factorization_cache[pattern_hash]
        
        if isinstance(pattern, str):
            result = self._factor_text(pattern)
        elif isinstance(pattern, np.ndarray):
            result = self._factor_embedding(pattern)
        elif isinstance(pattern, list):
            result = self._factor_sequence(pattern)
        else:
            result = self._factor_generic(pattern)
        
        # Cache the result
        self._factorization_cache[pattern_hash] = result
        return result
    
    def _hash_pattern(self, pattern: Any) -> str:
        """Create a hash of a pattern for caching.
        
        Args:
            pattern: The pattern to hash
            
        Returns:
            String hash of the pattern
        """
        if isinstance(pattern, str):
            return f"str:{hash(pattern)}"
        elif isinstance(pattern, np.ndarray):
            return f"array:{hash(pattern.tobytes())}"
        elif isinstance(pattern, list):
            return f"list:{hash(tuple(pattern))}"
        else:
            return f"generic:{hash(str(pattern))}"
    
    def _factor_text(self, text: str) -> List[Primitive]:
        """Factor text into primitives.
        
        Args:
            text: Text to factor
            
        Returns:
            List of primitives
        """
        # Simple keyword-based factorization
        primitives = []
        text_lower = text.lower()
        
        # Look for primitive names in the text
        for primitive in self.table.primitives.values():
            if primitive.name.lower() in text_lower:
                primitives.append(primitive)
        
        # If no direct matches, try to infer from content
        if not primitives:
            primitives = self._infer_primitives_from_text(text)
        
        return primitives
    
    def _factor_embedding(self, embedding: np.ndarray) -> List[Primitive]:
        """Factor an embedding vector into primitives using learned mappings.
        
        Args:
            embedding: Embedding vector to factor
            
        Returns:
            List of primitives
        """
        try:
            from .embedding_factorizer import EmbeddingFactorizer
            
            # Initialize embedding factorizer if not already done
            if not hasattr(self, '_embedding_factorizer'):
                self._embedding_factorizer = EmbeddingFactorizer(self.table)
            
            # Use env-configured embedding factorizer similarity threshold
            results = self._embedding_factorizer.factorize_text(text, top_k=3, similarity_threshold=None)
            for primitive_name, similarity_score in results:
                primitive = self.table.get_primitive(primitive_name)
                if primitive and primitive not in primitives:
                    primitives.append(primitive)
                    
        except Exception as e:
            logger.warning(f"Embedding factorization failed: {e}")
            return self._factor_embedding_heuristic(embedding)
    
    def _factor_embedding_heuristic(self, embedding: np.ndarray) -> List[Primitive]:
        """Heuristic fallback for embedding factorization."""
        useful_primitives = []
        
        # Get primitives that are likely to be semantically useful
        for primitive in self.table.primitives.values():
            # Prefer primitives with clear semantic content
            if (primitive.name in ['IsA', 'PartOf', 'AtLocation', 'Before', 'After', 
                                 'Causes', 'SimilarTo', 'DifferentFrom', 'HasProperty'] or
                len(primitive.name) > 8):  # Longer names tend to be more specific
                useful_primitives.append(primitive)
                if len(useful_primitives) >= 3:  # Limit to top 3
                    break
        
        return useful_primitives
    
    def _factor_sequence(self, sequence: List[Any]) -> List[Primitive]:
        """Factor a sequence into primitives.
        
        Args:
            sequence: Sequence to factor
            
        Returns:
            List of primitives
        """
        primitives = []
        
        # Factor each element and combine
        for item in sequence:
            item_primitives = self.factor(item)
            primitives.extend(item_primitives)
        
        # Add sequence-specific primitives only if meaningful
        if len(sequence) > 1 and any(primitives):
            # Get first available category or use structural as default
            category = next(iter(self.table.categories), PrimitiveCategory.STRUCTURAL)
            sequence_primitive = Primitive(
                name="Sequence",
                category=category,
                signature=PrimitiveSignature(arity=len(sequence)),
                description="Sequence of elements",
            )
            primitives.append(sequence_primitive)
        
        return primitives
    
    def _factor_generic(self, pattern: Any) -> List[Primitive]:
        """Factor a generic pattern into primitives.
        
        Args:
            pattern: Generic pattern to factor
            
        Returns:
            List of primitives
        """
        # No generic placeholder; return empty when unknown
        return []
    
    def _infer_primitives_from_text(self, text: str) -> List[Primitive]:
        """Infer primitives from text using multiple detection strategies.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected primitives
        """
        primitives = []
        
        # Strategy 1: Direct pattern detection
        try:
            from ..detect.text_detectors import detect_primitives_in_text
            from ..detect.srl_ud_detectors import detect_primitives_structured, detect_primitives_dep
            
            # Use regex-based detection
            regex_names = detect_primitives_in_text(text)
            for name in regex_names:
                primitive = self.table.get_primitive(name)
                if primitive and primitive not in primitives:
                    primitives.append(primitive)
            
            # Use structured detection (SRL/UD)
            structured_results = detect_primitives_structured(text)
            for result in structured_results:
                if isinstance(result, dict) and 'name' in result:
                    name = result['name']
                elif isinstance(result, str):
                    name = result
                else:
                    continue
                primitive = self.table.get_primitive(name)
                if primitive and primitive not in primitives:
                    primitives.append(primitive)
            
            # Use dependency-based detection
            dep_names = detect_primitives_dep(text)
            for name in dep_names:
                primitive = self.table.get_primitive(name)
                if primitive and primitive not in primitives:
                    primitives.append(primitive)
                    
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
        
        # Strategy 2: Embedding-based factorization
        if len(primitives) < 2:  # If we didn't find enough patterns
            try:
                from .embedding_factorizer import EmbeddingFactorizer
                
                # Initialize embedding factorizer if not already done
                if not hasattr(self, '_embedding_factorizer'):
                    self._embedding_factorizer = EmbeddingFactorizer(self.table)
                
                # Use embedding-based factorization
                results = self._embedding_factorizer.factorize_text(text, top_k=3, similarity_threshold=None)
                for primitive_name, similarity_score in results:
                    primitive = self.table.get_primitive(primitive_name)
                    if primitive and primitive not in primitives:
                        primitives.append(primitive)
                        
            except Exception as e:
                logger.warning(f"Embedding factorization failed: {e}")
        
        # Strategy 3: Semantic similarity backoff
        if len(primitives) < 2:  # If we still didn't find enough patterns
            try:
                # Use TF-IDF similarity
                tfidf_names = self._vec2vec_backoff(text)
                for name in tfidf_names:
                    primitive = self.table.get_primitive(name)
                    if primitive and primitive not in primitives:
                        primitives.append(primitive)
                        
                # Use SBERT similarity if available
                if len(primitives) < 3:
                    sbert_names = self._sbert_backoff(text)
                    for name in sbert_names:
                        primitive = self.table.get_primitive(name)
                        if primitive and primitive not in primitives:
                            primitives.append(primitive)
                            
            except Exception as e:
                logger.warning(f"Similarity backoff failed: {e}")
        
        # Strategy 4: Distant supervision backoff
        if len(primitives) < 1:  # If we still have nothing
            try:
                ds_names = self._predict_primitives_ds(text)
                for name in ds_names:
                    primitive = self.table.get_primitive(name)
                    if primitive and primitive not in primitives:
                        primitives.append(primitive)
            except Exception as e:
                logger.warning(f"Distant supervision backoff failed: {e}")
        
        # Strategy 5: Fallback to high-value primitives
        if not primitives:
            # Return a few high-value primitives that are likely to be useful
            fallback_primitives = ['IsA', 'PartOf', 'AtLocation', 'Before', 'After']
            for name in fallback_primitives:
                primitive = self.table.get_primitive(name)
                if primitive:
                    primitives.append(primitive)
                    if len(primitives) >= 2:
                        break
        
        return primitives

    def _ensure_vec_backoff(self) -> None:
        """Initialize TF-IDF prototypes for primitives if not ready."""
        if self._vec_ready:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        # Build prototypes from primitive metadata
        protos: List[str] = []
        names: List[str] = []
        for prim in self.table.primitives.values():
            pieces: List[str] = [prim.name]
            if getattr(prim, "description", ""):
                pieces.append(prim.description)
            if getattr(prim, "examples", None):
                try:
                    ex = prim.examples[0]
                    if isinstance(ex, str):
                        pieces.append(ex)
                except Exception:
                    pass
            protos.append(". ".join(pieces))
            names.append(prim.name)
        if not protos:
            self._vec_ready = True
            return
        vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        try:
            mat = vec.fit_transform(protos)
        except Exception:
            self._vec_ready = True
            return
        self._vec_vectorizer = vec
        self._vec_matrix = mat
        self._vec_names = names
        self._vec_ready = True

    def _vec2vec_backoff(self, text: str) -> List[str]:
        """Return top candidate primitive names by TF-IDF cosine nearness."""
        self._ensure_vec_backoff()
        if not self._vec_vectorizer or self._vec_matrix is None or not self._vec_names:
            return []
        try:
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
            import numpy as _np
            q = self._vec_vectorizer.transform([text])
            sims = cosine_similarity(q, self._vec_matrix)[0]
            # Consider top-2
            order = _np.argsort(-sims)[:2]
            names: List[str] = []
            # legacy internal threshold, now env-configurable
            try:
                thresh = float(os.getenv('PERIODIC_INTERNAL_THRESHOLD', '0.20'))
            except ValueError:
                thresh = 0.20
            for idx in order:
                score = float(sims[int(idx)])
                if score >= thresh:
                    names.append(self._vec_names[int(idx)])
            return names
        except Exception:
            return []

    def _ensure_sbert(self) -> None:
        """Initialize SBERT model and primitive prototypes if not ready."""
        if self._sbert_ready:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            self._sbert_ready = True
            return
        names: List[str] = []
        texts: List[str] = []
        for prim in self.table.primitives.values():
            names.append(prim.name)
            parts = [prim.name]
            if getattr(prim, "description", ""):
                parts.append(prim.description)
            texts.append(". ".join(parts))
        if not texts:
            self._sbert_ready = True
            return
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        except Exception:
            self._sbert_ready = True
            return
        self._sbert_model = model
        self._sbert_matrix = embs
        self._sbert_names = names
        self._sbert_ready = True

    def _sbert_backoff(self, text: str) -> List[str]:
        """Return top candidate primitive by SBERT cosine similarity (dot with normalized vecs)."""
        self._ensure_sbert()
        if not self._sbert_model or self._sbert_matrix is None or not self._sbert_names:
            return []
        try:
            q = self._sbert_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
            sims = self._sbert_matrix @ q
            idx = int(np.argmax(sims))
            score = float(sims[idx])
            if score >= 0.35:
                return [self._sbert_names[idx]]
            return []
        except Exception:
            return []
    
    def delta(self, prior_state: Any, new_state: Any) -> List[Primitive]:
        """Compute the difference operator (Δ) between states.
        
        The difference operator captures only the "differences that make a difference"
        between the prior and new states.
        
        Args:
            prior_state: The prior state
            new_state: The new state
            
        Returns:
            List of primitives representing the differences
        """
        if prior_state == new_state:
            return []  # No difference
        
        # Factor both states
        prior_primitives_list = self.factor(prior_state)
        new_primitives_list = self.factor(new_state)

        # Safely convert to sets using names to avoid unhashable dataclass contents
        prior_names = {p.name for p in prior_primitives_list}
        new_names = {p.name for p in new_primitives_list}
        
        # Find differences
        added_names = new_names - prior_names
        removed_names = prior_names - new_names
        
        # Create difference primitives
        differences = []
        
        # Map names back to primitive metadata (fallback to informational)
        name_to_primitive = {p.name: p for p in (prior_primitives_list + new_primitives_list)}

        for name in added_names:
            base = name_to_primitive.get(name)
            if base is None:
                continue
            diff_primitive = Primitive(
                name=f"Add_{name}",
                category=base.category,
                signature=base.signature,
                description=f"Addition of {name}",
            )
            differences.append(diff_primitive)
        
        for name in removed_names:
            base = name_to_primitive.get(name)
            if base is None:
                continue
            diff_primitive = Primitive(
                name=f"Remove_{name}",
                category=base.category,
                signature=base.signature,
                description=f"Removal of {name}",
            )
            differences.append(diff_primitive)
        
        return differences
    
    def simplify(self, composition: List[Primitive]) -> List[Primitive]:
        """Simplify a composition of primitives.
        
        Args:
            composition: List of primitives to simplify
            
        Returns:
            Simplified list of primitives
        """
        if len(composition) <= 1:
            return composition
        
        simplified = []
        i = 0
        
        while i < len(composition):
            if i + 1 < len(composition):
                # Try to compose adjacent primitives
                try:
                    comp_result = self.compose(composition[i], composition[i + 1])
                    if isinstance(comp_result.result, Primitive):
                        simplified.append(comp_result.result)
                        i += 2  # Skip both composed primitives
                    else:
                        # Composition resulted in multiple primitives
                        simplified.append(composition[i])
                        i += 1
                except ValueError:
                    # Cannot compose, keep both
                    simplified.append(composition[i])
                    i += 1
            else:
                # Last primitive
                simplified.append(composition[i])
                i += 1
        
        # Recursively simplify if changes were made
        if len(simplified) < len(composition):
            return self.simplify(simplified)
        
        return simplified
    
    def closure_check(self, sample_size: int = 100) -> float:
        """Check closure under composition.
        
        Args:
            sample_size: Number of random primitive pairs to test
            
        Returns:
            Fraction of successful compositions
        """
        primitives = list(self.table.primitives.values())
        if len(primitives) < 2:
            return 1.0
        
        successful = 0
        total = 0
        
        for _ in range(sample_size):
            # Sample random pair
            a, b = np.random.choice(primitives, 2, replace=False)
            
            try:
                self.compose(a, b)
                successful += 1
            except ValueError:
                pass
            
            total += 1
        
        return successful / total if total > 0 else 0.0
    
    def find_equivalences(self) -> List[Tuple[List[Primitive], List[Primitive]]]:
        """Find equivalent compositions of primitives.
        
        Returns:
            List of (composition1, composition2) pairs that are equivalent
        """
        equivalences = []
        primitives = list(self.table.primitives.values())
        
        # Check for inverse pairs
        inverse_pairs = self.table.get_inverse_pairs()
        for a_name, b_name in inverse_pairs:
            a = self.table.get_primitive(a_name)
            b = self.table.get_primitive(b_name)
            if a and b:
                equivalences.append(([a, b], []))  # Composition with inverse = neutral
        
        # Check for neutral elements
        for primitive in primitives:
            if primitive.neutral:
                neutral = self.table.get_primitive(primitive.neutral)
                if neutral:
                    equivalences.append(([primitive, neutral], [primitive]))
                    equivalences.append(([neutral, primitive], [primitive]))
        
        return equivalences
