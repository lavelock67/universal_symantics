"""
Semantic Alignment Engine

Cross-lingual semantic validation and alignment:
- Semantic similarity across languages
- Cross-lingual validation of generated text
- Semantic alignment analysis
- Universal concept mapping
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ..domain.models import Language, GenerationResult
from ...shared.config import get_settings
from ...shared.logging import get_logger

logger = get_logger(__name__)

@dataclass
class AlignmentConfig:
    """Configuration for semantic alignment."""
    similarity_threshold: float = 0.7
    alignment_threshold: float = 0.8
    max_alignments: int = 10
    use_multilingual_model: bool = True
    normalize_embeddings: bool = True

@dataclass
class AlignmentResult:
    """Result of semantic alignment analysis."""
    source_text: str
    target_text: str
    source_language: Language
    target_language: Language
    similarity_score: float
    alignment_score: float
    is_aligned: bool
    alignment_confidence: float
    semantic_differences: List[str]
    universal_concepts: List[str]

@dataclass
class CrossLingualValidationResult:
    """Result of cross-lingual validation."""
    original_text: str
    generated_text: str
    source_language: Language
    target_language: Language
    semantic_fidelity: float
    cross_lingual_consistency: float
    validation_score: float
    is_valid: bool
    validation_notes: List[str]
    alignment_details: Dict[str, Any]

class SemanticAlignmentEngine:
    """Engine for semantic alignment across languages."""
    
    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Initialize multilingual model
        if self.config.use_multilingual_model:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.logger.info("SemanticAlignmentEngine initialized successfully")
    
    def align_semantics(self, 
                       source_text: str,
                       target_text: str,
                       source_language: Language,
                       target_language: Language) -> AlignmentResult:
        """Align semantics between source and target texts."""
        try:
            # Encode texts
            source_embedding = self.model.encode([source_text])[0]
            target_embedding = self.model.encode([target_text])[0]
            
            # Normalize embeddings if configured
            if self.config.normalize_embeddings:
                source_embedding = source_embedding / np.linalg.norm(source_embedding)
                target_embedding = target_embedding / np.linalg.norm(target_embedding)
            
            # Calculate similarity
            similarity_score = np.dot(source_embedding, target_embedding)
            
            # Calculate alignment score (more sophisticated than simple similarity)
            alignment_score = self._calculate_alignment_score(source_embedding, target_embedding)
            
            # Determine if aligned
            is_aligned = alignment_score >= self.config.alignment_threshold
            
            # Calculate confidence
            alignment_confidence = min(alignment_score, 1.0)
            
            # Identify semantic differences
            semantic_differences = self._identify_semantic_differences(
                source_text, target_text, source_language, target_language
            )
            
            # Extract universal concepts
            universal_concepts = self._extract_universal_concepts(
                source_text, target_text, source_language, target_language
            )
            
            return AlignmentResult(
                source_text=source_text,
                target_text=target_text,
                source_language=source_language,
                target_language=target_language,
                similarity_score=similarity_score,
                alignment_score=alignment_score,
                is_aligned=is_aligned,
                alignment_confidence=alignment_confidence,
                semantic_differences=semantic_differences,
                universal_concepts=universal_concepts
            )
            
        except Exception as e:
            self.logger.error(f"Semantic alignment failed: {str(e)}")
            raise
    
    def validate_cross_lingual_generation(self, 
                                        original_text: str,
                                        generated_text: str,
                                        source_language: Language,
                                        target_language: Language) -> CrossLingualValidationResult:
        """Validate cross-lingual generation for semantic fidelity."""
        try:
            # Perform semantic alignment
            alignment_result = self.align_semantics(
                original_text, generated_text, source_language, target_language
            )
            
            # Calculate semantic fidelity
            semantic_fidelity = alignment_result.similarity_score
            
            # Calculate cross-lingual consistency
            cross_lingual_consistency = self._calculate_cross_lingual_consistency(
                original_text, generated_text, source_language, target_language
            )
            
            # Combined validation score
            validation_score = (semantic_fidelity + cross_lingual_consistency) / 2
            
            # Determine if valid
            is_valid = validation_score >= self.config.similarity_threshold
            
            # Generate validation notes
            validation_notes = self._generate_validation_notes(
                alignment_result, semantic_fidelity, cross_lingual_consistency
            )
            
            # Prepare alignment details
            alignment_details = {
                "similarity_score": alignment_result.similarity_score,
                "alignment_score": alignment_result.alignment_score,
                "semantic_differences": alignment_result.semantic_differences,
                "universal_concepts": alignment_result.universal_concepts
            }
            
            return CrossLingualValidationResult(
                original_text=original_text,
                generated_text=generated_text,
                source_language=source_language,
                target_language=target_language,
                semantic_fidelity=semantic_fidelity,
                cross_lingual_consistency=cross_lingual_consistency,
                validation_score=validation_score,
                is_valid=is_valid,
                validation_notes=validation_notes,
                alignment_details=alignment_details
            )
            
        except Exception as e:
            self.logger.error(f"Cross-lingual validation failed: {str(e)}")
            raise
    
    def batch_align_semantics(self, 
                            text_pairs: List[Tuple[str, str, Language, Language]]) -> List[AlignmentResult]:
        """Align semantics for multiple text pairs."""
        results = []
        
        try:
            for source_text, target_text, source_lang, target_lang in text_pairs:
                try:
                    result = self.align_semantics(source_text, target_text, source_lang, target_lang)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch alignment failed for pair: {str(e)}")
                    # Create error result
                    error_result = AlignmentResult(
                        source_text=source_text,
                        target_text=target_text,
                        source_language=source_lang,
                        target_language=target_lang,
                        similarity_score=0.0,
                        alignment_score=0.0,
                        is_aligned=False,
                        alignment_confidence=0.0,
                        semantic_differences=["Alignment failed"],
                        universal_concepts=[]
                    )
                    results.append(error_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch semantic alignment failed: {str(e)}")
            raise
    
    def analyze_semantic_alignment(self, 
                                 texts_by_language: Dict[Language, List[str]]) -> Dict[str, Any]:
        """Analyze semantic alignment across multiple languages."""
        try:
            analysis_results = {
                "alignment_scores": {},
                "semantic_differences": {},
                "universal_concepts": [],
                "language_specific_concepts": {},
                "cross_lingual_similarity_matrix": {}
            }
            
            # Encode all texts
            all_texts = []
            text_to_language = {}
            
            for language, texts in texts_by_language.items():
                for text in texts:
                    all_texts.append(text)
                    text_to_language[text] = language
            
            if not all_texts:
                return analysis_results
            
            # Encode all texts
            embeddings = self.model.encode(all_texts)
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(embeddings)
            
            # Analyze alignments
            for i, text1 in enumerate(all_texts):
                lang1 = text_to_language[text1]
                
                for j, text2 in enumerate(all_texts):
                    if i != j:
                        lang2 = text_to_language[text2]
                        
                        # Calculate alignment
                        alignment_result = self.align_semantics(text1, text2, lang1, lang2)
                        
                        # Store results
                        pair_key = f"{lang1.value}_{lang2.value}_{i}_{j}"
                        analysis_results["alignment_scores"][pair_key] = alignment_result.alignment_score
                        analysis_results["semantic_differences"][pair_key] = alignment_result.semantic_differences
                        
                        # Collect universal concepts
                        analysis_results["universal_concepts"].extend(alignment_result.universal_concepts)
            
            # Remove duplicates from universal concepts
            analysis_results["universal_concepts"] = list(set(analysis_results["universal_concepts"]))
            
            # Store similarity matrix
            analysis_results["cross_lingual_similarity_matrix"] = similarity_matrix.tolist()
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Semantic alignment analysis failed: {str(e)}")
            raise
    
    def _calculate_alignment_score(self, 
                                 source_embedding: np.ndarray, 
                                 target_embedding: np.ndarray) -> float:
        """Calculate alignment score between embeddings."""
        try:
            # Basic cosine similarity
            cosine_sim = np.dot(source_embedding, target_embedding)
            
            # Additional alignment factors could be added here
            # For now, use cosine similarity as the alignment score
            alignment_score = cosine_sim
            
            return alignment_score
            
        except Exception as e:
            self.logger.error(f"Alignment score calculation failed: {str(e)}")
            return 0.0
    
    def _identify_semantic_differences(self, 
                                     source_text: str,
                                     target_text: str,
                                     source_language: Language,
                                     target_language: Language) -> List[str]:
        """Identify semantic differences between texts."""
        differences = []
        
        try:
            # This would implement sophisticated semantic difference detection
            # For now, use a simplified approach
            
            # Check for length differences
            source_words = len(source_text.split())
            target_words = len(target_text.split())
            
            if abs(source_words - target_words) > 2:
                differences.append(f"Length difference: {source_words} vs {target_words} words")
            
            # Check for key concept differences
            source_concepts = set(source_text.lower().split())
            target_concepts = set(target_text.lower().split())
            
            missing_concepts = source_concepts - target_concepts
            extra_concepts = target_concepts - source_concepts
            
            if missing_concepts:
                differences.append(f"Missing concepts: {list(missing_concepts)[:3]}")
            
            if extra_concepts:
                differences.append(f"Extra concepts: {list(extra_concepts)[:3]}")
            
            return differences
            
        except Exception as e:
            self.logger.error(f"Semantic difference identification failed: {str(e)}")
            return ["Error identifying differences"]
    
    def _extract_universal_concepts(self, 
                                  source_text: str,
                                  target_text: str,
                                  source_language: Language,
                                  target_language: Language) -> List[str]:
        """Extract universal concepts from text pair."""
        concepts = []
        
        try:
            # This would implement sophisticated universal concept extraction
            # For now, use a simplified approach
            
            # Extract common words (potential universal concepts)
            source_words = set(source_text.lower().split())
            target_words = set(target_text.lower().split())
            
            common_words = source_words & target_words
            
            # Filter for potential universal concepts
            for word in common_words:
                if len(word) > 2 and word not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
                    concepts.append(word)
            
            return concepts[:self.config.max_alignments]
            
        except Exception as e:
            self.logger.error(f"Universal concept extraction failed: {str(e)}")
            return []
    
    def _calculate_cross_lingual_consistency(self, 
                                           original_text: str,
                                           generated_text: str,
                                           source_language: Language,
                                           target_language: Language) -> float:
        """Calculate cross-lingual consistency."""
        try:
            # This would implement sophisticated cross-lingual consistency checking
            # For now, use semantic similarity as a proxy
            
            # Encode texts
            original_embedding = self.model.encode([original_text])[0]
            generated_embedding = self.model.encode([generated_text])[0]
            
            # Calculate similarity
            consistency_score = np.dot(original_embedding, generated_embedding)
            
            return consistency_score
            
        except Exception as e:
            self.logger.error(f"Cross-lingual consistency calculation failed: {str(e)}")
            return 0.0
    
    def _generate_validation_notes(self, 
                                 alignment_result: AlignmentResult,
                                 semantic_fidelity: float,
                                 cross_lingual_consistency: float) -> List[str]:
        """Generate validation notes based on alignment results."""
        notes = []
        
        try:
            if semantic_fidelity < 0.6:
                notes.append("Low semantic fidelity detected")
            
            if cross_lingual_consistency < 0.6:
                notes.append("Low cross-lingual consistency")
            
            if alignment_result.semantic_differences:
                notes.append(f"Semantic differences: {len(alignment_result.semantic_differences)} issues found")
            
            if alignment_result.universal_concepts:
                notes.append(f"Universal concepts identified: {len(alignment_result.universal_concepts)}")
            
            if alignment_result.alignment_score >= 0.9:
                notes.append("Excellent semantic alignment")
            elif alignment_result.alignment_score >= 0.7:
                notes.append("Good semantic alignment")
            else:
                notes.append("Poor semantic alignment")
            
            return notes
            
        except Exception as e:
            self.logger.error(f"Validation note generation failed: {str(e)}")
            return ["Error generating validation notes"]

class CrossLingualValidator:
    """Specialized validator for cross-lingual semantic validation."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.alignment_engine = SemanticAlignmentEngine()
    
    def validate_generation(self, 
                          generation_result: GenerationResult,
                          original_text: str,
                          source_language: Language) -> CrossLingualValidationResult:
        """Validate a generation result for cross-lingual semantic fidelity."""
        try:
            return self.alignment_engine.validate_cross_lingual_generation(
                original_text,
                generation_result.generated_text,
                source_language,
                generation_result.target_language
            )
            
        except Exception as e:
            self.logger.error(f"Cross-lingual validation failed: {str(e)}")
            raise
    
    def batch_validate_generations(self, 
                                 generation_results: List[GenerationResult],
                                 original_texts: List[str],
                                 source_languages: List[Language]) -> List[CrossLingualValidationResult]:
        """Validate multiple generation results."""
        results = []
        
        try:
            for gen_result, original_text, source_lang in zip(generation_results, original_texts, source_languages):
                try:
                    validation_result = self.validate_generation(gen_result, original_text, source_lang)
                    results.append(validation_result)
                except Exception as e:
                    self.logger.error(f"Batch validation failed for generation: {str(e)}")
                    # Create error result
                    error_result = CrossLingualValidationResult(
                        original_text=original_text,
                        generated_text=gen_result.generated_text if gen_result else "",
                        source_language=source_lang,
                        target_language=gen_result.target_language if gen_result else Language.ENGLISH,
                        semantic_fidelity=0.0,
                        cross_lingual_consistency=0.0,
                        validation_score=0.0,
                        is_valid=False,
                        validation_notes=[f"Validation error: {str(e)}"],
                        alignment_details={}
                    )
                    results.append(error_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch cross-lingual validation failed: {str(e)}")
            raise
