#!/usr/bin/env python3
"""
Enhanced NSM Grammar Molecules System.

This script implements a comprehensive NSM grammar molecules system to tighten
the NSM legality micro-grammar and create a curated molecule registry for
improved translation quality and explication generation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import time
from collections import defaultdict, Counter
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.nsm.translate import NSMTranslator
    from src.nsm.explicator import NSMExplicator
    from src.nsm.enhanced_explicator import EnhancedNSMExplicator
    from src.table.schema import PeriodicTable
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    exit(1)


def convert_numpy_types(obj):
    """Convert numpy types and other non-serializable types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        # Convert tuple keys to strings
        converted_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                # Convert tuple key to string representation
                str_key = f"{key[0]}_{key[1]}" if len(key) == 2 else str(key)
                converted_dict[str_key] = convert_numpy_types(value)
            else:
                converted_dict[str(key)] = convert_numpy_types(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(obj.__dict__)
    else:
        return obj


class NSMGrammarMolecule:
    """Represents an NSM grammar molecule with structural and semantic properties."""
    
    def __init__(self, name: str, structure: str, primitives: List[str], 
                 language: str, legality_score: float = 0.0):
        """Initialize an NSM grammar molecule."""
        self.name = name
        self.structure = structure
        self.primitives = primitives
        self.language = language
        self.legality_score = legality_score
        self.usage_count = 0
        self.semantic_coherence = 0.0
        self.cross_language_consistency = 0.0
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert molecule to dictionary."""
        return {
            'name': self.name,
            'structure': self.structure,
            'primitives': self.primitives,
            'language': self.language,
            'legality_score': self.legality_score,
            'usage_count': self.usage_count,
            'semantic_coherence': self.semantic_coherence,
            'cross_language_consistency': self.cross_language_consistency,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NSMGrammarMolecule':
        """Create molecule from dictionary."""
        molecule = cls(
            name=data['name'],
            structure=data['structure'],
            primitives=data['primitives'],
            language=data['language'],
            legality_score=data.get('legality_score', 0.0)
        )
        molecule.usage_count = data.get('usage_count', 0)
        molecule.semantic_coherence = data.get('semantic_coherence', 0.0)
        molecule.cross_language_consistency = data.get('cross_language_consistency', 0.0)
        molecule.metadata = data.get('metadata', {})
        return molecule


class NSMGrammarMoleculeRegistry:
    """Registry for NSM grammar molecules with curation and validation."""
    
    def __init__(self):
        """Initialize the NSM grammar molecule registry."""
        self.molecules: Dict[str, NSMGrammarMolecule] = {}
        self.primitive_patterns: Dict[str, List[str]] = {}
        self.structural_templates: Dict[str, List[str]] = {}
        self.language_molecules: Dict[str, List[str]] = defaultdict(list)
        
        # Grammar validation rules
        self.grammar_rules = {
            'structural': {
                'min_primitives': 1,
                'max_primitives': 5,
                'required_connectors': ['AND', 'OR', 'IF', 'BECAUSE'],
                'forbidden_patterns': ['AND AND', 'OR OR', 'IF IF']
            },
            'semantic': {
                'min_coherence': 0.6,
                'max_ambiguity': 0.3,
                'required_semantic_roles': ['agent', 'patient', 'location', 'time']
            },
            'cross_language': {
                'min_consistency': 0.7,
                'required_languages': ['en', 'es', 'fr']
            }
        }
        
        self._load_base_molecules()
    
    def _load_base_molecules(self):
        """Load base NSM grammar molecules."""
        base_molecules = [
            # Basic property molecules
            NSMGrammarMolecule(
                name="has_property",
                structure="X has property Y",
                primitives=["HasProperty"],
                language="en",
                legality_score=0.9
            ),
            NSMGrammarMolecule(
                name="tiene_propiedad",
                structure="X tiene la propiedad Y",
                primitives=["HasProperty"],
                language="es",
                legality_score=0.9
            ),
            NSMGrammarMolecule(
                name="a_propriete",
                structure="X a la propriété Y",
                primitives=["HasProperty"],
                language="fr",
                legality_score=0.9
            ),
            
            # Location molecules
            NSMGrammarMolecule(
                name="at_location",
                structure="X is at Y",
                primitives=["AtLocation"],
                language="en",
                legality_score=0.95
            ),
            NSMGrammarMolecule(
                name="esta_en",
                structure="X está en Y",
                primitives=["AtLocation"],
                language="es",
                legality_score=0.95
            ),
            NSMGrammarMolecule(
                name="est_a",
                structure="X est à Y",
                primitives=["AtLocation"],
                language="fr",
                legality_score=0.95
            ),
            
            # Similarity molecules
            NSMGrammarMolecule(
                name="similar_to",
                structure="X is similar to Y",
                primitives=["SimilarTo"],
                language="en",
                legality_score=0.85
            ),
            NSMGrammarMolecule(
                name="similar_a",
                structure="X es similar a Y",
                primitives=["SimilarTo"],
                language="es",
                legality_score=0.85
            ),
            NSMGrammarMolecule(
                name="similaire_a",
                structure="X est similaire à Y",
                primitives=["SimilarTo"],
                language="fr",
                legality_score=0.85
            ),
            
            # Purpose molecules
            NSMGrammarMolecule(
                name="used_for",
                structure="X is used for Y",
                primitives=["UsedFor"],
                language="en",
                legality_score=0.8
            ),
            NSMGrammarMolecule(
                name="se_usa_para",
                structure="X se usa para Y",
                primitives=["UsedFor"],
                language="es",
                legality_score=0.8
            ),
            NSMGrammarMolecule(
                name="utilise_pour",
                structure="X est utilisé pour Y",
                primitives=["UsedFor"],
                language="fr",
                legality_score=0.8
            ),
            
            # Containment molecules
            NSMGrammarMolecule(
                name="contains",
                structure="X contains Y",
                primitives=["Contains"],
                language="en",
                legality_score=0.9
            ),
            NSMGrammarMolecule(
                name="contiene",
                structure="X contiene Y",
                primitives=["Contains"],
                language="es",
                legality_score=0.9
            ),
            NSMGrammarMolecule(
                name="contient",
                structure="X contient Y",
                primitives=["Contains"],
                language="fr",
                legality_score=0.9
            )
        ]
        
        for molecule in base_molecules:
            self.add_molecule(molecule)
    
    def add_molecule(self, molecule: NSMGrammarMolecule) -> bool:
        """Add a molecule to the registry with validation."""
        if self._validate_molecule(molecule):
            molecule_id = f"{molecule.language}_{molecule.name}"
            self.molecules[molecule_id] = molecule
            self.language_molecules[molecule.language].append(molecule_id)
            
            # Update primitive patterns
            for primitive in molecule.primitives:
                if primitive not in self.primitive_patterns:
                    self.primitive_patterns[primitive] = []
                self.primitive_patterns[primitive].append(molecule_id)
            
            # Update structural templates
            structure_type = self._classify_structure(molecule.structure)
            if structure_type not in self.structural_templates:
                self.structural_templates[structure_type] = []
            self.structural_templates[structure_type].append(molecule_id)
            
            return True
        return False
    
    def _validate_molecule(self, molecule: NSMGrammarMolecule) -> bool:
        """Validate a molecule against grammar rules."""
        # Structural validation
        if len(molecule.primitives) < self.grammar_rules['structural']['min_primitives']:
            return False
        
        if len(molecule.primitives) > self.grammar_rules['structural']['max_primitives']:
            return False
        
        # Check for forbidden patterns
        for pattern in self.grammar_rules['structural']['forbidden_patterns']:
            if pattern in molecule.structure:
                return False
        
        # Legality score validation
        if molecule.legality_score < 0.5:
            return False
        
        return True
    
    def _classify_structure(self, structure: str) -> str:
        """Classify molecule structure type."""
        if "has" in structure.lower() or "tiene" in structure.lower() or "a" in structure.lower():
            return "property"
        elif "at" in structure.lower() or "en" in structure.lower():
            return "location"
        elif "similar" in structure.lower():
            return "similarity"
        elif "used" in structure.lower() or "usa" in structure.lower() or "utilise" in structure.lower():
            return "purpose"
        elif "contains" in structure.lower() or "contiene" in structure.lower() or "contient" in structure.lower():
            return "containment"
        else:
            return "general"
    
    def get_molecules_by_primitive(self, primitive: str) -> List[NSMGrammarMolecule]:
        """Get molecules that use a specific primitive."""
        molecule_ids = self.primitive_patterns.get(primitive, [])
        return [self.molecules[m_id] for m_id in molecule_ids if m_id in self.molecules]
    
    def get_molecules_by_language(self, language: str) -> List[NSMGrammarMolecule]:
        """Get molecules for a specific language."""
        molecule_ids = self.language_molecules.get(language, [])
        return [self.molecules[m_id] for m_id in molecule_ids if m_id in self.molecules]
    
    def get_molecules_by_structure(self, structure_type: str) -> List[NSMGrammarMolecule]:
        """Get molecules by structure type."""
        molecule_ids = self.structural_templates.get(structure_type, [])
        return [self.molecules[m_id] for m_id in molecule_ids if m_id in self.molecules]
    
    def find_best_molecule(self, primitives: List[str], language: str, 
                          context: str = "") -> Optional[NSMGrammarMolecule]:
        """Find the best molecule for given primitives and language."""
        candidates = []
        
        for primitive in primitives:
            candidates.extend(self.get_molecules_by_primitive(primitive))
        
        # Filter by language
        candidates = [c for c in candidates if c.language == language]
        
        if not candidates:
            return None
        
        # Score candidates
        scored_candidates = []
        for candidate in candidates:
            score = self._score_molecule(candidate, primitives, context)
            scored_candidates.append((score, candidate))
        
        # Return best candidate
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1] if scored_candidates else None
    
    def _score_molecule(self, molecule: NSMGrammarMolecule, primitives: List[str], 
                       context: str) -> float:
        """Score a molecule based on primitives and context."""
        score = molecule.legality_score
        
        # Primitive coverage bonus
        covered_primitives = set(molecule.primitives) & set(primitives)
        coverage_ratio = len(covered_primitives) / len(primitives) if primitives else 0
        score += coverage_ratio * 0.2
        
        # Usage frequency bonus
        score += min(molecule.usage_count / 100, 0.1)
        
        # Semantic coherence bonus
        score += molecule.semantic_coherence * 0.1
        
        # Cross-language consistency bonus
        score += molecule.cross_language_consistency * 0.1
        
        return min(score, 1.0)
    
    def generate_explication(self, primitives: List[str], language: str, 
                           context: str = "") -> str:
        """Generate an explication using the best molecule."""
        molecule = self.find_best_molecule(primitives, language, context)
        
        if molecule:
            # Update usage count
            molecule.usage_count += 1
            
            # Generate explication from structure
            explication = self._fill_structure(molecule.structure, primitives, context)
            return explication
        
        # Fallback to simple explication
        return f"{' '.join(primitives)}({context})" if context else ' '.join(primitives)
    
    def _fill_structure(self, structure: str, primitives: List[str], context: str) -> str:
        """Fill a structure template with primitives and context."""
        # Simple template filling
        if "X" in structure and "Y" in structure:
            if len(primitives) >= 2:
                return structure.replace("X", primitives[0]).replace("Y", primitives[1])
            elif len(primitives) == 1:
                return structure.replace("X", primitives[0]).replace("Y", context)
        
        # Single variable replacement
        if "X" in structure:
            if primitives:
                return structure.replace("X", primitives[0])
            else:
                return structure.replace("X", context)
        
        return structure


class EnhancedNSMGrammarSystem:
    """Enhanced NSM grammar system with molecule registry and validation."""
    
    def __init__(self):
        """Initialize the enhanced NSM grammar system."""
        self.registry = NSMGrammarMoleculeRegistry()
        self.nsm_translator = NSMTranslator()
        self.enhanced_explicator = EnhancedNSMExplicator()
        self.sbert_model = None
        
        # Load periodic table
        try:
            with open("data/nsm_periodic_table.json", 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            self.periodic_table = PeriodicTable.from_dict(table_data)
        except Exception as e:
            logger.warning(f"Failed to load periodic table: {e}")
            self.periodic_table = PeriodicTable()
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for NSM grammar system...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def validate_explication_grammar(self, explication: str, language: str) -> Dict[str, Any]:
        """Validate explication grammar using molecule registry."""
        validation = {
            'is_valid': False,
            'legality_score': 0.0,
            'structural_score': 0.0,
            'semantic_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Extract primitives from explication
            primitives = self.nsm_translator.detect_primitives_in_text(explication, language)
            
            if not primitives:
                validation['issues'].append("No primitives detected in explication")
                return validation
            
            # Find matching molecules
            matching_molecules = []
            for primitive in primitives:
                molecules = self.registry.get_molecules_by_primitive(primitive)
                matching_molecules.extend(molecules)
            
            # Filter by language
            matching_molecules = [m for m in matching_molecules if m.language == language]
            
            if not matching_molecules:
                validation['issues'].append(f"No grammar molecules found for primitives: {primitives}")
                validation['recommendations'].append("Add grammar molecules for these primitives")
                return validation
            
            # Calculate structural score
            structural_score = self._calculate_structural_score(explication, matching_molecules)
            validation['structural_score'] = structural_score
            
            # Calculate semantic score
            semantic_score = self._calculate_semantic_score(explication, primitives, language)
            validation['semantic_score'] = semantic_score
            
            # Calculate overall legality score
            legality_score = (structural_score + semantic_score) / 2
            validation['legality_score'] = legality_score
            
            # Determine validity
            validation['is_valid'] = legality_score >= 0.7
            
            if not validation['is_valid']:
                validation['issues'].append(f"Low legality score: {legality_score:.3f}")
                validation['recommendations'].append("Improve explication structure and semantics")
            
        except Exception as e:
            validation['issues'].append(f"Validation error: {e}")
        
        return validation
    
    def _calculate_structural_score(self, explication: str, molecules: List[NSMGrammarMolecule]) -> float:
        """Calculate structural score based on molecule patterns."""
        if not molecules:
            return 0.0
        
        # Check if explication matches any molecule structure
        best_match_score = 0.0
        
        for molecule in molecules:
            # Simple pattern matching
            if molecule.structure.lower() in explication.lower():
                best_match_score = max(best_match_score, molecule.legality_score)
            
            # Check for structural similarity
            similarity = self._calculate_structure_similarity(explication, molecule.structure)
            best_match_score = max(best_match_score, similarity * molecule.legality_score)
        
        return best_match_score
    
    def _calculate_structure_similarity(self, explication: str, structure: str) -> float:
        """Calculate similarity between explication and structure."""
        # Simple word overlap similarity
        explication_words = set(explication.lower().split())
        structure_words = set(structure.lower().split())
        
        if not structure_words:
            return 0.0
        
        overlap = len(explication_words & structure_words)
        similarity = overlap / len(structure_words)
        
        return similarity
    
    def _calculate_semantic_score(self, explication: str, primitives: List[str], language: str) -> float:
        """Calculate semantic score using SBERT."""
        if not self.sbert_model or not primitives:
            return 0.5  # Default score
        
        try:
            # Create reference explication using molecules
            reference_explication = self.registry.generate_explication(primitives, language, "")
            
            # Calculate semantic similarity
            embeddings = self.sbert_model.encode([explication, reference_explication])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return max(0.0, float(similarity))
        
        except Exception as e:
            logger.warning(f"Semantic score calculation failed: {e}")
            return 0.5
    
    def generate_improved_explication(self, text: str, language: str) -> str:
        """Generate improved explication using grammar molecules."""
        try:
            # Extract primitives
            primitives = self.nsm_translator.detect_primitives_in_text(text, language)
            
            if not primitives:
                # Fallback to basic explication
                return f"unknown({text})"
            
            # Generate explication using registry
            explication = self.registry.generate_explication(primitives, language, text)
            
            # Validate the generated explication
            validation = self.validate_explication_grammar(explication, language)
            
            if validation['is_valid']:
                return explication
            else:
                # Try to improve the explication
                improved_explication = self._improve_explication(explication, primitives, language)
                return improved_explication
        
        except Exception as e:
            logger.warning(f"Failed to generate improved explication: {e}")
            return f"error({text})"
    
    def _improve_explication(self, explication: str, primitives: List[str], language: str) -> str:
        """Improve explication based on grammar validation."""
        # Try different molecule combinations
        for i in range(len(primitives)):
            for j in range(i + 1, len(primitives) + 1):
                subset = primitives[i:j]
                improved = self.registry.generate_explication(subset, language, "")
                
                validation = self.validate_explication_grammar(improved, language)
                if validation['is_valid']:
                    return improved
        
        # If no improvement found, return original
        return explication
    
    def run_grammar_enhancement(self, test_texts: List[str], languages: List[str] = ["en", "es", "fr"]) -> Dict[str, Any]:
        """Run comprehensive grammar enhancement analysis."""
        logger.info(f"Running NSM grammar enhancement for {len(test_texts)} texts")
        
        enhancement_results = {
            'test_configuration': {
                'num_test_texts': len(test_texts),
                'languages': languages,
                'timestamp': time.time()
            },
            'grammar_analysis': {},
            'molecule_usage': {},
            'improvement_metrics': {},
            'recommendations': []
        }
        
        all_validations = []
        molecule_usage = Counter()
        
        for language in languages:
            lang_results = {
                'validations': [],
                'improvements': [],
                'avg_legality': 0.0,
                'avg_structural': 0.0,
                'avg_semantic': 0.0
            }
            
            for text in test_texts:
                # Generate explication
                explication = self.generate_improved_explication(text, language)
                
                # Validate grammar
                validation = self.validate_explication_grammar(explication, language)
                lang_results['validations'].append(validation)
                all_validations.append(validation)
                
                # Track molecule usage
                primitives = self.nsm_translator.detect_primitives_in_text(text, language)
                for primitive in primitives:
                    molecules = self.registry.get_molecules_by_primitive(primitive)
                    for molecule in molecules:
                        if molecule.language == language:
                            molecule_usage[molecule.name] += 1
                
                # Compare with original explication
                try:
                    original_explication = self.enhanced_explicator.explicate(text, language)
                    if original_explication:
                        original_validation = self.validate_explication_grammar(original_explication, language)
                        improvement = {
                            'text': text,
                            'original': original_explication,
                            'improved': explication,
                            'original_score': original_validation['legality_score'],
                            'improved_score': validation['legality_score'],
                            'improvement': validation['legality_score'] - original_validation['legality_score']
                        }
                        lang_results['improvements'].append(improvement)
                except Exception as e:
                    logger.warning(f"Failed to compare with original: {e}")
            
            # Calculate averages
            if lang_results['validations']:
                lang_results['avg_legality'] = np.mean([v['legality_score'] for v in lang_results['validations']])
                lang_results['avg_structural'] = np.mean([v['structural_score'] for v in lang_results['validations']])
                lang_results['avg_semantic'] = np.mean([v['semantic_score'] for v in lang_results['validations']])
            
            enhancement_results['grammar_analysis'][language] = lang_results
        
        # Overall metrics
        enhancement_results['improvement_metrics'] = {
            'avg_legality': np.mean([v['legality_score'] for v in all_validations]),
            'avg_structural': np.mean([v['structural_score'] for v in all_validations]),
            'avg_semantic': np.mean([v['semantic_score'] for v in all_validations]),
            'valid_explications': sum(1 for v in all_validations if v['is_valid']),
            'total_explications': len(all_validations)
        }
        
        # Molecule usage statistics
        enhancement_results['molecule_usage'] = dict(molecule_usage.most_common(10))
        
        # Generate recommendations
        enhancement_results['recommendations'] = self._generate_grammar_recommendations(
            enhancement_results['improvement_metrics'], molecule_usage
        )
        
        return enhancement_results
    
    def _generate_grammar_recommendations(self, metrics: Dict[str, float], 
                                        molecule_usage: Counter) -> List[str]:
        """Generate recommendations based on grammar analysis."""
        recommendations = []
        
        # Legality score recommendations
        if metrics['avg_legality'] < 0.7:
            recommendations.append("Low average legality score - focus on improving explication structure")
        
        if metrics['avg_structural'] < 0.6:
            recommendations.append("Low structural score - add more grammar molecules for common patterns")
        
        if metrics['avg_semantic'] < 0.6:
            recommendations.append("Low semantic score - improve semantic coherence in explications")
        
        # Molecule usage recommendations
        if not molecule_usage:
            recommendations.append("No molecule usage detected - ensure molecules are properly registered")
        else:
            most_used = molecule_usage.most_common(1)[0]
            recommendations.append(f"Most used molecule: {most_used[0]} ({most_used[1]} times)")
        
        # Validity recommendations
        validity_rate = metrics['valid_explications'] / metrics['total_explications']
        if validity_rate < 0.8:
            recommendations.append(f"Low validity rate ({validity_rate:.1%}) - improve grammar validation")
        
        return recommendations


def main():
    """Main function to run NSM grammar molecules enhancement."""
    logger.info("Starting NSM grammar molecules enhancement...")
    
    # Initialize enhanced grammar system
    grammar_system = EnhancedNSMGrammarSystem()
    
    # Test texts
    test_texts = [
        "The red car is parked near the building",
        "The cat is on the mat",
        "This is similar to that",
        "The book contains important information about science",
        "The weather is cold today",
        "She works at the hospital",
        "The movie was very long",
        "I need to buy groceries",
        "Children play in the park",
        "The restaurant serves Italian food"
    ]
    
    # Run grammar enhancement
    enhancement_results = grammar_system.run_grammar_enhancement(test_texts, ["en", "es", "fr"])
    
    # Print results
    print("\n" + "="*80)
    print("NSM GRAMMAR MOLECULES ENHANCEMENT RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Texts: {enhancement_results['test_configuration']['num_test_texts']}")
    print(f"  Languages: {enhancement_results['test_configuration']['languages']}")
    
    print(f"\nOverall Improvement Metrics:")
    metrics = enhancement_results['improvement_metrics']
    print(f"  Average Legality Score: {metrics['avg_legality']:.3f}")
    print(f"  Average Structural Score: {metrics['avg_structural']:.3f}")
    print(f"  Average Semantic Score: {metrics['avg_semantic']:.3f}")
    print(f"  Valid Explications: {metrics['valid_explications']}/{metrics['total_explications']}")
    print(f"  Validity Rate: {metrics['valid_explications']/metrics['total_explications']:.1%}")
    
    print(f"\nMolecule Usage (Top 5):")
    for molecule, count in list(enhancement_results['molecule_usage'].items())[:5]:
        print(f"  {molecule}: {count} times")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(enhancement_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Show detailed results for each language
    for lang, results in enhancement_results['grammar_analysis'].items():
        print(f"\n{lang.upper()} Language Results:")
        print(f"  Average Legality: {results['avg_legality']:.3f}")
        print(f"  Average Structural: {results['avg_structural']:.3f}")
        print(f"  Average Semantic: {results['avg_semantic']:.3f}")
        
        # Show some example improvements
        if results['improvements']:
            print(f"  Example Improvements:")
            for i, improvement in enumerate(results['improvements'][:3]):
                print(f"    {i+1}. {improvement['text']}")
                print(f"       Original: {improvement['original']} (score: {improvement['original_score']:.3f})")
                print(f"       Improved: {improvement['improved']} (score: {improvement['improved_score']:.3f})")
                print(f"       Improvement: {improvement['improvement']:+.3f}")
    
    # Save results
    output_path = "data/nsm_grammar_molecules_enhanced_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(enhancement_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"NSM grammar molecules enhancement report saved to: {output_path}")
    
    print("="*80)
    print("NSM grammar molecules enhancement completed!")
    print("="*80)


if __name__ == "__main__":
    main()
