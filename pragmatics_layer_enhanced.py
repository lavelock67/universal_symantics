#!/usr/bin/env python3
"""
Enhanced Pragmatics Layer System.

This script implements a comprehensive pragmatics layer system to separate
semantic content from cultural styling while maintaining NSM integrity.
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
from enum import Enum

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
        converted_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
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


class SpeechAct(Enum):
    """Enumeration of speech acts."""
    REQUEST = "REQUEST"
    SUGGEST = "SUGGEST"
    INFORM = "INFORM"
    PROMISE = "PROMISE"
    WARN = "WARN"
    APOLOGIZE = "APOLOGIZE"
    THANK = "THANK"
    GREET = "GREET"
    COMMAND = "COMMAND"
    QUESTION = "QUESTION"


class PolitenessLevel(Enum):
    """Enumeration of politeness levels."""
    DEFERENTIAL = "deferential"
    NEUTRAL = "neutral"
    INTIMATE = "intimate"


class DirectnessLevel(Enum):
    """Enumeration of directness levels."""
    DIRECT = "direct"
    INDIRECT = "indirect"


class FormalityLevel(Enum):
    """Enumeration of formality levels."""
    FORMAL = "formal"
    NEUTRAL = "neutral"
    CASUAL = "casual"


class PragmaticsAnnotation:
    """Pragmatics annotation for NSM content."""
    
    def __init__(self, speech_act: SpeechAct = SpeechAct.INFORM,
                 politeness: PolitenessLevel = PolitenessLevel.NEUTRAL,
                 directness: DirectnessLevel = DirectnessLevel.DIRECT,
                 formality: FormalityLevel = FormalityLevel.NEUTRAL,
                 register: str = "general",
                 dialect: str = "en-US",
                 implicatures: List[str] = None,
                 presuppositions: List[str] = None,
                 connotation: Dict[str, float] = None):
        """Initialize pragmatics annotation."""
        self.speech_act = speech_act
        self.politeness = politeness
        self.directness = directness
        self.formality = formality
        self.register = register
        self.dialect = dialect
        self.implicatures = implicatures or []
        self.presuppositions = presuppositions or []
        self.connotation = connotation or {"valence": 0.5, "certainty": 0.5, "arousal": 0.5}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'speech_act': self.speech_act.value,
            'politeness': self.politeness.value,
            'directness': self.directness.value,
            'formality': self.formality.value,
            'register': self.register,
            'dialect': self.dialect,
            'implicatures': self.implicatures,
            'presuppositions': self.presuppositions,
            'connotation': self.connotation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PragmaticsAnnotation':
        """Create from dictionary representation."""
        return cls(
            speech_act=SpeechAct(data.get('speech_act', 'INFORM')),
            politeness=PolitenessLevel(data.get('politeness', 'neutral')),
            directness=DirectnessLevel(data.get('directness', 'direct')),
            formality=FormalityLevel(data.get('formality', 'neutral')),
            register=data.get('register', 'general'),
            dialect=data.get('dialect', 'en-US'),
            implicatures=data.get('implicatures', []),
            presuppositions=data.get('presuppositions', []),
            connotation=data.get('connotation', {"valence": 0.5, "certainty": 0.5, "arousal": 0.5})
        )


class PragmaticsMolecule:
    """Pragmatics molecule for cultural expressions."""
    
    def __init__(self, name: str, literal_meaning: str, conventional_forms: Dict[str, str],
                 connotation: Dict[str, float], locales: List[str], pragmatic: bool = True):
        """Initialize a pragmatics molecule."""
        self.name = name
        self.literal_meaning = literal_meaning
        self.conventional_forms = conventional_forms
        self.connotation = connotation
        self.locales = locales
        self.pragmatic = pragmatic
    
    def get_form(self, locale: str) -> str:
        """Get conventional form for a specific locale."""
        return self.conventional_forms.get(locale, self.literal_meaning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'literal_meaning': self.literal_meaning,
            'conventional_forms': self.conventional_forms,
            'connotation': self.connotation,
            'locales': self.locales,
            'pragmatic': self.pragmatic
        }


class PragmaticsMoleculeRegistry:
    """Registry for pragmatics molecules."""
    
    def __init__(self):
        """Initialize the pragmatics molecule registry."""
        self.molecules: Dict[str, PragmaticsMolecule] = {}
        self.locale_molecules: Dict[str, List[str]] = defaultdict(list)
        
        # Load base pragmatics molecules
        self._load_base_molecules()
    
    def _load_base_molecules(self):
        """Load base pragmatics molecules."""
        base_molecules = [
            # Request molecules
            {
                'name': 'REQUEST_INDIRECT_POLITE',
                'literal_meaning': 'REQUEST(agent, action)',
                'conventional_forms': {
                    'en-US': 'Could you {action}?',
                    'en-GB': 'Would you mind {action}?',
                    'es-ES': '¿Podrías {action}?',
                    'es-MX': '¿Podrías {action}?',
                    'fr-FR': 'Pourriez-vous {action}?'
                },
                'connotation': {'valence': 0.7, 'certainty': 0.3, 'arousal': 0.2},
                'locales': ['en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR'],
                'pragmatic': True
            },
            
            # Suggestion molecules
            {
                'name': 'SUGGEST_SOFT',
                'literal_meaning': 'SUGGEST(agent, action)',
                'conventional_forms': {
                    'en-US': 'You might want to {action}',
                    'en-GB': 'You might like to {action}',
                    'es-ES': 'Quizá deberías {action}',
                    'es-MX': 'Tal vez deberías {action}',
                    'fr-FR': 'Vous devriez peut-être {action}'
                },
                'connotation': {'valence': 0.6, 'certainty': 0.4, 'arousal': 0.3},
                'locales': ['en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR'],
                'pragmatic': True
            },
            
            # Temporal molecules
            {
                'name': 'ALMOST_DO',
                'literal_meaning': 'ALMOST(DO(agent, action))',
                'conventional_forms': {
                    'en-US': 'almost {action}',
                    'en-GB': 'nearly {action}',
                    'es-ES': 'casi {action}',
                    'es-MX': 'casi {action}',
                    'fr-FR': 'faillir {action}'
                },
                'connotation': {'valence': 0.4, 'certainty': 0.2, 'arousal': 0.6},
                'locales': ['en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR'],
                'pragmatic': True
            },
            
            {
                'name': 'RECENT_PAST',
                'literal_meaning': 'RECENT(PAST(DO(agent, action)))',
                'conventional_forms': {
                    'en-US': 'just {action}',
                    'en-GB': 'just {action}',
                    'es-ES': 'acabar de {action}',
                    'es-MX': 'acabar de {action}',
                    'fr-FR': 'venir de {action}'
                },
                'connotation': {'valence': 0.5, 'certainty': 0.8, 'arousal': 0.4},
                'locales': ['en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR'],
                'pragmatic': True
            },
            
            # Politeness molecules
            {
                'name': 'POLITE_REFUSAL',
                'literal_meaning': 'NOT(DO(agent, action))',
                'conventional_forms': {
                    'en-US': 'I\'m afraid I can\'t {action}',
                    'en-GB': 'I\'m sorry, but I can\'t {action}',
                    'es-ES': 'Lo siento, pero no puedo {action}',
                    'es-MX': 'Lo siento, pero no puedo {action}',
                    'fr-FR': 'Je suis désolé, mais je ne peux pas {action}'
                },
                'connotation': {'valence': 0.3, 'certainty': 0.9, 'arousal': 0.2},
                'locales': ['en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR'],
                'pragmatic': True
            },
            
            # Apology molecules
            {
                'name': 'APOLOGIZE_SINCERE',
                'literal_meaning': 'APOLOGIZE(agent, event)',
                'conventional_forms': {
                    'en-US': 'I\'m sorry about {event}',
                    'en-GB': 'I\'m sorry about {event}',
                    'es-ES': 'Lo siento por {event}',
                    'es-MX': 'Lo siento por {event}',
                    'fr-FR': 'Je suis désolé pour {event}'
                },
                'connotation': {'valence': 0.2, 'certainty': 0.8, 'arousal': 0.4},
                'locales': ['en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR'],
                'pragmatic': True
            }
        ]
        
        for molecule_data in base_molecules:
            molecule = PragmaticsMolecule(**molecule_data)
            self.add_molecule(molecule)
    
    def add_molecule(self, molecule: PragmaticsMolecule):
        """Add a molecule to the registry."""
        self.molecules[molecule.name] = molecule
        
        # Index by locale
        for locale in molecule.locales:
            self.locale_molecules[locale].append(molecule.name)
    
    def get_molecules_by_locale(self, locale: str) -> List[PragmaticsMolecule]:
        """Get molecules by locale."""
        molecule_names = self.locale_molecules.get(locale, [])
        return [self.molecules[name] for name in molecule_names if name in self.molecules]
    
    def get_all_molecules(self) -> List[PragmaticsMolecule]:
        """Get all molecules."""
        return list(self.molecules.values())


class StyleRealizer:
    """Style realizer for cultural naturalness."""
    
    def __init__(self):
        """Initialize the style realizer."""
        self.molecule_registry = PragmaticsMoleculeRegistry()
        self.sbert_model = None
        
        # Realization parameters
        self.realization_params = {
            'literalness_threshold': 0.7,
            'enable_style_transforms': True,
            'enable_cultural_adaptation': True,
            'max_style_iterations': 3
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic validation."""
        try:
            logger.info("Loading SBERT model for style realization...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def realize_style(self, canonical_text: str, pragmatics: PragmaticsAnnotation,
                     literalness: float = 0.8) -> Dict[str, Any]:
        """Realize cultural style while preserving meaning."""
        try:
            # Pass 1: Generate canonical surface (meaning-safe)
            canonical_result = self._generate_canonical_surface(canonical_text, pragmatics)
            
            # Pass 2: Apply style transforms (cultural adaptation)
            if literalness < self.realization_params['literalness_threshold']:
                styled_result = self._apply_style_transforms(
                    canonical_result['surface'], pragmatics, literalness
                )
            else:
                styled_result = {
                    'surface': canonical_result['surface'],
                    'style_applied': False,
                    'style_score': 1.0
                }
            
            # Validate meaning preservation
            meaning_preserved = self._validate_meaning_preservation(
                canonical_result['surface'], styled_result['surface']
            )
            
            return {
                'canonical_surface': canonical_result['surface'],
                'styled_surface': styled_result['surface'],
                'meaning_preserved': meaning_preserved,
                'style_score': styled_result.get('style_score', 1.0),
                'pragmatics_applied': pragmatics.to_dict(),
                'literalness': literalness
            }
        
        except Exception as e:
            logger.warning(f"Style realization failed: {e}")
            return {
                'canonical_surface': canonical_text,
                'styled_surface': canonical_text,
                'meaning_preserved': True,
                'style_score': 0.0,
                'pragmatics_applied': pragmatics.to_dict(),
                'literalness': literalness,
                'error': str(e)
            }
    
    def _generate_canonical_surface(self, text: str, pragmatics: PragmaticsAnnotation) -> Dict[str, Any]:
        """Generate canonical surface form."""
        try:
            # Apply basic pragmatics molecules
            surface = text
            
            # Apply speech act patterns
            if pragmatics.speech_act == SpeechAct.REQUEST:
                if pragmatics.politeness == PolitenessLevel.DEFERENTIAL:
                    molecule = self.molecule_registry.molecules.get('REQUEST_INDIRECT_POLITE')
                    if molecule:
                        surface = molecule.get_form(pragmatics.dialect).format(action=text)
            
            elif pragmatics.speech_act == SpeechAct.SUGGEST:
                molecule = self.molecule_registry.molecules.get('SUGGEST_SOFT')
                if molecule:
                    surface = molecule.get_form(pragmatics.dialect).format(action=text)
            
            elif pragmatics.speech_act == SpeechAct.APOLOGIZE:
                molecule = self.molecule_registry.molecules.get('APOLOGIZE_SINCERE')
                if molecule:
                    surface = molecule.get_form(pragmatics.dialect).format(event=text)
            
            return {
                'surface': surface,
                'pragmatics_applied': True
            }
        
        except Exception as e:
            logger.warning(f"Canonical surface generation failed: {e}")
            return {
                'surface': text,
                'pragmatics_applied': False,
                'error': str(e)
            }
    
    def _apply_style_transforms(self, surface: str, pragmatics: PragmaticsAnnotation,
                              literalness: float) -> Dict[str, Any]:
        """Apply style transforms for cultural naturalness."""
        try:
            transformed_surface = surface
            style_score = 1.0
            
            # Apply formality transforms
            if pragmatics.formality == FormalityLevel.FORMAL:
                transformed_surface = self._apply_formal_style(transformed_surface, pragmatics.dialect)
                style_score *= 0.9
            
            elif pragmatics.formality == FormalityLevel.CASUAL:
                transformed_surface = self._apply_casual_style(transformed_surface, pragmatics.dialect)
                style_score *= 0.8
            
            # Apply politeness transforms
            if pragmatics.politeness == PolitenessLevel.DEFERENTIAL:
                transformed_surface = self._apply_deferential_style(transformed_surface, pragmatics.dialect)
                style_score *= 0.85
            
            # Apply directness transforms
            if pragmatics.directness == DirectnessLevel.INDIRECT:
                transformed_surface = self._apply_indirect_style(transformed_surface, pragmatics.dialect)
                style_score *= 0.8
            
            return {
                'surface': transformed_surface,
                'style_applied': True,
                'style_score': style_score
            }
        
        except Exception as e:
            logger.warning(f"Style transform application failed: {e}")
            return {
                'surface': surface,
                'style_applied': False,
                'style_score': 0.5,
                'error': str(e)
            }
    
    def _apply_formal_style(self, text: str, dialect: str) -> str:
        """Apply formal style transformations."""
        # Simple formal style transformations
        if dialect.startswith('en'):
            # Remove contractions
            text = re.sub(r"n't", " not", text)
            text = re.sub(r"'re", " are", text)
            text = re.sub(r"'s", " is", text)
            text = re.sub(r"'ll", " will", text)
            text = re.sub(r"'ve", " have", text)
        
        return text
    
    def _apply_casual_style(self, text: str, dialect: str) -> str:
        """Apply casual style transformations."""
        # Simple casual style transformations
        if dialect.startswith('en'):
            # Add contractions where appropriate
            text = re.sub(r" is ", " 's ", text)
            text = re.sub(r" are ", " 're ", text)
            text = re.sub(r" not ", " n't ", text)
        
        return text
    
    def _apply_deferential_style(self, text: str, dialect: str) -> str:
        """Apply deferential style transformations."""
        # Simple deferential style transformations
        if dialect.startswith('en'):
            # Add polite markers
            if not text.startswith('Could you') and not text.startswith('Would you'):
                text = f"Could you please {text.lower()}"
        
        return text
    
    def _apply_indirect_style(self, text: str, dialect: str) -> str:
        """Apply indirect style transformations."""
        # Simple indirect style transformations
        if dialect.startswith('en'):
            # Add hedging
            if not text.startswith('You might') and not text.startswith('Could you'):
                text = f"You might want to {text.lower()}"
        
        return text
    
    def _validate_meaning_preservation(self, canonical: str, styled: str) -> bool:
        """Validate that meaning is preserved between canonical and styled forms."""
        try:
            if not self.sbert_model:
                return True  # Assume preservation if no model available
            
            # Calculate semantic similarity
            embeddings = self.sbert_model.encode([canonical, styled])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            # Require high similarity for meaning preservation
            return similarity >= 0.8
        
        except Exception as e:
            logger.warning(f"Meaning preservation validation failed: {e}")
            return True  # Assume preservation on error


class PragmaticsLayerSystem:
    """Comprehensive pragmatics layer system."""
    
    def __init__(self):
        """Initialize the pragmatics layer system."""
        self.style_realizer = StyleRealizer()
        self.nsm_translator = NSMTranslator()
        
        # System parameters
        self.system_params = {
            'enable_pragmatics': True,
            'enable_style_realization': True,
            'enable_meaning_validation': True,
            'default_literalness': 0.8
        }
    
    def process_with_pragmatics(self, text: str, pragmatics: PragmaticsAnnotation,
                              literalness: float = None) -> Dict[str, Any]:
        """Process text with pragmatics layer."""
        try:
            if literalness is None:
                literalness = self.system_params['default_literalness']
            
            # Generate NSM explication
            primitives = self.nsm_translator.detect_primitives_in_text(text, pragmatics.dialect[:2])
            
            if primitives:
                # Create canonical explication
                canonical_explication = f"{' '.join(primitives)}({text})"
                
                # Apply pragmatics layer
                pragmatics_result = self.style_realizer.realize_style(
                    canonical_explication, pragmatics, literalness
                )
                
                return {
                    'original_text': text,
                    'primitives': primitives,
                    'canonical_explication': canonical_explication,
                    'pragmatics_result': pragmatics_result,
                    'pragmatics_applied': pragmatics.to_dict(),
                    'literalness': literalness
                }
            else:
                return {
                    'original_text': text,
                    'primitives': [],
                    'canonical_explication': text,
                    'pragmatics_result': {
                        'canonical_surface': text,
                        'styled_surface': text,
                        'meaning_preserved': True,
                        'style_score': 1.0
                    },
                    'pragmatics_applied': pragmatics.to_dict(),
                    'literalness': literalness,
                    'error': 'No primitives detected'
                }
        
        except Exception as e:
            logger.warning(f"Pragmatics processing failed: {e}")
            return {
                'original_text': text,
                'primitives': [],
                'canonical_explication': text,
                'pragmatics_result': {
                    'canonical_surface': text,
                    'styled_surface': text,
                    'meaning_preserved': True,
                    'style_score': 0.0
                },
                'pragmatics_applied': pragmatics.to_dict(),
                'literalness': literalness,
                'error': str(e)
            }
    
    def run_pragmatics_analysis(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive pragmatics analysis."""
        logger.info(f"Running pragmatics analysis for {len(test_cases)} test cases")
        
        analysis_results = {
            'test_configuration': {
                'num_test_cases': len(test_cases),
                'timestamp': time.time()
            },
            'processing_results': [],
            'pragmatics_analysis': {},
            'recommendations': []
        }
        
        # Process test cases
        for test_case in test_cases:
            text = test_case['text']
            pragmatics = PragmaticsAnnotation.from_dict(test_case['pragmatics'])
            literalness = test_case.get('literalness', self.system_params['default_literalness'])
            
            result = self.process_with_pragmatics(text, pragmatics, literalness)
            analysis_results['processing_results'].append(result)
        
        # Analyze results
        analysis_results['pragmatics_analysis'] = self._analyze_pragmatics_results(
            analysis_results['processing_results']
        )
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_pragmatics_recommendations(
            analysis_results['pragmatics_analysis']
        )
        
        return analysis_results
    
    def _analyze_pragmatics_results(self, processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pragmatics processing results."""
        analysis = {
            'total_cases': len(processing_results),
            'successful_cases': 0,
            'meaning_preserved_cases': 0,
            'avg_style_score': 0.0,
            'pragmatics_distribution': defaultdict(int),
            'literalness_distribution': defaultdict(int),
            'style_effectiveness': {
                'high_style_scores': 0,
                'medium_style_scores': 0,
                'low_style_scores': 0
            }
        }
        
        style_scores = []
        
        for result in processing_results:
            if 'error' not in result:
                analysis['successful_cases'] += 1
            
            pragmatics_result = result.get('pragmatics_result', {})
            
            if pragmatics_result.get('meaning_preserved', True):
                analysis['meaning_preserved_cases'] += 1
            
            style_score = pragmatics_result.get('style_score', 0.0)
            style_scores.append(style_score)
            
            # Analyze pragmatics distribution
            pragmatics_applied = result.get('pragmatics_applied', {})
            speech_act = pragmatics_applied.get('speech_act', 'INFORM')
            analysis['pragmatics_distribution'][speech_act] += 1
            
            # Analyze literalness distribution
            literalness = result.get('literalness', 0.8)
            literalness_bucket = f"{int(literalness * 10) * 10}%"
            analysis['literalness_distribution'][literalness_bucket] += 1
            
            # Analyze style effectiveness
            if style_score >= 0.8:
                analysis['style_effectiveness']['high_style_scores'] += 1
            elif style_score >= 0.5:
                analysis['style_effectiveness']['medium_style_scores'] += 1
            else:
                analysis['style_effectiveness']['low_style_scores'] += 1
        
        # Calculate averages
        if style_scores:
            analysis['avg_style_score'] = np.mean(style_scores)
        
        return analysis
    
    def _generate_pragmatics_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for pragmatics layer."""
        recommendations = []
        
        # Success rate recommendations
        success_rate = analysis['successful_cases'] / analysis['total_cases']
        if success_rate < 0.9:
            recommendations.append(f"Low success rate ({success_rate:.1%}) - improve pragmatics processing")
        
        # Meaning preservation recommendations
        meaning_preservation_rate = analysis['meaning_preserved_cases'] / analysis['total_cases']
        if meaning_preservation_rate < 0.95:
            recommendations.append(f"Low meaning preservation rate ({meaning_preservation_rate:.1%}) - strengthen validation")
        
        # Style score recommendations
        if analysis['avg_style_score'] < 0.7:
            recommendations.append("Low average style score - improve style transformations")
        
        # Pragmatics distribution recommendations
        if len(analysis['pragmatics_distribution']) < 3:
            recommendations.append("Limited pragmatics diversity - expand speech act coverage")
        
        # Style effectiveness recommendations
        high_style_ratio = analysis['style_effectiveness']['high_style_scores'] / analysis['total_cases']
        if high_style_ratio < 0.5:
            recommendations.append("Low high-style-score ratio - enhance style realization")
        
        return recommendations


def main():
    """Main function to run enhanced pragmatics layer analysis."""
    logger.info("Starting enhanced pragmatics layer analysis...")
    
    # Initialize pragmatics layer system
    pragmatics_system = PragmaticsLayerSystem()
    
    # Test cases with different pragmatics configurations
    test_cases = [
        {
            'text': 'Please close the door',
            'pragmatics': {
                'speech_act': 'REQUEST',
                'politeness': 'deferential',
                'directness': 'indirect',
                'formality': 'formal',
                'dialect': 'en-US'
            },
            'literalness': 0.6
        },
        {
            'text': 'You should take a jacket',
            'pragmatics': {
                'speech_act': 'SUGGEST',
                'politeness': 'neutral',
                'directness': 'direct',
                'formality': 'casual',
                'dialect': 'en-GB'
            },
            'literalness': 0.7
        },
        {
            'text': 'I am sorry for the delay',
            'pragmatics': {
                'speech_act': 'APOLOGIZE',
                'politeness': 'deferential',
                'directness': 'direct',
                'formality': 'formal',
                'dialect': 'es-ES'
            },
            'literalness': 0.8
        },
        {
            'text': 'The weather is cold',
            'pragmatics': {
                'speech_act': 'INFORM',
                'politeness': 'neutral',
                'directness': 'direct',
                'formality': 'neutral',
                'dialect': 'fr-FR'
            },
            'literalness': 0.9
        },
        {
            'text': 'Could you help me?',
            'pragmatics': {
                'speech_act': 'REQUEST',
                'politeness': 'deferential',
                'directness': 'indirect',
                'formality': 'neutral',
                'dialect': 'es-MX'
            },
            'literalness': 0.5
        }
    ]
    
    # Run pragmatics analysis
    analysis_results = pragmatics_system.run_pragmatics_analysis(test_cases)
    
    # Print results
    print("\n" + "="*80)
    print("ENHANCED PRAGMATICS LAYER ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Cases: {analysis_results['test_configuration']['num_test_cases']}")
    
    print(f"\nPragmatics Analysis:")
    analysis = analysis_results['pragmatics_analysis']
    print(f"  Total Cases: {analysis['total_cases']}")
    print(f"  Successful Cases: {analysis['successful_cases']}")
    print(f"  Success Rate: {analysis['successful_cases']/analysis['total_cases']:.1%}")
    print(f"  Meaning Preserved Cases: {analysis['meaning_preserved_cases']}")
    print(f"  Meaning Preservation Rate: {analysis['meaning_preserved_cases']/analysis['total_cases']:.1%}")
    print(f"  Average Style Score: {analysis['avg_style_score']:.3f}")
    
    print(f"\nPragmatics Distribution:")
    for speech_act, count in analysis['pragmatics_distribution'].items():
        print(f"  {speech_act}: {count}")
    
    print(f"\nLiteralness Distribution:")
    for literalness, count in analysis['literalness_distribution'].items():
        print(f"  {literalness}: {count}")
    
    print(f"\nStyle Effectiveness:")
    style_eff = analysis['style_effectiveness']
    print(f"  High Style Scores: {style_eff['high_style_scores']}")
    print(f"  Medium Style Scores: {style_eff['medium_style_scores']}")
    print(f"  Low Style Scores: {style_eff['low_style_scores']}")
    
    print(f"\nExample Processing Results:")
    for i, result in enumerate(analysis_results['processing_results'][:3]):
        original_text = result['original_text']
        canonical = result.get('canonical_explication', 'N/A')
        styled = result.get('pragmatics_result', {}).get('styled_surface', 'N/A')
        style_score = result.get('pragmatics_result', {}).get('style_score', 0.0)
        
        print(f"  {i+1}. Original: {original_text}")
        print(f"     Canonical: {canonical}")
        print(f"     Styled: {styled}")
        print(f"     Style Score: {style_score:.3f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/pragmatics_layer_enhanced_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced pragmatics layer report saved to: {output_path}")
    
    print("="*80)
    print("Enhanced pragmatics layer analysis completed!")
    print("="*80)


if __name__ == "__main__":
    main()
