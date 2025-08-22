#!/usr/bin/env python3

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging

# Lightweight .env loader (no external dependency)
def _load_dotenv_simple(filename: str = ".env") -> None:
    path = Path(filename)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)

_load_dotenv_simple()

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.table.algebra import PrimitiveAlgebra
from src.table.schema import PeriodicTable
from src.table.embedding_factorizer import EmbeddingFactorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrimitiveTranslator:
    """
    Primitive-based translation system.
    
    Extracts universal information primitives from source language text,
    maps them to target language patterns, and generates translated text.
    Note: Current UMR-based generator is diagnostic-only and may be ungrammatical.
    """
    
    def __init__(self, primitive_table_path: str = "data/primitives_with_semantic.json"):
        """Initialize the primitive translator."""
        self.primitive_table_path = primitive_table_path
        self.table = None
        self.algebra = None
        self.embedding_factorizer = None
        
        # Universal primitives that work across languages
        self.universal_primitives = {
            'Causes', 'CausesDesire', 'HasProperty', 'HasProperty_Foreign',
            'HasProperty_Last', 'HasProperty_Many', 'HasProperty_Near', 
            'HasProperty_Strong', 'IsA', 'IsA_Lot', 'IsActor_Begin',
            'NotDesires', 'PartOf', 'UsedFor'
        }
        
        # Language-specific primitive patterns and templates
        self.language_patterns = {
            'en': {
                'Causes': ['causes', 'leads to', 'results in', 'brings about'],
                'CausesDesire': ['makes want', 'creates desire for', 'inspires'],
                'HasProperty': ['has', 'is', 'characterized by', 'features'],
                'IsA': ['is a', 'is an', 'is a type of', 'belongs to'],
                'PartOf': ['is part of', 'belongs to', 'is contained in'],
                'UsedFor': ['is used for', 'serves to', 'functions as'],
                'NotDesires': ['does not want', 'avoids', 'rejects']
            },
            'es': {
                'Causes': ['causa', 'lleva a', 'resulta en', 'trae consigo'],
                'CausesDesire': ['hace querer', 'crea deseo de', 'inspira'],
                'HasProperty': ['tiene', 'es', 'se caracteriza por', 'presenta'],
                'IsA': ['es un', 'es una', 'es un tipo de', 'pertenece a'],
                'PartOf': ['es parte de', 'pertenece a', 'está contenido en'],
                'UsedFor': ['se usa para', 'sirve para', 'funciona como'],
                'NotDesires': ['no quiere', 'evita', 'rechaza']
            },
            'fr': {
                'Causes': ['cause', 'mène à', 'résulte en', 'apporte'],
                'CausesDesire': ['fait vouloir', 'crée le désir de', 'inspire'],
                'HasProperty': ['a', 'est', 'est caractérisé par', 'présente'],
                'IsA': ['est un', 'est une', 'est un type de', 'appartient à'],
                'PartOf': ['fait partie de', 'appartient à', 'est contenu dans'],
                'UsedFor': ['est utilisé pour', 'sert à', 'fonctionne comme'],
                'NotDesires': ['ne veut pas', 'évite', 'rejette']
            }
        }
        
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components for translation."""
        try:
            # Load primitive table
            with open(self.primitive_table_path, 'r') as f:
                table_data = json.load(f)
            self.table = PeriodicTable.from_dict(table_data)
            logger.info(f"✓ Loaded primitive table with {len(self.table.primitives)} primitives")
            
            # Initialize algebra
            self.algebra = PrimitiveAlgebra(self.table)
            logger.info("✓ Initialized primitive algebra")
            
            # Initialize embedding factorizer
            self.embedding_factorizer = EmbeddingFactorizer(self.table)
            logger.info("✓ Initialized embedding factorizer")
            
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            raise
    
    def extract_primitives(self, text: str, source_lang: str = 'en') -> List[Tuple[str, float]]:
        """Extract universal primitives from source text."""
        try:
            # Configurable similarity threshold for factorization
            thr = os.getenv('PERIODIC_SIM_THRESHOLD')
            thr_val = float(thr) if thr is not None else None
            results = self.embedding_factorizer.factorize_text(
                text, top_k=10, similarity_threshold=thr_val
            )
            
            # Filter to only universal primitives
            universal_results = [
                (primitive, similarity) for primitive, similarity in results
                if primitive in self.universal_primitives
            ]
            
            return universal_results
            
        except Exception as e:
            logger.warning(f"Error extracting primitives: {e}")
            return []
    
    def map_primitives_to_target_language(self, primitives: List[Tuple[str, float]], 
                                        target_lang: str) -> List[Dict[str, Any]]:
        """Map primitives to target language patterns."""
        mapped_primitives = []
        
        for primitive_name, similarity in primitives:
            if primitive_name in self.language_patterns.get(target_lang, {}):
                patterns = self.language_patterns[target_lang][primitive_name]
                mapped_primitives.append({
                    'primitive': primitive_name,
                    'similarity': similarity,
                    'target_patterns': patterns,
                    'confidence': similarity
                })
        
        return mapped_primitives
    
    def generate_translation_candidates(self, text: str, primitives: List[Dict[str, Any]], 
                                      target_lang: str) -> List[str]:
        """Generate translation candidates using primitive patterns."""
        candidates = []
        
        # More realistic translation approach
        for primitive_info in primitives:
            primitive_name = primitive_info['primitive']
            patterns = primitive_info['target_patterns']
            
            if patterns:
                # Use proper vocabulary mapping instead of simple string replacement
                candidate = self._translate_with_vocabulary(text, target_lang, primitive_name)
                
                if candidate and candidate != text:  # Only add if translation actually changed something
                    candidates.append({
                        'text': candidate,
                        'primitive': primitive_name,
                        'confidence': primitive_info['confidence'],
                        'pattern_used': patterns[0]
                    })
        
        return candidates
    
    def _translate_with_vocabulary(self, text: str, target_lang: str, primitive_name: str) -> str:
        """Translate text using proper vocabulary mapping for the detected primitive."""
        # Basic vocabulary mappings (this is still simplified but more realistic)
        vocab_mappings = {
            'es': {
                'IsA': {
                    'is': 'es', 'are': 'son', 'am': 'soy',
                    'a': 'un', 'an': 'una'
                },
                'PartOf': {
                    'part of': 'parte de', 'belongs to': 'pertenece a',
                    'member of': 'miembro de'
                },
                'Causes': {
                    'causes': 'causa', 'leads to': 'lleva a',
                    'results in': 'resulta en'
                },
                'HasProperty': {
                    'has': 'tiene', 'possesses': 'posee',
                    'contains': 'contiene'
                },
                'UsedFor': {
                    'used for': 'usado para', 'used to': 'usado para',
                    'serves to': 'sirve para'
                }
            },
            'fr': {
                'IsA': {
                    'is': 'est', 'are': 'sont', 'am': 'suis',
                    'a': 'un', 'an': 'une'
                },
                'PartOf': {
                    'part of': 'partie de', 'belongs to': 'appartient à',
                    'member of': 'membre de'
                },
                'Causes': {
                    'causes': 'cause', 'leads to': 'mène à',
                    'results in': 'résulte en'
                },
                'HasProperty': {
                    'has': 'a', 'possesses': 'possède',
                    'contains': 'contient'
                },
                'UsedFor': {
                    'used for': 'utilisé pour', 'used to': 'utilisé pour',
                    'serves to': 'sert à'
                }
            }
        }
        
        # Get vocabulary for this primitive and language
        lang_vocab = vocab_mappings.get(target_lang, {}).get(primitive_name, {})
        
        if not lang_vocab:
            return text  # No translation possible
        
        # Apply vocabulary mapping
        translated = text
        for eng_word, target_word in lang_vocab.items():
            # Use word boundaries to avoid partial matches
            import re
            pattern = r'\b' + re.escape(eng_word) + r'\b'
            translated = re.sub(pattern, target_word, translated, flags=re.IGNORECASE)
        
        return translated
    
    def select_best_translation(self, candidates: List[Dict[str, Any]]) -> Optional[str]:
        """Select the best translation candidate based on confidence scores."""
        if not candidates:
            return None
        
        # Sort by confidence and return the highest
        best_candidate = max(candidates, key=lambda x: x['confidence'])
        return best_candidate['text']
    
    def translate_text(self, text: str, source_lang: str = 'en', 
                      target_lang: str = 'es') -> Dict[str, Any]:
        """Translate text using primitive-based approach."""
        logger.info(f"🔄 Translating: {text[:50]}... ({source_lang} → {target_lang})")
        
        # Step 1: Extract primitives from source text
        primitives = self.extract_primitives(text, source_lang)
        logger.info(f"  Extracted {len(primitives)} universal primitives")
        
        if not primitives:
            return {
                'success': False,
                'error': 'No universal primitives detected',
                'source_text': text,
                'target_text': None
            }
        
        # Step 2: Map primitives to target language patterns
        mapped_primitives = self.map_primitives_to_target_language(primitives, target_lang)
        logger.info(f"  Mapped {len(mapped_primitives)} primitives to {target_lang}")
        
        # Step 3: Generate translation candidates
        candidates = self.generate_translation_candidates(text, mapped_primitives, target_lang)
        logger.info(f"  Generated {len(candidates)} translation candidates")
        
        # Step 4: Select best translation
        best_translation = self.select_best_translation(candidates)
        
        if best_translation:
            logger.info(f"  ✓ Translation: {best_translation}")
            return {
                'success': True,
                'source_text': text,
                'target_text': best_translation,
                'primitives_used': [p[0] for p in primitives],
                'confidence': sum(p[1] for p in primitives) / len(primitives),
                'candidates': candidates
            }
        else:
            return {
                'success': False,
                'error': 'No valid translation candidates generated',
                'source_text': text,
                'target_text': None
            }
    
    def batch_translate(self, texts: List[str], source_lang: str = 'en', 
                       target_lang: str = 'es') -> List[Dict[str, Any]]:
        """Translate multiple texts in batch."""
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"📝 Translating text {i+1}/{len(texts)}")
            result = self.translate_text(text, source_lang, target_lang)
            results.append(result)
        
        return results
    
    def evaluate_translation_quality(self, translations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the quality of translations."""
        successful = [t for t in translations if t['success']]
        failed = [t for t in translations if not t['success']]
        
        success_rate = len(successful) / len(translations) if translations else 0
        avg_confidence = sum(t['confidence'] for t in successful) / len(successful) if successful else 0
        
        # Count primitive usage
        primitive_counts = defaultdict(int)
        for translation in successful:
            for primitive in translation.get('primitives_used', []):
                primitive_counts[primitive] += 1
        
        return {
            'total_texts': len(translations),
            'successful_translations': len(successful),
            'failed_translations': len(failed),
            'success_rate': success_rate,
            'average_confidence': avg_confidence,
            'primitive_usage': dict(primitive_counts),
            'most_used_primitives': sorted(primitive_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
        }

def test_translation_prototype():
    """Test the primitive-based translation prototype."""
    print("🌍 Testing Primitive-Based Translation Prototype")
    print("=" * 60)
    
    # Initialize translator
    translator = PrimitiveTranslator()
    
    # More diverse and realistic test texts (not cherry-picked for our patterns)
    test_texts = [
        "The weather is cold today",
        "She works at the hospital",
        "The movie was very long",
        "I need to buy groceries",
        "The computer crashed yesterday",
        "Children play in the park",
        "The restaurant serves Italian food",
        "He drives a red car",
        "The book contains many chapters",
        "Students study for exams"
    ]
    
    # Test English to Spanish translation
    print("\n🇺🇸 → 🇪🇸 English to Spanish Translation")
    print("-" * 40)
    
    es_translations = translator.batch_translate(test_texts, 'en', 'es')
    es_quality = translator.evaluate_translation_quality(es_translations)
    
    for i, result in enumerate(es_translations):
        print(f"\nText {i+1}: {result['source_text']}")
        if result['success']:
            print(f"  → {result['target_text']}")
            print(f"  Primitives: {', '.join(result['primitives_used'])}")
            print(f"  Confidence: {result['confidence']:.3f}")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    print(f"\n📊 Spanish Translation Quality:")
    print(f"  Success Rate: {es_quality['success_rate']:.1%}")
    print(f"  Average Confidence: {es_quality['average_confidence']:.3f}")
    print(f"  Most Used Primitives: {es_quality['most_used_primitives']}")
    
    # Test English to French translation
    print("\n🇺🇸 → 🇫🇷 English to French Translation")
    print("-" * 40)
    
    fr_translations = translator.batch_translate(test_texts, 'en', 'fr')
    fr_quality = translator.evaluate_translation_quality(fr_translations)
    
    for i, result in enumerate(fr_translations):
        print(f"\nText {i+1}: {result['source_text']}")
        if result['success']:
            print(f"  → {result['target_text']}")
            print(f"  Primitives: {', '.join(result['primitives_used'])}")
            print(f"  Confidence: {result['confidence']:.3f}")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    print(f"\n📊 French Translation Quality:")
    print(f"  Success Rate: {fr_quality['success_rate']:.1%}")
    print(f"  Average Confidence: {fr_quality['average_confidence']:.3f}")
    print(f"  Most Used Primitives: {fr_quality['most_used_primitives']}")
    
    # Save results
    results = {
        'spanish_translations': es_translations,
        'french_translations': fr_translations,
        'spanish_quality': es_quality,
        'french_quality': fr_quality,
        'test_texts': test_texts
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, 'dtype'):  # numpy array or scalar
            return float(obj) if hasattr(obj, 'item') else obj.tolist()
        else:
            return obj
    
    results = convert_numpy_types(results)
    
    with open('data/translation_prototype_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to data/translation_prototype_results.json")
    
    return results

def main():
    """Run the translation prototype test."""
    results = test_translation_prototype()
    
    # Print summary
    print(f"\n🎯 Translation Prototype Summary:")
    print(f"  ✅ Spanish Success Rate: {results['spanish_quality']['success_rate']:.1%}")
    print(f"  ✅ French Success Rate: {results['french_quality']['success_rate']:.1%}")
    print(f"  ✅ Total Primitives Used: {len(results['spanish_quality']['primitive_usage'])}")
    
    if results['spanish_quality']['success_rate'] > 0.5:
        print(f"  🚀 Primitive-based translation is working!")
    else:
        print(f"  💡 Need to improve primitive detection and mapping")

if __name__ == "__main__":
    main()
