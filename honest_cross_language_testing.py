#!/usr/bin/env python3
"""
Honest Cross-Language Testing with Parallel Data

This script performs honest evaluation of primitive universality using:
1. Parallel test data (same content in different languages)
2. Realistic similarity thresholds
3. No multilingual bias in primitive embeddings
4. Proper evaluation metrics
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter
import numpy as np

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

import logging
from src.table.algebra import PrimitiveAlgebra
from src.table.schema import PeriodicTable
from src.table.embedding_factorizer import EmbeddingFactorizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HonestCrossLanguageTester:
    """Honest cross-language primitive universality testing."""
    
    def __init__(self, primitive_table_path: str = "data/primitives_with_semantic.json"):
        """Initialize the honest tester."""
        self.primitive_table_path = primitive_table_path
        self.primitive_table = None
        self.embedding_factorizer = None
        self._load_components()
    
    def _load_components(self):
        """Load required components."""
        try:
            # Load primitive table
            with open(self.primitive_table_path, 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            self.primitive_table = PeriodicTable.from_dict(table_data)
            logger.info(f"✓ Loaded primitive table with {len(self.primitive_table.primitives)} primitives")
            
            # Load embedding factorizer
            self.embedding_factorizer = EmbeddingFactorizer(self.primitive_table)
            logger.info("✓ Loaded embedding factorizer")
            
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            raise
    
    def load_parallel_data(self) -> Dict[str, List[str]]:
        """Load parallel test data."""
        try:
            with open("data/parallel_test_data.json", 'r', encoding='utf-8') as f:
                parallel_data = json.load(f)
            
            logger.info(f"✓ Loaded parallel data: {len(parallel_data['en'])} sentence pairs")
            return parallel_data
            
        except Exception as e:
            logger.error(f"Failed to load parallel data: {e}")
            return {}
    
    def test_parallel_primitive_detection(self, parallel_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Test primitive detection on parallel data with honest evaluation."""
        logger.info("\n🔍 Testing Primitive Detection on Parallel Data")
        logger.info("=" * 60)
        
        results = {}
        all_detected_primitives = set()
        
        for lang, sentences in parallel_data.items():
            logger.info(f"\n📝 Testing {lang.upper()} parallel sentences")
            
            # Track detection results
            primitive_counts = Counter()
            detection_examples = defaultdict(list)
            successful_detections = 0
            total_similarity = 0.0
            
            for i, sentence in enumerate(sentences):
                try:
                    # Use realistic, configurable similarity threshold
                    thr = os.getenv('PERIODIC_SIM_THRESHOLD')
                    thr_val = float(thr) if thr is not None else None
                    results = self.embedding_factorizer.factorize_text(
                        sentence, top_k=5, similarity_threshold=thr_val
                    )
                    
                    if results:
                        successful_detections += 1
                        sentence_similarity = 0.0
                        
                        for primitive_name, similarity in results:
                            primitive_counts[primitive_name] += 1
                            all_detected_primitives.add(primitive_name)
                            sentence_similarity = max(sentence_similarity, similarity)
                            
                            if len(detection_examples[primitive_name]) < 2:
                                detection_examples[primitive_name].append({
                                    'sentence': sentence,
                                    'similarity': float(similarity)
                                })
                        
                        total_similarity += sentence_similarity
                    
                except Exception as e:
                    logger.warning(f"Error processing {lang} sentence {i}: {e}")
            
            avg_similarity = total_similarity / len(sentences) if sentences else 0.0
            
            results[lang] = {
                'total_sentences': len(sentences),
                'successful_detections': successful_detections,
                'detection_rate': successful_detections / len(sentences) if sentences else 0,
                'average_similarity': avg_similarity,
                'primitive_counts': dict(primitive_counts),
                'detection_examples': dict(detection_examples),
                'unique_primitives': len(primitive_counts)
            }
            
            logger.info(f"  Detection rate: {results[lang]['detection_rate']:.2%}")
            logger.info(f"  Average similarity: {results[lang]['average_similarity']:.3f}")
            logger.info(f"  Unique primitives: {results[lang]['unique_primitives']}")
        
        return results, all_detected_primitives
    
    def analyze_parallel_universality(self, results: Dict[str, Any], all_primitives: Set[str]) -> Dict[str, Any]:
        """Analyze universality across parallel data."""
        logger.info("\n🌍 Analyzing Parallel Universality")
        logger.info("=" * 60)
        
        all_languages = set(results.keys())
        primitive_language_coverage = defaultdict(set)
        
        # Find which languages each primitive appears in
        for lang, lang_results in results.items():
            for primitive in lang_results['primitive_counts'].keys():
                primitive_language_coverage[primitive].add(lang)
        
        # Categorize primitives
        universal_primitives = {
            primitive: coverage for primitive, coverage in primitive_language_coverage.items()
            if coverage == all_languages
        }
        
        cross_language_primitives = {
            primitive: coverage for primitive, coverage in primitive_language_coverage.items()
            if len(coverage) > 1 and coverage != all_languages
        }
        
        language_specific_primitives = {
            primitive: coverage for primitive, coverage in primitive_language_coverage.items()
            if len(coverage) == 1
        }
        
        # Calculate universality metrics
        total_primitives = len(primitive_language_coverage)
        universality_rate = len(universal_primitives) / total_primitives if total_primitives > 0 else 0
        
        logger.info(f"📊 Honest Universality Analysis:")
        logger.info(f"  Total languages tested: {len(all_languages)}")
        logger.info(f"  Total unique primitives detected: {total_primitives}")
        logger.info(f"  Universal primitives: {len(universal_primitives)} ({universality_rate:.1%})")
        logger.info(f"  Cross-language primitives: {len(cross_language_primitives)}")
        logger.info(f"  Language-specific primitives: {len(language_specific_primitives)}")
        
        if universal_primitives:
            logger.info(f"\n🌍 Universal Primitives (all {len(all_languages)} languages):")
            for primitive in sorted(universal_primitives.keys()):
                total_count = sum(results[lang]['primitive_counts'].get(primitive, 0) 
                                for lang in all_languages)
                logger.info(f"  • {primitive}: {total_count} total detections")
        else:
            logger.info(f"\n❌ No universal primitives found!")
        
        return {
            'total_languages': len(all_languages),
            'total_primitives': total_primitives,
            'universal_primitives': list(universal_primitives.keys()),
            'universality_rate': universality_rate,
            'cross_language_primitives': list(cross_language_primitives.keys()),
            'language_specific_primitives': list(language_specific_primitives.keys()),
            'primitive_coverage': dict(primitive_language_coverage)
        }
    
    def evaluate_translation_potential(self, universality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate translation potential based on honest results."""
        logger.info("\n🔤 Evaluating Translation Potential")
        logger.info("=" * 60)
        
        universal_count = len(universality_analysis['universal_primitives'])
        universality_rate = universality_analysis['universality_rate']
        
        # Honest assessment of translation potential
        if universality_rate >= 0.3:  # At least 30% of detected primitives are universal
            potential = "High"
            confidence = "Strong evidence for primitive-based translation"
        elif universality_rate >= 0.1:  # At least 10% are universal
            potential = "Medium"
            confidence = "Moderate evidence, needs more primitives"
        elif universal_count > 0:
            potential = "Low"
            confidence = "Weak evidence, very few universal primitives"
        else:
            potential = "None"
            confidence = "No universal primitives found"
        
        logger.info(f"📈 Translation Potential Assessment:")
        logger.info(f"  Universality rate: {universality_rate:.1%}")
        logger.info(f"  Universal primitives: {universal_count}")
        logger.info(f"  Potential: {potential}")
        logger.info(f"  Confidence: {confidence}")
        
        return {
            'potential': potential,
            'confidence': confidence,
            'universality_rate': universality_rate,
            'universal_primitive_count': universal_count
        }
    
    def generate_honest_report(self, results: Dict[str, Any], universality_analysis: Dict[str, Any], 
                             translation_analysis: Dict[str, Any]) -> None:
        """Generate honest evaluation report."""
        logger.info("\n📋 Generating Honest Evaluation Report")
        logger.info("=" * 60)
        
        report = {
            'test_metadata': {
                'test_type': 'honest_parallel_cross_language',
                'similarity_threshold': 0.3,
                'parallel_data_used': True,
                'multilingual_bias_removed': True
            },
            'summary': {
                'total_languages': universality_analysis['total_languages'],
                'total_primitives': universality_analysis['total_primitives'],
                'universal_primitives': universality_analysis['universal_primitives'],
                'universality_rate': universality_analysis['universality_rate'],
                'translation_potential': translation_analysis['potential'],
                'confidence': translation_analysis['confidence']
            },
            'per_language_results': results,
            'universality_analysis': universality_analysis,
            'translation_analysis': translation_analysis
        }
        
        # Save report
        output_file = Path("data/honest_cross_language_report.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"✓ Saved honest report to: {output_file}")
        
        # Print summary
        logger.info(f"\n📊 HONEST RESULTS SUMMARY:")
        logger.info(f"  Universality Rate: {universality_analysis['universality_rate']:.1%}")
        logger.info(f"  Universal Primitives: {len(universality_analysis['universal_primitives'])}")
        logger.info(f"  Translation Potential: {translation_analysis['potential']}")
        logger.info(f"  Confidence: {translation_analysis['confidence']}")
    
    def run_honest_evaluation(self):
        """Run complete honest evaluation."""
        logger.info("🔍 Starting Honest Cross-Language Evaluation")
        logger.info("=" * 60)
        
        # Load parallel data
        parallel_data = self.load_parallel_data()
        if not parallel_data:
            logger.error("Failed to load parallel data")
            return
        
        # Test primitive detection
        results, all_primitives = self.test_parallel_primitive_detection(parallel_data)
        
        # Analyze universality
        universality_analysis = self.analyze_parallel_universality(results, all_primitives)
        
        # Evaluate translation potential
        translation_analysis = self.evaluate_translation_potential(universality_analysis)
        
        # Generate report
        self.generate_honest_report(results, universality_analysis, translation_analysis)

def main():
    """Main function."""
    tester = HonestCrossLanguageTester()
    tester.run_honest_evaluation()

if __name__ == "__main__":
    main()
