#!/usr/bin/env python3

import json
import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
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

class CrossLanguageTester:
    """
    Comprehensive cross-language testing framework for information primitives.
    
    Tests primitive detection and universality across multiple languages
    to validate our primitive-based translation hypothesis.
    """
    
    def __init__(self, primitive_table_path: str = "data/primitives_with_semantic.json"):
        """Initialize the cross-language tester."""
        self.primitive_table_path = primitive_table_path
        self.table = None
        self.algebra = None
        self.embedding_factorizer = None
        self.results = {}
        
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components for cross-language testing."""
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
    
    def load_multilingual_corpora(self):
        """Load multilingual corpora for testing."""
        corpora = {}
        
        # Define corpus paths
        corpus_paths = {
            'en': 'data/tatoeba_en_10000.txt',
            'es': 'data/tatoeba_es_10000.txt', 
            'fr': 'data/tatoeba_fr_10000.txt'
        }
        
        for lang, path in corpus_paths.items():
            try:
                if Path(path).exists():
                    texts = Path(path).read_text(encoding='utf-8').strip().split('\n')
                    # Take a sample for testing (first 100 texts)
                    corpora[lang] = texts[:100]
                    logger.info(f"✓ Loaded {lang} corpus: {len(corpora[lang])} texts")
                else:
                    logger.warning(f"Corpus not found: {path}")
                    corpora[lang] = []
            except Exception as e:
                logger.error(f"Failed to load {lang} corpus: {e}")
                corpora[lang] = []
        
        return corpora
    
    def test_primitive_detection_by_language(self, corpora: dict):
        """Test primitive detection for each language."""
        results = {}
        
        for lang, texts in corpora.items():
            if not texts:
                continue
                
            logger.info(f"\n🔍 Testing {lang.upper()} corpus ({len(texts)} texts)")
            
            # Track detection results
            primitive_counts = Counter()
            detection_examples = defaultdict(list)
            successful_detections = 0
            
            for i, text in enumerate(texts):
                try:
                    # Use configurable similarity threshold; if None factorizer will use env PERIODIC_SIM_THRESHOLD
                    thr = os.getenv('PERIODIC_SIM_THRESHOLD')
                    thr_val = float(thr) if thr is not None else None
                    factorized_results = self.embedding_factorizer.factorize_text(
                        text, top_k=5, similarity_threshold=thr_val  # env-configurable
                    )
                    
                    if factorized_results:
                        successful_detections += 1
                        for primitive_name, similarity in factorized_results:
                            primitive_counts[primitive_name] += 1
                            if len(detection_examples[primitive_name]) < 3:
                                detection_examples[primitive_name].append({
                                    'text': text[:100] + "...",
                                    'similarity': similarity
                                })
                    
                except Exception as e:
                    logger.warning(f"Error processing {lang} text {i}: {e}")
            
            results[lang] = {
                'total_texts': len(texts),
                'successful_detections': successful_detections,
                'detection_rate': successful_detections / len(texts) if texts else 0,
                'primitive_counts': dict(primitive_counts),
                'detection_examples': dict(detection_examples),
                'unique_primitives': len(primitive_counts)
            }
            
            logger.info(f"  Detection rate: {results[lang]['detection_rate']:.2%}")
            logger.info(f"  Unique primitives: {results[lang]['unique_primitives']}")
        
        return results
    
    def analyze_cross_language_universality(self, results: dict):
        """Analyze which primitives are universal across languages."""
        logger.info("\n🌍 Analyzing Cross-Language Universality")
        logger.info("=" * 60)
        
        # Find primitives that appear in all languages
        all_languages = set(results.keys())
        primitive_language_coverage = defaultdict(set)
        
        for lang, lang_results in results.items():
            for primitive in lang_results['primitive_counts'].keys():
                primitive_language_coverage[primitive].add(lang)
        
        # Categorize primitives by universality
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
        
        # Print analysis
        logger.info(f"📊 Universality Analysis:")
        logger.info(f"  Total languages tested: {len(all_languages)}")
        logger.info(f"  Total unique primitives detected: {len(primitive_language_coverage)}")
        logger.info(f"  Universal primitives: {len(universal_primitives)}")
        logger.info(f"  Cross-language primitives: {len(cross_language_primitives)}")
        logger.info(f"  Language-specific primitives: {len(language_specific_primitives)}")
        
        logger.info(f"\n🌍 Universal Primitives (all {len(all_languages)} languages):")
        for primitive, coverage in sorted(universal_primitives.items()):
            total_count = sum(results[lang]['primitive_counts'].get(primitive, 0) 
                            for lang in coverage)
            logger.info(f"  • {primitive}: {total_count} total detections")
        
        logger.info(f"\n🌐 Cross-Language Primitives:")
        for primitive, coverage in sorted(cross_language_primitives.items()):
            total_count = sum(results[lang]['primitive_counts'].get(primitive, 0) 
                            for lang in coverage)
            languages = ", ".join(sorted(coverage))
            logger.info(f"  • {primitive}: {total_count} detections ({languages})")
        
        logger.info(f"\n🎯 Language-Specific Primitives:")
        for primitive, coverage in sorted(language_specific_primitives.items()):
            lang = list(coverage)[0]
            count = results[lang]['primitive_counts'].get(primitive, 0)
            logger.info(f"  • {primitive}: {count} detections (only in {lang})")
        
        return {
            'universal_primitives': universal_primitives,
            'cross_language_primitives': cross_language_primitives,
            'language_specific_primitives': language_specific_primitives,
            'primitive_language_coverage': dict(primitive_language_coverage)
        }
    
    def test_primitive_based_translation_hypothesis(self, results: dict, universality_analysis: dict):
        """Test the hypothesis that primitives can enable translation."""
        logger.info("\n🔄 Testing Primitive-Based Translation Hypothesis")
        logger.info("=" * 60)
        
        # Find parallel examples (same primitives across languages)
        universal_primitives = universality_analysis['universal_primitives']
        
        if not universal_primitives:
            logger.warning("No universal primitives found - translation hypothesis may not be viable")
            return
        
        logger.info(f"✅ Found {len(universal_primitives)} universal primitives for translation")
        
        # Analyze translation potential
        translation_analysis = {
            'universal_primitives_count': len(universal_primitives),
            'translation_coverage': len(universal_primitives) / len(self.table.primitives),
            'languages_supported': len(results),
            'translation_potential': 'High' if len(universal_primitives) > 10 else 'Medium'
        }
        
        logger.info(f"📈 Translation Potential Analysis:")
        logger.info(f"  Universal primitives: {translation_analysis['universal_primitives_count']}")
        logger.info(f"  Coverage of total primitives: {translation_analysis['translation_coverage']:.1%}")
        logger.info(f"  Languages supported: {translation_analysis['languages_supported']}")
        logger.info(f"  Translation potential: {translation_analysis['translation_potential']}")
        
        # Show example parallel detections
        logger.info(f"\n📝 Example Parallel Detections:")
        for primitive in list(universal_primitives.keys())[:5]:  # Show first 5
            examples = []
            for lang in results.keys():
                lang_examples = results[lang]['detection_examples'].get(primitive, [])
                if lang_examples:
                    examples.append(f"{lang}: {lang_examples[0]['text']}")
            
            if examples:
                logger.info(f"  {primitive}:")
                for example in examples:
                    logger.info(f"    {example}")
        
        return translation_analysis
    
    def generate_comprehensive_report(self, results: dict, universality_analysis: dict, translation_analysis: dict):
        """Generate a comprehensive cross-language testing report."""
        report = {
            'summary': {
                'languages_tested': list(results.keys()),
                'total_texts_tested': sum(r['total_texts'] for r in results.values()),
                'average_detection_rate': sum(r['detection_rate'] for r in results.values()) / len(results),
                'universal_primitives_count': len(universality_analysis['universal_primitives']),
                'translation_potential': translation_analysis['translation_potential']
            },
            'language_results': results,
            'universality_analysis': universality_analysis,
            'translation_analysis': translation_analysis,
            'recommendations': self._generate_recommendations(results, universality_analysis, translation_analysis)
        }
        
        # Save report
        report_path = "data/cross_language_testing_report.json"
        
        # Convert numpy types and sets to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, set):
                return list(convert_numpy_types(v) for v in obj)
            elif hasattr(obj, 'dtype'):  # numpy array or scalar
                return float(obj) if hasattr(obj, 'item') else obj.tolist()
            else:
                return obj
        
        report = convert_numpy_types(report)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n💾 Comprehensive report saved to {report_path}")
        
        return report
    
    def _generate_recommendations(self, results: dict, universality_analysis: dict, translation_analysis: dict):
        """Generate recommendations based on cross-language testing results."""
        recommendations = []
        
        # Analyze detection rates
        avg_detection_rate = sum(r['detection_rate'] for r in results.values()) / len(results)
        
        if avg_detection_rate < 0.3:
            recommendations.append({
                'type': 'detection_improvement',
                'priority': 'high',
                'description': 'Low detection rates across languages - need to improve primitive detection',
                'action': 'Enhance embedding factorizer with language-specific examples'
            })
        
        # Analyze universality
        universal_count = len(universality_analysis['universal_primitives'])
        if universal_count < 10:
            recommendations.append({
                'type': 'universality_improvement',
                'priority': 'high',
                'description': f'Only {universal_count} universal primitives found - need more cross-language patterns',
                'action': 'Mine more universal patterns from multilingual corpora'
            })
        
        # Translation potential
        if translation_analysis['translation_potential'] == 'High':
            recommendations.append({
                'type': 'translation_development',
                'priority': 'medium',
                'description': 'High translation potential detected - ready for primitive-based translation prototype',
                'action': 'Develop primitive-based translation system'
            })
        
        # Language coverage
        if len(results) < 3:
            recommendations.append({
                'type': 'language_expansion',
                'priority': 'medium',
                'description': f'Only tested {len(results)} languages - need more language coverage',
                'action': 'Add more languages to testing corpus'
            })
        
        return recommendations
    
    def run_comprehensive_test(self):
        """Run the complete cross-language testing suite."""
        logger.info("🌍 Starting Comprehensive Cross-Language Testing")
        logger.info("=" * 60)
        
        # Load multilingual corpora
        corpora = self.load_multilingual_corpora()
        
        if not corpora:
            logger.error("No corpora loaded - cannot proceed with testing")
            return
        
        # Test primitive detection by language
        results = self.test_primitive_detection_by_language(corpora)
        
        # Analyze cross-language universality
        universality_analysis = self.analyze_cross_language_universality(results)
        
        # Test translation hypothesis
        translation_analysis = self.test_primitive_based_translation_hypothesis(results, universality_analysis)
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(results, universality_analysis, translation_analysis)
        
        # Print final summary
        logger.info(f"\n🎯 Cross-Language Testing Complete!")
        logger.info(f"  Languages tested: {len(results)}")
        logger.info(f"  Universal primitives: {len(universality_analysis['universal_primitives'])}")
        logger.info(f"  Translation potential: {translation_analysis['translation_potential']}")
        
        return report

def main():
    """Run cross-language testing."""
    tester = CrossLanguageTester()
    report = tester.run_comprehensive_test()
    
    # Print key insights
    if report:
        summary = report['summary']
        print(f"\n🎯 Key Insights:")
        print(f"  ✅ Tested {len(summary['languages_tested'])} languages")
        print(f"  ✅ Found {summary['universal_primitives_count']} universal primitives")
        print(f"  ✅ Translation potential: {summary['translation_potential']}")
        
        if summary['translation_potential'] == 'High':
            print(f"  🚀 Ready to develop primitive-based translation system!")
        else:
            print(f"  💡 Need to improve universality before translation development")

if __name__ == "__main__":
    main()
