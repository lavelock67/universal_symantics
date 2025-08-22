#!/usr/bin/env python3
"""
Test Enhanced NSM System.

This script tests the enhanced NSM legality validation and explication
generation, comparing it with the original system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import both systems for comparison
try:
    from src.nsm.explicator import NSMExplicator
    from src.nsm.enhanced_explicator import EnhancedNSMExplicator
    from src.nsm.enhanced_legality import EnhancedNSMLegalityValidator
except ImportError as e:
    logger.error(f"Failed to import NSM components: {e}")
    exit(1)

def test_legality_comparison():
    """Compare legality scores between original and enhanced systems."""
    logger.info("Testing legality comparison...")
    
    # Initialize both systems
    original_explicator = NSMExplicator()
    enhanced_explicator = EnhancedNSMExplicator()
    
    # Test cases with expected legality levels
    test_cases = [
        # High legality (NSM-like)
        ("en", "I see you", "High legality (NSM primes)"),
        ("en", "This is good", "High legality (NSM primes)"),
        ("en", "There is something", "High legality (NSM primes)"),
        ("en", "Someone wants something", "High legality (NSM primes)"),
        ("en", "Something happens because something else happens", "High legality (NSM primes)"),
        
        # Medium legality
        ("en", "The cat is on the mat", "Medium legality (location)"),
        ("en", "This is similar to that", "Medium legality (similarity)"),
        ("en", "A book is used for reading", "Medium legality (purpose)"),
        
        # Low legality
        ("en", "The complex algorithmic optimization procedure", "Low legality (technical)"),
        ("en", "Supercalifragilisticexpialidocious", "Low legality (nonsense)"),
        ("en", "However, nevertheless, furthermore, moreover, consequently", "Low legality (complex discourse)"),
        
        # Spanish test cases
        ("es", "Yo veo algo", "Spanish high legality"),
        ("es", "Esto es bueno", "Spanish high legality"),
        ("es", "Hay algo aquí", "Spanish high legality"),
        
        # French test cases
        ("fr", "Je vois quelque chose", "French high legality"),
        ("fr", "Ceci est bon", "French high legality"),
        ("fr", "Il y a quelque chose", "French high legality"),
    ]
    
    results = []
    
    for lang, text, description in test_cases:
        # Get original legality score
        original_score = original_explicator.legality_score(text, lang)
        original_legal = original_explicator.validate_legality(text, lang)
        
        # Get enhanced legality score
        enhanced_score = enhanced_explicator.legality_score(text, lang)
        enhanced_legal = enhanced_explicator.validate_legality(text, lang)
        
        # Get detailed analysis
        detailed_analysis = enhanced_explicator.detailed_legality_analysis(text, lang)
        
        result = {
            "text": text,
            "lang": lang,
            "description": description,
            "original_score": original_score,
            "original_legal": original_legal,
            "enhanced_score": enhanced_score,
            "enhanced_legal": enhanced_legal,
            "improvement": enhanced_score - original_score,
            "detailed_analysis": {
                "structural_score": detailed_analysis.structural_score,
                "semantic_score": detailed_analysis.semantic_score,
                "grammar_score": detailed_analysis.grammar_score,
                "violations": detailed_analysis.violations,
                "detected_primes": detailed_analysis.detected_primes
            }
        }
        
        results.append(result)
        
        logger.info(f"📝 {description}")
        logger.info(f"   Text: {text}")
        logger.info(f"   Original Score: {original_score:.3f} (Legal: {original_legal})")
        logger.info(f"   Enhanced Score: {enhanced_score:.3f} (Legal: {enhanced_legal})")
        logger.info(f"   Improvement: {enhanced_score - original_score:+.3f}")
        logger.info(f"   Detected Primes: {detailed_analysis.detected_primes}")
        if detailed_analysis.violations:
            logger.info(f"   Violations: {detailed_analysis.violations}")
        logger.info("")
    
    return results

def test_template_comparison():
    """Compare template generation between original and enhanced systems."""
    logger.info("Testing template comparison...")
    
    # Initialize both systems
    original_explicator = NSMExplicator()
    enhanced_explicator = EnhancedNSMExplicator()
    
    # Test primitives
    primitives = [
        "AtLocation", "HasProperty", "UsedFor", "SimilarTo", "DifferentFrom",
        "PartOf", "Causes", "Not", "Exist", "Can", "Want", "Think", "Feel",
        "See", "Say", "Do", "Happen", "Good", "Bad", "Big", "Small"
    ]
    
    languages = ["en", "es", "fr"]
    
    results = []
    
    for primitive in primitives:
        for lang in languages:
            try:
                original_template = original_explicator.template_for_primitive(primitive, lang)
                enhanced_template = enhanced_explicator.template_for_primitive(primitive, lang)
                
                # Evaluate legality of both templates
                original_legality = original_explicator.legality_score(original_template, lang)
                enhanced_legality = enhanced_explicator.legality_score(enhanced_template, lang)
                
                result = {
                    "primitive": primitive,
                    "lang": lang,
                    "original_template": original_template,
                    "enhanced_template": enhanced_template,
                    "original_legality": original_legality,
                    "enhanced_legality": enhanced_legality,
                    "legality_improvement": enhanced_legality - original_legality,
                    "template_different": original_template != enhanced_template
                }
                
                results.append(result)
                
                logger.info(f"🔧 {primitive} ({lang})")
                logger.info(f"   Original: {original_template} (Legality: {original_legality:.3f})")
                logger.info(f"   Enhanced: {enhanced_template} (Legality: {enhanced_legality:.3f})")
                logger.info(f"   Improvement: {enhanced_legality - original_legality:+.3f}")
                logger.info("")
                
            except Exception as e:
                logger.warning(f"Error testing {primitive} in {lang}: {e}")
    
    return results

def test_substitutability_evaluation():
    """Test substitutability evaluation on the expanded dataset."""
    logger.info("Testing substitutability evaluation...")
    
    # Load expanded dataset
    data_path = Path('data/parallel_test_data_1k.json')
    if not data_path.exists():
        logger.error("Expanded dataset not found")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    enhanced_explicator = EnhancedNSMExplicator()
    
    # Test on a subset for efficiency
    test_subset = {
        "en": dataset["en"][:20],  # First 20 sentences
        "es": dataset["es"][:20],
        "fr": dataset["fr"][:20]
    }
    
    results = []
    
    for lang in ["en", "es", "fr"]:
        for idx, sentence in enumerate(test_subset[lang]):
            try:
                # Generate explication
                explication = enhanced_explicator.generate_explication(sentence, lang)
                
                # Evaluate substitutability
                substitutability = enhanced_explicator.evaluate_substitutability(sentence, explication, lang)
                
                # Get legality scores
                sentence_legality = enhanced_explicator.legality_score(sentence, lang)
                explication_legality = enhanced_explicator.legality_score(explication, lang)
                
                result = {
                    "idx": idx,
                    "lang": lang,
                    "sentence": sentence,
                    "explication": explication,
                    "substitutability": substitutability,
                    "sentence_legality": sentence_legality,
                    "explication_legality": explication_legality,
                    "avg_legality": (sentence_legality + explication_legality) / 2
                }
                
                results.append(result)
                
                logger.info(f"📊 {lang.upper()} #{idx+1}")
                logger.info(f"   Sentence: {sentence}")
                logger.info(f"   Explication: {explication}")
                logger.info(f"   Substitutability: {substitutability:.3f}")
                logger.info(f"   Sentence Legality: {sentence_legality:.3f}")
                logger.info(f"   Explication Legality: {explication_legality:.3f}")
                logger.info("")
                
            except Exception as e:
                logger.warning(f"Error evaluating {lang} #{idx+1}: {e}")
    
    return results

def generate_comparison_report(legality_results, template_results, substitutability_results):
    """Generate a comprehensive comparison report."""
    logger.info("Generating comparison report...")
    
    # Calculate statistics
    legality_improvements = [r["improvement"] for r in legality_results]
    avg_legality_improvement = sum(legality_improvements) / len(legality_improvements) if legality_improvements else 0
    
    template_improvements = [r["legality_improvement"] for r in template_results]
    avg_template_improvement = sum(template_improvements) / len(template_improvements) if template_improvements else 0
    
    substitutability_scores = [r["substitutability"] for r in substitutability_results]
    avg_substitutability = sum(substitutability_scores) / len(substitutability_scores) if substitutability_scores else 0
    
    legality_scores = [r["avg_legality"] for r in substitutability_results]
    avg_legality = sum(legality_scores) / len(legality_scores) if legality_scores else 0
    
    report = {
        "metadata": {
            "test_type": "Enhanced NSM System Comparison",
            "description": "Comparison between original and enhanced NSM systems"
        },
        "summary": {
            "avg_legality_improvement": avg_legality_improvement,
            "avg_template_legality_improvement": avg_template_improvement,
            "avg_substitutability": avg_substitutability,
            "avg_legality_score": avg_legality,
            "total_test_cases": len(legality_results) + len(template_results) + len(substitutability_results)
        },
        "detailed_results": {
            "legality_comparison": legality_results,
            "template_comparison": template_results,
            "substitutability_evaluation": substitutability_results
        }
    }
    
    # Save report
    output_path = Path("data/enhanced_nsm_comparison_report.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved comparison report to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ENHANCED NSM SYSTEM COMPARISON SUMMARY")
    print("="*80)
    print(f"Average Legality Improvement: {avg_legality_improvement:+.3f}")
    print(f"Average Template Legality Improvement: {avg_template_improvement:+.3f}")
    print(f"Average Substitutability Score: {avg_substitutability:.3f}")
    print(f"Average Legality Score: {avg_legality:.3f}")
    print(f"Total Test Cases: {report['summary']['total_test_cases']}")
    print("="*80)
    
    return report

def main():
    """Run all enhanced NSM tests."""
    logger.info("Starting enhanced NSM system tests...")
    
    # Run all tests
    legality_results = test_legality_comparison()
    template_results = test_template_comparison()
    substitutability_results = test_substitutability_evaluation()
    
    # Generate comprehensive report
    report = generate_comparison_report(legality_results, template_results, substitutability_results)
    
    logger.info("Enhanced NSM system tests completed!")

if __name__ == "__main__":
    main()
