#!/usr/bin/env python3

"""
UMR to Text Generation - Interlingual Baseline

This script demonstrates UMR-based translation as an interlingual baseline,
converting text to UMR graphs and back to text in different languages.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.umr import UMRParser, UMRGenerator, UMREvaluator, UMRGraph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UMRToTextTranslator:
    """UMR-based translation system using interlingual baseline."""
    
    def __init__(self):
        """Initialize the UMR-to-text translator."""
        self.parsers = {}
        self.generators = {}
        self.evaluator = UMREvaluator()
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize UMR parsers and generators for supported languages."""
        languages = ["en", "es", "fr"]
        
        for language in languages:
            try:
                self.parsers[language] = UMRParser(language)
                self.generators[language] = UMRGenerator(language)
                logger.info(f"Initialized UMR components for {language}")
            except Exception as e:
                logger.warning(f"Could not initialize UMR components for {language}: {e}")
    
    def translate_via_umr(self, source_text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate text via UMR interlingual representation.
        
        Args:
            source_text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dictionary with translation results and metrics
        """
        if source_lang not in self.parsers:
            raise ValueError(f"Source language {source_lang} not supported")
        if target_lang not in self.generators:
            raise ValueError(f"Target language {target_lang} not supported")
        
        # Step 1: Parse source text to UMR graph
        logger.info(f"Parsing {source_lang} text to UMR graph...")
        source_parser = self.parsers[source_lang]
        umr_graph = source_parser.parse_text(source_text)
        
        # Step 2: Generate target text from UMR graph
        logger.info(f"Generating {target_lang} text from UMR graph...")
        target_generator = self.generators[target_lang]
        target_text = target_generator.generate_text(umr_graph)
        
        # Step 3: Evaluate translation quality
        logger.info("Evaluating translation quality...")
        round_trip_score = self.evaluator.evaluate_round_trip(source_text, target_text)
        
        # Step 4: Analyze UMR graph structure
        graph_metrics = self.evaluator.extract_primitive_metrics(umr_graph)
        
        return {
            "source_text": source_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "umr_graph": umr_graph.to_dict(),
            "target_text": target_text,
            "translation_metrics": round_trip_score,
            "graph_metrics": graph_metrics,
            "graph_stats": {
                "num_nodes": len(umr_graph.nodes),
                "num_edges": len(umr_graph.edges),
                "node_types": {node.node_type: sum(1 for n in umr_graph.nodes.values() if n.node_type == node.node_type) 
                              for node in umr_graph.nodes.values()},
                "relation_types": {edge.relation: sum(1 for e in umr_graph.edges if e.relation == edge.relation) 
                                 for edge in umr_graph.edges}
            }
        }
    
    def batch_translate(self, parallel_texts: Dict[str, str], target_lang: str = "en") -> Dict[str, Any]:
        """Translate multiple texts to a target language via UMR.
        
        Args:
            parallel_texts: Dictionary mapping source languages to texts
            target_lang: Target language for all translations
            
        Returns:
            Dictionary with batch translation results
        """
        results = {}
        
        for source_lang, source_text in parallel_texts.items():
            try:
                result = self.translate_via_umr(source_text, source_lang, target_lang)
                results[source_lang] = result
            except Exception as e:
                logger.error(f"Error translating {source_lang} text: {e}")
                results[source_lang] = {"error": str(e)}
        
        return results
    
    def evaluate_translation_quality(self, translations: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the quality of UMR-based translations.
        
        Args:
            translations: Dictionary with translation results
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            "content_preservation": [],
            "text_similarity": [],
            "graph_complexity": [],
            "translation_success_rate": 0
        }
        
        successful_translations = 0
        total_translations = len(translations)
        
        for lang, result in translations.items():
            if "error" not in result:
                successful_translations += 1
                
                # Content preservation
                content_pres = result["translation_metrics"]["content_preservation"]
                quality_metrics["content_preservation"].append(content_pres)
                
                # Text similarity
                text_sim = result["translation_metrics"]["text_similarity"]
                quality_metrics["text_similarity"].append(text_sim)
                
                # Graph complexity (number of nodes)
                graph_complexity = result["graph_stats"]["num_nodes"]
                quality_metrics["graph_complexity"].append(graph_complexity)
        
        # Calculate averages
        if quality_metrics["content_preservation"]:
            quality_metrics["avg_content_preservation"] = sum(quality_metrics["content_preservation"]) / len(quality_metrics["content_preservation"])
        else:
            quality_metrics["avg_content_preservation"] = 0.0
            
        if quality_metrics["text_similarity"]:
            quality_metrics["avg_text_similarity"] = sum(quality_metrics["text_similarity"]) / len(quality_metrics["text_similarity"])
        else:
            quality_metrics["avg_text_similarity"] = 0.0
            
        if quality_metrics["graph_complexity"]:
            quality_metrics["avg_graph_complexity"] = sum(quality_metrics["graph_complexity"]) / len(quality_metrics["graph_complexity"])
        else:
            quality_metrics["avg_graph_complexity"] = 0.0
        
        quality_metrics["translation_success_rate"] = successful_translations / total_translations if total_translations > 0 else 0.0
        
        return quality_metrics


def main():
    """Demonstrate UMR-to-text translation capabilities."""
    # Sample parallel texts for testing
    parallel_texts = {
        "en": "The cat is sleeping on the red chair in the living room.",
        "es": "El gato está durmiendo en la silla roja en la sala de estar.",
        "fr": "Le chat dort sur la chaise rouge dans le salon."
    }
    
    # Initialize translator
    translator = UMRToTextTranslator()
    
    print("="*70)
    print("UMR-TO-TEXT TRANSLATION DEMONSTRATION")
    print("="*70)
    
    # Test individual translations
    print("\nIndividual Translation Examples:")
    print("-" * 50)
    
    for source_lang, source_text in parallel_texts.items():
        if source_lang != "en":  # Translate to English
            try:
                result = translator.translate_via_umr(source_text, source_lang, "en")
                print(f"\n{source_lang.upper()} → EN:")
                print(f"  Source: {result['source_text']}")
                print(f"  Target: {result['target_text']}")
                print(f"  Content preservation: {result['translation_metrics']['content_preservation']:.3f}")
                print(f"  Graph nodes: {result['graph_stats']['num_nodes']}")
            except Exception as e:
                print(f"\n{source_lang.upper()} → EN: Error - {e}")
    
    # Test batch translation
    print("\n" + "="*70)
    print("BATCH TRANSLATION TO ENGLISH")
    print("="*70)
    
    batch_results = translator.batch_translate(parallel_texts, "en")
    
    for source_lang, result in batch_results.items():
        if "error" not in result:
            print(f"\n{source_lang.upper()} → EN:")
            print(f"  Source: {result['source_text']}")
            print(f"  Target: {result['target_text']}")
            print(f"  Quality: {result['translation_metrics']['content_preservation']:.3f}")
        else:
            print(f"\n{source_lang.upper()} → EN: Error - {result['error']}")
    
    # Evaluate overall quality
    print("\n" + "="*70)
    print("TRANSLATION QUALITY EVALUATION")
    print("="*70)
    
    quality_metrics = translator.evaluate_translation_quality(batch_results)
    
    print(f"Translation Success Rate: {quality_metrics['translation_success_rate']:.1%}")
    print(f"Average Content Preservation: {quality_metrics['avg_content_preservation']:.3f}")
    print(f"Average Text Similarity: {quality_metrics['avg_text_similarity']:.3f}")
    print(f"Average Graph Complexity: {quality_metrics['avg_graph_complexity']:.1f} nodes")
    
    # Save results
    output_file = "umr_translation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "batch_results": batch_results,
            "quality_metrics": quality_metrics,
            "metadata": {
                "num_languages": len(parallel_texts),
                "target_language": "en",
                "supported_languages": list(translator.parsers.keys())
            }
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
    print("\n" + "="*70)
    print("UMR-to-text translation demonstration completed!")
    print("="*70)


if __name__ == "__main__":
    main()
