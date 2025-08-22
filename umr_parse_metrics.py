#!/usr/bin/env python3

"""
UMR Parse Metrics Integration

This script integrates Uniform Meaning Representation (UMR) parsing with the
existing primitive detection system to provide comprehensive cross-language
semantic analysis and evaluation.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.umr import UMRParser, UMRGenerator, UMREvaluator, UMRGraph
from src.table.algebra import PrimitiveAlgebra
from src.table.schema import PeriodicTable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UMRParseMetrics:
    """Integrates UMR parsing with primitive detection metrics."""
    
    def __init__(self, primitive_table_path: str = "data/primitives_with_semantic.json"):
        """Initialize UMR parse metrics.
        
        Args:
            primitive_table_path: Path to primitive table JSON file
        """
        self.primitive_table_path = primitive_table_path
        self.table = None
        self.algebra = None
        self.parsers = {}
        self.generators = {}
        self.evaluator = UMREvaluator()
        
        self._load_primitives()
        self._initialize_umr_components()
        
    def _load_primitives(self):
        """Load primitive table and algebra."""
        try:
            with open(self.primitive_table_path, 'r') as f:
                data = json.load(f)
            
            # Store primitive data directly
            self.primitive_data = data.get("primitives", [])
            self.primitive_categories = data.get("categories", [])
            
            # Create a simple table structure for compatibility
            self.table = PeriodicTable()
            self.algebra = PrimitiveAlgebra(self.table)
            
            logger.info(f"Loaded {len(self.primitive_data)} primitives")
            
        except Exception as e:
            logger.error(f"Error loading primitives: {e}")
            self.primitive_data = []
            self.primitive_categories = []
            self.table = PeriodicTable()
            self.algebra = PrimitiveAlgebra(self.table)
    
    def _initialize_umr_components(self):
        """Initialize UMR parsers and generators for supported languages."""
        languages = ["en", "es", "fr"]
        
        for language in languages:
            try:
                self.parsers[language] = UMRParser(language)
                self.generators[language] = UMRGenerator(language)
                logger.info(f"Initialized UMR components for {language}")
            except Exception as e:
                logger.warning(f"Could not initialize UMR components for {language}: {e}")
    
    def analyze_text_with_umr(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Analyze text using UMR parsing and primitive detection.
        
        Args:
            text: Input text to analyze
            language: Language code
            
        Returns:
            Dictionary with UMR analysis and primitive metrics
        """
        if language not in self.parsers:
            raise ValueError(f"Language {language} not supported")
        
        # Parse text to UMR graph
        parser = self.parsers[language]
        graph = parser.parse_text(text)
        
        # Extract primitive patterns
        patterns = parser.extract_primitive_patterns(graph)
        
        # Extract primitive metrics
        metrics = self.evaluator.extract_primitive_metrics(graph)
        
        # Map UMR patterns to primitive table
        primitive_matches = self._map_umr_to_primitives(graph, patterns)
        
        return {
            "text": text,
            "language": language,
            "umr_graph": graph.to_dict(),
            "patterns": patterns,
            "metrics": metrics,
            "primitive_matches": primitive_matches,
            "graph_stats": {
                "num_nodes": len(graph.nodes),
                "num_edges": len(graph.edges),
                "node_types": {node.node_type: sum(1 for n in graph.nodes.values() if n.node_type == node.node_type) 
                              for node in graph.nodes.values()},
                "relation_types": {edge.relation: sum(1 for e in graph.edges if e.relation == edge.relation) 
                                 for edge in graph.edges}
            }
        }
    
    def _map_umr_to_primitives(self, graph: UMRGraph, patterns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
        """Map UMR patterns to primitive table entries.
        
        Args:
            graph: UMR graph
            patterns: Extracted primitive patterns
            
        Returns:
            Dictionary mapping primitive categories to matched primitives
        """
        matches = defaultdict(list)
        
        # Map spatial patterns
        for pattern in patterns.get("spatial", []):
            matches["spatial"].extend([
                "AtLocation", "HasLocation", "LocatedNear", "LocatedFar"
            ])
        
        # Map temporal patterns
        for pattern in patterns.get("temporal", []):
            matches["temporal"].extend([
                "Before", "After", "During", "AtTime", "HasDuration"
            ])
        
        # Map causal patterns
        for pattern in patterns.get("causal", []):
            matches["causal"].extend([
                "Causes", "CausesDesire", "HasPrerequisite", "HasEffect"
            ])
        
        # Map logical patterns
        for pattern in patterns.get("logical", []):
            matches["logical"].extend([
                "And", "Or", "Not", "Implies", "Equivalent"
            ])
        
        # Map quantitative patterns
        for pattern in patterns.get("quantitative", []):
            matches["quantitative"].extend([
                "HasQuantity", "MoreThan", "LessThan", "Equal", "HasMeasure"
            ])
        
        # Map structural patterns
        for pattern in patterns.get("structural", []):
            matches["structural"].extend([
                "PartOf", "HasPart", "HasProperty", "IsA", "InstanceOf"
            ])
        
        return dict(matches)
    
    def evaluate_cross_language_umr(self, parallel_texts: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate UMR parsing across multiple languages.
        
        Args:
            parallel_texts: Dictionary mapping language codes to parallel texts
            
        Returns:
            Dictionary with cross-language UMR evaluation results
        """
        results = {}
        graphs = {}
        
        # Parse texts in each language
        for language, text in parallel_texts.items():
            if language in self.parsers:
                try:
                    analysis = self.analyze_text_with_umr(text, language)
                    results[language] = analysis
                    graphs[language] = UMRGraph.from_dict(analysis["umr_graph"])
                except Exception as e:
                    logger.error(f"Error analyzing {language} text: {e}")
                    results[language] = {"error": str(e)}
            else:
                results[language] = {"error": f"Language {language} not supported"}
        
        # Cross-language comparisons
        comparisons = {}
        languages = list(graphs.keys())
        
        for i, lang1 in enumerate(languages):
            for lang2 in languages[i+1:]:
                if lang1 in graphs and lang2 in graphs:
                    similarity = self.evaluator.compute_graph_similarity(graphs[lang1], graphs[lang2])
                    comparisons[f"{lang1}_vs_{lang2}"] = similarity
        
        # Round-trip evaluation
        round_trip_results = {}
        for language, text in parallel_texts.items():
            if language in self.generators and language in graphs:
                try:
                    generator = self.generators[language]
                    generated_text = generator.generate_text(graphs[language])
                    round_trip_score = self.evaluator.evaluate_round_trip(text, generated_text)
                    round_trip_results[language] = {
                        "original": text,
                        "generated": generated_text,
                        "scores": round_trip_score
                    }
                except Exception as e:
                    logger.error(f"Error in round-trip evaluation for {language}: {e}")
                    round_trip_results[language] = {"error": str(e)}
        
        return {
            "language_analyses": results,
            "cross_language_comparisons": comparisons,
            "round_trip_evaluation": round_trip_results,
            "summary": {
                "languages_processed": len([r for r in results.values() if "error" not in r]),
                "avg_graph_size": sum(len(r["umr_graph"]["nodes"]) for r in results.values() if "error" not in r) / max(1, len([r for r in results.values() if "error" not in r])),
                "avg_similarity": sum(c["overall_similarity"] for c in comparisons.values()) / max(1, len(comparisons))
            }
        }
    
    def generate_umr_report(self, parallel_texts: Dict[str, str], output_file: str = "umr_analysis_report.json"):
        """Generate comprehensive UMR analysis report.
        
        Args:
            parallel_texts: Dictionary mapping language codes to parallel texts
            output_file: Output file path for the report
        """
        logger.info("Generating UMR analysis report...")
        
        # Perform cross-language evaluation
        evaluation_results = self.evaluate_cross_language_umr(parallel_texts)
        
        # Add metadata
        report = {
            "metadata": {
                "timestamp": str(Path().cwd()),
                "num_languages": len(parallel_texts),
                "num_primitives": len(self.primitive_data),
                "supported_languages": list(self.parsers.keys())
            },
            "evaluation_results": evaluation_results,
            "primitive_table_info": {
                "categories": self.primitive_categories,
                "primitive_counts": {cat: len([p for p in self.primitive_data if p.get("kind") == cat]) 
                                   for cat in self.primitive_categories}
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"UMR analysis report saved to {output_file}")
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
    def _print_report_summary(self, report: Dict[str, Any]):
        """Print a summary of the UMR analysis report.
        
        Args:
            report: UMR analysis report
        """
        print("\n" + "="*60)
        print("UMR ANALYSIS REPORT SUMMARY")
        print("="*60)
        
        eval_results = report["evaluation_results"]
        summary = eval_results["summary"]
        
        print(f"\nLanguages Processed: {summary['languages_processed']}")
        print(f"Average Graph Size: {summary['avg_graph_size']:.1f} nodes")
        print(f"Average Cross-language Similarity: {summary['avg_similarity']:.3f}")
        
        print(f"\nCross-language Comparisons:")
        for comparison, similarity in eval_results["cross_language_comparisons"].items():
            print(f"  {comparison}: {similarity['overall_similarity']:.3f}")
        
        print(f"\nRound-trip Evaluation:")
        for language, result in eval_results["round_trip_evaluation"].items():
            if "error" not in result:
                scores = result["scores"]
                print(f"  {language.upper()}: Content preservation = {scores['content_preservation']:.3f}")
            else:
                print(f"  {language.upper()}: Error - {result['error']}")
        
        print(f"\nPrimitive Categories:")
        for category, count in report["primitive_table_info"]["primitive_counts"].items():
            print(f"  {category}: {count} primitives")
        
        print("\n" + "="*60)


def main():
    """Main function to demonstrate UMR parse metrics integration."""
    # Sample parallel texts for testing
    parallel_texts = {
        "en": "The big red car is parked in the garage near the house.",
        "es": "El coche rojo grande está aparcado en el garaje cerca de la casa.",
        "fr": "La grande voiture rouge est garée dans le garage près de la maison."
    }
    
    # Initialize UMR parse metrics
    umr_metrics = UMRParseMetrics()
    
    # Generate comprehensive report
    report = umr_metrics.generate_umr_report(parallel_texts)
    
    # Test individual text analysis
    print(f"\nIndividual Text Analysis Example:")
    analysis = umr_metrics.analyze_text_with_umr(parallel_texts["en"], "en")
    print(f"English text: {analysis['text']}")
    print(f"Graph nodes: {analysis['graph_stats']['num_nodes']}")
    print(f"Graph edges: {analysis['graph_stats']['num_edges']}")
    print(f"Primitive matches: {len(analysis['primitive_matches'])} categories")


if __name__ == "__main__":
    main()
