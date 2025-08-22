#!/usr/bin/env python3

"""
UMR Integration Test Script

This script demonstrates the Uniform Meaning Representation (UMR) integration
for cross-language primitive detection and evaluation.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.umr import UMRParser, UMRGenerator, UMREvaluator, UMRGraph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_umr_parsing():
    """Test UMR parsing on sample texts."""
    logger.info("Testing UMR parsing...")
    
    # Sample texts in different languages
    test_texts = {
        "en": [
            "The cat runs quickly.",
            "A big red car is in the garage.",
            "John eats an apple because he is hungry."
        ],
        "es": [
            "El gato corre rápidamente.",
            "Un coche rojo grande está en el garaje.",
            "Juan come una manzana porque tiene hambre."
        ],
        "fr": [
            "Le chat court rapidement.",
            "Une grande voiture rouge est dans le garage.",
            "Jean mange une pomme parce qu'il a faim."
        ]
    }
    
    results = {}
    
    for language, texts in test_texts.items():
        logger.info(f"Testing {language} parsing...")
        
        try:
            parser = UMRParser(language)
            graphs = parser.parse_batch(texts)
            
            language_results = []
            for i, (text, graph) in enumerate(zip(texts, graphs)):
                # Extract primitive patterns
                patterns = parser.extract_primitive_patterns(graph)
                
                result = {
                    "text": text,
                    "graph_id": graph.graph_id,
                    "num_nodes": len(graph.nodes),
                    "num_edges": len(graph.edges),
                    "patterns": patterns,
                    "graph_data": graph.to_dict()
                }
                language_results.append(result)
                
            results[language] = language_results
            
        except Exception as e:
            logger.error(f"Error testing {language} parsing: {e}")
            results[language] = []
    
    return results


def test_umr_generation():
    """Test UMR text generation."""
    logger.info("Testing UMR text generation...")
    
    # Create a simple test graph
    test_graph = UMRGraph("test_generation")
    
    # Add nodes
    from src.umr.graph import UMRNode, UMREdge
    
    # "The cat runs quickly" structure
    test_graph.add_node(UMRNode("n1", "cat", "concept", surface_form="cat"))
    test_graph.add_node(UMRNode("n2", "run", "event", surface_form="runs"))
    test_graph.add_node(UMRNode("n3", "quickly", "property", surface_form="quickly"))
    test_graph.add_node(UMRNode("n4", "the", "function", surface_form="the"))
    
    # Add edges
    test_graph.add_edge(UMREdge("n2", "n1", "ARG0"))  # run -> cat (subject)
    test_graph.add_edge(UMREdge("n2", "n3", "mod"))    # run -> quickly (modifier)
    test_graph.add_edge(UMREdge("n1", "n4", "det"))    # cat -> the (determiner)
    
    # Test generation in different languages
    results = {}
    
    for language in ["en", "es", "fr"]:
        try:
            generator = UMRGenerator(language)
            generated_text = generator.generate_text(test_graph)
            
            results[language] = {
                "original_graph": test_graph.to_dict(),
                "generated_text": generated_text
            }
            
        except Exception as e:
            logger.error(f"Error testing {language} generation: {e}")
            results[language] = {"error": str(e)}
    
    return results


def test_umr_evaluation():
    """Test UMR evaluation metrics."""
    logger.info("Testing UMR evaluation...")
    
    # Create test graphs for evaluation
    from src.umr.graph import UMRNode, UMREdge
    
    # Gold graph: "The cat runs quickly"
    gold_graph = UMRGraph("gold")
    gold_graph.add_node(UMRNode("n1", "cat", "concept", surface_form="cat"))
    gold_graph.add_node(UMRNode("n2", "run", "event", surface_form="runs"))
    gold_graph.add_node(UMRNode("n3", "quickly", "property", surface_form="quickly"))
    gold_graph.add_edge(UMREdge("n2", "n1", "ARG0"))
    gold_graph.add_edge(UMREdge("n2", "n3", "mod"))
    
    # Predicted graph: "The cat runs fast" (similar but not identical)
    pred_graph = UMRGraph("pred")
    pred_graph.add_node(UMRNode("n1", "cat", "concept", surface_form="cat"))
    pred_graph.add_node(UMRNode("n2", "run", "event", surface_form="runs"))
    pred_graph.add_node(UMRNode("n3", "fast", "property", surface_form="fast"))
    pred_graph.add_edge(UMREdge("n2", "n1", "ARG0"))
    pred_graph.add_edge(UMREdge("n2", "n3", "mod"))
    
    # Test evaluation
    evaluator = UMREvaluator()
    
    # Smatch score
    smatch_score = evaluator.compute_smatch_score(gold_graph, pred_graph)
    
    # Graph similarity
    similarity_score = evaluator.compute_graph_similarity(gold_graph, pred_graph)
    
    # Primitive metrics
    gold_metrics = evaluator.extract_primitive_metrics(gold_graph)
    pred_metrics = evaluator.extract_primitive_metrics(pred_graph)
    
    # Round-trip evaluation
    generator = UMRGenerator("en")
    generated_text = generator.generate_text(pred_graph)
    round_trip_score = evaluator.evaluate_round_trip("The cat runs quickly", generated_text)
    
    return {
        "smatch_score": smatch_score,
        "similarity_score": similarity_score,
        "gold_metrics": gold_metrics,
        "pred_metrics": pred_metrics,
        "round_trip_score": round_trip_score,
        "generated_text": generated_text
    }


def test_cross_language_umr():
    """Test cross-language UMR parsing and comparison."""
    logger.info("Testing cross-language UMR...")
    
    # Parallel sentences
    parallel_texts = {
        "en": "The big red car is in the garage.",
        "es": "El coche rojo grande está en el garaje.",
        "fr": "La grande voiture rouge est dans le garage."
    }
    
    # Parse in each language
    graphs = {}
    for language, text in parallel_texts.items():
        try:
            parser = UMRParser(language)
            graph = parser.parse_text(text)
            graphs[language] = graph
        except Exception as e:
            logger.error(f"Error parsing {language}: {e}")
            graphs[language] = UMRGraph()
    
    # Compare graphs across languages
    evaluator = UMREvaluator()
    comparisons = {}
    
    languages = list(graphs.keys())
    for i, lang1 in enumerate(languages):
        for lang2 in languages[i+1:]:
            if graphs[lang1] and graphs[lang2]:
                similarity = evaluator.compute_graph_similarity(graphs[lang1], graphs[lang2])
                comparisons[f"{lang1}_vs_{lang2}"] = {
                    "similarity": similarity,
                    "graph1_size": len(graphs[lang1].nodes),
                    "graph2_size": len(graphs[lang2].nodes)
                }
    
    return {
        "graphs": {lang: graph.to_dict() for lang, graph in graphs.items()},
        "comparisons": comparisons
    }


def main():
    """Run all UMR integration tests."""
    logger.info("Starting UMR integration tests...")
    
    results = {
        "parsing_test": test_umr_parsing(),
        "generation_test": test_umr_generation(),
        "evaluation_test": test_umr_evaluation(),
        "cross_language_test": test_cross_language_umr()
    }
    
    # Save results
    output_file = "umr_integration_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("UMR INTEGRATION TEST SUMMARY")
    print("="*50)
    
    # Parsing summary
    print("\nParsing Test Results:")
    for language, language_results in results["parsing_test"].items():
        if language_results:
            avg_nodes = sum(r["num_nodes"] for r in language_results) / len(language_results)
            avg_edges = sum(r["num_edges"] for r in language_results) / len(language_results)
            print(f"  {language.upper()}: {len(language_results)} texts, avg {avg_nodes:.1f} nodes, {avg_edges:.1f} edges")
        else:
            print(f"  {language.upper()}: No results (error)")
    
    # Generation summary
    print("\nGeneration Test Results:")
    for language, result in results["generation_test"].items():
        if "error" not in result:
            print(f"  {language.upper()}: '{result['generated_text']}'")
        else:
            print(f"  {language.upper()}: Error - {result['error']}")
    
    # Evaluation summary
    eval_results = results["evaluation_test"]
    print(f"\nEvaluation Test Results:")
    print(f"  Smatch F1: {eval_results['smatch_score']['f1']:.3f}")
    print(f"  Graph Similarity: {eval_results['similarity_score']['overall_similarity']:.3f}")
    print(f"  Round-trip Score: {eval_results['round_trip_score']['content_preservation']:.3f}")
    
    # Cross-language summary
    print(f"\nCross-language Test Results:")
    for comparison, data in results["cross_language_test"]["comparisons"].items():
        print(f"  {comparison}: {data['similarity']['overall_similarity']:.3f}")
    
    print("\n" + "="*50)
    print("UMR integration test completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
