#!/usr/bin/env python3
"""
Threshold calibration script for primitive detection system.

Sweeps through different threshold values and reports per-primitive precision/recall
metrics to optimize detection accuracy.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.table.algebra import PrimitiveAlgebra
    from src.table.schema import PeriodicTable
    from src.nsm.explicator import NSMExplicator
    from src.detect.srl_ud_detectors import detect_primitives_multilingual
    from src.umr.parser import UMRParser
    from src.umr.evaluator import UMREvaluator
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    logger.error("Make sure you're running from the project root directory")
    exit(1)


class ThresholdCalibrator:
    """Calibrates thresholds for primitive detection across different components."""
    
    def __init__(self):
        """Initialize the calibrator with system components."""
        # Create a basic periodic table for testing
        self.periodic_table = PeriodicTable()
        self.algebra = PrimitiveAlgebra(self.periodic_table)
        self.nsm_explicator = NSMExplicator()
        self.umr_parser = UMRParser()
        self.umr_evaluator = UMREvaluator()
        
        # Test data
        self.test_texts = self._load_test_data()
        
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data for calibration."""
        # Simple test cases for calibration
        test_cases = [
            {
                "text": "The cat is on the mat",
                "expected_primitives": ["AtLocation", "Entity"],
                "language": "en"
            },
            {
                "text": "El gato está en la alfombra",
                "expected_primitives": ["AtLocation", "Entity"],
                "language": "es"
            },
            {
                "text": "Le chat est sur le tapis",
                "expected_primitives": ["AtLocation", "Entity"],
                "language": "fr"
            },
            {
                "text": "The dog runs quickly",
                "expected_primitives": ["Entity", "Action", "Property"],
                "language": "en"
            },
            {
                "text": "El perro corre rápidamente",
                "expected_primitives": ["Entity", "Action", "Property"],
                "language": "es"
            },
            {
                "text": "Le chien court rapidement",
                "expected_primitives": ["Entity", "Action", "Property"],
                "language": "fr"
            },
            {
                "text": "This is similar to that",
                "expected_primitives": ["SimilarTo", "Entity"],
                "language": "en"
            },
            {
                "text": "Esto es similar a eso",
                "expected_primitives": ["SimilarTo", "Entity"],
                "language": "es"
            },
            {
                "text": "Ceci est similaire à cela",
                "expected_primitives": ["SimilarTo", "Entity"],
                "language": "fr"
            }
        ]
        
        # Load parallel test data if available
        parallel_path = Path("data/parallel_test_data.json")
        if parallel_path.exists():
            try:
                with open(parallel_path, 'r', encoding='utf-8') as f:
                    parallel_data = json.load(f)
                    # Add parallel test cases
                    for item in parallel_data.get("data", []):
                        test_cases.append({
                            "text": item.get("en", ""),
                            "expected_primitives": item.get("primitives", []),
                            "language": "en"
                        })
                        test_cases.append({
                            "text": item.get("es", ""),
                            "expected_primitives": item.get("primitives", []),
                            "language": "es"
                        })
                        test_cases.append({
                            "text": item.get("fr", ""),
                            "expected_primitives": item.get("primitives", []),
                            "language": "fr"
                        })
            except Exception as e:
                logger.warning(f"Failed to load parallel test data: {e}")
        
        return test_cases
    
    def calibrate_similarity_threshold(self) -> Dict[str, Any]:
        """Calibrate the similarity threshold for primitive detection."""
        logger.info("Calibrating similarity threshold...")
        
        thresholds = np.arange(0.1, 0.9, 0.05)  # 0.1 to 0.85 in 0.05 steps
        results = []
        
        for threshold in thresholds:
            logger.info(f"Testing threshold: {threshold:.2f}")
            
            # Temporarily set environment variable
            os.environ["PERIODIC_SIM_THRESHOLD"] = str(threshold)
            
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            valid_tests = 0
            
            for test_case in self.test_texts:
                try:
                    # Test algebra-based detection
                    detected_primitives = self.algebra._infer_primitives_from_text(test_case["text"])
                    detected_names = [p.name for p in detected_primitives]
                    
                    # Calculate metrics
                    expected = set(test_case["expected_primitives"])
                    detected = set(detected_names)
                    
                    if len(detected) > 0:
                        precision = len(expected & detected) / len(detected)
                    else:
                        precision = 0.0
                        
                    if len(expected) > 0:
                        recall = len(expected & detected) / len(expected)
                    else:
                        recall = 0.0
                        
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0.0
                    
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1
                    valid_tests += 1
                    
                except Exception as e:
                    logger.warning(f"Error testing case '{test_case['text']}': {e}")
                    continue
            
            if valid_tests > 0:
                avg_precision = total_precision / valid_tests
                avg_recall = total_recall / valid_tests
                avg_f1 = total_f1 / valid_tests
                
                results.append({
                    "threshold": float(threshold),
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": avg_f1,
                    "valid_tests": valid_tests
                })
        
        # Find optimal threshold
        if results:
            best_result = max(results, key=lambda x: x["f1"])
            logger.info(f"Optimal threshold: {best_result['threshold']:.2f} (F1: {best_result['f1']:.3f})")
        
        return {
            "results": results,
            "optimal_threshold": best_result["threshold"] if results else 0.3,
            "test_cases": len(self.test_texts)
        }
    
    def calibrate_per_primitive(self) -> Dict[str, Any]:
        """Calibrate thresholds per primitive type."""
        logger.info("Calibrating per-primitive thresholds...")
        
        # Group test cases by primitive type
        primitive_groups = {}
        for test_case in self.test_texts:
            for primitive in test_case["expected_primitives"]:
                if primitive not in primitive_groups:
                    primitive_groups[primitive] = []
                primitive_groups[primitive].append(test_case)
        
        per_primitive_results = {}
        
        for primitive, test_cases in primitive_groups.items():
            logger.info(f"Calibrating {primitive}...")
            
            thresholds = np.arange(0.1, 0.9, 0.1)  # 0.1 to 0.8 in 0.1 steps
            primitive_results = []
            
            for threshold in thresholds:
                os.environ["PERIODIC_SIM_THRESHOLD"] = str(threshold)
                
                total_precision = 0.0
                total_recall = 0.0
                valid_tests = 0
                
                for test_case in test_cases:
                    try:
                        detected_primitives = self.algebra._infer_primitives_from_text(test_case["text"])
                        detected_names = [p.name for p in detected_primitives]
                        
                        expected = set(test_case["expected_primitives"])
                        detected = set(detected_names)
                        
                        # Check if target primitive was detected
                        target_detected = primitive in detected
                        target_expected = primitive in expected
                        
                        if target_expected:
                            recall = 1.0 if target_detected else 0.0
                            total_recall += recall
                            valid_tests += 1
                        
                        if target_detected:
                            precision = 1.0 if target_expected else 0.0
                            total_precision += precision
                        
                    except Exception as e:
                        logger.warning(f"Error in primitive-specific test: {e}")
                        continue
                
                if valid_tests > 0:
                    avg_precision = total_precision / max(1, len([c for c in test_cases if primitive in c["expected_primitives"]]))
                    avg_recall = total_recall / valid_tests
                    
                    if avg_precision + avg_recall > 0:
                        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                    else:
                        f1 = 0.0
                    
                    primitive_results.append({
                        "threshold": float(threshold),
                        "precision": avg_precision,
                        "recall": avg_recall,
                        "f1": f1,
                        "test_count": valid_tests
                    })
            
            if primitive_results:
                best_result = max(primitive_results, key=lambda x: x["f1"])
                per_primitive_results[primitive] = {
                    "optimal_threshold": best_result["threshold"],
                    "optimal_f1": best_result["f1"],
                    "all_results": primitive_results
                }
        
        return per_primitive_results
    
    def calibrate_ud_thresholds(self) -> Dict[str, Any]:
        """Calibrate UD-based detection thresholds."""
        logger.info("Calibrating UD detection thresholds...")
        
        # Test UD patterns with different confidence thresholds
        ud_test_cases = [
            {"text": "The cat is on the mat", "pattern": "AtLocation"},
            {"text": "This is similar to that", "pattern": "SimilarTo"},
            {"text": "The tool is used for cutting", "pattern": "UsedFor"},
            {"text": "The red car", "pattern": "HasProperty"}
        ]
        
        thresholds = np.arange(0.1, 0.9, 0.1)
        ud_results = []
        
        for threshold in thresholds:
            logger.info(f"Testing UD threshold: {threshold:.1f}")
            
            total_precision = 0.0
            total_recall = 0.0
            valid_tests = 0
            
            for test_case in ud_test_cases:
                try:
                    # Test UD detection (simplified)
                    detected_patterns = detect_primitives_multilingual(test_case["text"])
                    
                    expected_pattern = test_case["pattern"]
                    
                    precision = 1.0 if expected_pattern in detected_patterns else 0.0
                    recall = 1.0 if expected_pattern in detected_patterns else 0.0
                    
                    total_precision += precision
                    total_recall += recall
                    valid_tests += 1
                    
                except Exception as e:
                    logger.warning(f"Error in UD test: {e}")
                    continue
            
            if valid_tests > 0:
                avg_precision = total_precision / valid_tests
                avg_recall = total_recall / valid_tests
                
                if avg_precision + avg_recall > 0:
                    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                else:
                    f1 = 0.0
                
                ud_results.append({
                    "threshold": float(threshold),
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": f1,
                    "test_count": valid_tests
                })
        
        return {
            "ud_results": ud_results,
            "optimal_ud_threshold": max(ud_results, key=lambda x: x["f1"])["threshold"] if ud_results else 0.5
        }
    
    def run_full_calibration(self) -> Dict[str, Any]:
        """Run complete threshold calibration."""
        logger.info("Starting full threshold calibration...")
        
        results = {
            "similarity_threshold": self.calibrate_similarity_threshold(),
            "per_primitive": self.calibrate_per_primitive(),
            "ud_thresholds": self.calibrate_ud_thresholds(),
            "timestamp": str(np.datetime64('now')),
            "test_data_size": len(self.test_texts)
        }
        
        # Save results
        output_path = Path("data/threshold_calibration.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Calibration results saved to {output_path}")
        
        # Print summary
        self._print_calibration_summary(results)
        
        return results
    
    def _print_calibration_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of calibration results."""
        print("\n" + "="*60)
        print("THRESHOLD CALIBRATION SUMMARY")
        print("="*60)
        
        # Similarity threshold
        sim_results = results["similarity_threshold"]
        print(f"\n📊 Similarity Threshold:")
        print(f"   Optimal: {sim_results['optimal_threshold']:.2f}")
        print(f"   Test cases: {sim_results['test_cases']}")
        
        # Per-primitive results
        print(f"\n📊 Per-Primitive Results:")
        for primitive, data in results["per_primitive"].items():
            print(f"   {primitive}: threshold={data['optimal_threshold']:.2f}, F1={data['optimal_f1']:.3f}")
        
        # UD thresholds
        ud_results = results["ud_thresholds"]
        print(f"\n📊 UD Detection:")
        print(f"   Optimal threshold: {ud_results['optimal_ud_threshold']:.2f}")
        
        print("\n" + "="*60)


def main():
    """Main calibration function."""
    try:
        calibrator = ThresholdCalibrator()
        results = calibrator.run_full_calibration()
        
        # Suggest .env updates
        print("\n🔧 Suggested .env Updates:")
        print(f"PERIODIC_SIM_THRESHOLD={results['similarity_threshold']['optimal_threshold']:.2f}")
        print(f"PERIODIC_UD_THRESHOLD={results['ud_thresholds']['optimal_ud_threshold']:.2f}")
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
