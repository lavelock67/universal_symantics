#!/usr/bin/env python3
"""
Combined report generator for primitive detection system.

Integrates UD recalls, NSM legality, UMR metrics, and other evaluation metrics
into a comprehensive report.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
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
    from src.umr.generator import UMRGenerator
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    logger.error("Make sure you're running from the project root directory")
    exit(1)


class CombinedReportGenerator:
    """Generates comprehensive reports combining multiple evaluation metrics."""
    
    def __init__(self):
        """Initialize the report generator."""
        # Create a basic periodic table for testing
        self.periodic_table = PeriodicTable()
        self.algebra = PrimitiveAlgebra(self.periodic_table)
        self.nsm_explicator = NSMExplicator()
        self.umr_parser = UMRParser()
        self.umr_evaluator = UMREvaluator()
        self.umr_generator = UMRGenerator()
        
        # Load test data
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data for evaluation."""
        # Parallel test data (try expanded first, then fallback to original)
        parallel_data = {}
        parallel_path = Path("data/expanded_parallel_test_data.json")
        if not parallel_path.exists():
            parallel_path = Path("data/parallel_test_data.json")
        if parallel_path.exists():
            try:
                with open(parallel_path, 'r', encoding='utf-8') as f:
                    parallel_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load parallel test data: {e}")
        
        # Gold labels (try expanded first, then fallback to original)
        gold_data = {}
        gold_path = Path("data/expanded_parallel_gold.json")
        if not gold_path.exists():
            gold_path = Path("data/parallel_gold.json")
        if gold_path.exists():
            try:
                with open(gold_path, 'r', encoding='utf-8') as f:
                    gold_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load gold data: {e}")
        
        # Threshold calibration results
        calibration_data = {}
        calibration_path = Path("data/threshold_calibration.json")
        if calibration_path.exists():
            try:
                with open(calibration_path, 'r', encoding='utf-8') as f:
                    calibration_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load calibration data: {e}")
        
        return {
            "parallel": parallel_data,
            "gold": gold_data,
            "calibration": calibration_data
        }
    
    def evaluate_ud_recalls(self) -> Dict[str, Any]:
        """Evaluate UD pattern detection recalls."""
        logger.info("Evaluating UD pattern detection recalls...")
        
        if not self.test_data["parallel"]:
            return {"error": "No parallel test data available"}
        
        results = {
            "languages": {},
            "patterns": {},
            "overall": {}
        }
        
        # Test each language
        for lang in ["en", "es", "fr"]:
            lang_results = {
                "total_sentences": 0,
                "detected_patterns": 0,
                "recall": 0.0,
                "pattern_breakdown": {}
            }
            
            for item in self.test_data["parallel"].get("data", []):
                text = item.get(lang, "")
                if not text:
                    continue
                
                lang_results["total_sentences"] += 1
                
                # Detect patterns
                try:
                    detected_patterns = detect_primitives_multilingual(text)
                    lang_results["detected_patterns"] += len(detected_patterns)
                    
                    # Count by pattern type
                    for pattern in detected_patterns:
                        if pattern not in lang_results["pattern_breakdown"]:
                            lang_results["pattern_breakdown"][pattern] = 0
                        lang_results["pattern_breakdown"][pattern] += 1
                        
                except Exception as e:
                    logger.warning(f"Error detecting patterns in {lang}: {e}")
                    continue
            
            # Calculate recall
            if lang_results["total_sentences"] > 0:
                lang_results["recall"] = lang_results["detected_patterns"] / lang_results["total_sentences"]
            
            results["languages"][lang] = lang_results
        
        # Aggregate pattern statistics
        all_patterns = set()
        for lang_data in results["languages"].values():
            all_patterns.update(lang_data["pattern_breakdown"].keys())
        
        for pattern in all_patterns:
            pattern_counts = []
            for lang_data in results["languages"].values():
                pattern_counts.append(lang_data["pattern_breakdown"].get(pattern, 0))
            
            results["patterns"][pattern] = {
                "total_detections": sum(pattern_counts),
                "avg_per_language": sum(pattern_counts) / len(pattern_counts),
                "detection_by_language": {
                    lang: results["languages"][lang]["pattern_breakdown"].get(pattern, 0)
                    for lang in results["languages"]
                }
            }
        
        # Overall statistics
        total_sentences = sum(lang_data["total_sentences"] for lang_data in results["languages"].values())
        total_detections = sum(lang_data["detected_patterns"] for lang_data in results["languages"].values())
        
        results["overall"] = {
            "total_sentences": total_sentences,
            "total_detections": total_detections,
            "overall_recall": total_detections / total_sentences if total_sentences > 0 else 0.0,
            "avg_recall_by_language": sum(lang_data["recall"] for lang_data in results["languages"].values()) / len(results["languages"])
        }
        
        return results
    
    def evaluate_nsm_legality(self) -> Dict[str, Any]:
        """Evaluate NSM legality across languages."""
        logger.info("Evaluating NSM legality...")
        
        if not self.test_data["parallel"]:
            return {"error": "No parallel test data available"}
        
        results = {
            "languages": {},
            "overall": {}
        }
        
        # Test each language
        for lang in ["en", "es", "fr"]:
            lang_results = {
                "total_sentences": 0,
                "legal_sentences": 0,
                "legality_rate": 0.0,
                "prime_detection": {
                    "total_primes": 0,
                    "unique_primes": set(),
                    "prime_frequency": {}
                }
            }
            
            for item in self.test_data["parallel"].get("data", []):
                text = item.get(lang, "")
                if not text:
                    continue
                
                lang_results["total_sentences"] += 1
                
                try:
                    # Check NSM legality
                    is_legal = self.nsm_explicator.validate_legality(text, lang)
                    if is_legal:
                        lang_results["legal_sentences"] += 1
                    
                    # Detect NSM primes
                    primes = self.nsm_explicator.detect_primes(text, lang)
                    lang_results["prime_detection"]["total_primes"] += len(primes)
                    lang_results["prime_detection"]["unique_primes"].update(primes)
                    
                    # Count prime frequency
                    for prime in primes:
                        if prime not in lang_results["prime_detection"]["prime_frequency"]:
                            lang_results["prime_detection"]["prime_frequency"][prime] = 0
                        lang_results["prime_detection"]["prime_frequency"][prime] += 1
                        
                except Exception as e:
                    logger.warning(f"Error evaluating NSM in {lang}: {e}")
                    continue
            
            # Calculate legality rate
            if lang_results["total_sentences"] > 0:
                lang_results["legality_rate"] = lang_results["legal_sentences"] / lang_results["total_sentences"]
            
            # Convert set to list for JSON serialization
            lang_results["prime_detection"]["unique_primes"] = list(lang_results["prime_detection"]["unique_primes"])
            
            results["languages"][lang] = lang_results
        
        # Overall statistics
        total_sentences = sum(lang_data["total_sentences"] for lang_data in results["languages"].values())
        total_legal = sum(lang_data["legal_sentences"] for lang_data in results["languages"].values())
        
        results["overall"] = {
            "total_sentences": total_sentences,
            "total_legal_sentences": total_legal,
            "overall_legality_rate": total_legal / total_sentences if total_sentences > 0 else 0.0,
            "avg_legality_by_language": sum(lang_data["legality_rate"] for lang_data in results["languages"].values()) / len(results["languages"])
        }
        
        return results
    
    def evaluate_umr_metrics(self) -> Dict[str, Any]:
        """Evaluate UMR parsing and generation metrics."""
        logger.info("Evaluating UMR metrics...")
        
        if not self.test_data["parallel"]:
            return {"error": "No parallel test data available"}
        
        results = {
            "languages": {},
            "overall": {}
        }
        
        # Test each language
        for lang in ["en", "es", "fr"]:
            lang_results = {
                "total_sentences": 0,
                "successful_parses": 0,
                "parse_success_rate": 0.0,
                "avg_graph_size": 0.0,
                "avg_round_trip_similarity": 0.0,
                "graph_metrics": []
            }
            
            for item in self.test_data["parallel"].get("data", []):
                text = item.get(lang, "")
                if not text:
                    continue
                
                lang_results["total_sentences"] += 1
                
                try:
                    # Parse text to UMR graph
                    graph = self.umr_parser.parse_text(text)
                    
                    if graph and len(graph.nodes) > 0:
                        lang_results["successful_parses"] += 1
                        
                        # Extract graph metrics
                        metrics = self.umr_evaluator.extract_primitive_metrics(graph)
                        lang_results["graph_metrics"].append(metrics)
                        
                        # Generate text from graph
                        generated_text = self.umr_generator.generate_text(graph)
                        
                        # Evaluate round-trip
                        round_trip_metrics = self.umr_evaluator.evaluate_round_trip(text, generated_text)
                        lang_results["avg_round_trip_similarity"] += round_trip_metrics["text_similarity"]
                        
                except Exception as e:
                    logger.warning(f"Error evaluating UMR in {lang}: {e}")
                    continue
            
            # Calculate averages
            if lang_results["successful_parses"] > 0:
                lang_results["parse_success_rate"] = lang_results["successful_parses"] / lang_results["total_sentences"]
                lang_results["avg_round_trip_similarity"] /= lang_results["successful_parses"]
                
                # Calculate average graph size
                total_nodes = sum(m["total_nodes"] for m in lang_results["graph_metrics"])
                total_edges = sum(m["total_edges"] for m in lang_results["graph_metrics"])
                lang_results["avg_graph_size"] = {
                    "nodes": total_nodes / lang_results["successful_parses"],
                    "edges": total_edges / lang_results["successful_parses"]
                }
            
            results["languages"][lang] = lang_results
        
        # Overall statistics
        total_sentences = sum(lang_data["total_sentences"] for lang_data in results["languages"].values())
        total_successful = sum(lang_data["successful_parses"] for lang_data in results["languages"].values())
        
        results["overall"] = {
            "total_sentences": total_sentences,
            "total_successful_parses": total_successful,
            "overall_parse_success_rate": total_successful / total_sentences if total_sentences > 0 else 0.0,
            "avg_parse_success_by_language": sum(lang_data["parse_success_rate"] for lang_data in results["languages"].values()) / len(results["languages"])
        }
        
        return results
    
    def evaluate_cross_language_consistency(self) -> Dict[str, Any]:
        """Evaluate cross-language consistency of primitive detection."""
        logger.info("Evaluating cross-language consistency...")
        
        if not self.test_data["parallel"]:
            return {"error": "No parallel test data available"}
        
        results = {
            "consistency_scores": [],
            "pattern_agreement": {},
            "overall_consistency": 0.0
        }
        
        # Test each parallel sentence group
        for item in self.test_data["parallel"].get("data", []):
            sentence_group = {
                "en": item.get("en", ""),
                "es": item.get("es", ""),
                "fr": item.get("fr", "")
            }
            
            # Skip if any language is missing
            if not all(sentence_group.values()):
                continue
            
            # Detect patterns in each language
            patterns_by_lang = {}
            for lang, text in sentence_group.items():
                try:
                    patterns = detect_primitives_multilingual(text)
                    patterns_by_lang[lang] = set(patterns)
                except Exception as e:
                    logger.warning(f"Error detecting patterns in {lang}: {e}")
                    patterns_by_lang[lang] = set()
            
            # Calculate consistency score
            all_patterns = set().union(*patterns_by_lang.values())
            if all_patterns:
                # Count how many languages detected each pattern
                pattern_agreement = {}
                for pattern in all_patterns:
                    agreement_count = sum(1 for patterns in patterns_by_lang.values() if pattern in patterns)
                    pattern_agreement[pattern] = agreement_count / len(patterns_by_lang)
                
                # Overall consistency is average agreement
                consistency_score = sum(pattern_agreement.values()) / len(pattern_agreement)
                results["consistency_scores"].append(consistency_score)
                
                # Aggregate pattern agreement
                for pattern, agreement in pattern_agreement.items():
                    if pattern not in results["pattern_agreement"]:
                        results["pattern_agreement"][pattern] = []
                    results["pattern_agreement"][pattern].append(agreement)
        
        # Calculate overall consistency
        if results["consistency_scores"]:
            results["overall_consistency"] = sum(results["consistency_scores"]) / len(results["consistency_scores"])
        
        # Calculate average agreement per pattern
        for pattern in results["pattern_agreement"]:
            results["pattern_agreement"][pattern] = sum(results["pattern_agreement"][pattern]) / len(results["pattern_agreement"][pattern])
        
        return results
    
    def generate_combined_report(self) -> Dict[str, Any]:
        """Generate comprehensive combined report."""
        logger.info("Generating combined report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "thresholds": {
                    "PERIODIC_SIM_THRESHOLD": os.getenv("PERIODIC_SIM_THRESHOLD", "0.3"),
                    "PERIODIC_TFIDF_THRESHOLD": os.getenv("PERIODIC_TFIDF_THRESHOLD", "0.1"),
                    "PERIODIC_DS_THRESHOLD": os.getenv("PERIODIC_DS_THRESHOLD", "0.6"),
                    "PERIODIC_INTERNAL_THRESHOLD": os.getenv("PERIODIC_INTERNAL_THRESHOLD", "0.20")
                }
            },
            "evaluations": {
                "ud_recalls": self.evaluate_ud_recalls(),
                "nsm_legality": self.evaluate_nsm_legality(),
                "umr_metrics": self.evaluate_umr_metrics(),
                "cross_language_consistency": self.evaluate_cross_language_consistency()
            },
            "calibration_data": self.test_data["calibration"],
            "summary": {}
        }
        
        # Generate summary statistics
        summary = {}
        
        # UD recall summary
        ud_results = report["evaluations"]["ud_recalls"]
        if "overall" in ud_results:
            summary["ud_overall_recall"] = ud_results["overall"]["overall_recall"]
            summary["ud_avg_recall_by_language"] = ud_results["overall"]["avg_recall_by_language"]
        
        # NSM legality summary
        nsm_results = report["evaluations"]["nsm_legality"]
        if "overall" in nsm_results:
            summary["nsm_overall_legality_rate"] = nsm_results["overall"]["overall_legality_rate"]
            summary["nsm_avg_legality_by_language"] = nsm_results["overall"]["avg_legality_by_language"]
        
        # UMR metrics summary
        umr_results = report["evaluations"]["umr_metrics"]
        if "overall" in umr_results:
            summary["umr_overall_parse_success_rate"] = umr_results["overall"]["overall_parse_success_rate"]
            summary["umr_avg_parse_success_by_language"] = umr_results["overall"]["avg_parse_success_by_language"]
        
        # Cross-language consistency summary
        consistency_results = report["evaluations"]["cross_language_consistency"]
        summary["cross_language_consistency"] = consistency_results["overall_consistency"]
        
        # Overall system score (weighted average)
        scores = []
        weights = []
        
        if "ud_overall_recall" in summary:
            scores.append(summary["ud_overall_recall"])
            weights.append(0.3)  # 30% weight for UD detection
        
        if "nsm_overall_legality_rate" in summary:
            scores.append(summary["nsm_overall_legality_rate"])
            weights.append(0.3)  # 30% weight for NSM legality
        
        if "umr_overall_parse_success_rate" in summary:
            scores.append(summary["umr_overall_parse_success_rate"])
            weights.append(0.2)  # 20% weight for UMR parsing
        
        if "cross_language_consistency" in summary:
            scores.append(summary["cross_language_consistency"])
            weights.append(0.2)  # 20% weight for cross-language consistency
        
        if scores and weights:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Calculate weighted average
            overall_score = sum(s * w for s, w in zip(scores, normalized_weights))
            summary["overall_system_score"] = overall_score
        
        report["summary"] = summary
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Save the combined report to file."""
        if output_path is None:
            output_path = f"data/combined_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Combined report saved to {output_file}")
        return str(output_file)
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print a summary of the combined report."""
        print("\n" + "="*80)
        print("COMBINED PRIMITIVE DETECTION SYSTEM REPORT")
        print("="*80)
        
        summary = report["summary"]
        
        print(f"\n📊 OVERALL SYSTEM SCORE: {summary.get('overall_system_score', 'N/A'):.3f}")
        
        print(f"\n🔍 UD PATTERN DETECTION:")
        print(f"   Overall Recall: {summary.get('ud_overall_recall', 'N/A'):.3f}")
        print(f"   Avg by Language: {summary.get('ud_avg_recall_by_language', 'N/A'):.3f}")
        
        print(f"\n🧠 NSM LEGALITY:")
        print(f"   Overall Legality Rate: {summary.get('nsm_overall_legality_rate', 'N/A'):.3f}")
        print(f"   Avg by Language: {summary.get('nsm_avg_legality_by_language', 'N/A'):.3f}")
        
        print(f"\n🔄 UMR METRICS:")
        print(f"   Parse Success Rate: {summary.get('umr_overall_parse_success_rate', 'N/A'):.3f}")
        print(f"   Avg by Language: {summary.get('umr_avg_parse_success_by_language', 'N/A'):.3f}")
        
        print(f"\n🌍 CROSS-LANGUAGE CONSISTENCY:")
        print(f"   Overall Consistency: {summary.get('cross_language_consistency', 'N/A'):.3f}")
        
        print(f"\n⚙️  SYSTEM THRESHOLDS:")
        thresholds = report["system_info"]["thresholds"]
        for key, value in thresholds.items():
            print(f"   {key}: {value}")
        
        print("\n" + "="*80)


def main():
    """Main function to generate combined report."""
    try:
        generator = CombinedReportGenerator()
        report = generator.generate_combined_report()
        
        # Save report
        output_file = generator.save_report(report)
        
        # Print summary
        generator.print_summary(report)
        
        print(f"\n📄 Full report saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



