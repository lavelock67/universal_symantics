#!/usr/bin/env python3
"""
Evaluation Manifest Stamping System.

This script implements the evaluation manifest system to stamp all reports
with run metadata and prevent metric drift, as specified in ChatGPT5's feedback.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
from datetime import datetime
import hashlib
import time
from dataclasses import dataclass, asdict
from enum import Enum

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class MetricScale(Enum):
    """Standardized metric scales to prevent drift."""
    COSINE_0_1 = "cosine_0_1"
    PERCENTAGE_0_100 = "percentage_0_100"
    PROBABILITY_0_1 = "probability_0_1"
    RATIO_0_INF = "ratio_0_inf"
    COUNT_0_INF = "count_0_inf"


@dataclass
class EvaluationManifest:
    """Evaluation manifest with run metadata to prevent metric drift."""
    run_id: str
    commit_sha: str
    dataset_version: str
    split_name: str
    similarity_model_name: str
    metric_scales: Dict[str, str]
    timestamp: float
    evaluation_config: Dict[str, Any]
    system_components: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'run_id': self.run_id,
            'commit_sha': self.commit_sha,
            'dataset_version': self.dataset_version,
            'split_name': self.split_name,
            'similarity_model_name': self.similarity_model_name,
            'metric_scales': self.metric_scales,
            'timestamp': self.timestamp,
            'evaluation_config': self.evaluation_config,
            'system_components': self.system_components
        }


class ManifestGenerator:
    """Generates evaluation manifests with run metadata."""
    
    def __init__(self):
        """Initialize the manifest generator."""
        self.scale_registry = {
            'sbert_similarity': MetricScale.COSINE_0_1,
            'xnli_entailment': MetricScale.PROBABILITY_0_1,
            'prime_alignment': MetricScale.PERCENTAGE_0_100,
            'polarity_alignment': MetricScale.PERCENTAGE_0_100,
            'scope_alignment': MetricScale.PERCENTAGE_0_100,
            'umr_structural_consistency': MetricScale.PERCENTAGE_0_100,
            'bmr_structural_consistency': MetricScale.PERCENTAGE_0_100,
            'mps_composite': MetricScale.PERCENTAGE_0_100,
            'detection_rate': MetricScale.PERCENTAGE_0_100,
            'legality_rate': MetricScale.PERCENTAGE_0_100,
            'cross_language_consistency': MetricScale.PERCENTAGE_0_100
        }
    
    def generate_manifest(self, evaluation_config: Dict[str, Any]) -> EvaluationManifest:
        """Generate evaluation manifest with current run metadata."""
        # Generate run ID
        run_id = self._generate_run_id()
        
        # Get commit SHA
        commit_sha = self._get_commit_sha()
        
        # Get dataset version
        dataset_version = self._get_dataset_version()
        
        # Get similarity model name
        similarity_model_name = evaluation_config.get('similarity_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        
        # Get metric scales
        metric_scales = {metric: scale.value for metric, scale in self.scale_registry.items()}
        
        # Get system components
        system_components = self._get_system_components()
        
        return EvaluationManifest(
            run_id=run_id,
            commit_sha=commit_sha,
            dataset_version=dataset_version,
            split_name=evaluation_config.get('split_name', 'test'),
            similarity_model_name=similarity_model_name,
            metric_scales=metric_scales,
            timestamp=time.time(),
            evaluation_config=evaluation_config,
            system_components=system_components
        )
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"eval_{timestamp}_{random_suffix}"
    
    def _get_commit_sha(self) -> str:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not get git commit SHA")
            return "unknown"
    
    def _get_dataset_version(self) -> str:
        """Get dataset version from data files."""
        try:
            # Check for dataset version files
            data_dir = Path("data")
            if data_dir.exists():
                # Look for version files
                version_files = list(data_dir.glob("*version*.json")) + list(data_dir.glob("*version*.txt"))
                if version_files:
                    # Use most recent version file
                    latest_file = max(version_files, key=lambda f: f.stat().st_mtime)
                    return f"dataset_{latest_file.stem}_{latest_file.stat().st_mtime}"
            
            # Fallback to directory modification time
            if data_dir.exists():
                return f"dataset_{data_dir.stat().st_mtime}"
            else:
                return "unknown"
        except Exception as e:
            logger.warning(f"Could not get dataset version: {e}")
            return "unknown"
    
    def _get_system_components(self) -> List[str]:
        """Get list of system components being evaluated."""
        components = []
        
        # Check for key system files
        system_files = [
            "src/nsm/translate.py",
            "src/nsm/explicator.py", 
            "src/nsm/enhanced_explicator.py",
            "src/detect/text_detectors.py",
            "unified_primitive_detection.py",
            "executable_interlingua_spec.py"
        ]
        
        for file_path in system_files:
            if Path(file_path).exists():
                components.append(file_path)
        
        return components


class MPSCalculator:
    """Calculates MPS (Meaning Preservation Score) as the headline metric."""
    
    def __init__(self):
        """Initialize the MPS calculator."""
        self.weights = {
            'xnli_entailment': 0.4,
            'prime_alignment': 0.3,
            'polarity_alignment': 0.1,
            'scope_alignment': 0.1,
            'umr_structural_consistency': 0.05,
            'bmr_structural_consistency': 0.05
        }
    
    def calculate_mps(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate MPS as weighted composite of meaning preservation metrics."""
        mps_components = {}
        total_weight = 0
        weighted_sum = 0
        
        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                # Normalize to 0-1 scale if needed
                normalized_value = self._normalize_metric(metric_name, value)
                mps_components[metric_name] = {
                    'raw_value': value,
                    'normalized_value': normalized_value,
                    'weight': weight,
                    'contribution': normalized_value * weight
                }
                weighted_sum += normalized_value * weight
                total_weight += weight
        
        # Calculate final MPS
        mps = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return {
            'mps_score': mps,
            'components': mps_components,
            'weights': self.weights,
            'interpretation': self._interpret_mps(mps)
        }
    
    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric to 0-1 scale based on expected range."""
        if metric_name == 'xnli_entailment':
            # XNLI typically returns 0-1 probabilities
            return max(0.0, min(1.0, value))
        elif metric_name in ['prime_alignment', 'polarity_alignment', 'scope_alignment', 
                           'umr_structural_consistency', 'bmr_structural_consistency']:
            # These are typically percentages, normalize to 0-1
            return max(0.0, min(1.0, value / 100.0))
        else:
            # Default normalization
            return max(0.0, min(1.0, value))
    
    def _interpret_mps(self, mps: float) -> str:
        """Interpret MPS score."""
        if mps >= 0.9:
            return "excellent"
        elif mps >= 0.8:
            return "good"
        elif mps >= 0.7:
            return "fair"
        elif mps >= 0.6:
            return "poor"
        else:
            return "very_poor"


class EvaluationStamper:
    """Stamps evaluation reports with manifest and MPS."""
    
    def __init__(self):
        """Initialize the evaluation stamper."""
        self.manifest_generator = ManifestGenerator()
        self.mps_calculator = MPSCalculator()
    
    def stamp_evaluation_report(self, evaluation_results: Dict[str, Any], 
                              evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Stamp evaluation report with manifest and MPS."""
        # Generate manifest
        manifest = self.manifest_generator.generate_manifest(evaluation_config)
        
        # Calculate MPS
        metrics = evaluation_results.get('metrics', {})
        mps_result = self.mps_calculator.calculate_mps(metrics)
        
        # Create stamped report
        stamped_report = {
            'manifest': manifest.to_dict(),
            'mps_headline': mps_result,
            'raw_metrics': metrics,
            'evaluation_results': evaluation_results,
            'stamp_timestamp': time.time()
        }
        
        return stamped_report
    
    def validate_manifest_consistency(self, report1: Dict[str, Any], 
                                    report2: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency between two evaluation reports."""
        manifest1 = report1.get('manifest', {})
        manifest2 = report2.get('manifest', {})
        
        consistency_check = {
            'same_commit': manifest1.get('commit_sha') == manifest2.get('commit_sha'),
            'same_dataset': manifest1.get('dataset_version') == manifest2.get('dataset_version'),
            'same_split': manifest1.get('split_name') == manifest2.get('split_name'),
            'same_model': manifest1.get('similarity_model_name') == manifest2.get('similarity_model_name'),
            'same_scales': manifest1.get('metric_scales') == manifest2.get('metric_scales'),
            'mps_comparison': {
                'report1_mps': report1.get('mps_headline', {}).get('mps_score', 0),
                'report2_mps': report2.get('mps_headline', {}).get('mps_score', 0),
                'mps_difference': abs(report1.get('mps_headline', {}).get('mps_score', 0) - 
                                    report2.get('mps_headline', {}).get('mps_score', 0))
            }
        }
        
        # Determine if reports are comparable
        consistency_check['comparable'] = (
            consistency_check['same_commit'] and 
            consistency_check['same_dataset'] and 
            consistency_check['same_split'] and 
            consistency_check['same_model'] and 
            consistency_check['same_scales']
        )
        
        return consistency_check


def main():
    """Main function to demonstrate evaluation manifest stamping."""
    logger.info("Starting evaluation manifest stamping demonstration...")
    
    # Initialize stamper
    stamper = EvaluationStamper()
    
    # Sample evaluation results
    sample_results = {
        'metrics': {
            'sbert_similarity': 0.85,
            'xnli_entailment': 0.78,
            'prime_alignment': 85.0,
            'polarity_alignment': 92.0,
            'scope_alignment': 88.0,
            'umr_structural_consistency': 76.0,
            'bmr_structural_consistency': 82.0,
            'detection_rate': 34.7,
            'legality_rate': 100.0
        },
        'test_configuration': {
            'num_test_texts': 40,
            'languages': ['en']
        }
    }
    
    # Sample evaluation config
    sample_config = {
        'split_name': 'test',
        'similarity_model': 'paraphrase-multilingual-MiniLM-L12-v2',
        'evaluation_type': 'comprehensive',
        'include_mps': True
    }
    
    # Stamp evaluation report
    stamped_report = stamper.stamp_evaluation_report(sample_results, sample_config)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION MANIFEST STAMPING RESULTS")
    print("="*80)
    
    print(f"Manifest:")
    manifest = stamped_report['manifest']
    print(f"  Run ID: {manifest['run_id']}")
    print(f"  Commit SHA: {manifest['commit_sha'][:8]}...")
    print(f"  Dataset Version: {manifest['dataset_version']}")
    print(f"  Split Name: {manifest['split_name']}")
    print(f"  Similarity Model: {manifest['similarity_model_name']}")
    print(f"  Timestamp: {datetime.fromtimestamp(manifest['timestamp'])}")
    
    print(f"\nMPS Headline:")
    mps = stamped_report['mps_headline']
    print(f"  MPS Score: {mps['mps_score']:.3f} ({mps['interpretation']})")
    print(f"  Components:")
    for component, details in mps['components'].items():
        print(f"    {component}: {details['raw_value']:.3f} → {details['normalized_value']:.3f} (weight: {details['weight']})")
    
    print(f"\nMetric Scales:")
    for metric, scale in manifest['metric_scales'].items():
        print(f"  {metric}: {scale}")
    
    print(f"\nSystem Components:")
    for component in manifest['system_components']:
        print(f"  {component}")
    
    # Save stamped report
    output_path = Path("data/evaluation_manifest_stamp_demo.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(convert_numpy_types(stamped_report), f, indent=2)
    
    logger.info(f"Stamped evaluation report saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Evaluation manifest stamping demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()
