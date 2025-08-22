#!/usr/bin/env python3
"""
Enhanced UDS Ingest Mining System.

This script implements a comprehensive UDS (Universal Decompositional Semantics)
ingest mining system to ingest UDS dataset and mine candidate idea-primes via
attributes for improved primitive detection and semantic analysis.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import time
from collections import defaultdict, Counter
import re
import csv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.nsm.translate import NSMTranslator
    from src.nsm.explicator import NSMExplicator
    from src.nsm.enhanced_explicator import EnhancedNSMExplicator
    from src.table.schema import PeriodicTable
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    exit(1)


def convert_numpy_types(obj):
    """Convert numpy types and other non-serializable types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        # Convert tuple keys to strings
        converted_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                # Convert tuple key to string representation
                str_key = f"{key[0]}_{key[1]}" if len(key) == 2 else str(key)
                converted_dict[str_key] = convert_numpy_types(value)
            else:
                converted_dict[str(key)] = convert_numpy_types(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(obj.__dict__)
    else:
        return obj


class UDSDatasetIngester:
    """Ingest and process UDS dataset for semantic analysis."""
    
    def __init__(self, data_dir: str = "data/uds"):
        """Initialize the UDS dataset ingester."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # UDS data structures
        self.uds_graphs = []
        self.uds_attributes = defaultdict(list)
        self.uds_events = []
        self.uds_participants = []
        
        # Attribute categories
        self.attribute_categories = {
            'semantic': ['agent', 'patient', 'theme', 'goal', 'source', 'path', 'location', 'time'],
            'temporal': ['duration', 'frequency', 'aspect', 'tense', 'mood'],
            'modality': ['necessity', 'possibility', 'ability', 'permission', 'obligation'],
            'polarity': ['positive', 'negative', 'neutral'],
            'intensity': ['high', 'medium', 'low', 'extreme'],
            'causality': ['cause', 'effect', 'condition', 'purpose', 'result']
        }
    
    def load_uds_data(self, file_path: str) -> bool:
        """Load UDS data from file."""
        try:
            if not Path(file_path).exists():
                logger.warning(f"UDS data file not found: {file_path}")
                return False
            
            # Try to load as JSON first
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._process_uds_json(data)
            except json.JSONDecodeError:
                # Try as CSV
                return self._process_uds_csv(file_path)
        
        except Exception as e:
            logger.error(f"Failed to load UDS data: {e}")
            return False
    
    def _process_uds_json(self, data: Dict[str, Any]) -> bool:
        """Process UDS data in JSON format."""
        try:
            if 'graphs' in data:
                self.uds_graphs = data['graphs']
            elif 'events' in data:
                self.uds_events = data['events']
            elif 'attributes' in data:
                self.uds_attributes = data['attributes']
            else:
                # Assume it's a list of graphs
                self.uds_graphs = data if isinstance(data, list) else [data]
            
            logger.info(f"Loaded {len(self.uds_graphs)} UDS graphs")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process UDS JSON: {e}")
            return False
    
    def _process_uds_csv(self, file_path: str) -> bool:
        """Process UDS data in CSV format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Extract graph information
                    graph = {
                        'id': row.get('id', ''),
                        'sentence': row.get('sentence', ''),
                        'attributes': {},
                        'events': [],
                        'participants': []
                    }
                    
                    # Extract attributes
                    for key, value in row.items():
                        if key.startswith('attr_'):
                            attr_name = key[5:]  # Remove 'attr_' prefix
                            graph['attributes'][attr_name] = value
                            self.uds_attributes[attr_name].append(value)
                    
                    self.uds_graphs.append(graph)
            
            logger.info(f"Loaded {len(self.uds_graphs)} UDS graphs from CSV")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process UDS CSV: {e}")
            return False
    
    def generate_sample_uds_data(self) -> bool:
        """Generate sample UDS data for testing."""
        sample_data = [
            {
                'id': 'uds_001',
                'sentence': 'The cat chased the mouse quickly through the garden.',
                'attributes': {
                    'agent': 'cat',
                    'patient': 'mouse',
                    'manner': 'quickly',
                    'location': 'garden',
                    'tense': 'past',
                    'aspect': 'perfective'
                },
                'events': ['chase'],
                'participants': ['cat', 'mouse']
            },
            {
                'id': 'uds_002',
                'sentence': 'She carefully placed the book on the shelf.',
                'attributes': {
                    'agent': 'she',
                    'patient': 'book',
                    'goal': 'shelf',
                    'manner': 'carefully',
                    'tense': 'past',
                    'aspect': 'perfective'
                },
                'events': ['place'],
                'participants': ['she', 'book', 'shelf']
            },
            {
                'id': 'uds_003',
                'sentence': 'The weather will be cold tomorrow.',
                'attributes': {
                    'theme': 'weather',
                    'property': 'cold',
                    'time': 'tomorrow',
                    'tense': 'future',
                    'polarity': 'positive'
                },
                'events': ['be'],
                'participants': ['weather']
            },
            {
                'id': 'uds_004',
                'sentence': 'He must finish the project by Friday.',
                'attributes': {
                    'agent': 'he',
                    'patient': 'project',
                    'time': 'Friday',
                    'modality': 'necessity',
                    'tense': 'present',
                    'aspect': 'imperfective'
                },
                'events': ['finish'],
                'participants': ['he', 'project']
            },
            {
                'id': 'uds_005',
                'sentence': 'The children played happily in the park.',
                'attributes': {
                    'agent': 'children',
                    'location': 'park',
                    'manner': 'happily',
                    'tense': 'past',
                    'aspect': 'imperfective',
                    'polarity': 'positive'
                },
                'events': ['play'],
                'participants': ['children', 'park']
            }
        ]
        
        self.uds_graphs = sample_data
        
        # Extract attributes
        for graph in sample_data:
            for attr_name, attr_value in graph['attributes'].items():
                self.uds_attributes[attr_name].append(attr_value)
        
        logger.info(f"Generated {len(sample_data)} sample UDS graphs")
        return True
    
    def get_attribute_statistics(self) -> Dict[str, Any]:
        """Get statistics about UDS attributes."""
        stats = {
            'total_graphs': len(self.uds_graphs),
            'attribute_counts': {},
            'attribute_values': {},
            'category_distribution': defaultdict(int)
        }
        
        for attr_name, attr_values in self.uds_attributes.items():
            stats['attribute_counts'][attr_name] = len(attr_values)
            stats['attribute_values'][attr_name] = list(set(attr_values))
            
            # Categorize attributes
            for category, attrs in self.attribute_categories.items():
                if attr_name in attrs:
                    stats['category_distribution'][category] += 1
                    break
        
        return stats


class UDSIdeaPrimeMiner:
    """Mine candidate idea-primes from UDS attributes."""
    
    def __init__(self):
        """Initialize the UDS idea-prime miner."""
        self.nsm_translator = NSMTranslator()
        self.sbert_model = None
        self.idea_primes = []
        self.prime_candidates = []
        
        # Mining parameters
        self.mining_params = {
            'min_frequency': 2,
            'min_similarity': 0.7,
            'max_primes_per_category': 10,
            'semantic_threshold': 0.6
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for UDS idea-prime mining...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def mine_idea_primes(self, uds_attributes: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Mine candidate idea-primes from UDS attributes."""
        logger.info("Mining idea-primes from UDS attributes...")
        
        idea_primes = []
        
        for attr_name, attr_values in uds_attributes.items():
            if len(attr_values) < self.mining_params['min_frequency']:
                continue
            
            # Count value frequencies
            value_counts = Counter(attr_values)
            
            # Find frequent values
            frequent_values = [
                value for value, count in value_counts.items() 
                if count >= self.mining_params['min_frequency']
            ]
            
            for value in frequent_values:
                # Check if value could be a primitive
                if self._is_potential_primitive(value, attr_name):
                    idea_prime = {
                        'value': value,
                        'attribute': attr_name,
                        'frequency': value_counts[value],
                        'category': self._categorize_attribute(attr_name),
                        'semantic_score': 0.0,
                        'nsm_compatibility': 0.0,
                        'confidence': 0.0
                    }
                    
                    # Calculate semantic score
                    idea_prime['semantic_score'] = self._calculate_semantic_score(value, attr_name)
                    
                    # Check NSM compatibility
                    idea_prime['nsm_compatibility'] = self._check_nsm_compatibility(value)
                    
                    # Calculate overall confidence
                    idea_prime['confidence'] = (
                        idea_prime['semantic_score'] * 0.4 +
                        idea_prime['nsm_compatibility'] * 0.4 +
                        min(value_counts[value] / 10, 1.0) * 0.2
                    )
                    
                    idea_primes.append(idea_prime)
        
        # Sort by confidence
        idea_primes.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit per category
        limited_primes = []
        category_counts = defaultdict(int)
        
        for prime in idea_primes:
            category = prime['category']
            if category_counts[category] < self.mining_params['max_primes_per_category']:
                limited_primes.append(prime)
                category_counts[category] += 1
        
        self.idea_primes = limited_primes
        return limited_primes
    
    def _is_potential_primitive(self, value: str, attribute: str) -> bool:
        """Check if a value could be a semantic primitive."""
        # Basic checks
        if len(value) < 2 or len(value) > 20:
            return False
        
        # Check for common primitive patterns
        primitive_patterns = [
            r'^[a-z]+$',  # All lowercase
            r'^[A-Z][a-z]+$',  # Title case
            r'^[a-z]+_[a-z]+$',  # Snake case
            r'^[a-z]+[A-Z][a-z]+$'  # Camel case
        ]
        
        for pattern in primitive_patterns:
            if re.match(pattern, value):
                return True
        
        # Check if it's a common semantic concept
        common_concepts = [
            'agent', 'patient', 'theme', 'goal', 'source', 'path', 'location',
            'time', 'manner', 'cause', 'effect', 'condition', 'purpose',
            'positive', 'negative', 'neutral', 'high', 'medium', 'low',
            'past', 'present', 'future', 'perfective', 'imperfective'
        ]
        
        if value.lower() in common_concepts:
            return True
        
        return False
    
    def _categorize_attribute(self, attribute: str) -> str:
        """Categorize an attribute."""
        attribute_categories = {
            'semantic': ['agent', 'patient', 'theme', 'goal', 'source', 'path', 'location', 'time'],
            'temporal': ['duration', 'frequency', 'aspect', 'tense', 'mood'],
            'modality': ['necessity', 'possibility', 'ability', 'permission', 'obligation'],
            'polarity': ['positive', 'negative', 'neutral'],
            'intensity': ['high', 'medium', 'low', 'extreme'],
            'causality': ['cause', 'effect', 'condition', 'purpose', 'result']
        }
        
        for category, attrs in attribute_categories.items():
            if attribute in attrs:
                return category
        
        return 'general'
    
    def _calculate_semantic_score(self, value: str, attribute: str) -> float:
        """Calculate semantic score for a value."""
        if not self.sbert_model:
            return 0.5
        
        try:
            # Create reference embeddings for semantic concepts
            reference_texts = [
                f"The {attribute} is {value}",
                f"This has {attribute} {value}",
                f"The {value} {attribute}"
            ]
            
            # Calculate semantic similarity
            embeddings = self.sbert_model.encode(reference_texts)
            
            # Calculate average similarity
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
        
        except Exception as e:
            logger.warning(f"Semantic score calculation failed: {e}")
            return 0.5
    
    def _check_nsm_compatibility(self, value: str) -> float:
        """Check compatibility with NSM primitives."""
        try:
            # Check if value matches any NSM primitive patterns
            nsm_patterns = [
                'HasProperty', 'AtLocation', 'SimilarTo', 'UsedFor', 'Contains',
                'Causes', 'PartOf', 'MadeOf', 'Desires', 'CapableOf'
            ]
            
            # Simple pattern matching
            value_lower = value.lower()
            for pattern in nsm_patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in value_lower or value_lower in pattern_lower:
                    return 0.8
            
            # Check if it's a common semantic concept
            common_semantic = [
                'agent', 'patient', 'location', 'time', 'manner', 'cause',
                'effect', 'property', 'goal', 'source', 'path'
            ]
            
            if value_lower in common_semantic:
                return 0.6
            
            return 0.3
        
        except Exception as e:
            logger.warning(f"NSM compatibility check failed: {e}")
            return 0.3
    
    def generate_prime_candidates(self, idea_primes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate prime candidates from idea-primes."""
        candidates = []
        
        for prime in idea_primes:
            if prime['confidence'] >= self.mining_params['semantic_threshold']:
                candidate = {
                    'prime_id': f"prime_{len(candidates):03d}",
                    'value': prime['value'],
                    'attribute': prime['attribute'],
                    'category': prime['category'],
                    'confidence': prime['confidence'],
                    'semantic_score': prime['semantic_score'],
                    'nsm_compatibility': prime['nsm_compatibility'],
                    'frequency': prime['frequency'],
                    'metadata': {
                        'source': 'uds_mining',
                        'timestamp': time.time(),
                        'mining_params': self.mining_params
                    }
                }
                candidates.append(candidate)
        
        self.prime_candidates = candidates
        return candidates


class EnhancedUDSMiningSystem:
    """Enhanced UDS mining system with comprehensive analysis."""
    
    def __init__(self):
        """Initialize the enhanced UDS mining system."""
        self.ingester = UDSDatasetIngester()
        self.miner = UDSIdeaPrimeMiner()
        self.nsm_translator = NSMTranslator()
        
        # Analysis parameters
        self.analysis_params = {
            'min_confidence': 0.5,
            'max_candidates': 50,
            'semantic_threshold': 0.6,
            'category_weights': {
                'semantic': 1.0,
                'temporal': 0.8,
                'modality': 0.9,
                'polarity': 0.7,
                'intensity': 0.6,
                'causality': 0.9,
                'general': 0.5
            }
        }
    
    def run_uds_mining(self, use_sample_data: bool = True) -> Dict[str, Any]:
        """Run comprehensive UDS mining analysis."""
        logger.info("Starting enhanced UDS mining analysis...")
        
        mining_results = {
            'test_configuration': {
                'use_sample_data': use_sample_data,
                'timestamp': time.time()
            },
            'uds_statistics': {},
            'idea_primes': [],
            'prime_candidates': [],
            'mining_analysis': {},
            'recommendations': []
        }
        
        # Load UDS data
        if use_sample_data:
            success = self.ingester.generate_sample_uds_data()
        else:
            # Try to load from file
            uds_file = self.ingester.data_dir / "uds_data.json"
            success = self.ingester.load_uds_data(str(uds_file))
        
        if not success:
            logger.error("Failed to load UDS data")
            return mining_results
        
        # Get UDS statistics
        mining_results['uds_statistics'] = self.ingester.get_attribute_statistics()
        
        # Mine idea-primes
        idea_primes = self.miner.mine_idea_primes(self.ingester.uds_attributes)
        mining_results['idea_primes'] = idea_primes
        
        # Generate prime candidates
        prime_candidates = self.miner.generate_prime_candidates(idea_primes)
        mining_results['prime_candidates'] = prime_candidates
        
        # Analyze mining results
        mining_results['mining_analysis'] = self._analyze_mining_results(
            idea_primes, prime_candidates
        )
        
        # Generate recommendations
        mining_results['recommendations'] = self._generate_mining_recommendations(
            mining_results['mining_analysis']
        )
        
        return mining_results
    
    def _analyze_mining_results(self, idea_primes: List[Dict[str, Any]], 
                              prime_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze mining results."""
        analysis = {
            'total_idea_primes': len(idea_primes),
            'total_candidates': len(prime_candidates),
            'category_distribution': defaultdict(int),
            'confidence_distribution': defaultdict(int),
            'semantic_score_distribution': defaultdict(int),
            'nsm_compatibility_distribution': defaultdict(int),
            'top_candidates': [],
            'category_analysis': {}
        }
        
        # Analyze idea-primes
        for prime in idea_primes:
            category = prime['category']
            analysis['category_distribution'][category] += 1
            
            confidence_bucket = int(prime['confidence'] * 10) / 10
            analysis['confidence_distribution'][confidence_bucket] += 1
            
            semantic_bucket = int(prime['semantic_score'] * 10) / 10
            analysis['semantic_score_distribution'][semantic_bucket] += 1
            
            nsm_bucket = int(prime['nsm_compatibility'] * 10) / 10
            analysis['nsm_compatibility_distribution'][nsm_bucket] += 1
        
        # Get top candidates
        analysis['top_candidates'] = prime_candidates[:10]
        
        # Analyze by category
        for category in set(prime['category'] for prime in idea_primes):
            category_primes = [p for p in idea_primes if p['category'] == category]
            category_candidates = [c for c in prime_candidates if c['category'] == category]
            
            analysis['category_analysis'][category] = {
                'total_primes': len(category_primes),
                'total_candidates': len(category_candidates),
                'avg_confidence': np.mean([p['confidence'] for p in category_primes]) if category_primes else 0.0,
                'avg_semantic_score': np.mean([p['semantic_score'] for p in category_primes]) if category_primes else 0.0,
                'avg_nsm_compatibility': np.mean([p['nsm_compatibility'] for p in category_primes]) if category_primes else 0.0
            }
        
        return analysis
    
    def _generate_mining_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on mining analysis."""
        recommendations = []
        
        # Overall recommendations
        if analysis['total_candidates'] < 10:
            recommendations.append("Low number of prime candidates - consider lowering confidence threshold")
        
        if analysis['total_idea_primes'] < 20:
            recommendations.append("Low number of idea-primes - consider expanding UDS dataset")
        
        # Category recommendations
        for category, cat_analysis in analysis['category_analysis'].items():
            if cat_analysis['avg_confidence'] < 0.6:
                recommendations.append(f"Low confidence for {category} category ({cat_analysis['avg_confidence']:.1%}) - improve semantic analysis")
            
            if cat_analysis['avg_nsm_compatibility'] < 0.5:
                recommendations.append(f"Low NSM compatibility for {category} category ({cat_analysis['avg_nsm_compatibility']:.1%}) - refine primitive patterns")
        
        # Distribution recommendations
        confidence_dist = analysis['confidence_distribution']
        if max(confidence_dist.values()) < 5:
            recommendations.append("Sparse confidence distribution - consider adjusting mining parameters")
        
        return recommendations


def main():
    """Main function to run enhanced UDS mining."""
    logger.info("Starting enhanced UDS mining...")
    
    # Initialize mining system
    mining_system = EnhancedUDSMiningSystem()
    
    # Run mining analysis
    mining_results = mining_system.run_uds_mining(use_sample_data=True)
    
    # Print results
    print("\n" + "="*80)
    print("ENHANCED UDS MINING RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Use Sample Data: {mining_results['test_configuration']['use_sample_data']}")
    
    print(f"\nUDS Statistics:")
    stats = mining_results['uds_statistics']
    print(f"  Total Graphs: {stats['total_graphs']}")
    print(f"  Attribute Categories: {len(stats['category_distribution'])}")
    print(f"  Total Attributes: {len(stats['attribute_counts'])}")
    
    print(f"\nMining Results:")
    print(f"  Total Idea-Primes: {mining_results['mining_analysis']['total_idea_primes']}")
    print(f"  Total Candidates: {mining_results['mining_analysis']['total_candidates']}")
    
    print(f"\nCategory Distribution:")
    for category, count in mining_results['mining_analysis']['category_distribution'].items():
        print(f"  {category}: {count}")
    
    print(f"\nTop 5 Prime Candidates:")
    for i, candidate in enumerate(mining_results['mining_analysis']['top_candidates'][:5], 1):
        print(f"  {i}. {candidate['value']} ({candidate['attribute']}) - confidence: {candidate['confidence']:.3f}")
    
    print(f"\nCategory Analysis:")
    for category, analysis in mining_results['mining_analysis']['category_analysis'].items():
        print(f"  {category}:")
        print(f"    Primes: {analysis['total_primes']}, Candidates: {analysis['total_candidates']}")
        print(f"    Avg Confidence: {analysis['avg_confidence']:.3f}")
        print(f"    Avg Semantic Score: {analysis['avg_semantic_score']:.3f}")
        print(f"    Avg NSM Compatibility: {analysis['avg_nsm_compatibility']:.3f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(mining_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/uds_ingest_mining_enhanced_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(mining_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced UDS mining report saved to: {output_path}")
    
    print("="*80)
    print("Enhanced UDS mining completed!")
    print("="*80)


if __name__ == "__main__":
    main()
