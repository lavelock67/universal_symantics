#!/usr/bin/env python3
"""
Scaled Parallel Dataset Generator.

This script generates a scaled parallel dataset with ≥100 items per phenomenon
and implements the tiny test pack specified by ChatGPT5 for validation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
import re
from dataclasses import dataclass, asdict
from enum import Enum
import time
from collections import defaultdict

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
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class PhenomenonType(Enum):
    """Types of linguistic phenomena."""
    NEGATION = "negation"
    MODALITY = "modality"
    ASPECT = "aspect"
    EXPERIENCER = "experiencer"
    QUANTIFIERS = "quantifiers"
    LOCATIVES = "locatives"
    COUNTERFACTUALS = "counterfactuals"
    IDIOMS = "idioms"
    POLITENESS = "politeness"


@dataclass
class ParallelItem:
    """A parallel text item across multiple languages."""
    item_id: str
    phenomenon: PhenomenonType
    texts: Dict[str, str]  # language -> text
    explications: Dict[str, str]  # language -> NSM explication
    detected_primes: Dict[str, List[str]]  # language -> list of primes
    synsets: Dict[str, List[str]]  # language -> list of BabelNet synsets
    expected_output: Dict[str, Any]  # expected analysis results
    test_type: str  # 'validation', 'adversarial', 'standard'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'item_id': self.item_id,
            'phenomenon': self.phenomenon.value,
            'texts': self.texts,
            'explications': self.explications,
            'detected_primes': self.detected_primes,
            'synsets': self.synsets,
            'expected_output': self.expected_output,
            'test_type': self.test_type
        }


class TinyTestPackGenerator:
    """Generates the tiny test pack specified by ChatGPT5."""
    
    def __init__(self):
        """Initialize the tiny test pack generator."""
        self.test_cases = self._create_tiny_test_pack()
    
    def _create_tiny_test_pack(self) -> List[ParallelItem]:
        """Create the tiny test pack with specific validation cases."""
        test_cases = []
        
        # 1. Neg+locative: "The cat is not on the mat." → NOT AT_LOC(cat,on(mat)); HERE must NOT fire
        test_cases.append(ParallelItem(
            item_id="tiny_test_001_neg_locative",
            phenomenon=PhenomenonType.NEGATION,
            texts={
                'en': "The cat is not on the mat",
                'es': "El gato no está en la alfombra",
                'fr': "Le chat n'est pas sur le tapis"
            },
            explications={
                'en': "NOT AT_LOC(cat, on(mat))",
                'es': "NOT AT_LOC(cat, on(mat))",
                'fr': "NOT AT_LOC(cat, on(mat))"
            },
            detected_primes={
                'en': ['NOT', 'AT_LOC'],
                'es': ['NOT', 'AT_LOC'],
                'fr': ['NOT', 'AT_LOC']
            },
            synsets={
                'en': ['not.r.01', 'at_location.v.01'],
                'es': ['no.r.01', 'estar_en.v.01'],
                'fr': ['ne.r.01', 'être_sur.v.01']
            },
            expected_output={
                'primes_expected': ['NOT', 'AT_LOC'],
                'primes_forbidden': ['HERE'],
                'synset_stability': True,
                'jaccard_threshold': 0.85
            },
            test_type='validation'
        ))
        
        # 2. Aux DO: "I do not like this weather." → NOT LIKE(i, weather(this)); no eventive DO
        test_cases.append(ParallelItem(
            item_id="tiny_test_002_aux_do",
            phenomenon=PhenomenonType.NEGATION,
            texts={
                'en': "I do not like this weather",
                'es': "No me gusta este tiempo",
                'fr': "Je n'aime pas ce temps"
            },
            explications={
                'en': "NOT LIKE(i, weather(this))",
                'es': "NOT LIKE(i, weather(this))",
                'fr': "NOT LIKE(i, weather(this))"
            },
            detected_primes={
                'en': ['NOT', 'LIKE'],
                'es': ['NOT', 'LIKE'],
                'fr': ['NOT', 'LIKE']
            },
            synsets={
                'en': ['not.r.01', 'like.v.01'],
                'es': ['no.r.01', 'gustar.v.01'],
                'fr': ['ne.r.01', 'aimer.v.01']
            },
            expected_output={
                'primes_expected': ['NOT', 'LIKE'],
                'primes_forbidden': ['DO'],
                'synset_stability': True,
                'jaccard_threshold': 0.85
            },
            test_type='validation'
        ))
        
        # 3. Quant scope (ambiguous): "All children aren't playing." → two parses flagged; no silent ∀¬ collapse
        test_cases.append(ParallelItem(
            item_id="tiny_test_003_quant_scope_ambiguous",
            phenomenon=PhenomenonType.QUANTIFIERS,
            texts={
                'en': "All children aren't playing",
                'es': "Todos los niños no están jugando",
                'fr': "Tous les enfants ne jouent pas"
            },
            explications={
                'en': "AMBIGUOUS: ALL(child, NOT(play(child))) OR NOT(ALL(child, play(child)))",
                'es': "AMBIGUOUS: ALL(child, NOT(play(child))) OR NOT(ALL(child, play(child)))",
                'fr': "AMBIGUOUS: ALL(child, NOT(play(child))) OR NOT(ALL(child, play(child)))"
            },
            detected_primes={
                'en': ['ALL', 'NOT', 'AMBIGUOUS'],
                'es': ['ALL', 'NOT', 'AMBIGUOUS'],
                'fr': ['ALL', 'NOT', 'AMBIGUOUS']
            },
            synsets={
                'en': ['all.a.01', 'not.r.01', 'play.v.01'],
                'es': ['todo.a.01', 'no.r.01', 'jugar.v.01'],
                'fr': ['tout.a.01', 'ne.r.01', 'jouer.v.01']
            },
            expected_output={
                'primes_expected': ['ALL', 'NOT'],
                'ambiguity_detected': True,
                'forbidden_collapse': True,
                'jaccard_threshold': 0.85
            },
            test_type='validation'
        ))
        
        # 4. Experiencer: "A María le gusta el café." / "Marie aime ce café." → LIKE(exp,stim) roles preserved
        test_cases.append(ParallelItem(
            item_id="tiny_test_004_experiencer",
            phenomenon=PhenomenonType.EXPERIENCER,
            texts={
                'en': "Maria likes coffee",
                'es': "A María le gusta el café",
                'fr': "Marie aime ce café"
            },
            explications={
                'en': "LIKE(maria, coffee)",
                'es': "LIKE(maria, coffee)",
                'fr': "LIKE(maria, coffee)"
            },
            detected_primes={
                'en': ['LIKE'],
                'es': ['LIKE'],
                'fr': ['LIKE']
            },
            synsets={
                'en': ['like.v.01'],
                'es': ['gustar.v.01'],
                'fr': ['aimer.v.01']
            },
            expected_output={
                'primes_expected': ['LIKE'],
                'roles_preserved': True,
                'experiencer_mapping': True,
                'synset_stability': True,
                'jaccard_threshold': 0.85
            },
            test_type='validation'
        ))
        
        # 5. Almost: "Il a failli tomber." / "Por poco se cae." → ALMOST_DO(fall); not fall
        test_cases.append(ParallelItem(
            item_id="tiny_test_005_almost",
            phenomenon=PhenomenonType.ASPECT,
            texts={
                'en': "He almost fell",
                'es': "Por poco se cae",
                'fr': "Il a failli tomber"
            },
            explications={
                'en': "ALMOST_DO(he, fall)",
                'es': "ALMOST_DO(he, fall)",
                'fr': "ALMOST_DO(he, fall)"
            },
            detected_primes={
                'en': ['ALMOST_DO'],
                'es': ['ALMOST_DO'],
                'fr': ['ALMOST_DO']
            },
            synsets={
                'en': ['almost.r.01', 'fall.v.01'],
                'es': ['casi.r.01', 'caer.v.01'],
                'fr': ['presque.r.01', 'tomber.v.01']
            },
            expected_output={
                'primes_expected': ['ALMOST_DO'],
                'primes_forbidden': ['fall'],  # Should not detect actual fall
                'aspect_molecule': True,
                'synset_stability': True,
                'jaccard_threshold': 0.85
            },
            test_type='validation'
        ))
        
        # 6. Recent past: "Je viens d'arriver." / "Acabo de llegar." → RECENT_PAST(arrive) with stable senses
        test_cases.append(ParallelItem(
            item_id="tiny_test_006_recent_past",
            phenomenon=PhenomenonType.ASPECT,
            texts={
                'en': "I just arrived",
                'es': "Acabo de llegar",
                'fr': "Je viens d'arriver"
            },
            explications={
                'en': "RECENT_PAST(arrive(i))",
                'es': "RECENT_PAST(arrive(i))",
                'fr': "RECENT_PAST(arrive(i))"
            },
            detected_primes={
                'en': ['RECENT_PAST'],
                'es': ['RECENT_PAST'],
                'fr': ['RECENT_PAST']
            },
            synsets={
                'en': ['just.r.01', 'arrive.v.01'],
                'es': ['acabar.v.01', 'llegar.v.01'],
                'fr': ['venir.v.01', 'arriver.v.01']
            },
            expected_output={
                'primes_expected': ['RECENT_PAST'],
                'aspect_molecule': True,
                'synset_stability': True,
                'jaccard_threshold': 0.85
            },
            test_type='validation'
        ))
        
        return test_cases
    
    def get_test_pack(self) -> List[ParallelItem]:
        """Get the tiny test pack."""
        return self.test_cases


class ScaledParallelDatasetGenerator:
    """Generates scaled parallel dataset with ≥100 items per phenomenon."""
    
    def __init__(self):
        """Initialize the scaled dataset generator."""
        self.phenomena_templates = self._create_phenomena_templates()
        self.tiny_test_pack = TinyTestPackGenerator()
    
    def _create_phenomena_templates(self) -> Dict[PhenomenonType, List[Dict[str, Any]]]:
        """Create templates for each phenomenon type."""
        templates = {
            PhenomenonType.NEGATION: [
                {
                    'en': "The {subject} is not {predicate}",
                    'es': "El {subject} no está {predicate}",
                    'fr': "Le {subject} n'est pas {predicate}",
                    'primes': ['NOT'],
                    'synsets': ['not.r.01']
                },
                {
                    'en': "I do not {action}",
                    'es': "No {action}",
                    'fr': "Je ne {action} pas",
                    'primes': ['NOT'],
                    'synsets': ['not.r.01']
                },
                {
                    'en': "Nobody {action}",
                    'es': "Nadie {action}",
                    'fr': "Personne ne {action}",
                    'primes': ['NOT', 'SOMEONE'],
                    'synsets': ['not.r.01', 'someone.n.01']
                }
            ],
            PhenomenonType.MODALITY: [
                {
                    'en': "I can {action}",
                    'es': "Puedo {action}",
                    'fr': "Je peux {action}",
                    'primes': ['CAN'],
                    'synsets': ['can.v.01']
                },
                {
                    'en': "I must {action}",
                    'es': "Debo {action}",
                    'fr': "Je dois {action}",
                    'primes': ['MUST'],
                    'synsets': ['must.v.01']
                },
                {
                    'en': "I want to {action}",
                    'es': "Quiero {action}",
                    'fr': "Je veux {action}",
                    'primes': ['WANT'],
                    'synsets': ['want.v.01']
                }
            ],
            PhenomenonType.ASPECT: [
                {
                    'en': "I am {action}ing",
                    'es': "Estoy {action}ando",
                    'fr': "Je suis en train de {action}",
                    'primes': ['DO', 'NOW'],
                    'synsets': ['do.v.01', 'now.r.01']
                },
                {
                    'en': "I have {action}ed",
                    'es': "He {action}ado",
                    'fr': "J'ai {action}é",
                    'primes': ['DO', 'BEFORE'],
                    'synsets': ['do.v.01', 'before.r.01']
                },
                {
                    'en': "I will {action}",
                    'es': "Voy a {action}",
                    'fr': "Je vais {action}",
                    'primes': ['DO', 'AFTER'],
                    'synsets': ['do.v.01', 'after.r.01']
                }
            ],
            PhenomenonType.EXPERIENCER: [
                {
                    'en': "I like {object}",
                    'es': "Me gusta {object}",
                    'fr': "J'aime {object}",
                    'primes': ['LIKE'],
                    'synsets': ['like.v.01']
                },
                {
                    'en': "I need {object}",
                    'es': "Necesito {object}",
                    'fr': "J'ai besoin de {object}",
                    'primes': ['NEED'],
                    'synsets': ['need.v.01']
                },
                {
                    'en': "I think {proposition}",
                    'es': "Pienso que {proposition}",
                    'fr': "Je pense que {proposition}",
                    'primes': ['THINK'],
                    'synsets': ['think.v.01']
                }
            ],
            PhenomenonType.QUANTIFIERS: [
                {
                    'en': "All {objects} are {property}",
                    'es': "Todos los {objects} son {property}",
                    'fr': "Tous les {objects} sont {property}",
                    'primes': ['ALL'],
                    'synsets': ['all.a.01']
                },
                {
                    'en': "Some {objects} are {property}",
                    'es': "Algunos {objects} son {property}",
                    'fr': "Certains {objects} sont {property}",
                    'primes': ['SOME'],
                    'synsets': ['some.a.01']
                },
                {
                    'en': "Many {objects} are {property}",
                    'es': "Muchos {objects} son {property}",
                    'fr': "Beaucoup de {objects} sont {property}",
                    'primes': ['MANY'],
                    'synsets': ['many.a.01']
                }
            ],
            PhenomenonType.LOCATIVES: [
                {
                    'en': "The {object} is on the {surface}",
                    'es': "El {object} está en la {surface}",
                    'fr': "Le {object} est sur la {surface}",
                    'primes': ['AT_LOC'],
                    'synsets': ['at_location.v.01']
                },
                {
                    'en': "The {object} is in the {container}",
                    'es': "El {object} está en el {container}",
                    'fr': "Le {object} est dans le {container}",
                    'primes': ['AT_LOC'],
                    'synsets': ['at_location.v.01']
                },
                {
                    'en': "I am here",
                    'es': "Estoy aquí",
                    'fr': "Je suis ici",
                    'primes': ['HERE'],
                    'synsets': ['here.r.01']
                }
            ]
        }
        return templates
    
    def generate_scaled_dataset(self, items_per_phenomenon: int = 100) -> Dict[str, Any]:
        """Generate scaled parallel dataset."""
        logger.info(f"Generating scaled parallel dataset with {items_per_phenomenon} items per phenomenon")
        
        dataset = {
            'metadata': {
                'total_items': 0,
                'items_per_phenomenon': items_per_phenomenon,
                'languages': ['en', 'es', 'fr'],
                'generation_timestamp': time.time()
            },
            'phenomena': {},
            'tiny_test_pack': [item.to_dict() for item in self.tiny_test_pack.get_test_pack()],
            'validation_set': [],
            'adversarial_set': []
        }
        
        # Generate items for each phenomenon
        for phenomenon in PhenomenonType:
            phenomenon_items = self._generate_phenomenon_items(phenomenon, items_per_phenomenon)
            dataset['phenomena'][phenomenon.value] = {
                'items': [item.to_dict() for item in phenomenon_items],
                'count': len(phenomenon_items)
            }
            dataset['metadata']['total_items'] += len(phenomenon_items)
        
        # Add tiny test pack to validation set
        dataset['validation_set'] = dataset['tiny_test_pack']
        
        # Generate adversarial examples
        dataset['adversarial_set'] = self._generate_adversarial_examples()
        
        return dataset
    
    def _generate_phenomenon_items(self, phenomenon: PhenomenonType, count: int) -> List[ParallelItem]:
        """Generate items for a specific phenomenon."""
        items = []
        templates = self.phenomena_templates.get(phenomenon, [])
        
        if not templates:
            logger.warning(f"No templates found for phenomenon {phenomenon.value}")
            return items
        
        # Vocabulary for slot filling
        vocabulary = {
            'subject': ['cat', 'dog', 'bird', 'fish', 'person', 'child', 'student', 'teacher'],
            'predicate': ['happy', 'sad', 'tired', 'hungry', 'thirsty', 'excited', 'worried'],
            'action': ['run', 'walk', 'jump', 'swim', 'read', 'write', 'sing', 'dance'],
            'object': ['book', 'car', 'house', 'tree', 'flower', 'food', 'water', 'music'],
            'property': ['big', 'small', 'red', 'blue', 'fast', 'slow', 'good', 'bad'],
            'objects': ['cats', 'dogs', 'birds', 'fish', 'people', 'children', 'students'],
            'surface': ['table', 'floor', 'ground', 'mat', 'bed', 'chair', 'desk'],
            'container': ['box', 'bag', 'room', 'house', 'car', 'bottle', 'cup'],
            'proposition': ['it is good', 'it is bad', 'it is true', 'it is false']
        }
        
        for i in range(count):
            # Select template
            template = templates[i % len(templates)]
            
            # Fill slots
            filled_template = self._fill_template_slots(template, vocabulary)
            
            # Create parallel item
            item = ParallelItem(
                item_id=f"{phenomenon.value}_{i+1:03d}",
                phenomenon=phenomenon,
                texts=filled_template['texts'],
                explications=filled_template['explications'],
                detected_primes=filled_template['primes'],
                synsets=filled_template['synsets'],
                expected_output={
                    'primes_expected': filled_template['primes'],
                    'synset_stability': True,
                    'jaccard_threshold': 0.85
                },
                test_type='standard'
            )
            items.append(item)
        
        return items
    
    def _fill_template_slots(self, template: Dict[str, Any], vocabulary: Dict[str, List[str]]) -> Dict[str, Any]:
        """Fill template slots with vocabulary items."""
        filled = {
            'texts': {},
            'explications': {},
            'primes': template.get('primes', []),
            'synsets': template.get('synsets', [])
        }
        
        # Fill text slots
        for lang, text_template in template.items():
            if lang in ['primes', 'synsets']:
                continue
            
            filled_text = text_template
            for slot, values in vocabulary.items():
                if f"{{{slot}}}" in filled_text:
                    value = np.random.choice(values)
                    filled_text = filled_text.replace(f"{{{slot}}}", value)
            
            filled['texts'][lang] = filled_text
        
        # Generate explications (simplified)
        for lang in filled['texts']:
            filled['explications'][lang] = self._generate_explication(filled['texts'][lang], template.get('primes', []))
        
        return filled
    
    def _generate_explication(self, text: str, primes: List[str]) -> str:
        """Generate NSM explication from text and primes."""
        # Simplified explication generation
        explication_parts = []
        for prime in primes:
            if prime == 'NOT':
                explication_parts.append('NOT')
            elif prime == 'CAN':
                explication_parts.append('CAN')
            elif prime == 'MUST':
                explication_parts.append('MUST')
            elif prime == 'WANT':
                explication_parts.append('WANT')
            elif prime == 'LIKE':
                explication_parts.append('LIKE')
            elif prime == 'ALL':
                explication_parts.append('ALL')
            elif prime == 'SOME':
                explication_parts.append('SOME')
            elif prime == 'AT_LOC':
                explication_parts.append('AT_LOC')
            elif prime == 'HERE':
                explication_parts.append('HERE')
        
        return ' '.join(explication_parts) if explication_parts else text
    
    def _generate_adversarial_examples(self) -> List[Dict[str, Any]]:
        """Generate adversarial examples for testing robustness."""
        adversarial_examples = [
            {
                'item_id': 'adv_001_negation_scope',
                'phenomenon': 'negation',
                'texts': {
                    'en': "Not all students passed the exam",
                    'es': "No todos los estudiantes aprobaron el examen",
                    'fr': "Pas tous les étudiants ont réussi l'examen"
                },
                'expected_analysis': {
                    'ambiguity_detected': True,
                    'scope_analysis': 'NOT(ALL) vs ALL(NOT)',
                    'primes_expected': ['NOT', 'ALL']
                }
            },
            {
                'item_id': 'adv_002_experiencer_drift',
                'phenomenon': 'experiencer',
                'texts': {
                    'en': "The music pleases me",
                    'es': "La música me gusta",
                    'fr': "La musique me plaît"
                },
                'expected_analysis': {
                    'experiencer_mapping': True,
                    'synset_stability': True,
                    'primes_expected': ['LIKE']
                }
            }
        ]
        return adversarial_examples


def main():
    """Main function to generate scaled parallel dataset."""
    logger.info("Starting scaled parallel dataset generation...")
    
    # Initialize generator
    generator = ScaledParallelDatasetGenerator()
    
    # Generate scaled dataset
    dataset = generator.generate_scaled_dataset(items_per_phenomenon=100)
    
    # Print results
    print("\n" + "="*80)
    print("SCALED PARALLEL DATASET GENERATION RESULTS")
    print("="*80)
    
    print(f"Dataset Metadata:")
    print(f"  Total Items: {dataset['metadata']['total_items']}")
    print(f"  Items per Phenomenon: {dataset['metadata']['items_per_phenomenon']}")
    print(f"  Languages: {dataset['metadata']['languages']}")
    
    print(f"\nPhenomena Distribution:")
    for phenomenon, data in dataset['phenomena'].items():
        print(f"  {phenomenon}: {data['count']} items")
    
    print(f"\nTiny Test Pack:")
    print(f"  Validation Items: {len(dataset['tiny_test_pack'])}")
    for item in dataset['tiny_test_pack']:
        print(f"    {item['item_id']}: {item['phenomenon']} - {item['texts']['en'][:50]}...")
    
    print(f"\nAdversarial Examples:")
    print(f"  Adversarial Items: {len(dataset['adversarial_set'])}")
    for item in dataset['adversarial_set']:
        print(f"    {item['item_id']}: {item['phenomenon']} - {item['texts']['en'][:50]}...")
    
    # Save dataset
    output_path = Path("data/scaled_parallel_dataset.json")
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        json_results = convert_numpy_types(dataset)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        # Save a simplified version
        simplified_dataset = {
            'metadata': dataset['metadata'],
            'phenomena_count': {k: v['count'] for k, v in dataset['phenomena'].items()},
            'tiny_test_pack_count': len(dataset['tiny_test_pack']),
            'adversarial_count': len(dataset['adversarial_set'])
        }
        with open(output_path, 'w') as f:
            json.dump(simplified_dataset, f, indent=2)
    
    logger.info(f"Scaled parallel dataset saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Scaled parallel dataset generation completed!")
    print("="*80)


if __name__ == "__main__":
    main()
