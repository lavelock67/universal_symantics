#!/usr/bin/env python3
"""
Locative/Deixis Split System.

This script implements the locative/deixis split to properly distinguish between
HERE (deictic) and WHERE (locative) based on ChatGPT5's guidance.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import time
from collections import defaultdict, Counter
import re
from dataclasses import dataclass, asdict
from enum import Enum

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocativeType(Enum):
    """Types of locative expressions."""
    DEICTIC = "deictic"      # HERE - speaker-anchored
    LOCATIVE = "locative"    # WHERE - general location
    SPATIAL = "spatial"      # AtLocation - spatial relation


@dataclass
class LocativeAnalysis:
    """Analysis of locative expressions."""
    text: str
    detected_locatives: List[Dict[str, Any]]
    deictic_indicators: List[str]
    spatial_indicators: List[str]
    analysis_summary: Dict[str, Any]


class LocativeDeixisSplitter:
    """System to properly distinguish between HERE (deictic) and WHERE (locative)."""
    
    def __init__(self):
        """Initialize the locative/deixis splitter."""
        # Deictic indicators (explicit speaker-anchored references)
        self.deictic_indicators = {
            'en': [
                r'\bhere\b',
                r'\bthis place\b',
                r'\bthis location\b',
                r'\bwhere i am\b',
                r'\bwhere we are\b',
                r'\bthis spot\b',
                r'\bthis area\b',
                r'\bthis site\b'
            ],
            'es': [
                r'\baquí\b',
                r'\beste lugar\b',
                r'\besta ubicación\b',
                r'\bdonde estoy\b',
                r'\bdonde estamos\b',
                r'\beste sitio\b',
                r'\besta zona\b'
            ],
            'fr': [
                r'\bici\b',
                r'\bce lieu\b',
                r'\bcet endroit\b',
                r'\boù je suis\b',
                r'\boù nous sommes\b',
                r'\bcet emplacement\b',
                r'\bcette zone\b'
            ]
        }
        
        # Spatial indicators (general location references)
        self.spatial_indicators = {
            'en': [
                r'\bon\b',
                r'\bin\b',
                r'\bat\b',
                r'\bunder\b',
                r'\bover\b',
                r'\bbeside\b',
                r'\bnear\b',
                r'\bfar\b',
                r'\binside\b',
                r'\boutside\b',
                r'\babove\b',
                r'\bbelow\b',
                r'\bwhere\b',
                r'\blocation\b',
                r'\bplace\b',
                r'\bposition\b'
            ],
            'es': [
                r'\ben\b',
                r'\bsobre\b',
                r'\bdebajo\b',
                r'\bcerca\b',
                r'\blejos\b',
                r'\bdentro\b',
                r'\bfuera\b',
                r'\bdonde\b',
                r'\bubicación\b',
                r'\blugar\b',
                r'\bposición\b'
            ],
            'fr': [
                r'\bsur\b',
                r'\bdans\b',
                r'\bà\b',
                r'\bdessous\b',
                r'\bdessus\b',
                r'\bprès\b',
                r'\bloin\b',
                r'\bdedans\b',
                r'\bdehors\b',
                r'\boù\b',
                r'\bemplacement\b',
                r'\blieu\b',
                r'\bposition\b'
            ]
        }
        
        # Context patterns that indicate deixis vs locative
        self.deictic_context_patterns = [
            r'\bcome here\b',
            r'\bgo here\b',
            r'\bstay here\b',
            r'\bleave here\b',
            r'\barrive here\b',
            r'\bmeet here\b',
            r'\bwait here\b',
            r'\bstand here\b',
            r'\bsit here\b',
            r'\blook here\b',
            r'\bpoint here\b',
            r'\bput.*here\b',
            r'\bplace.*here\b',
            r'\bleave.*here\b'
        ]
        
        self.locative_context_patterns = [
            r'\bis on\b',
            r'\bis in\b',
            r'\bis at\b',
            r'\bis under\b',
            r'\bis over\b',
            r'\bis near\b',
            r'\bis far\b',
            r'\bis inside\b',
            r'\bis outside\b',
            r'\bis above\b',
            r'\bis below\b',
            r'\blocated\b',
            r'\bsituated\b',
            r'\bfound\b',
            r'\bplaced\b',
            r'\bpositioned\b'
        ]
    
    def analyze_locatives(self, text: str, language: str = "en") -> LocativeAnalysis:
        """Analyze locative expressions in text."""
        text_lower = text.lower()
        
        # Detect deictic indicators
        deictic_indicators = []
        for pattern in self.deictic_indicators.get(language, self.deictic_indicators['en']):
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                deictic_indicators.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'deictic'
                })
        
        # Detect spatial indicators
        spatial_indicators = []
        for pattern in self.spatial_indicators.get(language, self.spatial_indicators['en']):
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                spatial_indicators.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'spatial'
                })
        
        # Analyze context patterns
        deictic_context = []
        for pattern in self.deictic_context_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                deictic_context.append({
                    'pattern': pattern,
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        locative_context = []
        for pattern in self.locative_context_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                locative_context.append({
                    'pattern': pattern,
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Determine locative types
        detected_locatives = []
        
        # Check for explicit deictic indicators
        for indicator in deictic_indicators:
            detected_locatives.append({
                'primitive': 'HERE',
                'type': LocativeType.DEICTIC,
                'confidence': 0.9,
                'evidence': indicator['text'],
                'evidence_type': 'explicit_deictic',
                'start': indicator['start'],
                'end': indicator['end']
            })
        
        # Check for spatial indicators (but not if already marked as deictic)
        for indicator in spatial_indicators:
            # Skip if this is already covered by a deictic indicator
            is_covered_by_deictic = any(
                d['start'] <= indicator['start'] <= d['end'] or 
                d['start'] <= indicator['end'] <= d['end']
                for d in deictic_indicators
            )
            
            if not is_covered_by_deictic:
                # Determine if it's WHERE or AtLocation based on context
                if indicator['text'] in ['where', 'donde', 'où']:
                    primitive = 'WHERE'
                    loc_type = LocativeType.LOCATIVE
                elif indicator['text'] in ['on', 'in', 'at', 'under', 'over', 'beside', 'near', 'far', 'inside', 'outside', 'above', 'below', 'sur', 'dans', 'à', 'dessous', 'dessus', 'près', 'loin', 'dedans', 'dehors', 'en', 'sobre', 'debajo', 'cerca', 'lejos', 'dentro', 'fuera']:
                    # These are spatial prepositions that should trigger WHERE detection
                    primitive = 'WHERE'
                    loc_type = LocativeType.LOCATIVE
                else:
                    primitive = 'AtLocation'
                    loc_type = LocativeType.SPATIAL
                
                detected_locatives.append({
                    'primitive': primitive,
                    'type': loc_type,
                    'confidence': 0.7,
                    'evidence': indicator['text'],
                    'evidence_type': 'spatial_indicator',
                    'start': indicator['start'],
                    'end': indicator['end']
                })
        
        # Apply context-based corrections
        detected_locatives = self._apply_context_corrections(
            detected_locatives, deictic_context, locative_context, text_lower
        )
        
        # Generate analysis summary
        analysis_summary = {
            'total_locatives': len(detected_locatives),
            'deictic_count': sum(1 for l in detected_locatives if l['type'] == LocativeType.DEICTIC),
            'locative_count': sum(1 for l in detected_locatives if l['type'] == LocativeType.LOCATIVE),
            'spatial_count': sum(1 for l in detected_locatives if l['type'] == LocativeType.SPATIAL),
            'deictic_context_patterns': len(deictic_context),
            'locative_context_patterns': len(locative_context),
            'confidence_distribution': {
                'high': sum(1 for l in detected_locatives if l['confidence'] >= 0.8),
                'medium': sum(1 for l in detected_locatives if 0.6 <= l['confidence'] < 0.8),
                'low': sum(1 for l in detected_locatives if l['confidence'] < 0.6)
            }
        }
        
        return LocativeAnalysis(
            text=text,
            detected_locatives=detected_locatives,
            deictic_indicators=[d['text'] for d in deictic_indicators],
            spatial_indicators=[s['text'] for s in spatial_indicators],
            analysis_summary=analysis_summary
        )
    
    def _apply_context_corrections(
        self, 
        detected_locatives: List[Dict[str, Any]], 
        deictic_context: List[Dict[str, Any]], 
        locative_context: List[Dict[str, Any]], 
        text_lower: str
    ) -> List[Dict[str, Any]]:
        """Apply context-based corrections to locative detections."""
        corrected_locatives = []
        
        for locative in detected_locatives:
            # Check if there's deictic context that should override
            has_deictic_context = any(
                dc['start'] <= locative['start'] <= dc['end'] or 
                dc['start'] <= locative['end'] <= dc['end'] or
                locative['start'] <= dc['start'] <= locative['end'] or
                locative['start'] <= dc['end'] <= locative['end']
                for dc in deictic_context
            )
            
            # Check if there's locative context that confirms the type
            has_locative_context = any(
                lc['start'] <= locative['start'] <= lc['end'] or 
                lc['start'] <= locative['end'] <= lc['end'] or
                locative['start'] <= lc['start'] <= locative['end'] or
                locative['start'] <= lc['end'] <= locative['end']
                for lc in locative_context
            )
            
            # Apply corrections
            if has_deictic_context and locative['type'] != LocativeType.DEICTIC:
                # Override with deictic
                corrected_locative = locative.copy()
                corrected_locative['primitive'] = 'HERE'
                corrected_locative['type'] = LocativeType.DEICTIC
                corrected_locative['confidence'] = min(0.95, locative['confidence'] + 0.2)
                corrected_locative['evidence_type'] = 'context_corrected_deictic'
                corrected_locatives.append(corrected_locative)
            elif has_locative_context and locative['type'] == LocativeType.DEICTIC:
                # Override with locative (rare case)
                corrected_locative = locative.copy()
                corrected_locative['primitive'] = 'WHERE'
                corrected_locative['type'] = LocativeType.LOCATIVE
                corrected_locative['confidence'] = min(0.95, locative['confidence'] + 0.1)
                corrected_locative['evidence_type'] = 'context_corrected_locative'
                corrected_locatives.append(corrected_locative)
            else:
                # Keep as is
                corrected_locatives.append(locative)
        
        return corrected_locatives
    
    def validate_detection(self, text: str, expected_here: bool, expected_where: bool) -> Dict[str, Any]:
        """Validate detection against expected results."""
        analysis = self.analyze_locatives(text)
        
        detected_here = any(l['primitive'] == 'HERE' for l in analysis.detected_locatives)
        detected_where = any(l['primitive'] == 'WHERE' for l in analysis.detected_locatives)
        
        return {
            'text': text,
            'expected_here': expected_here,
            'expected_where': expected_where,
            'detected_here': detected_here,
            'detected_where': detected_where,
            'here_correct': detected_here == expected_here,
            'where_correct': detected_where == expected_where,
            'all_correct': (detected_here == expected_here) and (detected_where == expected_where),
            'analysis': analysis
        }


class LocativeDeixisTestSuite:
    """Test suite for locative/deixis split validation."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.splitter = LocativeDeixisSplitter()
        
        # Test cases with expected results
        self.test_cases = [
            # Explicit deictic cases (should detect HERE)
            ("Come here", True, False),
            ("The cat is here", True, False),
            ("I am here now", True, False),
            ("Meet me here", True, False),
            ("Put it here", True, False),
            ("Stay here", True, False),
            ("Look here", True, False),
            
            # General locative cases (should detect WHERE/AtLocation, not HERE)
            ("The cat is on the mat", False, True),
            ("The book is in the box", False, True),
            ("The car is at the station", False, True),
            ("The bird is under the tree", False, True),
            ("The plane is above the clouds", False, True),
            ("The fish is in the water", False, True),
            ("The key is on the table", False, True),
            
            # Ambiguous cases that need context analysis
            ("The cat is here on the mat", True, True),  # Both deictic and locative
            ("I am here in the room", True, True),       # Both deictic and locative
            ("The meeting is here at the office", True, True),  # Both deictic and locative
            
            # Edge cases
            ("Where is the cat?", False, True),          # Question form
            ("Here is the answer", True, False),         # Deictic presentation
            ("The location is here", True, False),       # Deictic with location word
            ("The place is here", True, False),          # Deictic with place word
            
            # Spanish examples
            ("Ven aquí", True, False),                   # Come here
            ("El gato está aquí", True, False),          # The cat is here
            ("El libro está en la mesa", False, True),   # The book is on the table
            ("¿Dónde está el gato?", False, True),       # Where is the cat?
            
            # French examples
            ("Viens ici", True, False),                  # Come here
            ("Le chat est ici", True, False),            # The cat is here
            ("Le livre est sur la table", False, True),  # The book is on the table
            ("Où est le chat?", False, True),            # Where is the cat?
        ]
    
    def run_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        logger.info("Running locative/deixis split test suite...")
        
        results = []
        total_tests = len(self.test_cases)
        correct_here = 0
        correct_where = 0
        all_correct = 0
        
        for i, (text, expected_here, expected_where) in enumerate(self.test_cases):
            result = self.splitter.validate_detection(text, expected_here, expected_where)
            results.append(result)
            
            if result['here_correct']:
                correct_here += 1
            if result['where_correct']:
                correct_where += 1
            if result['all_correct']:
                all_correct += 1
            
            print(f"Test {i+1}: {text}")
            print(f"  Expected: HERE={expected_here}, WHERE={expected_where}")
            print(f"  Detected: HERE={result['detected_here']}, WHERE={result['detected_where']}")
            print(f"  Correct: {result['all_correct']}")
            print()
        
        # Calculate metrics
        here_accuracy = correct_here / total_tests if total_tests > 0 else 0
        where_accuracy = correct_where / total_tests if total_tests > 0 else 0
        overall_accuracy = all_correct / total_tests if total_tests > 0 else 0
        
        summary = {
            'total_tests': total_tests,
            'correct_here': correct_here,
            'correct_where': correct_where,
            'all_correct': all_correct,
            'here_accuracy': here_accuracy,
            'where_accuracy': where_accuracy,
            'overall_accuracy': overall_accuracy,
            'results': results
        }
        
        return summary
    
    def generate_detailed_report(self, summary: Dict[str, Any]) -> None:
        """Generate detailed test report."""
        print("="*80)
        print("LOCATIVE/DEIXIS SPLIT TEST SUITE RESULTS")
        print("="*80)
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Correct HERE Detections: {summary['correct_here']}/{summary['total_tests']} ({summary['here_accuracy']:.1%})")
        print(f"Correct WHERE Detections: {summary['correct_where']}/{summary['total_tests']} ({summary['where_accuracy']:.1%})")
        print(f"All Correct: {summary['all_correct']}/{summary['total_tests']} ({summary['overall_accuracy']:.1%})")
        
        print(f"\nDetailed Results:")
        for i, result in enumerate(summary['results']):
            status = "✅" if result['all_correct'] else "❌"
            print(f"{status} Test {i+1}: {result['text']}")
            print(f"    Expected: HERE={result['expected_here']}, WHERE={result['expected_where']}")
            print(f"    Detected: HERE={result['detected_here']}, WHERE={result['detected_where']}")
            
            if not result['all_correct']:
                print(f"    Issues:")
                if not result['here_correct']:
                    print(f"      - HERE detection incorrect")
                if not result['where_correct']:
                    print(f"      - WHERE detection incorrect")
        
        # Save detailed report
        report_data = {
            'summary': summary,
            'timestamp': time.time(),
            'test_cases': self.test_cases
        }
        
        # Convert LocativeAnalysis objects to dicts for JSON serialization
        serializable_results = []
        for result in summary['results']:
            # Convert LocativeType enums to strings
            detected_locatives = []
            for loc in result['analysis'].detected_locatives:
                loc_copy = loc.copy()
                loc_copy['type'] = loc_copy['type'].value if hasattr(loc_copy['type'], 'value') else str(loc_copy['type'])
                detected_locatives.append(loc_copy)
            
            serializable_result = {
                'text': result['text'],
                'expected_here': result['expected_here'],
                'expected_where': result['expected_where'],
                'detected_here': result['detected_here'],
                'detected_where': result['detected_where'],
                'here_correct': result['here_correct'],
                'where_correct': result['where_correct'],
                'all_correct': result['all_correct'],
                'analysis': {
                    'text': result['analysis'].text,
                    'detected_locatives': detected_locatives,
                    'deictic_indicators': result['analysis'].deictic_indicators,
                    'spatial_indicators': result['analysis'].spatial_indicators,
                    'analysis_summary': result['analysis'].analysis_summary
                }
            }
            serializable_results.append(serializable_result)
        
        report_data['results'] = serializable_results
        
        with open('data/locative_deixis_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info("Detailed test report saved to data/locative_deixis_test_report.json")


def main():
    """Main function to run locative/deixis split testing."""
    logger.info("Starting locative/deixis split testing...")
    
    # Initialize test suite
    test_suite = LocativeDeixisTestSuite()
    
    # Run tests
    summary = test_suite.run_test_suite()
    
    # Generate report
    test_suite.generate_detailed_report(summary)
    
    print("="*80)
    print("Locative/deixis split testing completed!")
    print("="*80)


if __name__ == "__main__":
    main()
