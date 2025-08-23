#!/usr/bin/env python3
"""
NSM Pivot Cleaner System.

This script implements the critical NSM pivot cleaning fixes identified by ChatGPT5:
- No meta "SAY" wrapper unless truly reporting speech
- Aux-DO gate: only emit NSM DO when UD root is action predicate
- Deixis vs location: HERE only on explicit deictics, else WHERE/AT-LOC
- ConceptNet quarantine: keep CN relations separate from NSM
- Dedup & canonicalization: collapse duplicates, normalize order
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
import re
from dataclasses import dataclass, asdict
from enum import Enum
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
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class PivotIssueType(Enum):
    """Types of NSM pivot issues to fix."""
    META_SAY_WRAPPER = "meta_say_wrapper"
    AUX_DO_MISUSE = "aux_do_misuse"
    DEICTIC_FALSE_POSITIVE = "deictic_false_positive"
    CONCEPTNET_IN_NSM = "conceptnet_in_nsm"
    DUPLICATE_PRIMES = "duplicate_primes"
    NON_CANONICAL_ORDER = "non_canonical_order"


@dataclass
class PivotIssue:
    """A specific NSM pivot issue to be fixed."""
    issue_type: PivotIssueType
    description: str
    original_text: str
    problematic_explication: str
    suggested_fix: str
    confidence: float
    line_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'issue_type': self.issue_type.value,
            'description': self.description,
            'original_text': self.original_text,
            'problematic_explication': self.problematic_explication,
            'suggested_fix': self.suggested_fix,
            'confidence': self.confidence,
            'line_number': self.line_number
        }


@dataclass
class CleanedExplication:
    """A cleaned NSM explication."""
    original_text: str
    original_explication: str
    cleaned_explication: str
    fixes_applied: List[str]
    confidence: float
    canonical_order: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_text': self.original_text,
            'original_explication': self.original_explication,
            'cleaned_explication': self.cleaned_explication,
            'fixes_applied': self.fixes_applied,
            'confidence': self.confidence,
            'canonical_order': self.canonical_order
        }


class NSMPivotCleaner:
    """Cleans NSM pivot explications according to ChatGPT5's specifications."""
    
    def __init__(self):
        """Initialize the NSM pivot cleaner."""
        # ConceptNet relations to quarantine
        self.conceptnet_relations = {
            'AtLocation', 'SimilarTo', 'UsedFor', 'HasProperty', 'Causes',
            'PartOf', 'CapableOf', 'Desires', 'Antonym', 'Synonym',
            'RelatedTo', 'IsA', 'InstanceOf', 'MadeOf', 'LocatedNear'
        }
        
        # Deictic indicators
        self.deictic_indicators = {
            'en': ['here', 'this', 'that', 'these', 'those'],
            'es': ['aquí', 'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas'],
            'fr': ['ici', 'ce', 'cette', 'ces', 'celui', 'celle', 'ceux', 'celles']
        }
        
        # Canonical order for NSM explications
        self.canonical_order = [
            'scope', 'modality', 'tense_aspect', 'predicate_roles', 'adjuncts'
        ]
        
        # Auxiliary DO patterns
        self.aux_do_patterns = [
            r'\bdo\s+not\b', r'\bdoes\s+not\b', r'\bdid\s+not\b',
            r'\bdo\s+you\b', r'\bdoes\s+he\b', r'\bdid\s+they\b',
            r'\bdo\s+we\b', r'\bdoes\s+she\b', r'\bdid\s+it\b'
        ]
    
    def clean_explication(self, original_text: str, explication: str, 
                         language: str = "en") -> CleanedExplication:
        """Clean an NSM explication according to all rules."""
        logger.info(f"Cleaning explication for: {original_text[:50]}...")
        
        original_explication = explication
        cleaned_explication = explication
        fixes_applied = []
        
        # Apply all cleaning rules
        cleaned_explication, fixes = self._remove_meta_say_wrapper(cleaned_explication)
        fixes_applied.extend(fixes)
        
        cleaned_explication, fixes = self._fix_aux_do_misuse(cleaned_explication, original_text)
        fixes_applied.extend(fixes)
        
        cleaned_explication, fixes = self._fix_deictic_false_positives(cleaned_explication, original_text, language)
        fixes_applied.extend(fixes)
        
        cleaned_explication, fixes = self._quarantine_conceptnet_relations(cleaned_explication)
        fixes_applied.extend(fixes)
        
        cleaned_explication, fixes = self._deduplicate_primes(cleaned_explication)
        fixes_applied.extend(fixes)
        
        cleaned_explication, fixes = self._canonicalize_order(cleaned_explication)
        fixes_applied.extend(fixes)
        
        # Calculate confidence based on number of fixes applied
        confidence = max(0.5, 1.0 - len(fixes_applied) * 0.1)
        
        # Get canonical order
        canonical_order = self._extract_canonical_order(cleaned_explication)
        
        return CleanedExplication(
            original_text=original_text,
            original_explication=original_explication,
            cleaned_explication=cleaned_explication,
            fixes_applied=fixes_applied,
            confidence=confidence,
            canonical_order=canonical_order
        )
    
    def _remove_meta_say_wrapper(self, explication: str) -> Tuple[str, List[str]]:
        """Remove meta 'SAY' wrapper unless truly reporting speech."""
        fixes = []
        
        # Pattern to match "X SAY: ..." wrapper
        say_pattern = r'^([A-Z]+\s+)?SAY:\s*(.+)$'
        match = re.match(say_pattern, explication.strip())
        
        if match:
            # Check if this is truly reported speech
            if not self._is_reported_speech(match.group(2)):
                # Remove the SAY wrapper
                cleaned = match.group(2).strip()
                fixes.append(f"Removed meta SAY wrapper: '{explication}' → '{cleaned}'")
                return cleaned, fixes
        
        return explication, fixes
    
    def _is_reported_speech(self, content: str) -> bool:
        """Check if content represents reported speech."""
        # Simple heuristics for reported speech
        reported_indicators = [
            'said', 'told', 'asked', 'replied', 'answered', 'explained',
            'dijo', 'contó', 'preguntó', 'respondió', 'explicó',
            'dit', 'raconté', 'demanda', 'répondit', 'expliqua'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in reported_indicators)
    
    def _fix_aux_do_misuse(self, explication: str, original_text: str) -> Tuple[str, List[str]]:
        """Fix auxiliary DO misuse - only emit NSM DO when UD root is action predicate."""
        fixes = []
        
        # Check if original text contains auxiliary DO patterns
        original_lower = original_text.lower()
        has_aux_do = any(re.search(pattern, original_lower) for pattern in self.aux_do_patterns)
        
        if has_aux_do:
            # Look for DO in explication that might be auxiliary
            do_pattern = r'\bDO\b'
            if re.search(do_pattern, explication):
                # Check if this DO is actually auxiliary (not eventive)
                if self._is_auxiliary_do(original_text):
                    # Remove auxiliary DO
                    cleaned = re.sub(do_pattern, '', explication).strip()
                    cleaned = re.sub(r'\s+', ' ', cleaned)  # Clean up extra spaces
                    fixes.append(f"Removed auxiliary DO: '{explication}' → '{cleaned}'")
                    return cleaned, fixes
        
        return explication, fixes
    
    def _is_auxiliary_do(self, text: str) -> bool:
        """Check if DO in text is auxiliary (not eventive)."""
        text_lower = text.lower()
        
        # Check for auxiliary DO patterns
        aux_patterns = [
            r'\bdo\s+not\b', r'\bdoes\s+not\b', r'\bdid\s+not\b',
            r'\bdo\s+you\b', r'\bdoes\s+he\b', r'\bdid\s+they\b'
        ]
        
        if any(re.search(pattern, text_lower) for pattern in aux_patterns):
            return True
        
        # Check if DO is followed by another verb (auxiliary pattern)
        do_verb_pattern = r'\bdo(es)?\s+[a-z]+(ing|ed)?\b'
        if re.search(do_verb_pattern, text_lower):
            return True
        
        return False
    
    def _fix_deictic_false_positives(self, explication: str, original_text: str, 
                                   language: str) -> Tuple[str, List[str]]:
        """Fix deictic false positives - HERE only on explicit deictics."""
        fixes = []
        
        # Check if original text has explicit deictic indicators
        original_lower = original_text.lower()
        deictic_indicators = self.deictic_indicators.get(language, self.deictic_indicators['en'])
        has_explicit_deictic = any(indicator in original_lower for indicator in deictic_indicators)
        
        # Look for HERE in explication
        here_pattern = r'\bHERE\b'
        if re.search(here_pattern, explication):
            if not has_explicit_deictic:
                # Replace HERE with WHERE or AT-LOC
                cleaned = re.sub(here_pattern, 'WHERE', explication)
                fixes.append(f"Replaced HERE with WHERE (no explicit deictic): '{explication}' → '{cleaned}'")
                return cleaned, fixes
        
        return explication, fixes
    
    def _quarantine_conceptnet_relations(self, explication: str) -> Tuple[str, List[str]]:
        """Quarantine ConceptNet relations - keep them separate from NSM."""
        fixes = []
        
        # Find ConceptNet relations in explication
        found_cn_relations = []
        for relation in self.conceptnet_relations:
            if relation in explication:
                found_cn_relations.append(relation)
        
        if found_cn_relations:
            # Replace ConceptNet relations with NSM equivalents or remove them
            cleaned = explication
            for relation in found_cn_relations:
                nsm_equivalent = self._get_nsm_equivalent(relation)
                if nsm_equivalent:
                    cleaned = cleaned.replace(relation, nsm_equivalent)
                    fixes.append(f"Replaced ConceptNet '{relation}' with NSM '{nsm_equivalent}'")
                else:
                    # Remove if no NSM equivalent
                    cleaned = cleaned.replace(relation, '')
                    fixes.append(f"Removed ConceptNet relation '{relation}' (no NSM equivalent)")
            
            # Clean up extra spaces
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return cleaned, fixes
        
        return explication, fixes
    
    def _get_nsm_equivalent(self, conceptnet_relation: str) -> Optional[str]:
        """Get NSM equivalent for ConceptNet relation."""
        cn_to_nsm = {
            'AtLocation': 'WHERE',
            'SimilarTo': 'LIKE',
            'UsedFor': 'FOR',
            'HasProperty': 'HAVE',
            'Causes': 'BECAUSE',
            'PartOf': 'PART',
            'CapableOf': 'CAN',
            'Desires': 'WANT',
            'IsA': 'BE',
            'InstanceOf': 'BE'
        }
        return cn_to_nsm.get(conceptnet_relation)
    
    def _deduplicate_primes(self, explication: str) -> Tuple[str, List[str]]:
        """Deduplicate primes in explication."""
        fixes = []
        
        # Split into words and find duplicates
        words = explication.split()
        seen = set()
        deduplicated = []
        
        for word in words:
            if word not in seen:
                deduplicated.append(word)
                seen.add(word)
            else:
                fixes.append(f"Removed duplicate prime '{word}'")
        
        cleaned = ' '.join(deduplicated)
        
        if fixes:
            return cleaned, fixes
        
        return explication, fixes
    
    def _canonicalize_order(self, explication: str) -> Tuple[str, List[str]]:
        """Canonicalize order of NSM elements."""
        fixes = []
        
        # Parse explication into components
        components = self._parse_explication_components(explication)
        
        # Reorder according to canonical order
        canonical_components = self._reorder_components(components)
        
        # Reconstruct explication
        cleaned = self._reconstruct_explication(canonical_components)
        
        if cleaned != explication:
            fixes.append(f"Canonicalized order: '{explication}' → '{cleaned}'")
        
        return cleaned, fixes
    
    def _parse_explication_components(self, explication: str) -> Dict[str, List[str]]:
        """Parse explication into component categories."""
        components = {
            'scope': [],
            'modality': [],
            'tense_aspect': [],
            'predicate_roles': [],
            'adjuncts': []
        }
        
        words = explication.split()
        
        for word in words:
            if word in ['NOT', 'NEG']:
                components['scope'].append(word)
            elif word in ['CAN', 'MUST', 'SHOULD', 'MIGHT', 'WILL']:
                components['modality'].append(word)
            elif word in ['PAST', 'FUTURE', 'PRESENT', 'BEFORE', 'AFTER']:
                components['tense_aspect'].append(word)
            elif word in ['DO', 'HAPPEN', 'CAUSE', 'LIKE', 'WANT', 'THINK', 'KNOW']:
                components['predicate_roles'].append(word)
            else:
                components['adjuncts'].append(word)
        
        return components
    
    def _reorder_components(self, components: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Reorder components according to canonical order."""
        return {category: components[category] for category in self.canonical_order}
    
    def _reconstruct_explication(self, components: Dict[str, List[str]]) -> str:
        """Reconstruct explication from ordered components."""
        parts = []
        for category, words in components.items():
            if words:
                parts.extend(words)
        
        return ' '.join(parts)
    
    def _extract_canonical_order(self, explication: str) -> List[str]:
        """Extract canonical order from cleaned explication."""
        components = self._parse_explication_components(explication)
        order = []
        
        for category in self.canonical_order:
            if components[category]:
                order.append(f"{category}: {', '.join(components[category])}")
        
        return order
    
    def detect_pivot_issues(self, explications: List[Tuple[str, str]], 
                          language: str = "en") -> List[PivotIssue]:
        """Detect pivot issues in a list of explications."""
        issues = []
        
        for i, (original_text, explication) in enumerate(explications):
            # Check for meta SAY wrapper
            if re.match(r'^([A-Z]+\s+)?SAY:\s*(.+)$', explication.strip()):
                if not self._is_reported_speech(re.match(r'^([A-Z]+\s+)?SAY:\s*(.+)$', explication.strip()).group(2)):
                    issues.append(PivotIssue(
                        issue_type=PivotIssueType.META_SAY_WRAPPER,
                        description="Meta SAY wrapper detected without reported speech",
                        original_text=original_text,
                        problematic_explication=explication,
                        suggested_fix="Remove SAY wrapper and keep propositional content",
                        confidence=0.9,
                        line_number=i + 1
                    ))
            
            # Check for auxiliary DO misuse
            if self._is_auxiliary_do(original_text) and 'DO' in explication:
                issues.append(PivotIssue(
                    issue_type=PivotIssueType.AUX_DO_MISUSE,
                    description="Auxiliary DO incorrectly treated as eventive",
                    original_text=original_text,
                    problematic_explication=explication,
                    suggested_fix="Remove auxiliary DO from explication",
                    confidence=0.8,
                    line_number=i + 1
                ))
            
            # Check for deictic false positives
            original_lower = original_text.lower()
            deictic_indicators = self.deictic_indicators.get(language, self.deictic_indicators['en'])
            has_explicit_deictic = any(indicator in original_lower for indicator in deictic_indicators)
            
            if 'HERE' in explication and not has_explicit_deictic:
                issues.append(PivotIssue(
                    issue_type=PivotIssueType.DEICTIC_FALSE_POSITIVE,
                    description="HERE used without explicit deictic indicator",
                    original_text=original_text,
                    problematic_explication=explication,
                    suggested_fix="Replace HERE with WHERE or AT-LOC",
                    confidence=0.7,
                    line_number=i + 1
                ))
            
            # Check for ConceptNet relations
            found_cn_relations = [rel for rel in self.conceptnet_relations if rel in explication]
            if found_cn_relations:
                issues.append(PivotIssue(
                    issue_type=PivotIssueType.CONCEPTNET_IN_NSM,
                    description=f"ConceptNet relations found in NSM: {', '.join(found_cn_relations)}",
                    original_text=original_text,
                    problematic_explication=explication,
                    suggested_fix="Replace with NSM equivalents or remove",
                    confidence=0.9,
                    line_number=i + 1
                ))
            
            # Check for duplicate primes
            words = explication.split()
            duplicates = [word for word in set(words) if words.count(word) > 1]
            if duplicates:
                issues.append(PivotIssue(
                    issue_type=PivotIssueType.DUPLICATE_PRIMES,
                    description=f"Duplicate primes found: {', '.join(duplicates)}",
                    original_text=original_text,
                    problematic_explication=explication,
                    suggested_fix="Remove duplicate primes",
                    confidence=0.8,
                    line_number=i + 1
                ))
        
        return issues


class NSMPivotCleaningSystem:
    """Comprehensive NSM pivot cleaning system."""
    
    def __init__(self):
        """Initialize the NSM pivot cleaning system."""
        self.cleaner = NSMPivotCleaner()
    
    def run_cleaning_analysis(self, test_cases: List[Tuple[str, str]], 
                            language: str = "en") -> Dict[str, Any]:
        """Run comprehensive NSM pivot cleaning analysis."""
        logger.info(f"Running NSM pivot cleaning analysis on {len(test_cases)} test cases")
        
        analysis_results = {
            'test_configuration': {
                'num_test_cases': len(test_cases),
                'language': language,
                'timestamp': time.time()
            },
            'cleaned_explications': [],
            'detected_issues': [],
            'cleaning_analysis': {},
            'recommendations': []
        }
        
        # Clean explications
        for original_text, explication in test_cases:
            cleaned = self.cleaner.clean_explication(original_text, explication, language)
            analysis_results['cleaned_explications'].append(cleaned.to_dict())
        
        # Detect issues
        issues = self.cleaner.detect_pivot_issues(test_cases, language)
        analysis_results['detected_issues'] = [issue.to_dict() for issue in issues]
        
        # Analyze results
        analysis_results['cleaning_analysis'] = self._analyze_cleaning_results(
            analysis_results['cleaned_explications'], issues
        )
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_cleaning_recommendations(
            analysis_results['cleaning_analysis']
        )
        
        return analysis_results
    
    def _analyze_cleaning_results(self, cleaned_explications: List[Dict[str, Any]], 
                                issues: List[PivotIssue]) -> Dict[str, Any]:
        """Analyze cleaning results."""
        analysis = {
            'total_explications': len(cleaned_explications),
            'total_issues_detected': len(issues),
            'issue_distribution': defaultdict(int),
            'fix_distribution': defaultdict(int),
            'confidence_distribution': defaultdict(int),
            'improvement_metrics': {}
        }
        
        # Analyze issue distribution
        for issue in issues:
            analysis['issue_distribution'][issue.issue_type.value] += 1
        
        # Analyze fix distribution
        for cleaned in cleaned_explications:
            for fix in cleaned['fixes_applied']:
                fix_type = fix.split(':')[0].strip()
                analysis['fix_distribution'][fix_type] += 1
        
        # Analyze confidence distribution
        confidences = [cleaned['confidence'] for cleaned in cleaned_explications]
        analysis['confidence_distribution'] = {
            'high': len([c for c in confidences if c >= 0.8]),
            'medium': len([c for c in confidences if 0.6 <= c < 0.8]),
            'low': len([c for c in confidences if c < 0.6])
        }
        
        # Calculate improvement metrics
        total_fixes = sum(len(cleaned['fixes_applied']) for cleaned in cleaned_explications)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        analysis['improvement_metrics'] = {
            'total_fixes_applied': total_fixes,
            'average_confidence': avg_confidence,
            'fixes_per_explication': total_fixes / len(cleaned_explications) if cleaned_explications else 0,
            'issue_resolution_rate': len([i for i in issues if i.confidence > 0.7]) / len(issues) if issues else 0
        }
        
        return analysis
    
    def _generate_cleaning_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations from cleaning analysis."""
        recommendations = []
        
        # Issue-based recommendations
        if analysis['total_issues_detected'] > 0:
            recommendations.append(f"Address {analysis['total_issues_detected']} detected pivot issues")
        
        # Most common issue
        if analysis['issue_distribution']:
            most_common_issue = max(analysis['issue_distribution'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Focus on fixing {most_common_issue} issues (most common)")
        
        # Confidence-based recommendations
        if analysis['confidence_distribution']['low'] > 0:
            recommendations.append(f"Improve confidence for {analysis['confidence_distribution']['low']} low-confidence explications")
        
        # Fix-based recommendations
        if analysis['fix_distribution']:
            most_common_fix = max(analysis['fix_distribution'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Prioritize {most_common_fix} fixes (most frequently applied)")
        
        return recommendations


def main():
    """Main function to demonstrate NSM pivot cleaning."""
    logger.info("Starting NSM pivot cleaning demonstration...")
    
    # Initialize cleaning system
    cleaning_system = NSMPivotCleaningSystem()
    
    # Test cases from ChatGPT5's feedback
    test_cases = [
        ("The cat is not on the mat", "X SAY: NOT AtLocation"),
        ("I do not like this weather", "X SAY: NOT DO SimilarTo"),
        ("All children can play here", "X SAY: CAN ALL HERE"),
        ("Some people want to help", "X SAY: SOME Desires Desires"),
        ("The weather is good today", "X SAY: GOOD GOOD")
    ]
    
    # Run cleaning analysis
    analysis_results = cleaning_system.run_cleaning_analysis(test_cases, "en")
    
    # Print results
    print("\n" + "="*80)
    print("NSM PIVOT CLEANING RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Cases: {analysis_results['test_configuration']['num_test_cases']}")
    print(f"  Language: {analysis_results['test_configuration']['language']}")
    
    print(f"\nCleaning Analysis:")
    analysis = analysis_results['cleaning_analysis']
    print(f"  Total Explications: {analysis['total_explications']}")
    print(f"  Total Issues Detected: {analysis['total_issues_detected']}")
    print(f"  Total Fixes Applied: {analysis['improvement_metrics']['total_fixes_applied']}")
    print(f"  Average Confidence: {analysis['improvement_metrics']['average_confidence']:.3f}")
    
    print(f"\nIssue Distribution:")
    for issue_type, count in analysis['issue_distribution'].items():
        print(f"  {issue_type}: {count}")
    
    print(f"\nFix Distribution:")
    for fix_type, count in analysis['fix_distribution'].items():
        print(f"  {fix_type}: {count}")
    
    print(f"\nConfidence Distribution:")
    for level, count in analysis['confidence_distribution'].items():
        print(f"  {level}: {count}")
    
    print(f"\nSample Cleaned Explications:")
    for i, cleaned in enumerate(analysis_results['cleaned_explications'][:3]):
        print(f"  {i+1}. Original: {cleaned['original_text']}")
        print(f"     Before: {cleaned['original_explication']}")
        print(f"     After:  {cleaned['cleaned_explication']}")
        if cleaned['fixes_applied']:
            print(f"     Fixes:  {', '.join(cleaned['fixes_applied'])}")
        print()
    
    print(f"\nRecommendations:")
    for i, recommendation in enumerate(analysis_results['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    # Save results
    output_path = Path("data/nsm_pivot_cleaning_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(convert_numpy_types(analysis_results), f, indent=2)
    
    logger.info(f"NSM pivot cleaning results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("NSM pivot cleaning demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()
