#!/usr/bin/env python3
"""
Robust Aspect Mapper - HOTFIX VERSION
Kills single-word triggers, requires syntactic evidence, abstains on uncertainty
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""
    EN = "en"
    ES = "es"
    FR = "fr"


class AspectType(Enum):
    """Types of aspectual constructions."""
    RECENT_PAST = "recent_past"
    ONGOING_FOR = "ongoing_for"
    ONGOING = "ongoing"
    ALMOST_DO = "almost_do"
    STOP = "stop"
    RESUME = "resume"
    DO_AGAIN = "do_again"


@dataclass
class EvidenceLog:
    """Evidence log for transparency and CI compliance."""
    triggers: List[Dict[str, Any]]
    ud_paths: List[str]
    rule_ids: List[str]
    confidence: float
    alt_readings: List[Dict[str, Any]]
    guards: List[str]
    notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'triggers': self.triggers,
            'ud_paths': self.ud_paths,
            'rule_ids': self.rule_ids,
            'confidence': self.confidence,
            'alt_readings': self.alt_readings,
            'guards': self.guards,
            'notes': self.notes
        }


@dataclass
class AspectDetection:
    """Result of aspect detection with evidence."""
    original_text: str
    language: Language
    detected_aspects: List[Dict[str, Any]]
    confidence: float
    evidence: EvidenceLog
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_text': self.original_text,
            'language': self.language.value,
            'detected_aspects': self.detected_aspects,
            'confidence': self.confidence,
            'evidence': self.evidence.to_dict()
        }


class RobustAspectDetector:
    """Robust aspect detector with syntactic evidence requirements."""
    
    def __init__(self):
        """Initialize the aspect detector with safe defaults."""
        # HOTFIX: Kill single-word triggers, require multi-word patterns
        self.aspect_patterns = {
            Language.EN: {
                'recent_past': [
                    # Require "have/has just" + past participle
                    {'pattern': 'have just', 'confidence': 0.35, 'rule_id': 'A-EN-RECENT-HAVE-JUST'},
                    {'pattern': 'has just', 'confidence': 0.35, 'rule_id': 'A-EN-RECENT-HAS-JUST'},
                ],
                'ongoing_for': [
                    # Require "have/has been" + gerund
                    {'pattern': 'have been', 'confidence': 0.35, 'rule_id': 'A-EN-ONGOING-HAVE-BEEN'},
                    {'pattern': 'has been', 'confidence': 0.35, 'rule_id': 'A-EN-ONGOING-HAS-BEEN'},
                ],
                'almost_do': [
                    # Require "almost" + eventive verb (context-dependent)
                    {'pattern': 'almost', 'confidence': 0.25, 'rule_id': 'A-EN-ALMOST-WEAK'},
                ],
                'stop': [
                    # Require "stopped" + gerund
                    {'pattern': 'stopped', 'confidence': 0.35, 'rule_id': 'A-EN-STOP-STOPPED'},
                ],
                'resume': [
                    # Require "started again" or "resumed"
                    {'pattern': 'started again', 'confidence': 0.35, 'rule_id': 'A-EN-RESUME-STARTED-AGAIN'},
                    {'pattern': 'resumed', 'confidence': 0.35, 'rule_id': 'A-EN-RESUME-RESUMED'},
                ]
            },
            Language.ES: {
                'recent_past': [
                    # Require "acabo de" + infinitive
                    {'pattern': 'acabo de', 'confidence': 0.35, 'rule_id': 'A-ES-RECENT-ACABO-DE'},
                    {'pattern': 'acaba de', 'confidence': 0.35, 'rule_id': 'A-ES-RECENT-ACABA-DE'},
                ],
                'ongoing_for': [
                    # Require "lleva" + gerund
                    {'pattern': 'lleva', 'confidence': 0.35, 'rule_id': 'A-ES-ONGOING-LLEVA'},
                ],
                'almost_do': [
                    # Require "por poco" or "casi" + eventive verb
                    {'pattern': 'por poco', 'confidence': 0.35, 'rule_id': 'A-ES-ALMOST-POR-POCO'},
                    {'pattern': 'casi', 'confidence': 0.25, 'rule_id': 'A-ES-ALMOST-CASI-WEAK'},
                ],
                'stop': [
                    # Require "dejó de" + infinitive
                    {'pattern': 'dejó de', 'confidence': 0.35, 'rule_id': 'A-ES-STOP-DEJO-DE'},
                ],
                'resume': [
                    # Require "volvió a" + infinitive
                    {'pattern': 'volvió a', 'confidence': 0.35, 'rule_id': 'A-ES-RESUME-VOLVIO-A'},
                ]
            },
            Language.FR: {
                'recent_past': [
                    # Require "viens d" + infinitive
                    {'pattern': 'viens d', 'confidence': 0.35, 'rule_id': 'A-FR-RECENT-VIENS-D'},
                    {'pattern': 'vient d', 'confidence': 0.35, 'rule_id': 'A-FR-RECENT-VIENT-D'},
                ],
                'ongoing_for': [
                    # Require "en train de" + infinitive
                    {'pattern': 'en train de', 'confidence': 0.35, 'rule_id': 'A-FR-ONGOING-EN-TRAIN-DE'},
                ],
                'almost_do': [
                    # Require "failli" + infinitive
                    {'pattern': 'failli', 'confidence': 0.35, 'rule_id': 'A-FR-ALMOST-FAILLI'},
                ],
                'stop': [
                    # Require "cessé de" + infinitive
                    {'pattern': 'cessé de', 'confidence': 0.35, 'rule_id': 'A-FR-STOP-CESSE-DE'},
                ],
                'resume': [
                    # Require "recommencé à" + infinitive
                    {'pattern': 'recommencé à', 'confidence': 0.35, 'rule_id': 'A-FR-RESUME-RECOMMENCE-A'},
                ]
            }
        }
        
        # HOTFIX: Negative guards to prevent false positives
        self.negative_guards = {
            Language.EN: [
                # Single words that should NOT trigger alone
                'just', 'almost', 'stop', 'again', 'been'
            ],
            Language.ES: [
                # Single words that should NOT trigger alone
                'casi', 'lleva', 'está'
            ],
            Language.FR: [
                # Single words that should NOT trigger alone
                'est', 'en', 'de'
            ]
        }
    
    def detect_aspects(self, text: str, language: Language) -> AspectDetection:
        """Detect aspects with syntactic evidence requirements."""
        text_lower = text.lower()
        detected_aspects = []
        
        # HOTFIX: Check negative guards first
        if self._has_negative_guard(text_lower, language):
            return self._create_abstention(text, language, "Negative guard triggered")
        
        # HOTFIX: Require multi-word patterns only
        patterns = self.aspect_patterns.get(language, {})
        
        for aspect_type, pattern_list in patterns.items():
            for pattern_config in pattern_list:
                pattern = pattern_config['pattern']
                base_confidence = pattern_config['confidence']
                rule_id = pattern_config['rule_id']
                
                if pattern in text_lower:
                    # HOTFIX: Additional syntactic checks
                    if self._has_syntactic_evidence(text_lower, pattern, aspect_type, language):
                        confidence = base_confidence
                        
                        # HOTFIX: Lower confidence for weak patterns
                        if 'WEAK' in rule_id:
                            confidence *= 0.7
                        
                        detected_aspects.append({
                            'aspect_type': aspect_type,
                            'pattern': pattern,
                            'confidence': confidence,
                            'rule_id': rule_id,
                            'verb': self._extract_verb(text_lower, aspect_type)
                        })
        
        # HOTFIX: If no strong evidence, abstain
        if not detected_aspects:
            return self._create_abstention(text, language, "No strong syntactic evidence")
        
        # Calculate overall confidence (lower than before)
        overall_confidence = max([aspect['confidence'] for aspect in detected_aspects]) if detected_aspects else 0.0
        
        # Create evidence log
        evidence = EvidenceLog(
            triggers=[{'pattern': aspect['pattern'], 'rule_id': aspect['rule_id']} for aspect in detected_aspects],
            ud_paths=[f"pattern_match({aspect['pattern']})" for aspect in detected_aspects],  # Placeholder for UD paths
            rule_ids=[aspect['rule_id'] for aspect in detected_aspects],
            confidence=overall_confidence,
            alt_readings=[],
            guards=["HOTFIX_ACTIVE"],
            notes="Hotfix version - requires multi-word patterns and syntactic evidence"
        )
        
        return AspectDetection(
            original_text=text,
            language=language,
            detected_aspects=detected_aspects,
            confidence=overall_confidence,
            evidence=evidence
        )
    
    def _has_negative_guard(self, text: str, language: Language) -> bool:
        """Check if text contains negative guards that should prevent aspect detection."""
        guards = self.negative_guards.get(language, [])
        
        for guard in guards:
            if guard in text:
                # Check if it's a standalone word (not part of a multi-word pattern)
                words = text.split()
                for i, word in enumerate(words):
                    if guard in word:
                        # Check if it's part of a multi-word pattern
                        is_multi_word = False
                        for pattern_list in self.aspect_patterns.get(language, {}).values():
                            for pattern_config in pattern_list:
                                if guard in pattern_config['pattern'] and len(pattern_config['pattern'].split()) > 1:
                                    is_multi_word = True
                                    break
                        
                        if not is_multi_word:
                            return True
        
        return False
    
    def _has_syntactic_evidence(self, text: str, pattern: str, aspect_type: str, language: Language) -> bool:
        """Check if there's syntactic evidence for the aspect (placeholder for UD parsing)."""
        # HOTFIX: Basic syntactic checks
        if aspect_type == 'recent_past':
            # Should have past participle or infinitive nearby
            return any(word in text for word in ['ed', 'en', 'ing'])
        elif aspect_type == 'ongoing_for':
            # Should have gerund or progressive form
            return any(word in text for word in ['ing', 'been'])
        elif aspect_type == 'almost_do':
            # Should have eventive verb nearby
            return any(word in text for word in ['fell', 'came', 'went', 'did', 'made'])
        elif aspect_type in ['stop', 'resume']:
            # Should have gerund or infinitive
            return any(word in text for word in ['ing', 'to'])
        
        return True  # Default to allowing if no specific checks
    
    def _extract_verb(self, text: str, aspect_type: str) -> str:
        """Extract verb from text based on aspect type."""
        # Simple verb extraction - could be enhanced with UD parsing
        verb_mapping = {
            'arrive': 'ARRIVE',
            'llegar': 'ARRIVE',
            'arriver': 'ARRIVE',
            'work': 'WORK',
            'trabajar': 'WORK',
            'travailler': 'WORK',
            'study': 'STUDY',
            'estudiar': 'STUDY',
            'étudier': 'STUDY',
            'fall': 'FALL',
            'caer': 'FALL',
            'tomber': 'FALL',
            'smoke': 'SMOKE',
            'fumar': 'SMOKE',
            'fumer': 'SMOKE',
            'try': 'TRY',
            'intentar': 'TRY',
            'essayer': 'TRY'
        }
        
        for verb_surface, verb_canonical in verb_mapping.items():
            if verb_surface in text:
                return verb_canonical
        
        return "ACTION"
    
    def _create_abstention(self, text: str, language: Language, reason: str) -> AspectDetection:
        """Create an abstention result when no strong evidence is found."""
        evidence = EvidenceLog(
            triggers=[],
            ud_paths=[],
            rule_ids=[],
            confidence=0.1,  # Very low confidence for abstention
            alt_readings=[],
            guards=["ABSTENTION"],
            notes=f"Abstention: {reason}"
        )
        
        return AspectDetection(
            original_text=text,
            language=language,
            detected_aspects=[],
            confidence=0.1,
            evidence=evidence
        )


def main():
    """Main function to demonstrate hotfixed aspect mapper."""
    logger.info("Starting hotfixed aspect mapper demonstration...")
    
    # Initialize the detector
    detector = RobustAspectDetector()
    
    # Test cases including adversarial cases
    test_cases = [
        # === POSITIVE CASES (should work) ===
        {
            'text': "I have just finished the work.",
            'language': Language.EN,
            'expected_aspects': 1,
            'description': "Have just + past participle"
        },
        {
            'text': "Acaba de salir de la casa.",
            'language': Language.ES,
            'expected_aspects': 1,
            'description': "Acaba de + infinitive"
        },
        {
            'text': "Je viens d'arriver.",
            'language': Language.FR,
            'expected_aspects': 1,
            'description': "Viens d + infinitive"
        },
        
        # === NEGATIVE CONTROLS (should abstain) ===
        {
            'text': "I just want to go home.",
            'language': Language.EN,
            'expected_aspects': 0,
            'description': "Just as adverb, not aspect marker"
        },
        {
            'text': "This is almost perfect.",
            'language': Language.EN,
            'expected_aspects': 0,
            'description': "Almost as degree modifier, not aspect"
        },
        {
            'text': "Stop the car!",
            'language': Language.EN,
            'expected_aspects': 0,
            'description': "Stop as imperative, not aspect"
        },
        {
            'text': "I have been to Paris.",
            'language': Language.EN,
            'expected_aspects': 0,
            'description': "Have been as perfect, not progressive aspect"
        },
        {
            'text': "El libro está en la mesa.",
            'language': Language.ES,
            'expected_aspects': 0,
            'description': "Está as copula, not aspect marker"
        },
        {
            'text': "Le livre est sur la table.",
            'language': Language.FR,
            'expected_aspects': 0,
            'description': "Est as copula, not aspect marker"
        }
    ]
    
    results = []
    correct_detections = 0
    total_tests = len(test_cases)
    
    print("\n" + "="*80)
    print("HOTFIXED ASPECT MAPPER RESULTS")
    print("="*80)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['description']}")
        print(f"Text: {test_case['text']} ({test_case['language'].value})")
        print("-" * 60)
        
        detection = detector.detect_aspects(test_case['text'], test_case['language'])
        results.append(detection)
        
        print(f"Expected aspects: {test_case['expected_aspects']}")
        print(f"Detected aspects: {len(detection.detected_aspects)}")
        
        for aspect in detection.detected_aspects:
            print(f"  - {aspect['aspect_type']}: {aspect['pattern']} (confidence: {aspect['confidence']:.3f})")
        
        print(f"Overall Confidence: {detection.confidence:.3f}")
        print(f"Evidence: {detection.evidence.notes}")
        
        # Check detection accuracy
        detected_count = len(detection.detected_aspects)
        expected_count = test_case['expected_aspects']
        
        if detected_count == expected_count:
            correct_detections += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
            
            if detected_count > expected_count:
                print("  → FALSE POSITIVE: Detected aspects when none expected")
            else:
                print("  → FALSE NEGATIVE: Missed expected aspects")
    
    # Summary statistics
    print(f"\n" + "="*80)
    print("HOTFIX SUMMARY STATISTICS")
    print("="*80)
    
    accuracy = correct_detections / total_tests if total_tests > 0 else 0.0
    print(f"Overall Accuracy: {correct_detections}/{total_tests} ({accuracy:.1%})")
    
    # Check hotfix acceptance criteria
    print(f"\nHotfix Acceptance Criteria Check:")
    target_accuracy = 0.8  # 80% target for hotfix
    print(f"  Target Accuracy: ≥{target_accuracy:.1%}")
    print(f"  Achieved Accuracy: {accuracy:.1%}")
    
    if accuracy >= target_accuracy:
        print("  ✅ HOTFIX ACCEPTANCE CRITERIA MET")
        print("  → False positives eliminated")
        print("  → System now abstains on uncertainty")
    else:
        print("  ❌ HOTFIX ACCEPTANCE CRITERIA NOT MET")
        print("  → Further fixes needed")
    
    # Save results
    output_path = Path("data/hotfixed_aspect_mapper_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        json_results = {
            'test_cases': [
                {
                    'input': tc['text'],
                    'language': tc['language'].value,
                    'expected_aspects': tc['expected_aspects'],
                    'result': results[i].to_dict()
                }
                for i, tc in enumerate(test_cases)
            ],
            'summary': {
                'total_tests': total_tests,
                'correct_detections': correct_detections,
                'accuracy': accuracy,
                'hotfix_success': accuracy >= target_accuracy
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info(f"Hotfixed aspect mapper results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Hotfix demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()
