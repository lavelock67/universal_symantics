#!/usr/bin/env python3
"""
Aspect Mapper: Wire Detection → NSM → EIL.

This script implements aspect molecule mapping as specified to fix the 0% success rate:
- RECENT_PAST: FR 'venir de' + INF; ES 'acabar de' + INF
- ONGOING_FOR: ES 'llevar + gerundio (+ duración)'; FR 'être en train de'
- ALMOST_DO: FR 'faillir + INF'; ES 'por poco (se) + VERB'
- STOP/RESUME/DO_AGAIN: ES 'dejar de / volver a'; FR 'cesser de / recommencer'
- Timestamps/durations: parse 'tres horas / trois heures / today/ayer/aujourd'hui'
- Map to EIL operators with temporal constraints
- Reasoner rules (A1 family) for aspect propagation
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
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class AspectType(Enum):
    """Types of aspect molecules."""
    RECENT_PAST = "recent_past"
    ONGOING_FOR = "ongoing_for"
    ALMOST_DO = "almost_do"
    STOP = "stop"
    RESUME = "resume"
    DO_AGAIN = "do_again"


class Language(Enum):
    """Supported languages."""
    EN = "en"
    ES = "es"
    FR = "fr"


@dataclass
class AspectPattern:
    """An aspect pattern in a specific language."""
    pattern: str
    aspect_type: AspectType
    language: Language
    nsm_template: str
    eil_template: str
    examples: List[str]
    confidence_boost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pattern': self.pattern,
            'aspect_type': self.aspect_type.value,
            'language': self.language.value,
            'nsm_template': self.nsm_template,
            'eil_template': self.eil_template,
            'examples': self.examples,
            'confidence_boost': self.confidence_boost
        }


@dataclass
class TemporalExpression:
    """A temporal expression (duration or timestamp)."""
    text: str
    duration_hours: Optional[float]
    timestamp: Optional[str]
    language: Language
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'text': self.text,
            'duration_hours': self.duration_hours,
            'timestamp': self.timestamp,
            'language': self.language.value,
            'confidence': self.confidence
        }


@dataclass
class AspectDetection:
    """Result of aspect detection."""
    original_text: str
    language: Language
    detected_aspects: List[AspectPattern]
    temporal_expressions: List[TemporalExpression]
    nsm_representation: str
    eil_representation: str
    confidence: float
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_text': self.original_text,
            'language': self.language.value,
            'detected_aspects': [a.to_dict() for a in self.detected_aspects],
            'temporal_expressions': [t.to_dict() for t in self.temporal_expressions],
            'nsm_representation': self.nsm_representation,
            'eil_representation': self.eil_representation,
            'confidence': self.confidence,
            'warnings': self.warnings
        }


class AspectPatternDatabase:
    """Database of aspect patterns for different languages."""
    
    def __init__(self):
        """Initialize the aspect pattern database."""
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[Language, List[AspectPattern]]:
        """Initialize aspect patterns for all languages."""
        patterns = {
            Language.EN: self._get_english_patterns(),
            Language.ES: self._get_spanish_patterns(),
            Language.FR: self._get_french_patterns()
        }
        return patterns
    
    def _get_english_patterns(self) -> List[AspectPattern]:
        """Get English aspect patterns."""
        return [
            AspectPattern(
                pattern=r"\bjust\s+(\w+ed|\w+)\b",
                aspect_type=AspectType.RECENT_PAST,
                language=Language.EN,
                nsm_template="RECENT_PAST({verb})",
                eil_template="PAST({verb}) ∧ close(now, {verb})",
                examples=["just arrived", "just finished", "just called"],
                confidence_boost=0.9
            ),
            AspectPattern(
                pattern=r"\b(has|have)\s+been\s+(\w+ing)\s+(for\s+\w+)?",
                aspect_type=AspectType.ONGOING_FOR,
                language=Language.EN,
                nsm_template="ONGOING_FOR({verb}, {duration})",
                eil_template="DURING({verb}, now-{duration}..now)",
                examples=["has been working for three hours", "have been studying"],
                confidence_boost=0.8
            ),
            AspectPattern(
                pattern=r"\balmost\s+(\w+ed|\w+)\b",
                aspect_type=AspectType.ALMOST_DO,
                language=Language.EN,
                nsm_template="ALMOST_DO({verb})",
                eil_template="¬{verb} ∧ near({verb})",
                examples=["almost fell", "almost finished", "almost called"],
                confidence_boost=0.9
            ),
            AspectPattern(
                pattern=r"\bstopped\s+(\w+ing)\b",
                aspect_type=AspectType.STOP,
                language=Language.EN,
                nsm_template="STOP({verb})",
                eil_template="BEFORE(DURING({verb}, t1..t2), now) ∧ ¬DURING({verb}, now)",
                examples=["stopped working", "stopped studying"],
                confidence_boost=0.8
            ),
            AspectPattern(
                pattern=r"\bresumed\s+(\w+ing)|\bstarted\s+(\w+ing)\s+again\b",
                aspect_type=AspectType.RESUME,
                language=Language.EN,
                nsm_template="RESUME({verb})",
                eil_template="BEFORE(DURING({verb}, t1..t2), t3) ∧ DURING({verb}, now)",
                examples=["resumed working", "started studying again"],
                confidence_boost=0.8
            ),
            AspectPattern(
                pattern=r"\b(\w+ed|\w+)\s+again\b",
                aspect_type=AspectType.DO_AGAIN,
                language=Language.EN,
                nsm_template="DO_AGAIN({verb})",
                eil_template="again({verb}_now, {verb}_before)",
                examples=["called again", "visited again", "tried again"],
                confidence_boost=0.8
            )
        ]
    
    def _get_spanish_patterns(self) -> List[AspectPattern]:
        """Get Spanish aspect patterns."""
        return [
            AspectPattern(
                pattern=r"\bacab[aáo]\s+de\s+(\w+r)\b",
                aspect_type=AspectType.RECENT_PAST,
                language=Language.ES,
                nsm_template="RECENT_PAST({verb})",
                eil_template="PAST({verb}) ∧ close(now, {verb})",
                examples=["acaba de llegar", "acabó de terminar", "acaban de salir"],
                confidence_boost=0.95
            ),
            AspectPattern(
                pattern=r"\bllev[aáo]\s+(\w+ndo)\s+(durante\s+|por\s+)?([^\.]+)?",
                aspect_type=AspectType.ONGOING_FOR,
                language=Language.ES,
                nsm_template="ONGOING_FOR({verb}, {duration})",
                eil_template="DURING({verb}, now-{duration}..now)",
                examples=["lleva estudiando tres horas", "llevaba trabajando desde ayer"],
                confidence_boost=0.9
            ),
            AspectPattern(
                pattern=r"\bpor\s+poco\s+(se\s+)?(\w+[aáoóeé]?)\b",
                aspect_type=AspectType.ALMOST_DO,
                language=Language.ES,
                nsm_template="ALMOST_DO({verb})",
                eil_template="¬{verb} ∧ near({verb})",
                examples=["por poco se cae", "por poco llueve", "por poco pierden"],
                confidence_boost=0.95
            ),
            AspectPattern(
                pattern=r"\bdej[aáoó]\s+de\s+(\w+r)\b",
                aspect_type=AspectType.STOP,
                language=Language.ES,
                nsm_template="STOP({verb})",
                eil_template="BEFORE(DURING({verb}, t1..t2), now) ∧ ¬DURING({verb}, now)",
                examples=["dejó de fumar", "dejaron de trabajar", "dejé de estudiar"],
                confidence_boost=0.9
            ),
            AspectPattern(
                pattern=r"\bvolv[íióoe]+\s+a\s+(\w+r?)\b",
                aspect_type=AspectType.RESUME,
                language=Language.ES,
                nsm_template="RESUME({verb})",
                eil_template="BEFORE(DURING({verb}, t1..t2), t3) ∧ DURING({verb}, now)",
                examples=["volvió a llamar", "volvieron a intentar", "volví a estudiar"],
                confidence_boost=0.9
            ),
            AspectPattern(
                pattern=r"\b(\w+[aáoóeé]?)\s+(otra\s+vez|de\s+nuevo)\b",
                aspect_type=AspectType.DO_AGAIN,
                language=Language.ES,
                nsm_template="DO_AGAIN({verb})",
                eil_template="again({verb}_now, {verb}_before)",
                examples=["llamó otra vez", "intentó de nuevo", "visitó otra vez"],
                confidence_boost=0.8
            )
        ]
    
    def _get_french_patterns(self) -> List[AspectPattern]:
        """Get French aspect patterns."""
        return [
            AspectPattern(
                pattern=r"\bvien[st]+\s+d['']\s*(\w+r)\b",
                aspect_type=AspectType.RECENT_PAST,
                language=Language.FR,
                nsm_template="RECENT_PAST({verb})",
                eil_template="PAST({verb}) ∧ close(now, {verb})",
                examples=["vient d'arriver", "viens de terminer", "viennent de partir"],
                confidence_boost=0.95
            ),
            AspectPattern(
                pattern=r"\b(être|est|suis|sommes|êtes|sont)\s+en\s+train\s+de\s+(\w+r)\b",
                aspect_type=AspectType.ONGOING_FOR,
                language=Language.FR,
                nsm_template="ONGOING_FOR({verb}, now)",
                eil_template="DURING({verb}, now)",
                examples=["est en train de travailler", "sont en train d'étudier"],
                confidence_boost=0.9
            ),
            AspectPattern(
                pattern=r"\bfailli[rtls]?\s+(\w+r)\b",
                aspect_type=AspectType.ALMOST_DO,
                language=Language.FR,
                nsm_template="ALMOST_DO({verb})",
                eil_template="¬{verb} ∧ near({verb})",
                examples=["a failli tomber", "faillit mourir", "avait failli oublier"],
                confidence_boost=0.95
            ),
            AspectPattern(
                pattern=r"\bcess[eéèaèr]+\s+(de\s+)?(\w+r)\b",
                aspect_type=AspectType.STOP,
                language=Language.FR,
                nsm_template="STOP({verb})",
                eil_template="BEFORE(DURING({verb}, t1..t2), now) ∧ ¬DURING({verb}, now)",
                examples=["cesse de fumer", "cessé de travailler", "cessent d'étudier"],
                confidence_boost=0.9
            ),
            AspectPattern(
                pattern=r"\brecommenc[eéèaèr]+\s+(à\s+)?(\w+r)\b",
                aspect_type=AspectType.RESUME,
                language=Language.FR,
                nsm_template="RESUME({verb})",
                eil_template="BEFORE(DURING({verb}, t1..t2), t3) ∧ DURING({verb}, now)",
                examples=["recommence à travailler", "recommencé à étudier"],
                confidence_boost=0.9
            ),
            AspectPattern(
                pattern=r"\b(\w+[eé]?)\s+(encore|à\s+nouveau)\b",
                aspect_type=AspectType.DO_AGAIN,
                language=Language.FR,
                nsm_template="DO_AGAIN({verb})",
                eil_template="again({verb}_now, {verb}_before)",
                examples=["appelé encore", "essayé à nouveau", "visité encore"],
                confidence_boost=0.8
            )
        ]
    
    def get_patterns_for_language(self, language: Language) -> List[AspectPattern]:
        """Get patterns for a specific language."""
        return self.patterns.get(language, [])


class TemporalParser:
    """Parses temporal expressions (durations and timestamps)."""
    
    def __init__(self):
        """Initialize the temporal parser."""
        self.duration_patterns = {
            Language.EN: [
                (r"\b(\d+)\s+hours?\b", 1.0),
                (r"\bthree\s+hours?\b", 3.0),
                (r"\btwo\s+hours?\b", 2.0),
                (r"\bone\s+hour\b", 1.0),
                (r"\ball\s+day\b", 8.0),
                (r"\ball\s+morning\b", 4.0)
            ],
            Language.ES: [
                (r"\b(\d+)\s+horas?\b", 1.0),
                (r"\btres\s+horas?\b", 3.0),
                (r"\bdos\s+horas?\b", 2.0),
                (r"\buna\s+hora\b", 1.0),
                (r"\btodo\s+el\s+día\b", 8.0),
                (r"\btoda\s+la\s+mañana\b", 4.0)
            ],
            Language.FR: [
                (r"\b(\d+)\s+heures?\b", 1.0),
                (r"\btrois\s+heures?\b", 3.0),
                (r"\bdeux\s+heures?\b", 2.0),
                (r"\bune\s+heure\b", 1.0),
                (r"\btoute\s+la\s+journée\b", 8.0),
                (r"\btoute\s+la\s+matinée\b", 4.0)
            ]
        }
        
        self.timestamp_patterns = {
            Language.EN: [
                (r"\btoday\b", "today"),
                (r"\byesterday\b", "yesterday"),
                (r"\btomorrow\b", "tomorrow"),
                (r"\bnow\b", "now"),
                (r"\bthis\s+morning\b", "this_morning")
            ],
            Language.ES: [
                (r"\bhoy\b", "today"),
                (r"\bayer\b", "yesterday"),
                (r"\bmañana\b", "tomorrow"),
                (r"\bahora\b", "now"),
                (r"\besta\s+mañana\b", "this_morning")
            ],
            Language.FR: [
                (r"\baujourd'hui\b", "today"),
                (r"\bhier\b", "yesterday"),
                (r"\bdemain\b", "tomorrow"),
                (r"\bmaintenant\b", "now"),
                (r"\bce\s+matin\b", "this_morning")
            ]
        }
    
    def parse_temporal_expressions(self, text: str, language: Language) -> List[TemporalExpression]:
        """Parse temporal expressions from text."""
        expressions = []
        text_lower = text.lower()
        
        # Parse durations
        duration_patterns = self.duration_patterns.get(language, [])
        for pattern, base_hours in duration_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if match.group(1) if len(match.groups()) > 0 and match.group(1).isdigit() else None:
                    duration = float(match.group(1))
                else:
                    duration = base_hours
                
                expressions.append(TemporalExpression(
                    text=match.group(0),
                    duration_hours=duration,
                    timestamp=None,
                    language=language,
                    confidence=0.9
                ))
        
        # Parse timestamps
        timestamp_patterns = self.timestamp_patterns.get(language, [])
        for pattern, timestamp_value in timestamp_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                expressions.append(TemporalExpression(
                    text=match.group(0),
                    duration_hours=None,
                    timestamp=timestamp_value,
                    language=language,
                    confidence=0.8
                ))
        
        return expressions


class AspectDetector:
    """Detects aspect patterns in text."""
    
    def __init__(self):
        """Initialize the aspect detector."""
        self.pattern_db = AspectPatternDatabase()
        self.temporal_parser = TemporalParser()
    
    def detect_aspects(self, text: str, language: Language) -> AspectDetection:
        """Detect aspect patterns in text."""
        logger.info(f"Detecting aspects in: {text} ({language.value})")
        
        # Get patterns for the language
        patterns = self.pattern_db.get_patterns_for_language(language)
        
        # Detect aspect patterns
        detected_aspects = self._detect_aspect_patterns(text, patterns)
        
        # Parse temporal expressions
        temporal_expressions = self.temporal_parser.parse_temporal_expressions(text, language)
        
        # Generate NSM and EIL representations
        nsm_repr, eil_repr, confidence = self._compile_representations(
            text, detected_aspects, temporal_expressions
        )
        
        warnings = []
        if not detected_aspects:
            warnings.append("No aspect patterns detected")
        if confidence < 0.7:
            warnings.append("Low confidence in aspect detection")
        
        return AspectDetection(
            original_text=text,
            language=language,
            detected_aspects=detected_aspects,
            temporal_expressions=temporal_expressions,
            nsm_representation=nsm_repr,
            eil_representation=eil_repr,
            confidence=confidence,
            warnings=warnings
        )
    
    def _detect_aspect_patterns(self, text: str, patterns: List[AspectPattern]) -> List[AspectPattern]:
        """Detect aspect patterns in text."""
        detected = []
        text_lower = text.lower()
        
        for pattern in patterns:
            if re.search(pattern.pattern, text_lower):
                detected.append(pattern)
        
        return detected
    
    def _compile_representations(self, text: str, aspects: List[AspectPattern], 
                               temporal_exprs: List[TemporalExpression]) -> Tuple[str, str, float]:
        """Compile NSM and EIL representations."""
        if not aspects:
            return "", "", 0.0
        
        # Use the first detected aspect (could be enhanced to handle multiple)
        primary_aspect = aspects[0]
        
        # Extract verb from text (simplified)
        verb = self._extract_verb(text, primary_aspect)
        
        # Extract duration if available
        duration = None
        for expr in temporal_exprs:
            if expr.duration_hours is not None:
                duration = f"{expr.duration_hours}h"
                break
        
        # Generate NSM representation
        nsm_repr = primary_aspect.nsm_template.format(
            verb=verb,
            duration=duration if duration else "now"
        )
        
        # Generate EIL representation
        eil_repr = primary_aspect.eil_template.format(
            verb=verb,
            duration=duration if duration else "now"
        )
        
        # Calculate confidence
        confidence = primary_aspect.confidence_boost
        if temporal_exprs:
            confidence = min(1.0, confidence + 0.1)  # Boost for temporal info
        
        return nsm_repr, eil_repr, confidence
    
    def _extract_verb(self, text: str, aspect_pattern: AspectPattern) -> str:
        """Extract verb from text based on aspect pattern."""
        # Simplified verb extraction
        verb_mapping = {
            Language.EN: {
                "arrive": "ARRIVE", "finish": "FINISH", "call": "CALL",
                "work": "WORK", "study": "STUDY", "fall": "FALL",
                "smoke": "SMOKE", "visit": "VISIT", "try": "TRY"
            },
            Language.ES: {
                "llegar": "ARRIVE", "terminar": "FINISH", "llamar": "CALL",
                "trabajar": "WORK", "estudiar": "STUDY", "caer": "FALL",
                "fumar": "SMOKE", "visitar": "VISIT", "intentar": "TRY",
                "intent": "TRY"  # Handle "intentar" without the -ar ending
            },
            Language.FR: {
                "arriver": "ARRIVE", "terminer": "FINISH", "appeler": "CALL",
                "travailler": "WORK", "étudier": "STUDY", "tomber": "FALL",
                "fumer": "SMOKE", "visiter": "VISIT", "essayer": "TRY",
                "arriv": "ARRIVE"  # Handle "arriver" without the -er ending
            }
        }
        
        text_lower = text.lower()
        language_verbs = verb_mapping.get(aspect_pattern.language, {})
        
        for verb_surface, verb_canonical in language_verbs.items():
            if verb_surface in text_lower:
                return verb_canonical
        
        return "ACTION"


class EILAspectReasoner:
    """Applies A1 family reasoning rules for aspects."""
    
    def __init__(self):
        """Initialize the EIL aspect reasoner."""
        self.aspect_rules = {
            "recent_past_closure": {
                "input_pattern": r"RECENT_PAST\((\w+)\)",
                "output_template": "PAST({verb}) ∧ close(now, {verb}) ∧ ¬LONG_AGO({verb})",
                "confidence": 0.9
            },
            "ongoing_temporal_bounds": {
                "input_pattern": r"ONGOING_FOR\((\w+), (.+)\)",
                "output_template": "DURING({verb}, now-{duration}..now) ∧ CONTINUOUS({verb})",
                "confidence": 0.9
            },
            "almost_negation": {
                "input_pattern": r"ALMOST_DO\((\w+)\)",
                "output_template": "¬{verb} ∧ near({verb}) ∧ POSSIBLE({verb})",
                "confidence": 0.9
            },
            "stop_resume_chain": {
                "input_pattern": r"STOP\((\w+)\)",
                "output_template": "∃t1,t2 [BEFORE(DURING({verb}, t1..t2), now) ∧ ¬DURING({verb}, now)]",
                "confidence": 0.8
            },
            "do_again_precedence": {
                "input_pattern": r"DO_AGAIN\((\w+)\)",
                "output_template": "∃t1 [BEFORE({verb}(t1), now) ∧ {verb}(now) ∧ same_type({verb}(t1), {verb}(now))]",
                "confidence": 0.8
            }
        }
    
    def apply_aspect_rules(self, eil_representation: str) -> List[Dict[str, Any]]:
        """Apply A1 family aspect reasoning rules."""
        applied_rules = []
        
        for rule_name, rule_config in self.aspect_rules.items():
            pattern = rule_config["input_pattern"]
            template = rule_config["output_template"]
            confidence = rule_config["confidence"]
            
            match = re.search(pattern, eil_representation)
            if match:
                # Extract variables from the pattern
                variables = {}
                if "verb" in template:
                    variables["verb"] = match.group(1) if match.groups() else "ACTION"
                if "duration" in template and len(match.groups()) > 1:
                    variables["duration"] = match.group(2)
                
                # Apply the rule
                output = template.format(**variables)
                
                applied_rules.append({
                    'rule_name': rule_name,
                    'input': eil_representation,
                    'output': output,
                    'confidence': confidence,
                    'variables': variables
                })
        
        return applied_rules


def main():
    """Main function to demonstrate aspect mapper."""
    logger.info("Starting aspect mapper demonstration...")
    
    # Initialize the aspect detector
    detector = AspectDetector()
    reasoner = EILAspectReasoner()
    
    # Test cases from the specification
    test_cases = [
        # Spanish cases
        {
            'text': "Acabo de llegar.",
            'language': Language.ES,
            'expected_aspect': AspectType.RECENT_PAST
        },
        {
            'text': "Lleva estudiando tres horas.",
            'language': Language.ES,
            'expected_aspect': AspectType.ONGOING_FOR
        },
        {
            'text': "Por poco se cae.",
            'language': Language.ES,
            'expected_aspect': AspectType.ALMOST_DO
        },
        {
            'text': "Dejó de fumar.",
            'language': Language.ES,
            'expected_aspect': AspectType.STOP
        },
        {
            'text': "Volvió a intentar.",
            'language': Language.ES,
            'expected_aspect': AspectType.RESUME
        },
        # French cases
        {
            'text': "Je viens d'arriver.",
            'language': Language.FR,
            'expected_aspect': AspectType.RECENT_PAST
        },
        {
            'text': "Il est en train de travailler.",
            'language': Language.FR,
            'expected_aspect': AspectType.ONGOING_FOR
        },
        {
            'text': "Il a failli tomber.",
            'language': Language.FR,
            'expected_aspect': AspectType.ALMOST_DO
        },
        {
            'text': "Elle a cessé de fumer.",
            'language': Language.FR,
            'expected_aspect': AspectType.STOP
        },
        {
            'text': "Il a recommencé à étudier.",
            'language': Language.FR,
            'expected_aspect': AspectType.RESUME
        },
        # English cases
        {
            'text': "I just arrived.",
            'language': Language.EN,
            'expected_aspect': AspectType.RECENT_PAST
        },
        {
            'text': "She has been working for three hours.",
            'language': Language.EN,
            'expected_aspect': AspectType.ONGOING_FOR
        },
        {
            'text': "He almost fell.",
            'language': Language.EN,
            'expected_aspect': AspectType.ALMOST_DO
        }
    ]
    
    results = []
    correct_detections = 0
    total_tests = len(test_cases)
    
    print("\n" + "="*80)
    print("ASPECT MAPPER RESULTS")
    print("="*80)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['text']} ({test_case['language'].value})")
        print("-" * 60)
        
        detection = detector.detect_aspects(test_case['text'], test_case['language'])
        results.append(detection)
        
        print(f"Detected Aspects: {len(detection.detected_aspects)}")
        for aspect in detection.detected_aspects:
            print(f"  - {aspect.aspect_type.value}: {aspect.pattern}")
        
        print(f"Temporal Expressions: {len(detection.temporal_expressions)}")
        for temporal in detection.temporal_expressions:
            if temporal.duration_hours:
                print(f"  - Duration: {temporal.text} ({temporal.duration_hours}h)")
            if temporal.timestamp:
                print(f"  - Timestamp: {temporal.text} ({temporal.timestamp})")
        
        print(f"NSM Representation: {detection.nsm_representation}")
        print(f"EIL Representation: {detection.eil_representation}")
        print(f"Confidence: {detection.confidence:.3f}")
        
        # Apply EIL reasoning rules
        if detection.eil_representation:
            applied_rules = reasoner.apply_aspect_rules(detection.eil_representation)
            if applied_rules:
                print(f"EIL A1 Rules Applied:")
                for rule in applied_rules:
                    print(f"  - {rule['rule_name']}: {rule['output']}")
        
        if detection.warnings:
            print(f"Warnings: {detection.warnings}")
        
        # Check detection accuracy
        if detection.detected_aspects:
            detected_type = detection.detected_aspects[0].aspect_type
            if detected_type == test_case['expected_aspect']:
                correct_detections += 1
                print("✅ Aspect detection: CORRECT")
            else:
                print("❌ Aspect detection: INCORRECT")
        else:
            print("❌ Aspect detection: NO DETECTION")
    
    # Summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    accuracy = correct_detections / total_tests if total_tests > 0 else 0.0
    print(f"Overall Accuracy: {correct_detections}/{total_tests} ({accuracy:.1%})")
    
    # Analyze by language and aspect type
    lang_stats = defaultdict(lambda: {'total': 0, 'detected': 0})
    aspect_stats = defaultdict(lambda: {'total': 0, 'detected': 0})
    
    for i, result in enumerate(results):
        lang = result.language.value
        expected_aspect = test_cases[i]['expected_aspect'].value
        
        lang_stats[lang]['total'] += 1
        aspect_stats[expected_aspect]['total'] += 1
        
        if result.detected_aspects:
            lang_stats[lang]['detected'] += 1
            if result.detected_aspects[0].aspect_type.value == expected_aspect:
                aspect_stats[expected_aspect]['detected'] += 1
    
    print(f"\nLanguage Statistics:")
    for lang, stats in lang_stats.items():
        detection_rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  {lang.upper()}: {stats['total']} tests, {detection_rate:.1%} detection")
    
    print(f"\nAspect Type Statistics:")
    for aspect, stats in aspect_stats.items():
        detection_rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  {aspect}: {stats['total']} tests, {detection_rate:.1%} accuracy")
    
    # Check acceptance criteria
    print(f"\nAcceptance Criteria Check:")
    target_accuracy = 0.9  # 90% target
    print(f"  Target Accuracy: ≥{target_accuracy:.1%}")
    print(f"  Achieved Accuracy: {accuracy:.1%}")
    
    if accuracy >= target_accuracy:
        print("  ✅ ACCEPTANCE CRITERIA MET")
    else:
        print("  ❌ ACCEPTANCE CRITERIA NOT MET")
    
    # Save results
    output_path = Path("data/aspect_mapper_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        json_results = convert_numpy_types({
            'test_cases': [
                {
                    'input': tc['text'],
                    'language': tc['language'].value,
                    'expected_aspect': tc['expected_aspect'].value,
                    'result': results[i].to_dict()
                }
                for i, tc in enumerate(test_cases)
            ],
            'summary': {
                'total_tests': total_tests,
                'correct_detections': correct_detections,
                'accuracy': accuracy,
                'acceptance_criteria_met': accuracy >= target_accuracy,
                'language_stats': dict(lang_stats),
                'aspect_stats': dict(aspect_stats)
            }
        })
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info(f"Aspect mapper results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Aspect mapper demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()
