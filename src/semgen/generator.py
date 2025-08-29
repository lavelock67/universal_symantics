"""
Semantic Generator
The ONLY place allowed to emit NSM primes. Uses UD + MWE + SRL to generate proper NSM primes.
"""

from pathlib import Path
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from ..eil.primes_registry import assert_only_allowed, ALLOWED_PRIMES
from .prime_sparsifier import PrimeSparsifier, PrimeSparsifierConfig
from .adapters.umr_adapter import UMRAdapter, SemFeatures

logger = logging.getLogger(__name__)

@dataclass
class EILGraph:
    """EIL (Elementary Interlingua) Graph representation"""
    nodes: List[Dict[str, Any]] = None
    edges: List[Dict[str, Any]] = None
    binders: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []
        if self.edges is None:
            self.edges = []
        if self.binders is None:
            self.binders = []
    
    def add_prime(self, prime: str, arguments: List[str] = None, confidence: float = 1.0):
        """Add an NSM prime to the graph."""
        assert_only_allowed([prime])
        
        node = {
            'type': 'prime',
            'prime': prime,
            'arguments': arguments or [],
            'confidence': confidence
        }
        self.nodes.append(node)
    
    def add_molecule(self, molecule: str, arguments: List[str] = None, confidence: float = 1.0):
        """Add an NSM molecule to the graph."""
        node = {
            'type': 'molecule',
            'molecule': molecule,
            'arguments': arguments or [],
            'confidence': confidence
        }
        self.nodes.append(node)
    
    def add_binder(self, surface: str, features: List[str] = None, prime_type: str = None):
        """Add a binder for open-class words."""
        binder = {
            'surface': surface,
            'features': features or [],
            'prime_type': prime_type
        }
        self.binders.append(binder)
    
    def get_primes(self) -> List[str]:
        """Get all primes from the graph."""
        primes = []
        for node in self.nodes:
            if node['type'] == 'prime':
                primes.append(node['prime'])
        return primes
    
    def remove_primes(self, primes_to_remove: set):
        """Remove specified primes from the graph."""
        self.nodes = [node for node in self.nodes 
                     if not (node['type'] == 'prime' and node['prime'] in primes_to_remove)]
    
    def get_edges(self) -> List[Dict[str, Any]]:
        """Get all edges from the graph."""
        return self.edges
    
    def get_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes from the graph."""
        return self.nodes

class SemanticGenerator:
    """Semantic Generator - the ONLY place allowed to emit NSM primes"""
    
    def __init__(self):
        self.spatial_maps = self._load_spatial_maps()
        self.mwe_patterns = self._load_mwe_patterns()
        self.umr_adapter = UMRAdapter()
        
    def _load_spatial_maps(self) -> Dict[str, Dict[str, List[str]]]:
        """Load spatial relation mappings from data file."""
        spatial_file = Path("assets/spatial_maps.json")
        if spatial_file.exists():
            with open(spatial_file, 'r') as f:
                return json.load(f)
        else:
            # Default spatial mappings
            return {
                "es": {
                    "NEAR": ["cerca de"],
                    "INSIDE": ["dentro de", "en el interior de"],
                    "ABOVE": ["encima de", "sobre"]
                },
                "fr": {
                    "NEAR": ["près de"],
                    "INSIDE": ["dans", "à l'intérieur de"],
                    "ABOVE": ["au-dessus de", "sur"]
                },
                "en": {
                    "NEAR": ["near"],
                    "INSIDE": ["inside", "inside of"],
                    "ABOVE": ["above", "over", "on top of"]
                }
            }
    
    def _load_mwe_patterns(self) -> Dict[str, List[str]]:
        """Load MWE patterns for quantifiers, intensifiers, negation."""
        return {
            "quantifiers": ["much", "many", "some", "all", "half"],
            "intensifiers": ["very", "more", "most"],
            "negation": ["not", "no", "ne", "pas", "nicht", "non"]
        }
    
    def generate(self, doc, lang: str, *, mwe_tags=None, srl=None, umr_graph=None) -> EILGraph:
        """
        Generate EIL graph from UD + MWE + SRL.
        This is the ONLY place allowed to emit NSM primes.
        """
        graph = EILGraph()
        
        # logger.info(f"Starting semantic generation for language: {lang}")
        
        # Generate primes from spatial relations
        self._spatial_from_ud(doc, lang, mwe_tags, graph)
        
        # Generate primes from semantic roles
        if srl:
            self._primes_from_srl(srl, graph)
        
        # Generate primes from morphological features
        self._primes_from_morphology(doc, graph)
        
        # Generate primes from negation
        self._primes_from_negation(doc, graph)
        
        # Generate primes from UMR features (if available)
        if umr_graph:
            self._primes_from_umr(umr_graph, lang, graph)
        
        # Generate binders for open-class words
        self._binders_from_open_class(doc, graph)
        
        # Sparsify primes to remove unlicensed global fillers
        sparsifier = PrimeSparsifier()
        graph = sparsifier.sparsify(graph)
        
        # Validate that only allowed primes are present
        primes = graph.get_primes()
        assert_only_allowed(primes)
        
        # Additional validation: no pseudo-primes
        pseudo_primes = {'BOY', 'HOUSE', 'BALL', 'KICK', 'THE', 'A', 'AN'}
        found_pseudo = [p for p in primes if p in pseudo_primes]
        if found_pseudo:
            raise ValueError(f"Pseudo-primes detected: {found_pseudo}. These should be binders, not primes.")
        
        logger.info(f"Generated {len(primes)} NSM primes: {primes}")
        return graph
    
    def _primes_from_umr(self, umr_graph, lang: str, graph: EILGraph):
        """Generate primes from UMR features."""
        # Convert UMR graph to semantic features
        umr_features = self.umr_adapter.umr_to_features(umr_graph, lang)
        
        # Extract events and add DO/HAPPEN primes
        for event in umr_features.events:
            if event["type"] == "agentive":
                graph.add_prime("DO", arguments=[event["label"]], confidence=event["confidence"])
            else:
                graph.add_prime("HAPPEN", arguments=[event["label"]], confidence=event["confidence"])
        
        # Extract arguments and add role-based primes
        for arg in umr_features.args:
            if arg["role"] == "AGENT":
                graph.add_prime("SOMEONE", arguments=[arg["argument"]], confidence=arg["confidence"])
            elif arg["role"] == "PATIENT":
                graph.add_prime("THING", arguments=[arg["argument"]], confidence=arg["confidence"])
        
        # Extract quantifiers
        for quant in umr_features.quantifiers:
            quant_prime = quant["quantifier"]
            if quant_prime in ["ALL", "SOME", "ONE", "MANY", "FEW", "HALF"]:
                graph.add_prime(quant_prime, confidence=quant["confidence"])
        
        # Extract negation
        for neg in umr_features.negations:
            graph.add_prime("NOT", confidence=neg["confidence"])
        
        # Extract modality (only CAN is in official NSM primes)
        for mod in umr_features.modality:
            mod_prime = mod["modality"]
            if mod_prime == "CAN":  # Only CAN is in official NSM primes
                graph.add_prime(mod_prime, confidence=mod["confidence"])
        
        # Extract spatial relations
        for spatial in umr_features.spatial_roles:
            spatial_prime = spatial["relation"]
            if spatial_prime in ["NEAR", "INSIDE", "ABOVE"]:
                graph.add_prime(spatial_prime, confidence=spatial["confidence"])
        
        # Extract temporal expressions (only if licensed)
        for time_expr in umr_features.times:
            # Only add TIME if it's properly licensed by temporal structure
            if self._is_temporal_licensed(time_expr):
                graph.add_prime("TIME", arguments=[time_expr["time_expression"]], confidence=time_expr["confidence"])
                # Add a temporal node to license the TIME prime
                graph.nodes.append({
                    "id": f"time_{time_expr['time_expression']}",
                    "type": "temporal",
                    "label": time_expr["time_expression"],
                    "features": ["time"]
                })
    
    def _is_temporal_licensed(self, time_expr: Dict[str, Any]) -> bool:
        """Check if temporal expression is properly licensed."""
        # Only license TIME for explicit temporal expressions
        time_words = {"today", "yesterday", "tomorrow", "now", "then", "when", "before", "after"}
        return any(word in time_expr["time_expression"].lower() for word in time_words)
    
    def _spatial_from_ud(self, doc, lang: str, mwe_tags, graph: EILGraph):
        """Generate spatial primes from UD analysis."""
        # First, try to detect multi-word spatial expressions
        text = " ".join([t.token for t in doc]).lower()
        # logger.info(f"Checking spatial patterns in text: '{text}'")
        
        # Check for multi-word spatial expressions
        if lang in self.spatial_maps:
            for prime, patterns in self.spatial_maps[lang].items():
                for pattern in patterns:
                    # Handle hyphenated compounds like "au-dessus de"
                    normalized_text = text.replace(" - ", "").replace("-", "").replace(" ", "")
                    normalized_pattern = pattern.replace(" ", "")
                    
                    # logger.info(f"Checking pattern '{pattern}' -> normalized '{normalized_pattern}' against text '{normalized_text}'")
                    
                    # Check for exact pattern match
                    if pattern in text:
                        # logger.info(f"Found spatial prime (exact): {prime}")
                        graph.add_prime(prime, confidence=0.9)
                        return  # Only add one spatial prime per sentence
                    
                    # Check for normalized pattern match
                    if normalized_pattern in normalized_text:
                        # logger.info(f"Found spatial prime (normalized): {prime}")
                        graph.add_prime(prime, confidence=0.9)
                        return  # Only add one spatial prime per sentence
                    
                    # Check for token sequence match (for hyphenated compounds)
                    tokens = [t.token.lower() for t in doc if t.pos != "PUNCT"]
                    pattern_tokens = pattern.split()
                    # logger.info(f"Checking token sequence: tokens={tokens}, pattern_tokens={pattern_tokens}")
                    for i in range(len(tokens) - len(pattern_tokens) + 1):
                        if tokens[i:i+len(pattern_tokens)] == pattern_tokens:
                            # logger.info(f"Found spatial prime (token sequence): {prime}")
                            graph.add_prime(prime, confidence=0.9)
                            return  # Only add one spatial prime per sentence
        
        # Fallback to single token detection
        for token in doc:
            if token.pos == "ADP":
                # Get MWE span if available
                span = None
                if mwe_tags:
                    span = mwe_tags.get_span(token)
                
                surface = (span or token).lemma.lower()
                
                # Look up spatial relation
                spatial_label = self._lookup_spatial(lang, surface)
                if not spatial_label:
                    continue
                
                # Guards: topical senses
                # Find the head token by index
                head_token = None
                for t in doc:
                    if t.index == token.head:
                        head_token = t
                        break
                
                if head_token and head_token.lemma.lower() in {"sujet", "thème", "tema", "asunto"}:
                    continue  # Skip topical 'sur/sobre'
                
                # Add spatial prime
                graph.add_prime(spatial_label, confidence=0.9)
    
    def _lookup_spatial(self, lang: str, surface: str) -> Optional[str]:
        """Look up spatial relation from data table."""
        if lang in self.spatial_maps:
            for prime, patterns in self.spatial_maps[lang].items():
                if surface in patterns:
                    return prime
        return None
    
    def _primes_from_srl(self, srl, graph: EILGraph):
        """Generate primes from semantic role labeling."""
        # Only emit primes for explicit semantic roles, not generic mappings
        # These should be represented as nodes with binders, not global primes
        
        for role in srl:
            # Only emit primes for truly semantic roles that have explicit linguistic markers
            if role.role in ['AGENT'] and role.confidence > 0.8:
                # AGENT can be SOMEONE if it's a clear agentive subject
                graph.add_prime('SOMEONE', arguments=[role.text], confidence=role.confidence)
                # Add edge for licensing
                graph.edges.append({
                    'type': 'agent',
                    'label': 'AGENT',
                    'source': role.text,
                    'target': 'action'
                })
            elif role.role in ['PATIENT', 'THEME'] and role.confidence >= 0.79:
                # PATIENT/THEME can be THING
                graph.add_prime('THING', arguments=[role.text], confidence=role.confidence)
                # Add edge for licensing
                graph.edges.append({
                    'type': 'patient',
                    'label': 'PATIENT',
                    'source': 'action',
                    'target': role.text
                })
            # Other roles become nodes with binders, not global primes
    
    def _primes_from_morphology(self, doc, graph: EILGraph):
        """Generate primes from morphological features."""
        for token in doc:
            # DO (verbs)
            if token.pos == "VERB" and token.dep != "aux":
                graph.add_prime("DO", confidence=0.8)
            
            # ONE (number)
            if (token.pos in {"DET", "NUM"} and 
                token.lemma.lower() in {"one", "un", "una", "uno"}):
                
                # Guard pronoun 'one' (EN) or 'on' (FR)
                if token.lemma.lower() not in {"on"}:  # Allow "one" as determiner
                    graph.add_prime("ONE", confidence=0.9)
            
            # WORDS
            if (token.lemma.lower() in {"word", "words", "mot", "mots", "palabra", "palabras"} and
                self._has_nearby_verb(doc, token, {"say", "dire", "decir", "write", "écrire", "escribir", "read", "lire", "leer"})):
                
                # Guard idioms
                if not self._is_speech_act_idiom(doc, token):
                    graph.add_prime("WORDS", confidence=0.9)
    
    def _has_nearby_verb(self, doc, token, verb_lemmas):
        """Check if token has nearby verb from the specified set."""
        for other_token in doc:
            if (other_token.pos == "VERB" and 
                other_token.lemma.lower() in verb_lemmas and
                abs(other_token.index - token.index) <= 3):
                return True
        return False
    
    def _is_speech_act_idiom(self, doc, token):
        """Check if token is part of a speech act idiom."""
        # Simple check for common idioms
        text = " ".join([t.token for t in doc]).lower()
        idioms = ["tenir parole", "dar la palabra", "keep word"]
        return any(idiom in text for idiom in idioms)
    
    def _primes_from_negation(self, doc, graph: EILGraph):
        """Generate primes from negation markers."""
        negation_markers = {"not", "no", "ne", "pas", "non"}
        
        for token in doc:
            if token.lemma.lower() in negation_markers:
                graph.add_prime("NOT", confidence=0.9)
                break
    
    def _binders_from_open_class(self, doc, graph: EILGraph):
        """Generate binders for open-class words (not primes)."""
        for token in doc:
            if token.pos in {"NOUN", "PROPN"}:
                # These become binders, not primes
                features = []
                if token.pos == "PROPN":
                    features.append("NAMED")
                
                graph.add_binder(
                    surface=token.token,
                    features=features,
                    prime_type="THING" if token.pos == "NOUN" else "PERSON"
                )
