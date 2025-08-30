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
from .timing import timed_method, get_metrics

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
    
    @timed_method("generate", "semantic")
    def generate(self, doc, lang: str, *, mwe_tags=None, srl=None, umr_graph=None, mwe_spans_by_token=None) -> EILGraph:
        """
        Generate EIL graph from UD + MWE + SRL.
        This is the ONLY place allowed to emit NSM primes.
        """
        graph = EILGraph()
        
        # logger.info(f"Starting semantic generation for language: {lang}")
        
        # Generate primes from spatial relations
        self._spatial_from_ud(doc, lang, mwe_tags, graph, mwe_spans_by_token)
        
        # Generate primes from semantic roles
        if srl:
            self._primes_from_srl(srl, graph)
        
        # Generate primes from morphological features
        self._primes_from_morphology(doc, graph)
        
        # Generate primes from negation
        self._primes_from_negation(doc, graph)
        
        # Generate primes from residence verbs (LIVE)
        self._primes_from_residence(doc, lang, graph)
        
        # Generate primes from truth predicates (FALSE/TRUE)
        self._primes_from_truth(doc, lang, graph)
        
        # Generate primes from quantifiers (NOT, MORE, HALF, PEOPLE, MANY)
        self._primes_from_quantifiers(doc, lang, graph)
        
        # Generate primes from mental predicates (THINK, PEOPLE, THIS, VERY, GOOD)
        self._primes_from_mental(doc, lang, graph)
        
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
    
    @timed_method("primes_from_umr", "semantic")
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
    
    @timed_method("spatial_from_ud", "semantic")
    def _spatial_from_ud(self, doc, lang: str, mwe_tags, graph: EILGraph, mwe_spans_by_token=None):
        """Generate spatial primes from UD analysis with bulletproof MWE normalization."""
        # PHASE 5: Ensure MWE processing happens before UD analysis
        # Use doc._.mwe_spans_by_token for full span access to ADP chains
        
        # Get MWE spans by token if available
        if mwe_spans_by_token is None:
            mwe_spans_by_token = {}
        
        # First, try to detect multi-word spatial expressions using MWE spans
        text = " ".join([t.text for t in doc]).lower()
        
        # Check for multi-word spatial expressions (MWE normalization)
        if lang in self.spatial_maps:
            for prime, patterns in self.spatial_maps[lang].items():
                for pattern in patterns:
                    # Handle hyphenated compounds like "au-dessus de"
                    normalized_text = text.replace(" - ", "").replace("-", "").replace(" ", "")
                    normalized_pattern = pattern.replace(" ", "")
                    
                    # Check for exact pattern match
                    if pattern in text:
                        if self._is_spatial_licensed(pattern, lang, prime, doc):
                            graph.add_prime(prime, confidence=0.9)
                            return  # Only add one spatial prime per sentence
                    
                    # Check for normalized pattern match
                    if normalized_pattern in normalized_text:
                        if self._is_spatial_licensed(pattern, lang, prime, doc):
                            graph.add_prime(prime, confidence=0.9)
                            return  # Only add one spatial prime per sentence
                    
                    # Check for token sequence match (for hyphenated compounds)
                    tokens = [t.text.lower() for t in doc if t.pos_ != "PUNCT"]
                    pattern_tokens = pattern.split()
                    for i in range(len(tokens) - len(pattern_tokens) + 1):
                        if tokens[i:i+len(pattern_tokens)] == pattern_tokens:
                            if self._is_spatial_licensed(pattern, lang, prime, doc):
                                graph.add_prime(prime, confidence=0.9)
                                return  # Only add one spatial prime per sentence
        
        # Fallback to single token detection with enhanced MWE span access
        for token in doc:
            if token.pos_ == "ADP":
                # Get MWE span if available using doc._.mwe_spans_by_token
                span = None
                if token.i in mwe_spans_by_token:
                    # Get the first MWE span for this token
                    span = mwe_spans_by_token[token.i][0] if mwe_spans_by_token[token.i] else None
                
                # Also check mwe_tags for backward compatibility
                if not span and mwe_tags:
                    for mwe in mwe_tags:
                        if hasattr(mwe, 'start') and hasattr(mwe, 'end'):
                            if mwe.start <= token.i < mwe.end:
                                span = mwe
                                break
                
                if span:
                    # Use MWE text for surface form
                    surface = span.text.lower()
                else:
                    # Use token lemma
                    surface = token.lemma_.lower()
                
                # Look up spatial relation
                spatial_label = self._lookup_spatial(lang, surface)
                if not spatial_label:
                    continue
                
                # Apply enhanced topical/figurative guards
                if not self._is_spatial_licensed(surface, lang, spatial_label, doc):
                    continue
                
                # Add spatial prime
                graph.add_prime(spatial_label, confidence=0.9)
    
    def _is_spatial_licensed(self, text: str, lang: str, relation: str, doc) -> bool:
        """Check if spatial expression is properly licensed (not topical/figurative) - enhanced guards."""
        # Enhanced topical/figurative guards for bulletproof MWE normalization
        topical_patterns = {
            "fr": {
                "sur": ["sujet", "thème", "question", "problème", "point", "aspect", "plan"],  # Topic markers
                "dans": ["équipe", "groupe", "organisation", "société", "entreprise", "institution", "association"]  # Social/organizational
            },
            "es": {
                "sobre": ["tema", "asunto", "sujeto", "problema", "punto", "aspecto", "plan"],  # Topic markers
                "en": ["equipo", "grupo", "organización", "sociedad", "empresa", "institución", "asociación"]  # Social/organizational
            },
            "en": {
                "on": ["topic", "subject", "matter", "issue", "point", "aspect", "plan"],  # Topic markers
                "in": ["team", "group", "organization", "society", "company", "institution", "association"]  # Social/organizational
            },
            "de": {
                "über": ["Thema", "Gegenstand", "Frage", "Problem", "Punkt", "Aspekt", "Plan"],  # Topic markers
                "in": ["Team", "Gruppe", "Organisation", "Gesellschaft", "Unternehmen", "Institution", "Verein"]  # Social/organizational
            }
        }
        
        # Check if this is a topical/figurative use
        if lang in topical_patterns:
            for prep, topics in topical_patterns[lang].items():
                if text.lower() == prep.lower():
                    # Check if followed by topical words
                    # Look for topical words in the sentence
                    sentence_text = " ".join([t.text.lower() for t in doc])
                    
                    # Check for topical words in the context
                    for topic in topics:
                        if topic.lower() in sentence_text:
                            # Additional check: is this actually a spatial use?
                            # Look for spatial context (location words, spatial verbs)
                            spatial_indicators = {
                                "fr": ["lieu", "endroit", "position", "place", "localisation"],
                                "es": ["lugar", "sitio", "posición", "ubicación"],
                                "en": ["place", "location", "position", "area", "spot"],
                                "de": ["Ort", "Platz", "Position", "Lage", "Stelle"]
                            }
                            
                            spatial_words = spatial_indicators.get(lang, [])
                            has_spatial_context = any(word in sentence_text for word in spatial_words)
                            
                            # If no spatial context, likely topical/figurative
                            if not has_spatial_context:
                                return False
        
        # Additional guards for metaphorical uses
        metaphorical_patterns = {
            "fr": {
                "sur": ["sur internet", "sur le web", "sur la télévision"],  # Media/technology
                "dans": ["dans le temps", "dans l'histoire", "dans la vie"]  # Abstract domains
            },
            "es": {
                "sobre": ["sobre internet", "sobre la web", "sobre la televisión"],  # Media/technology
                "en": ["en el tiempo", "en la historia", "en la vida"]  # Abstract domains
            },
            "en": {
                "on": ["on the internet", "on the web", "on television"],  # Media/technology
                "in": ["in time", "in history", "in life"]  # Abstract domains
            },
            "de": {
                "über": ["über das Internet", "über das Web", "über das Fernsehen"],  # Media/technology
                "in": ["in der Zeit", "in der Geschichte", "im Leben"]  # Abstract domains
            }
        }
        
        if lang in metaphorical_patterns:
            for prep, metaphors in metaphorical_patterns[lang].items():
                if text.lower() == prep.lower():
                    sentence_text = " ".join([t.text.lower() for t in doc])
                    for metaphor in metaphors:
                        if metaphor.lower() in sentence_text:
                            return False
        
        return True
    
    def _lookup_spatial(self, lang: str, surface: str) -> Optional[str]:
        """Look up spatial relation from data table."""
        if lang in self.spatial_maps:
            for prime, patterns in self.spatial_maps[lang].items():
                if surface in patterns:
                    return prime
        return None
    
    @timed_method("primes_from_srl", "semantic")
    def _primes_from_srl(self, srl, graph: EILGraph):
        """Generate primes from semantic role labeling."""
        # Only emit primes for explicit semantic roles, not generic mappings
        # These should be represented as nodes with binders, not global primes
        
        for role in srl:
            # Handle both object and dictionary formats
            role_name = role.role if hasattr(role, 'role') else role.get('role')
            role_text = role.text if hasattr(role, 'text') else role.get('text')
            role_confidence = role.confidence if hasattr(role, 'confidence') else role.get('confidence', 0.5)
            
            # Only emit primes for truly semantic roles that have explicit linguistic markers
            if role_name in ['AGENT'] and role_confidence > 0.8:
                # AGENT can be SOMEONE if it's a clear agentive subject
                graph.add_prime('SOMEONE', arguments=[role_text], confidence=role_confidence)
                # Add edge for licensing
                graph.edges.append({
                    'type': 'agent',
                    'label': 'AGENT',
                    'source': role_text,
                    'target': 'action'
                })
            elif role_name in ['PATIENT', 'THEME'] and role_confidence >= 0.79:
                # PATIENT/THEME can be THING
                graph.add_prime('THING', arguments=[role_text], confidence=role_confidence)
                # Add edge for licensing
                graph.edges.append({
                    'type': 'patient',
                    'label': 'PATIENT',
                    'source': 'action',
                    'target': role_text
                })
            # Other roles become nodes with binders, not global primes
    
    @timed_method("primes_from_morphology", "semantic")
    def _primes_from_morphology(self, doc, graph: EILGraph):
        """Generate primes from morphological features."""
        for token in doc:
            # DO (verbs)
            if token.pos_ == "VERB" and token.dep_ != "aux":
                graph.add_prime("DO", confidence=0.8)
            
            # ONE (number)
            if (token.pos_ in {"DET", "NUM"} and 
                token.lemma_.lower() in {"one", "un", "una", "uno"}):
                
                # Guard pronoun 'one' (EN) or 'on' (FR)
                if token.lemma_.lower() not in {"on"}:  # Allow "one" as determiner
                    graph.add_prime("ONE", confidence=0.9)
            
            # WORDS
            if (token.lemma_.lower() in {"word", "words", "mot", "mots", "palabra", "palabras"} and
                self._has_nearby_verb(doc, token, {"say", "dire", "decir", "write", "écrire", "escribir", "read", "lire", "leer"})):
                
                # Guard idioms
                if not self._is_speech_act_idiom(doc, token):
                    graph.add_prime("WORDS", confidence=0.9)
    
    def _has_nearby_verb(self, doc, token, verb_lemmas):
        """Check if token has nearby verb from the specified set."""
        for other_token in doc:
            if (other_token.pos_ == "VERB" and 
                other_token.lemma_.lower() in verb_lemmas and
                abs(other_token.i - token.i) <= 3):
                return True
        return False
    
    def _is_speech_act_idiom(self, doc, token):
        """Check if token is part of a speech act idiom."""
        # Simple check for common idioms
        text = " ".join([t.text for t in doc]).lower()
        idioms = ["tenir parole", "dar la palabra", "keep word"]
        return any(idiom in text for idiom in idioms)
    
    @timed_method("primes_from_negation", "semantic")
    def _primes_from_negation(self, doc, graph: EILGraph):
        """Generate primes from negation patterns."""
        for token in doc:
            if token.lemma_.lower() in ['not', 'no', 'ne', 'pas', 'nicht', 'non']:
                graph.add_prime("NOT", confidence=0.9)
    
    @timed_method("primes_from_residence", "semantic")
    def _primes_from_residence(self, doc, lang: str, graph: EILGraph):
        """Generate LIVE prime for residence verbs."""
        residence_verbs = {
            'en': ['live', 'reside', 'dwell'],
            'es': ['vivir', 'residir'],
            'fr': ['habiter', 'résider'],
            'de': ['wohnen', 'leben']
        }
        
        lang_verbs = residence_verbs.get(lang, [])
        for token in doc:
            if (token.pos_ == 'VERB' and 
                token.lemma_.lower() in lang_verbs and
                not self._is_metaphorical_residence(token)):
                graph.add_prime("LIVE", confidence=0.9)
                break
    
    def _is_metaphorical_residence(self, token) -> bool:
        """Check if residence verb is metaphorical (live stream, live wire, etc.)"""
        # Check for compound nouns or adjectives that indicate metaphorical use
        for child in token.children:
            if (child.pos_ in ['NOUN', 'ADJ'] and 
                child.dep_ in ['compound', 'amod']):
                return True
        return False
    
    @timed_method("primes_from_truth", "semantic")
    def _primes_from_truth(self, doc, lang: str, graph: EILGraph):
        """Generate FALSE/TRUE primes from copular constructions."""
        false_predicates = {
            'en': ['false', 'wrong', 'incorrect'],
            'es': ['falso', 'incorrecto', 'erróneo'],
            'fr': ['faux', 'incorrect', 'erroné'],
            'de': ['falsch', 'unrichtig', 'inkorrekt']
        }
        
        true_predicates = {
            'en': ['true', 'correct', 'right'],
            'es': ['verdadero', 'correcto', 'cierto'],
            'fr': ['vrai', 'correct', 'juste'],
            'de': ['wahr', 'richtig', 'korrekt']
        }
        
        lang_false = false_predicates.get(lang, [])
        lang_true = true_predicates.get(lang, [])
        
        # Check for "es falso que" pattern (Spanish)
        if lang == 'es':
            text = " ".join([t.text.lower() for t in doc])
            if "es falso que" in text:
                graph.add_prime("FALSE", confidence=0.9)
                return
        
        # Check for "it is false that" pattern (English)
        if lang == 'en':
            text = " ".join([t.text.lower() for t in doc])
            if "it is false that" in text or "it's false that" in text:
                graph.add_prime("FALSE", confidence=0.9)
                return
        
        # Check for "c'est faux que" pattern (French)
        if lang == 'fr':
            text = " ".join([t.text.lower() for t in doc])
            if "c'est faux que" in text:
                graph.add_prime("FALSE", confidence=0.9)
                return
        
        # Check for "es ist falsch dass" pattern (German)
        if lang == 'de':
            text = " ".join([t.text.lower() for t in doc])
            if "es ist falsch dass" in text:
                graph.add_prime("FALSE", confidence=0.9)
                return
        
        # Fallback to copular construction detection
        for token in doc:
            if (token.pos_ == 'VERB' and 
                token.lemma_.lower() in ['be', 'ser', 'être', 'sein']):
                # Look for predicate adjectives
                for child in token.children:
                    if (child.pos_ == 'ADJ' and 
                        child.dep_ in ['attr', 'pred']):
                        if child.lemma_.lower() in lang_false:
                            graph.add_prime("FALSE", confidence=0.9)
                            return
                        elif child.lemma_.lower() in lang_true:
                            graph.add_prime("TRUE", confidence=0.9)
                            return
    
    @timed_method("primes_from_quantifiers", "semantic")
    def _primes_from_quantifiers(self, doc, lang: str, graph: EILGraph):
        """Generate quantifier primes (NOT, MORE, HALF, PEOPLE, MANY)."""
        # Check for "at most" pattern
        for i, token in enumerate(doc):
            if (token.lemma_.lower() == 'at' and 
                i + 1 < len(doc) and 
                doc[i + 1].lemma_.lower() == 'most'):
                graph.add_prime("NOT", confidence=0.9)
                graph.add_prime("MORE", confidence=0.9)
                break
        
        # Check for "half" pattern
        for token in doc:
            if token.lemma_.lower() == 'half':
                graph.add_prime("HALF", confidence=0.9)
                break
        
        # Check for "a lot" pattern
        for i, token in enumerate(doc):
            if (token.lemma_.lower() == 'a' and 
                i + 1 < len(doc) and 
                doc[i + 1].lemma_.lower() == 'lot'):
                graph.add_prime("MANY", confidence=0.9)
                break
        
        # Check for plural nouns (PEOPLE)
        for token in doc:
            if (token.pos_ == 'NOUN' and 
                token.morph.get('Number') == 'Plur' and
                token.lemma_.lower() in ['student', 'person', 'people', 'child']):
                graph.add_prime("PEOPLE", confidence=0.8)
                break
    
    @timed_method("primes_from_mental", "semantic")
    def _primes_from_mental(self, doc, lang: str, graph: EILGraph):
        """Generate mental predicate primes (THINK, PEOPLE, THIS, VERY, GOOD)."""
        # Mental predicates
        think_verbs = {
            'en': ['think', 'believe', 'consider'],
            'es': ['pensar', 'creer', 'considerar'],
            'fr': ['penser', 'croire', 'considérer'],
            'de': ['denken', 'glauben', 'meinen']
        }
        
        # Evaluators
        good_words = {
            'en': ['good', 'great', 'excellent', 'nice'],
            'es': ['bueno', 'excelente', 'genial'],
            'fr': ['bon', 'excellent', 'génial'],
            'de': ['gut', 'ausgezeichnet', 'toll']
        }
        
        # Intensifiers
        very_words = {
            'en': ['very', 'really', 'quite'],
            'es': ['muy', 'realmente', 'bastante'],
            'fr': ['très', 'vraiment', 'assez'],
            'de': ['sehr', 'wirklich', 'ziemlich']
        }
        
        # Deictics
        this_words = {
            'en': ['this', 'that'],
            'es': ['esto', 'eso', 'este', 'ese'],
            'fr': ['ce', 'cette', 'ça'],
            'de': ['das', 'dies', 'jenes']
        }
        
        lang_think = think_verbs.get(lang, [])
        lang_good = good_words.get(lang, [])
        lang_very = very_words.get(lang, [])
        lang_this = this_words.get(lang, [])
        
        # Check for mental predicates
        for token in doc:
            if (token.pos_ == 'VERB' and 
                token.lemma_.lower() in lang_think):
                graph.add_prime("THINK", confidence=0.9)
                break
        
        # Check for evaluators
        for token in doc:
            if (token.pos_ == 'ADJ' and 
                token.lemma_.lower() in lang_good):
                graph.add_prime("GOOD", confidence=0.9)
                break
        
        # Check for intensifiers
        for token in doc:
            if (token.pos_ == 'ADV' and 
                token.lemma_.lower() in lang_very):
                graph.add_prime("VERY", confidence=0.9)
                break
        
        # Check for deictics
        for token in doc:
            if (token.pos_ in ['DET', 'PRON'] and 
                token.lemma_.lower() in lang_this):
                graph.add_prime("THIS", confidence=0.8)
                break
        
        # Check for plural nouns (PEOPLE)
        for token in doc:
            if (token.pos_ == 'NOUN' and 
                token.morph.get('Number') == 'Plur' and
                token.lemma_.lower() in ['gens', 'person', 'people']):
                graph.add_prime("PEOPLE", confidence=0.8)
                break
    
    @timed_method("binders_from_open_class", "semantic")
    def _binders_from_open_class(self, doc, graph: EILGraph):
        """Generate binders for open-class words (not primes)."""
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                # These become binders, not primes
                features = []
                if token.pos_ == "PROPN":
                    features.append("NAMED")
                
                graph.add_binder(
                    surface=token.text,
                    features=features,
                    prime_type="THING" if token.pos_ == "NOUN" else "PERSON"
                )
