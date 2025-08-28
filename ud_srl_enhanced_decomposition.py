#!/usr/bin/env python3
"""
UD + SRL Enhanced Semantic Decomposition

This integrates Universal Dependencies (UD) and Semantic Role Labeling (SRL)
for more robust semantic decomposition and analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

import spacy
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class SemanticRole:
    """Represents a semantic role in a sentence."""
    role: str  # AGENT, PATIENT, THEME, GOAL, LOCATION, TIME, etc.
    text: str
    start: int
    end: int
    confidence: float
    entity_type: Optional[str] = None

@dataclass
class UDNode:
    """Represents a Universal Dependencies node."""
    token: str
    pos: str
    dep: str
    head: int
    index: int
    lemma: str
    features: Dict[str, str]

@dataclass
class EnhancedSentenceStructure:
    """Enhanced sentence structure with UD and SRL information."""
    # Basic structure
    subject: Optional[str]
    predicate: Optional[str]
    object: Optional[str]
    indirect_object: Optional[str]
    
    # UD information
    ud_tree: List[UDNode]
    dependency_paths: Dict[str, List[str]]
    
    # SRL information
    semantic_roles: List[SemanticRole]
    role_mappings: Dict[str, List[SemanticRole]]
    
    # Enhanced features
    voice: str  # "active", "passive", "middle"
    mood: str   # "declarative", "interrogative", "imperative"
    tense: str  # "present", "past", "future", "progressive"
    aspect: str # "simple", "progressive", "perfect"
    modality: Optional[str] = None
    
    # Additional semantic information
    temporal_expressions: List[str] = None
    spatial_expressions: List[str] = None
    causal_relations: List[Tuple[str, str]] = None
    negation: bool = False

class UDSRLEnhancedDecompositionEngine:
    """Enhanced semantic decomposition with UD and SRL integration."""
    
    def __init__(self):
        # Load SpaCy with UD support
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  SpaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # SRL patterns and rules
        self.srl_patterns = self._initialize_srl_patterns()
        
        # UD dependency patterns for semantic roles
        self.ud_role_patterns = self._initialize_ud_role_patterns()
        
        # Enhanced concept decompositions
        self.concept_decompositions = self._initialize_enhanced_decompositions()
    
    def _initialize_srl_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize SRL patterns for different semantic roles."""
        
        return {
            "AGENT": [
                {"dep": "nsubj", "pos": ["NOUN", "PROPN", "PRON"]},
                {"dep": "nsubjpass", "pos": ["NOUN", "PROPN", "PRON"]},
                {"dep": "agent", "pos": ["NOUN", "PROPN", "PRON"]}
            ],
            "PATIENT": [
                {"dep": "dobj", "pos": ["NOUN", "PROPN", "PRON"]},
                {"dep": "pobj", "pos": ["NOUN", "PROPN", "PRON"]},
                {"dep": "nsubjpass", "pos": ["NOUN", "PROPN", "PRON"]}
            ],
            "THEME": [
                {"dep": "dobj", "pos": ["NOUN", "PROPN", "PRON"]},
                {"dep": "pobj", "pos": ["NOUN", "PROPN", "PRON"]},
                {"dep": "attr", "pos": ["NOUN", "PROPN", "PRON"]}
            ],
            "GOAL": [
                {"dep": "pobj", "pos": ["NOUN", "PROPN", "PRON"]},
                {"dep": "dative", "pos": ["NOUN", "PROPN", "PRON"]},
                {"dep": "iobj", "pos": ["NOUN", "PROPN", "PRON"]}
            ],
            "LOCATION": [
                {"dep": "pobj", "pos": ["NOUN", "PROPN", "PRON"]},
                {"dep": "nmod", "pos": ["NOUN", "PROPN"]},
                {"dep": "advmod", "pos": ["ADV"]}
            ],
            "TIME": [
                {"dep": "nmod", "pos": ["NOUN", "PROPN"]},
                {"dep": "advmod", "pos": ["ADV"]},
                {"dep": "pobj", "pos": ["NOUN", "PROPN"]}
            ],
            "INSTRUMENT": [
                {"dep": "pobj", "pos": ["NOUN", "PROPN"]},
                {"dep": "nmod", "pos": ["NOUN", "PROPN"]},
                {"dep": "advmod", "pos": ["ADV"]}
            ],
            "MANNER": [
                {"dep": "advmod", "pos": ["ADV", "ADJ"]},
                {"dep": "amod", "pos": ["ADJ"]},
                {"dep": "nmod", "pos": ["NOUN"]}
            ]
        }
    
    def _initialize_ud_role_patterns(self) -> Dict[str, List[str]]:
        """Initialize UD dependency patterns for semantic role identification."""
        
        return {
            "subject_patterns": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
            "object_patterns": ["dobj", "iobj", "pobj", "attr"],
            "modifier_patterns": ["amod", "advmod", "nummod", "det"],
            "temporal_patterns": ["nmod:tmod", "advmod", "pobj"],
            "spatial_patterns": ["nmod:loc", "advmod", "pobj"],
            "causal_patterns": ["advcl", "mark", "conj"],
            "negation_patterns": ["neg", "advmod"],
            "auxiliary_patterns": ["aux", "auxpass", "cop"]
        }
    
    def _initialize_enhanced_decompositions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize enhanced concept decompositions with semantic roles."""
        
        return {
            # People and beings (with semantic role information)
            "boy": {
                "type": "person",
                "semantic_roles": ["AGENT", "EXPERIENCER", "POSSESSOR"],
                "decomposition": [
                    "someone of one kind",
                    "when someone is someone of this kind, this someone's body is small",
                    "this someone is not someone of another kind (girl)"
                ],
                "ud_features": {
                    "pos": "NOUN",
                    "gender": "masculine",
                    "number": "singular",
                    "animacy": "animate"
                }
            },
            "girl": {
                "type": "person",
                "semantic_roles": ["AGENT", "EXPERIENCER", "POSSESSOR"],
                "decomposition": [
                    "someone of one kind",
                    "when someone is someone of this kind, this someone's body is small",
                    "this someone is not someone of another kind (boy)"
                ],
                "ud_features": {
                    "pos": "NOUN",
                    "gender": "feminine",
                    "number": "singular",
                    "animacy": "animate"
                }
            },
            "cat": {
                "type": "animal",
                "semantic_roles": ["AGENT", "EXPERIENCER", "THEME"],
                "decomposition": [
                    "a living thing of one kind",
                    "this thing is not a person",
                    "this thing can move",
                    "this thing has four legs",
                    "this thing is small"
                ],
                "ud_features": {
                    "pos": "NOUN",
                    "number": "singular",
                    "animacy": "animate"
                }
            },
            
            # Objects and things
            "ball": {
                "type": "object",
                "semantic_roles": ["THEME", "PATIENT", "INSTRUMENT"],
                "decomposition": [
                    "a thing of one kind",
                    "when people see it, they think: it is small, it is round",
                    "this thing can move because of what people do to it"
                ],
                "ud_features": {
                    "pos": "NOUN",
                    "number": "singular",
                    "animacy": "inanimate"
                }
            },
            "mat": {
                "type": "object",
                "semantic_roles": ["LOCATION", "THEME"],
                "decomposition": [
                    "a thing of one kind",
                    "this thing is flat",
                    "this thing is on the ground",
                    "people can sit on this thing"
                ],
                "ud_features": {
                    "pos": "NOUN",
                    "number": "singular",
                    "animacy": "inanimate"
                }
            },
            
            # Actions with semantic role information
            "kick": {
                "type": "action",
                "semantic_roles": ["AGENT", "PATIENT", "INSTRUMENT"],
                "decomposition": [
                    "someone X did something to someone else Y",
                    "this someone's leg moved",
                    "this someone's leg touched the thing",
                    "because of this, the thing moved",
                    "it happened in one moment"
                ],
                "ud_features": {
                    "pos": "VERB",
                    "voice": "active",
                    "tense": "present"
                }
            },
            "kicked": {
                "type": "action",
                "semantic_roles": ["AGENT", "PATIENT", "INSTRUMENT"],
                "decomposition": [
                    "someone X did something to someone else Y",
                    "this someone's leg moved",
                    "this someone's leg touched the thing",
                    "because of this, the thing moved",
                    "it happened in one moment"
                ],
                "ud_features": {
                    "pos": "VERB",
                    "voice": "active",
                    "tense": "past"
                }
            },
            "sat": {
                "type": "action",
                "semantic_roles": ["AGENT", "LOCATION"],
                "decomposition": [
                    "someone did something",
                    "this someone was not moving",
                    "this someone was on something",
                    "this happened for some time"
                ],
                "ud_features": {
                    "pos": "VERB",
                    "voice": "active",
                    "tense": "past"
                }
            },
            "wrote": {
                "type": "action",
                "semantic_roles": ["AGENT", "THEME", "GOAL"],
                "decomposition": [
                    "someone did something",
                    "this someone made many words",
                    "because of this, other people can see these words",
                    "because of this, other people can know many things"
                ],
                "ud_features": {
                    "pos": "VERB",
                    "voice": "active",
                    "tense": "past"
                }
            }
        }
    
    def parse_ud_structure(self, text: str) -> List[UDNode]:
        """Parse Universal Dependencies structure."""
        
        doc = self.nlp(text)
        ud_nodes = []
        
        for token in doc:
            node = UDNode(
                token=token.text,
                pos=token.pos_,
                dep=token.dep_,
                head=token.head.i,
                index=token.i,
                lemma=token.lemma_,
                features={
                    "tag": token.tag_,
                    "morph": str(token.morph),
                    "is_sent_start": token.is_sent_start,
                    "is_sent_end": token.is_sent_end
                }
            )
            ud_nodes.append(node)
        
        return ud_nodes
    
    def extract_semantic_roles(self, ud_nodes: List[UDNode]) -> List[SemanticRole]:
        """Extract semantic roles using UD patterns and SRL rules."""
        
        semantic_roles = []
        
        for node in ud_nodes:
            # Check each SRL pattern
            for role, patterns in self.srl_patterns.items():
                for pattern in patterns:
                    if (node.dep == pattern["dep"] and 
                        node.pos in pattern["pos"]):
                        
                        # Calculate confidence based on pattern match and context
                        confidence = self._calculate_role_confidence(node, role, ud_nodes)
                        
                        semantic_role = SemanticRole(
                            role=role,
                            text=node.token,
                            start=node.index,
                            end=node.index + 1,
                            confidence=confidence,
                            entity_type=self._determine_entity_type(node, ud_nodes)
                        )
                        semantic_roles.append(semantic_role)
                        break
        
        return semantic_roles
    
    def _calculate_role_confidence(self, node: UDNode, role: str, all_nodes: List[UDNode]) -> float:
        """Calculate confidence score for a semantic role assignment."""
        
        base_confidence = 0.7
        
        # Higher confidence for clear role indicators
        if role == "AGENT" and node.dep in ["nsubj", "nsubjpass"]:
            base_confidence += 0.2
        elif role == "PATIENT" and node.dep in ["dobj", "nsubjpass"]:
            base_confidence += 0.2
        elif role == "LOCATION" and any(word in node.token.lower() for word in ["in", "at", "on", "near"]):
            base_confidence += 0.1
        elif role == "TIME" and any(word in node.token.lower() for word in ["when", "time", "year", "date"]):
            base_confidence += 0.1
        
        # Higher confidence for longer, more specific entities
        if len(node.token) > 3:
            base_confidence += 0.1
        
        # Higher confidence for proper nouns in certain roles
        if node.pos == "PROPN" and role in ["AGENT", "LOCATION", "THEME"]:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _determine_entity_type(self, node: UDNode, all_nodes: List[UDNode]) -> Optional[str]:
        """Determine entity type based on UD features and context."""
        
        if node.pos == "PROPN":
            # Check for location indicators
            if any(word in node.token.lower() for word in ["city", "country", "state", "river", "mountain"]):
                return "LOCATION"
            # Check for person indicators
            elif any(word in node.token.lower() for word in ["mr", "mrs", "dr", "prof"]):
                return "PERSON"
            else:
                return "ORGANIZATION"
        elif node.pos == "NOUN":
            if node.token.lower() in ["boy", "girl", "man", "woman", "child"]:
                return "PERSON"
            elif node.token.lower() in ["cat", "dog", "bird", "fish"]:
                return "ANIMAL"
            else:
                return "OBJECT"
        
        return None
    
    def build_enhanced_structure(self, text: str) -> EnhancedSentenceStructure:
        """Build enhanced sentence structure with UD and SRL information."""
        
        # Parse UD structure
        ud_nodes = self.parse_ud_structure(text)
        
        # Extract semantic roles
        semantic_roles = self.extract_semantic_roles(ud_nodes)
        
        # Build dependency paths
        dependency_paths = self._build_dependency_paths(ud_nodes)
        
        # Group semantic roles
        role_mappings = self._group_semantic_roles(semantic_roles)
        
        # Extract basic structure
        subject, predicate, obj, indirect_obj = self._extract_basic_structure(ud_nodes)
        
        # Determine voice, mood, tense, aspect
        voice, mood, tense, aspect, modality = self._extract_grammatical_features(ud_nodes)
        
        # Extract temporal and spatial expressions
        temporal_expr = self._extract_temporal_expressions(ud_nodes, semantic_roles)
        spatial_expr = self._extract_spatial_expressions(ud_nodes, semantic_roles)
        
        # Extract causal relations
        causal_relations = self._extract_causal_relations(ud_nodes, semantic_roles)
        
        # Check for negation
        negation = self._check_negation(ud_nodes)
        
        return EnhancedSentenceStructure(
            subject=subject,
            predicate=predicate,
            object=obj,
            indirect_object=indirect_obj,
            ud_tree=ud_nodes,
            dependency_paths=dependency_paths,
            semantic_roles=semantic_roles,
            role_mappings=role_mappings,
            voice=voice,
            mood=mood,
            tense=tense,
            aspect=aspect,
            modality=modality,
            temporal_expressions=temporal_expr,
            spatial_expressions=spatial_expr,
            causal_relations=causal_relations,
            negation=negation
        )
    
    def _build_dependency_paths(self, ud_nodes: List[UDNode]) -> Dict[str, List[str]]:
        """Build dependency paths from root to each node."""
        
        paths = {}
        
        for node in ud_nodes:
            path = []
            current = node
            
            while current.head != current.index:  # Not root
                path.append(current.dep)
                current = ud_nodes[current.head]
            
            path.append("ROOT")
            paths[node.token] = list(reversed(path))
        
        return paths
    
    def _group_semantic_roles(self, semantic_roles: List[SemanticRole]) -> Dict[str, List[SemanticRole]]:
        """Group semantic roles by role type."""
        
        grouped = defaultdict(list)
        for role in semantic_roles:
            grouped[role.role].append(role)
        
        return dict(grouped)
    
    def _extract_basic_structure(self, ud_nodes: List[UDNode]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Extract basic subject, predicate, object structure."""
        
        subject = None
        predicate = None
        obj = None
        indirect_obj = None
        
        for node in ud_nodes:
            if node.dep in ["nsubj", "nsubjpass"]:
                subject = node.token
            elif node.dep == "ROOT" and node.pos == "VERB":
                predicate = node.token
            elif node.dep == "dobj":
                obj = node.token
            elif node.dep == "iobj":
                indirect_obj = node.token
        
        return subject, predicate, obj, indirect_obj
    
    def _extract_grammatical_features(self, ud_nodes: List[UDNode]) -> Tuple[str, str, str, str, Optional[str]]:
        """Extract grammatical features from UD structure."""
        
        voice = "active"
        mood = "declarative"
        tense = "present"
        aspect = "simple"
        modality = None
        
        for node in ud_nodes:
            # Voice detection
            if node.dep == "nsubjpass":
                voice = "passive"
            
            # Tense and aspect detection
            if node.pos == "VERB":
                if "VBD" in node.features.get("tag", ""):
                    tense = "past"
                elif "VBG" in node.features.get("tag", ""):
                    aspect = "progressive"
                elif "VBN" in node.features.get("tag", ""):
                    aspect = "perfect"
            
            # Modality detection
            if node.lemma in ["can", "could", "will", "would", "should", "must", "may", "might"]:
                modality = node.lemma
        
        return voice, mood, tense, aspect, modality
    
    def _extract_temporal_expressions(self, ud_nodes: List[UDNode], semantic_roles: List[SemanticRole]) -> List[str]:
        """Extract temporal expressions."""
        
        temporal_expr = []
        
        for role in semantic_roles:
            if role.role == "TIME":
                temporal_expr.append(role.text)
        
        # Also check for temporal adverbs
        for node in ud_nodes:
            if node.pos == "ADV" and any(word in node.token.lower() for word in ["when", "time", "year", "date", "today", "yesterday", "tomorrow"]):
                temporal_expr.append(node.token)
        
        return temporal_expr
    
    def _extract_spatial_expressions(self, ud_nodes: List[UDNode], semantic_roles: List[SemanticRole]) -> List[str]:
        """Extract spatial expressions."""
        
        spatial_expr = []
        
        for role in semantic_roles:
            if role.role == "LOCATION":
                spatial_expr.append(role.text)
        
        # Also check for spatial prepositions and adverbs
        for node in ud_nodes:
            if node.pos == "ADP" and any(word in node.token.lower() for word in ["in", "at", "on", "near", "under", "over"]):
                spatial_expr.append(node.token)
        
        return spatial_expr
    
    def _extract_causal_relations(self, ud_nodes: List[UDNode], semantic_roles: List[SemanticRole]) -> List[Tuple[str, str]]:
        """Extract causal relations."""
        
        causal_relations = []
        
        for i, node in enumerate(ud_nodes):
            if node.lemma in ["because", "since", "as", "therefore", "thus"]:
                # Look for cause and effect in surrounding context
                if i > 0:
                    cause = ud_nodes[i-1].token
                else:
                    cause = "unknown"
                
                if i < len(ud_nodes) - 1:
                    effect = ud_nodes[i+1].token
                else:
                    effect = "unknown"
                
                causal_relations.append((cause, effect))
        
        return causal_relations
    
    def _check_negation(self, ud_nodes: List[UDNode]) -> bool:
        """Check for negation in the sentence."""
        
        for node in ud_nodes:
            if node.dep == "neg" or node.lemma in ["not", "no", "never", "none"]:
                return True
        
        return False
    
    def decompose_with_ud_srl(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Decompose text using UD and SRL enhanced analysis."""
        
        print(f"🔍 UD + SRL Enhanced Decomposition: '{text}'")
        print("-" * 70)
        
        # Step 1: Build enhanced structure
        structure = self.build_enhanced_structure(text)
        print(f"📐 Enhanced Structure:")
        print(f"  Subject: {structure.subject}")
        print(f"  Predicate: {structure.predicate}")
        print(f"  Object: {structure.object}")
        print(f"  Voice: {structure.voice}")
        print(f"  Tense: {structure.tense}")
        print(f"  Aspect: {structure.aspect}")
        print(f"  Negation: {structure.negation}")
        
        # Step 2: Show UD analysis
        print(f"\n🌳 Universal Dependencies:")
        for node in structure.ud_tree:
            print(f"  {node.token} ({node.pos}) --{node.dep}--> {structure.ud_tree[node.head].token if node.head != node.index else 'ROOT'}")
        
        # Step 3: Show SRL analysis
        print(f"\n🎭 Semantic Role Labeling:")
        for role in structure.semantic_roles:
            print(f"  {role.role}: '{role.text}' (confidence: {role.confidence:.2f}, type: {role.entity_type})")
        
        # Step 4: Show role mappings
        print(f"\n📋 Role Mappings:")
        for role_type, roles in structure.role_mappings.items():
            print(f"  {role_type}: {[r.text for r in roles]}")
        
        # Step 5: Enhanced semantic decomposition
        decomposed_concepts = self._decompose_with_semantic_roles(text, structure)
        print(f"\n🧩 Enhanced Concept Decomposition:")
        for concept_type, decompositions in decomposed_concepts.items():
            print(f"  {concept_type}: {decompositions}")
        
        # Step 6: Generate enhanced prime language
        prime_language = self._generate_enhanced_prime_language(structure, decomposed_concepts)
        print(f"\n🎯 Enhanced Prime Language:")
        print(f"  {prime_language}")
        
        return {
            "original": text,
            "enhanced_structure": structure,
            "decomposed_concepts": decomposed_concepts,
            "prime_language": prime_language
        }
    
    def _decompose_with_semantic_roles(self, text: str, structure: EnhancedSentenceStructure) -> Dict[str, List[str]]:
        """Decompose concepts using semantic role information."""
        
        decomposed = {}
        
        # Decompose based on semantic roles
        for role_type, roles in structure.role_mappings.items():
            for role in roles:
                if role.text.lower() in self.concept_decompositions:
                    concept_info = self.concept_decompositions[role.text.lower()]
                    
                    # Add semantic role context to decomposition
                    enhanced_decomposition = concept_info["decomposition"].copy()
                    enhanced_decomposition.append(f"this thing has the semantic role: {role.role}")
                    
                    if role.entity_type:
                        enhanced_decomposition.append(f"this thing is of type: {role.entity_type}")
                    
                    decomposed[f"{role.text}_{role.role}"] = enhanced_decomposition
        
        return decomposed
    
    def _generate_enhanced_prime_language(self, structure: EnhancedSentenceStructure, decomposed_concepts: Dict[str, List[str]]) -> str:
        """Generate enhanced Prime language with UD and SRL information."""
        
        prime_components = []
        
        # Add temporal information
        if structure.temporal_expressions:
            prime_components.append(f"this happened at: {', '.join(structure.temporal_expressions)}")
        
        # Add spatial information
        if structure.spatial_expressions:
            prime_components.append(f"this happened at: {', '.join(structure.spatial_expressions)}")
        
        # Add causal relations
        for cause, effect in structure.causal_relations:
            prime_components.append(f"because of {cause}, {effect} happened")
        
        # Add negation
        if structure.negation:
            prime_components.append("this did not happen")
        
        # Add concept decompositions
        for concept_type, decomposition in decomposed_concepts.items():
            prime_components.extend(decomposition)
        
        # Add voice and aspect information
        if structure.voice == "passive":
            prime_components.append("something happened to someone")
        else:
            prime_components.append("someone did something")
        
        if structure.aspect == "progressive":
            prime_components.append("this was happening")
        elif structure.aspect == "perfect":
            prime_components.append("this had happened")
        
        # Join into coherent representation
        if prime_components:
            return " ".join(prime_components)
        else:
            return "someone did something"

def demonstrate_ud_srl_enhancement():
    """Demonstrate the UD + SRL enhanced decomposition."""
    
    print("🌳 UD + SRL ENHANCED DECOMPOSITION DEMONSTRATION")
    print("=" * 80)
    print()
    
    enhanced_engine = UDSRLEnhancedDecompositionEngine()
    
    # Test cases that benefit from UD + SRL
    test_cases = [
        "The boy kicked the ball in Paris.",
        "Einstein was born in Germany.",
        "Shakespeare wrote many plays.",
        "The cat sat on the mat.",
        "Apple Inc. was founded by Steve Jobs in California.",
        "The Great Wall of China is a famous landmark.",
        "Microsoft released Windows 11 in 2021.",
        "The teacher gave the book to the student."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"🎯 EXAMPLE {i}: '{text}'")
        print("-" * 60)
        
        try:
            result = enhanced_engine.decompose_with_ud_srl(text)
            
            print(f"\n📊 ENHANCED RESULTS:")
            print(f"  Semantic Roles: {len(result['enhanced_structure'].semantic_roles)}")
            print(f"  UD Nodes: {len(result['enhanced_structure'].ud_tree)}")
            print(f"  Decomposed Concepts: {len(result['decomposed_concepts'])}")
            print(f"  Enhanced Prime Language: {result['prime_language'][:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print()

if __name__ == "__main__":
    demonstrate_ud_srl_enhancement()
