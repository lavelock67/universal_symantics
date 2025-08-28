#!/usr/bin/env python3
"""
Refined UD + SRL Enhanced Semantic Decomposition

This refines the UD + SRL integration to address over-assignment
and provide better semantic role disambiguation.
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
class RefinedSemanticRole:
    """Represents a refined semantic role with disambiguation."""
    role: str
    text: str
    start: int
    end: int
    confidence: float
    entity_type: Optional[str] = None
    context: str = ""
    disambiguation_score: float = 0.0

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

class RefinedUDSRLDecompositionEngine:
    """Refined semantic decomposition with better UD and SRL integration."""
    
    def __init__(self):
        # Load SpaCy with UD support
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  SpaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Refined SRL patterns with priority and exclusivity
        self.srl_patterns = self._initialize_refined_srl_patterns()
        
        # UD dependency patterns
        self.ud_role_patterns = self._initialize_ud_role_patterns()
        
        # Enhanced concept decompositions
        self.concept_decompositions = self._initialize_enhanced_decompositions()
        
        # Role exclusivity rules
        self.role_exclusivity = self._initialize_role_exclusivity()
    
    def _initialize_refined_srl_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize refined SRL patterns with priority and context rules."""
        
        return {
            "AGENT": {
                "patterns": [
                    {"dep": "nsubj", "pos": ["NOUN", "PROPN", "PRON"], "priority": 1.0},
                    {"dep": "nsubjpass", "pos": ["NOUN", "PROPN", "PRON"], "priority": 0.8},
                    {"dep": "agent", "pos": ["NOUN", "PROPN", "PRON"], "priority": 0.9}
                ],
                "exclusive_with": ["PATIENT", "THEME"],
                "context_indicators": ["by", "from", "of"]
            },
            "PATIENT": {
                "patterns": [
                    {"dep": "dobj", "pos": ["NOUN", "PROPN", "PRON"], "priority": 1.0},
                    {"dep": "nsubjpass", "pos": ["NOUN", "PROPN", "PRON"], "priority": 0.9}
                ],
                "exclusive_with": ["AGENT"],
                "context_indicators": ["to", "for", "at"]
            },
            "THEME": {
                "patterns": [
                    {"dep": "dobj", "pos": ["NOUN", "PROPN", "PRON"], "priority": 0.8},
                    {"dep": "attr", "pos": ["NOUN", "PROPN", "PRON"], "priority": 0.9},
                    {"dep": "pobj", "pos": ["NOUN", "PROPN", "PRON"], "priority": 0.7}
                ],
                "exclusive_with": ["AGENT"],
                "context_indicators": ["about", "of", "concerning"]
            },
            "GOAL": {
                "patterns": [
                    {"dep": "iobj", "pos": ["NOUN", "PROPN", "PRON"], "priority": 1.0},
                    {"dep": "pobj", "pos": ["NOUN", "PROPN", "PRON"], "priority": 0.8}
                ],
                "exclusive_with": ["AGENT", "PATIENT"],
                "context_indicators": ["to", "toward", "for"]
            },
            "LOCATION": {
                "patterns": [
                    {"dep": "pobj", "pos": ["NOUN", "PROPN"], "priority": 0.9},
                    {"dep": "nmod", "pos": ["NOUN", "PROPN"], "priority": 0.8}
                ],
                "exclusive_with": ["AGENT", "PATIENT", "THEME"],
                "context_indicators": ["in", "at", "on", "near", "inside", "outside"]
            },
            "TIME": {
                "patterns": [
                    {"dep": "nmod", "pos": ["NOUN", "PROPN"], "priority": 0.9},
                    {"dep": "advmod", "pos": ["ADV"], "priority": 0.8}
                ],
                "exclusive_with": ["AGENT", "PATIENT", "THEME", "LOCATION"],
                "context_indicators": ["when", "during", "at", "in", "on", "before", "after"]
            },
            "INSTRUMENT": {
                "patterns": [
                    {"dep": "pobj", "pos": ["NOUN", "PROPN"], "priority": 0.8},
                    {"dep": "nmod", "pos": ["NOUN", "PROPN"], "priority": 0.7}
                ],
                "exclusive_with": ["AGENT", "PATIENT", "THEME"],
                "context_indicators": ["with", "using", "by", "through"]
            },
            "MANNER": {
                "patterns": [
                    {"dep": "advmod", "pos": ["ADV", "ADJ"], "priority": 0.9},
                    {"dep": "amod", "pos": ["ADJ"], "priority": 0.8}
                ],
                "exclusive_with": ["AGENT", "PATIENT", "THEME", "GOAL"],
                "context_indicators": ["quickly", "slowly", "carefully", "well", "badly"]
            }
        }
    
    def _initialize_role_exclusivity(self) -> Dict[str, List[str]]:
        """Initialize role exclusivity rules."""
        
        return {
            "AGENT": ["PATIENT", "THEME"],
            "PATIENT": ["AGENT"],
            "THEME": ["AGENT"],
            "GOAL": ["AGENT", "PATIENT"],
            "LOCATION": ["AGENT", "PATIENT", "THEME"],
            "TIME": ["AGENT", "PATIENT", "THEME", "LOCATION"],
            "INSTRUMENT": ["AGENT", "PATIENT", "THEME"],
            "MANNER": ["AGENT", "PATIENT", "THEME", "GOAL"]
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
        """Initialize enhanced concept decompositions."""
        
        return {
            # People and beings
            "boy": {
                "type": "person",
                "semantic_roles": ["AGENT", "EXPERIENCER"],
                "decomposition": [
                    "someone of one kind",
                    "when someone is someone of this kind, this someone's body is small",
                    "this someone is not someone of another kind (girl)"
                ]
            },
            "girl": {
                "type": "person",
                "semantic_roles": ["AGENT", "EXPERIENCER"],
                "decomposition": [
                    "someone of one kind",
                    "when someone is someone of this kind, this someone's body is small",
                    "this someone is not someone of another kind (boy)"
                ]
            },
            "cat": {
                "type": "animal",
                "semantic_roles": ["AGENT", "EXPERIENCER"],
                "decomposition": [
                    "a living thing of one kind",
                    "this thing is not a person",
                    "this thing can move",
                    "this thing has four legs",
                    "this thing is small"
                ]
            },
            "teacher": {
                "type": "person",
                "semantic_roles": ["AGENT", "EXPERIENCER"],
                "decomposition": [
                    "someone of one kind",
                    "this someone knows many things",
                    "this someone helps other people know things"
                ]
            },
            "student": {
                "type": "person",
                "semantic_roles": ["EXPERIENCER", "RECIPIENT"],
                "decomposition": [
                    "someone of one kind",
                    "this someone wants to know many things",
                    "other people help this someone know things"
                ]
            },
            
            # Objects and things
            "ball": {
                "type": "object",
                "semantic_roles": ["THEME", "PATIENT"],
                "decomposition": [
                    "a thing of one kind",
                    "when people see it, they think: it is small, it is round",
                    "this thing can move because of what people do to it"
                ]
            },
            "mat": {
                "type": "object",
                "semantic_roles": ["LOCATION"],
                "decomposition": [
                    "a thing of one kind",
                    "this thing is flat",
                    "this thing is on the ground",
                    "people can sit on this thing"
                ]
            },
            "book": {
                "type": "object",
                "semantic_roles": ["THEME", "PATIENT"],
                "decomposition": [
                    "a thing of one kind",
                    "this thing has many words",
                    "people can see these words",
                    "because of this, people can know many things"
                ]
            },
            
            # Actions
            "kick": {
                "type": "action",
                "semantic_roles": ["AGENT", "PATIENT", "INSTRUMENT"],
                "decomposition": [
                    "someone X did something to someone else Y",
                    "this someone's leg moved",
                    "this someone's leg touched the thing",
                    "because of this, the thing moved",
                    "it happened in one moment"
                ]
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
                ]
            },
            "sat": {
                "type": "action",
                "semantic_roles": ["AGENT", "LOCATION"],
                "decomposition": [
                    "someone did something",
                    "this someone was not moving",
                    "this someone was on something",
                    "this happened for some time"
                ]
            },
            "wrote": {
                "type": "action",
                "semantic_roles": ["AGENT", "THEME"],
                "decomposition": [
                    "someone did something",
                    "this someone made many words",
                    "because of this, other people can see these words",
                    "because of this, other people can know many things"
                ]
            },
            "gave": {
                "type": "action",
                "semantic_roles": ["AGENT", "THEME", "RECIPIENT"],
                "decomposition": [
                    "someone did something",
                    "this someone had something",
                    "this someone made someone else have this something",
                    "because of this, the other someone had this something"
                ]
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
    
    def extract_refined_semantic_roles(self, ud_nodes: List[UDNode]) -> List[RefinedSemanticRole]:
        """Extract refined semantic roles with disambiguation."""
        
        # First pass: collect all potential roles
        potential_roles = []
        
        for node in ud_nodes:
            for role_name, role_config in self.srl_patterns.items():
                for pattern in role_config["patterns"]:
                    if (node.dep == pattern["dep"] and 
                        node.pos in pattern["pos"]):
                        
                        # Calculate base confidence
                        confidence = pattern["priority"]
                        
                        # Check context indicators
                        context_score = self._check_context_indicators(node, ud_nodes, role_config["context_indicators"])
                        confidence *= context_score
                        
                        # Create potential role
                        potential_role = RefinedSemanticRole(
                            role=role_name,
                            text=node.token,
                            start=node.index,
                            end=node.index + 1,
                            confidence=confidence,
                            entity_type=self._determine_entity_type(node, ud_nodes),
                            context=self._extract_context(node, ud_nodes),
                            disambiguation_score=0.0
                        )
                        potential_roles.append(potential_role)
        
        # Second pass: resolve conflicts and assign final roles
        final_roles = self._resolve_role_conflicts(potential_roles, ud_nodes)
        
        return final_roles
    
    def _check_context_indicators(self, node: UDNode, all_nodes: List[UDNode], indicators: List[str]) -> float:
        """Check for context indicators that support a semantic role."""
        
        # Check if any parent or child nodes contain context indicators
        for other_node in all_nodes:
            if other_node.head == node.index or node.head == other_node.index:
                if any(indicator in other_node.token.lower() for indicator in indicators):
                    return 1.2  # Boost confidence
        
        return 1.0  # No context boost
    
    def _extract_context(self, node: UDNode, all_nodes: List[UDNode]) -> str:
        """Extract context around a node."""
        
        context_words = []
        for other_node in all_nodes:
            if abs(other_node.index - node.index) <= 2:  # Within 2 tokens
                context_words.append(other_node.token)
        
        return " ".join(context_words)
    
    def _resolve_role_conflicts(self, potential_roles: List[RefinedSemanticRole], ud_nodes: List[UDNode]) -> List[RefinedSemanticRole]:
        """Resolve conflicts between semantic roles and assign final roles."""
        
        # Group roles by text
        roles_by_text = defaultdict(list)
        for role in potential_roles:
            roles_by_text[role.text].append(role)
        
        final_roles = []
        
        for text, roles in roles_by_text.items():
            if len(roles) == 1:
                # No conflict, keep the role
                final_roles.append(roles[0])
            else:
                # Resolve conflict by selecting the best role
                best_role = self._select_best_role(roles, ud_nodes)
                if best_role:
                    final_roles.append(best_role)
        
        return final_roles
    
    def _select_best_role(self, roles: List[RefinedSemanticRole], ud_nodes: List[UDNode]) -> Optional[RefinedSemanticRole]:
        """Select the best semantic role from conflicting options."""
        
        if not roles:
            return None
        
        # Calculate disambiguation scores
        for role in roles:
            role.disambiguation_score = self._calculate_disambiguation_score(role, roles, ud_nodes)
        
        # Select role with highest disambiguation score
        best_role = max(roles, key=lambda r: r.disambiguation_score)
        
        # Only keep if disambiguation score is above threshold
        if best_role.disambiguation_score > 0.5:
            return best_role
        
        return None
    
    def _calculate_disambiguation_score(self, role: RefinedSemanticRole, all_roles: List[RefinedSemanticRole], ud_nodes: List[UDNode]) -> float:
        """Calculate disambiguation score for a semantic role."""
        
        base_score = role.confidence
        
        # Boost for exclusive roles
        if role.role in ["AGENT", "PATIENT"]:
            base_score *= 1.2
        
        # Penalize for conflicts with exclusive roles
        exclusive_roles = self.role_exclusivity.get(role.role, [])
        for other_role in all_roles:
            if other_role.role in exclusive_roles and other_role.text != role.text:
                base_score *= 0.7
        
        # Boost for strong context indicators
        if any(word in role.context.lower() for word in ["by", "to", "in", "at", "on"]):
            base_score *= 1.1
        
        # Boost for proper nouns in appropriate roles
        if role.entity_type in ["PERSON", "ORGANIZATION"] and role.role in ["AGENT", "THEME"]:
            base_score *= 1.1
        
        return base_score
    
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
            if node.token.lower() in ["boy", "girl", "man", "woman", "child", "teacher", "student"]:
                return "PERSON"
            elif node.token.lower() in ["cat", "dog", "bird", "fish"]:
                return "ANIMAL"
            else:
                return "OBJECT"
        
        return None
    
    def decompose_with_refined_ud_srl(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Decompose text using refined UD and SRL analysis."""
        
        print(f"🔍 Refined UD + SRL Decomposition: '{text}'")
        print("-" * 70)
        
        # Parse UD structure
        ud_nodes = self.parse_ud_structure(text)
        
        # Extract refined semantic roles
        semantic_roles = self.extract_refined_semantic_roles(ud_nodes)
        
        # Group roles by type
        role_mappings = defaultdict(list)
        for role in semantic_roles:
            role_mappings[role.role].append(role)
        
        # Extract basic structure
        subject, predicate, obj, indirect_obj = self._extract_basic_structure(ud_nodes)
        
        # Determine grammatical features
        voice, mood, tense, aspect, modality = self._extract_grammatical_features(ud_nodes)
        
        # Extract temporal and spatial expressions
        temporal_expr = self._extract_temporal_expressions(ud_nodes, semantic_roles)
        spatial_expr = self._extract_spatial_expressions(ud_nodes, semantic_roles)
        
        # Extract causal relations
        causal_relations = self._extract_causal_relations(ud_nodes, semantic_roles)
        
        # Check for negation
        negation = self._check_negation(ud_nodes)
        
        # Show analysis
        print(f"📐 Structure:")
        print(f"  Subject: {subject}")
        print(f"  Predicate: {predicate}")
        print(f"  Object: {obj}")
        print(f"  Voice: {voice}")
        print(f"  Tense: {tense}")
        print(f"  Aspect: {aspect}")
        print(f"  Negation: {negation}")
        
        print(f"\n🌳 Universal Dependencies:")
        for node in ud_nodes:
            print(f"  {node.token} ({node.pos}) --{node.dep}--> {ud_nodes[node.head].token if node.head != node.index else 'ROOT'}")
        
        print(f"\n🎭 Refined Semantic Role Labeling:")
        for role in semantic_roles:
            print(f"  {role.role}: '{role.text}' (confidence: {role.confidence:.2f}, disambiguation: {role.disambiguation_score:.2f}, type: {role.entity_type})")
        
        print(f"\n📋 Role Mappings:")
        for role_type, roles in role_mappings.items():
            print(f"  {role_type}: {[r.text for r in roles]}")
        
        # Enhanced semantic decomposition
        decomposed_concepts = self._decompose_with_refined_roles(text, semantic_roles)
        print(f"\n🧩 Enhanced Concept Decomposition:")
        for concept_type, decompositions in decomposed_concepts.items():
            print(f"  {concept_type}: {decompositions}")
        
        # Generate enhanced prime language
        prime_language = self._generate_refined_prime_language(
            semantic_roles, decomposed_concepts, temporal_expr, spatial_expr, 
            causal_relations, negation, voice, aspect
        )
        print(f"\n🎯 Refined Prime Language:")
        print(f"  {prime_language}")
        
        return {
            "original": text,
            "ud_nodes": ud_nodes,
            "semantic_roles": semantic_roles,
            "role_mappings": dict(role_mappings),
            "decomposed_concepts": decomposed_concepts,
            "prime_language": prime_language
        }
    
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
            if node.dep == "nsubjpass":
                voice = "passive"
            
            if node.pos == "VERB":
                if "VBD" in node.features.get("tag", ""):
                    tense = "past"
                elif "VBG" in node.features.get("tag", ""):
                    aspect = "progressive"
                elif "VBN" in node.features.get("tag", ""):
                    aspect = "perfect"
            
            if node.lemma in ["can", "could", "will", "would", "should", "must", "may", "might"]:
                modality = node.lemma
        
        return voice, mood, tense, aspect, modality
    
    def _extract_temporal_expressions(self, ud_nodes: List[UDNode], semantic_roles: List[RefinedSemanticRole]) -> List[str]:
        """Extract temporal expressions."""
        
        temporal_expr = []
        
        for role in semantic_roles:
            if role.role == "TIME":
                temporal_expr.append(role.text)
        
        for node in ud_nodes:
            if node.pos == "ADV" and any(word in node.token.lower() for word in ["when", "time", "year", "date", "today", "yesterday", "tomorrow"]):
                temporal_expr.append(node.token)
        
        return temporal_expr
    
    def _extract_spatial_expressions(self, ud_nodes: List[UDNode], semantic_roles: List[RefinedSemanticRole]) -> List[str]:
        """Extract spatial expressions."""
        
        spatial_expr = []
        
        for role in semantic_roles:
            if role.role == "LOCATION":
                spatial_expr.append(role.text)
        
        for node in ud_nodes:
            if node.pos == "ADP" and any(word in node.token.lower() for word in ["in", "at", "on", "near", "under", "over"]):
                spatial_expr.append(node.token)
        
        return spatial_expr
    
    def _extract_causal_relations(self, ud_nodes: List[UDNode], semantic_roles: List[RefinedSemanticRole]) -> List[Tuple[str, str]]:
        """Extract causal relations."""
        
        causal_relations = []
        
        for i, node in enumerate(ud_nodes):
            if node.lemma in ["because", "since", "as", "therefore", "thus"]:
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
    
    def _decompose_with_refined_roles(self, text: str, semantic_roles: List[RefinedSemanticRole]) -> Dict[str, List[str]]:
        """Decompose concepts using refined semantic role information."""
        
        decomposed = {}
        
        for role in semantic_roles:
            if role.text.lower() in self.concept_decompositions:
                concept_info = self.concept_decompositions[role.text.lower()]
                
                # Add semantic role context to decomposition
                enhanced_decomposition = concept_info["decomposition"].copy()
                enhanced_decomposition.append(f"this thing has the semantic role: {role.role}")
                
                if role.entity_type:
                    enhanced_decomposition.append(f"this thing is of type: {role.entity_type}")
                
                decomposed[f"{role.text}_{role.role}"] = enhanced_decomposition
        
        return decomposed
    
    def _generate_refined_prime_language(self, semantic_roles: List[RefinedSemanticRole], 
                                       decomposed_concepts: Dict[str, List[str]], 
                                       temporal_expr: List[str], spatial_expr: List[str],
                                       causal_relations: List[Tuple[str, str]], 
                                       negation: bool, voice: str, aspect: str) -> str:
        """Generate refined Prime language with UD and SRL information."""
        
        prime_components = []
        
        # Add temporal information
        if temporal_expr:
            prime_components.append(f"this happened at: {', '.join(temporal_expr)}")
        
        # Add spatial information
        if spatial_expr:
            prime_components.append(f"this happened at: {', '.join(spatial_expr)}")
        
        # Add causal relations
        for cause, effect in causal_relations:
            prime_components.append(f"because of {cause}, {effect} happened")
        
        # Add negation
        if negation:
            prime_components.append("this did not happen")
        
        # Add concept decompositions
        for concept_type, decomposition in decomposed_concepts.items():
            prime_components.extend(decomposition)
        
        # Add voice and aspect information
        if voice == "passive":
            prime_components.append("something happened to someone")
        else:
            prime_components.append("someone did something")
        
        if aspect == "progressive":
            prime_components.append("this was happening")
        elif aspect == "perfect":
            prime_components.append("this had happened")
        
        # Join into coherent representation
        if prime_components:
            return " ".join(prime_components)
        else:
            return "someone did something"

def demonstrate_refined_decomposition():
    """Demonstrate the refined UD + SRL decomposition."""
    
    print("🌳 REFINED UD + SRL DECOMPOSITION DEMONSTRATION")
    print("=" * 80)
    print()
    
    refined_engine = RefinedUDSRLDecompositionEngine()
    
    # Test cases that benefit from refined analysis
    test_cases = [
        "The boy kicked the ball in Paris.",
        "Einstein was born in Germany.",
        "Shakespeare wrote many plays.",
        "The cat sat on the mat.",
        "The teacher gave the book to the student.",
        "Apple Inc. was founded by Steve Jobs in California.",
        "Microsoft released Windows 11 in 2021."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"🎯 EXAMPLE {i}: '{text}'")
        print("-" * 60)
        
        try:
            result = refined_engine.decompose_with_refined_ud_srl(text)
            
            print(f"\n📊 REFINED RESULTS:")
            print(f"  Semantic Roles: {len(result['semantic_roles'])}")
            print(f"  UD Nodes: {len(result['ud_nodes'])}")
            print(f"  Decomposed Concepts: {len(result['decomposed_concepts'])}")
            print(f"  Refined Prime Language: {result['prime_language'][:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print()

if __name__ == "__main__":
    demonstrate_refined_decomposition()
