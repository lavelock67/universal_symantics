#!/usr/bin/env python3
"""
Semantic Decomposition Engine

This implements proper semantic decomposition that breaks down complex concepts
into NSM prime combinations with causal and temporal relationships.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language, PrimeType
from typing import List, Dict, Any, Tuple, Optional
import re

class SemanticDecompositionEngine:
    """Decomposes complex concepts into NSM prime combinations."""
    
    def __init__(self):
        self.nsm_service = NSMDetectionService()
        
        # Complex concept decompositions
        self.concept_decompositions = {
            # People and beings
            "boy": {
                "type": "person",
                "decomposition": [
                    "someone of one kind",
                    "when someone is someone of this kind, this someone's body is small",
                    "this someone is not someone of another kind (girl)"
                ]
            },
            "girl": {
                "type": "person", 
                "decomposition": [
                    "someone of one kind",
                    "when someone is someone of this kind, this someone's body is small",
                    "this someone is not someone of another kind (boy)"
                ]
            },
            "man": {
                "type": "person",
                "decomposition": [
                    "someone of one kind", 
                    "when someone is someone of this kind, this someone's body is big",
                    "this someone is not someone of another kind (woman)"
                ]
            },
            "woman": {
                "type": "person",
                "decomposition": [
                    "someone of one kind",
                    "when someone is someone of this kind, this someone's body is big", 
                    "this someone is not someone of another kind (man)"
                ]
            },
            "child": {
                "type": "person",
                "decomposition": [
                    "someone of one kind",
                    "when someone is someone of this kind, this someone's body is small",
                    "this someone has not lived for a long time"
                ]
            },
            
            # Objects and things
            "ball": {
                "type": "object",
                "decomposition": [
                    "a thing of one kind",
                    "when people see it, they think: it is small, it is round",
                    "this thing can move because of what people do to it"
                ]
            },
            "car": {
                "type": "object", 
                "decomposition": [
                    "a thing of one kind",
                    "this thing is big",
                    "this thing can move",
                    "people can be inside this thing",
                    "this thing moves because of what people do to it"
                ]
            },
            "house": {
                "type": "object",
                "decomposition": [
                    "a thing of one kind",
                    "this thing is big",
                    "this thing is a place where people live",
                    "people can be inside this thing"
                ]
            },
            "book": {
                "type": "object",
                "decomposition": [
                    "a thing of one kind", 
                    "this thing has many words",
                    "people can see these words",
                    "because of this, people can know many things"
                ]
            },
            "food": {
                "type": "object",
                "decomposition": [
                    "a thing of one kind",
                    "this thing is something people eat",
                    "because of this, people can live",
                    "this thing goes inside people's bodies"
                ]
            },
            
            # Actions and events
            "kick": {
                "type": "action",
                "decomposition": [
                    "someone X did something to someone else Y",
                    "this someone's leg moved",
                    "this someone's leg touched the thing",
                    "because of this, the thing moved",
                    "it happened in one moment"
                ]
            },
            "threw": {
                "type": "action",
                "decomposition": [
                    "someone X did something to someone else Y",
                    "this someone's hand moved",
                    "this someone's hand touched the thing",
                    "because of this, the thing moved",
                    "the thing moved far from this someone"
                ]
            },
            "ate": {
                "type": "action",
                "decomposition": [
                    "someone did something",
                    "this someone put something in this someone's mouth",
                    "because of this, this something went inside this someone's body",
                    "because of this, this someone can live"
                ]
            },
            "slept": {
                "type": "action",
                "decomposition": [
                    "someone did something",
                    "this someone's eyes were not open",
                    "this someone was not thinking",
                    "this someone was not moving",
                    "this happened for some time"
                ]
            },
            
            # Abstract concepts
            "loves": {
                "type": "abstract",
                "decomposition": [
                    "someone feels something good",
                    "this someone thinks good things about someone else",
                    "this someone wants to be near this someone else",
                    "this someone wants good things to happen to this someone else"
                ]
            },
            "hate": {
                "type": "abstract",
                "decomposition": [
                    "someone feels something bad",
                    "this someone thinks bad things about someone else", 
                    "this someone does not want to be near this someone else",
                    "this someone does not want good things to happen to this someone else"
                ]
            },
            "understand": {
                "type": "abstract",
                "decomposition": [
                    "someone knows something",
                    "this someone can think about this something",
                    "this someone can say something about this something",
                    "this someone thinks: this is true"
                ]
            }
        }
        
        # Action patterns for complex verbs
        self.action_patterns = {
            "physical_contact": {
                "pattern": r"\b(kick|hit|touch|push|pull|hold|grab)\b",
                "decomposition": [
                    "someone's body part moved",
                    "this body part touched something",
                    "because of this, something happened to this something"
                ]
            },
            "movement": {
                "pattern": r"\b(walk|run|jump|climb|swim|fly)\b", 
                "decomposition": [
                    "someone moved",
                    "this someone's body moved",
                    "this someone was not in the same place",
                    "this happened for some time"
                ]
            },
            "communication": {
                "pattern": r"\b(say|tell|speak|talk|ask|answer)\b",
                "decomposition": [
                    "someone said something",
                    "this someone wanted someone else to know something",
                    "because of this, someone else could hear this something"
                ]
            }
        }
    
    def decompose_sentence(self, text: str, language: Language) -> Dict[str, Any]:
        """Decompose a sentence into semantic prime combinations."""
        
        print(f"🔍 Decomposing: '{text}'")
        print("-" * 60)
        
        # Step 1: Parse sentence structure
        structure = self._parse_sentence_structure(text)
        print(f"📐 Structure: {structure}")
        
        # Step 2: Decompose concepts
        decomposed_concepts = self._decompose_concepts(text, structure)
        print(f"🧩 Concepts: {decomposed_concepts}")
        
        # Step 3: Decompose actions
        decomposed_actions = self._decompose_actions(text, structure)
        print(f"⚡ Actions: {decomposed_actions}")
        
        # Step 4: Build semantic relationships
        semantic_relationships = self._build_semantic_relationships(
            structure, decomposed_concepts, decomposed_actions
        )
        print(f"🔗 Relationships: {semantic_relationships}")
        
        # Step 5: Generate prime language representation
        prime_language = self._generate_prime_language(semantic_relationships)
        print(f"🎯 Prime Language: {prime_language}")
        
        return {
            "original": text,
            "structure": structure,
            "concepts": decomposed_concepts,
            "actions": decomposed_actions,
            "relationships": semantic_relationships,
            "prime_language": prime_language
        }
    
    def _parse_sentence_structure(self, text: str) -> Dict[str, Any]:
        """Parse basic sentence structure."""
        
        # Clean words by removing punctuation
        words = [word.lower().rstrip('.,!?;:') for word in text.split()]
        
        structure = {
            "subject": None,
            "predicate": None,
            "object": None,
            "modifiers": [],
            "temporal": None,
            "spatial": None
        }
        
        # Find subject (first noun phrase)
        for i, word in enumerate(words):
            if word in ["the", "a", "an"]:
                continue
            if word in ["boy", "girl", "man", "woman", "child", "person", "someone"]:
                structure["subject"] = word
                break
        
        # Find predicate (verb)
        for word in words:
            if word in ["kicked", "threw", "ate", "slept", "walked", "ran", "said", "thought", "loves", "loved"]:
                structure["predicate"] = word
                break
        
        # Find object (second noun phrase) - look after the verb
        verb_found = False
        for word in words:
            if word == structure["predicate"]:
                verb_found = True
                continue
            if verb_found and word in ["ball", "car", "house", "book", "food", "thing", "something", "child"]:
                if word != structure["subject"]:
                    structure["object"] = word
                    break
        
        # If no object found after verb, try to find any object in the sentence
        if not structure["object"]:
            for word in words:
                if word in ["ball", "car", "house", "book", "food", "thing", "something", "child"]:
                    if word != structure["subject"]:
                        structure["object"] = word
                        break
        
        return structure
    
    def _decompose_concepts(self, text: str, structure: Dict[str, Any]) -> Dict[str, List[str]]:
        """Decompose complex concepts into prime combinations."""
        
        decomposed = {}
        
        # Decompose subject
        if structure["subject"] and structure["subject"] in self.concept_decompositions:
            decomposed["subject"] = self.concept_decompositions[structure["subject"]]["decomposition"]
        
        # Decompose object
        if structure["object"] and structure["object"] in self.concept_decompositions:
            decomposed["object"] = self.concept_decompositions[structure["object"]]["decomposition"]
        
        return decomposed
    
    def _decompose_actions(self, text: str, structure: Dict[str, Any]) -> List[str]:
        """Decompose complex actions into prime combinations."""
        
        if not structure["predicate"]:
            return []
        
        predicate = structure["predicate"]
        
        # Check specific action decompositions
        if predicate in self.concept_decompositions:
            return self.concept_decompositions[predicate]["decomposition"]
        
        # Check action patterns
        for pattern_name, pattern_info in self.action_patterns.items():
            if re.search(pattern_info["pattern"], predicate):
                return pattern_info["decomposition"]
        
        # Default action decomposition
        return [
            "someone did something",
            "because of this, something happened"
        ]
    
    def _build_semantic_relationships(self, structure: Dict[str, Any], 
                                   concepts: Dict[str, List[str]], 
                                   actions: List[str]) -> Dict[str, Any]:
        """Build semantic relationships between components."""
        
        relationships = {
            "agent": None,
            "action": None,
            "patient": None,
            "temporal": None,
            "causal": [],
            "spatial": None
        }
        
        # Agent (subject)
        if "subject" in concepts:
            relationships["agent"] = {
                "type": "person",
                "decomposition": concepts["subject"]
            }
        
        # Action
        if actions:
            relationships["action"] = {
                "type": "event",
                "decomposition": actions
            }
        
        # Patient (object)
        if "object" in concepts:
            relationships["patient"] = {
                "type": "object", 
                "decomposition": concepts["object"]
            }
        
        # Causal relationships
        if relationships["agent"] and relationships["action"]:
            relationships["causal"].append({
                "cause": "agent did action",
                "effect": "something happened to patient"
            })
        
        return relationships
    
    def _generate_prime_language(self, relationships: Dict[str, Any]) -> str:
        """Generate coherent Prime language representation."""
        
        prime_components = []
        
        # About the agent
        if relationships["agent"]:
            agent_decomp = relationships["agent"]["decomposition"]
            prime_components.extend(agent_decomp)
        
        # About the action
        if relationships["action"]:
            action_decomp = relationships["action"]["decomposition"]
            prime_components.extend(action_decomp)
        
        # About the patient
        if relationships["patient"]:
            patient_decomp = relationships["patient"]["decomposition"]
            prime_components.extend(patient_decomp)
        
        # Causal connections
        for causal in relationships["causal"]:
            prime_components.append(f"because of this, {causal['effect']}")
        
        # Join into coherent representation
        if prime_components:
            return " ".join(prime_components)
        else:
            return "someone did something"
    
    def test_decomposition(self, text: str, language: Language = Language.ENGLISH):
        """Test the decomposition engine."""
        
        result = self.decompose_sentence(text, language)
        
        print("\n📋 DECOMPOSITION SUMMARY:")
        print(f"Original: {result['original']}")
        print(f"Prime Language: {result['prime_language']}")
        print()
        
        if result["concepts"]:
            print("🧩 Concept Decompositions:")
            for role, decomp in result["concepts"].items():
                print(f"  {role}: {decomp}")
            print()
        
        if result["actions"]:
            print("⚡ Action Decomposition:")
            for action in result["actions"]:
                print(f"  - {action}")
            print()
        
        if result["relationships"]["causal"]:
            print("🔗 Causal Relationships:")
            for causal in result["relationships"]["causal"]:
                print(f"  {causal['cause']} → {causal['effect']}")
        
        return result

def test_semantic_decomposition():
    """Test the semantic decomposition engine."""
    
    print("🧠 SEMANTIC DECOMPOSITION ENGINE TEST")
    print("=" * 70)
    print()
    
    engine = SemanticDecompositionEngine()
    
    # Test cases
    test_cases = [
        "The boy kicked the ball.",
        "The man ate the food.",
        "The woman loves the child.",
        "The girl threw the book.",
        "The child slept in the house."
    ]
    
    for text in test_cases:
        print(f"🎯 Testing: {text}")
        result = engine.test_decomposition(text)
        print("-" * 60)
        print()

if __name__ == "__main__":
    test_semantic_decomposition()
