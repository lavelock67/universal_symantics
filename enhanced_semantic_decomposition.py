#!/usr/bin/env python3
"""
Enhanced Semantic Decomposition Engine

This extends our semantic decomposition to handle more sentence types
including passive voice, questions, and complex structures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

import spacy
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SentenceStructure:
    """Enhanced sentence structure with more detailed analysis."""
    subject: Optional[str]
    predicate: Optional[str]
    object: Optional[str]
    indirect_object: Optional[str]
    modifiers: List[str]
    temporal: Optional[str]
    spatial: Optional[str]
    voice: str  # "active" or "passive"
    mood: str   # "declarative", "interrogative", "imperative"
    tense: str  # "present", "past", "future"

class EnhancedSemanticDecompositionEngine:
    """Enhanced semantic decomposition engine with broader sentence coverage."""
    
    def __init__(self):
        # Load SpaCy for better parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  SpaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Extended concept decompositions
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
            "cat": {
                "type": "animal",
                "decomposition": [
                    "a living thing of one kind",
                    "this thing is not a person",
                    "this thing can move",
                    "this thing has four legs",
                    "this thing is small"
                ]
            },
            "dog": {
                "type": "animal",
                "decomposition": [
                    "a living thing of one kind",
                    "this thing is not a person",
                    "this thing can move",
                    "this thing has four legs",
                    "this thing can make sounds"
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
            "mat": {
                "type": "object",
                "decomposition": [
                    "a thing of one kind",
                    "this thing is flat",
                    "this thing is on the ground",
                    "people can sit on this thing"
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
            "plays": {
                "type": "object",
                "decomposition": [
                    "things of one kind",
                    "these things have many words",
                    "people can see these things",
                    "because of this, people can think about many things"
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
            "kicked": {
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
            "sat": {
                "type": "action",
                "decomposition": [
                    "someone did something",
                    "this someone was not moving",
                    "this someone was on something",
                    "this happened for some time"
                ]
            },
            "wrote": {
                "type": "action",
                "decomposition": [
                    "someone did something",
                    "this someone made many words",
                    "because of this, other people can see these words",
                    "because of this, other people can know many things"
                ]
            },
            "was_born": {
                "type": "action",
                "decomposition": [
                    "something happened to someone",
                    "this someone began to live",
                    "this happened in one place",
                    "this happened at one time"
                ]
            },
            "founded": {
                "type": "action",
                "decomposition": [
                    "someone did something",
                    "this someone made something new",
                    "because of this, this something began to exist",
                    "this someone was the first person to do this"
                ]
            },
            "released": {
                "type": "action",
                "decomposition": [
                    "someone did something",
                    "this someone made something available to many people",
                    "because of this, many people could get this something",
                    "this happened at one time"
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
    
    def parse_sentence_structure(self, text: str) -> SentenceStructure:
        """Parse sentence structure using SpaCy dependency parsing."""
        
        doc = self.nlp(text)
        
        # Initialize structure
        structure = SentenceStructure(
            subject=None,
            predicate=None,
            object=None,
            indirect_object=None,
            modifiers=[],
            temporal=None,
            spatial=None,
            voice="active",
            mood="declarative",
            tense="present"
        )
        
        # Determine mood
        if text.strip().endswith('?'):
            structure.mood = "interrogative"
        elif text.strip().endswith('!'):
            structure.mood = "imperative"
        
        # Find subject, predicate, and object using dependency parsing
        for token in doc:
            # Find subject (nsubj or nsubjpass)
            if token.dep_ in ["nsubj", "nsubjpass"]:
                structure.subject = token.text.lower()
                if token.dep_ == "nsubjpass":
                    structure.voice = "passive"
            
            # Find predicate (root verb)
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                structure.predicate = token.text.lower()
            
            # Find direct object (dobj)
            if token.dep_ == "dobj":
                structure.object = token.text.lower()
            
            # Find indirect object (iobj)
            if token.dep_ == "iobj":
                structure.indirect_object = token.text.lower()
            
            # Find modifiers (amod, advmod)
            if token.dep_ in ["amod", "advmod"]:
                structure.modifiers.append(token.text.lower())
            
            # Find temporal and spatial information
            if token.dep_ in ["nmod", "prep"]:
                if any(word in token.text.lower() for word in ["when", "time", "year", "date"]):
                    structure.temporal = token.text.lower()
                elif any(word in token.text.lower() for word in ["where", "place", "location", "in", "at", "on"]):
                    structure.spatial = token.text.lower()
        
        # Determine tense
        for token in doc:
            if token.pos_ == "VERB":
                if token.tag_ in ["VBD", "VBN"]:
                    structure.tense = "past"
                elif token.tag_ in ["VBG"]:
                    structure.tense = "present"
                elif token.tag_ in ["VBP", "VBZ"]:
                    structure.tense = "present"
                break
        
        return structure
    
    def decompose_sentence(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Decompose a sentence into semantic prime combinations."""
        
        print(f"🔍 Enhanced Decomposition: '{text}'")
        print("-" * 60)
        
        # Step 1: Parse sentence structure
        structure = self.parse_sentence_structure(text)
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
    
    def _decompose_concepts(self, text: str, structure: SentenceStructure) -> Dict[str, List[str]]:
        """Decompose complex concepts into prime combinations."""
        
        decomposed = {}
        
        # Decompose subject
        if structure.subject and structure.subject in self.concept_decompositions:
            decomposed["subject"] = self.concept_decompositions[structure.subject]["decomposition"]
        
        # Decompose object
        if structure.object and structure.object in self.concept_decompositions:
            decomposed["object"] = self.concept_decompositions[structure.object]["decomposition"]
        
        # Decompose indirect object
        if structure.indirect_object and structure.indirect_object in self.concept_decompositions:
            decomposed["indirect_object"] = self.concept_decompositions[structure.indirect_object]["decomposition"]
        
        return decomposed
    
    def _decompose_actions(self, text: str, structure: SentenceStructure) -> List[str]:
        """Decompose complex actions into prime combinations."""
        
        if not structure.predicate:
            return []
        
        predicate = structure.predicate
        
        # Check specific action decompositions
        if predicate in self.concept_decompositions:
            return self.concept_decompositions[predicate]["decomposition"]
        
        # Handle passive voice
        if structure.voice == "passive":
            return [
                "something happened to someone",
                "because of this, something changed",
                "this happened at one time"
            ]
        
        # Default action decomposition
        return [
            "someone did something",
            "because of this, something happened"
        ]
    
    def _build_semantic_relationships(self, structure: SentenceStructure, 
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
        
        # Temporal and spatial
        if structure.temporal:
            relationships["temporal"] = structure.temporal
        if structure.spatial:
            relationships["spatial"] = structure.spatial
        
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
        
        # Add temporal/spatial information
        if relationships["temporal"]:
            prime_components.append(f"this happened {relationships['temporal']}")
        if relationships["spatial"]:
            prime_components.append(f"this happened {relationships['spatial']}")
        
        # Causal connections
        for causal in relationships["causal"]:
            prime_components.append(f"because of this, {causal['effect']}")
        
        # Join into coherent representation
        if prime_components:
            return " ".join(prime_components)
        else:
            return "someone did something"

def demonstrate_enhanced_decomposition():
    """Demonstrate the enhanced semantic decomposition."""
    
    print("🧠 ENHANCED SEMANTIC DECOMPOSITION DEMONSTRATION")
    print("=" * 70)
    print()
    
    enhanced_engine = EnhancedSemanticDecompositionEngine()
    
    # Test cases that were problematic before
    test_cases = [
        "The boy kicked the ball in Paris.",
        "Einstein was born in Germany.",
        "The Eiffel Tower is in France.",
        "Shakespeare wrote many plays.",
        "The cat sat on the mat.",
        "Apple Inc. was founded by Steve Jobs in California.",
        "The Great Wall of China is a famous landmark.",
        "Microsoft released Windows 11 in 2021."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"🎯 EXAMPLE {i}: '{text}'")
        print("-" * 50)
        
        try:
            result = enhanced_engine.decompose_sentence(text)
            
            print(f"\n📊 RESULTS:")
            print(f"  Structure: {result['structure']}")
            print(f"  Concepts: {len(result['concepts'])}")
            print(f"  Actions: {len(result['actions'])}")
            print(f"  Prime Language: {result['prime_language']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 50)
        print()

if __name__ == "__main__":
    demonstrate_enhanced_decomposition()
