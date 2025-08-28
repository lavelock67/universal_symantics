#!/usr/bin/env python3
"""
Improved Entity Extraction

This fixes the entity extraction issues in our knowledge graph integration
by implementing proper NER and filtering out common words.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

import spacy
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExtractedEntity:
    """Represents an extracted entity with context."""
    text: str
    start: int
    end: int
    label: str
    confidence: float
    context: str

class ImprovedEntityExtractor:
    """Improved entity extraction with proper NER and filtering."""
    
    def __init__(self):
        # Load SpaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  SpaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Common words to filter out
        self.common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "down", "out", "off", "over", "under", "into", "onto",
            "this", "that", "these", "those", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "can", "may", "might", "must", "shall", "am", "is", "are", "was", "were"
        }
        
        # Entity type mappings for better semantic understanding
        self.entity_type_mappings = {
            "PERSON": ["PERSON"],
            "PLACE": ["GPE", "LOC", "FAC"],
            "ORGANIZATION": ["ORG"],
            "EVENT": ["EVENT"],
            "OBJECT": ["PRODUCT", "WORK_OF_ART"],
            "DATE": ["DATE", "TIME"],
            "QUANTITY": ["QUANTITY", "MONEY", "PERCENT"],
            "LANGUAGE": ["LANGUAGE"]
        }
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using SpaCy NER with filtering."""
        
        # Process text with SpaCy
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            # Filter out common words and low-confidence entities
            if self._should_include_entity(ent):
                entity = ExtractedEntity(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_,
                    confidence=self._calculate_confidence(ent),
                    context=self._extract_context(text, ent.start_char, ent.end_char)
                )
                entities.append(entity)
        
        return entities
    
    def _should_include_entity(self, ent) -> bool:
        """Determine if an entity should be included."""
        
        # Filter out common words
        if ent.text.lower() in self.common_words:
            return False
        
        # Filter out very short entities (likely false positives)
        if len(ent.text.strip()) < 3:
            return False
        
        # Filter out entities that are just numbers or punctuation
        if re.match(r'^[\d\s\W]+$', ent.text):
            return False
        
        # Include all named entities from SpaCy
        return True
    
    def _calculate_confidence(self, ent) -> float:
        """Calculate confidence score for an entity."""
        
        # Base confidence on entity length and type
        base_confidence = 0.7
        
        # Longer entities are more likely to be correct
        if len(ent.text) > 5:
            base_confidence += 0.2
        
        # Certain entity types are more reliable
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around an entity."""
        
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        return text[context_start:context_end].strip()
    
    def get_entity_type(self, entity: ExtractedEntity) -> str:
        """Get semantic entity type for knowledge graph integration."""
        
        for semantic_type, spacy_labels in self.entity_type_mappings.items():
            if entity.label in spacy_labels:
                return semantic_type
        
        return "UNKNOWN"
    
    def extract_entities_for_grounding(self, text: str) -> List[str]:
        """Extract entity texts for knowledge graph grounding."""
        
        entities = self.extract_entities(text)
        return [entity.text for entity in entities]
    
    def get_entity_contexts(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Get entities with their contexts for better disambiguation."""
        
        entities = self.extract_entities(text)
        contexts = {}
        
        for entity in entities:
            contexts[entity.text] = {
                "label": entity.label,
                "semantic_type": self.get_entity_type(entity),
                "confidence": entity.confidence,
                "context": entity.context,
                "position": (entity.start, entity.end)
            }
        
        return contexts

class EnhancedSemanticDecompositionEngine:
    """Enhanced semantic decomposition with improved entity extraction."""
    
    def __init__(self):
        from semantic_decomposition_engine import SemanticDecompositionEngine
        from knowledge_graph_integrator import KnowledgeGraphIntegrator
        
        self.base_engine = SemanticDecompositionEngine()
        self.kg_integrator = KnowledgeGraphIntegrator()
        self.entity_extractor = ImprovedEntityExtractor()
    
    def decompose_with_improved_entities(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Decompose text with improved entity extraction and knowledge graph grounding."""
        
        print(f"🔍 Enhanced Decomposition with Improved Entities: '{text}'")
        print("-" * 70)
        
        # Step 1: Improved entity extraction
        print("📋 Step 1: Entity Extraction")
        entities = self.entity_extractor.extract_entities(text)
        entity_contexts = self.entity_extractor.get_entity_contexts(text)
        
        print(f"  Found {len(entities)} entities:")
        for entity in entities:
            print(f"    - '{entity.text}' ({entity.label}) - confidence: {entity.confidence:.2f}")
        
        # Step 2: Basic semantic decomposition
        print("\n📐 Step 2: Semantic Decomposition")
        base_decomposition = self.base_engine.decompose_sentence(text, language)
        print(f"  Base decomposition completed")
        
        # Step 3: Ground entities in knowledge graph
        print("\n🔗 Step 3: Knowledge Graph Grounding")
        grounded_entities = []
        for entity in entities:
            print(f"  Grounding: '{entity.text}' ({entity.label})")
            wikidata_entity = self.kg_integrator.ground_entity(entity.text, text)
            if wikidata_entity:
                grounded_entities.append(wikidata_entity)
                print(f"    ✅ Grounded to: {wikidata_entity.label} ({wikidata_entity.id})")
            else:
                print(f"    ❌ Could not ground")
        
        # Step 4: Create enhanced interlingua graph
        print("\n🧠 Step 4: Enhanced Interlingua Graph")
        enhanced_graph = self.kg_integrator.create_enhanced_interlingua(
            base_decomposition, grounded_entities
        )
        
        # Step 5: Generate JSON-LD representation
        json_ld = self.kg_integrator.to_json_ld(enhanced_graph)
        
        result = {
            "original_text": text,
            "extracted_entities": entities,
            "entity_contexts": entity_contexts,
            "base_decomposition": base_decomposition,
            "grounded_entities": grounded_entities,
            "enhanced_interlingua": enhanced_graph,
            "json_ld": json_ld
        }
        
        return result

def demonstrate_improved_entity_extraction():
    """Demonstrate the improved entity extraction."""
    
    print("🔍 IMPROVED ENTITY EXTRACTION DEMONSTRATION")
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
            result = enhanced_engine.decompose_with_improved_entities(text)
            
            print(f"\n📊 RESULTS:")
            print(f"  Extracted Entities: {len(result['extracted_entities'])}")
            for entity in result['extracted_entities']:
                semantic_type = enhanced_engine.entity_extractor.get_entity_type(entity)
                print(f"    - '{entity.text}' ({entity.label} → {semantic_type}) - {entity.confidence:.2f}")
            
            print(f"\n🔗 Knowledge Graph Grounding:")
            print(f"  Grounded Entities: {len(result['grounded_entities'])}")
            for entity in result['grounded_entities']:
                print(f"    - {entity.label} ({entity.entity_type}) - {entity.id}")
            
            print(f"\n🧠 Enhanced Interlingua Graph:")
            print(f"  Type: {result['enhanced_interlingua']['type']}")
            print(f"  Entities: {len(result['enhanced_interlingua']['knowledge_graph_entities'])}")
            print(f"  Relationships: {len(result['enhanced_interlingua']['grounded_relationships'])}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 50)
        print()

def compare_extraction_methods():
    """Compare old vs new entity extraction methods."""
    
    print("⚖️ COMPARISON: Old vs New Entity Extraction")
    print("=" * 70)
    print()
    
    test_text = "The boy kicked the ball in Paris."
    
    # Old method (simple capitalization)
    print("📦 OLD METHOD (Simple Capitalization):")
    words = test_text.split()
    old_entities = []
    for word in words:
        clean_word = word.strip('.,!?;:')
        if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
            old_entities.append(clean_word)
    
    print(f"  Entities found: {old_entities}")
    print(f"  Problems: Picks up 'The', no semantic understanding")
    
    # New method (SpaCy NER)
    print("\n🔍 NEW METHOD (SpaCy NER):")
    extractor = ImprovedEntityExtractor()
    new_entities = extractor.extract_entities(test_text)
    
    print(f"  Entities found: {[e.text for e in new_entities]}")
    print(f"  Types: {[e.label for e in new_entities]}")
    print(f"  Advantages: Semantic understanding, filtered common words")
    
    print("\n✅ IMPROVEMENTS:")
    print("  - Filters out common words like 'The', 'A', 'An'")
    print("  - Provides semantic entity types (PERSON, GPE, ORG)")
    print("  - Includes confidence scores")
    print("  - Extracts context for better disambiguation")
    print("  - Handles multi-word entities properly")

if __name__ == "__main__":
    demonstrate_improved_entity_extraction()
    print()
    compare_extraction_methods()
