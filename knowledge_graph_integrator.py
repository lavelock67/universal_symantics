#!/usr/bin/env python3
"""
Knowledge Graph Integrator

This implements the first critical step of the enhanced universal translator:
integrating Wikidata for entity grounding and enhanced semantic representation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

import requests
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import quote
import time

@dataclass
class WikidataEntity:
    """Represents a Wikidata entity with its properties."""
    id: str
    label: str
    description: str
    entity_type: str
    properties: Dict[str, Any]
    aliases: List[str]

class KnowledgeGraphIntegrator:
    """Integrates Wikidata for entity grounding and enhanced semantic representation."""
    
    def __init__(self):
        self.wikidata_api_base = "https://www.wikidata.org/w/api.php"
        self.sparql_endpoint = "https://query.wikidata.org/sparql"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UniversalTranslator/1.0 (https://github.com/your-repo)'
        })
        
        # Entity type mappings for better semantic grounding
        self.entity_type_mappings = {
            "PERSON": ["Q5"],  # human
            "PLACE": ["Q532", "Q2221906"],  # populated place, geographical object
            "ORGANIZATION": ["Q43229"],  # organization
            "EVENT": ["Q1190554"],  # occurrence
            "CONCEPT": ["Q151885"],  # concept
            "OBJECT": ["Q223557"],  # physical object
            "ANIMAL": ["Q729"],  # animal
            "PLANT": ["Q756"],  # plant
            "FOOD": ["Q2095"],  # food
            "SPORT": ["Q349"],  # sport
            "VEHICLE": ["Q42889"],  # vehicle
            "BUILDING": ["Q41176"],  # building
        }
    
    def search_entity(self, query: str, language: str = "en") -> List[WikidataEntity]:
        """Search for entities in Wikidata."""
        
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': language,
            'type': 'item',
            'search': query,
            'limit': 5
        }
        
        try:
            response = self.session.get(self.wikidata_api_base, params=params)
            response.raise_for_status()
            data = response.json()
            
            entities = []
            for item in data.get('search', []):
                entity = WikidataEntity(
                    id=item['id'],
                    label=item['label'],
                    description=item.get('description', ''),
                    entity_type=self._determine_entity_type(item),
                    properties={},
                    aliases=item.get('aliases', [])
                )
                entities.append(entity)
            
            return entities
        
        except Exception as e:
            print(f"Error searching Wikidata: {e}")
            return []
    
    def get_entity_details(self, entity_id: str) -> Optional[WikidataEntity]:
        """Get detailed information about a Wikidata entity."""
        
        # Get basic entity info
        params = {
            'action': 'wbgetentities',
            'format': 'json',
            'ids': entity_id,
            'languages': 'en',
            'props': 'labels|descriptions|aliases|claims'
        }
        
        try:
            response = self.session.get(self.wikidata_api_base, params=params)
            response.raise_for_status()
            data = response.json()
            
            if entity_id not in data.get('entities', {}):
                return None
            
            entity_data = data['entities'][entity_id]
            
            # Extract properties from claims
            properties = self._extract_properties(entity_data.get('claims', {}))
            
            entity = WikidataEntity(
                id=entity_id,
                label=entity_data.get('labels', {}).get('en', {}).get('value', ''),
                description=entity_data.get('descriptions', {}).get('en', {}).get('value', ''),
                entity_type=self._determine_entity_type(entity_data),
                properties=properties,
                aliases=[alias['value'] for alias in entity_data.get('aliases', {}).get('en', [])]
            )
            
            return entity
        
        except Exception as e:
            print(f"Error getting entity details: {e}")
            return None
    
    def _extract_properties(self, claims: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key properties from Wikidata claims."""
        
        properties = {}
        
        # Common property mappings
        property_mappings = {
            'P31': 'instance_of',  # instance of
            'P279': 'subclass_of',  # subclass of
            'P17': 'country',  # country
            'P131': 'located_in',  # located in
            'P19': 'place_of_birth',  # place of birth
            'P20': 'place_of_death',  # place of death
            'P27': 'country_of_citizenship',  # country of citizenship
            'P106': 'occupation',  # occupation
            'P569': 'date_of_birth',  # date of birth
            'P570': 'date_of_death',  # date of death
        }
        
        for claim_id, claim_list in claims.items():
            if claim_id in property_mappings:
                property_name = property_mappings[claim_id]
                values = []
                
                for claim in claim_list:
                    if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                        value = claim['mainsnak']['datavalue']['value']
                        if isinstance(value, dict) and 'id' in value:
                            # It's a reference to another entity
                            values.append(value['id'])
                        else:
                            values.append(value)
                
                if values:
                    properties[property_name] = values
        
        return properties
    
    def _determine_entity_type(self, entity_data: Dict[str, Any]) -> str:
        """Determine the semantic type of an entity based on its properties."""
        
        # Check instance of claims
        claims = entity_data.get('claims', {})
        instance_of = claims.get('P31', [])
        
        for claim in instance_of:
            if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                value_id = claim['mainsnak']['datavalue']['value']['id']
                
                # Map to our entity types
                for entity_type, type_ids in self.entity_type_mappings.items():
                    if value_id in type_ids:
                        return entity_type
        
        return "UNKNOWN"
    
    def ground_entity(self, text: str, context: str = "") -> Optional[WikidataEntity]:
        """Ground a text entity to a Wikidata entity."""
        
        # Search for the entity
        entities = self.search_entity(text)
        
        if not entities:
            return None
        
        # For now, return the first (best) match
        # In a full implementation, you'd use context to disambiguate
        best_entity = entities[0]
        
        # Get full details
        detailed_entity = self.get_entity_details(best_entity.id)
        
        return detailed_entity if detailed_entity else best_entity
    
    def create_enhanced_interlingua(self, semantic_decomposition: Dict[str, Any], 
                                  entities: List[WikidataEntity]) -> Dict[str, Any]:
        """Create an enhanced interlingua graph with knowledge graph grounding."""
        
        enhanced_graph = {
            "type": "EnhancedInterlinguaGraph",
            "version": "1.0",
            "semantic_decomposition": semantic_decomposition,
            "knowledge_graph_entities": [],
            "grounded_relationships": [],
            "metadata": {
                "created_at": time.time(),
                "source": "KnowledgeGraphIntegrator"
            }
        }
        
        # Add grounded entities
        for entity in entities:
            grounded_entity = {
                "wikidata_id": entity.id,
                "label": entity.label,
                "description": entity.description,
                "entity_type": entity.entity_type,
                "properties": entity.properties,
                "aliases": entity.aliases
            }
            enhanced_graph["knowledge_graph_entities"].append(grounded_entity)
        
        # Create grounded relationships
        for entity in entities:
            if entity.entity_type == "PERSON":
                enhanced_graph["grounded_relationships"].append({
                    "type": "agent",
                    "entity_id": entity.id,
                    "nsm_representation": "someone of one kind"
                })
            elif entity.entity_type == "PLACE":
                enhanced_graph["grounded_relationships"].append({
                    "type": "location",
                    "entity_id": entity.id,
                    "nsm_representation": "this place"
                })
            elif entity.entity_type == "OBJECT":
                enhanced_graph["grounded_relationships"].append({
                    "type": "patient",
                    "entity_id": entity.id,
                    "nsm_representation": "this thing"
                })
        
        return enhanced_graph
    
    def to_json_ld(self, enhanced_graph: Dict[str, Any]) -> str:
        """Convert enhanced interlingua graph to JSON-LD format."""
        
        json_ld = {
            "@context": {
                "@vocab": "https://schema.org/",
                "nsm": "https://universal-translator.org/nsm/",
                "wikidata": "https://www.wikidata.org/entity/",
                "interlingua": "https://universal-translator.org/interlingua/"
            },
            "@type": "EnhancedInterlinguaGraph",
            "@id": f"interlingua:graph_{int(time.time())}",
            "semanticDecomposition": enhanced_graph["semantic_decomposition"],
            "knowledgeGraphEntities": [],
            "groundedRelationships": enhanced_graph["grounded_relationships"]
        }
        
        # Add entities as JSON-LD nodes
        for entity in enhanced_graph["knowledge_graph_entities"]:
            entity_node = {
                "@type": "Thing",
                "@id": f"wikidata:{entity['wikidata_id']}",
                "name": entity["label"],
                "description": entity["description"],
                "entityType": entity["entity_type"]
            }
            json_ld["knowledgeGraphEntities"].append(entity_node)
        
        return json.dumps(json_ld, indent=2)

class EnhancedSemanticDecompositionEngine:
    """Enhanced semantic decomposition engine with knowledge graph integration."""
    
    def __init__(self):
        from semantic_decomposition_engine import SemanticDecompositionEngine
        self.base_engine = SemanticDecompositionEngine()
        self.kg_integrator = KnowledgeGraphIntegrator()
    
    def decompose_with_knowledge_graph(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Decompose text with knowledge graph grounding."""
        
        print(f"🔍 Enhanced Decomposition: '{text}'")
        print("-" * 60)
        
        # Step 1: Basic semantic decomposition
        base_decomposition = self.base_engine.decompose_sentence(text, language)
        print(f"📐 Base decomposition completed")
        
        # Step 2: Extract potential entities for grounding
        entities_to_ground = self._extract_entities_for_grounding(text)
        print(f"🔗 Entities to ground: {entities_to_ground}")
        
        # Step 3: Ground entities in knowledge graph
        grounded_entities = []
        for entity_text in entities_to_ground:
            entity = self.kg_integrator.ground_entity(entity_text, text)
            if entity:
                grounded_entities.append(entity)
                print(f"✅ Grounded: '{entity_text}' → {entity.label} ({entity.id})")
            else:
                print(f"❌ Could not ground: '{entity_text}'")
        
        # Step 4: Create enhanced interlingua graph
        enhanced_graph = self.kg_integrator.create_enhanced_interlingua(
            base_decomposition, grounded_entities
        )
        
        # Step 5: Generate JSON-LD representation
        json_ld = self.kg_integrator.to_json_ld(enhanced_graph)
        
        result = {
            "original_text": text,
            "base_decomposition": base_decomposition,
            "grounded_entities": grounded_entities,
            "enhanced_interlingua": enhanced_graph,
            "json_ld": json_ld
        }
        
        return result
    
    def _extract_entities_for_grounding(self, text: str) -> List[str]:
        """Extract potential entities from text for knowledge graph grounding."""
        
        # Simple entity extraction - in a full implementation, you'd use NER
        words = text.split()
        entities = []
        
        # Look for capitalized words (potential named entities)
        for word in words:
            # Remove punctuation
            clean_word = word.strip('.,!?;:')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                entities.append(clean_word)
        
        return entities

def demonstrate_knowledge_graph_integration():
    """Demonstrate the knowledge graph integration."""
    
    print("🧠 KNOWLEDGE GRAPH INTEGRATION DEMONSTRATION")
    print("=" * 70)
    print()
    
    enhanced_engine = EnhancedSemanticDecompositionEngine()
    
    # Test cases with named entities
    test_cases = [
        "The boy kicked the ball in Paris.",
        "Einstein was born in Germany.",
        "The Eiffel Tower is in France.",
        "Shakespeare wrote many plays.",
        "The cat sat on the mat."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"🎯 EXAMPLE {i}: '{text}'")
        print("-" * 50)
        
        try:
            result = enhanced_engine.decompose_with_knowledge_graph(text)
            
            print(f"\n📊 RESULTS:")
            print(f"  Grounded Entities: {len(result['grounded_entities'])}")
            for entity in result['grounded_entities']:
                print(f"    - {entity.label} ({entity.entity_type}) - {entity.id}")
            
            print(f"\n🔗 Enhanced Interlingua Graph:")
            print(f"  Type: {result['enhanced_interlingua']['type']}")
            print(f"  Entities: {len(result['enhanced_interlingua']['knowledge_graph_entities'])}")
            print(f"  Relationships: {len(result['enhanced_interlingua']['grounded_relationships'])}")
            
            print(f"\n📄 JSON-LD Preview:")
            json_ld_preview = result['json_ld'][:200] + "..." if len(result['json_ld']) > 200 else result['json_ld']
            print(f"  {json_ld_preview}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 50)
        print()

if __name__ == "__main__":
    demonstrate_knowledge_graph_integration()
