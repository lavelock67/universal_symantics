"""
EIL Realizer

Converts EIL graphs back to surface text in target languages.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

from .graph import EILGraph, EILNode, EILRelation, EILNodeType, EILRelationType

logger = logging.getLogger(__name__)


@dataclass
class RealizationResult:
    """Result of EIL realization."""
    text: str
    confidence: float
    realization_method: str
    metadata: Dict[str, Any]


class EILRealizer:
    """Realizes EIL graphs as surface text."""
    
    def __init__(self):
        """Initialize the realizer."""
        # Language-specific templates
        self.templates = {
            "en": {
                "entity": "{label}",
                "event": "{label}",
                "property": "{label}",
                "negation": "not {target}",
                "cause": "{source} because {target}",
                "if_then": "if {source} then {target}",
                "before": "{source} before {target}",
                "after": "{source} after {target}",
                "all": "all {target}",
                "some": "some {target}",
                "must": "must {target}",
                "can": "can {target}",
                "should": "should {target}",
                "in": "{source} in {target}",
                "on": "{source} on {target}",
                "near": "{source} near {target}",
                "far": "{source} far from {target}",
                "above": "{source} above {target}",
                "below": "{source} below {target}",
            },
            "es": {
                "entity": "{label}",
                "event": "{label}",
                "property": "{label}",
                "negation": "no {target}",
                "cause": "{source} porque {target}",
                "if_then": "si {source} entonces {target}",
                "before": "{source} antes de {target}",
                "after": "{source} después de {target}",
                "all": "todos {target}",
                "some": "algunos {target}",
                "must": "debe {target}",
                "can": "puede {target}",
                "should": "debería {target}",
                "in": "{source} en {target}",
                "on": "{source} en {target}",
                "near": "{source} cerca de {target}",
                "far": "{source} lejos de {target}",
                "above": "{source} arriba de {target}",
                "below": "{source} abajo de {target}",
            },
            "fr": {
                "entity": "{label}",
                "event": "{label}",
                "property": "{label}",
                "negation": "ne pas {target}",
                "cause": "{source} parce que {target}",
                "if_then": "si {source} alors {target}",
                "before": "{source} avant {target}",
                "after": "{source} après {target}",
                "all": "tous {target}",
                "some": "quelques {target}",
                "must": "doit {target}",
                "can": "peut {target}",
                "should": "devrait {target}",
                "in": "{source} dans {target}",
                "on": "{source} sur {target}",
                "near": "{source} près de {target}",
                "far": "{source} loin de {target}",
                "above": "{source} au-dessus de {target}",
                "below": "{source} en-dessous de {target}",
            }
        }
        
        # NSM prime to surface form mapping
        self.prime_surface_forms = {
            "en": {
                "I": "I",
                "YOU": "you",
                "SOMEONE": "someone",
                "PEOPLE": "people",
                "SOMETHING": "something",
                "THING": "thing",
                "BODY": "body",
                "THINK": "think",
                "KNOW": "know",
                "WANT": "want",
                "FEEL": "feel",
                "SEE": "see",
                "HEAR": "hear",
                "SAY": "say",
                "NOT": "not",
                "BECAUSE": "because",
                "IF": "if",
                "MAYBE": "maybe",
                "WHEN": "when",
                "NOW": "now",
                "BEFORE": "before",
                "AFTER": "after",
                "WHERE": "where",
                "HERE": "here",
                "ABOVE": "above",
                "BELOW": "below",
                "NEAR": "near",
                "FAR": "far",
                "INSIDE": "inside",
                "ALL": "all",
                "SOME": "some",
                "MANY": "many",
                "FEW": "few",
                "ONE": "one",
                "TWO": "two",
                "GOOD": "good",
                "BAD": "bad",
                "BIG": "big",
                "SMALL": "small",
                "DO": "do",
                "HAPPEN": "happen",
                "MOVE": "move",
                "TOUCH": "touch",
                "LIVE": "live",
                "DIE": "die",
                "THIS": "this",
                "OTHER": "other",
                "SAME": "same",
                "DIFFERENT": "different",
                "VERY": "very",
                "MORE": "more",
                "LIKE": "like",
                "WORDS": "words",
                "BE_SOMEONE": "be someone",
                "BE_SOMEWHERE": "be somewhere",
                "THERE_IS": "there is",
                "A_LONG_TIME": "a long time",
                "A_SHORT_TIME": "a short time",
                "FOR_SOME_TIME": "for some time",
                "THE_SAME": "the same",
            },
            "es": {
                "I": "yo",
                "YOU": "tú",
                "SOMEONE": "alguien",
                "PEOPLE": "gente",
                "SOMETHING": "algo",
                "THING": "cosa",
                "BODY": "cuerpo",
                "THINK": "pensar",
                "KNOW": "saber",
                "WANT": "querer",
                "FEEL": "sentir",
                "SEE": "ver",
                "HEAR": "oír",
                "SAY": "decir",
                "NOT": "no",
                "BECAUSE": "porque",
                "IF": "si",
                "MAYBE": "tal vez",
                "WHEN": "cuando",
                "NOW": "ahora",
                "BEFORE": "antes",
                "AFTER": "después",
                "WHERE": "dónde",
                "HERE": "aquí",
                "ABOVE": "arriba",
                "BELOW": "abajo",
                "NEAR": "cerca",
                "FAR": "lejos",
                "INSIDE": "dentro",
                "ALL": "todos",
                "SOME": "algunos",
                "MANY": "muchos",
                "FEW": "pocos",
                "ONE": "uno",
                "TWO": "dos",
                "GOOD": "bueno",
                "BAD": "malo",
                "BIG": "grande",
                "SMALL": "pequeño",
                "DO": "hacer",
                "HAPPEN": "pasar",
                "MOVE": "mover",
                "TOUCH": "tocar",
                "LIVE": "vivir",
                "DIE": "morir",
                "THIS": "este",
                "OTHER": "otro",
                "SAME": "mismo",
                "DIFFERENT": "diferente",
                "VERY": "muy",
                "MORE": "más",
                "LIKE": "como",
                "WORDS": "palabras",
                "BE_SOMEONE": "ser alguien",
                "BE_SOMEWHERE": "estar en algún lugar",
                "THERE_IS": "hay",
                "A_LONG_TIME": "mucho tiempo",
                "A_SHORT_TIME": "poco tiempo",
                "FOR_SOME_TIME": "por algún tiempo",
                "THE_SAME": "el mismo",
            },
            "fr": {
                "I": "je",
                "YOU": "tu",
                "SOMEONE": "quelqu'un",
                "PEOPLE": "gens",
                "SOMETHING": "quelque chose",
                "THING": "chose",
                "BODY": "corps",
                "THINK": "penser",
                "KNOW": "savoir",
                "WANT": "vouloir",
                "FEEL": "sentir",
                "SEE": "voir",
                "HEAR": "entendre",
                "SAY": "dire",
                "NOT": "ne pas",
                "BECAUSE": "parce que",
                "IF": "si",
                "MAYBE": "peut-être",
                "WHEN": "quand",
                "NOW": "maintenant",
                "BEFORE": "avant",
                "AFTER": "après",
                "WHERE": "où",
                "HERE": "ici",
                "ABOVE": "au-dessus",
                "BELOW": "en-dessous",
                "NEAR": "près",
                "FAR": "loin",
                "INSIDE": "dedans",
                "ALL": "tous",
                "SOME": "quelques",
                "MANY": "beaucoup",
                "FEW": "peu",
                "ONE": "un",
                "TWO": "deux",
                "GOOD": "bon",
                "BAD": "mauvais",
                "BIG": "grand",
                "SMALL": "petit",
                "DO": "faire",
                "HAPPEN": "arriver",
                "MOVE": "bouger",
                "TOUCH": "toucher",
                "LIVE": "vivre",
                "DIE": "mourir",
                "THIS": "ce",
                "OTHER": "autre",
                "SAME": "même",
                "DIFFERENT": "différent",
                "VERY": "très",
                "MORE": "plus",
                "LIKE": "comme",
                "WORDS": "mots",
                "BE_SOMEONE": "être quelqu'un",
                "BE_SOMEWHERE": "être quelque part",
                "THERE_IS": "il y a",
                "A_LONG_TIME": "longtemps",
                "A_SHORT_TIME": "peu de temps",
                "FOR_SOME_TIME": "pendant quelque temps",
                "THE_SAME": "le même",
            }
        }
    
    def realize(self, graph: EILGraph, target_language: str = "en") -> RealizationResult:
        """Realize EIL graph as surface text."""
        try:
            if not graph.nodes:
                return RealizationResult(
                    text="",
                    confidence=0.0,
                    realization_method="empty_graph",
                    metadata={"error": "Empty graph"}
                )
            
            # Get language-specific resources
            templates = self.templates.get(target_language, self.templates["en"])
            surface_forms = self.prime_surface_forms.get(target_language, self.prime_surface_forms["en"])
            
            # Realize nodes
            node_texts = self._realize_nodes(graph, surface_forms)
            
            # Realize relations
            relation_texts = self._realize_relations(graph, templates, surface_forms)
            
            # Combine into final text
            final_text = self._combine_texts(node_texts, relation_texts, target_language)
            
            # Calculate confidence
            confidence = self._calculate_realization_confidence(graph)
            
            return RealizationResult(
                text=final_text,
                confidence=confidence,
                realization_method="template_based",
                metadata={
                    "node_count": len(graph.nodes),
                    "relation_count": len(graph.relations),
                    "target_language": target_language
                }
            )
            
        except Exception as e:
            logger.error(f"EIL realization failed: {str(e)}")
            return RealizationResult(
                text="",
                confidence=0.0,
                realization_method="error",
                metadata={"error": str(e)}
            )
    
    def _realize_nodes(self, graph: EILGraph, surface_forms: Dict[str, str]) -> Dict[str, str]:
        """Realize individual nodes."""
        node_texts = {}
        
        for node_id, node in graph.nodes.items():
            # Get surface form for the node label
            surface_form = surface_forms.get(node.label, node.label.lower())
            
            # Apply node type specific formatting
            if node.node_type == EILNodeType.ENTITY:
                node_texts[node_id] = surface_form
            elif node.node_type == EILNodeType.EVENT:
                node_texts[node_id] = surface_form
            elif node.node_type == EILNodeType.PROPERTY:
                node_texts[node_id] = surface_form
            elif node.node_type == EILNodeType.QUANTIFIER:
                node_texts[node_id] = surface_form
            elif node.node_type == EILNodeType.MODAL:
                node_texts[node_id] = surface_form
            elif node.node_type == EILNodeType.NEGATION:
                node_texts[node_id] = surface_form
            elif node.node_type == EILNodeType.TEMPORAL:
                node_texts[node_id] = surface_form
            elif node.node_type == EILNodeType.SPATIAL:
                node_texts[node_id] = surface_form
            else:
                node_texts[node_id] = surface_form
        
        return node_texts
    
    def _realize_relations(self, graph: EILGraph, templates: Dict[str, str], surface_forms: Dict[str, str]) -> List[str]:
        """Realize relations between nodes."""
        relation_texts = []
        
        for relation in graph.relations.values():
            source_node = graph.get_node(relation.source_id)
            target_node = graph.get_node(relation.target_id)
            
            if not source_node or not target_node:
                continue
            
            # Get template for relation type
            template_key = relation.relation_type.value
            template = templates.get(template_key, "{source} {target}")
            
            # Get surface forms for nodes
            source_text = surface_forms.get(source_node.label, source_node.label.lower())
            target_text = surface_forms.get(target_node.label, target_node.label.lower())
            
            # Apply template
            relation_text = template.format(source=source_text, target=target_text)
            relation_texts.append(relation_text)
        
        return relation_texts
    
    def _combine_texts(self, node_texts: Dict[str, str], relation_texts: List[str], language: str) -> str:
        """Combine node and relation texts into final text."""
        # Simple combination strategy
        all_texts = list(node_texts.values()) + relation_texts
        
        if not all_texts:
            return ""
        
        # Join with appropriate separators based on language
        if language == "en":
            separator = " "
        elif language == "es":
            separator = " "
        elif language == "fr":
            separator = " "
        else:
            separator = " "
        
        # Remove duplicates and join
        unique_texts = list(dict.fromkeys(all_texts))  # Preserve order
        final_text = separator.join(unique_texts)
        
        # Basic capitalization for English
        if language == "en" and final_text:
            final_text = final_text[0].upper() + final_text[1:]
        
        return final_text
    
    def _calculate_realization_confidence(self, graph: EILGraph) -> float:
        """Calculate confidence for the realization."""
        if not graph.nodes:
            return 0.0
        
        # Average confidence of all nodes
        node_confidence = sum(node.confidence for node in graph.nodes.values()) / len(graph.nodes)
        
        # Relation confidence if available
        relation_confidence = 0.0
        if graph.relations:
            relation_confidence = sum(rel.confidence for rel in graph.relations.values()) / len(graph.relations)
        
        # Weighted average
        if graph.relations:
            return (node_confidence * 0.6) + (relation_confidence * 0.4)
        else:
            return node_confidence
    
    def realize_with_neural(self, graph: EILGraph, target_language: str = "en") -> RealizationResult:
        """Realize EIL graph using neural generation (placeholder for future implementation)."""
        # This would integrate with a neural text generation model
        # For now, fall back to template-based realization
        logger.warning("Neural realization not yet implemented, falling back to templates")
        return self.realize(graph, target_language)

