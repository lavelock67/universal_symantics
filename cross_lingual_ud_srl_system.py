#!/usr/bin/env python3
"""
Cross-Lingual UD + SRL System

This extends our UD + SRL integration to support multiple languages
for true universal translation capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

import spacy
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class CrossLingualSemanticRole:
    """Represents a semantic role with cross-lingual information."""
    role: str
    text: str
    start: int
    end: int
    confidence: float
    entity_type: Optional[str] = None
    language: str = "en"
    universal_role: str = ""  # Universal semantic role across languages

@dataclass
class CrossLingualUDNode:
    """Represents a Universal Dependencies node with cross-lingual features."""
    token: str
    pos: str
    dep: str
    head: int
    index: int
    lemma: str
    features: Dict[str, str]
    language: str = "en"
    universal_pos: str = ""  # Universal POS tag
    universal_dep: str = ""  # Universal dependency relation

class CrossLingualUDSRLSystem:
    """Cross-lingual UD + SRL system for universal translation."""
    
    def __init__(self):
        # Language-specific SpaCy models
        self.nlp_models = {}
        self.supported_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
        
        # Load models for supported languages
        self._load_language_models()
        
        # Cross-lingual SRL patterns
        self.universal_srl_patterns = self._initialize_universal_srl_patterns()
        
        # Language-specific SRL patterns
        self.language_specific_patterns = self._initialize_language_specific_patterns()
        
        # Universal dependency mappings
        self.universal_dep_mappings = self._initialize_universal_dep_mappings()
        
        # Cross-lingual concept mappings
        self.cross_lingual_concepts = self._initialize_cross_lingual_concepts()
    
    def _load_language_models(self):
        """Load SpaCy models for supported languages."""
        
        language_model_map = {
            "en": "en_core_web_sm",
            "es": "es_core_news_sm", 
            "fr": "fr_core_news_sm",
            "de": "de_core_news_sm",
            "it": "it_core_news_sm",
            "pt": "pt_core_news_sm",
            "ru": "ru_core_news_sm",
            "zh": "zh_core_web_sm",
            "ja": "ja_core_news_sm",
            "ko": "ko_core_news_sm"
        }
        
        for lang_code, model_name in language_model_map.items():
            try:
                self.nlp_models[lang_code] = spacy.load(model_name)
                print(f"✅ Loaded {model_name} for {lang_code}")
            except OSError:
                print(f"⚠️  Model {model_name} not found for {lang_code}")
                # For now, fall back to English model
                if lang_code != "en":
                    try:
                        self.nlp_models[lang_code] = spacy.load("en_core_web_sm")
                        print(f"  Using English model as fallback for {lang_code}")
                    except OSError:
                        print(f"  ❌ No model available for {lang_code}")
    
    def _initialize_universal_srl_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize universal SRL patterns that work across languages."""
        
        return {
            "AGENT": {
                "universal_patterns": [
                    {"dep": "nsubj", "pos": ["NOUN", "PROPN", "PRON"]},
                    {"dep": "nsubjpass", "pos": ["NOUN", "PROPN", "PRON"]},
                    {"dep": "agent", "pos": ["NOUN", "PROPN", "PRON"]}
                ],
                "cross_lingual_indicators": ["by", "from", "of", "por", "de", "par", "von", "durch"],
                "exclusive_with": ["PATIENT", "THEME"]
            },
            "PATIENT": {
                "universal_patterns": [
                    {"dep": "dobj", "pos": ["NOUN", "PROPN", "PRON"]},
                    {"dep": "nsubjpass", "pos": ["NOUN", "PROPN", "PRON"]},
                    {"dep": "pobj", "pos": ["NOUN", "PROPN", "PRON"]}
                ],
                "cross_lingual_indicators": ["to", "for", "at", "a", "para", "pour", "zu", "für"],
                "exclusive_with": ["AGENT"]
            },
            "THEME": {
                "universal_patterns": [
                    {"dep": "dobj", "pos": ["NOUN", "PROPN", "PRON"]},
                    {"dep": "attr", "pos": ["NOUN", "PROPN", "PRON"]},
                    {"dep": "pobj", "pos": ["NOUN", "PROPN", "PRON"]}
                ],
                "cross_lingual_indicators": ["about", "of", "concerning", "sobre", "de", "sur", "über"],
                "exclusive_with": ["AGENT"]
            },
            "GOAL": {
                "universal_patterns": [
                    {"dep": "iobj", "pos": ["NOUN", "PROPN", "PRON"]},
                    {"dep": "pobj", "pos": ["NOUN", "PROPN", "PRON"]}
                ],
                "cross_lingual_indicators": ["to", "toward", "for", "a", "para", "vers", "zu", "nach"],
                "exclusive_with": ["AGENT", "PATIENT"]
            },
            "LOCATION": {
                "universal_patterns": [
                    {"dep": "pobj", "pos": ["NOUN", "PROPN"]},
                    {"dep": "nmod", "pos": ["NOUN", "PROPN"]}
                ],
                "cross_lingual_indicators": ["in", "at", "on", "near", "en", "dans", "sur", "in", "auf"],
                "exclusive_with": ["AGENT", "PATIENT", "THEME"]
            },
            "TIME": {
                "universal_patterns": [
                    {"dep": "nmod", "pos": ["NOUN", "PROPN"]},
                    {"dep": "advmod", "pos": ["ADV"]}
                ],
                "cross_lingual_indicators": ["when", "time", "year", "date", "cuando", "quand", "wann", "quando"],
                "exclusive_with": ["AGENT", "PATIENT", "THEME", "LOCATION"]
            },
            "INSTRUMENT": {
                "universal_patterns": [
                    {"dep": "pobj", "pos": ["NOUN", "PROPN"]},
                    {"dep": "nmod", "pos": ["NOUN", "PROPN"]}
                ],
                "cross_lingual_indicators": ["with", "using", "by", "con", "avec", "mit", "com"],
                "exclusive_with": ["AGENT", "PATIENT", "THEME"]
            },
            "MANNER": {
                "universal_patterns": [
                    {"dep": "advmod", "pos": ["ADV", "ADJ"]},
                    {"dep": "amod", "pos": ["ADJ"]}
                ],
                "cross_lingual_indicators": ["quickly", "slowly", "carefully", "rápidamente", "lentement", "schnell", "velocemente"],
                "exclusive_with": ["AGENT", "PATIENT", "THEME", "GOAL"]
            }
        }
    
    def _initialize_language_specific_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language-specific SRL patterns."""
        
        return {
            "es": {  # Spanish
                "AGENT": {
                    "additional_patterns": [
                        {"dep": "nsubj", "pos": ["NOUN", "PROPN", "PRON"]},
                        {"dep": "agent", "pos": ["NOUN", "PROPN", "PRON"]}
                    ],
                    "language_indicators": ["por", "de parte de", "mediante"]
                },
                "PATIENT": {
                    "additional_patterns": [
                        {"dep": "dobj", "pos": ["NOUN", "PROPN", "PRON"]},
                        {"dep": "iobj", "pos": ["NOUN", "PROPN", "PRON"]}
                    ],
                    "language_indicators": ["a", "para", "hacia"]
                }
            },
            "fr": {  # French
                "AGENT": {
                    "additional_patterns": [
                        {"dep": "nsubj", "pos": ["NOUN", "PROPN", "PRON"]},
                        {"dep": "agent", "pos": ["NOUN", "PROPN", "PRON"]}
                    ],
                    "language_indicators": ["par", "de la part de", "grâce à"]
                },
                "PATIENT": {
                    "additional_patterns": [
                        {"dep": "dobj", "pos": ["NOUN", "PROPN", "PRON"]},
                        {"dep": "iobj", "pos": ["NOUN", "PROPN", "PRON"]}
                    ],
                    "language_indicators": ["à", "pour", "vers"]
                }
            },
            "de": {  # German
                "AGENT": {
                    "additional_patterns": [
                        {"dep": "nsubj", "pos": ["NOUN", "PROPN", "PRON"]},
                        {"dep": "agent", "pos": ["NOUN", "PROPN", "PRON"]}
                    ],
                    "language_indicators": ["von", "durch", "mittels"]
                },
                "PATIENT": {
                    "additional_patterns": [
                        {"dep": "dobj", "pos": ["NOUN", "PROPN", "PRON"]},
                        {"dep": "iobj", "pos": ["NOUN", "PROPN", "PRON"]}
                    ],
                    "language_indicators": ["zu", "für", "an"]
                }
            }
        }
    
    def _initialize_universal_dep_mappings(self) -> Dict[str, Dict[str, str]]:
        """Initialize universal dependency relation mappings."""
        
        return {
            "subject": {
                "en": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
                "es": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
                "fr": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
                "de": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
                "it": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
                "pt": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
                "ru": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
                "zh": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
                "ja": ["nsubj", "nsubjpass", "csubj", "csubjpass"],
                "ko": ["nsubj", "nsubjpass", "csubj", "csubjpass"]
            },
            "object": {
                "en": ["dobj", "iobj", "pobj", "attr"],
                "es": ["dobj", "iobj", "pobj", "attr"],
                "fr": ["dobj", "iobj", "pobj", "attr"],
                "de": ["dobj", "iobj", "pobj", "attr"],
                "it": ["dobj", "iobj", "pobj", "attr"],
                "pt": ["dobj", "iobj", "pobj", "attr"],
                "ru": ["dobj", "iobj", "pobj", "attr"],
                "zh": ["dobj", "iobj", "pobj", "attr"],
                "ja": ["dobj", "iobj", "pobj", "attr"],
                "ko": ["dobj", "iobj", "pobj", "attr"]
            },
            "modifier": {
                "en": ["amod", "advmod", "nummod", "det"],
                "es": ["amod", "advmod", "nummod", "det"],
                "fr": ["amod", "advmod", "nummod", "det"],
                "de": ["amod", "advmod", "nummod", "det"],
                "it": ["amod", "advmod", "nummod", "det"],
                "pt": ["amod", "advmod", "nummod", "det"],
                "ru": ["amod", "advmod", "nummod", "det"],
                "zh": ["amod", "advmod", "nummod", "det"],
                "ja": ["amod", "advmod", "nummod", "det"],
                "ko": ["amod", "advmod", "nummod", "det"]
            }
        }
    
    def _initialize_cross_lingual_concepts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cross-lingual concept mappings."""
        
        return {
            "person": {
                "en": ["boy", "girl", "man", "woman", "child", "teacher", "student"],
                "es": ["niño", "niña", "hombre", "mujer", "niño", "profesor", "estudiante"],
                "fr": ["garçon", "fille", "homme", "femme", "enfant", "professeur", "étudiant"],
                "de": ["Junge", "Mädchen", "Mann", "Frau", "Kind", "Lehrer", "Student"],
                "it": ["ragazzo", "ragazza", "uomo", "donna", "bambino", "professore", "studente"],
                "pt": ["menino", "menina", "homem", "mulher", "criança", "professor", "estudante"],
                "ru": ["мальчик", "девочка", "мужчина", "женщина", "ребенок", "учитель", "студент"],
                "zh": ["男孩", "女孩", "男人", "女人", "孩子", "老师", "学生"],
                "ja": ["男の子", "女の子", "男", "女", "子供", "先生", "学生"],
                "ko": ["소년", "소녀", "남자", "여자", "아이", "선생님", "학생"]
            },
            "object": {
                "en": ["ball", "book", "car", "house", "mat"],
                "es": ["pelota", "libro", "coche", "casa", "alfombrilla"],
                "fr": ["balle", "livre", "voiture", "maison", "tapis"],
                "de": ["Ball", "Buch", "Auto", "Haus", "Matte"],
                "it": ["palla", "libro", "auto", "casa", "tappetino"],
                "pt": ["bola", "livro", "carro", "casa", "tapete"],
                "ru": ["мяч", "книга", "машина", "дом", "коврик"],
                "zh": ["球", "书", "车", "房子", "垫子"],
                "ja": ["ボール", "本", "車", "家", "マット"],
                "ko": ["공", "책", "차", "집", "매트"]
            },
            "action": {
                "en": ["kick", "give", "write", "eat", "sleep"],
                "es": ["patear", "dar", "escribir", "comer", "dormir"],
                "fr": ["donner un coup de pied", "donner", "écrire", "manger", "dormir"],
                "de": ["treten", "geben", "schreiben", "essen", "schlafen"],
                "it": ["calciare", "dare", "scrivere", "mangiare", "dormire"],
                "pt": ["chutar", "dar", "escrever", "comer", "dormir"],
                "ru": ["пинать", "давать", "писать", "есть", "спать"],
                "zh": ["踢", "给", "写", "吃", "睡"],
                "ja": ["蹴る", "与える", "書く", "食べる", "寝る"],
                "ko": ["차다", "주다", "쓰다", "먹다", "자다"]
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        
        # Simple language detection based on common words
        text_lower = text.lower()
        
        # Spanish indicators
        if any(word in text_lower for word in ["el", "la", "los", "las", "es", "son", "está", "están"]):
            return "es"
        
        # French indicators
        if any(word in text_lower for word in ["le", "la", "les", "est", "sont", "être", "avoir"]):
            return "fr"
        
        # German indicators
        if any(word in text_lower for word in ["der", "die", "das", "ist", "sind", "haben", "sein"]):
            return "de"
        
        # Italian indicators
        if any(word in text_lower for word in ["il", "la", "gli", "le", "è", "sono", "essere"]):
            return "it"
        
        # Portuguese indicators
        if any(word in text_lower for word in ["o", "a", "os", "as", "é", "são", "estar"]):
            return "pt"
        
        # Russian indicators
        if any(char in text for char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"):
            return "ru"
        
        # Chinese indicators
        if any(char in text for char in "的一是在有和人大中上为以国我地时用出会可也你对他生说年着就那要下可能过子还"):
            return "zh"
        
        # Japanese indicators
        if any(char in text for char in "のをはにがでとしもなかたきれいすてまよわ"):
            return "ja"
        
        # Korean indicators
        if any(char in text for char in "의에가을를이도로다고하나지그것때문에"):
            return "ko"
        
        # Default to English
        return "en"
    
    def parse_cross_lingual_ud(self, text: str, language: str = None) -> List[CrossLingualUDNode]:
        """Parse Universal Dependencies structure for any supported language."""
        
        if language is None:
            language = self.detect_language(text)
        
        if language not in self.nlp_models:
            print(f"⚠️  No model available for {language}, using English fallback")
            language = "en"
        
        nlp = self.nlp_models[language]
        doc = nlp(text)
        
        ud_nodes = []
        for token in doc:
            node = CrossLingualUDNode(
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
                },
                language=language,
                universal_pos=self._map_to_universal_pos(token.pos_, language),
                universal_dep=self._map_to_universal_dep(token.dep_, language)
            )
            ud_nodes.append(node)
        
        return ud_nodes
    
    def _map_to_universal_pos(self, pos: str, language: str) -> str:
        """Map language-specific POS tags to universal POS tags."""
        
        # Universal POS mapping (simplified)
        universal_pos_map = {
            "NOUN": "NOUN",
            "PROPN": "PROPN", 
            "VERB": "VERB",
            "ADJ": "ADJ",
            "ADV": "ADV",
            "PRON": "PRON",
            "DET": "DET",
            "ADP": "ADP",
            "CONJ": "CONJ",
            "INTJ": "INTJ",
            "NUM": "NUM",
            "PART": "PART",
            "PUNCT": "PUNCT",
            "SYM": "SYM",
            "X": "X"
        }
        
        return universal_pos_map.get(pos, pos)
    
    def _map_to_universal_dep(self, dep: str, language: str) -> str:
        """Map language-specific dependency relations to universal relations."""
        
        # Universal dependency mapping (simplified)
        universal_dep_map = {
            "nsubj": "nsubj",
            "nsubjpass": "nsubjpass",
            "dobj": "dobj",
            "iobj": "iobj",
            "pobj": "pobj",
            "attr": "attr",
            "amod": "amod",
            "advmod": "advmod",
            "det": "det",
            "prep": "prep",
            "conj": "conj",
            "cc": "cc",
            "mark": "mark",
            "aux": "aux",
            "cop": "cop",
            "neg": "neg",
            "punct": "punct",
            "root": "root"
        }
        
        return universal_dep_map.get(dep, dep)
    
    def extract_cross_lingual_semantic_roles(self, ud_nodes: List[CrossLingualUDNode]) -> List[CrossLingualSemanticRole]:
        """Extract semantic roles using cross-lingual patterns."""
        
        language = ud_nodes[0].language if ud_nodes else "en"
        semantic_roles = []
        
        # First pass: apply universal patterns
        for node in ud_nodes:
            for role_name, role_config in self.universal_srl_patterns.items():
                for pattern in role_config["universal_patterns"]:
                    if (node.universal_dep == pattern["dep"] and 
                        node.universal_pos in pattern["pos"]):
                        
                        # Calculate confidence
                        confidence = self._calculate_cross_lingual_confidence(
                            node, role_name, role_config, ud_nodes, language
                        )
                        
                        # Check context indicators
                        context_score = self._check_cross_lingual_context(
                            node, ud_nodes, role_config["cross_lingual_indicators"]
                        )
                        confidence *= context_score
                        
                        semantic_role = CrossLingualSemanticRole(
                            role=role_name,
                            text=node.token,
                            start=node.index,
                            end=node.index + 1,
                            confidence=confidence,
                            entity_type=self._determine_cross_lingual_entity_type(node, ud_nodes),
                            language=language,
                            universal_role=role_name
                        )
                        semantic_roles.append(semantic_role)
                        break
        
        # Second pass: apply language-specific patterns
        if language in self.language_specific_patterns:
            for node in ud_nodes:
                for role_name, role_config in self.language_specific_patterns[language].items():
                    for pattern in role_config["additional_patterns"]:
                        if (node.dep == pattern["dep"] and 
                            node.pos in pattern["pos"]):
                            
                            # Check if this role is already assigned
                            existing_role = next((r for r in semantic_roles if r.text == node.token), None)
                            if existing_role:
                                # Update confidence if this pattern is stronger
                                new_confidence = self._calculate_cross_lingual_confidence(
                                    node, role_name, role_config, ud_nodes, language
                                )
                                if new_confidence > existing_role.confidence:
                                    existing_role.role = role_name
                                    existing_role.universal_role = role_name
                                    existing_role.confidence = new_confidence
                            else:
                                # Add new role
                                confidence = self._calculate_cross_lingual_confidence(
                                    node, role_name, role_config, ud_nodes, language
                                )
                                semantic_role = CrossLingualSemanticRole(
                                    role=role_name,
                                    text=node.token,
                                    start=node.index,
                                    end=node.index + 1,
                                    confidence=confidence,
                                    entity_type=self._determine_cross_lingual_entity_type(node, ud_nodes),
                                    language=language,
                                    universal_role=role_name
                                )
                                semantic_roles.append(semantic_role)
                            break
        
        # Third pass: resolve conflicts
        final_roles = self._resolve_cross_lingual_conflicts(semantic_roles, ud_nodes)
        
        return final_roles
    
    def _calculate_cross_lingual_confidence(self, node: CrossLingualUDNode, role: str, 
                                         role_config: Dict[str, Any], all_nodes: List[CrossLingualUDNode], 
                                         language: str) -> float:
        """Calculate confidence score for cross-lingual semantic role assignment."""
        
        base_confidence = 0.7
        
        # Higher confidence for clear role indicators
        if role == "AGENT" and node.universal_dep in ["nsubj", "nsubjpass"]:
            base_confidence += 0.2
        elif role == "PATIENT" and node.universal_dep in ["dobj", "nsubjpass"]:
            base_confidence += 0.2
        elif role == "LOCATION" and any(word in node.token.lower() for word in ["in", "at", "on", "en", "dans", "sur", "in", "auf"]):
            base_confidence += 0.1
        
        # Higher confidence for longer, more specific entities
        if len(node.token) > 3:
            base_confidence += 0.1
        
        # Higher confidence for proper nouns in certain roles
        if node.universal_pos == "PROPN" and role in ["AGENT", "LOCATION", "THEME"]:
            base_confidence += 0.1
        
        # Language-specific confidence adjustments
        if language in ["es", "fr", "de"]:
            # These languages have more explicit case marking
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _check_cross_lingual_context(self, node: CrossLingualUDNode, all_nodes: List[CrossLingualUDNode], 
                                   indicators: List[str]) -> float:
        """Check for cross-lingual context indicators."""
        
        # Check if any parent or child nodes contain context indicators
        for other_node in all_nodes:
            if other_node.head == node.index or node.head == other_node.index:
                if any(indicator in other_node.token.lower() for indicator in indicators):
                    return 1.2  # Boost confidence
        
        return 1.0  # No context boost
    
    def _determine_cross_lingual_entity_type(self, node: CrossLingualUDNode, all_nodes: List[CrossLingualUDNode]) -> Optional[str]:
        """Determine entity type based on cross-lingual features."""
        
        if node.universal_pos == "PROPN":
            # Check for location indicators across languages
            location_indicators = ["city", "country", "state", "river", "mountain", 
                                 "ciudad", "país", "estado", "río", "montaña",
                                 "ville", "pays", "état", "rivière", "montagne",
                                 "Stadt", "Land", "Staat", "Fluss", "Berg"]
            if any(indicator in node.token.lower() for indicator in location_indicators):
                return "LOCATION"
            # Check for person indicators
            person_indicators = ["mr", "mrs", "dr", "prof", "señor", "señora", "doctor", "profesor",
                               "monsieur", "madame", "docteur", "professeur",
                               "herr", "frau", "doktor", "professor"]
            if any(indicator in node.token.lower() for indicator in person_indicators):
                return "PERSON"
            else:
                return "ORGANIZATION"
        elif node.universal_pos == "NOUN":
            # Check against cross-lingual concept mappings
            for concept_type, language_mappings in self.cross_lingual_concepts.items():
                if node.language in language_mappings:
                    if node.token.lower() in language_mappings[node.language]:
                        if concept_type == "person":
                            return "PERSON"
                        elif concept_type == "object":
                            return "OBJECT"
                        elif concept_type == "action":
                            return "ACTION"
            return "OBJECT"
        
        return None
    
    def _resolve_cross_lingual_conflicts(self, semantic_roles: List[CrossLingualSemanticRole], 
                                       ud_nodes: List[CrossLingualUDNode]) -> List[CrossLingualSemanticRole]:
        """Resolve conflicts between cross-lingual semantic roles."""
        
        # Group roles by text
        roles_by_text = defaultdict(list)
        for role in semantic_roles:
            roles_by_text[role.text].append(role)
        
        final_roles = []
        
        for text, roles in roles_by_text.items():
            if len(roles) == 1:
                # No conflict, keep the role
                final_roles.append(roles[0])
            else:
                # Resolve conflict by selecting the best role
                best_role = max(roles, key=lambda r: r.confidence)
                if best_role.confidence > 0.5:
                    final_roles.append(best_role)
        
        return final_roles
    
    def analyze_cross_lingual_text(self, text: str, language: str = None) -> Dict[str, Any]:
        """Analyze text using cross-lingual UD + SRL."""
        
        if language is None:
            language = self.detect_language(text)
        
        print(f"🔍 Cross-Lingual Analysis: '{text}' (Language: {language})")
        print("-" * 70)
        
        # Parse UD structure
        ud_nodes = self.parse_cross_lingual_ud(text, language)
        
        # Extract semantic roles
        semantic_roles = self.extract_cross_lingual_semantic_roles(ud_nodes)
        
        # Group roles by type
        role_mappings = defaultdict(list)
        for role in semantic_roles:
            role_mappings[role.role].append(role)
        
        # Extract basic structure
        subject, predicate, obj, indirect_obj = self._extract_cross_lingual_structure(ud_nodes)
        
        # Determine grammatical features
        voice, mood, tense, aspect, modality = self._extract_cross_lingual_grammatical_features(ud_nodes)
        
        # Show analysis
        print(f"📐 Cross-Lingual Structure:")
        print(f"  Subject: {subject}")
        print(f"  Predicate: {predicate}")
        print(f"  Object: {obj}")
        print(f"  Voice: {voice}")
        print(f"  Tense: {tense}")
        print(f"  Aspect: {aspect}")
        print(f"  Language: {language}")
        
        print(f"\n🌳 Cross-Lingual Universal Dependencies:")
        for node in ud_nodes:
            print(f"  {node.token} ({node.universal_pos}) --{node.universal_dep}--> {ud_nodes[node.head].token if node.head != node.index else 'ROOT'}")
        
        print(f"\n🎭 Cross-Lingual Semantic Role Labeling:")
        for role in semantic_roles:
            print(f"  {role.universal_role}: '{role.text}' (confidence: {role.confidence:.2f}, type: {role.entity_type}, lang: {role.language})")
        
        print(f"\n📋 Cross-Lingual Role Mappings:")
        for role_type, roles in role_mappings.items():
            print(f"  {role_type}: {[r.text for r in roles]}")
        
        return {
            "original": text,
            "language": language,
            "ud_nodes": ud_nodes,
            "semantic_roles": semantic_roles,
            "role_mappings": dict(role_mappings),
            "structure": {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "indirect_object": indirect_obj,
                "voice": voice,
                "tense": tense,
                "aspect": aspect
            }
        }
    
    def _extract_cross_lingual_structure(self, ud_nodes: List[CrossLingualUDNode]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Extract basic structure using universal dependency relations."""
        
        subject = None
        predicate = None
        obj = None
        indirect_obj = None
        
        for node in ud_nodes:
            if node.universal_dep in ["nsubj", "nsubjpass"]:
                subject = node.token
            elif node.universal_dep == "ROOT" and node.universal_pos == "VERB":
                predicate = node.token
            elif node.universal_dep == "dobj":
                obj = node.token
            elif node.universal_dep == "iobj":
                indirect_obj = node.token
        
        return subject, predicate, obj, indirect_obj
    
    def _extract_cross_lingual_grammatical_features(self, ud_nodes: List[CrossLingualUDNode]) -> Tuple[str, str, str, str, Optional[str]]:
        """Extract grammatical features using universal patterns."""
        
        voice = "active"
        mood = "declarative"
        tense = "present"
        aspect = "simple"
        modality = None
        
        for node in ud_nodes:
            if node.universal_dep == "nsubjpass":
                voice = "passive"
            
            if node.universal_pos == "VERB":
                # Simplified tense detection
                if "VBD" in node.features.get("tag", "") or "VBN" in node.features.get("tag", ""):
                    tense = "past"
                elif "VBG" in node.features.get("tag", ""):
                    aspect = "progressive"
                elif "VBN" in node.features.get("tag", ""):
                    aspect = "perfect"
            
            # Modality detection across languages
            modality_words = ["can", "could", "will", "would", "should", "must", "may", "might",
                            "poder", "deber", "tener que", "pouvoir", "devoir", "falloir",
                            "können", "müssen", "sollen", "wollen", "dürfen"]
            if node.lemma.lower() in modality_words:
                modality = node.lemma
        
        return voice, mood, tense, aspect, modality

def demonstrate_cross_lingual_system():
    """Demonstrate the cross-lingual UD + SRL system."""
    
    print("🌍 CROSS-LINGUAL UD + SRL SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()
    
    cross_lingual_system = CrossLingualUDSRLSystem()
    
    # Test cases in different languages
    test_cases = [
        {
            "text": "The boy kicked the ball in Paris.",
            "language": "en",
            "description": "English - Basic action with location"
        },
        {
            "text": "El niño pateó la pelota en París.",
            "language": "es", 
            "description": "Spanish - Same structure, different language"
        },
        {
            "text": "Le garçon a donné un coup de pied au ballon à Paris.",
            "language": "fr",
            "description": "French - Different verb construction"
        },
        {
            "text": "Der Junge hat den Ball in Paris getreten.",
            "language": "de",
            "description": "German - Different word order and verb placement"
        },
        {
            "text": "Il ragazzo ha calciato la palla a Parigi.",
            "language": "it",
            "description": "Italian - Similar to Spanish structure"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🎯 EXAMPLE {i}: {test_case['description']}")
        print(f"Text: '{test_case['text']}'")
        print("-" * 60)
        
        try:
            result = cross_lingual_system.analyze_cross_lingual_text(
                test_case['text'], test_case['language']
            )
            
            print(f"\n📊 CROSS-LINGUAL RESULTS:")
            print(f"  Detected Language: {result['language']}")
            print(f"  Semantic Roles: {len(result['semantic_roles'])}")
            print(f"  UD Nodes: {len(result['ud_nodes'])}")
            print(f"  Universal Structure: {result['structure']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print()

if __name__ == "__main__":
    demonstrate_cross_lingual_system()
