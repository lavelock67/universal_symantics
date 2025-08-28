#!/usr/bin/env python3
"""
Cultural Adaptation System

This implements cultural context and adaptation for universal translation,
handling idiomatic expressions, politeness levels, and cultural norms.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class CulturalContext:
    """Represents cultural context for a language/region."""
    language: str
    region: str
    politeness_levels: Dict[str, str]  # formal, informal, polite, rude
    cultural_norms: Dict[str, Any]
    idiomatic_mappings: Dict[str, str]
    formality_markers: Dict[str, List[str]]

@dataclass
class CulturalAdaptation:
    """Represents a cultural adaptation for translation."""
    original_text: str
    adapted_text: str
    cultural_context: CulturalContext
    adaptations_applied: List[str]
    confidence: float
    reasoning: str

class CulturalAdaptationSystem:
    """Cultural adaptation system for universal translation."""
    
    def __init__(self):
        # Cultural context database
        self.cultural_contexts = self._initialize_cultural_contexts()
        
        # Idiomatic expression database
        self.idiomatic_expressions = self._initialize_idiomatic_expressions()
        
        # Politeness and formality patterns
        self.politeness_patterns = self._initialize_politeness_patterns()
        
        # Cultural norm mappings
        self.cultural_norms = self._initialize_cultural_norms()
        
        # Formality detection patterns
        self.formality_patterns = self._initialize_formality_patterns()
    
    def _initialize_cultural_contexts(self) -> Dict[str, CulturalContext]:
        """Initialize cultural contexts for different languages/regions."""
        
        return {
            "en_US": CulturalContext(
                language="en",
                region="US",
                politeness_levels={
                    "formal": "very polite, respectful",
                    "informal": "casual, friendly",
                    "polite": "respectful but not overly formal",
                    "rude": "direct, potentially offensive"
                },
                cultural_norms={
                    "directness": "high",
                    "personal_space": "large",
                    "time_orientation": "future",
                    "individualism": "high",
                    "power_distance": "low"
                },
                idiomatic_mappings={},
                formality_markers={
                    "formal": ["sir", "madam", "please", "would you", "could you"],
                    "informal": ["hey", "dude", "guy", "buddy", "cool"]
                }
            ),
            "en_GB": CulturalContext(
                language="en",
                region="GB",
                politeness_levels={
                    "formal": "very polite, reserved",
                    "informal": "polite but casual",
                    "polite": "respectful, understated",
                    "rude": "direct, potentially offensive"
                },
                cultural_norms={
                    "directness": "medium",
                    "personal_space": "medium",
                    "time_orientation": "present",
                    "individualism": "medium",
                    "power_distance": "medium"
                },
                idiomatic_mappings={},
                formality_markers={
                    "formal": ["sir", "madam", "please", "would you mind", "if you don't mind"],
                    "informal": ["mate", "cheers", "brilliant", "lovely"]
                }
            ),
            "es_ES": CulturalContext(
                language="es",
                region="ES",
                politeness_levels={
                    "formal": "very respectful, using usted",
                    "informal": "friendly, using tú",
                    "polite": "respectful but warm",
                    "rude": "direct, potentially offensive"
                },
                cultural_norms={
                    "directness": "medium",
                    "personal_space": "small",
                    "time_orientation": "present",
                    "individualism": "medium",
                    "power_distance": "high"
                },
                idiomatic_mappings={},
                formality_markers={
                    "formal": ["usted", "señor", "señora", "por favor", "si fuera tan amable"],
                    "informal": ["tú", "hombre", "tío", "vale", "genial"]
                }
            ),
            "es_MX": CulturalContext(
                language="es",
                region="MX",
                politeness_levels={
                    "formal": "very respectful, using usted",
                    "informal": "friendly, using tú",
                    "polite": "respectful but warm",
                    "rude": "direct, potentially offensive"
                },
                cultural_norms={
                    "directness": "medium",
                    "personal_space": "small",
                    "time_orientation": "present",
                    "individualism": "medium",
                    "power_distance": "high"
                },
                idiomatic_mappings={},
                formality_markers={
                    "formal": ["usted", "señor", "señora", "por favor", "si fuera tan amable"],
                    "informal": ["tú", "güey", "wey", "chido", "padre"]
                }
            ),
            "fr_FR": CulturalContext(
                language="fr",
                region="FR",
                politeness_levels={
                    "formal": "very polite, using vous",
                    "informal": "friendly, using tu",
                    "polite": "respectful but warm",
                    "rude": "direct, potentially offensive"
                },
                cultural_norms={
                    "directness": "medium",
                    "personal_space": "medium",
                    "time_orientation": "present",
                    "individualism": "medium",
                    "power_distance": "high"
                },
                idiomatic_mappings={},
                formality_markers={
                    "formal": ["vous", "monsieur", "madame", "s'il vous plaît", "pourriez-vous"],
                    "informal": ["tu", "mec", "copain", "super", "génial"]
                }
            ),
            "de_DE": CulturalContext(
                language="de",
                region="DE",
                politeness_levels={
                    "formal": "very formal, using Sie",
                    "informal": "friendly, using du",
                    "polite": "respectful and precise",
                    "rude": "direct, potentially offensive"
                },
                cultural_norms={
                    "directness": "high",
                    "personal_space": "large",
                    "time_orientation": "future",
                    "individualism": "medium",
                    "power_distance": "medium"
                },
                idiomatic_mappings={},
                formality_markers={
                    "formal": ["Sie", "Herr", "Frau", "bitte", "könnten Sie"],
                    "informal": ["du", "Mann", "Kumpel", "super", "toll"]
                }
            ),
            "ja_JP": CulturalContext(
                language="ja",
                region="JP",
                politeness_levels={
                    "formal": "very polite, using keigo",
                    "informal": "casual, using plain form",
                    "polite": "respectful, using teineigo",
                    "rude": "direct, potentially offensive"
                },
                cultural_norms={
                    "directness": "low",
                    "personal_space": "small",
                    "time_orientation": "present",
                    "individualism": "low",
                    "power_distance": "high"
                },
                idiomatic_mappings={},
                formality_markers={
                    "formal": ["です", "ます", "ございます", "お", "ご"],
                    "informal": ["だ", "る", "よ", "ね", "さ"]
                }
            ),
            "zh_CN": CulturalContext(
                language="zh",
                region="CN",
                politeness_levels={
                    "formal": "very polite, using honorifics",
                    "informal": "casual, using plain form",
                    "polite": "respectful, using polite form",
                    "rude": "direct, potentially offensive"
                },
                cultural_norms={
                    "directness": "medium",
                    "personal_space": "small",
                    "time_orientation": "present",
                    "individualism": "low",
                    "power_distance": "high"
                },
                idiomatic_mappings={},
                formality_markers={
                    "formal": ["您", "请", "谢谢", "对不起", "不好意思"],
                    "informal": ["你", "哥们", "朋友", "好", "棒"]
                }
            )
        }
    
    def _initialize_idiomatic_expressions(self) -> Dict[str, Dict[str, str]]:
        """Initialize idiomatic expression mappings."""
        
        return {
            "en": {
                "break a leg": "good luck",
                "piece of cake": "very easy",
                "hit the nail on the head": "exactly right",
                "let the cat out of the bag": "reveal a secret",
                "pull someone's leg": "joke with someone",
                "cost an arm and a leg": "very expensive",
                "break the ice": "start a conversation",
                "get out of hand": "become uncontrollable",
                "on the ball": "alert and competent",
                "pull yourself together": "calm down and control yourself"
            },
            "es": {
                "romper una pierna": "buena suerte",
                "pan comido": "muy fácil",
                "dar en el clavo": "exactamente correcto",
                "se le escapó el gato": "revelar un secreto",
                "tomar el pelo": "bromear con alguien",
                "costar un ojo de la cara": "muy caro",
                "romper el hielo": "iniciar una conversación",
                "irse de las manos": "volverse incontrolable",
                "estar al tanto": "estar alerta y competente",
                "ponerse las pilas": "animarse y controlarse"
            },
            "fr": {
                "casser une jambe": "bonne chance",
                "du gâteau": "très facile",
                "mettre dans le mille": "exactement correct",
                "vendre la mèche": "révéler un secret",
                "tirer la jambe": "plaisanter avec quelqu'un",
                "coûter les yeux de la tête": "très cher",
                "briser la glace": "commencer une conversation",
                "échapper au contrôle": "devenir incontrôlable",
                "être sur le ballon": "être alerte et compétent",
                "se ressaisir": "se calmer et se contrôler"
            },
            "de": {
                "ein Bein brechen": "viel Glück",
                "ein Kinderspiel": "sehr einfach",
                "den Nagel auf den Kopf treffen": "genau richtig",
                "die Katze aus dem Sack lassen": "ein Geheimnis verraten",
                "jemanden auf den Arm nehmen": "mit jemandem scherzen",
                "ein Vermögen kosten": "sehr teuer",
                "das Eis brechen": "ein Gespräch beginnen",
                "außer Kontrolle geraten": "unkontrollierbar werden",
                "auf dem Ball sein": "wachsam und kompetent sein",
                "sich zusammenreißen": "sich beruhigen und kontrollieren"
            },
            "ja": {
                "足を折る": "頑張って",
                "朝飯前": "とても簡単",
                "的を射る": "まさに正しい",
                "猫を袋から出す": "秘密を漏らす",
                "人の足を引っ張る": "人をからかう",
                "目玉が飛び出る": "とても高い",
                "氷を割る": "会話を始める",
                "手に負えない": "制御不能になる",
                "ボールの上にいる": "警戒して有能である",
                "気を引き締める": "落ち着いて自制する"
            },
            "zh": {
                "断腿": "祝你好运",
                "小菜一碟": "很容易",
                "一针见血": "完全正确",
                "泄露天机": "泄露秘密",
                "开玩笑": "和某人开玩笑",
                "贵得要命": "非常贵",
                "破冰": "开始对话",
                "失控": "变得无法控制",
                "保持警惕": "保持警觉和能干",
                "振作起来": "冷静下来控制自己"
            }
        }
    
    def _initialize_politeness_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize politeness patterns for different languages."""
        
        return {
            "en": {
                "formal": {
                    "greetings": ["Good morning", "Good afternoon", "Good evening", "Hello"],
                    "requests": ["Would you please", "Could you please", "I would appreciate if", "Might I ask"],
                    "apologies": ["I sincerely apologize", "I deeply regret", "Please accept my apologies"],
                    "thanks": ["Thank you very much", "I greatly appreciate", "I'm very grateful"]
                },
                "informal": {
                    "greetings": ["Hi", "Hey", "Hello", "What's up"],
                    "requests": ["Can you", "Could you", "Would you mind", "Do you think you could"],
                    "apologies": ["Sorry", "I'm sorry", "My bad", "Oops"],
                    "thanks": ["Thanks", "Thank you", "Appreciate it", "Cheers"]
                }
            },
            "es": {
                "formal": {
                    "greetings": ["Buenos días", "Buenas tardes", "Buenas noches", "Hola"],
                    "requests": ["¿Podría por favor", "¿Le importaría", "Agradecería si", "¿Me permitiría"],
                    "apologies": ["Me disculpo sinceramente", "Lamento profundamente", "Por favor acepte mis disculpas"],
                    "thanks": ["Muchas gracias", "Le agradezco mucho", "Estoy muy agradecido"]
                },
                "informal": {
                    "greetings": ["Hola", "¿Qué tal?", "¿Qué pasa?", "¿Cómo estás?"],
                    "requests": ["¿Puedes", "¿Podrías", "¿Te importaría", "¿Crees que podrías"],
                    "apologies": ["Lo siento", "Perdón", "Mi culpa", "Ups"],
                    "thanks": ["Gracias", "Te agradezco", "Vale", "Genial"]
                }
            },
            "fr": {
                "formal": {
                    "greetings": ["Bonjour", "Bonne après-midi", "Bonsoir", "Salut"],
                    "requests": ["Pourriez-vous s'il vous plaît", "Auriez-vous l'obligeance", "Je vous serais reconnaissant si", "Permettez-moi de"],
                    "apologies": ["Je m'excuse sincèrement", "Je regrette profondément", "Veuillez accepter mes excuses"],
                    "thanks": ["Merci beaucoup", "Je vous remercie infiniment", "Je suis très reconnaissant"]
                },
                "informal": {
                    "greetings": ["Salut", "Coucou", "Bonjour", "Comment ça va?"],
                    "requests": ["Peux-tu", "Pourrais-tu", "Ça te dérangerait", "Tu crois que tu pourrais"],
                    "apologies": ["Désolé", "Pardon", "C'est ma faute", "Oups"],
                    "thanks": ["Merci", "Je te remercie", "Super", "Génial"]
                }
            },
            "de": {
                "formal": {
                    "greetings": ["Guten Morgen", "Guten Tag", "Guten Abend", "Hallo"],
                    "requests": ["Könnten Sie bitte", "Würden Sie bitte", "Ich wäre Ihnen dankbar wenn", "Dürfte ich Sie bitten"],
                    "apologies": ["Ich entschuldige mich aufrichtig", "Es tut mir sehr leid", "Bitte akzeptieren Sie meine Entschuldigung"],
                    "thanks": ["Vielen Dank", "Ich bin Ihnen sehr dankbar", "Ich schätze es sehr"]
                },
                "informal": {
                    "greetings": ["Hallo", "Hi", "Servus", "Wie geht's?"],
                    "requests": ["Kannst du", "Könntest du", "Würde es dich stören", "Denkst du du könntest"],
                    "apologies": ["Entschuldigung", "Es tut mir leid", "Mein Fehler", "Ups"],
                    "thanks": ["Danke", "Ich danke dir", "Super", "Toll"]
                }
            }
        }
    
    def _initialize_cultural_norms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural norm mappings."""
        
        return {
            "directness": {
                "high": ["en_US", "de_DE"],
                "medium": ["en_GB", "es_ES", "es_MX", "fr_FR", "zh_CN"],
                "low": ["ja_JP"]
            },
            "personal_space": {
                "large": ["en_US", "de_DE"],
                "medium": ["en_GB", "fr_FR"],
                "small": ["es_ES", "es_MX", "ja_JP", "zh_CN"]
            },
            "time_orientation": {
                "future": ["en_US", "de_DE"],
                "present": ["en_GB", "es_ES", "es_MX", "fr_FR", "ja_JP", "zh_CN"]
            },
            "individualism": {
                "high": ["en_US"],
                "medium": ["en_GB", "es_ES", "es_MX", "fr_FR", "de_DE"],
                "low": ["ja_JP", "zh_CN"]
            },
            "power_distance": {
                "high": ["es_ES", "es_MX", "fr_FR", "ja_JP", "zh_CN"],
                "medium": ["en_GB", "de_DE"],
                "low": ["en_US"]
            }
        }
    
    def _initialize_formality_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize formality detection patterns."""
        
        return {
            "en": {
                "formal_indicators": ["sir", "madam", "please", "would you", "could you", "thank you", "excuse me"],
                "informal_indicators": ["hey", "dude", "guy", "buddy", "cool", "awesome", "yeah", "okay"]
            },
            "es": {
                "formal_indicators": ["usted", "señor", "señora", "por favor", "gracias", "disculpe"],
                "informal_indicators": ["tú", "hombre", "tío", "vale", "genial", "sí", "vale"]
            },
            "fr": {
                "formal_indicators": ["vous", "monsieur", "madame", "s'il vous plaît", "merci", "excusez-moi"],
                "informal_indicators": ["tu", "mec", "copain", "super", "génial", "oui", "ok"]
            },
            "de": {
                "formal_indicators": ["Sie", "Herr", "Frau", "bitte", "danke", "entschuldigung"],
                "informal_indicators": ["du", "Mann", "Kumpel", "super", "toll", "ja", "ok"]
            }
        }
    
    def detect_formality_level(self, text: str, language: str) -> str:
        """Detect the formality level of the input text."""
        
        text_lower = text.lower()
        formal_count = 0
        informal_count = 0
        
        if language in self.formality_patterns:
            patterns = self.formality_patterns[language]
            
            # Count formal indicators
            for indicator in patterns["formal_indicators"]:
                if indicator.lower() in text_lower:
                    formal_count += 1
            
            # Count informal indicators
            for indicator in patterns["informal_indicators"]:
                if indicator.lower() in text_lower:
                    informal_count += 1
        
        # Determine formality level
        if formal_count > informal_count:
            return "formal"
        elif informal_count > formal_count:
            return "informal"
        else:
            return "neutral"
    
    def adapt_idiomatic_expressions(self, text: str, source_language: str, target_language: str) -> str:
        """Adapt idiomatic expressions from source to target language."""
        
        adapted_text = text
        
        if source_language in self.idiomatic_expressions:
            source_idioms = self.idiomatic_expressions[source_language]
            
            for idiom, meaning in source_idioms.items():
                if idiom.lower() in text.lower():
                    # Find equivalent in target language
                    if target_language in self.idiomatic_expressions:
                        target_idioms = self.idiomatic_expressions[target_language]
                        
                        # Look for equivalent meaning
                        for target_idiom, target_meaning in target_idioms.items():
                            if target_meaning.lower() == meaning.lower():
                                # Replace idiom
                                adapted_text = re.sub(
                                    re.escape(idiom), 
                                    target_idiom, 
                                    adapted_text, 
                                    flags=re.IGNORECASE
                                )
                                break
                        else:
                            # If no equivalent found, use literal meaning
                            adapted_text = re.sub(
                                re.escape(idiom), 
                                meaning, 
                                adapted_text, 
                                flags=re.IGNORECASE
                            )
        
        return adapted_text
    
    def adapt_politeness_level(self, text: str, source_context: str, target_context: str) -> str:
        """Adapt politeness level based on cultural context."""
        
        source_culture = self.cultural_contexts.get(source_context)
        target_culture = self.cultural_contexts.get(target_context)
        
        if not source_culture or not target_culture:
            return text
        
        # Detect current politeness level
        current_level = self.detect_formality_level(text, source_culture.language)
        
        # Get politeness patterns for both languages
        source_patterns = self.politeness_patterns.get(source_culture.language, {})
        target_patterns = self.politeness_patterns.get(target_culture.language, {})
        
        adapted_text = text
        
        # Adapt greetings
        if current_level in source_patterns and current_level in target_patterns:
            source_greetings = source_patterns[current_level]["greetings"]
            target_greetings = target_patterns[current_level]["greetings"]
            
            for source_greeting in source_greetings:
                if source_greeting.lower() in text.lower():
                    target_greeting = random.choice(target_greetings)
                    adapted_text = re.sub(
                        re.escape(source_greeting), 
                        target_greeting, 
                        adapted_text, 
                        flags=re.IGNORECASE
                    )
        
        return adapted_text
    
    def apply_cultural_norms(self, text: str, source_context: str, target_context: str) -> str:
        """Apply cultural norm adaptations."""
        
        source_culture = self.cultural_contexts.get(source_context)
        target_culture = self.cultural_contexts.get(target_context)
        
        if not source_culture or not target_culture:
            return text
        
        adapted_text = text
        
        # Adapt directness
        source_directness = None
        target_directness = None
        
        for level, contexts in self.cultural_norms["directness"].items():
            if source_context in contexts:
                source_directness = level
            if target_context in contexts:
                target_directness = level
        
        if source_directness and target_directness and source_directness != target_directness:
            # Adapt directness level
            if source_directness == "high" and target_directness == "low":
                # Make more indirect
                adapted_text = self._make_more_indirect(adapted_text, target_culture.language)
            elif source_directness == "low" and target_directness == "high":
                # Make more direct
                adapted_text = self._make_more_direct(adapted_text, target_culture.language)
        
        return adapted_text
    
    def _make_more_indirect(self, text: str, language: str) -> str:
        """Make text more indirect for high-context cultures."""
        
        indirect_markers = {
            "en": ["perhaps", "maybe", "possibly", "I think", "it seems"],
            "es": ["quizás", "tal vez", "posiblemente", "creo que", "parece que"],
            "fr": ["peut-être", "possiblement", "je pense que", "il semble que"],
            "de": ["vielleicht", "möglicherweise", "ich denke", "es scheint"],
            "ja": ["かもしれません", "多分", "思います", "ようです"],
            "zh": ["也许", "可能", "我想", "似乎"]
        }
        
        if language in indirect_markers:
            markers = indirect_markers[language]
            # Add indirect marker at the beginning
            marker = random.choice(markers)
            return f"{marker}, {text}"
        
        return text
    
    def _make_more_direct(self, text: str, language: str) -> str:
        """Make text more direct for low-context cultures."""
        
        # Remove indirect markers
        indirect_patterns = {
            "en": r"\b(perhaps|maybe|possibly|I think|it seems)\b,?\s*",
            "es": r"\b(quizás|tal vez|posiblemente|creo que|parece que)\b,?\s*",
            "fr": r"\b(peut-être|possiblement|je pense que|il semble que)\b,?\s*",
            "de": r"\b(vielleicht|möglicherweise|ich denke|es scheint)\b,?\s*"
        }
        
        if language in indirect_patterns:
            adapted_text = re.sub(indirect_patterns[language], "", text, flags=re.IGNORECASE)
            return adapted_text
        
        return text
    
    def adapt_culturally(self, text: str, source_context: str, target_context: str) -> CulturalAdaptation:
        """Apply comprehensive cultural adaptation."""
        
        print(f"🌍 Cultural Adaptation: '{text}'")
        print(f"Source: {source_context} → Target: {target_context}")
        print("-" * 70)
        
        adapted_text = text
        adaptations_applied = []
        confidence = 1.0
        reasoning = []
        
        # Step 1: Adapt idiomatic expressions
        original_text = adapted_text
        adapted_text = self.adapt_idiomatic_expressions(adapted_text, 
                                                       self.cultural_contexts[source_context].language,
                                                       self.cultural_contexts[target_context].language)
        if adapted_text != original_text:
            adaptations_applied.append("idiomatic_expressions")
            reasoning.append("Adapted idiomatic expressions to target language equivalents")
        
        # Step 2: Adapt politeness level
        original_text = adapted_text
        adapted_text = self.adapt_politeness_level(adapted_text, source_context, target_context)
        if adapted_text != original_text:
            adaptations_applied.append("politeness_level")
            reasoning.append("Adapted politeness level to target cultural norms")
        
        # Step 3: Apply cultural norms
        original_text = adapted_text
        adapted_text = self.apply_cultural_norms(adapted_text, source_context, target_context)
        if adapted_text != original_text:
            adaptations_applied.append("cultural_norms")
            reasoning.append("Applied cultural norm adaptations")
        
        # Calculate confidence based on adaptations applied
        if len(adaptations_applied) > 0:
            confidence = 0.8 + (len(adaptations_applied) * 0.05)
            confidence = min(confidence, 1.0)
        
        # Show adaptation details
        print(f"📝 Original Text: {text}")
        print(f"🎯 Adapted Text: {adapted_text}")
        print(f"🔧 Adaptations Applied: {adaptations_applied}")
        print(f"📊 Confidence: {confidence:.2f}")
        print(f"💭 Reasoning: {'; '.join(reasoning)}")
        
        # Show cultural context comparison
        source_culture = self.cultural_contexts[source_context]
        target_culture = self.cultural_contexts[target_context]
        
        print(f"\n🌍 Cultural Context Comparison:")
        print(f"Source ({source_context}):")
        for norm, value in source_culture.cultural_norms.items():
            print(f"  {norm}: {value}")
        
        print(f"Target ({target_context}):")
        for norm, value in target_culture.cultural_norms.items():
            print(f"  {norm}: {value}")
        
        return CulturalAdaptation(
            original_text=text,
            adapted_text=adapted_text,
            cultural_context=target_culture,
            adaptations_applied=adaptations_applied,
            confidence=confidence,
            reasoning="; ".join(reasoning)
        )

def demonstrate_cultural_adaptation():
    """Demonstrate the cultural adaptation system."""
    
    print("🌍 CULTURAL ADAPTATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()
    
    cultural_system = CulturalAdaptationSystem()
    
    # Test cases for cultural adaptation
    test_cases = [
        {
            "text": "Hey dude, can you help me with this piece of cake?",
            "source": "en_US",
            "target": "ja_JP",
            "description": "Informal English to formal Japanese"
        },
        {
            "text": "Good morning, sir. Would you please assist me with this very easy task?",
            "source": "en_GB",
            "target": "es_MX",
            "description": "Formal English to informal Spanish"
        },
        {
            "text": "Hola, ¿podrías ayudarme con esta tarea?",
            "source": "es_ES",
            "target": "de_DE",
            "description": "Spanish to German"
        },
        {
            "text": "Bonjour, pourriez-vous m'aider avec cette tâche?",
            "source": "fr_FR",
            "target": "zh_CN",
            "description": "French to Chinese"
        },
        {
            "text": "Break a leg on your presentation!",
            "source": "en_US",
            "target": "es_ES",
            "description": "English idiom to Spanish"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🎯 EXAMPLE {i}: {test_case['description']}")
        print(f"Text: '{test_case['text']}'")
        print("-" * 60)
        
        try:
            adaptation = cultural_system.adapt_culturally(
                test_case['text'], 
                test_case['source'], 
                test_case['target']
            )
            
            print(f"\n📊 ADAPTATION RESULTS:")
            print(f"  Confidence: {adaptation.confidence:.2f}")
            print(f"  Adaptations: {len(adaptation.adaptations_applied)}")
            print(f"  Reasoning: {adaptation.reasoning}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print()

if __name__ == "__main__":
    demonstrate_cultural_adaptation()
