#!/usr/bin/env python3
"""
Language Expansion Module

This module expands the universal translator to support generation
for all 10 languages currently supported for detection.
"""

from typing import Dict, List, Optional
import logging

from ..domain.models import Language

logger = logging.getLogger(__name__)

class LanguageExpansion:
    """Expands language support for the universal translator."""
    
    def __init__(self):
        """Initialize the language expansion module."""
        self.extended_mappings = {}
        self.grammar_rules = {}
        self._load_extended_mappings()
        self._load_extended_grammar_rules()
    
    def _load_extended_mappings(self):
        """Load extended language mappings for all supported languages."""
        
        # English mappings (comprehensive - 65 NSM primes)
        self.extended_mappings[Language.ENGLISH] = {
            # Phase 1: Substantives (7 primes)
            "I": "I", "YOU": "you", "SOMEONE": "someone", "PEOPLE": "people",
            "SOMETHING": "something", "THING": "thing", "BODY": "body",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "kind", "PART": "part",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "this", "THE_SAME": "the same", "OTHER": "other",
            "ONE": "one", "TWO": "two", "SOME": "some", "ALL": "all",
            "MUCH": "much", "MANY": "many",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "good", "BAD": "bad", "BIG": "big", "SMALL": "small",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "think", "KNOW": "know", "WANT": "want",
            "FEEL": "feel", "SEE": "see", "HEAR": "hear",
            
            # Phase 6: Speech (4 primes)
            "SAY": "say", "WORDS": "words", "TRUE": "true", "FALSE": "false",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "do", "HAPPEN": "happen", "MOVE": "move", "TOUCH": "touch",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "somewhere", "THERE_IS": "there is",
            "HAVE": "have", "BE_SOMEONE": "someone",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "live", "DIE": "die",
            
            # Phase 10: Time (8 primes)
            "WHEN": "when", "NOW": "now", "BEFORE": "before", "AFTER": "after",
            "A_LONG_TIME": "a long time", "A_SHORT_TIME": "a short time",
            "FOR_SOME_TIME": "for some time", "MOMENT": "moment",
            
            # Phase 11: Space (7 primes)
            "WHERE": "where", "HERE": "here", "ABOVE": "above", "BELOW": "below",
            "FAR": "far", "NEAR": "near", "INSIDE": "inside",
            
            # Logical concepts (5 primes)
            "NOT": "not", "MAYBE": "maybe", "CAN": "can",
            "BECAUSE": "because", "IF": "if",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "very", "MORE": "more", "LIKE": "like",
            
            # Additional UD primes (4 primes)
            "ABILITY": "ability", "OBLIGATION": "obligation", "AGAIN": "again", "FINISH": "finish"
        }
        
        # Spanish mappings (comprehensive - 65 NSM primes)
        self.extended_mappings[Language.SPANISH] = {
            # Phase 1: Substantives (7 primes)
            "I": "yo", "YOU": "tú", "SOMEONE": "alguien", "PEOPLE": "gente",
            "SOMETHING": "algo", "THING": "cosa", "BODY": "cuerpo",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "tipo", "PART": "parte",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "esto", "THE_SAME": "lo mismo", "OTHER": "otro",
            "ONE": "uno", "TWO": "dos", "SOME": "algunos", "ALL": "todos",
            "MUCH": "mucho", "MANY": "muchos",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "bueno", "BAD": "malo", "BIG": "grande", "SMALL": "pequeño",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "pensar", "KNOW": "saber", "WANT": "querer",
            "FEEL": "sentir", "SEE": "ver", "HEAR": "oír",
            
            # Phase 6: Speech (4 primes)
            "SAY": "decir", "WORDS": "palabras", "TRUE": "verdadero", "FALSE": "falso",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "hacer", "HAPPEN": "pasar", "MOVE": "mover", "TOUCH": "tocar",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "en algún lugar", "THERE_IS": "hay",
            "HAVE": "tener", "BE_SOMEONE": "ser alguien",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "vivir", "DIE": "morir",
            
            # Phase 10: Time (8 primes)
            "WHEN": "cuándo", "NOW": "ahora", "BEFORE": "antes", "AFTER": "después",
            "A_LONG_TIME": "mucho tiempo", "A_SHORT_TIME": "poco tiempo",
            "FOR_SOME_TIME": "por algún tiempo", "MOMENT": "momento",
            
            # Phase 11: Space (7 primes)
            "WHERE": "dónde", "HERE": "aquí", "ABOVE": "arriba", "BELOW": "abajo",
            "FAR": "lejos", "NEAR": "cerca", "INSIDE": "dentro",
            
            # Logical concepts (5 primes)
            "NOT": "no", "MAYBE": "tal vez", "CAN": "poder",
            "BECAUSE": "porque", "IF": "si",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "muy", "MORE": "más", "LIKE": "como",
            
            # Additional UD primes (4 primes)
            "ABILITY": "habilidad", "OBLIGATION": "obligación", "AGAIN": "otra vez", "FINISH": "terminar"
        }
        
        # French mappings (comprehensive - 65 NSM primes)
        self.extended_mappings[Language.FRENCH] = {
            # Phase 1: Substantives (7 primes)
            "I": "je", "YOU": "tu", "SOMEONE": "quelqu'un", "PEOPLE": "gens",
            "SOMETHING": "quelque chose", "THING": "chose", "BODY": "corps",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "genre", "PART": "partie",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "ceci", "THE_SAME": "le même", "OTHER": "autre",
            "ONE": "un", "TWO": "deux", "SOME": "quelques", "ALL": "tous",
            "MUCH": "beaucoup", "MANY": "beaucoup",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "bon", "BAD": "mauvais", "BIG": "grand", "SMALL": "petit",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "penser", "KNOW": "savoir", "WANT": "vouloir",
            "FEEL": "sentir", "SEE": "voir", "HEAR": "entendre",
            
            # Phase 6: Speech (4 primes)
            "SAY": "dire", "WORDS": "mots", "TRUE": "vrai", "FALSE": "faux",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "faire", "HAPPEN": "arriver", "MOVE": "bouger", "TOUCH": "toucher",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "quelque part", "THERE_IS": "il y a",
            "HAVE": "avoir", "BE_SOMEONE": "être quelqu'un",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "vivre", "DIE": "mourir",
            
            # Phase 10: Time (8 primes)
            "WHEN": "quand", "NOW": "maintenant", "BEFORE": "avant", "AFTER": "après",
            "A_LONG_TIME": "longtemps", "A_SHORT_TIME": "peu de temps",
            "FOR_SOME_TIME": "pendant quelque temps", "MOMENT": "moment",
            
            # Phase 11: Space (7 primes)
            "WHERE": "où", "HERE": "ici", "ABOVE": "au-dessus", "BELOW": "en-dessous",
            "FAR": "loin", "NEAR": "près", "INSIDE": "dedans",
            
            # Logical concepts (5 primes)
            "NOT": "ne pas", "MAYBE": "peut-être", "CAN": "pouvoir",
            "BECAUSE": "parce que", "IF": "si",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "très", "MORE": "plus", "LIKE": "comme",
            
            # Additional UD primes (4 primes)
            "ABILITY": "capacité", "OBLIGATION": "obligation", "AGAIN": "encore", "FINISH": "finir"
        }
        
        # German mappings (comprehensive - 65 NSM primes)
        self.extended_mappings[Language.GERMAN] = {
            # Phase 1: Substantives (7 primes)
            "I": "ich", "YOU": "du", "SOMEONE": "jemand", "PEOPLE": "leute",
            "SOMETHING": "etwas", "THING": "ding", "BODY": "körper",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "art", "PART": "teil",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "dies", "THE_SAME": "das gleiche", "OTHER": "anderer",
            "ONE": "eins", "TWO": "zwei", "SOME": "einige", "ALL": "alle",
            "MUCH": "viel", "MANY": "viele",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "gut", "BAD": "schlecht", "BIG": "groß", "SMALL": "klein",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "denken", "KNOW": "wissen", "WANT": "wollen",
            "FEEL": "fühlen", "SEE": "sehen", "HEAR": "hören",
            
            # Phase 6: Speech (4 primes)
            "SAY": "sagen", "WORDS": "worte", "TRUE": "wahr", "FALSE": "falsch",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "tun", "HAPPEN": "passieren", "MOVE": "bewegen", "TOUCH": "berühren",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "irgendwo", "THERE_IS": "es gibt",
            "HAVE": "haben", "BE_SOMEONE": "jemand sein",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "leben", "DIE": "sterben",
            
            # Phase 10: Time (8 primes)
            "WHEN": "wann", "NOW": "jetzt", "BEFORE": "vor", "AFTER": "nach",
            "A_LONG_TIME": "lange zeit", "A_SHORT_TIME": "kurze zeit",
            "FOR_SOME_TIME": "für einige zeit", "MOMENT": "moment",
            
            # Phase 11: Space (7 primes)
            "WHERE": "wo", "HERE": "hier", "ABOVE": "über", "BELOW": "unter",
            "FAR": "weit", "NEAR": "nah", "INSIDE": "innen",
            
            # Logical concepts (5 primes)
            "NOT": "nicht", "MAYBE": "vielleicht", "CAN": "können",
            "BECAUSE": "weil", "IF": "wenn",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "sehr", "MORE": "mehr", "LIKE": "wie",
            
            # Additional UD primes (4 primes)
            "ABILITY": "fähigkeit", "OBLIGATION": "verpflichtung", "AGAIN": "wieder", "FINISH": "beenden"
        }
        
        # Italian mappings (comprehensive - 65 NSM primes)
        self.extended_mappings[Language.ITALIAN] = {
            # Phase 1: Substantives (7 primes)
            "I": "io", "YOU": "tu", "SOMEONE": "qualcuno", "PEOPLE": "gente",
            "SOMETHING": "qualcosa", "THING": "cosa", "BODY": "corpo",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "tipo", "PART": "parte",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "questo", "THE_SAME": "lo stesso", "OTHER": "altro",
            "ONE": "uno", "TWO": "due", "SOME": "alcuni", "ALL": "tutti",
            "MUCH": "molto", "MANY": "molti",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "buono", "BAD": "cattivo", "BIG": "grande", "SMALL": "piccolo",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "pensare", "KNOW": "sapere", "WANT": "volere",
            "FEEL": "sentire", "SEE": "vedere", "HEAR": "sentire",
            
            # Phase 6: Speech (4 primes)
            "SAY": "dire", "WORDS": "parole", "TRUE": "vero", "FALSE": "falso",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "fare", "HAPPEN": "accadere", "MOVE": "muovere", "TOUCH": "toccare",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "da qualche parte", "THERE_IS": "c'è",
            "HAVE": "avere", "BE_SOMEONE": "essere qualcuno",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "vivere", "DIE": "morire",
            
            # Phase 10: Time (8 primes)
            "WHEN": "quando", "NOW": "ora", "BEFORE": "prima", "AFTER": "dopo",
            "A_LONG_TIME": "lungo tempo", "A_SHORT_TIME": "poco tempo",
            "FOR_SOME_TIME": "per qualche tempo", "MOMENT": "momento",
            
            # Phase 11: Space (7 primes)
            "WHERE": "dove", "HERE": "qui", "ABOVE": "sopra", "BELOW": "sotto",
            "FAR": "lontano", "NEAR": "vicino", "INSIDE": "dentro",
            
            # Logical concepts (5 primes)
            "NOT": "non", "MAYBE": "forse", "CAN": "potere",
            "BECAUSE": "perché", "IF": "se",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "molto", "MORE": "più", "LIKE": "come",
            
            # Additional UD primes (4 primes)
            "ABILITY": "abilità", "OBLIGATION": "obbligo", "AGAIN": "di nuovo", "FINISH": "finire"
        }
        
        # Portuguese mappings (comprehensive - 69 primes)
        self.extended_mappings[Language.PORTUGUESE] = {
            # Phase 1: Substantives (7 primes)
            "I": "eu", "YOU": "tu", "SOMEONE": "alguém", "PEOPLE": "gente",
            "SOMETHING": "algo", "THING": "coisa", "BODY": "corpo",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "tipo", "PART": "parte",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "isto", "THE_SAME": "o mesmo", "OTHER": "outro",
            "ONE": "um", "TWO": "dois", "SOME": "alguns", "ALL": "todos",
            "MUCH": "muito", "MANY": "muitos",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "bom", "BAD": "mau", "BIG": "grande", "SMALL": "pequeno",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "pensar", "KNOW": "saber", "WANT": "querer",
            "FEEL": "sentir", "SEE": "ver", "HEAR": "ouvir",
            
            # Phase 6: Speech (4 primes)
            "SAY": "dizer", "WORDS": "palavras", "TRUE": "verdadeiro", "FALSE": "falso",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "fazer", "HAPPEN": "acontecer", "MOVE": "mover", "TOUCH": "tocar",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "em algum lugar", "THERE_IS": "há",
            "HAVE": "ter", "BE_SOMEONE": "ser alguém",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "viver", "DIE": "morrer",
            
            # Phase 10: Time (8 primes)
            "WHEN": "quando", "NOW": "agora", "BEFORE": "antes", "AFTER": "depois",
            "A_LONG_TIME": "muito tempo", "A_SHORT_TIME": "pouco tempo",
            "FOR_SOME_TIME": "por algum tempo", "MOMENT": "momento",
            
            # Phase 11: Space (7 primes)
            "WHERE": "onde", "HERE": "aqui", "ABOVE": "acima", "BELOW": "abaixo",
            "FAR": "longe", "NEAR": "perto", "INSIDE": "dentro",
            
            # Logical concepts (5 primes)
            "NOT": "não", "MAYBE": "talvez", "CAN": "poder",
            "BECAUSE": "porque", "IF": "se",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "muito", "MORE": "mais", "LIKE": "como",
            
            # Additional UD primes (4 primes)
            "ABILITY": "habilidade", "OBLIGATION": "obrigação", "AGAIN": "novamente", "FINISH": "terminar"
        }
        
        # Russian mappings (comprehensive - 69 primes)
        self.extended_mappings[Language.RUSSIAN] = {
            # Phase 1: Substantives (7 primes)
            "I": "я", "YOU": "ты", "SOMEONE": "кто-то", "PEOPLE": "люди",
            "SOMETHING": "что-то", "THING": "вещь", "BODY": "тело",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "вид", "PART": "часть",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "это", "THE_SAME": "то же самое", "OTHER": "другой",
            "ONE": "один", "TWO": "два", "SOME": "некоторые", "ALL": "все",
            "MUCH": "много", "MANY": "много",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "хороший", "BAD": "плохой", "BIG": "большой", "SMALL": "маленький",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "думать", "KNOW": "знать", "WANT": "хотеть",
            "FEEL": "чувствовать", "SEE": "видеть", "HEAR": "слышать",
            
            # Phase 6: Speech (4 primes)
            "SAY": "говорить", "WORDS": "слова", "TRUE": "правда", "FALSE": "ложь",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "делать", "HAPPEN": "происходить", "MOVE": "двигаться", "TOUCH": "трогать",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "где-то", "THERE_IS": "есть",
            "HAVE": "иметь", "BE_SOMEONE": "быть кем-то",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "жить", "DIE": "умирать",
            
            # Phase 10: Time (8 primes)
            "WHEN": "когда", "NOW": "сейчас", "BEFORE": "до", "AFTER": "после",
            "A_LONG_TIME": "долгое время", "A_SHORT_TIME": "короткое время",
            "FOR_SOME_TIME": "некоторое время", "MOMENT": "момент",
            
            # Phase 11: Space (7 primes)
            "WHERE": "где", "HERE": "здесь", "ABOVE": "выше", "BELOW": "ниже",
            "FAR": "далеко", "NEAR": "близко", "INSIDE": "внутри",
            
            # Logical concepts (5 primes)
            "NOT": "не", "MAYBE": "может быть", "CAN": "мочь",
            "BECAUSE": "потому что", "IF": "если",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "очень", "MORE": "больше", "LIKE": "как",
            
            # Additional UD primes (4 primes)
            "ABILITY": "способность", "OBLIGATION": "обязанность", "AGAIN": "снова", "FINISH": "закончить"
        }
        
        # Chinese mappings (comprehensive - 69 primes)
        self.extended_mappings[Language.CHINESE] = {
            # Phase 1: Substantives (7 primes)
            "I": "我", "YOU": "你", "SOMEONE": "某人", "PEOPLE": "人们",
            "SOMETHING": "某事", "THING": "东西", "BODY": "身体",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "种类", "PART": "部分",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "这个", "THE_SAME": "相同", "OTHER": "其他",
            "ONE": "一", "TWO": "二", "SOME": "一些", "ALL": "所有",
            "MUCH": "很多", "MANY": "很多",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "好", "BAD": "坏", "BIG": "大", "SMALL": "小",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "想", "KNOW": "知道", "WANT": "想要",
            "FEEL": "感觉", "SEE": "看", "HEAR": "听",
            
            # Phase 6: Speech (4 primes)
            "SAY": "说", "WORDS": "话", "TRUE": "真", "FALSE": "假",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "做", "HAPPEN": "发生", "MOVE": "移动", "TOUCH": "触摸",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "某处", "THERE_IS": "有",
            "HAVE": "有", "BE_SOMEONE": "是某人",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "生活", "DIE": "死",
            
            # Phase 10: Time (8 primes)
            "WHEN": "什么时候", "NOW": "现在", "BEFORE": "之前", "AFTER": "之后",
            "A_LONG_TIME": "很长时间", "A_SHORT_TIME": "短时间",
            "FOR_SOME_TIME": "一段时间", "MOMENT": "时刻",
            
            # Phase 11: Space (7 primes)
            "WHERE": "哪里", "HERE": "这里", "ABOVE": "上面", "BELOW": "下面",
            "FAR": "远", "NEAR": "近", "INSIDE": "里面",
            
            # Logical concepts (5 primes)
            "NOT": "不", "MAYBE": "也许", "CAN": "能",
            "BECAUSE": "因为", "IF": "如果",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "很", "MORE": "更多", "LIKE": "像",
            
            # Additional UD primes (4 primes)
            "ABILITY": "能力", "OBLIGATION": "义务", "AGAIN": "再次", "FINISH": "完成"
        }
        
        # Japanese mappings (comprehensive - 69 primes)
        self.extended_mappings[Language.JAPANESE] = {
            # Phase 1: Substantives (7 primes)
            "I": "私", "YOU": "あなた", "SOMEONE": "誰か", "PEOPLE": "人々",
            "SOMETHING": "何か", "THING": "物", "BODY": "体",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "種類", "PART": "部分",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "これ", "THE_SAME": "同じ", "OTHER": "他の",
            "ONE": "一", "TWO": "二", "SOME": "いくつか", "ALL": "すべて",
            "MUCH": "たくさん", "MANY": "多い",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "良い", "BAD": "悪い", "BIG": "大きい", "SMALL": "小さい",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "考える", "KNOW": "知る", "WANT": "欲しい",
            "FEEL": "感じる", "SEE": "見る", "HEAR": "聞く",
            
            # Phase 6: Speech (4 primes)
            "SAY": "言う", "WORDS": "言葉", "TRUE": "真", "FALSE": "偽",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "する", "HAPPEN": "起こる", "MOVE": "動く", "TOUCH": "触る",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "どこか", "THERE_IS": "ある",
            "HAVE": "持つ", "BE_SOMEONE": "誰かである",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "生きる", "DIE": "死ぬ",
            
            # Phase 10: Time (8 primes)
            "WHEN": "いつ", "NOW": "今", "BEFORE": "前", "AFTER": "後",
            "A_LONG_TIME": "長い時間", "A_SHORT_TIME": "短い時間",
            "FOR_SOME_TIME": "しばらく", "MOMENT": "瞬間",
            
            # Phase 11: Space (7 primes)
            "WHERE": "どこ", "HERE": "ここ", "ABOVE": "上", "BELOW": "下",
            "FAR": "遠い", "NEAR": "近い", "INSIDE": "中",
            
            # Logical concepts (5 primes)
            "NOT": "ない", "MAYBE": "多分", "CAN": "できる",
            "BECAUSE": "なぜなら", "IF": "もし",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "とても", "MORE": "もっと", "LIKE": "ように",
            
            # Additional UD primes (4 primes)
            "ABILITY": "能力", "OBLIGATION": "義務", "AGAIN": "再び", "FINISH": "終わる"
        }
        
        # Korean mappings (comprehensive - 69 primes)
        self.extended_mappings[Language.KOREAN] = {
            # Phase 1: Substantives (7 primes)
            "I": "나", "YOU": "너", "SOMEONE": "누군가", "PEOPLE": "사람들",
            "SOMETHING": "무언가", "THING": "것", "BODY": "몸",
            
            # Phase 2: Relational substantives (2 primes)
            "KIND": "종류", "PART": "부분",
            
            # Phase 3: Determiners and quantifiers (9 primes)
            "THIS": "이것", "THE_SAME": "같은", "OTHER": "다른",
            "ONE": "하나", "TWO": "둘", "SOME": "일부", "ALL": "모든",
            "MUCH": "많은", "MANY": "많은",
            
            # Phase 4: Evaluators and descriptors (4 primes)
            "GOOD": "좋은", "BAD": "나쁜", "BIG": "큰", "SMALL": "작은",
            
            # Phase 5: Mental predicates (6 primes)
            "THINK": "생각하다", "KNOW": "알다", "WANT": "원하다",
            "FEEL": "느끼다", "SEE": "보다", "HEAR": "듣다",
            
            # Phase 6: Speech (4 primes)
            "SAY": "말하다", "WORDS": "말", "TRUE": "참", "FALSE": "거짓",
            
            # Phase 7: Actions and events (4 primes)
            "DO": "하다", "HAPPEN": "일어나다", "MOVE": "움직이다", "TOUCH": "만지다",
            
            # Phase 8: Location, existence, possession, specification (4 primes)
            "BE_SOMEWHERE": "어딘가", "THERE_IS": "있다",
            "HAVE": "가지다", "BE_SOMEONE": "누군가이다",
            
            # Phase 9: Life and death (2 primes)
            "LIVE": "살다", "DIE": "죽다",
            
            # Phase 10: Time (8 primes)
            "WHEN": "언제", "NOW": "지금", "BEFORE": "전", "AFTER": "후",
            "A_LONG_TIME": "오랜 시간", "A_SHORT_TIME": "짧은 시간",
            "FOR_SOME_TIME": "잠시", "MOMENT": "순간",
            
            # Phase 11: Space (7 primes)
            "WHERE": "어디", "HERE": "여기", "ABOVE": "위", "BELOW": "아래",
            "FAR": "멀리", "NEAR": "가까이", "INSIDE": "안",
            
            # Logical concepts (5 primes)
            "NOT": "아니다", "MAYBE": "아마", "CAN": "할 수 있다",
            "BECAUSE": "왜냐하면", "IF": "만약",
            
            # Intensifier and augmentor (3 primes)
            "VERY": "매우", "MORE": "더", "LIKE": "같이",
            
            # Additional UD primes (4 primes)
            "ABILITY": "능력", "OBLIGATION": "의무", "AGAIN": "다시", "FINISH": "끝내다"
        }
    
    def _load_extended_grammar_rules(self):
        """Load extended grammar rules for all languages."""
        
        # English grammar rules
        self.grammar_rules[Language.ENGLISH] = {
            "word_order": "SVO",
            "adjective_position": "before_noun",
            "adverb_position": "before_verb",
            "negation_word": "not",
            "question_inversion": True,
            "articles": ["a", "an", "the"],
            "auxiliary_verbs": ["be", "have", "do", "can", "will", "would", "should"]
        }
        
        # Spanish grammar rules
        self.grammar_rules[Language.SPANISH] = {
            "word_order": "SVO",
            "adjective_position": "after_noun",
            "adverb_position": "after_verb",
            "negation_word": "no",
            "question_inversion": False,
            "articles": ["el", "la", "los", "las", "un", "una", "unos", "unas"],
            "auxiliary_verbs": ["ser", "estar", "haber", "tener", "poder", "deber"]
        }
        
        # French grammar rules
        self.grammar_rules[Language.FRENCH] = {
            "word_order": "SVO",
            "adjective_position": "after_noun",
            "adverb_position": "after_verb",
            "negation_word": "ne pas",
            "question_inversion": True,
            "articles": ["le", "la", "les", "un", "une", "des"],
            "auxiliary_verbs": ["être", "avoir", "faire", "pouvoir", "devoir"]
        }
        
        # German grammar rules
        self.grammar_rules[Language.GERMAN] = {
            "word_order": "SVO",
            "adjective_position": "before_noun",
            "adverb_position": "before_verb",
            "negation_word": "nicht",
            "question_inversion": True,
            "articles": ["der", "die", "das", "ein", "eine", "eines"],
            "auxiliary_verbs": ["sein", "haben", "werden", "können", "müssen", "sollen"]
        }
        
        # Italian grammar rules
        self.grammar_rules[Language.ITALIAN] = {
            "word_order": "SVO",
            "adjective_position": "after_noun",
            "adverb_position": "after_verb",
            "negation_word": "non",
            "question_inversion": False,
            "articles": ["il", "la", "i", "gli", "le", "un", "una", "uno"],
            "auxiliary_verbs": ["essere", "avere", "fare", "potere", "dovere"]
        }
        
        # Portuguese grammar rules
        self.grammar_rules[Language.PORTUGUESE] = {
            "word_order": "SVO",
            "adjective_position": "after_noun",
            "adverb_position": "after_verb",
            "negation_word": "não",
            "question_inversion": False,
            "articles": ["o", "a", "os", "as", "um", "uma"],
            "auxiliary_verbs": ["ser", "estar", "ter", "fazer", "poder", "dever"]
        }
        
        # Russian grammar rules
        self.grammar_rules[Language.RUSSIAN] = {
            "word_order": "SVO",
            "adjective_position": "before_noun",
            "adverb_position": "before_verb",
            "negation_word": "не",
            "question_inversion": False,
            "articles": [],  # Russian doesn't use articles
            "auxiliary_verbs": ["быть", "иметь", "делать", "мочь", "должен"]
        }
        
        # Chinese grammar rules
        self.grammar_rules[Language.CHINESE] = {
            "word_order": "SVO",
            "adjective_position": "before_noun",
            "adverb_position": "before_verb",
            "negation_word": "不",
            "question_inversion": False,
            "articles": [],  # Chinese doesn't use articles
            "auxiliary_verbs": ["是", "有", "能", "会", "要"]
        }
        
        # Japanese grammar rules
        self.grammar_rules[Language.JAPANESE] = {
            "word_order": "SOV",  # Subject-Object-Verb
            "adjective_position": "before_noun",
            "adverb_position": "before_verb",
            "negation_word": "ない",
            "question_inversion": False,
            "articles": [],  # Japanese doesn't use articles
            "auxiliary_verbs": ["です", "ある", "いる", "できる", "する"]
        }
        
        # Korean grammar rules
        self.grammar_rules[Language.KOREAN] = {
            "word_order": "SOV",  # Subject-Object-Verb
            "adjective_position": "before_noun",
            "adverb_position": "before_verb",
            "negation_word": "아니다",
            "question_inversion": False,
            "articles": [],  # Korean doesn't use articles
            "auxiliary_verbs": ["이다", "있다", "없다", "하다", "되다"]
        }
    
    def get_mappings(self, language: Language) -> Dict[str, str]:
        """Get language mappings for a specific language."""
        return self.extended_mappings.get(language, {})
    
    def get_grammar_rules(self, language: Language) -> Dict[str, any]:
        """Get grammar rules for a specific language."""
        return self.grammar_rules.get(language, {})
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of languages supported for generation."""
        return list(self.extended_mappings.keys())
    
    def get_coverage_stats(self, language: Language) -> Dict[str, any]:
        """Get coverage statistics for a language."""
        mappings = self.extended_mappings.get(language, {})
        total_primes = 65  # Total NSM primes
        
        return {
            "total_primes": total_primes,
            "mapped_primes": len(mappings),
            "coverage_percentage": (len(mappings) / total_primes * 100) if total_primes > 0 else 0,
            "grammar_rules_available": language in self.grammar_rules
        }
    
    def validate_language_support(self, language: Language) -> bool:
        """Validate if a language is fully supported for generation."""
        return (language in self.extended_mappings and 
                language in self.grammar_rules and
                len(self.extended_mappings[language]) >= 30)  # Minimum threshold
