"""Optional SRL/UD-style detectors using spaCy if available.

These detectors use light dependency patterns to infer primitives from text.
If spaCy or the language model is not available, the functions return empty lists.
"""

from typing import List, Dict, Any

# Define all 65 NSM primes
ALL_NSM_PRIMES = {
    # Phase 1: Substantives
    "I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY",
    # Phase 2: Relational substantives
    "KIND", "PART",
    # Phase 3: Determiners and quantifiers
    "THIS", "THE_SAME", "OTHER", "ONE", "TWO", "SOME", "ALL", "MUCH", "MANY",
    # Phase 4: Evaluators and descriptors
    "GOOD", "BAD", "BIG", "SMALL",
    # Phase 5: Mental predicates
    "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR",
    # Phase 6: Speech
    "SAY", "WORDS", "TRUE", "FALSE",
    # Phase 7: Actions and events
    "DO", "HAPPEN", "MOVE", "TOUCH",
    # Phase 8: Location, existence, possession, specification
    "BE_SOMEWHERE", "THERE_IS", "HAVE", "BE_SOMEONE",
    # Phase 9: Life and death
    "LIVE", "DIE",
    # Phase 10: Time
    "WHEN", "NOW", "BEFORE", "AFTER", "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT",
    # Phase 11: Space
    "WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "SIDE", "INSIDE", "TOUCH",
    # Logical concepts
    "NOT", "MAYBE", "CAN", "BECAUSE", "IF",
    # Intensifier and augmentor
    "VERY", "MORE",
    # Similarity
    "LIKE"
}

try:
	from spacy.matcher import DependencyMatcher  # type: ignore
except Exception:  # pragma: no cover
	DependencyMatcher = None  # type: ignore


def _load_nlp():
	try:
		import spacy  # type: ignore
		# Try English and Spanish small models
		pipelines: Dict[str, Any] = {}
		for code, model in ("en", "en_core_web_sm"), ("es", "es_core_news_sm"), ("fr", "fr_core_news_sm"):
			try:
				pipelines[code] = spacy.load(model)
			except Exception:
				pipelines[code] = None
		return pipelines
	except Exception:
		return {}


_NLP = _load_nlp()


def _pick_lang(text: str) -> str:
	"""Lightweight language pick without external deps.

	Heuristics:
	- Accented chars map to ES/FR
	- Common stopwords
	- Default EN
	"""
	lower = text.lower()
	
	# French-specific patterns (check first to avoid confusion with Spanish)
	french_patterns = ("d'", "l'", "qu'", "n'", "s'", "c'", "j'", "m'", "t'")
	if any(pattern in lower for pattern in french_patterns):
		return "fr"
	
	# Accented characters
	spanish_accents = set("áéíóúñü")
	french_accents = set("àâçéèêëîïôùûüÿœ")
	if any(c in french_accents for c in lower):
		return "fr"
	if any(c in spanish_accents for c in lower):
		return "es"
	
	# Stopword cues
	es_sw = (" el ", " la ", " los ", " las ", " de ", " que ", " para ", " puede ", " puedo ", " debes ", " ¿puedo ", " ¿puede ", " pocos ", " mayoría ", " todos ", " algunos ")
	fr_sw = (" le ", " la ", " les ", " des ", " que ", " pour ", " pas ", " peut ", " puis ", " devez ", " puis-je ", " peu ", " plupart ", " tous ", " quelques ", " étudiants ", " étudiants ")
	if any(w in f" {lower} " for w in fr_sw):
		return "fr"
	if any(w in f" {lower} " for w in es_sw):
		return "es"
	return "en"


def detect_primitives_spacy(text: str) -> List[str]:
	lang = _pick_lang(text)
	nlp = _NLP.get(lang)
	if not nlp:
		return []
	try:
		doc = nlp(text)
	except Exception:
		return []

	detected: List[str] = []

	# Heuristic IsA: copular constructions: nsubj --cop--> be --attr--> NOUN
	for token in doc:
		# "X is Y" patterns (English) and equivalents in Spanish
		if token.dep_ in {"attr", "acomp", "oprd"} and token.head.lemma_ in {"be", "ser"}:
			subj = [c for c in token.head.children if c.dep_ in {"nsubj", "nsubjpass"}]
			if subj and (token.pos_ in {"NOUN", "PROPN"} or token.tag_ in {"NN", "NNS", "NNP", "NNPS"}):
				if "IsA" not in detected:
					detected.append("IsA")
				break

	# Additional IsA: be + nsubj (fallback when attr not tagged as noun)
	for token in doc:
		if token.lemma_ == "be" and any(c.dep_ in {"nsubj", "nsubjpass"} for c in token.children):
			if "IsA" not in detected:
				detected.append("IsA")
			break

	# Heuristic PartOf: nmod:of pattern "X part of Y"
	for token in doc:
		if token.lemma_.lower() == "part" and token.pos_ in {"NOUN", "PROPN"}:
			# Look for 'of' and its pobj
			of = next((c for c in token.children if c.lower_ == "of" and c.pos_ == "ADP"), None)
			if of is not None:
				pobj = next((c for c in of.children if c.dep_ == "pobj" and c.pos_ in {"NOUN", "PROPN"}), None)
				if pobj is not None:
					detected.append("PartOf")
					break
	# FR: "partie de X"
	for token in doc:
		if token.lemma_.lower() in {"partie"} and token.pos_ == "NOUN":
			de = next((c for c in token.children if c.lower_ == "de" and c.pos_ == "ADP"), None)
			if de is not None:
				pobj = next((c for c in de.children if c.dep_ in {"pobj", "obj"} and c.pos_ in {"NOUN", "PROPN"}), None)
				if pobj is not None:
					detected.append("PartOf")
					break

	# Heuristic Before/After: temporal markers around verbs or events
	for token in doc:
		if token.lemma_.lower() == "before" and token.pos_ in {"SCONJ", "ADP", "ADV"}:
			detected.append("Before")
			break
	for token in doc:
		if token.lemma_.lower() == "after" and token.pos_ in {"SCONJ", "ADP", "ADV"}:
			detected.append("After")
			break

	# Heuristic AtLocation: prepositional locatives
	for token in doc:
		if token.pos_ == "ADP" and token.lemma_.lower() in {"at", "in", "on", "inside", "within", "into", "onto"}:
			if "AtLocation" not in detected:
				detected.append("AtLocation")
			break

	# Heuristic causals and conditionals
	for token in doc:
		if token.lemma_.lower() == "because" and token.pos_ in {"SCONJ", "ADP"}:
			detected.append("Because")
			break
	for token in doc:
		if token.lemma_.lower() in {"therefore", "so"} and token.pos_ in {"ADV", "CCONJ"}:
			detected.append("Therefore")
			break
	for token in doc:
		if token.lemma_.lower() == "if" and token.pos_ in {"SCONJ"}:
			detected.append("If")
			break
	for token in doc:
		if token.lemma_.lower() == "unless" and token.pos_ in {"SCONJ"}:
			detected.append("Unless")
			break

	# Purpose: in order to / to + VERB (ensure 'to' functions as marker)
	for i, token in enumerate(doc):
		if token.text.lower() == "in" and i+2 < len(doc):
			if doc[i+1].lemma_.lower() == "order" and doc[i+2].lemma_.lower() == "to":
				if "InOrderTo" not in detected:
					detected.append("InOrderTo")
				break
	for i, token in enumerate(doc):
		if token.lemma_.lower() == "to" and token.pos_ in {"PART"} and i+1 < len(doc):
			if doc[i+1].pos_ == "VERB":
				if "InOrderTo" not in detected:
					detected.append("InOrderTo")
				break

	# Aspect: progressive (be + VERB+ing), perfect (have + VERB+ed)
	for token in doc:
		if token.lemma_ == "be":
			if any(child.pos_ == "VERB" and child.tag_ in {"VBG"} for child in token.children):
				detected.append("Progressive")
				break
	for token in doc:
		if token.lemma_ == "have":
			if any(child.pos_ == "VERB" and child.tag_ in {"VBN"} for child in token.children):
				detected.append("Perfect")
				break

	# Quantifiers and comparisons (rough)
	for token in doc:
		if token.lemma_.lower() in {"most"}:
			detected.append("Most")
			break
	for token in doc:
		if token.lemma_.lower() in {"no"}:
			detected.append("None")
			break

	# Expanded causal connectives
	for token in doc:
		if token.lemma_.lower() in {"since", "due"}:
			if token.lemma_.lower() == "due" and token.nbor(1).lemma_.lower() if token.i+1 < len(doc) else None == "to":
				detected.append("Because")
				break
			if token.pos_ in {"SCONJ", "ADP"}:
				detected.append("Because")
				break
	for token in doc:
		if token.text.lower() in {"as", "because"} and token.pos_ in {"SCONJ", "ADP"}:
			detected.append("Because")
			break

	# Comprehensive quantifier patterns
	quantifier_patterns = {
		"en": {
			"All": ["all", "every", "each"],
			"Some": ["some", "several", "a few"],
			"Many": ["many", "numerous", "lots of"],
			"Few": ["few", "hardly any", "scarcely any"],
			"Most": ["most", "majority", "the majority of"],
			"None": ["no", "none", "nobody", "nothing"],
			"Only": ["only", "merely", "just"],
			"AtMost": ["at most", "no more than", "maximum"],
			"LessThan": ["less than", "fewer than", "under"]
		},
		"es": {
			"All": ["todos", "todas", "cada", "todo"],
			"Some": ["algunos", "algunas", "varios", "varias"],
			"Many": ["muchos", "muchas", "numerosos", "numerosas"],
			"Few": ["pocos", "pocas", "poco", "poca", "apenas", "escasos"],
			"Most": ["mayoría", "mayor", "mayor parte"],
			"None": ["ningún", "ninguna", "nadie", "nada"],
			"Only": ["solo", "únicamente", "solamente"],
			"AtMost": ["a lo sumo", "no más de", "máximo"],
			"LessThan": ["menos de", "menos que", "bajo"]
		},
		"fr": {
			"All": ["tous", "toutes", "chaque", "tout"],
			"Some": ["quelques", "plusieurs", "certains", "certaines"],
			"Many": ["beaucoup", "nombreux", "nombreuses"],
			"Few": ["peu", "quelques", "à peine"],
			"Most": ["plupart", "majorité", "la plupart de"],
			"None": ["aucun", "aucune", "personne", "rien"],
			"Only": ["seulement", "uniquement", "juste"],
			"AtMost": ["au plus", "pas plus de", "maximum"],
			"LessThan": ["moins de", "moins que", "sous"]
		}
	}
	
	patterns = quantifier_patterns.get(lang, quantifier_patterns["en"])
	
	# Check for quantifier patterns
	text_lower = text.lower()
	
	# Multi-word patterns first (e.g., "at most", "no more than")
	for quant_type, words in patterns.items():
		for word in words:
			if " " in word and word in text_lower:
				if quant_type not in detected:
					detected.append(quant_type)
	
	# Single-word patterns
	for token in doc:
		lemma = token.lemma_.lower()
		
		# Direct determiner patterns
		if token.dep_ == "det":
			for quant_type, words in patterns.items():
				if lemma in words:
					if quant_type not in detected:
						detected.append(quant_type)
					break
		
		# Adjective modifier patterns (e.g., "Few students")
		if token.dep_ == "amod" and token.pos_ == "ADJ":
			for quant_type, words in patterns.items():
				if lemma in words:
					if quant_type not in detected:
						detected.append(quant_type)
					break
		
		# Noun modifier patterns (e.g., "mayoría", "plupart")
		if token.dep_ in {"nmod", "nsubj"} and token.pos_ in {"NOUN", "ADV"}:
			for quant_type, words in patterns.items():
				if lemma in words:
					if quant_type not in detected:
						detected.append(quant_type)
					break
		
		# Root adverb patterns (e.g., "Peu")
		if token.dep_ == "ROOT" and token.pos_ == "ADV":
			for quant_type, words in patterns.items():
				if lemma in words:
					if quant_type not in detected:
						detected.append(quant_type)
					break
		
		# Check for "not all" patterns (wide scope negation)
		if lemma == "not" and any(c.lemma_.lower() in patterns["All"] for c in token.head.children):
			if "NotAll" not in detected:
				detected.append("NotAll")
		
		# Check for "all ... not" patterns (ambiguous scope)
		if lemma in patterns["All"] and any(c.lemma_.lower() == "not" for c in token.head.children):
			if "AllNot" not in detected:
				detected.append("AllNot")

	# Comparatives: advmod(head)=more/less + than
	for token in doc:
		if token.lemma_.lower() in {"more", "less"} and token.dep_ == "advmod":
			head = token.head
			if any(child.lemma_.lower() == "than" for child in head.children):
				detected.append("MoreThan" if token.lemma_.lower() == "more" else "LessThan")
				break

	# Existential: there-expl with 'be'
	for token in doc:
		if token.lemma_ == "be" and any(ch.dep_ == "expl" and ch.lower_ == "there" for ch in token.children):
			if "Exist" not in detected:
				detected.append("Exist")
			break

	# STILL: continuation of state/action
	still_tokens = {"en": ["still"], "es": ["todavía", "aún"], "fr": ["encore", "toujours"]}.get(lang, ["still"])
	for token in doc:
		if token.lemma_.lower() in still_tokens and token.pos_ in {"ADV", "ADJ"}:
			if "Still" not in detected:
				detected.append("Still")
			break

	# NOT_YET: anticipated completion (negative + temporal)
	not_yet_tokens = {"en": ["yet"], "es": ["todavía", "aún"], "fr": ["encore"]}.get(lang, ["yet"])
	for token in doc:
		if token.lemma_.lower() in not_yet_tokens and token.pos_ in {"ADV"}:
			# Check for negation context
			has_negation = any(t.lemma_.lower() in {"not", "no", "n't", "no", "ne"} for t in doc)
			if has_negation and "NotYet" not in detected:
				detected.append("NotYet")
			break

	# START: beginning of action/state
	start_tokens = {"en": ["start", "begin"], "es": ["empezar", "comenzar"], "fr": ["commencer", "débuter"]}.get(lang, ["start"])
	for token in doc:
		if token.lemma_.lower() in start_tokens and token.pos_ == "VERB":
			if "Start" not in detected:
				detected.append("Start")
			break

	# FINISH: completion of action/state
	finish_tokens = {"en": ["finish", "complete", "end"], "es": ["terminar", "acabar", "finalizar"], "fr": ["finir", "terminer", "achever"]}.get(lang, ["finish"])
	for token in doc:
		if token.lemma_.lower() in finish_tokens and token.pos_ == "VERB":
			if "Finish" not in detected:
				detected.append("Finish")
			break

	# AGAIN: repetition of action
	again_tokens = {"en": ["again"], "es": ["otra vez", "de nuevo"], "fr": ["encore", "de nouveau"]}.get(lang, ["again"])
	for token in doc:
		if token.lemma_.lower() in again_tokens and token.pos_ in {"ADV"}:
			if "Again" not in detected:
				detected.append("Again")
			break

	# KEEP: continuation/maintenance of action
	keep_tokens = {"en": ["keep", "continue"], "es": ["seguir", "continuar"], "fr": ["continuer", "garder"]}.get(lang, ["keep"])
	for token in doc:
		if token.lemma_.lower() in keep_tokens and token.pos_ == "VERB":
			if "Keep" not in detected:
				detected.append("Keep")
			break

	# === PHASE 1: CORE SUBSTANTIVES (NSM Primes) - UD-Based Detection ===
	
	# I: first person singular pronoun (nsubj, nsubjpass, or direct object)
	for token in doc:
		if (token.pos_ == "PRON" and 
			token.lemma_.lower() in {"i", "me", "my", "myself", "yo", "je"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "iobj"}):
			if "I" not in detected:
				detected.append("I")
			break

	# YOU: second person pronoun (nsubj, nsubjpass, or direct object)
	for token in doc:
		if (token.pos_ == "PRON" and 
			token.lemma_.lower() in {"you", "your", "yourself", "tú", "tu", "vous"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "iobj"}):
			if "YOU" not in detected:
				detected.append("YOU")
			break

	# SOMEONE: indefinite person (subject, object, or with determiner)
	for token in doc:
		if (token.pos_ in {"PRON", "NOUN"} and 
			token.lemma_.lower() in {"someone", "somebody", "anyone", "anybody", "alguien", "quelqu'un", "person"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			if "SOMEONE" not in detected:
				detected.append("SOMEONE")
			break
	
	# Also detect "a person" as SOMEONE
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"person", "persona", "personne"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			# Check for indefinite determiner
			det = next((c for c in token.children if c.dep_ == "det" and c.lemma_.lower() in {"a", "an", "un", "una", "un", "une"}), None)
			if det:
				if "SOMEONE" not in detected:
					detected.append("SOMEONE")
				break

	# PEOPLE: plural persons (subject, object, or collective noun)
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"people", "persons", "humans", "individuals", "gente", "personnes", "individus"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			if "PEOPLE" not in detected:
				detected.append("PEOPLE")
			break

	# SOMETHING: indefinite thing (subject, object, or with determiner)
	for token in doc:
		if (token.pos_ in {"PRON", "NOUN"} and 
			token.lemma_.lower() in {"something", "anything", "whatever", "algo", "quelque chose", "event", "evento", "événement"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			if "SOMETHING" not in detected:
				detected.append("SOMETHING")
			break
	
	# Also detect "an event" as SOMETHING
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"event", "evento", "événement", "thing", "cosa", "chose"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			# Check for indefinite determiner
			det = next((c for c in token.children if c.dep_ == "det" and c.lemma_.lower() in {"a", "an", "un", "una", "un", "une"}), None)
			if det:
				if "SOMETHING" not in detected:
					detected.append("SOMETHING")
				break

	# THING: generic object (subject, object, or with determiner)
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"thing", "object", "item", "cosa", "objet", "article"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			if "THING" not in detected:
				detected.append("THING")
			break
	
	# Also detect "the object" as THING
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"object", "objeto", "objet"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			if "THING" not in detected:
				detected.append("THING")
			break

	# BODY: physical entity (subject, object, or with determiner)
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"body", "person", "human", "cuerpo", "personne", "humain"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			if "BODY" not in detected:
				detected.append("BODY")
			break
	
	# Also detect "the person" as BODY
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"person", "persona", "personne"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			# Check for definite determiner
			det = next((c for c in token.children if c.dep_ == "det" and c.lemma_.lower() in {"the", "el", "la", "le", "les"}), None)
			if det:
				if "BODY" not in detected:
					detected.append("BODY")
				break

	# === PHASE 2: MENTAL PREDICATES (NSM Primes) - UD-Based Detection ===
	
	# THINK: mental cognition and reasoning
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"think", "believe", "consider", "pensar", "creer", "penser", "croire"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "THINK" not in detected:
				detected.append("THINK")
			break

	# KNOW: mental knowledge and understanding
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"know", "understand", "realize", "saber", "conocer", "savoir", "connaître"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "KNOW" not in detected:
				detected.append("KNOW")
			break

	# WANT: mental desire and intention
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"want", "desire", "wish", "querer", "desear", "vouloir", "souhaiter"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "WANT" not in detected:
				detected.append("WANT")
			break

	# FEEL: mental emotion and sensation
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"feel", "sense", "sentir", "sentir", "ressentir"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "FEEL" not in detected:
				detected.append("FEEL")
			break

	# SEE: sensory visual perception
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"see", "look", "watch", "ver", "mirar", "voir", "regarder"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "SEE" not in detected:
				detected.append("SEE")
			break

	# HEAR: sensory auditory perception
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"hear", "listen", "escuchar", "oir", "entendre", "écouter"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "HEAR" not in detected:
				detected.append("HEAR")
			break

	# === PHASE 3: LOGICAL OPERATORS (NSM Primes) - UD-Based Detection ===
	
	# BECAUSE: logical causation and reasoning
	for token in doc:
		if (token.pos_ in {"SCONJ", "ADV"} and 
			token.lemma_.lower() in {"because", "since", "as", "porque", "ya", "puisque", "car"} and
			token.dep_ in {"mark", "advmod"}):
			if "BECAUSE" not in detected:
				detected.append("BECAUSE")
			break

	# IF: logical condition and implication
	for token in doc:
		if (token.pos_ == "SCONJ" and 
			token.lemma_.lower() in {"if", "si", "si"} and
			token.dep_ == "mark"):
			if "IF" not in detected:
				detected.append("IF")
			break

	# NOT: logical negation
	for token in doc:
		if (token.pos_ in {"PART", "ADV"} and 
			token.lemma_.lower() in {"not", "no", "ne", "pas"} and
			token.dep_ in {"neg", "advmod"}):
			if "NOT" not in detected:
				detected.append("NOT")
			break

	# SAME: logical identity and equivalence
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV"} and 
			token.lemma_.lower() in {"same", "identical", "igual", "même", "pareil"} and
			token.dep_ in {"amod", "advmod", "attr"}):
			if "SAME" not in detected:
				detected.append("SAME")
			break

	# DIFFERENT: logical difference and distinction
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV"} and 
			token.lemma_.lower() in {"different", "distinct", "diferente", "différent", "autre"} and
			token.dep_ in {"amod", "advmod", "attr"}):
			if "DIFFERENT" not in detected:
				detected.append("DIFFERENT")
			break

	# MAYBE: logical possibility and uncertainty
	for token in doc:
		if (token.pos_ in {"ADV", "PART"} and 
			token.lemma_.lower() in {"maybe", "perhaps", "possibly", "tal", "vez", "peut-être"} and
			token.dep_ in {"advmod", "discourse"}):
			if "MAYBE" not in detected:
				detected.append("MAYBE")
			break

	# === PHASE 4: TEMPORAL & CAUSAL (NSM Primes) - UD-Based Detection ===
	
	# BEFORE: temporal precedence and ordering
	for token in doc:
		if (token.pos_ in {"ADP", "ADV", "SCONJ"} and 
			token.lemma_.lower() in {"before", "prior", "antes", "avant"} and
			token.dep_ in {"prep", "advmod", "mark"}):
			if "BEFORE" not in detected:
				detected.append("BEFORE")
			break

	# AFTER: temporal succession and sequence
	for token in doc:
		if (token.pos_ in {"ADP", "ADV", "SCONJ"} and 
			token.lemma_.lower() in {"after", "later", "después", "après"} and
			token.dep_ in {"prep", "advmod", "mark"}):
			if "AFTER" not in detected:
				detected.append("AFTER")
			break

	# WHEN: temporal simultaneity and coincidence
	for token in doc:
		if (token.pos_ in {"ADV", "SCONJ"} and 
			token.lemma_.lower() in {"when", "while", "cuando", "quand"} and
			token.dep_ in {"advmod", "mark"}):
			if "WHEN" not in detected:
				detected.append("WHEN")
			break

	# CAUSE: causal agency and responsibility
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"cause", "lead", "causar", "causer"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "CAUSE" not in detected:
				detected.append("CAUSE")
			break

	# MAKE: causal creation and production
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"make", "create", "hacer", "faire"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "MAKE" not in detected:
				detected.append("MAKE")
			break

	# LET: causal permission and allowance
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"let", "allow", "permitir", "permettre"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "LET" not in detected:
				detected.append("LET")
			break

	# === PHASE 5: SPATIAL & PHYSICAL (NSM Primes) - UD-Based Detection ===
	
	# IN: spatial containment and inclusion
	for token in doc:
		if (token.pos_ == "ADP" and 
			token.lemma_.lower() in {"in", "within", "dentro", "dans"} and
			token.dep_ == "prep"):
			if "IN" not in detected:
				detected.append("IN")
			break

	# ON: spatial support and contact
	for token in doc:
		if (token.pos_ == "ADP" and 
			token.lemma_.lower() in {"on", "upon", "sobre", "sur"} and
			token.dep_ == "prep"):
			if "ON" not in detected:
				detected.append("ON")
			break

	# UNDER: spatial subordination and coverage
	for token in doc:
		if (token.pos_ == "ADP" and 
			token.lemma_.lower() in {"under", "beneath", "bajo", "sous"} and
			token.dep_ == "prep"):
			if "UNDER" not in detected:
				detected.append("UNDER")
			break

	# NEAR: spatial proximity and closeness
	for token in doc:
		if (token.pos_ in {"ADP", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"near", "close", "cerca", "près"} and
			token.dep_ in {"prep", "amod", "advmod"}):
			if "NEAR" not in detected:
				detected.append("NEAR")
			break

	# FAR: spatial distance and separation
	for token in doc:
		if (token.pos_ in {"ADP", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"far", "away", "lejos", "loin"} and
			token.dep_ in {"prep", "amod", "advmod"}):
			if "FAR" not in detected:
				detected.append("FAR")
			break

	# INSIDE: spatial interiority and enclosure
	for token in doc:
		if (token.pos_ in {"ADP", "ADV"} and 
			token.lemma_.lower() in {"inside", "within", "dentro", "dedans"} and
			token.dep_ in {"prep", "advmod"}):
			if "INSIDE" not in detected:
				detected.append("INSIDE")
			break

	# === PHASE 6: QUANTIFIERS (NSM Primes) - UD-Based Detection ===
	
	# ALL: universal quantification and totality
	for token in doc:
		if (token.pos_ in {"DET", "PRON", "ADJ"} and 
			token.lemma_.lower() in {"all", "every", "each", "todo", "tous"} and
			token.dep_ in {"det", "amod", "nsubj", "dobj"}):
			if "ALL" not in detected:
				detected.append("ALL")
			break

	# MANY: large quantity and plurality
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"many", "numerous", "muchos", "nombreux"} and
			token.dep_ in {"det", "amod", "advmod"}):
			if "MANY" not in detected:
				detected.append("MANY")
			break

	# SOME: partial quantity and existence
	for token in doc:
		if (token.pos_ in {"DET", "PRON", "ADJ"} and 
			token.lemma_.lower() in {"some", "several", "algunos", "quelques"} and
			token.dep_ in {"det", "amod", "nsubj", "dobj"}):
			if "SOME" not in detected:
				detected.append("SOME")
			break

	# FEW: small quantity and scarcity
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"few", "little", "pocos", "peu"} and
			token.dep_ in {"det", "amod", "advmod"}):
			if "FEW" not in detected:
				detected.append("FEW")
			break

	# MUCH: large amount and abundance
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"much", "a lot", "mucho", "beaucoup"} and
			token.dep_ in {"det", "amod", "advmod"}):
			if "MUCH" not in detected:
				detected.append("MUCH")
			break

	# LITTLE: small amount and paucity
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"little", "small", "poco", "peu"} and
			token.dep_ in {"det", "amod", "advmod"}):
			if "LITTLE" not in detected:
				detected.append("LITTLE")
			break

	# === PHASE 7: EVALUATORS (NSM Primes) - UD-Based Detection ===
	
	# GOOD: positive evaluation and desirability
	for token in doc:
		if (token.pos_ == "ADJ" and 
			token.lemma_.lower() in {"good", "great", "excellent", "bueno", "bon"} and
			token.dep_ in {"amod", "attr", "pred"}):
			if "GOOD" not in detected:
				detected.append("GOOD")
			break

	# BAD: negative evaluation and undesirability
	for token in doc:
		if (token.pos_ == "ADJ" and 
			token.lemma_.lower() in {"bad", "terrible", "awful", "malo", "mauvais"} and
			token.dep_ in {"amod", "attr", "pred"}):
			if "BAD" not in detected:
				detected.append("BAD")
			break

	# BIG: large size and magnitude
	for token in doc:
		if (token.pos_ == "ADJ" and 
			token.lemma_.lower() in {"big", "large", "huge", "grande", "grand"} and
			token.dep_ in {"amod", "attr", "pred"}):
			if "BIG" not in detected:
				detected.append("BIG")
			break

	# SMALL: small size and magnitude
	for token in doc:
		if (token.pos_ == "ADJ" and 
			token.lemma_.lower() in {"small", "tiny", "little", "pequeño", "petit"} and
			token.dep_ in {"amod", "attr", "pred"}):
			if "SMALL" not in detected:
				detected.append("SMALL")
			break

	# RIGHT: correctness and appropriateness
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV"} and 
			token.lemma_.lower() in {"right", "correct", "proper", "correcto", "juste"} and
			token.dep_ in {"amod", "attr", "pred", "advmod"}):
			if "RIGHT" not in detected:
				detected.append("RIGHT")
			break

	# WRONG: incorrectness and inappropriateness (avoid conflict with FALSE)
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV"} and 
			token.lemma_.lower() in {"wrong", "incorrect", "incorrecto"} and
			token.dep_ in {"amod", "attr", "pred", "advmod"} and
			token.lemma_.lower() not in {"false", "fake", "falso"}):  # Avoid conflict with FALSE
			if "WRONG" not in detected:
				detected.append("WRONG")
			break

	# === PHASE 8: ACTIONS (NSM Primes) - UD-Based Detection ===
	
	# DO: action performance and execution
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"do", "make", "perform", "hacer", "faire"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "DO" not in detected:
				detected.append("DO")
			break

	# HAPPEN: event occurrence and happening
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"happen", "occur", "take", "pasar", "arriver"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "HAPPEN" not in detected:
				detected.append("HAPPEN")
			break

	# MOVE: physical movement and motion
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"move", "go", "walk", "mover", "bouger"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "MOVE" not in detected:
				detected.append("MOVE")
			break

	# TOUCH: physical contact and interaction
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"touch", "contact", "reach", "tocar", "toucher"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "TOUCH" not in detected:
				detected.append("TOUCH")
			break

	# LIVE: existence and being alive
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"live", "exist", "be", "vivir", "vivre"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "LIVE" not in detected:
				detected.append("LIVE")
			break

	# DIE: death and cessation of life
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"die", "death", "end", "morir", "mourir"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			if "DIE" not in detected:
				detected.append("DIE")
			break

	# === PHASE 9: DESCRIPTORS (NSM Primes) - UD-Based Detection ===
	
	# THIS: proximate reference and identification
	for token in doc:
		if (token.pos_ == "DET" and 
			token.lemma_.lower() in {"this", "ese", "ce"} and
			token.dep_ in {"det", "attr"}):
			if "THIS" not in detected:
				detected.append("THIS")
			break

	# THE SAME: identity and sameness
	for token in doc:
		if (token.pos_ in {"DET", "ADJ"} and 
			token.lemma_.lower() in {"same", "mismo", "même"} and
			token.dep_ in {"det", "amod", "attr"}):
			if "THE SAME" not in detected:
				detected.append("THE SAME")
			break

	# OTHER: distinction and difference
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "PRON"} and 
			token.lemma_.lower() in {"other", "otro", "autre"} and
			token.dep_ in {"det", "amod", "attr", "nsubj", "dobj"}):
			if "OTHER" not in detected:
				detected.append("OTHER")
			break

	# ONE: singularity and unity
	for token in doc:
		if (token.pos_ in {"NUM", "DET"} and 
			token.lemma_.lower() in {"one", "uno", "un"} and
			token.dep_ in {"nummod", "det"}):
			if "ONE" not in detected:
				detected.append("ONE")
			break

	# TWO: duality and pairing
	for token in doc:
		if (token.pos_ == "NUM" and 
			token.lemma_.lower() in {"two", "dos", "deux"} and
			token.dep_ in {"nummod"}):
			if "TWO" not in detected:
				detected.append("TWO")
			break

	# SOME: indefinite quantity and selection
	for token in doc:
		if (token.pos_ in {"DET", "PRON"} and 
			token.lemma_.lower() in {"some", "algunos", "quelques"} and
			token.dep_ in {"det", "nsubj", "dobj"}):
			if "SOME" not in detected:
				detected.append("SOME")
			break

	# === PHASE 10: INTENSIFIERS (NSM Primes) - UD-Based Detection ===
	
	# VERY: high degree and intensity
	for token in doc:
		if (token.pos_ == "ADV" and 
			token.lemma_.lower() in {"very", "muy", "très"} and
			token.dep_ in {"advmod", "amod"}):
			if "VERY" not in detected:
				detected.append("VERY")
			break

	# MORE: comparative degree and increase
	for token in doc:
		if (token.pos_ in {"ADV", "ADJ"} and 
			token.lemma_.lower() in {"more", "más", "plus"} and
			token.dep_ in {"advmod", "amod", "attr"}):
			if "MORE" not in detected:
				detected.append("MORE")
			break

	# LIKE: similarity and resemblance
	for token in doc:
		if (token.pos_ in {"ADP", "ADV", "ADJ"} and 
			token.lemma_.lower() in {"like", "como", "comme"} and
			token.dep_ in {"prep", "advmod", "amod"}):
			if "LIKE" not in detected:
				detected.append("LIKE")
			break

	# KIND OF: partial degree and approximation
	for token in doc:
		if (token.pos_ in {"ADV", "ADJ"} and 
			token.lemma_.lower() in {"kind", "tipo", "genre"} and
			token.dep_ in {"advmod", "amod"}):
			if "KIND OF" not in detected:
				detected.append("KIND OF")
			break

	# === PHASE 11: FINAL PRIMES (NSM Primes) - UD-Based Detection ===
	
	# SAY: speech and communication (enhanced detection)
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"say", "tell", "speak", "decir", "dire"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp", "acl", "advcl"}):
			if "SAY" not in detected:
				detected.append("SAY")
			break
	
	# Check for SAY in relative clauses and complex constructions
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"say", "tell", "speak", "decir", "dire"} and
			token.dep_ in {"relcl", "acl:relcl", "advcl"}):
			if "SAY" not in detected:
				detected.append("SAY")
			break

	# WORDS: linguistic expression
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"word", "words", "palabra", "mot"} and
			token.dep_ in {"nsubj", "dobj", "pobj"}):
			if "WORDS" not in detected:
				detected.append("WORDS")
			break

	# TRUE: truth and factuality (enhanced detection)
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV", "NOUN"} and 
			token.lemma_.lower() in {"true", "real", "truth", "verdadero", "vrai", "vérité"} and
			token.dep_ in {"amod", "attr", "pred", "advmod", "nsubj", "dobj", "pobj"}):
			if "TRUE" not in detected:
				detected.append("TRUE")
			break
	
	# Check for copula constructions with TRUE
	for token in doc:
		if (token.lemma_.lower() in {"true", "real", "truth", "verdadero", "vrai", "vérité"} and
			(token.dep_ in {"attr", "pred"} or 
			 (token.dep_ == "acomp" and token.head.pos_ == "AUX")) and
			(token.head.lemma_.lower() in {"be", "is", "are", "was", "were", "ser", "estar", "être"} or
			 any(child.lemma_.lower() in {"be", "is", "are", "was", "were", "ser", "estar", "être"} for child in token.head.children) or
			 token.head.pos_ == "AUX")):
			if "TRUE" not in detected:
				detected.append("TRUE")
			break

	# FALSE: falsity and deception (enhanced detection)
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV", "NOUN"} and 
			token.lemma_.lower() in {"false", "fake", "falsity", "falso", "faux", "fausseté"} and
			token.dep_ in {"amod", "attr", "pred", "advmod", "nsubj", "dobj", "pobj"}):
			if "FALSE" not in detected:
				detected.append("FALSE")
			break
	
	# Check for copula constructions with FALSE
	for token in doc:
		if (token.lemma_.lower() in {"false", "fake", "falsity", "falso", "faux", "fausseté"} and
			(token.dep_ in {"attr", "pred"} or 
			 (token.dep_ == "acomp" and token.head.pos_ == "AUX")) and
			(token.head.lemma_.lower() in {"be", "is", "are", "was", "were", "ser", "estar", "être"} or
			 any(child.lemma_.lower() in {"be", "is", "are", "was", "were", "ser", "estar", "être"} for child in token.head.children) or
			 token.head.pos_ == "AUX")):
			if "FALSE" not in detected:
				detected.append("FALSE")
			break

	# WHERE: location specification (enhanced detection)
	for token in doc:
		if (token.pos_ in {"ADV", "PRON", "SCONJ"} and 
			token.lemma_.lower() in {"where", "dónde", "où", "donde"} and
			token.dep_ in {"advmod", "pobj", "nsubj", "mark", "advcl"}):
			if "WHERE" not in detected:
				detected.append("WHERE")
			break

	# WHEN: time specification (enhanced)
	for token in doc:
		if (token.pos_ in {"ADV", "PRON", "SCONJ"} and 
			token.lemma_.lower() in {"when", "cuándo", "quand"} and
			token.dep_ in {"advmod", "pobj", "mark"}):
			if "WHEN" not in detected:
				detected.append("WHEN")
			break

	return detected



def detect_primitives_structured(text: str) -> List[Dict[str, Any]]:
	"""Detect primitives with simple argument extraction using spaCy.

	Returns a list of dicts: {"name": str, "args": [str, ...], "confidence": float}
	"""
	lang = _pick_lang(text)
	nlp = _NLP.get(lang)
	if not nlp:
		return []
	try:
		doc = nlp(text)
	except Exception:
		return []

	detections: List[Dict[str, Any]] = []

	def add(name: str, args: List[str], conf: float = 0.6) -> None:
		detections.append({"name": name, "args": args, "confidence": conf})

	# Language-specific lexicons
	be_lemmas = {"en": {"be"}, "es": {"ser"}, "fr": {"être"}}.get(lang, {"be"})
	loc_adps = {"en": {"in", "on", "at"}, "es": {"en", "a", "sobre"}, "fr": {"à", "dans", "sur", "en"}}.get(lang, {"in", "on", "at"})
	# Expanded causal cues; note tokenization splits multi-word cues
	because_tokens = {"en": {"because", "since"}, "es": {"porque", "ya", "puesto", "debido"}, "fr": {"parce", "car", "puisque"}}.get(lang, {"because"})
	to_markers = {"en": {"to"}, "es": {"para"}, "fr": {"pour"}}.get(lang, {"to"})
	# Expanded similarity adjectives
	sim_adj = {"en": {"similar"}, "es": {"similar", "parecido", "parecida", "parecidos", "parecidas"}, "fr": {"similaire", "semblable", "pareil", "pareille", "pareils", "pareilles"}}.get(lang, {"similar"})
	like_adp = {"en": {"like"}, "es": {"como"}, "fr": {"comme"}}.get(lang, {"like"})
	# Expanded difference adjectives
	diff_adj = {"en": {"different"}, "es": {"diferente", "distinto", "distinta", "diferentes", "distintos", "distintas"}, "fr": {"différent", "différente", "différents", "différentes"}}.get(lang, {"different"})
	have_lemmas = {"en": {"have"}, "es": {"tener"}, "fr": {"avoir"}}.get(lang, {"have"})

	# IsA: X copula Y (attr/compl are nouns)
	for token in doc:
		if token.dep_ in {"attr", "acomp", "oprd"} and token.head.lemma_ in be_lemmas:
			subj = next((c for c in token.head.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
			if subj is not None and (token.pos_ in {"NOUN", "PROPN"}):
				add("IsA", [subj.text, token.text], 0.7)
				break

	# FR: copular auxiliary pattern: token(dep=cop, lemma=être), head is NOUN/ADJ
	if lang == "fr":
		for token in doc:
			if token.dep_ == "cop" and token.lemma_.lower() == "être":
				head = token.head
				if head is not None and head.pos_ in {"NOUN", "ADJ"}:
					subj = next((c for c in head.children if c.dep_ in {"nsubj", "nsubj:pass"}), None)
					if subj is not None:
						add("IsA", [subj.text, head.text], 0.7)
						break

	# PartOf: X part of Y
	for token in doc:
		if token.lemma_.lower() == "part" and token.dep_.startswith("nmod"):
			head = token.head
			pobj = next((c for c in token.children if c.dep_ in {"pobj", "nmod"}), None)
			if head is not None:
				phrase_x = head.subtree
				y = next((w for w in token.children if w.dep_ == "pobj"), None)
				add("PartOf", [" ".join(w.text for w in phrase_x), y.text if y else ""], 0.6)
				break

	# ES/FR: 'parte de' / 'partie de' + OBJ
	for token in doc:
		if lang in {"es", "fr"} and token.text.lower() in {"parte", "partie"} and token.pos_ == "NOUN":
			prep_de = next((c for c in token.children if c.text.lower() == "de" and c.pos_ == "ADP"), None)
			if prep_de is not None:
				obj = next((c for c in prep_de.children if c.dep_ in {"pobj", "obj"}), None)
				if obj is not None:
					add("PartOf", [token.text, obj.text], 0.6)
					break

	# ES/FR UD v2: 'de' as 'case' attached to the object noun; head is the container
	for token in doc:
		if lang in {"es", "fr"} and token.pos_ in {"NOUN", "PROPN"}:
			case_de = next((c for c in token.children if c.dep_ == "case" and c.text.lower() == "de"), None)
			if case_de is not None and token.head is not None and token.head.pos_ in {"NOUN", "PROPN"}:
				add("PartOf", [token.head.text, token.text], 0.6)
				break

	# Before/After: marker between two verbs/clauses
	for token in doc:
		if token.lemma_.lower() in {"before", "after"} and token.pos_ in {"SCONJ", "ADP", "ADV"}:
			left_verb = token.head if token.head.pos_ == "VERB" else None
			right_verb = next((w for w in token.children if w.pos_ == "VERB"), None)
			if left_verb or right_verb:
				name = "Before" if token.lemma_.lower() == "before" else "After"
				add(name, [left_verb.lemma_ if left_verb else "", right_verb.lemma_ if right_verb else ""], 0.6)
				break

	# AtLocation: ADP + pobj/obj
	for token in doc:
		if token.pos_ == "ADP" and token.lemma_.lower() in loc_adps:
			pobj = next((c for c in token.children if c.dep_ in {"pobj", "obj"}), None)
			if pobj is not None:
				add("AtLocation", [pobj.text], 0.6)
				break
			# UD v2 style where ADP is 'case' of a head NOUN
			if token.head is not None and token.head.pos_ in {"NOUN", "PROPN"}:
				add("AtLocation", [token.head.text], 0.6)
				break

	# UD v2: NOUN/PROPN with case ADP in location set
	for token in doc:
		if token.pos_ in {"NOUN", "PROPN"}:
			case_adps = [c for c in token.children if c.dep_ == "case" and c.lemma_.lower() in loc_adps]
			if case_adps:
				add("AtLocation", [token.text], 0.6)
				break

	# FR: verb with oblique nominal having case in loc_adps (e.g., "joue dans le parc")
	if lang == "fr":
		for token in doc:
			if token.pos_ == "VERB":
				obl = next((c for c in token.children if c.dep_ in {"obl", "nmod"} and c.pos_ in {"NOUN", "PROPN"}), None)
				if obl is not None:
					case = next((cc for cc in obl.children if cc.dep_ == "case" and cc.lemma_.lower() in loc_adps), None)
					if case is not None:
						add("AtLocation", [obl.text], 0.6)
						break

	# Because/Therefore/If/Unless
	for token in doc:
		if token.text.lower() in because_tokens or token.lemma_.lower() in because_tokens:
			add("Because", [], 0.6)
			break
	for token in doc:
		if token.lemma_.lower() in {"therefore", "so"} and token.pos_ in {"ADV", "CCONJ"}:
			add("Therefore", [], 0.5)
			break
	for token in doc:
		if token.lemma_.lower() == "if" and token.pos_ == "SCONJ":
			add("If", [], 0.6)
			break
	for token in doc:
		if token.lemma_.lower() == "unless" and token.pos_ == "SCONJ":
			add("Unless", [], 0.6)
			break

	# Purpose: in order to / to|para|pour + VERB
	for i, token in enumerate(doc):
		if token.text.lower() == "in" and i+2 < len(doc):
			if doc[i+1].lemma_.lower() == "order" and doc[i+2].lemma_.lower() == "to":
				v = doc[i+3].lemma_ if i+3 < len(doc) and doc[i+3].pos_ == "VERB" else ""
				add("InOrderTo", [v], 0.6)
				break
	for i, token in enumerate(doc):
		if token.text.lower() in to_markers and i+1 < len(doc) and doc[i+1].pos_ == "VERB":
			add("InOrderTo", [doc[i+1].lemma_], 0.5)
			break

	# Comprehensive quantifiers
	quantifier_patterns = {
		"en": {
			"All": ["all", "every", "each"],
			"Some": ["some", "several", "a few"],
			"Many": ["many", "numerous", "lots of"],
			"Few": ["few", "hardly any", "scarcely any"],
			"Most": ["most", "majority", "the majority of"],
			"None": ["no", "none", "nobody", "nothing"],
			"Only": ["only", "merely", "just"],
			"AtMost": ["at most", "no more than", "maximum"],
			"LessThan": ["less than", "fewer than", "under"]
		},
		"es": {
			"All": ["todos", "todas", "cada", "todo"],
			"Some": ["algunos", "algunas", "varios", "varias"],
			"Many": ["muchos", "muchas", "numerosos", "numerosas"],
			"Few": ["pocos", "pocas", "apenas", "escasos"],
			"Most": ["mayoría", "la mayoría de", "mayor parte"],
			"None": ["ningún", "ninguna", "nadie", "nada"],
			"Only": ["solo", "únicamente", "solamente"],
			"AtMost": ["a lo sumo", "no más de", "máximo"],
			"LessThan": ["menos de", "menos que", "bajo"]
		},
		"fr": {
			"All": ["tous", "toutes", "chaque", "tout"],
			"Some": ["quelques", "plusieurs", "certains", "certaines"],
			"Many": ["beaucoup", "nombreux", "nombreuses"],
			"Few": ["peu", "quelques", "à peine"],
			"Most": ["plupart", "majorité", "la plupart de"],
			"None": ["aucun", "aucune", "personne", "rien"],
			"Only": ["seulement", "uniquement", "juste"],
			"AtMost": ["au plus", "pas plus de", "maximum"],
			"LessThan": ["moins de", "moins que", "sous"]
		}
	}
	
	patterns = quantifier_patterns.get(lang, quantifier_patterns["en"])
	
	# Check for quantifier patterns
	for token in doc:
		lemma = token.lemma_.lower()
		
		# Direct determiner patterns
		if token.dep_ == "det":
			for quant_type, words in patterns.items():
				if lemma in words:
					add(quant_type, [token.head.text], 0.6)
					break
		
		# Multi-word patterns
		if token.pos_ in {"ADV", "ADJ", "DET"}:
			text_lower = text.lower()
			for quant_type, words in patterns.items():
				for word in words:
					if " " in word and word in text_lower:
						# Extract the noun phrase that follows
						noun_phrase = ""
						for t in doc:
							if t.pos_ in {"NOUN", "PROPN"} and t.text.lower() not in word:
								noun_phrase = t.text
								break
						add(quant_type, [noun_phrase], 0.6)
						break
		
		# Check for "not all" patterns
		if lemma == "not" and any(c.lemma_.lower() in patterns["All"] for c in token.head.children):
			add("NotAll", [token.head.text], 0.6)
		
		# Check for "all ... not" patterns
		if lemma in patterns["All"] and any(c.lemma_.lower() == "not" for c in token.head.children):
			add("AllNot", [token.head.text], 0.6)

	# Comparatives: more/less than
	for token in doc:
		if token.lemma_.lower() in {"more", "less"} and token.dep_ == "advmod":
			head = token.head
			if any(child.lemma_.lower() == "than" for child in head.children):
				add("MoreThan" if token.lemma_.lower() == "more" else "LessThan", [head.text], 0.6)
				break

	# Negation
	# Generic: any verb with neg child
	for token in doc:
		if token.pos_ == "VERB" and any(child.dep_ == "neg" for child in token.children):
			add("Not", [token.text], 0.5)
			break
	# French: 'ne ... pas' pattern
	if lang == "fr":
		if any(t.text.lower() == "pas" for t in doc):
			add("Not", ["pas"], 0.5)
	# Spanish: standalone 'no' before verb
	if lang == "es":
		for i, t in enumerate(doc):
			if t.text.lower() == "no" and i+1 < len(doc) and doc[i+1].pos_ == "VERB":
				add("Not", [doc[i+1].text], 0.5)
				break

	# SimilarTo / Like
	for token in doc:
		if token.lemma_.lower() in sim_adj and token.pos_ == "ADJ":
			prep = next((c for c in token.children if c.lower_ in {"to", "a", "à"}), None)
			if prep is not None:
				obj = next((c for c in prep.children if c.dep_ in {"pobj", "obj"}), None)
				if obj is not None:
					add("SimilarTo", [obj.text], 0.6)
					break
	for token in doc:
		if token.text.lower() in like_adp and token.pos_ == "ADP":
			obj = next((c for c in token.children if c.dep_ in {"pobj", "obj"}), None)
			if obj is not None:
				add("SimilarTo", [obj.text], 0.5)
				break
	# FR case-based: ADJ head with NOUN child having case=à (obl/nmod)
	if lang == "fr":
		for token in doc:
			if token.pos_ == "ADJ" and token.lemma_.lower() in sim_adj:
				nmod = next((c for c in token.children if c.dep_ in {"obl", "nmod"} and c.pos_ in {"NOUN", "PROPN"}), None)
				if nmod is not None and any(cc.dep_ == "case" and cc.lemma_.lower() == "à" for cc in nmod.children):
					add("SimilarTo", [nmod.text], 0.6)
					break
		# Fallback: ADJ with any child/adposition 'à' and a nearby NOUN
		for token in doc:
			if token.pos_ == "ADJ" and token.lemma_.lower() in sim_adj:
				has_a = any((c.text.lower() == "à" or c.lemma_.lower() == "à") for c in token.children)
				if has_a:
					noun = next((w for w in token.subtree if w.pos_ in {"NOUN", "PROPN"}), None)
					if noun is not None:
						add("SimilarTo", [noun.text], 0.5)
						break
	# ES case-based: ADJ head with NOUN child having case=a
	if lang == "es":
		for token in doc:
			if token.pos_ == "ADJ" and token.lemma_.lower() in sim_adj:
				nmod = next((c for c in token.children if c.dep_ in {"obl", "nmod"} and c.pos_ in {"NOUN", "PROPN"}), None)
				if nmod is not None and any(cc.dep_ == "case" and cc.lemma_.lower() == "a" for cc in nmod.children):
					add("SimilarTo", [nmod.text], 0.6)
					break
		# Fallback: ADJ with any child/adposition 'a' and a nearby NOUN
		for token in doc:
			if token.pos_ == "ADJ" and token.lemma_.lower() in sim_adj:
				has_a = any((c.text.lower() == "a" or c.lemma_.lower() == "a") for c in token.children)
				if has_a:
					noun = next((w for w in token.subtree if w.pos_ in {"NOUN", "PROPN"}), None)
					if noun is not None:
						add("SimilarTo", [noun.text], 0.5)
						break

	# DifferentFrom
	for token in doc:
		if token.lemma_.lower() in diff_adj and token.pos_ == "ADJ":
			prep = next((c for c in token.children if c.lower_ in {"from", "de"}), None)
			if prep is not None:
				obj = next((c for c in prep.children if c.dep_ in {"pobj", "obj"}), None)
				if obj is not None:
					add("DifferentFrom", [obj.text], 0.6)
					break
	# FR case-based: ADJ head with NOUN child having case=de
	if lang == "fr":
		for token in doc:
			if token.pos_ == "ADJ" and token.lemma_.lower() in diff_adj:
				nmod = next((c for c in token.children if c.dep_ in {"obl", "nmod"} and c.pos_ in {"NOUN", "PROPN"}), None)
				if nmod is not None and any(cc.dep_ == "case" and cc.lemma_.lower() == "de" for cc in nmod.children):
					add("DifferentFrom", [nmod.text], 0.6)
					break
		# Fallback: ADJ containing token 'différent(e)' and a 'de' then NOUN in subtree
		for token in doc:
			if token.pos_ == "ADJ" and token.lemma_.lower() in diff_adj:
				de_present = any((c.text.lower() == "de" or c.lemma_.lower() == "de") for c in token.children)
				if de_present:
					noun = next((w for w in token.subtree if w.pos_ in {"NOUN", "PROPN"}), None)
					if noun is not None:
						add("DifferentFrom", [noun.text], 0.5)
						break
	# ES case-based: ADJ head with NOUN child having case=de
	if lang == "es":
		for token in doc:
			if token.pos_ == "ADJ" and token.lemma_.lower() in diff_adj:
				nmod = next((c for c in token.children if c.dep_ in {"obl", "nmod"} and c.pos_ in {"NOUN", "PROPN"}), None)
				if nmod is not None and any(cc.dep_ == "case" and cc.lemma_.lower() == "de" for cc in nmod.children):
					add("DifferentFrom", [nmod.text], 0.6)
					break

	# HasProperty: have/tenir/avoir + ADJ
	for token in doc:
		if token.lemma_.lower() in have_lemmas and token.pos_ == "VERB":
			adj = next((c for c in token.children if c.pos_ == "ADJ"), None)
			if adj is not None:
				add("HasProperty", [adj.text], 0.6)
				break
	# HasProperty: have + NUM NOUN (e.g., has four wheels)
	for token in doc:
		if token.lemma_.lower() in have_lemmas and token.pos_ == "VERB":
			obj = next((c for c in token.children if c.dep_ in {"dobj", "obj"}), None)
			if obj is not None:
				num = next((c for c in obj.children if c.pos_ == "NUM"), None)
				if num is not None or obj.pos_ in {"NOUN", "PROPN"}:
					add("HasProperty", [obj.text], 0.5)
					break
	# FR: avoir + de + ADJ/NOUN (e.g., "a de grandes roues", "a de la patience")
	if lang == "fr":
		for token in doc:
			if token.lemma_.lower() == "avoir" and token.pos_ == "VERB":
				dep = next((c for c in token.children if c.dep_ in {"obj", "obl", "nmod"}), None)
				if dep is not None:
					case = next((cc for cc in dep.children if cc.dep_ == "case" and cc.lemma_.lower() == "de"), None)
					if case is not None and dep.pos_ in {"NOUN", "ADJ"}:
						add("HasProperty", [dep.text], 0.5)
						break

	# UsedFor: use/usar/utiliser + for/para/pour + OBJ/VERB
	for token in doc:
		if token.lemma_.lower() in {"use", "usar", "utiliser", "servir"} and token.pos_ == "VERB":
			prep = next((c for c in token.children if c.text.lower() in {"for", "para", "pour", "à"}), None)
			if prep is not None:
				obj = next((c for c in prep.children if c.dep_ in {"pobj", "obj"} or c.pos_ == "VERB"), None)
				if obj is not None:
					add("UsedFor", [obj.text], 0.6)
					break
	# ES: reflexive/passive and servir variants: "se usa para", "sirve para"
	if lang == "es":
		for i, t in enumerate(doc):
			if t.lemma_.lower() in {"usar", "servir"} and i+1 < len(doc):
				para = next((c for c in t.children if c.text.lower() == "para"), None)
				if para is not None:
					obj = next((c for c in para.children if c.dep_ in {"pobj", "obj"} or c.pos_ == "VERB"), None)
					if obj is not None:
						add("UsedFor", [obj.text], 0.6)
						break
			# Fallback: find 'para' in sentence and a following VERB
			if t.text.lower() == "para":
				# look ahead for a VERB in the next few tokens
				for j in range(i+1, min(i+5, len(doc))):
					if doc[j].pos_ == "VERB":
						add("UsedFor", [doc[j].lemma_], 0.5)
						break

	# Exist: language-specific patterns
	# EN: existential there + be (handled in spaCy section), plus 'exist' verb
	for t in doc:
		if t.lemma_.lower() in {"exist"} and t.pos_ == "VERB":
			add("Exist", [], 0.6)
			break
	# ES: 'hay' (haber), 'existe'
	if lang == "es":
		lower = text.lower()
		if " hay " in f" {lower} " or " existe" in lower:
			add("Exist", [], 0.6)
	# FR: 'il y a', 'existe'
	if lang == "fr":
		lower = text.lower()
		if " il y a " in f" {lower} " or " existe" in lower:
			add("Exist", [], 0.6)

	# STILL: continuation of state/action
	still_tokens = {"en": ["still"], "es": ["todavía", "aún"], "fr": ["encore", "toujours"]}.get(lang, ["still"])
	for token in doc:
		if token.lemma_.lower() in still_tokens and token.pos_ in {"ADV", "ADJ"}:
			# Find the verb or state being continued
			verb = token.head if token.head.pos_ == "VERB" else next((t for t in doc if t.pos_ == "VERB"), None)
			add("Still", [verb.lemma_ if verb else ""], 0.6)
			break

	# NOT_YET: anticipated completion (negative + temporal)
	not_yet_tokens = {"en": ["yet"], "es": ["todavía", "aún"], "fr": ["encore"]}.get(lang, ["yet"])
	for token in doc:
		if token.lemma_.lower() in not_yet_tokens and token.pos_ in {"ADV"}:
			# Check for negation context
			has_negation = any(t.lemma_.lower() in {"not", "no", "n't", "no", "ne"} for t in doc)
			if has_negation:
				verb = token.head if token.head.pos_ == "VERB" else next((t for t in doc if t.pos_ == "VERB"), None)
				add("NotYet", [verb.lemma_ if verb else ""], 0.6)
			break
	# Also check for "haven't/hasn't" + "yet" pattern
	if lang == "en":
		for i, token in enumerate(doc):
			if token.lemma_.lower() in {"have", "has"} and i+2 < len(doc):
				if doc[i+1].text.lower() in {"n't", "not"} and doc[i+2].lemma_.lower() == "yet":
					verb = next((t for t in doc if t.pos_ == "VERB" and t.lemma_ not in {"have", "has", "be"}), None)
					add("NotYet", [verb.lemma_ if verb else ""], 0.6)
					break
	# Also check for "yet" as adverb with negation context
	for token in doc:
		if token.lemma_.lower() == "yet" and token.pos_ == "ADV":
			# Check for negation context
			has_negation = any(t.lemma_.lower() in {"not", "no", "n't"} for t in doc)
			if has_negation:
				verb = next((t for t in doc if t.pos_ == "VERB" and t.lemma_ not in {"have", "has", "be"}), None)
				add("NotYet", [verb.lemma_ if verb else ""], 0.6)
				break

	# START: beginning of action/state
	start_tokens = {"en": ["start", "begin"], "es": ["empezar", "comenzar"], "fr": ["commencer", "débuter"]}.get(lang, ["start"])
	for token in doc:
		if token.lemma_.lower() in start_tokens and token.pos_ == "VERB":
			# Find the object or infinitive being started
			obj = next((c for c in token.children if c.dep_ in {"dobj", "obj", "xcomp"}), None)
			add("Start", [obj.lemma_ if obj else ""], 0.6)
			break

	# FINISH: completion of action/state
	finish_tokens = {"en": ["finish", "complete", "end"], "es": ["terminar", "acabar", "finalizar"], "fr": ["finir", "terminer", "achever"]}.get(lang, ["finish"])
	for token in doc:
		if token.lemma_.lower() in finish_tokens and token.pos_ == "VERB":
			# Find the object or infinitive being finished
			obj = next((c for c in token.children if c.dep_ in {"dobj", "obj", "xcomp"}), None)
			add("Finish", [obj.lemma_ if obj else ""], 0.6)
			break

	# AGAIN: repetition of action
	again_tokens = {"en": ["again"], "es": ["otra", "vez", "nuevo"], "fr": ["encore", "nouveau"]}.get(lang, ["again"])
	for token in doc:
		if token.lemma_.lower() in again_tokens and token.pos_ in {"ADV", "ADJ"}:
			# Find the verb being repeated
			verb = token.head if token.head.pos_ == "VERB" else next((t for t in doc if t.pos_ == "VERB"), None)
			add("Again", [verb.lemma_ if verb else ""], 0.6)
			break

	# KEEP: continuation/maintenance of action
	keep_tokens = {"en": ["keep", "continue"], "es": ["seguir", "continuar"], "fr": ["continuer", "garder"]}.get(lang, ["keep"])
	for token in doc:
		if token.lemma_.lower() in keep_tokens and token.pos_ == "VERB":
			# Find the object or infinitive being continued
			obj = next((c for c in token.children if c.dep_ in {"dobj", "obj", "xcomp"}), None)
			add("Keep", [obj.lemma_ if obj else ""], 0.6)
			break

	# === PHASE 1: CORE SUBSTANTIVES (NSM Primes) - UD-Based Detection ===
	
	# I: first person singular pronoun (nsubj, nsubjpass, or direct object)
	for token in doc:
		if (token.pos_ == "PRON" and 
			token.lemma_.lower() in {"i", "me", "my", "myself", "yo", "je"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "iobj"}):
			add("I", [token.text], 0.8)
			break

	# YOU: second person pronoun (nsubj, nsubjpass, or direct object)
	for token in doc:
		if (token.pos_ == "PRON" and 
			token.lemma_.lower() in {"you", "your", "yourself", "tú", "tu", "vous"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "iobj"}):
			add("YOU", [token.text], 0.8)
			break

	# SOMEONE: indefinite person (subject, object, or with determiner)
	for token in doc:
		if (token.pos_ in {"PRON", "NOUN"} and 
			token.lemma_.lower() in {"someone", "somebody", "anyone", "anybody", "alguien", "quelqu'un", "person"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			add("SOMEONE", [token.text], 0.7)
			break
	
	# Also detect "a person" as SOMEONE
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"person", "persona", "personne"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			# Check for indefinite determiner
			det = next((c for c in token.children if c.dep_ == "det" and c.lemma_.lower() in {"a", "an", "un", "una", "un", "une"}), None)
			if det:
				add("SOMEONE", [token.text], 0.7)
				break

	# PEOPLE: plural persons (subject, object, or collective noun)
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"people", "persons", "humans", "individuals", "gente", "personnes", "individus"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			add("PEOPLE", [token.text], 0.7)
			break

	# SOMETHING: indefinite thing (subject, object, or with determiner)
	for token in doc:
		if (token.pos_ in {"PRON", "NOUN"} and 
			token.lemma_.lower() in {"something", "anything", "whatever", "algo", "quelque chose", "event", "evento", "événement"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			add("SOMETHING", [token.text], 0.7)
			break
	
	# Also detect "an event" as SOMETHING
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"event", "evento", "événement", "thing", "cosa", "chose"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			# Check for indefinite determiner
			det = next((c for c in token.children if c.dep_ == "det" and c.lemma_.lower() in {"a", "an", "un", "una", "un", "une"}), None)
			if det:
				add("SOMETHING", [token.text], 0.7)
				break

	# THING: generic object (subject, object, or with determiner)
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"thing", "object", "item", "cosa", "objet", "article"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			add("THING", [token.text], 0.7)
			break
	
	# Also detect "the object" as THING
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"object", "objeto", "objet"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			add("THING", [token.text], 0.7)
			break

	# BODY: physical entity (subject, object, or with determiner)
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"body", "person", "human", "cuerpo", "personne", "humain"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			add("BODY", [token.text], 0.7)
			break
	
	# Also detect "the person" as BODY
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"person", "persona", "personne"} and
			token.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}):
			# Check for definite determiner
			det = next((c for c in token.children if c.dep_ == "det" and c.lemma_.lower() in {"the", "el", "la", "le", "les"}), None)
			if det:
				add("BODY", [token.text], 0.7)
				break

	# === PHASE 2: MENTAL PREDICATES (NSM Primes) - UD-Based Detection ===
	
	# THINK: mental cognition and reasoning
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"think", "believe", "consider", "pensar", "creer", "penser", "croire"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("THINK", [token.text], 0.8)
			break

	# KNOW: mental knowledge and understanding
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"know", "understand", "realize", "saber", "conocer", "savoir", "connaître"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("KNOW", [token.text], 0.8)
			break

	# WANT: mental desire and intention
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"want", "desire", "wish", "querer", "desear", "vouloir", "souhaiter"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("WANT", [token.text], 0.8)
			break

	# FEEL: mental emotion and sensation
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"feel", "sense", "sentir", "sentir", "ressentir"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("FEEL", [token.text], 0.8)
			break

	# SEE: sensory visual perception
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"see", "look", "watch", "ver", "mirar", "voir", "regarder"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("SEE", [token.text], 0.8)
			break

	# HEAR: sensory auditory perception
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"hear", "listen", "escuchar", "oir", "entendre", "écouter"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("HEAR", [token.text], 0.8)
			break

	# === PHASE 3: LOGICAL OPERATORS (NSM Primes) - UD-Based Detection ===
	
	# BECAUSE: logical causation and reasoning
	for token in doc:
		if (token.pos_ in {"SCONJ", "ADV"} and 
			token.lemma_.lower() in {"because", "since", "as", "porque", "ya", "puisque", "car"} and
			token.dep_ in {"mark", "advmod"}):
			add("BECAUSE", [token.text], 0.8)
			break

	# IF: logical condition and implication
	for token in doc:
		if (token.pos_ == "SCONJ" and 
			token.lemma_.lower() in {"if", "si", "si"} and
			token.dep_ == "mark"):
			add("IF", [token.text], 0.8)
			break

	# NOT: logical negation
	for token in doc:
		if (token.pos_ in {"PART", "ADV"} and 
			token.lemma_.lower() in {"not", "no", "ne", "pas"} and
			token.dep_ in {"neg", "advmod"}):
			add("NOT", [token.text], 0.8)
			break

	# SAME: logical identity and equivalence
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV"} and 
			token.lemma_.lower() in {"same", "identical", "igual", "même", "pareil"} and
			token.dep_ in {"amod", "advmod", "attr"}):
			add("SAME", [token.text], 0.8)
			break

	# DIFFERENT: logical difference and distinction
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV"} and 
			token.lemma_.lower() in {"different", "distinct", "diferente", "différent", "autre"} and
			token.dep_ in {"amod", "advmod", "attr"}):
			add("DIFFERENT", [token.text], 0.8)
			break

	# MAYBE: logical possibility and uncertainty
	for token in doc:
		if (token.pos_ in {"ADV", "PART"} and 
			token.lemma_.lower() in {"maybe", "perhaps", "possibly", "tal", "vez", "peut-être"} and
			token.dep_ in {"advmod", "discourse"}):
			add("MAYBE", [token.text], 0.8)
			break

	# === PHASE 4: TEMPORAL & CAUSAL (NSM Primes) - UD-Based Detection ===
	
	# BEFORE: temporal precedence and ordering
	for token in doc:
		if (token.pos_ in {"ADP", "ADV", "SCONJ"} and 
			token.lemma_.lower() in {"before", "prior", "antes", "avant"} and
			token.dep_ in {"prep", "advmod", "mark"}):
			add("BEFORE", [token.text], 0.8)
			break

	# AFTER: temporal succession and sequence
	for token in doc:
		if (token.pos_ in {"ADP", "ADV", "SCONJ"} and 
			token.lemma_.lower() in {"after", "later", "después", "après"} and
			token.dep_ in {"prep", "advmod", "mark"}):
			add("AFTER", [token.text], 0.8)
			break

	# WHEN: temporal simultaneity and coincidence
	for token in doc:
		if (token.pos_ in {"ADV", "SCONJ"} and 
			token.lemma_.lower() in {"when", "while", "cuando", "quand"} and
			token.dep_ in {"advmod", "mark"}):
			add("WHEN", [token.text], 0.8)
			break

	# CAUSE: causal agency and responsibility
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"cause", "lead", "causar", "causer"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("CAUSE", [token.text], 0.8)
			break

	# MAKE: causal creation and production
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"make", "create", "hacer", "faire"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("MAKE", [token.text], 0.8)
			break

	# LET: causal permission and allowance
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"let", "allow", "permitir", "permettre"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("LET", [token.text], 0.8)
			break

	# === PHASE 5: SPATIAL & PHYSICAL (NSM Primes) - UD-Based Detection ===
	
	# IN: spatial containment and inclusion
	for token in doc:
		if (token.pos_ == "ADP" and 
			token.lemma_.lower() in {"in", "within", "dentro", "dans"} and
			token.dep_ == "prep"):
			add("IN", [token.text], 0.8)
			break

	# ON: spatial support and contact
	for token in doc:
		if (token.pos_ == "ADP" and 
			token.lemma_.lower() in {"on", "upon", "sobre", "sur"} and
			token.dep_ == "prep"):
			add("ON", [token.text], 0.8)
			break

	# UNDER: spatial subordination and coverage
	for token in doc:
		if (token.pos_ == "ADP" and 
			token.lemma_.lower() in {"under", "beneath", "bajo", "sous"} and
			token.dep_ == "prep"):
			add("UNDER", [token.text], 0.8)
			break

	# NEAR: spatial proximity and closeness
	for token in doc:
		if (token.pos_ in {"ADP", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"near", "close", "cerca", "près"} and
			token.dep_ in {"prep", "amod", "advmod"}):
			add("NEAR", [token.text], 0.8)
			break

	# FAR: spatial distance and separation
	for token in doc:
		if (token.pos_ in {"ADP", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"far", "away", "lejos", "loin"} and
			token.dep_ in {"prep", "amod", "advmod"}):
			add("FAR", [token.text], 0.8)
			break

	# INSIDE: spatial interiority and enclosure
	for token in doc:
		if (token.pos_ in {"ADP", "ADV"} and 
			token.lemma_.lower() in {"inside", "within", "dentro", "dedans"} and
			token.dep_ in {"prep", "advmod"}):
			add("INSIDE", [token.text], 0.8)
			break

	# === PHASE 6: QUANTIFIERS (NSM Primes) - UD-Based Detection ===
	
	# ALL: universal quantification and totality
	for token in doc:
		if (token.pos_ in {"DET", "PRON", "ADJ"} and 
			token.lemma_.lower() in {"all", "every", "each", "todo", "tous"} and
			token.dep_ in {"det", "amod", "nsubj", "dobj"}):
			add("ALL", [token.text], 0.8)
			break

	# MANY: large quantity and plurality
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"many", "numerous", "muchos", "nombreux"} and
			token.dep_ in {"det", "amod", "advmod"}):
			add("MANY", [token.text], 0.8)
			break

	# SOME: partial quantity and existence
	for token in doc:
		if (token.pos_ in {"DET", "PRON", "ADJ"} and 
			token.lemma_.lower() in {"some", "several", "algunos", "quelques"} and
			token.dep_ in {"det", "amod", "nsubj", "dobj"}):
			add("SOME", [token.text], 0.8)
			break

	# FEW: small quantity and scarcity
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"few", "little", "pocos", "peu"} and
			token.dep_ in {"det", "amod", "advmod"}):
			add("FEW", [token.text], 0.8)
			break

	# MUCH: large amount and abundance
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"much", "a lot", "mucho", "beaucoup"} and
			token.dep_ in {"det", "amod", "advmod"}):
			add("MUCH", [token.text], 0.8)
			break

	# LITTLE: small amount and paucity
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "ADV"} and 
			token.lemma_.lower() in {"little", "small", "poco", "peu"} and
			token.dep_ in {"det", "amod", "advmod"}):
			add("LITTLE", [token.text], 0.8)
			break

	# === PHASE 7: EVALUATORS (NSM Primes) - UD-Based Detection ===
	
	# GOOD: positive evaluation and desirability
	for token in doc:
		if (token.pos_ == "ADJ" and 
			token.lemma_.lower() in {"good", "great", "excellent", "bueno", "bon"} and
			token.dep_ in {"amod", "attr", "pred"}):
			add("GOOD", [token.text], 0.8)
			break

	# BAD: negative evaluation and undesirability
	for token in doc:
		if (token.pos_ == "ADJ" and 
			token.lemma_.lower() in {"bad", "terrible", "awful", "malo", "mauvais"} and
			token.dep_ in {"amod", "attr", "pred"}):
			add("BAD", [token.text], 0.8)
			break

	# BIG: large size and magnitude
	for token in doc:
		if (token.pos_ == "ADJ" and 
			token.lemma_.lower() in {"big", "large", "huge", "grande", "grand"} and
			token.dep_ in {"amod", "attr", "pred"}):
			add("BIG", [token.text], 0.8)
			break

	# SMALL: small size and magnitude
	for token in doc:
		if (token.pos_ == "ADJ" and 
			token.lemma_.lower() in {"small", "tiny", "little", "pequeño", "petit"} and
			token.dep_ in {"amod", "attr", "pred"}):
			add("SMALL", [token.text], 0.8)
			break

	# RIGHT: correctness and appropriateness
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV"} and 
			token.lemma_.lower() in {"right", "correct", "proper", "correcto", "juste"} and
			token.dep_ in {"amod", "attr", "pred", "advmod"}):
			add("RIGHT", [token.text], 0.8)
			break

	# WRONG: incorrectness and inappropriateness (avoid conflict with FALSE)
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV"} and 
			token.lemma_.lower() in {"wrong", "incorrect", "incorrecto"} and
			token.dep_ in {"amod", "attr", "pred", "advmod"} and
			token.lemma_.lower() not in {"false", "fake", "falso"}):  # Avoid conflict with FALSE
			add("WRONG", [token.text], 0.8)
			break

	# === PHASE 8: ACTIONS (NSM Primes) - UD-Based Detection ===
	
	# DO: action performance and execution
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"do", "make", "perform", "hacer", "faire"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("DO", [token.text], 0.8)
			break

	# HAPPEN: event occurrence and happening
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"happen", "occur", "take", "pasar", "arriver"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("HAPPEN", [token.text], 0.8)
			break

	# MOVE: physical movement and motion
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"move", "go", "walk", "mover", "bouger"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("MOVE", [token.text], 0.8)
			break

	# TOUCH: physical contact and interaction
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"touch", "contact", "reach", "tocar", "toucher"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("TOUCH", [token.text], 0.8)
			break

	# LIVE: existence and being alive
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"live", "exist", "be", "vivir", "vivre"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("LIVE", [token.text], 0.8)
			break

	# DIE: death and cessation of life
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"die", "death", "end", "morir", "mourir"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp"}):
			add("DIE", [token.text], 0.8)
			break

	# === PHASE 9: DESCRIPTORS (NSM Primes) - UD-Based Detection ===
	
	# THIS: proximate reference and identification
	for token in doc:
		if (token.pos_ == "DET" and 
			token.lemma_.lower() in {"this", "ese", "ce"} and
			token.dep_ in {"det", "attr"}):
			add("THIS", [token.text], 0.8)
			break

	# THE SAME: identity and sameness
	for token in doc:
		if (token.pos_ in {"DET", "ADJ"} and 
			token.lemma_.lower() in {"same", "mismo", "même"} and
			token.dep_ in {"det", "amod", "attr"}):
			add("THE SAME", [token.text], 0.8)
			break

	# OTHER: distinction and difference
	for token in doc:
		if (token.pos_ in {"DET", "ADJ", "PRON"} and 
			token.lemma_.lower() in {"other", "otro", "autre"} and
			token.dep_ in {"det", "amod", "attr", "nsubj", "dobj"}):
			add("OTHER", [token.text], 0.8)
			break

	# ONE: singularity and unity
	for token in doc:
		if (token.pos_ in {"NUM", "DET"} and 
			token.lemma_.lower() in {"one", "uno", "un"} and
			token.dep_ in {"nummod", "det"}):
			add("ONE", [token.text], 0.8)
			break

	# TWO: duality and pairing
	for token in doc:
		if (token.pos_ == "NUM" and 
			token.lemma_.lower() in {"two", "dos", "deux"} and
			token.dep_ in {"nummod"}):
			add("TWO", [token.text], 0.8)
			break

	# SOME: indefinite quantity and selection
	for token in doc:
		if (token.pos_ in {"DET", "PRON"} and 
			token.lemma_.lower() in {"some", "algunos", "quelques"} and
			token.dep_ in {"det", "nsubj", "dobj"}):
			add("SOME", [token.text], 0.8)
			break

	# === PHASE 10: INTENSIFIERS (NSM Primes) - UD-Based Detection ===
	
	# VERY: high degree and intensity
	for token in doc:
		if (token.pos_ == "ADV" and 
			token.lemma_.lower() in {"very", "muy", "très"} and
			token.dep_ in {"advmod", "amod"}):
			add("VERY", [token.text], 0.8)
			break

	# MORE: comparative degree and increase
	for token in doc:
		if (token.pos_ in {"ADV", "ADJ"} and 
			token.lemma_.lower() in {"more", "más", "plus"} and
			token.dep_ in {"advmod", "amod", "attr"}):
			add("MORE", [token.text], 0.8)
			break

	# LIKE: similarity and resemblance
	for token in doc:
		if (token.pos_ in {"ADP", "ADV", "ADJ"} and 
			token.lemma_.lower() in {"like", "como", "comme"} and
			token.dep_ in {"prep", "advmod", "amod"}):
			add("LIKE", [token.text], 0.8)
			break

	# KIND OF: partial degree and approximation
	for token in doc:
		if (token.pos_ in {"ADV", "ADJ"} and 
			token.lemma_.lower() in {"kind", "tipo", "genre"} and
			token.dep_ in {"advmod", "amod"}):
			add("KIND OF", [token.text], 0.8)
			break

	# === PHASE 11: FINAL PRIMES (NSM Primes) - UD-Based Detection ===
	
	# SAY: speech and communication (main verb or complement)
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"say", "tell", "speak", "decir", "dire"} and
			token.dep_ in {"ROOT", "ccomp", "xcomp", "acl", "advcl"}):
			add("SAY", [token.text], 0.8)
			break
	
	# Check for SAY in relative clauses and complex constructions
	for token in doc:
		if (token.pos_ == "VERB" and 
			token.lemma_.lower() in {"say", "tell", "speak", "decir", "dire"} and
			token.dep_ in {"relcl", "acl:relcl", "advcl"}):
			add("SAY", [token.text], 0.8)
			break

	# WORDS: linguistic expression (noun)
	for token in doc:
		if (token.pos_ == "NOUN" and 
			token.lemma_.lower() in {"word", "words", "palabra", "mot"} and
			token.dep_ in {"nsubj", "dobj", "pobj"}):
			add("WORDS", [token.text], 0.8)
			break

	# TRUE: truth and factuality (adjective or adverb)
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV", "NOUN"} and 
			token.lemma_.lower() in {"true", "real", "truth", "verdadero", "vrai", "vérité"} and
			token.dep_ in {"amod", "attr", "pred", "advmod", "nsubj", "dobj", "pobj"}):
			add("TRUE", [token.text], 0.8)
			break
	
	# Check for copula constructions with TRUE
	for token in doc:
		if (token.lemma_.lower() in {"true", "real", "truth", "verdadero", "vrai", "vérité"} and
			token.dep_ in {"attr", "pred"} and
			any(child.lemma_.lower() in {"be", "is", "are", "was", "were", "ser", "estar", "être"} for child in token.head.children)):
			add("TRUE", [token.text], 0.8)
			break

	# FALSE: falsity and deception (adjective or adverb)
	for token in doc:
		if (token.pos_ in {"ADJ", "ADV", "NOUN"} and 
			token.lemma_.lower() in {"false", "fake", "falsity", "falso", "faux", "fausseté"} and
			token.dep_ in {"amod", "attr", "pred", "advmod", "nsubj", "dobj", "pobj"}):
			add("FALSE", [token.text], 0.8)
			break
	
	# Check for copula constructions with FALSE
	for token in doc:
		if (token.lemma_.lower() in {"false", "fake", "falsity", "falso", "faux", "fausseté"} and
			token.dep_ in {"attr", "pred"} and
			any(child.lemma_.lower() in {"be", "is", "are", "was", "were", "ser", "estar", "être"} for child in token.head.children)):
			add("FALSE", [token.text], 0.8)
			break

	# WHERE: location specification
	for token in doc:
		if (token.pos_ in {"ADV", "PRON"} and 
			token.lemma_.lower() in {"where", "dónde", "où"} and
			token.dep_ in {"advmod", "pobj", "nsubj"}):
			add("WHERE", [token.text], 0.8)
			break

	# WHEN: time specification (enhanced)
	for token in doc:
		if (token.pos_ in {"ADV", "PRON", "SCONJ"} and 
			token.lemma_.lower() in {"when", "cuándo", "quand"} and
			token.dep_ in {"advmod", "pobj", "mark"}):
			add("WHEN", [token.text], 0.8)
			break

	return detections


def detect_primitives_dep(text: str) -> List[str]:
	"""Strict dependency-pattern detector using spaCy's DependencyMatcher.

	Patterns:
	- IsA: nsubj(HEAD=be), attr NOUN/PROPN
	- AtLocation: ADP (in/on/at) with pobj NOUN/PROPN
	- InOrderTo: 'in' 'order' 'to' + VERB, or 'to' + VERB
	- Not: any neg dependency on a verb
	"""
	lang = _pick_lang(text)
	nlp = _NLP.get(lang)
	if not nlp or DependencyMatcher is None:
		return []
	try:
		doc = nlp(text)
	except Exception:
		return []
	matcher = DependencyMatcher(nlp.vocab)
	patterns: Dict[str, List[List[Dict[str, Any]]]] = {}
	# Language-specific lemma/lex sets
	if lang == "en":
		be_lemmas = ["be"]
		loc_adps = ["in", "on", "at"]
		because_tokens = ["because", "since"]
		used_for_prep = ["for"]
		to_marker = ["to"]
		like_adp = ["like"]
		diff_tokens = ["different"]
		sim_tokens = ["similar"]
		have_lemmas = ["have"]
		part_of_tokens = ("part", "of")
	elif lang == "es":
		be_lemmas = ["ser"]
		loc_adps = ["en", "a", "sobre"]
		because_tokens = ["porque", "debido"]
		used_for_prep = ["para"]
		to_marker = ["para"]
		like_adp = ["como"]
		diff_tokens = ["diferente"]
		sim_tokens = ["similar"]
		have_lemmas = ["tener"]
		part_of_tokens = ("parte", "de")
	else:  # fr
		be_lemmas = ["être"]
		loc_adps = ["à", "dans", "sur", "en"]
		because_tokens = ["parce", "car"]  # capture "parce que" by token "parce"
		used_for_prep = ["pour"]
		to_marker = ["pour"]
		like_adp = ["comme"]
		diff_tokens = ["différent"]
		sim_tokens = ["similaire"]
		have_lemmas = ["avoir"]
		part_of_tokens = ("partie", "de")

	# === PHASE 1: CORE SUBSTANTIVES (NSM Primes) - UD-Based Patterns ===
	
	# I: first person singular pronoun (subject or object)
	patterns["I"] = [[
		{"RIGHT_ID": "i", "RIGHT_ATTRS": {"LOWER": {"IN": ["i", "me", "my", "myself", "yo", "je"]}, "POS": "PRON", "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "iobj"]}}}
	]]
	
	# YOU: second person pronoun (subject or object)
	patterns["YOU"] = [[
		{"RIGHT_ID": "you", "RIGHT_ATTRS": {"LOWER": {"IN": ["you", "your", "yourself", "tú", "tu", "vous"]}, "POS": "PRON", "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "iobj"]}}}
	]]
	
	# SOMEONE: indefinite person (subject, object, or with indefinite determiner)
	patterns["SOMEONE"] = [[
		{"RIGHT_ID": "someone", "RIGHT_ATTRS": {"LOWER": {"IN": ["someone", "somebody", "anyone", "anybody", "alguien", "quelqu'un", "person"]}, "POS": {"IN": ["PRON", "NOUN"]}, "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "pobj"]}}}
	]]
	
	# SOMEONE: "a person" pattern
	patterns["SOMEONE_A_PERSON"] = [[
		{"RIGHT_ID": "det", "RIGHT_ATTRS": {"LOWER": {"IN": ["a", "an", "un", "una", "un", "une"]}, "POS": "DET", "DEP": "det"}},
		{"LEFT_ID": "det", "REL_OP": ">", "RIGHT_ID": "person", "RIGHT_ATTRS": {"LOWER": {"IN": ["person", "persona", "personne"]}, "POS": "NOUN", "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "pobj"]}}}
	]]
	
	# PEOPLE: plural persons (subject, object, or collective noun)
	patterns["PEOPLE"] = [[
		{"RIGHT_ID": "people", "RIGHT_ATTRS": {"LOWER": {"IN": ["people", "persons", "humans", "individuals", "gente", "personnes", "individus"]}, "POS": "NOUN", "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "pobj"]}}}
	]]
	
	# SOMETHING: indefinite thing (subject, object, or with indefinite determiner)
	patterns["SOMETHING"] = [[
		{"RIGHT_ID": "something", "RIGHT_ATTRS": {"LOWER": {"IN": ["something", "anything", "whatever", "algo", "quelque chose", "event", "evento", "événement"]}, "POS": {"IN": ["PRON", "NOUN"]}, "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "pobj"]}}}
	]]
	
	# SOMETHING: "an event" pattern
	patterns["SOMETHING_AN_EVENT"] = [[
		{"RIGHT_ID": "det", "RIGHT_ATTRS": {"LOWER": {"IN": ["a", "an", "un", "una", "un", "une"]}, "POS": "DET", "DEP": "det"}},
		{"LEFT_ID": "det", "REL_OP": ">", "RIGHT_ID": "event", "RIGHT_ATTRS": {"LOWER": {"IN": ["event", "evento", "événement", "thing", "cosa", "chose"]}, "POS": "NOUN", "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "pobj"]}}}
	]]
	
	# THING: generic object (subject, object, or with determiner)
	patterns["THING"] = [[
		{"RIGHT_ID": "thing", "RIGHT_ATTRS": {"LOWER": {"IN": ["thing", "object", "item", "cosa", "objet", "article"]}, "POS": "NOUN", "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "pobj"]}}}
	]]
	
	# THING: "the object" pattern
	patterns["THING_THE_OBJECT"] = [[
		{"RIGHT_ID": "det", "RIGHT_ATTRS": {"LOWER": {"IN": ["the", "el", "la", "le", "les"]}, "POS": "DET", "DEP": "det"}},
		{"LEFT_ID": "det", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"LOWER": {"IN": ["object", "objeto", "objet"]}, "POS": "NOUN", "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "pobj"]}}}
	]]
	
	# BODY: physical entity (subject, object, or with determiner)
	patterns["BODY"] = [[
		{"RIGHT_ID": "body", "RIGHT_ATTRS": {"LOWER": {"IN": ["body", "person", "human", "cuerpo", "personne", "humain"]}, "POS": "NOUN", "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "pobj"]}}}
	]]
	
	# BODY: "the person" pattern
	patterns["BODY_THE_PERSON"] = [[
		{"RIGHT_ID": "det", "RIGHT_ATTRS": {"LOWER": {"IN": ["the", "el", "la", "le", "les"]}, "POS": "DET", "DEP": "det"}},
		{"LEFT_ID": "det", "REL_OP": ">", "RIGHT_ID": "person", "RIGHT_ATTRS": {"LOWER": {"IN": ["person", "persona", "personne"]}, "POS": "NOUN", "DEP": {"IN": ["nsubj", "nsubjpass", "dobj", "obj", "pobj"]}}}
	]]
	
	# === PHASE 2: MENTAL PREDICATES (NSM Primes) - UD-Based Patterns ===
	
	# THINK: mental cognition and reasoning (main verb or complement)
	patterns["THINK"] = [[
		{"RIGHT_ID": "think", "RIGHT_ATTRS": {"LOWER": {"IN": ["think", "believe", "consider", "pensar", "creer", "penser", "croire"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# KNOW: mental knowledge and understanding (main verb or complement)
	patterns["KNOW"] = [[
		{"RIGHT_ID": "know", "RIGHT_ATTRS": {"LOWER": {"IN": ["know", "understand", "realize", "saber", "conocer", "savoir", "connaître"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# WANT: mental desire and intention (main verb or complement)
	patterns["WANT"] = [[
		{"RIGHT_ID": "want", "RIGHT_ATTRS": {"LOWER": {"IN": ["want", "desire", "wish", "querer", "desear", "vouloir", "souhaiter"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# FEEL: mental emotion and sensation (main verb or complement)
	patterns["FEEL"] = [[
		{"RIGHT_ID": "feel", "RIGHT_ATTRS": {"LOWER": {"IN": ["feel", "sense", "sentir", "ressentir"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# SEE: sensory visual perception (main verb or complement)
	patterns["SEE"] = [[
		{"RIGHT_ID": "see", "RIGHT_ATTRS": {"LOWER": {"IN": ["see", "look", "watch", "ver", "mirar", "voir", "regarder"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# HEAR: sensory auditory perception (main verb or complement)
	patterns["HEAR"] = [[
		{"RIGHT_ID": "hear", "RIGHT_ATTRS": {"LOWER": {"IN": ["hear", "listen", "escuchar", "oir", "entendre", "écouter"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# === PHASE 3: LOGICAL OPERATORS (NSM Primes) - UD-Based Patterns ===
	
	# BECAUSE: logical causation and reasoning (subordinating conjunction or adverb)
	patterns["BECAUSE"] = [[
		{"RIGHT_ID": "because", "RIGHT_ATTRS": {"LOWER": {"IN": ["because", "since", "as", "porque", "ya", "puisque", "car"]}, "POS": {"IN": ["SCONJ", "ADV"]}, "DEP": {"IN": ["mark", "advmod"]}}}
	]]
	
	# IF: logical condition and implication (subordinating conjunction)
	patterns["IF"] = [[
		{"RIGHT_ID": "if", "RIGHT_ATTRS": {"LOWER": {"IN": ["if", "si", "si"]}, "POS": "SCONJ", "DEP": "mark"}}
	]]
	
	# NOT: logical negation (particle or adverb)
	patterns["NOT"] = [[
		{"RIGHT_ID": "not", "RIGHT_ATTRS": {"LOWER": {"IN": ["not", "no", "ne", "pas"]}, "POS": {"IN": ["PART", "ADV"]}, "DEP": {"IN": ["neg", "advmod"]}}}
	]]
	
	# SAME: logical identity and equivalence (adjective or adverb)
	patterns["SAME"] = [[
		{"RIGHT_ID": "same", "RIGHT_ATTRS": {"LOWER": {"IN": ["same", "identical", "igual", "même", "pareil"]}, "POS": {"IN": ["ADJ", "ADV"]}, "DEP": {"IN": ["amod", "advmod", "attr"]}}}
	]]
	
	# DIFFERENT: logical difference and distinction (adjective or adverb)
	patterns["DIFFERENT"] = [[
		{"RIGHT_ID": "different", "RIGHT_ATTRS": {"LOWER": {"IN": ["different", "distinct", "diferente", "différent", "autre"]}, "POS": {"IN": ["ADJ", "ADV"]}, "DEP": {"IN": ["amod", "advmod", "attr"]}}}
	]]
	
	# MAYBE: logical possibility and uncertainty (adverb or particle)
	patterns["MAYBE"] = [[
		{"RIGHT_ID": "maybe", "RIGHT_ATTRS": {"LOWER": {"IN": ["maybe", "perhaps", "possibly", "tal", "vez", "peut-être"]}, "POS": {"IN": ["ADV", "PART"]}, "DEP": {"IN": ["advmod", "discourse"]}}}
	]]
	
	# === PHASE 4: TEMPORAL & CAUSAL (NSM Primes) - UD-Based Patterns ===
	
	# BEFORE: temporal precedence and ordering (preposition, adverb, or subordinating conjunction)
	patterns["BEFORE"] = [[
		{"RIGHT_ID": "before", "RIGHT_ATTRS": {"LOWER": {"IN": ["before", "prior", "antes", "avant"]}, "POS": {"IN": ["ADP", "ADV", "SCONJ"]}, "DEP": {"IN": ["prep", "advmod", "mark"]}}}
	]]
	
	# AFTER: temporal succession and sequence (preposition, adverb, or subordinating conjunction)
	patterns["AFTER"] = [[
		{"RIGHT_ID": "after", "RIGHT_ATTRS": {"LOWER": {"IN": ["after", "later", "después", "après"]}, "POS": {"IN": ["ADP", "ADV", "SCONJ"]}, "DEP": {"IN": ["prep", "advmod", "mark"]}}}
	]]
	
	# WHEN: temporal simultaneity and coincidence (adverb or subordinating conjunction)
	patterns["WHEN"] = [[
		{"RIGHT_ID": "when", "RIGHT_ATTRS": {"LOWER": {"IN": ["when", "while", "cuando", "quand"]}, "POS": {"IN": ["ADV", "SCONJ"]}, "DEP": {"IN": ["advmod", "mark"]}}}
	]]
	
	# CAUSE: causal agency and responsibility (main verb or complement)
	patterns["CAUSE"] = [[
		{"RIGHT_ID": "cause", "RIGHT_ATTRS": {"LOWER": {"IN": ["cause", "lead", "causar", "causer"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# MAKE: causal creation and production (main verb or complement)
	patterns["MAKE"] = [[
		{"RIGHT_ID": "make", "RIGHT_ATTRS": {"LOWER": {"IN": ["make", "create", "hacer", "faire"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# LET: causal permission and allowance (main verb or complement)
	patterns["LET"] = [[
		{"RIGHT_ID": "let", "RIGHT_ATTRS": {"LOWER": {"IN": ["let", "allow", "permitir", "permettre"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# === PHASE 5: SPATIAL & PHYSICAL (NSM Primes) - UD-Based Patterns ===
	
	# IN: spatial containment and inclusion (preposition)
	patterns["IN"] = [[
		{"RIGHT_ID": "in", "RIGHT_ATTRS": {"LOWER": {"IN": ["in", "within", "dentro", "dans"]}, "POS": "ADP", "DEP": "prep"}}
	]]
	
	# ON: spatial support and contact (preposition)
	patterns["ON"] = [[
		{"RIGHT_ID": "on", "RIGHT_ATTRS": {"LOWER": {"IN": ["on", "upon", "sobre", "sur"]}, "POS": "ADP", "DEP": "prep"}}
	]]
	
	# UNDER: spatial subordination and coverage (preposition)
	patterns["UNDER"] = [[
		{"RIGHT_ID": "under", "RIGHT_ATTRS": {"LOWER": {"IN": ["under", "beneath", "bajo", "sous"]}, "POS": "ADP", "DEP": "prep"}}
	]]
	
	# NEAR: spatial proximity and closeness (preposition, adjective, or adverb)
	patterns["NEAR"] = [[
		{"RIGHT_ID": "near", "RIGHT_ATTRS": {"LOWER": {"IN": ["near", "close", "cerca", "près"]}, "POS": {"IN": ["ADP", "ADJ", "ADV"]}, "DEP": {"IN": ["prep", "amod", "advmod"]}}}
	]]
	
	# FAR: spatial distance and separation (preposition, adjective, or adverb)
	patterns["FAR"] = [[
		{"RIGHT_ID": "far", "RIGHT_ATTRS": {"LOWER": {"IN": ["far", "away", "lejos", "loin"]}, "POS": {"IN": ["ADP", "ADJ", "ADV"]}, "DEP": {"IN": ["prep", "amod", "advmod"]}}}
	]]
	
	# INSIDE: spatial interiority and enclosure (preposition or adverb)
	patterns["INSIDE"] = [[
		{"RIGHT_ID": "inside", "RIGHT_ATTRS": {"LOWER": {"IN": ["inside", "within", "dentro", "dedans"]}, "POS": {"IN": ["ADP", "ADV"]}, "DEP": {"IN": ["prep", "advmod"]}}}
	]]
	
	# === PHASE 6: QUANTIFIERS (NSM Primes) - UD-Based Patterns ===
	
	# ALL: universal quantification and totality (determiner, pronoun, or adjective)
	patterns["ALL"] = [[
		{"RIGHT_ID": "all", "RIGHT_ATTRS": {"LOWER": {"IN": ["all", "every", "each", "todo", "tous"]}, "POS": {"IN": ["DET", "PRON", "ADJ"]}, "DEP": {"IN": ["det", "amod", "nsubj", "dobj"]}}}
	]]
	
	# MANY: large quantity and plurality (determiner, adjective, or adverb)
	patterns["MANY"] = [[
		{"RIGHT_ID": "many", "RIGHT_ATTRS": {"LOWER": {"IN": ["many", "numerous", "muchos", "nombreux"]}, "POS": {"IN": ["DET", "ADJ", "ADV"]}, "DEP": {"IN": ["det", "amod", "advmod"]}}}
	]]
	
	# SOME: partial quantity and existence (determiner, pronoun, or adjective)
	patterns["SOME"] = [[
		{"RIGHT_ID": "some", "RIGHT_ATTRS": {"LOWER": {"IN": ["some", "several", "algunos", "quelques"]}, "POS": {"IN": ["DET", "PRON", "ADJ"]}, "DEP": {"IN": ["det", "amod", "nsubj", "dobj"]}}}
	]]
	
	# FEW: small quantity and scarcity (determiner, adjective, or adverb)
	patterns["FEW"] = [[
		{"RIGHT_ID": "few", "RIGHT_ATTRS": {"LOWER": {"IN": ["few", "little", "pocos", "peu"]}, "POS": {"IN": ["DET", "ADJ", "ADV"]}, "DEP": {"IN": ["det", "amod", "advmod"]}}}
	]]
	
	# MUCH: large amount and abundance (determiner, adjective, or adverb)
	patterns["MUCH"] = [[
		{"RIGHT_ID": "much", "RIGHT_ATTRS": {"LOWER": {"IN": ["much", "a lot", "mucho", "beaucoup"]}, "POS": {"IN": ["DET", "ADJ", "ADV"]}, "DEP": {"IN": ["det", "amod", "advmod"]}}}
	]]
	
	# LITTLE: small amount and paucity (determiner, adjective, or adverb)
	patterns["LITTLE"] = [[
		{"RIGHT_ID": "little", "RIGHT_ATTRS": {"LOWER": {"IN": ["little", "small", "poco", "peu"]}, "POS": {"IN": ["DET", "ADJ", "ADV"]}, "DEP": {"IN": ["det", "amod", "advmod"]}}}
	]]
	
	# === PHASE 7: EVALUATORS (NSM Primes) - UD-Based Patterns ===
	
	# GOOD: positive evaluation and desirability (adjective)
	patterns["GOOD"] = [[
		{"RIGHT_ID": "good", "RIGHT_ATTRS": {"LOWER": {"IN": ["good", "great", "excellent", "bueno", "bon"]}, "POS": "ADJ", "DEP": {"IN": ["amod", "attr", "pred"]}}}
	]]
	
	# BAD: negative evaluation and undesirability (adjective)
	patterns["BAD"] = [[
		{"RIGHT_ID": "bad", "RIGHT_ATTRS": {"LOWER": {"IN": ["bad", "terrible", "awful", "malo", "mauvais"]}, "POS": "ADJ", "DEP": {"IN": ["amod", "attr", "pred"]}}}
	]]
	
	# BIG: large size and magnitude (adjective)
	patterns["BIG"] = [[
		{"RIGHT_ID": "big", "RIGHT_ATTRS": {"LOWER": {"IN": ["big", "large", "huge", "grande", "grand"]}, "POS": "ADJ", "DEP": {"IN": ["amod", "attr", "pred"]}}}
	]]
	
	# SMALL: small size and magnitude (adjective)
	patterns["SMALL"] = [[
		{"RIGHT_ID": "small", "RIGHT_ATTRS": {"LOWER": {"IN": ["small", "tiny", "little", "pequeño", "petit"]}, "POS": "ADJ", "DEP": {"IN": ["amod", "attr", "pred"]}}}
	]]
	
	# RIGHT: correctness and appropriateness (adjective or adverb)
	patterns["RIGHT"] = [[
		{"RIGHT_ID": "right", "RIGHT_ATTRS": {"LOWER": {"IN": ["right", "correct", "proper", "correcto", "juste"]}, "POS": {"IN": ["ADJ", "ADV"]}, "DEP": {"IN": ["amod", "attr", "pred", "advmod"]}}}
	]]
	
	# WRONG: incorrectness and inappropriateness (adjective or adverb) - avoid conflict with FALSE
	patterns["WRONG"] = [[
		{"RIGHT_ID": "wrong", "RIGHT_ATTRS": {"LOWER": {"IN": ["wrong", "incorrect", "incorrecto"]}, "POS": {"IN": ["ADJ", "ADV"]}, "DEP": {"IN": ["amod", "attr", "pred", "advmod"]}}}
	]]
	
	# === PHASE 8: ACTIONS (NSM Primes) - UD-Based Patterns ===
	
	# DO: action performance and execution (main verb or complement)
	patterns["DO"] = [[
		{"RIGHT_ID": "do", "RIGHT_ATTRS": {"LOWER": {"IN": ["do", "make", "perform", "hacer", "faire"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# HAPPEN: event occurrence and happening (main verb or complement)
	patterns["HAPPEN"] = [[
		{"RIGHT_ID": "happen", "RIGHT_ATTRS": {"LOWER": {"IN": ["happen", "occur", "take", "pasar", "arriver"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# MOVE: physical movement and motion (main verb or complement)
	patterns["MOVE"] = [[
		{"RIGHT_ID": "move", "RIGHT_ATTRS": {"LOWER": {"IN": ["move", "go", "walk", "mover", "bouger"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# TOUCH: physical contact and interaction (main verb or complement)
	patterns["TOUCH"] = [[
		{"RIGHT_ID": "touch", "RIGHT_ATTRS": {"LOWER": {"IN": ["touch", "contact", "reach", "tocar", "toucher"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# LIVE: existence and being alive (main verb or complement)
	patterns["LIVE"] = [[
		{"RIGHT_ID": "live", "RIGHT_ATTRS": {"LOWER": {"IN": ["live", "exist", "be", "vivir", "vivre"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# DIE: death and cessation of life (main verb or complement)
	patterns["DIE"] = [[
		{"RIGHT_ID": "die", "RIGHT_ATTRS": {"LOWER": {"IN": ["die", "death", "end", "morir", "mourir"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# === PHASE 9: DESCRIPTORS (NSM Primes) - UD-Based Patterns ===
	
	# THIS: proximate reference and identification (determiner)
	patterns["THIS"] = [[
		{"RIGHT_ID": "this", "RIGHT_ATTRS": {"LOWER": {"IN": ["this", "ese", "ce"]}, "POS": "DET", "DEP": {"IN": ["det", "attr"]}}}
	]]
	
	# THE SAME: identity and sameness (determiner or adjective)
	patterns["THE SAME"] = [[
		{"RIGHT_ID": "same", "RIGHT_ATTRS": {"LOWER": {"IN": ["same", "mismo", "même"]}, "POS": {"IN": ["DET", "ADJ"]}, "DEP": {"IN": ["det", "amod", "attr"]}}}
	]]
	
	# OTHER: distinction and difference (determiner, adjective, or pronoun)
	patterns["OTHER"] = [[
		{"RIGHT_ID": "other", "RIGHT_ATTRS": {"LOWER": {"IN": ["other", "otro", "autre"]}, "POS": {"IN": ["DET", "ADJ", "PRON"]}, "DEP": {"IN": ["det", "amod", "attr", "nsubj", "dobj"]}}}
	]]
	
	# ONE: singularity and unity (number or determiner)
	patterns["ONE"] = [[
		{"RIGHT_ID": "one", "RIGHT_ATTRS": {"LOWER": {"IN": ["one", "uno", "un"]}, "POS": {"IN": ["NUM", "DET"]}, "DEP": {"IN": ["nummod", "det"]}}}
	]]
	
	# TWO: duality and pairing (number)
	patterns["TWO"] = [[
		{"RIGHT_ID": "two", "RIGHT_ATTRS": {"LOWER": {"IN": ["two", "dos", "deux"]}, "POS": "NUM", "DEP": "nummod"}}
	]]
	
	# SOME: indefinite quantity and selection (determiner or pronoun)
	patterns["SOME"] = [[
		{"RIGHT_ID": "some", "RIGHT_ATTRS": {"LOWER": {"IN": ["some", "algunos", "quelques"]}, "POS": {"IN": ["DET", "PRON"]}, "DEP": {"IN": ["det", "nsubj", "dobj"]}}}
	]]
	
	# === PHASE 10: INTENSIFIERS (NSM Primes) - UD-Based Patterns ===
	
	# VERY: high degree and intensity (adverb)
	patterns["VERY"] = [[
		{"RIGHT_ID": "very", "RIGHT_ATTRS": {"LOWER": {"IN": ["very", "muy", "très"]}, "POS": "ADV", "DEP": {"IN": ["advmod", "amod"]}}}
	]]
	
	# MORE: comparative degree and increase (adverb or adjective)
	patterns["MORE"] = [[
		{"RIGHT_ID": "more", "RIGHT_ATTRS": {"LOWER": {"IN": ["more", "más", "plus"]}, "POS": {"IN": ["ADV", "ADJ"]}, "DEP": {"IN": ["advmod", "amod", "attr"]}}}
	]]
	
	# LIKE: similarity and resemblance (preposition, adverb, or adjective)
	patterns["LIKE"] = [[
		{"RIGHT_ID": "like", "RIGHT_ATTRS": {"LOWER": {"IN": ["like", "como", "comme"]}, "POS": {"IN": ["ADP", "ADV", "ADJ"]}, "DEP": {"IN": ["prep", "advmod", "amod"]}}}
	]]
	
	# KIND OF: partial degree and approximation (adverb or adjective)
	patterns["KIND OF"] = [[
		{"RIGHT_ID": "kind", "RIGHT_ATTRS": {"LOWER": {"IN": ["kind", "tipo", "genre"]}, "POS": {"IN": ["ADV", "ADJ"]}, "DEP": {"IN": ["advmod", "amod"]}}}
	]]
	
	# === PHASE 11: FINAL PRIMES (NSM Primes) - UD-Based Patterns ===
	
	# SAY: speech and communication (main verb or complement)
	patterns["SAY"] = [[
		{"RIGHT_ID": "say", "RIGHT_ATTRS": {"LOWER": {"IN": ["say", "tell", "speak", "decir", "dire"]}, "POS": "VERB", "DEP": {"IN": ["ROOT", "ccomp", "xcomp"]}}}
	]]
	
	# WORDS: linguistic expression (noun)
	patterns["WORDS"] = [[
		{"RIGHT_ID": "words", "RIGHT_ATTRS": {"LOWER": {"IN": ["word", "words", "palabra", "mot"]}, "POS": "NOUN", "DEP": {"IN": ["nsubj", "dobj", "pobj"]}}}
	]]
	
	# TRUE: truth and factuality (adjective or adverb)
	patterns["TRUE"] = [[
		{"RIGHT_ID": "true", "RIGHT_ATTRS": {"LOWER": {"IN": ["true", "real", "verdadero", "vrai"]}, "POS": {"IN": ["ADJ", "ADV"]}, "DEP": {"IN": ["amod", "attr", "pred", "advmod"]}}}
	]]
	
	# FALSE: falsity and deception (adjective or adverb)
	patterns["FALSE"] = [[
		{"RIGHT_ID": "false", "RIGHT_ATTRS": {"LOWER": {"IN": ["false", "fake", "falso", "faux"]}, "POS": {"IN": ["ADJ", "ADV"]}, "DEP": {"IN": ["amod", "attr", "pred", "advmod"]}}}
	]]
	
	# WHERE: location specification (adverb or pronoun)
	patterns["WHERE"] = [[
		{"RIGHT_ID": "where", "RIGHT_ATTRS": {"LOWER": {"IN": ["where", "dónde", "où"]}, "POS": {"IN": ["ADV", "PRON"]}, "DEP": {"IN": ["advmod", "pobj", "nsubj"]}}}
	]]
	
	# WHEN: time specification (adverb, pronoun, or subordinating conjunction)
	patterns["WHEN"] = [[
		{"RIGHT_ID": "when", "RIGHT_ATTRS": {"LOWER": {"IN": ["when", "cuándo", "quand"]}, "POS": {"IN": ["ADV", "PRON", "SCONJ"]}, "DEP": {"IN": ["advmod", "pobj", "mark"]}}}
	]]

	# IsA: subj and attr under copula
	patterns["IsA"] = [[
		{"SPEC": {"NODE_NAME": "be"}, "PATTERN": {"LEMMA": {"IN": be_lemmas}}},
		{"SPEC": {"NODE_NAME": "subj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}, "REL_OP": ">"},
		{"SPEC": {"NODE_NAME": "attr", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": {"IN": ["attr", "acomp", "oprd"]}, "POS": {"IN": ["NOUN", "PROPN"]}}, "REL_OP": ">"},
	]]
	# AtLocation: ADP with pobj NOUN/PROPN
	patterns["AtLocation"] = [[
		{"SPEC": {"NODE_NAME": "adp"}, "PATTERN": {"POS": "ADP", "LEMMA": {"IN": loc_adps}}},
		{"SPEC": {"NODE_NAME": "pobj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": {"IN": ["pobj", "obj"]}, "POS": {"IN": ["NOUN", "PROPN"]}}, "REL_OP": ">"},
	]]
	# Existential (EN only reliable): there + be
	if lang == "en":
		patterns["Exist"] = [[
			{"SPEC": {"NODE_NAME": "be"}, "PATTERN": {"LEMMA": "be"}},
			{"SPEC": {"NODE_NAME": "there", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": "expl", "LOWER": "there"}, "REL_OP": ">"},
		]]
	# InOrderTo: exact phrase 'in order to' followed by VERB, or 'to' + VERB
	if lang == "en":
		patterns["InOrderTo"] = [[
			{"SPEC": {"NODE_NAME": "in"}, "PATTERN": {"LOWER": "in"}},
			{"SPEC": {"NODE_NAME": "order", "NBOR_RELOP": ">"}, "PATTERN": {"LOWER": "order"}},
			{"SPEC": {"NODE_NAME": "to", "NBOR_RELOP": ">"}, "PATTERN": {"LOWER": "to"}},
		]]
		patterns["InOrderTo_to"] = [[
			{"SPEC": {"NODE_NAME": "to"}, "PATTERN": {"LOWER": {"IN": to_marker}, "POS": {"IN": ["PART", "SCONJ"]}}},
			{"SPEC": {"NODE_NAME": "verb", "NBOR_RELOP": ">"}, "PATTERN": {"POS": "VERB"}},
		]]
	else:
		# ES/FR: 'para'/'pour' + VERB
		patterns["InOrderTo_to"] = [[
			{"SPEC": {"NODE_NAME": "to"}, "PATTERN": {"LOWER": {"IN": to_marker}, "POS": {"IN": ["SCONJ", "ADP"]}}},
			{"SPEC": {"NODE_NAME": "verb", "NBOR_RELOP": ">"}, "PATTERN": {"POS": "VERB"}},
		]]
	# Not: any verb with neg child
	patterns["Not"] = [[
		{
			"RIGHT_ID": "verb",
			"RIGHT_ATTRS": {"POS": "VERB"}
		},
		{
			"LEFT_ID": "verb",
			"REL_OP": ">",
			"RIGHT_ID": "neg", 
			"RIGHT_ATTRS": {"DEP": "neg"}
		}
	]]
	
	# CAUSE: cause/causar/causer + VERB
	patterns["Cause"] = [[
		{
			"RIGHT_ID": "cause",
			"RIGHT_ATTRS": {"LEMMA": {"IN": ["cause", "causar", "causer"]}, "POS": "VERB"}
		},
		{
			"LEFT_ID": "cause",
			"REL_OP": ">",
			"RIGHT_ID": "effect", 
			"RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "xcomp"]}}
		}
	]]
	
	# BEFORE/AFTER: temporal markers
	patterns["Before"] = [[
		{
			"RIGHT_ID": "before",
			"RIGHT_ATTRS": {"LEMMA": {"IN": ["before", "antes", "avant"]}, "POS": {"IN": ["ADP", "SCONJ", "ADV"]}}
		}
	]]
	
	patterns["After"] = [[
		{
			"RIGHT_ID": "after",
			"RIGHT_ATTRS": {"LEMMA": {"IN": ["after", "después", "après"]}, "POS": {"IN": ["ADP", "SCONJ", "ADV"]}}
		}
	]]
	
	# MORE/LESS: comparative constructions
	patterns["More"] = [[
		{
			"RIGHT_ID": "more",
			"RIGHT_ATTRS": {"LEMMA": {"IN": ["more", "más", "plus"]}, "POS": "ADV"}
		}
	]]
	
	patterns["Less"] = [[
		{
			"RIGHT_ID": "less",
			"RIGHT_ATTRS": {"LEMMA": {"IN": ["less", "menos", "moins"]}, "POS": "ADV"}
		}
	]]
	
	# Modality patterns
	if lang == "en":
		# ABILITY: can/could + VERB (modal aux + main verb)
		patterns["Ability"] = [[
			{
				"RIGHT_ID": "modal",
				"RIGHT_ATTRS": {"LEMMA": {"IN": ["can", "could"]}, "POS": "AUX"}
			},
			{
				"LEFT_ID": "modal",
				"REL_OP": "<",
				"RIGHT_ID": "verb", 
				"RIGHT_ATTRS": {"POS": "VERB"}
			}
		]]
		# PERMISSION: may + VERB (or can + I/we)
		patterns["Permission"] = [[
			{
				"RIGHT_ID": "modal",
				"RIGHT_ATTRS": {"LEMMA": "may", "POS": "AUX"}
			},
			{
				"LEFT_ID": "modal",
				"REL_OP": "<",
				"RIGHT_ID": "verb", 
				"RIGHT_ATTRS": {"POS": "VERB"}
			}
		]]
		# OBLIGATION: must/should + VERB or have to + VERB
		patterns["Obligation"] = [[
			{
				"RIGHT_ID": "modal",
				"RIGHT_ATTRS": {"LEMMA": {"IN": ["must", "should"]}, "POS": "AUX"}
			},
			{
				"LEFT_ID": "modal",
				"REL_OP": "<",
				"RIGHT_ID": "verb", 
				"RIGHT_ATTRS": {"POS": "VERB"}
			}
		]]
	elif lang == "es":
		# ABILITY: poder + VERB (Spanish modal aux + main verb)
		patterns["Ability"] = [[
			{
				"RIGHT_ID": "modal",
				"RIGHT_ATTRS": {"LEMMA": "poder", "POS": "AUX"}
			},
			{
				"LEFT_ID": "modal",
				"REL_OP": "<",
				"RIGHT_ID": "verb", 
				"RIGHT_ATTRS": {"POS": "VERB"}
			}
		]]
		# PERMISSION: poder + VERB (context dependent)
		patterns["Permission"] = [[
			{
				"RIGHT_ID": "modal",
				"RIGHT_ATTRS": {"LEMMA": "poder", "POS": "AUX"}
			},
			{
				"LEFT_ID": "modal",
				"REL_OP": "<",
				"RIGHT_ID": "verb", 
				"RIGHT_ATTRS": {"POS": "VERB"}
			}
		]]
		# OBLIGATION: deber + VERB or tener que + VERB
		patterns["Obligation"] = [[
			{
				"RIGHT_ID": "modal",
				"RIGHT_ATTRS": {"LEMMA": "deber", "POS": "AUX"}
			},
			{
				"LEFT_ID": "modal",
				"REL_OP": "<",
				"RIGHT_ID": "verb", 
				"RIGHT_ATTRS": {"POS": "VERB"}
			}
		]]
	else:  # fr
		# ABILITY: pouvoir + VERB (French modal verb + xcomp)
		patterns["Ability"] = [[
			{
				"RIGHT_ID": "modal",
				"RIGHT_ATTRS": {"LEMMA": "pouvoir", "POS": "VERB"}
			},
			{
				"LEFT_ID": "modal",
				"REL_OP": ">",
				"RIGHT_ID": "verb", 
				"RIGHT_ATTRS": {"DEP": "xcomp", "POS": "VERB"}
			}
		]]
		# PERMISSION: pouvoir + VERB (context dependent)
		patterns["Permission"] = [[
			{
				"RIGHT_ID": "modal",
				"RIGHT_ATTRS": {"LEMMA": "pouvoir", "POS": "VERB"}
			},
			{
				"LEFT_ID": "modal",
				"REL_OP": ">",
				"RIGHT_ID": "verb", 
				"RIGHT_ATTRS": {"DEP": "xcomp", "POS": "VERB"}
			}
		]]
		# OBLIGATION: devoir + VERB
		patterns["Obligation"] = [[
			{
				"RIGHT_ID": "modal",
				"RIGHT_ATTRS": {"LEMMA": "devoir", "POS": "VERB"}
			},
			{
				"LEFT_ID": "modal",
				"REL_OP": ">",
				"RIGHT_ID": "verb", 
				"RIGHT_ATTRS": {"DEP": "xcomp", "POS": "VERB"}
			}
		]]
	
	# PartOf: 'part of' + NOUN
	if lang == "en":
		patterns["PartOf"] = [[
			{"SPEC": {"NODE_NAME": "part"}, "PATTERN": {"LEMMA": "part", "POS": "NOUN"}},
			{"SPEC": {"NODE_NAME": "of", "NBOR_RELOP": ">"}, "PATTERN": {"LOWER": "of"}, "REL_OP": ">"},
			{"SPEC": {"NODE_NAME": "pobj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": "pobj", "POS": {"IN": ["NOUN", "PROPN"]}}, "REL_OP": ">"},
		]]
	else:
		# ES/FR: 'parte de' / 'partie de'
		patterns["PartOf"] = [[
			{"SPEC": {"NODE_NAME": "part"}, "PATTERN": {"LOWER": part_of_tokens[0], "POS": "NOUN"}},
			{"SPEC": {"NODE_NAME": "de", "NBOR_RELOP": ">"}, "PATTERN": {"LOWER": part_of_tokens[1]} , "REL_OP": ">"},
			{"SPEC": {"NODE_NAME": "obj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": {"IN": ["pobj", "obj"]}, "POS": {"IN": ["NOUN", "PROPN"]}}, "REL_OP": ">"},
		]]
	# Because: token 'because'/'since' SCONJ/ADP (single-node pattern)
	patterns["Because"] = [[
		{"SPEC": {"NODE_NAME": "cue"}, "PATTERN": {"LOWER": {"IN": because_tokens}}},
	]]
	# If: token 'if' SCONJ
	patterns["If"] = [[
		{"SPEC": {"NODE_NAME": "if"}, "PATTERN": {"LOWER": {"IN": ["if", "si"]}}},
	]]
	# SuchAs: 'such as' + NOUN (exemplification)
	patterns["SuchAs"] = [[
		{"SPEC": {"NODE_NAME": "such"}, "PATTERN": {"LOWER": "such"}},
		{"SPEC": {"NODE_NAME": "as", "NBOR_RELOP": ">"}, "PATTERN": {"LOWER": "as"}, "REL_OP": ">"},
		{"SPEC": {"NODE_NAME": "pobj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": "pobj", "POS": {"IN": ["NOUN", "PROPN"]}}, "REL_OP": ">"},
	]]
	# ConsistsOf: 'consists of' + NOUN (composition)
	if lang == "en":
		patterns["ConsistsOf"] = [[
			{"SPEC": {"NODE_NAME": "consists"}, "PATTERN": {"LEMMA": "consist", "POS": "VERB"}},
			{"SPEC": {"NODE_NAME": "of", "NBOR_RELOP": ">"}, "PATTERN": {"LOWER": "of"}, "REL_OP": ">"},
			{"SPEC": {"NODE_NAME": "pobj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": "pobj", "POS": {"IN": ["NOUN", "PROPN"]}}, "REL_OP": ">"},
		]]
	# else: skip for now
	# Causes: 'causes'/'caused by' + NOUN/VERB (causation)
	patterns["Causes"] = [[
		{"SPEC": {"NODE_NAME": "causes"}, "PATTERN": {"LEMMA": {"IN": ["cause", "causar", "causer"]}, "POS": "VERB"}},
		{"SPEC": {"NODE_NAME": "obj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": {"IN": ["dobj", "obj"]}}, "REL_OP": ">"},
	]]
	# SimilarTo: 'similar to'/'like' + NOUN (similarity)
	patterns["SimilarTo"] = [[
		{"SPEC": {"NODE_NAME": "similar"}, "PATTERN": {"LEMMA": {"IN": sim_tokens}, "POS": "ADJ"}},
		{"SPEC": {"NODE_NAME": "to", "NBOR_RELOP": ">"}, "PATTERN": {"LOWER": {"IN": ["to", "a", "à"]}}, "REL_OP": ">"},
		{"SPEC": {"NODE_NAME": "pobj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": {"IN": ["pobj", "obj"]}, "POS": {"IN": ["NOUN", "PROPN"]}}, "REL_OP": ">"},
	]]
	patterns["Like"] = [[
		{"SPEC": {"NODE_NAME": "like"}, "PATTERN": {"LOWER": {"IN": like_adp}, "POS": "ADP"}},
		{"SPEC": {"NODE_NAME": "pobj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": {"IN": ["pobj", "obj"]}, "POS": {"IN": ["NOUN", "PROPN"]}}, "REL_OP": ">"},
	]]
	# DifferentFrom: 'different from' + NOUN (contrast)
	patterns["DifferentFrom"] = [[
		{"SPEC": {"NODE_NAME": "different"}, "PATTERN": {"LEMMA": {"IN": diff_tokens}, "POS": "ADJ"}},
		{"SPEC": {"NODE_NAME": "from", "NBOR_RELOP": ">"}, "PATTERN": {"LOWER": {"IN": ["from", "de"]}}, "REL_OP": ">"},
		{"SPEC": {"NODE_NAME": "pobj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": {"IN": ["pobj", "obj"]}, "POS": {"IN": ["NOUN", "PROPN"]}}, "REL_OP": ">"},
	]]
	# HasProperty: 'has'/'with' + ADJ (possession/attribute)
	patterns["HasProperty"] = [[
		{"SPEC": {"NODE_NAME": "has"}, "PATTERN": {"LEMMA": {"IN": have_lemmas}, "POS": "VERB"}},
		{"SPEC": {"NODE_NAME": "adj", "NBOR_RELOP": ">"}, "PATTERN": {"POS": "ADJ"}, "REL_OP": ">"},
	]]
	# UsedFor: 'used for' + VERB/NOUN (purpose)
	patterns["UsedFor"] = [[
		{"SPEC": {"NODE_NAME": "used"}, "PATTERN": {"LEMMA": {"IN": ["use", "usar", "utiliser"]}, "POS": "VERB"}},
		{"SPEC": {"NODE_NAME": "for", "NBOR_RELOP": ">"}, "PATTERN": {"LOWER": {"IN": used_for_prep}}, "REL_OP": ">"},
		{"SPEC": {"NODE_NAME": "pobj", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": {"IN": ["pobj", "obj"]}}, "REL_OP": ">"},
	]]
	
	# STILL: continuation of state/action
	patterns["Still"] = [[
		{"RIGHT_ID": "still", "RIGHT_ATTRS": {"LOWER": {"IN": ["still", "todavía", "aún", "encore", "toujours"]}, "POS": {"IN": ["ADV", "ADJ"]}}}
	]]
	
	# NOT_YET: anticipated completion (negative + temporal)
	patterns["NotYet"] = [[
		{"RIGHT_ID": "yet", "RIGHT_ATTRS": {"LOWER": {"IN": ["yet", "todavía", "aún", "encore"]}, "POS": "ADV"}}
	]]
	
	# START: beginning of action/state
	patterns["Start"] = [[
		{"RIGHT_ID": "start", "RIGHT_ATTRS": {"LEMMA": {"IN": ["start", "begin", "empezar", "comenzar", "commencer", "débuter"]}, "POS": "VERB"}}
	]]
	
	# FINISH: completion of action/state
	patterns["Finish"] = [[
		{"RIGHT_ID": "finish", "RIGHT_ATTRS": {"LEMMA": {"IN": ["finish", "complete", "end", "terminar", "acabar", "finir", "terminer"]}, "POS": "VERB"}}
	]]
	
	# AGAIN: repetition of action
	patterns["Again"] = [[
		{"RIGHT_ID": "again", "RIGHT_ATTRS": {"LOWER": {"IN": ["again", "otra", "vez", "nuevo", "encore", "nouveau"]}, "POS": {"IN": ["ADV", "ADJ"]}}}
	]]
	
	# KEEP: continuation/maintenance of action
	patterns["Keep"] = [[
		{"RIGHT_ID": "keep", "RIGHT_ATTRS": {"LEMMA": {"IN": ["keep", "continue", "seguir", "continuar", "continuer"]}, "POS": "VERB"}}
	]]
	
	added = 0
	for name, pat in patterns.items():
		try:
			matcher.add(name, pat)
			added += 1
		except Exception:
			continue
	if added == 0:
		return []
	matches = matcher(doc)
	out: List[str] = []
	for match_id, _ in matches:
		out.append(nlp.vocab.strings[match_id])
	# Post-process: unify pattern variants
	final: List[str] = []
	for n in out:
		if n == "InOrderTo_to":
			final.append("InOrderTo")
		elif n == "Like":
			final.append("SimilarTo")
		elif n in ["SOMEONE_A_PERSON"]:
			final.append("SOMEONE")
		elif n in ["SOMETHING_AN_EVENT"]:
			final.append("SOMETHING")
		elif n in ["THING_THE_OBJECT"]:
			final.append("THING")
		elif n in ["BODY_THE_PERSON"]:
			final.append("BODY")
		else:
			final.append(n)
	return final


def detect_primitives_lexical(text: str) -> List[str]:
	"""Lexical fallback detector for EN/ES/FR without dependency parse.

	Uses simple phrase cues to detect a subset of NSM-like primitives.
	Honest, conservative: exact substrings with spaces/punctuation boundaries.
	"""
	lower = f" {text.lower()} "
	out: List[str] = []

	# IsA / HasProperty cues
	if any(p in lower for p in [" is a ", " is an ", " es un ", " es una ", " est un ", " est une "]):
		out.append("IsA")
	# HasProperty (additional patterns)
	if any(p in lower for p in [" is ", " es ", " est ", " tiene ", " a ", " tiene de "]):
		out.append("HasProperty")
	# PartOf
	if any(p in lower for p in [" part of ", " parte de ", " partie de "]):
		out.append("PartOf")
	# AtLocation
	if any(p in lower for p in [" is on ", " is in ", " está en ", " está sobre ", " est sur ", " est dans "]):
		out.append("AtLocation")
	# UsedFor
	if any(p in lower for p in [" used for ", " usado para ", " utilisée pour ", " utilisé pour ", " sert à ", " sirve para ", " se usa para ", " se utiliza para "]):
		out.append("UsedFor")
	# Causes
	if any(p in lower for p in [" causes ", " causa ", " cause "]):
		out.append("Causes")
	# SimilarTo
	if any(p in lower for p in [" similar to ", " similar a ", " similaire à ", " parecido a ", " pareil à ", " semblable à "]):
		out.append("SimilarTo")
	# DifferentFrom
	if any(p in lower for p in [" different from ", " diferente de ", " différent de ", " distinto de "]):
		out.append("DifferentFrom")
	# Not
	if any(p in lower for p in [" not ", " no ", " ne "]):
		out.append("Not")
	# Exist
	if any(p in lower for p in [" there is ", " there are ", " existe ", " il y a "]):
		out.append("Exist")

	# STILL: continuation
	if any(p in lower for p in [" still ", " todavía ", " aún ", " encore ", " toujours "]):
		out.append("Still")

	# NOT_YET: anticipated completion
	if any(p in lower for p in [" not yet ", " not yet ", " aún no ", " todavía no ", " pas encore "]):
		out.append("NotYet")

	# START: beginning
	if any(p in lower for p in [" start ", " begin ", " empezar ", " comenzar ", " commencer ", " débuter "]):
		out.append("Start")

	# FINISH: completion
	if any(p in lower for p in [" finish ", " complete ", " end ", " terminar ", " acabar ", " finir ", " terminer "]):
		out.append("Finish")

	# AGAIN: repetition
	if any(p in lower for p in [" again ", " otra vez ", " de nuevo ", " encore ", " de nouveau "]):
		out.append("Again")

	# KEEP: continuation
	if any(p in lower for p in [" keep ", " continue ", " seguir ", " continuar ", " continuer "]):
		out.append("Keep")

	# === PHASE 1: CORE SUBSTANTIVES (NSM Primes) ===
	
	# I: first person singular pronoun
	if any(p in lower for p in [" i ", " me ", " my ", " myself ", " yo ", " je "]):
		out.append("I")
	
	# YOU: second person pronoun
	if any(p in lower for p in [" you ", " your ", " yourself ", " tú ", " tu ", " vous "]):
		out.append("YOU")
	
	# SOMEONE: indefinite person
	if any(p in lower for p in [" someone ", " somebody ", " anyone ", " anybody ", " alguien ", " quelqu'un "]):
		out.append("SOMEONE")
	
	# PEOPLE: plural persons
	if any(p in lower for p in [" people ", " persons ", " humans ", " gente ", " personnes "]):
		out.append("PEOPLE")
	
	# SOMETHING: indefinite thing
	if any(p in lower for p in [" something ", " anything ", " whatever ", " algo ", " quelque chose "]):
		out.append("SOMETHING")
	
	# THING: generic object
	if any(p in lower for p in [" thing ", " object ", " item ", " cosa ", " objet "]):
		out.append("THING")
	
	# BODY: physical entity
	if any(p in lower for p in [" body ", " person ", " human ", " cuerpo ", " personne "]):
		out.append("BODY")

	return out


def detect_primitives_multilingual(text: str) -> List[str]:
	"""Run dependency-based detection if available; otherwise lexical fallback."""
	res = detect_primitives_dep(text)
	if res:
		return res
	return detect_primitives_lexical(text)

