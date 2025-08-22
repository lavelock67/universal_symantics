"""Optional SRL/UD-style detectors using spaCy if available.

These detectors use light dependency patterns to infer primitives from text.
If spaCy or the language model is not available, the functions return empty lists.
"""

from typing import List, Dict, Any

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
	# Accented characters
	spanish_accents = set("áéíóúñü")
	french_accents = set("àâçéèêëîïôùûüÿœ")
	if any(c in spanish_accents for c in lower):
		return "es"
	if any(c in french_accents for c in lower):
		return "fr"
	# Stopword cues
	es_sw = (" el ", " la ", " los ", " las ", " de ", " que ", " para ")
	fr_sw = (" le ", " la ", " les ", " des ", " que ", " pour ", " pas ")
	if any(w in f" {lower} " for w in es_sw):
		return "es"
	if any(w in f" {lower} " for w in fr_sw):
		return "fr"
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

	# Dependency-based quantifier patterns: det(head)=all/some/many/few/most/no
	for token in doc:
		if token.dep_ == "det" and token.lemma_.lower() in {"all", "some", "many", "few", "most", "no"}:
			lemma = token.lemma_.lower()
			if lemma == "all":
				detected.append("All")
			elif lemma == "some":
				detected.append("Some")
			elif lemma == "many":
				detected.append("Many")
			elif lemma == "few":
				detected.append("Few")
			elif lemma == "most":
				detected.append("Most")
			elif lemma == "no":
				detected.append("None")
			break

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

	# Quantifiers
	for token in doc:
		if token.dep_ == "det" and token.lemma_.lower() in {"all", "some", "many", "few", "most", "no"}:
			add(token.lemma_.capitalize() if token.lemma_ != "no" else "None", [token.head.text], 0.6)
			break

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
		{"SPEC": {"NODE_NAME": "verb"}, "PATTERN": {"POS": "VERB"}},
		{"SPEC": {"NODE_NAME": "neg", "NBOR_RELOP": ">"}, "PATTERN": {"DEP": "neg"}, "REL_OP": ">"},
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
	# Post-process: unify InOrderTo variants and similar patterns
	final: List[str] = []
	for n in out:
		if n == "InOrderTo_to":
			final.append("InOrderTo")
		elif n == "Like":
			final.append("SimilarTo")
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

	return out


def detect_primitives_multilingual(text: str) -> List[str]:
	"""Run dependency-based detection if available; otherwise lexical fallback."""
	res = detect_primitives_dep(text)
	if res:
		return res
	return detect_primitives_lexical(text)

