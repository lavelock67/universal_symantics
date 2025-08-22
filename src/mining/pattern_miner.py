"""Dependency/SRL-style pattern miner for candidate primitive discovery.

Extract frequent, simple argumented patterns from text corpora using spaCy
UD parses. Output candidates with pattern signature, example spans, and
frequency stats. Optionally merge high-support candidates into a new table.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import click
from tqdm import tqdm

from ..table.schema import PeriodicTable, Primitive, PrimitiveSignature, PrimitiveCategory


logger = logging.getLogger(__name__)


class PatternMiner:
    """Mines dependency patterns from text corpora to discover candidate primitives."""
    
    def __init__(self):
        """Initialize the pattern miner with spaCy models."""
        try:
            import spacy
            self.nlp_en = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp_en = None
    
    def _extract_patterns(self, texts: List[str], min_support: int = 5) -> List[PatternCandidate]:
        """Extract dependency patterns from text corpora.
        
        Args:
            texts: List of text strings to mine from
            min_support: Minimum frequency for a pattern to be considered
            
        Returns:
            List of pattern candidates
        """
        patterns: Dict[str, Dict[str, Any]] = {}
        
        # Function words to filter out (low semantic value)
        FUNCTION_WORDS = {
            'be', 'have', 'do', 'get', 'make', 'go', 'come', 'take', 'see', 'know',
            'think', 'say', 'want', 'need', 'like', 'look', 'work', 'call', 'try',
            'ask', 'tell', 'give', 'find', 'put', 'keep', 'let', 'help', 'show',
            'play', 'run', 'move', 'turn', 'start', 'stop', 'begin', 'end', 'use',
            'feel', 'seem', 'become', 'remain', 'stay', 'leave', 'arrive', 'reach',
            'enter', 'exit', 'open', 'close', 'break', 'fix', 'build', 'create',
            'destroy', 'change', 'grow', 'shrink', 'increase', 'decrease', 'rise',
            'fall', 'win', 'lose', 'buy', 'sell', 'send', 'receive', 'write', 'read',
            'speak', 'listen', 'watch', 'hear', 'smell', 'taste', 'touch', 'hold',
            'carry', 'bring', 'send', 'return', 'forget', 'remember', 'learn', 'teach',
            'study', 'understand', 'explain', 'describe', 'discuss', 'agree', 'disagree',
            'believe', 'doubt', 'hope', 'worry', 'enjoy', 'hate', 'love', 'miss',
            'meet', 'join', 'leave', 'follow', 'lead', 'guide', 'direct', 'control',
            'manage', 'organize', 'plan', 'decide', 'choose', 'select', 'pick',
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'when', 'where',
            'how', 'why', 'what', 'which', 'who', 'whom', 'whose', 'that', 'this',
            'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'hers', 'i',
            'me', 'my', 'mine', 'is', 'are', 'was', 'were', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        # High-value dependency relations (semantic content)
        SEMANTIC_DEPS = {
            'nsubj', 'dobj', 'iobj', 'pobj', 'attr', 'acomp', 'oprd',  # Core arguments
            'advcl', 'ccomp', 'xcomp', 'csubj', 'csubjpass',  # Clausal relations
            'acl', 'relcl', 'appos', 'nummod', 'amod', 'compound',  # Modifiers
            'prep', 'agent', 'pcomp', 'prt', 'quantmod',  # Prepositional/complex
            'expl', 'aux', 'auxpass', 'cop', 'mark', 'discourse',  # Functional but useful
            'det', 'predet', 'poss', 'case', 'cc', 'conj', 'list',  # Structural
            'parataxis', 'vocative', 'dep', 'root'  # Other
        }
        
        # Low-value dependency relations (grammatical function)
        GRAMMATICAL_DEPS = {
            'punct', 'cc', 'det', 'predet', 'aux', 'auxpass', 'cop', 'mark',
            'discourse', 'expl', 'case', 'cc:preconj'
        }
        
        for text in texts:
            if not text.strip():
                continue
                
            try:
                # Use English model for now (can be extended)
                doc = self.nlp_en(text)
                
                for token in doc:
                    # Skip function words and punctuation
                    if (token.lemma_.lower() in FUNCTION_WORDS or 
                        token.pos_ in {'PUNCT', 'SPACE', 'X'} or
                        token.dep_ in GRAMMATICAL_DEPS):
                        continue
                    
                    # Focus on content words with semantic dependencies
                    if (token.pos_ in {'NOUN', 'VERB', 'ADJ', 'PROPN'} and
                        token.dep_ in SEMANTIC_DEPS):
                        
                        # Create pattern: head_lemma-dep-child_lemma
                        if token.head != token:  # Not root
                            pattern_key = f"{token.head.lemma_}-{token.dep_}-{token.lemma_}"
                            
                            if pattern_key not in patterns:
                                patterns[pattern_key] = {
                                    'key': pattern_key,
                                    'head': token.head.lemma_,
                                    'relation': token.dep_,
                                    'arg_types': [token.head.pos_, token.pos_],
                                    'frequency': 0,
                                    'examples': [],
                                    'head_pos': token.head.pos_,
                                    'child_pos': token.pos_,
                                    'head_dep': token.head.dep_,
                                    'child_dep': token.dep_
                                }
                            
                            patterns[pattern_key]['frequency'] += 1
                            if len(patterns[pattern_key]['examples']) < 3:
                                patterns[pattern_key]['examples'].append({
                                    'text': text[:100] + '...' if len(text) > 100 else text,
                                    'head': token.head.text,
                                    'child': token.text,
                                    'head_lemma': token.head.lemma_,
                                    'child_lemma': token.lemma_
                                })
                            
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                continue
        
        # Filter by minimum support and convert to candidates
        candidates = []
        for pattern_data in patterns.values():
            if pattern_data['frequency'] >= min_support:
                # Additional filtering: prefer patterns with content words
                head_pos = pattern_data['head_pos']
                child_pos = pattern_data['child_pos']
                
                # Score based on content word combinations
                score = 0
                if head_pos in {'NOUN', 'VERB', 'ADJ', 'PROPN'}:
                    score += 2
                if child_pos in {'NOUN', 'VERB', 'ADJ', 'PROPN'}:
                    score += 2
                if head_pos == 'VERB' and child_pos == 'NOUN':  # Verb-object
                    score += 3
                if head_pos == 'NOUN' and child_pos == 'ADJ':   # Noun-adjective
                    score += 2
                if head_pos == 'VERB' and child_pos == 'ADJ':   # Verb-adjective
                    score += 2
                
                # Only keep high-scoring patterns
                if score >= 3:
                    candidates.append(PatternCandidate(
                        key=pattern_data['key'],
                        head=pattern_data['head'],
                        relation=pattern_data['relation'],
                        arg_types=pattern_data['arg_types'],
                        frequency=pattern_data['frequency'],
                        examples=pattern_data['examples']
                    ))
        
        # Sort by frequency and score
        candidates.sort(key=lambda x: x.frequency, reverse=True)
        
        return candidates
    
    def _candidate_to_primitive(self, c: PatternCandidate, name_prefix: str = "Cand") -> Primitive:
        """Convert a pattern candidate to a primitive with a meaningful name."""
        
        # Generate a unique name based on the pattern content
        head_word = c.head
        relation = c.relation
        child_word = c.key.split('-')[-1] if '-' in c.key else "unknown"
        
        # Create unique names that incorporate the actual words
        if c.relation == 'amod':
            # Adjective modifier: "X is Y" -> "HasProperty_Y"
            name = f"HasProperty_{child_word.capitalize()}"
        elif c.relation == 'dobj':
            # Direct object: "X does Y" -> "DoesAction_Y"
            name = f"DoesAction_{child_word.capitalize()}"
        elif c.relation == 'compound':
            # Compound noun: "X Y" -> "IsTypeOf_Y"
            name = f"IsTypeOf_{child_word.capitalize()}"
        elif c.relation == 'nsubj':
            # Subject: "X does" -> "IsActor_X"
            name = f"IsActor_{head_word.capitalize()}"
        elif c.relation == 'pobj':
            # Prepositional object: "X in/at Y" -> "AtLocation_Y"
            name = f"AtLocation_{child_word.capitalize()}"
        elif c.relation == 'advcl':
            # Adverbial clause: "X when Y" -> "WhenCondition_Y"
            name = f"WhenCondition_{child_word.capitalize()}"
        elif c.relation == 'ccomp':
            # Clausal complement: "X that Y" -> "ThatContent_Y"
            name = f"ThatContent_{child_word.capitalize()}"
        elif c.relation == 'xcomp':
            # Open clausal complement: "X to Y" -> "InOrderTo_Y"
            name = f"InOrderTo_{child_word.capitalize()}"
        elif c.relation == 'attr':
            # Attribute: "X is Y" -> "IsA_Y"
            name = f"IsA_{child_word.capitalize()}"
        elif c.relation == 'prep':
            # Preposition: "X of Y" -> "PartOf_Y"
            name = f"PartOf_{child_word.capitalize()}"
        else:
            # Generic name based on relation and words
            name = f"{name_prefix}_{relation}_{head_word}_{child_word}"
        
        # Determine arity based on the pattern
        if c.relation in ['amod', 'compound', 'attr']:
            arity = 2  # Binary relations
        elif c.relation in ['dobj', 'nsubj', 'pobj', 'advcl', 'ccomp', 'xcomp']:
            arity = 2  # Binary relations
        else:
            arity = len(c.arg_types) + 1  # Default to number of arguments + head
        
        # Determine category based on semantic content
        if c.relation in ['amod', 'attr', 'compound']:
            category = PrimitiveCategory.INFORMATIONAL
        elif c.relation in ['dobj', 'nsubj', 'pobj']:
            category = PrimitiveCategory.STRUCTURAL
        elif c.relation in ['advcl', 'ccomp', 'xcomp']:
            category = PrimitiveCategory.LOGICAL
        else:
            category = PrimitiveCategory.INFORMATIONAL
        
        return Primitive(
            name=name,
            category=category,
            signature=PrimitiveSignature(arity=arity),
            description=f"Pattern '{c.key}' with frequency {c.frequency}",
            examples=c.examples[:3]
        )


def _load_nlp():
    try:
        import spacy  # type: ignore
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            return None
    except Exception:
        return None


def _load_srl():
    try:
        from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
        from allennlp.models.archival import load_archive
        archive = load_archive('https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz')
        return SentenceTaggerPredictor.from_archive(archive, 'semantic_role_labeling')
    except Exception:
        return None


_NLP = _load_nlp()
_SRL = _load_srl()

_SEMANTIC_DEPS = {"nsubj", "nsubjpass", "obj", "dobj", "iobj", "obl", "nmod", "cop", "mark", "ccomp", "xcomp", "advcl", "acl"}

def _load_nlp_multi() -> Dict[str, Any]:
    try:
        import spacy
        pipes: Dict[str, Any] = {}
        for code, model in (
            ("en", "en_core_web_sm"),
            ("es", "es_core_news_sm"),
            ("fr", "fr_core_news_sm"),
        ):
            try:
                pipes[code] = spacy.load(model)
            except Exception:
                pipes[code] = None
        return pipes
    except Exception:
        return {}

_NLP_MULTI = _load_nlp_multi()

def _guess_lang_from_path(path: str) -> str:
    p = path.lower()
    if "_en" in p or "/en_" in p or "/eng" in p:
        return "en"
    if "_es" in p or "/es_" in p or "/spa" in p:
        return "es"
    if "_fr" in p or "/fr_" in p or "/fra" in p:
        return "fr"
    return "en"


@dataclass
class PatternCandidate:
    key: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    args_schema: Dict[str, str] = field(default_factory=dict)
    head: str = "?"             
    relation: str = "?"         
    arg_types: List[str] = field(default_factory=list)  


def _read_corpora(paths: Iterable[str]) -> List[str]:
    lines: List[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for ln in fh:
                t = ln.strip()
                if t:
                    lines.append(t)
    return lines


def _candidate_to_primitive(c: PatternCandidate, name_prefix: str = "Cand") -> Primitive:
    name = f"{name_prefix}_{c.relation}_{'_'.join(c.arg_types)}"
    arity = len(c.args_schema) + 1  # +1 for head
    category = PrimitiveCategory.STRUCTURAL if "prep" in c.relation or "loc" in c.relation else PrimitiveCategory.INFORMATIONAL
    return Primitive(
        name=name,
        category=category,
        signature=PrimitiveSignature(arity=arity),
        description=f"Pattern '{c.key}' with frequency {c.frequency}",
        examples=c.examples[:3]
    )


@click.command()
@click.option("--corpus", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True,
              help="Corpus file(s) to mine")
@click.option("--min-support", default=20, show_default=True, help="Minimum frequency to keep a pattern")
@click.option("--top-k", default=50, show_default=True, help="Top-K patterns by frequency")
@click.option("--out", default="candidates.json", show_default=True, help="Output candidates JSON")
@click.option("--merge-into", default=None, type=click.Path(dir_okay=False), help="Optional path to write a new table JSON with merged candidates")
@click.option("--base-table", default="primitives.json", show_default=True, type=click.Path(exists=True), help="Base table to merge into")
def main(corpus: Tuple[str, ...], min_support: int, top_k: int, out: str, merge_into: str | None, base_table: str) -> None:
    """Mine dependency patterns from text corpora."""
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load texts from corpus files
    texts = []
    for corpus_file in corpus:
        if not Path(corpus_file).exists():
            logger.error(f"Corpus file not found: {corpus_file}")
            continue
        texts.extend(Path(corpus_file).read_text(encoding="utf-8").splitlines())
    
    logger.info(f"Mining patterns from {len(texts)} text lines")
    
    # Initialize pattern miner and extract patterns
    miner = PatternMiner()
    patterns = miner._extract_patterns(texts, min_support=min_support)
    
    # Take top-k patterns
    patterns = patterns[:top_k]
    
    logger.info(f"Found {len(patterns)} patterns with min_support={min_support}")
    
    # Convert to JSON format
    candidates = []
    for pattern in patterns:
        candidates.append({
            "key": pattern.key,
            "name": miner._candidate_to_primitive(pattern).name,
            "category": miner._candidate_to_primitive(pattern).category.value,
            "arity": miner._candidate_to_primitive(pattern).arity,
            "frequency": pattern.frequency,
            "examples": pattern.examples,
            "head": pattern.head,
            "relation": pattern.relation,
            "arg_types": pattern.arg_types
        })
    
    # Write candidates
    Path(out).write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    logger.info(f"Wrote {len(candidates)} candidates to {out}")
    
    # Optionally merge into base table
    if merge_into:
        if not Path(base_table).exists():
            logger.error(f"Base table not found: {base_table}")
            return
        
        # Load base table using from_dict
        base = PeriodicTable.from_dict(json.load(open(base_table)))
        added = 0
        
        for pattern in patterns:
            try:
                primitive = miner._candidate_to_primitive(pattern)
                if not base.get_primitive(primitive.name):
                    base.add_primitive(primitive)
                    added += 1
            except Exception as e:
                logger.warning(f"Failed to add pattern {pattern.key}: {e}")
        
        # Save merged table
        Path(merge_into).write_text(json.dumps(base.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Merged {added} new primitives into {merge_into}")


if __name__ == "__main__":
    main()


