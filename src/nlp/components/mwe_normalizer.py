"""
MWE Normalizer - SpaCy pipe component for multi-word expression normalization.

This component runs BEFORE UD parsing to normalize multi-word adpositions
and provide canonical forms for downstream processing.

Key features:
- Normalizes spatial MWEs (in front of, au-dessus de, cerca de, etc.)
- Applies figurative guards to prevent false spatial hits
- Exposes spans via doc._.mwe_spans_by_token
- Uses assets/spatial_maps.json as single source of truth
- Maintains ADR-001 compliance (hints only, no prime emission)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Doc, Span, SpanGroup
from spacy.language import Language

# Register extensions
if not Doc.has_extension("mwe_spans_by_token"):
    Doc.set_extension("mwe_spans_by_token", default={}, force=True)
if not Span.has_extension("mwe_meta"):
    Span.set_extension("mwe_meta", default={}, force=True)


def load_spatial_maps() -> Dict[str, Dict]:
    """Load spatial maps from assets/spatial_maps.json."""
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent.parent.parent / "assets" / "spatial_maps.json",  # src/nlp/components -> assets
        Path(__file__).parent.parent.parent.parent / "assets" / "spatial_maps.json",  # src/nlp/components -> assets (alternative)
        Path.cwd() / "assets" / "spatial_maps.json",  # Current working directory
    ]
    
    for maps_path in possible_paths:
        if maps_path.exists():
            with open(maps_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    # If no file found, return empty dict
    return {}


class MWENormalizer:
    """
    SpaCy pipe component for MWE normalization.
    
    Runs before UD parsing to normalize multi-word adpositions and provide
    canonical forms for downstream semantic analysis.
    """
    
    def __init__(self, nlp: Language, spatial_maps: Optional[Dict] = None, merge_adp: bool = False):
        """
        Initialize MWE normalizer.
        
        Args:
            nlp: SpaCy language model
            spatial_maps: Spatial maps dictionary (loads from assets if None)
            merge_adp: Whether to merge ADP tokens (default: False)
        """
        self.nlp = nlp
        self.lang = nlp.lang
        self.merge_adp = merge_adp
        
        # Load spatial maps
        if spatial_maps is None:
            spatial_maps = load_spatial_maps()
        self.maps = spatial_maps.get(self.lang, {})
        
        # Initialize matchers
        self.matcher = Matcher(nlp.vocab)
        self.dep_matcher = DependencyMatcher(nlp.vocab)
        
        # Build patterns
        self._build_patterns()
        
        # Initialize metrics
        self.metrics = {
            'mwe_detected_total': 0,
            'mwe_guard_filtered_total': 0,
            'mwe_retokenize_applied_total': 0
        }
    
    def _build_patterns(self):
        """Build SpaCy matcher patterns from spatial maps."""
        if not self.maps:
            return
        
        for entry in self.maps.get('entries', []):
            entry_id = entry.get('id', '')
            prime_hint = entry.get('prime_hint', '')
            canonical_lemma = entry.get('canonical_lemma', '')
            
            for variant in entry.get('surface_variants', []):
                pattern = variant.get('pattern', [])
                tag_constraints = variant.get('tag_constraints', [])
                
                # Build pattern for Matcher
                matcher_pattern = []
                for i, token in enumerate(pattern):
                    token_dict = {"TEXT": token.lower()}
                    if i < len(tag_constraints):
                        # Map tag constraints to POS constraints
                        tag = tag_constraints[i]
                        if tag == "ADP":
                            token_dict["POS"] = "ADP"
                        elif tag == "ADJ":
                            token_dict["POS"] = "ADJ"
                        elif tag == "NOUN":
                            token_dict["POS"] = "NOUN"
                        elif tag == "DET":
                            token_dict["POS"] = "DET"
                        elif tag == "ADV":
                            token_dict["POS"] = "ADV"
                        elif tag == "SCONJ":
                            token_dict["POS"] = "SCONJ"
                        elif tag == "IN":
                            # Map IN tag to both ADP and SCONJ since it can be either
                            token_dict["POS"] = "ADP"
                        elif tag == "RB":
                            # Map RB tag to both ADV and ADP since "inside" can be either
                            token_dict["POS"] = "ADV"
                        else:
                            # Fallback to TAG if POS mapping not found
                            token_dict["TAG"] = tag
                    matcher_pattern.append(token_dict)
                
                if matcher_pattern:
                    self.matcher.add(
                        f"{entry_id}_{len(matcher_pattern)}",
                        [matcher_pattern],
                        on_match=self._on_match
                    )
                    
                    # For patterns with "than", also add a variant with ADP instead of SCONJ
                    if any("SCONJ" in str(token_dict) for token_dict in matcher_pattern):
                        adp_pattern = []
                        for token_dict in matcher_pattern:
                            new_dict = token_dict.copy()
                            if "POS" in new_dict and new_dict["POS"] == "SCONJ":
                                # Create ADP variant
                                new_dict["POS"] = "ADP"
                            adp_pattern.append(new_dict)
                        
                        if adp_pattern != matcher_pattern:
                            self.matcher.add(
                                f"{entry_id}_{len(matcher_pattern)}_adp",
                                [adp_pattern],
                                on_match=self._on_match
                            )
                    
                    # For patterns with "inside", also add a variant with ADV instead of ADP
                    if any("ADP" in str(token_dict) and "inside" in str(token_dict) for token_dict in matcher_pattern):
                        adv_pattern = []
                        for token_dict in matcher_pattern:
                            new_dict = token_dict.copy()
                            if "POS" in new_dict and new_dict["POS"] == "ADP" and "inside" in str(token_dict):
                                # Create ADV variant for "inside"
                                new_dict["POS"] = "ADV"
                            adv_pattern.append(new_dict)
                        
                        if adv_pattern != matcher_pattern:
                            self.matcher.add(
                                f"{entry_id}_{len(matcher_pattern)}_adv",
                                [adv_pattern],
                                on_match=self._on_match
                            )
    
    def _on_match(self, matcher, doc, match_id, matches):
        """Callback for matcher matches."""
        pass  # Handled in __call__
    
    def _prefer_longest(self, matches: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Filter matches to prefer longest spans, avoiding overlaps."""
        if not matches:
            return []
        
        # Sort by length (longest first) and start position
        matches = sorted(matches, key=lambda x: (x[2] - x[1], -x[1]), reverse=True)
        
        filtered = []
        used_positions = set()
        
        for match_id, start, end in matches:
            # Check if this match overlaps with any existing match
            overlap = False
            for pos in range(start, end):
                if pos in used_positions:
                    overlap = True
                    break
            
            if not overlap:
                filtered.append((match_id, start, end))
                for pos in range(start, end):
                    used_positions.add(pos)
        
        return filtered
    
    def _meta_for_span(self, span: Span) -> Dict[str, Any]:
        """Extract metadata for a matched span."""
        span_text = span.text.lower()
        
        # Find matching entry in spatial maps
        for entry in self.maps.get('entries', []):
            for variant in entry.get('surface_variants', []):
                pattern = variant.get('pattern', [])
                if ' '.join(pattern).lower() == span_text:
                    return {
                        "prime_hint": entry.get('prime_hint', ''),
                        "canonical": entry.get('canonical_lemma', ''),
                        "lang": self.lang,
                        "source": "spatial_maps.json",
                        "entry_id": entry.get('id', ''),
                        "ud_constraints": entry.get('ud_constraints', {}),
                        "guards": entry.get('guards', {})
                    }
        
        return {}
    
    def _passes_guards(self, span: Span, meta: Dict[str, Any]) -> bool:
        """Apply figurative guards to prevent false spatial hits."""
        if not meta or not meta.get('guards'):
            return True
        
        guards = meta['guards']
        
        # Check forbidden lemmas on object
        if 'forbid_lemmas_on_object' in guards:
            for token in span:
                if token.lemma_.lower() in guards['forbid_lemmas_on_object']:
                    self.metrics['mwe_guard_filtered_total'] += 1
                    return False
        
        # Check NER requirements
        if 'require_object_ner_any' in guards:
            has_required_ner = False
            # Check the object (noun after the preposition)
            for token in span.doc[span.end:]:
                if token.ent_type_ in guards['require_object_ner_any']:
                    has_required_ner = True
                    break
            # If no NER requirement or no entities found, allow it (don't be too strict)
            if not has_required_ner and any(token.ent_type_ for token in span.doc):
                # Only filter if we have NER data but no matching entities
                self.metrics['mwe_guard_filtered_total'] += 1
                return False
        
        # Check POS constraints
        if 'disallow_heads_pos' in guards:
            for token in span:
                for pos_constraint in guards['disallow_heads_pos']:
                    if ':' in pos_constraint:
                        pos, tag = pos_constraint.split(':', 1)
                        if token.pos_ == pos and token.tag_ == tag:
                            self.metrics['mwe_guard_filtered_total'] += 1
                            return False
                    elif token.pos_ == pos_constraint:
                        self.metrics['mwe_guard_filtered_total'] += 1
                        return False
        
        # Check right context regex
        if 'regex_forbid_right_context' in guards:
            right_context = span.doc.text[span.end_char:span.end_char + 50]
            for pattern in guards['regex_forbid_right_context']:
                if re.search(pattern, right_context, re.IGNORECASE):
                    self.metrics['mwe_guard_filtered_total'] += 1
                    return False
        
        return True
    
    def __call__(self, doc: Doc) -> Doc:
        """
        Process document to normalize MWEs.
        
        Args:
            doc: SpaCy document
            
        Returns:
            Processed document with MWE spans and metadata
        """
        spans = []
        
        # Check if POS attributes are available
        if not any(token.pos_ for token in doc):
            # If no POS attributes, return doc without processing
            doc._.mwe_spans_by_token = {}
            return doc
        
        # 1) Match surface variants
        matches = self.matcher(doc)
        longest_matches = self._prefer_longest(matches)
        
        for match_id, start, end in longest_matches:
            span = doc[start:end]
            meta = self._meta_for_span(span)
            
            if not meta:
                continue
            
            if not self._passes_guards(span, meta):
                continue
            
            span._.mwe_meta = meta
            spans.append(span)
            self.metrics['mwe_detected_total'] += 1
        
        # 2) Index by token
        by_token = {}
        for sp in spans:
            for token in sp:
                if token.i not in by_token:
                    by_token[token.i] = []
                by_token[token.i].append(sp)
        
        doc._.mwe_spans_by_token = by_token
        
        # 3) Optional merge/retokenization
        if self.merge_adp and spans:
            with doc.retokenize() as retok:
                for sp in spans:
                    if sp._.mwe_meta:
                        attrs = {
                            "LEMMA": sp._.mwe_meta.get("canonical", sp.text),
                            "POS": "ADP"
                        }
                        retok.merge(sp, attrs=attrs)
                        self.metrics['mwe_retokenize_applied_total'] += 1
        
        return doc
    
    def get_metrics(self) -> Dict[str, int]:
        """Get current metrics."""
        return self.metrics.copy()


@Language.factory("mwe_normalizer")
def create_mwe_normalizer(nlp: Language, name: str) -> MWENormalizer:
    """Factory function for SpaCy pipeline integration."""
    # Ensure extensions are registered on this specific nlp instance
    if not Doc.has_extension("mwe_spans_by_token"):
        Doc.set_extension("mwe_spans_by_token", default={}, force=True)
    if not Span.has_extension("mwe_meta"):
        Span.set_extension("mwe_meta", default={}, force=True)
    
    # Load spatial maps for the factory
    spatial_maps = load_spatial_maps()
    
    return MWENormalizer(nlp, spatial_maps)
