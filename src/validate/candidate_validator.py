"""Validate mined candidate primitives by marginal MDL gain across corpora.

For each candidate, temporarily add a heuristic primitive to the table and
measure the change in average MDL on given corpora. Keep candidates that show
consistent positive gain across corpora beyond a small threshold.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click

from ..table.schema import PeriodicTable, Primitive, PrimitiveSignature, PrimitiveCategory
from ..validate.compression import CompressionValidator
from ..table.bootstrap import augment_with_basics
from ..table.nsm_seed import augment_with_nsm_primes


logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    key: str
    head: str
    relation: str
    arg_types: List[str]
    frequency: int
    examples: List[str]


def load_candidates(path: str) -> List[Candidate]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[Candidate] = []
    for c in data:
        out.append(Candidate(
            key=c.get("key", ""),
            head=c.get("head", ""),
            relation=c.get("relation", ""),
            arg_types=c.get("arg_types", []) or [],
            frequency=int(c.get("frequency", 0)),
            examples=c.get("examples", []) or [],
        ))
    return out


def candidate_to_primitive(c: Candidate, name_prefix: str = "Cand") -> Primitive:
    # Map to reasonable category + arity
    if c.key.startswith("cop:"):
        category = PrimitiveCategory.INFORMATIONAL
        arity = 2
        name = f"{name_prefix}_IsA_like_{c.head}"
    elif c.key.startswith("prep:of"):
        category = PrimitiveCategory.STRUCTURAL
        arity = 2
        name = f"{name_prefix}_Of_{c.head}"
    elif c.key.startswith("temp:"):
        category = PrimitiveCategory.TEMPORAL
        arity = 2
        name = f"{name_prefix}_Temporal_{c.head}"
    elif c.key.startswith("loc:"):
        category = PrimitiveCategory.SPATIAL
        arity = 2
        name = f"{name_prefix}_Loc_{c.head}"
    else:
        category = PrimitiveCategory.INFORMATIONAL
        arity = max(1, len(c.arg_types) or 1)
        name = f"{name_prefix}_{c.relation}_{c.head}"
    return Primitive(name=name, category=category, signature=PrimitiveSignature(arity=arity),
                     description=f"Validated from pattern {c.key}")


def eval_avg_mdl(table: PeriodicTable, corps: List[str]) -> Dict[str, float]:
    validator = CompressionValidator(table)
    metrics: Dict[str, float] = {}
    for path in corps:
        res = validator.evaluate_text_corpus(path)
        metrics[path] = float(res["avg_mdl_score"])
    return metrics


def eval_delta_gain(table: PeriodicTable, n_pairs: int = 100) -> float:
    validator = CompressionValidator(table)
    smoke = validator.smoke_test_delta_vs_raw(n_pairs=n_pairs)
    # Simple gain: fraction better - worse, normalized
    better = smoke.get("better", 0)
    worse = smoke.get("worse", 0)
    total = smoke.get("N", 1)
    return (better - worse) / total


@click.command()
@click.option("--base-table", required=True, type=click.Path(exists=True, dir_okay=False), help="Base table JSON")
@click.option("--candidates", required=True, type=click.Path(exists=True, dir_okay=False), help="Candidates JSON from miner")
@click.option("--corpus", multiple=True, required=True, type=click.Path(exists=True, dir_okay=False), help="One or more corpora paths")
@click.option("--augment-basics", is_flag=True, help="Augment with basics before validation")
@click.option("--augment-nsm", is_flag=True, help="Augment with NSM before validation")
@click.option("--gain-threshold", default=0.005, show_default=True, help="Minimum average MDL reduction per corpus")
@click.option("--delta-threshold", default=0.01, show_default=True, help="Minimum delta_cost gain fraction")
@click.option("--delta-pairs", default=100, show_default=True, help="Number of pairs for delta stability test")
@click.option("--out-keep", default="kept_candidates.json", show_default=True, help="Output kept candidates JSON")
@click.option("--merge-into", default=None, type=click.Path(dir_okay=False), help="Optional output merged table JSON")
@click.option("--bundle-size", default=0, help="Test candidates in bundles of this size (0 = individual testing)")
@click.option("--bundle-threshold", default=0.01, help="Minimum MDL gain for bundle acceptance")
@click.option("--fast-mode", is_flag=True, help="Use fast validation with smaller corpus sample")
@click.option("--sample-size", default=1000, help="Number of lines to sample for fast validation")
@click.option("--simple-mode", is_flag=True, help="Use simple validation based on frequency and pattern quality")
@click.option("--min-frequency", default=10, help="Minimum frequency for simple mode acceptance")
def main(base_table: str, candidates: str, corpus: Tuple[str, ...], augment_basics: bool, augment_nsm: bool,
         gain_threshold: float, delta_threshold: float, delta_pairs: int, out_keep: str, merge_into: str | None,
         bundle_size: int, bundle_threshold: float, fast_mode: bool, sample_size: int, simple_mode: bool, min_frequency: int) -> None:
    """Validate and select candidate primitives by consistent MDL gains."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    base = PeriodicTable.from_dict(json.load(open(base_table, "r")))
    if augment_basics:
        augment_with_basics(base)
    if augment_nsm:
        augment_with_nsm_primes(base)

    # Load and prepare corpus
    all_texts = []
    for corpus_file in corpus:
        if Path(corpus_file).exists():
            texts = Path(corpus_file).read_text(encoding="utf-8").strip().split("\n")
            all_texts.extend(texts)
        else:
            logger.warning(f"Corpus file not found: {corpus_file}")
    
    if not all_texts:
        logger.error("No corpus texts found")
        return
    
    # Fast mode: sample a smaller corpus for faster validation
    if fast_mode:
        import random
        random.seed(42)  # For reproducible results
        if len(all_texts) > sample_size:
            all_texts = random.sample(all_texts, sample_size)
            logger.info(f"Fast mode: Using {len(all_texts)} lines from corpus (sampled from {len(all_texts) * 10})")
    
    logger.info(f"Loaded {len(all_texts)} corpus lines")

    corps = list(corpus)
    base_mdl = eval_avg_mdl(base, corps)
    base_delta = eval_delta_gain(base, n_pairs=delta_pairs)
    logger.info("Base avg MDL: %s | Base Δ gain: %s", base_mdl, base_delta)

    cands = load_candidates(candidates)
    kept: List[Dict[str, Any]] = []
    added = 0
    
    # Simple mode: accept candidates based on frequency and pattern quality
    if simple_mode:
        logger.info(f"Simple mode: Accepting candidates with frequency >= {min_frequency}")
        accepted = []
        for candidate in cands:
            if candidate.frequency >= min_frequency:
                # Additional quality checks
                key = candidate.key
                if any(relation in key for relation in ["amod", "dobj", "compound", "nsubj"]):
                    accepted.append(candidate)
        
        logger.info(f"Simple mode: Accepted {len(accepted)}/{len(cands)} candidates")
        
        # Save accepted candidates
        if out_keep:
            # Convert Candidate objects to dictionaries for JSON serialization
            accepted_dicts = []
            for candidate in accepted:
                accepted_dicts.append({
                    "key": candidate.key,
                    "frequency": candidate.frequency,
                    "relation": candidate.relation,
                    "head": candidate.head,
                    "arg_types": candidate.arg_types
                })
            Path(out_keep).write_text(json.dumps(accepted_dicts, indent=2), encoding="utf-8")
            logger.info(f"Wrote {len(accepted)} accepted candidates to {out_keep}")
        
        # Optionally merge into base table
        if merge_into and accepted:
            for candidate in accepted:
                try:
                    primitive = candidate_to_primitive(candidate)
                    if not base.get_primitive(primitive.name):
                        base.add_primitive(primitive)
                except Exception as e:
                    logger.warning(f"Failed to add candidate {candidate.key}: {e}")
            
            Path(merge_into).write_text(json.dumps(base.to_dict(), indent=2), encoding="utf-8")
            logger.info(f"Merged {len(accepted)} candidates into {merge_into}")
        
        return

    if bundle_size > 0:
        # Bundle validation: test candidates in groups
        logger.info(f"Testing candidates in bundles of size {bundle_size}")
        from itertools import combinations
        
        # Test all possible bundles of the specified size
        for bundle in combinations(cands, min(bundle_size, len(cands))):
            tmp = PeriodicTable.from_dict(base.to_dict())
            bundle_primitives = []
            
            # Add all candidates in the bundle
            for c in bundle:
                prim = candidate_to_primitive(c)
                if tmp.get_primitive(prim.name):
                    continue
                try:
                    tmp.add_primitive(prim)
                    bundle_primitives.append(prim)
                except Exception:
                    continue
            
            if not bundle_primitives:
                continue
                
            # Test the bundle
            mdl = eval_avg_mdl(tmp, corps)
            bundle_gain = sum(base_mdl[path] - mdl[path] for path in corps) / len(corps)
            
            if bundle_gain >= bundle_threshold:
                # Accept the entire bundle
                for c in bundle:
                    kept.append({
                        "key": c.key,
                        "name": candidate_to_primitive(c).name,
                        "category": candidate_to_primitive(c).category.value,
                        "arity": candidate_to_primitive(c).arity,
                        "frequency": c.frequency,
                        "examples": c.examples,
                        "bundle_gain": bundle_gain,
                        "bundle_size": len(bundle_primitives),
                    })
                    added += 1
                logger.info(f"Accepted bundle with {len(bundle_primitives)} candidates, gain: {bundle_gain:.4f}")
    else:
        # Individual validation (original logic)
        for c in cands:
            tmp = PeriodicTable.from_dict(base.to_dict())
            prim = candidate_to_primitive(c)
            if tmp.get_primitive(prim.name):
                continue
            try:
                tmp.add_primitive(prim)
            except Exception:
                continue
            mdl = eval_avg_mdl(tmp, corps)
            gains_ok = True
            for path in corps:
                gain = base_mdl[path] - mdl[path]
                if gain < gain_threshold:
                    gains_ok = False
                    break
            delta = eval_delta_gain(tmp, n_pairs=delta_pairs)
            delta_ok = (delta - base_delta) >= delta_threshold
            if gains_ok and delta_ok:
                kept.append({
                    "key": c.key,
                    "name": prim.name,
                    "category": prim.category.value,
                    "arity": prim.arity,
                    "frequency": c.frequency,
                    "examples": c.examples,
                    "gains": {p: base_mdl[p] - mdl[p] for p in corps},
                    "delta_gain": delta - base_delta,
                })
                added += 1

    Path(out_keep).write_text(json.dumps(kept, indent=2), encoding="utf-8")
    logger.info("Kept %d/%d candidates -> %s", added, len(cands), out_keep)

    if merge_into:
        merged = PeriodicTable.from_dict(base.to_dict())
        for k in kept:
            prim = Primitive(
                name=k["name"],
                category=PrimitiveCategory(k["category"]),
                signature=PrimitiveSignature(arity=int(k["arity"]))
            )
            try:
                merged.add_primitive(prim)
            except Exception:
                pass
        Path(merge_into).write_text(json.dumps(merged.to_dict(), indent=2), encoding="utf-8")
        logger.info("Merged kept candidates into %s", merge_into)


if __name__ == "__main__":
    main()


