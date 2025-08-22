#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import sys
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.detect.srl_ud_detectors import (
    detect_primitives_structured,
    detect_primitives_dep,
    detect_primitives_lexical,
)

# Expected primitive per index in the parallel data arrays
def load_gold(allow_fallback: bool = False) -> dict[int, set[str]]:
    gold_path = Path('data/parallel_gold.json')
    if gold_path.exists():
        data = json.loads(gold_path.read_text(encoding='utf-8'))
        labels = data.get('labels', [])
        return {i: {labels[i]} for i in range(len(labels))}
    # If no gold is present, either fail or allow minimal fallback
    if not allow_fallback:
        logger.error("Gold mapping 'data/parallel_gold.json' not found. Rerun data prep or pass --allow-fallback to use a 10-item stub.")
        raise FileNotFoundError("Missing data/parallel_gold.json")
    # fallback to 10-sentence mapping (diagnostic only)
    logger.warning("Using minimal 10-item fallback gold map (diagnostic only).")
    return {
        0: {"AtLocation"},
        1: {"HasProperty"},
        2: {"PartOf"},
        3: {"Causes"},
        4: {"HasProperty"},
        5: {"UsedFor"},
        6: {"SimilarTo"},
        7: {"DifferentFrom"},
        8: {"Not"},
        9: {"Exist"},
    }


def main():
    parser = argparse.ArgumentParser(description='UD-based parallel primitive testing')
    parser.add_argument('--allow-fallback', action='store_true', help='Allow minimal hardcoded fallback gold map if gold file is missing')
    args = parser.parse_args()

    data_path = Path('data/parallel_test_data.json')
    if not data_path.exists():
        logger.error('Parallel data not found. Run create_parallel_test_data.py first.')
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        parallel = json.load(f)

    results = {}
    recalls = {}

    try:
        gold_map = load_gold(allow_fallback=args.allow_fallback)
    except FileNotFoundError:
        sys.exit(1)

    for lang, sentences in parallel.items():
        counts = Counter()
        examples = defaultdict(list)
        det = 0
        struct_hits = dep_hits = lex_hits = 0
        # Per-primitive tallies
        gold_counts = Counter()
        correct_counts = Counter()

        for idx, s in enumerate(sentences):
            names_struct = [d['name'] for d in detect_primitives_structured(s)]
            backend = None
            names = []
            if names_struct:
                names = names_struct
                backend = 'struct'
            else:
                names_dep = detect_primitives_dep(s)
                if names_dep:
                    names = names_dep
                    backend = 'dep'
                else:
                    names = detect_primitives_lexical(s)
                    backend = 'lex'

            if names:
                det += 1
                if backend == 'struct':
                    struct_hits += 1
                elif backend == 'dep':
                    dep_hits += 1
                else:
                    lex_hits += 1

            # Update counts and examples
            for n in set(names):
                counts[n] += 1
                if len(examples[n]) < 2:
                    examples[n].append({'text': s, 'backend': backend})

            # Per-primitive recall
            gold_set = gold_map.get(idx, set())
            for g in gold_set:
                gold_counts[g] += 1
                if g in names:
                    correct_counts[g] += 1

        total = len(sentences)
        # Compute recall per primitive over known gold keys
        recall_per_prim = {}
        for prim in sorted({p for s in gold_map.values() for p in s}):
            denom = gold_counts[prim]
            recall_per_prim[prim] = (correct_counts[prim] / denom) if denom else 0.0

        results[lang] = {
            'detection_rate': det / total if total else 0,
            'primitive_counts': dict(counts),
            'examples': dict(examples),
            'struct_sentence_hits': struct_hits,
            'dep_sentence_hits': dep_hits,
            'lex_sentence_hits': lex_hits,
            'recall_per_primitive': recall_per_prim,
        }
        recalls[lang] = recall_per_prim

        logger.info(
            f"{lang.upper()} detection_rate={results[lang]['detection_rate']:.2f} "
            f"uniques={len(counts)} struct={struct_hits} dep={dep_hits} lex={lex_hits}"
        )

    out = Path('data/ud_parallel_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {out}")

    # Print concise per-primitive recalls
    prims = sorted({p for s in gold_map.values() for p in s})
    for lang in ['en','es','fr']:
        if lang in recalls:
            line = '  '.join(f"{p}:{recalls[lang].get(p,0):.2f}" for p in prims)
            logger.info(f"{lang.upper()} recall per primitive: {line}")


if __name__ == '__main__':
    main()

# TODO: Expand ES/FR UD patterns for SimilarTo, UsedFor, AtLocation, HasProperty
# TODO: Improve FR AtLocation with verb oblique + case + det variants
# TODO: Add FR SimilarTo patterns: 'semblable à', 'pareil(le) à'
# TODO: Add FR DifferentFrom patterns: 'différent(e) de' UD v2
# TODO: Extend FR HasProperty: avoir + de + ADJ/NOUN; NUM NOUN
# TODO: Add ES SimilarTo patterns: 'parecido(a) a'
# TODO: Broaden ES UsedFor: 'servir/sirve para' variants
