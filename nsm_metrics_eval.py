#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.nsm.explicator import NSMExplicator

GOLD_BY_INDEX = {
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


def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        # multilingual for cross-lingual similarity
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        logger.warning(f"SBERT unavailable: {e}")
        return None


def cosine(a, b):
    import numpy as np
    na = a / (np.linalg.norm(a) + 1e-12)
    nb = b / (np.linalg.norm(b) + 1e-12)
    return float((na * nb).sum())


def main():
    # Try the expanded dataset first, fall back to original
    data_path = Path('data/parallel_test_data_1k.json')
    if not data_path.exists():
        data_path = Path('data/parallel_test_data.json')
        if not data_path.exists():
            logger.error('Parallel data not found.')
            return

    with open(data_path, 'r', encoding='utf-8') as f:
        parallel = json.load(f)

    explicator = NSMExplicator()
    sbert = load_sbert()

    report: Dict[str, Dict] = {"per_lang": {}, "overall": {}}

    langs = ['en', 'es', 'fr']
    for lang in langs:
        sentences: List[str] = parallel[lang]
        entries = []
        subs_scores = []
        legal_scores = []
        for idx, sent in enumerate(sentences):
            prim = next(iter(GOLD_BY_INDEX.get(idx, {"HasProperty"})))
            exp = explicator.template_for_primitive(prim, lang)
            # legality
            leg = explicator.legality_score(exp, lang)
            legal_scores.append(leg)
            # substitutability via cosine sim in multilingual embedding space
            if sbert is not None:
                emb = sbert.encode([sent, exp])
                sim = cosine(emb[0], emb[1])
            else:
                sim = 0.0
            subs_scores.append(sim)
            entries.append({
                'idx': idx,
                'primitive': prim,
                'source': sent,
                'explication': exp,
                'legality': leg,
                'substitutability': sim,
            })

        avg_leg = sum(legal_scores) / len(legal_scores) if legal_scores else 0.0
        avg_subs = sum(subs_scores) / len(subs_scores) if subs_scores else 0.0
        report['per_lang'][lang] = {
            'avg_legality': avg_leg,
            'avg_substitutability': avg_subs,
            'entries': entries,
        }
        logger.info(f"{lang.upper()} legality={avg_leg:.2f} substitutability={avg_subs:.2f}")

    # Cross-translatability: same primitive explications are available across langs with reasonable legality
    cross_ok = 0
    total_idxs = len(parallel['en'])
    for idx in range(total_idxs):
        prim = next(iter(GOLD_BY_INDEX.get(idx, {"HasProperty"})))
        legal_ok = 0
        for lang in langs:
            exp = explicator.template_for_primitive(prim, lang)
            if explicator.legality_score(exp, lang) >= 0.5:
                legal_ok += 1
        if legal_ok == len(langs):
            cross_ok += 1
    cross_translatability = cross_ok / total_idxs if total_idxs else 0.0
    report['overall'] = {
        'cross_translatability': cross_translatability
    }

    out = Path('data/nsm_metrics_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {out}")


if __name__ == '__main__':
    main()



