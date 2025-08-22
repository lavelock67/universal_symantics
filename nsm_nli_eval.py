#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from typing import Dict, List
import sys
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.nsm.explicator import NSMExplicator

def load_gold(allow_fallback: bool = False) -> dict[int, set[str]]:
    data = json.loads(Path('data/parallel_gold.json').read_text(encoding='utf-8')) if Path('data/parallel_gold.json').exists() else None
    if data:
        labels = data.get('labels', [])
        return {i: {labels[i]} for i in range(len(labels))}
    if not allow_fallback:
        logger.error("Gold mapping 'data/parallel_gold.json' not found. Rerun data prep or pass --allow-fallback to use a 10-item stub.")
        raise FileNotFoundError("Missing data/parallel_gold.json")
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


def load_xnli(model_name: str = 'joeddav/xlm-roberta-large-xnli', allow_missing: bool = False):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    except Exception as e:
        if allow_missing:
            logger.warning(f"transformers unavailable ({e}); proceeding without NLI")
            return None
        logger.error("transformers is required for NLI. Install transformers or pass --allow-missing-nli.")
        raise
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok)
    return pipe


def prob_entails(pipe, premise: str, hypothesis: str) -> float:
    if pipe is None:
        return 0.0
    # XNLI labels: contradiction, neutral, entailment
    # Use top_k=None to get all scores robustly across transformers versions
    res = pipe(premise, text_pair=hypothesis, top_k=None)
    # res is a list[List[dict]] or list[dict] depending on version; normalize
    if isinstance(res, list):
        first = res[0]
        if isinstance(first, list):
            items = first
        elif isinstance(first, dict):
            items = res
        else:
            items = []
    else:
        items = []
    out = {str(x.get('label', '')).lower(): float(x.get('score', 0.0)) for x in items}
    return float(out.get('entailment', 0.0))


def main():
    parser = argparse.ArgumentParser(description='NSM NLI substitutability evaluation')
    parser.add_argument('--allow-fallback', action='store_true', help='Allow minimal hardcoded fallback gold map if gold file is missing')
    parser.add_argument('--allow-missing-nli', action='store_true', help='Allow running without transformers/XNLI; scores default to 0.0')
    parser.add_argument('--model-name', default='joeddav/xlm-roberta-large-xnli', help='HuggingFace model name for XNLI')
    args = parser.parse_args()

    data_path = Path('data/parallel_test_data.json')
    if not data_path.exists():
        logger.error('Parallel data not found.')
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        parallel = json.load(f)

    explicator = NSMExplicator()
    try:
        gold = load_gold(allow_fallback=args.allow_fallback)
    except FileNotFoundError:
        sys.exit(1)

    try:
        pipe = load_xnli(model_name=args.model_name, allow_missing=args.allow_missing_nli)
    except Exception:
        sys.exit(1)

    report: Dict[str, Dict] = {"per_lang": {}, "overall": {}}
    langs = ['en', 'es', 'fr']
    for lang in langs:
        sentences: List[str] = parallel[lang]
        entries = []
        ent_scores = []
        bi_scores = []
        for idx, sent in enumerate(sentences):
            prim = next(iter(gold.get(idx, {"HasProperty"})))
            exp = explicator.template_for_primitive(prim, lang)
            e1 = prob_entails(pipe, sent, exp)
            e2 = prob_entails(pipe, exp, sent)
            ent_scores.append(e1)
            bi_scores.append(0.5 * (e1 + e2))
            entries.append({'idx': idx, 'primitive': prim, 'source': sent, 'explication': exp, 'entails_sent_to_exp': e1, 'entails_exp_to_sent': e2, 'bi_entail': 0.5 * (e1 + e2)})
        report['per_lang'][lang] = {
            'avg_entail_sent_to_exp': sum(ent_scores) / len(ent_scores) if ent_scores else 0.0,
            'avg_bi_entail': sum(bi_scores) / len(bi_scores) if bi_scores else 0.0,
            'entries': entries,
        }
        logger.info(f"{lang.upper()} NLI bi-entailment: {report['per_lang'][lang]['avg_bi_entail']:.3f}")

    out = Path('data/nsm_nli_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {out}")


if __name__ == '__main__':
    main()


