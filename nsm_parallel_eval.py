#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.nsm.explicator import NSMExplicator

# Gold primitive per sentence index (reused)
def load_gold() -> dict[int, set[str]]:
    gold_path = Path('data/parallel_gold.json')
    if gold_path.exists():
        data = json.loads(gold_path.read_text(encoding='utf-8'))
        labels = data.get('labels', [])
        return {i: {labels[i]} for i in range(len(labels))}
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

LANGS = ["en", "es", "fr"]


def main():
    data_path = Path('data/parallel_test_data.json')
    if not data_path.exists():
        logger.error('Parallel data not found.')
        return

    explicator = NSMExplicator()

    with open(data_path, 'r', encoding='utf-8') as f:
        parallel = json.load(f)

    report = {"per_lang": {}, "overall": {}}

    gold = load_gold()
    for lang in LANGS:
        templates = []
        legalities = []
        for idx in range(len(parallel[lang])):
            prim = next(iter(gold.get(idx, {"HasProperty"})))
            template = explicator.template_for_primitive(prim, lang)
            score = explicator.legality_score(template, lang)
            templates.append({"idx": idx, "primitive": prim, "template": template, "legality": score})
            legalities.append(score)
        avg_legality = sum(legalities) / len(legalities) if legalities else 0.0
        report["per_lang"][lang] = {
            "avg_legality": avg_legality,
            "templates": templates,
        }
        logger.info(f"{lang.upper()} avg legality: {avg_legality:.2f}")

    report["overall"]["avg_legality"] = sum(report["per_lang"][l]["avg_legality"] for l in LANGS) / len(LANGS)

    out = Path('data/nsm_legality_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {out}")


if __name__ == '__main__':
    main()
