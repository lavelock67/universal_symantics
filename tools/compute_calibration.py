#!/usr/bin/env python3
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

from robust_aspect_mapper import (
    RobustAspectDetector,
    Language,
    compute_ece,
    compute_raw_confidence,
    compute_raw_confidence_for_type,
)
from quant_scope_normalizer import QuantifierScopeNormalizer

ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = ROOT / 'data' / 'suites'
CALIB_DIR = ROOT / 'data' / 'calibration'

random.seed(123)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_dev_eval(rows: List[Dict[str, Any]], ratio: float = 0.7) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    idx = int(len(rows) * ratio)
    return rows[:idx], rows[idx:]


def risk_coverage(scores: List[float], labels: List[int], taus: List[float]) -> List[Dict[str, Any]]:
    curve: List[Dict[str, Any]] = []
    n = len(scores)
    for t in taus:
        accepted = [i for i, s in enumerate(scores) if s >= t]
        cov = len(accepted) / n if n else 0.0
        if not accepted:
            curve.append({'tau': t, 'coverage': cov, 'risk': 0.0, 'fpr': 0.0})
            continue
        acc = sum(1 for i in accepted if labels[i] == 1) / len(accepted)
        risk = 1 - acc
        neg_idx = [i for i, y in enumerate(labels) if y == 0]
        fp = sum(1 for i in neg_idx if scores[i] >= t)
        tn = sum(1 for i in neg_idx if scores[i] < t)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        curve.append({'tau': t, 'coverage': cov, 'risk': risk, 'fpr': fpr})
    return curve


def _top_bin_over95_frac(scores: List[float]) -> float:
    if not scores:
        return 0.0
    cnt = sum(1 for s in scores if s >= 0.95)
    return cnt / len(scores)


def calib_aspect(lang: str) -> Dict[str, Any]:
    det = RobustAspectDetector()
    rows = read_jsonl(SUITES_DIR / 'aspect' / f'{lang}.jsonl')
    dev, eval_rows = split_dev_eval(rows)
    # Overall calibration
    dev_scores: List[float] = []
    dev_labels: List[int] = []
    for r in dev:
        lang_enum = Language(lang)
        d = det.detect_aspects(r['text'], lang_enum)
        y = 1 if r['label'] == 1 else 0
        s = compute_raw_confidence(r['text'], lang_enum, d)
        dev_scores.append(float(s))
        dev_labels.append(int(y))
    # Dev reliability
    ece_dev, reliability_dev = compute_ece(dev_scores, dev_labels)
    top_bin_frac_dev = _top_bin_over95_frac(dev_scores)
    # Eval
    eval_scores: List[float] = []
    eval_labels: List[int] = []
    for r in eval_rows:
        lang_enum = Language(lang)
        d = det.detect_aspects(r['text'], lang_enum)
        y = 1 if r['label'] == 1 else 0
        s = compute_raw_confidence(r['text'], lang_enum, d)
        eval_scores.append(float(s))
        eval_labels.append(int(y))
    ece, reliability = compute_ece(eval_scores, eval_labels)
    taus = [i/100 for i in range(0, 100)]
    rc = risk_coverage(eval_scores, eval_labels, taus)
    # choose tau: first with fpr < 0.01
    tau_choice = next((pt['tau'] for pt in rc if pt['fpr'] < 0.01), 1.0)
    # Per-class calibration using suite 'type' field (map to detector aspect types)
    type_map = {
        'recent': 'recent_past',
        'ongoing_for': 'ongoing_for',
        'ongoing': 'ongoing',
        'almost': 'almost_do',
        'stop': 'stop',
        'resume': 'resume',
    }
    per_class: Dict[str, Any] = {}
    # Dev per-class
    dev_by_class: Dict[str, Dict[str, List[float]]] = {}
    for r in dev:
        t = r.get('type')
        cls = type_map.get(t)
        y = 1 if r['label'] == 1 and cls is not None else 0
        lang_enum = Language(lang)
        d = det.detect_aspects(r['text'], lang_enum)
        for cls_name in ['recent_past','ongoing_for','ongoing','almost_do','stop','resume']:
            s = compute_raw_confidence_for_type(r['text'], lang_enum, d, cls_name)
            key = cls_name
            dev_by_class.setdefault(key, {}).setdefault('scores', []).append(float(s))
            # positive for its own class rows only, otherwise negative
            dev_by_class.setdefault(key, {}).setdefault('labels', []).append(1 if cls_name == cls and r['label'] == 1 else 0)
    # Eval per-class
    eval_by_class: Dict[str, Dict[str, List[float]]] = {}
    for r in eval_rows:
        t = r.get('type')
        cls = type_map.get(t)
        lang_enum = Language(lang)
        d = det.detect_aspects(r['text'], lang_enum)
        for cls_name in ['recent_past','ongoing_for','ongoing','almost_do','stop','resume']:
            s = compute_raw_confidence_for_type(r['text'], lang_enum, d, cls_name)
            eval_by_class.setdefault(cls_name, {}).setdefault('scores', []).append(float(s))
            eval_by_class.setdefault(cls_name, {}).setdefault('labels', []).append(1 if cls_name == cls and r['label'] == 1 else 0)
    for cls_name, data in eval_by_class.items():
        scores = data.get('scores', [])
        labels = data.get('labels', [])
        if not scores:
            continue
        ece_c, rel_c = compute_ece(scores, labels)
        rc_c = risk_coverage(scores, labels, taus)
        tau_c = next((pt['tau'] for pt in rc_c if pt['fpr'] < 0.01), 1.0)
        per_class[cls_name] = {'ece': ece_c, 'reliability': rel_c, 'risk_coverage': rc_c, 'tau': tau_c}
    return {
        'ece': ece,
        'reliability': reliability,
        'risk_coverage': rc,
        'tau': tau_choice,
        'per_class': per_class,
        'dev': {
            'ece': ece_dev,
            'reliability': reliability_dev,
            'top_bin_over95_frac': top_bin_frac_dev,
        },
    }


def calib_quant(lang: str) -> Dict[str, Any]:
    qn = QuantifierScopeNormalizer()
    rows = read_jsonl(SUITES_DIR / 'quantifiers' / f'{lang}.jsonl')
    dev, eval_rows = split_dev_eval(rows)
    # Use reported confidence from quantifier analyzer
    dev_scores: List[float] = []
    dev_labels: List[int] = []
    for r in dev:
        qa = qn.normalize_quantifier_scope(r['text'], Language(lang))
        dev_scores.append(float(qa.confidence or 0.0))
        dev_labels.append(int(r['label']))
    eval_scores: List[float] = []
    eval_labels: List[int] = []
    for r in eval_rows:
        qa = qn.normalize_quantifier_scope(r['text'], Language(lang))
        eval_scores.append(float(qa.confidence or 0.0))
        eval_labels.append(int(r['label']))
    ece, reliability = compute_ece(eval_scores, eval_labels)
    taus = [i/100 for i in range(0, 100)]
    rc = risk_coverage(eval_scores, eval_labels, taus)
    tau_choice = next((pt['tau'] for pt in rc if pt['fpr'] < 0.01), 1.0)
    ece_dev, reliability_dev = compute_ece(dev_scores, dev_labels)
    top_bin_frac_dev = _top_bin_over95_frac(dev_scores)
    return {
        'ece': ece,
        'reliability': reliability,
        'risk_coverage': rc,
        'tau': tau_choice,
        'dev': {
            'ece': ece_dev,
            'reliability': reliability_dev,
            'top_bin_over95_frac': top_bin_frac_dev,
        }
    }


def main():
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Any] = {'aspect': {}, 'quantifiers': {}}
    for lang in ['en','es','fr']:
        artifacts['aspect'][lang] = calib_aspect(lang)
        artifacts['quantifiers'][lang] = calib_quant(lang)
    (CALIB_DIR / 'calibration_suite_artifacts.json').write_text(json.dumps(artifacts, indent=2))
    print(f"Calibration artifacts written to {CALIB_DIR}")


if __name__ == '__main__':
    main()
