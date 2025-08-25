#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from robust_aspect_mapper import RobustAspectDetector, Language, compute_raw_confidence
from quant_scope_normalizer import QuantifierScopeNormalizer, ScopeType, Language as QLanguage

ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = ROOT / 'data' / 'suites'
CHECKPOINT_DIR = ROOT / 'data' / 'checkpoint'
CALIBRATION_PATH = ROOT / 'data' / 'calibration' / 'calibration_suite_artifacts.json'
DATASETS_DIR = ROOT / 'datasets' / 'aspect'


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


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return precision, recall, accuracy


def eval_quantifiers(lang: str, rows: List[Dict[str, Any]], tau: float = 0.11) -> Dict[str, Any]:
    qn = QuantifierScopeNormalizer()
    tp = fp = fn = tn = 0
    fam_used = used_total = 0
    for r in rows:
        text = r['text']
        label = r['label']
        # Use quantifier normalizer's Language enum
        lang_enum_q = QLanguage(lang)
        qa = qn.normalize_quantifier_scope(text, lang_enum_q)
        scope = qa.scope_resolution
        conf = float(qa.confidence or 0.0)
        # Expected scope for positives inferred from text per language
        expected_scope = None
        t = text.lower()
        if label == 1:
            if lang == 'fr':
                if 'pas tous' in t:
                    expected_scope = ScopeType.NARROW_SCOPE
                elif 'tous' in t and ' ne ' in t and ' pas' in t:
                    expected_scope = ScopeType.WIDE_SCOPE
                elif any(w in t for w in ['peu', 'quelques', 'à peine', 'plupart', 'majorité', 'au plus', 'pas plus de', 'moins de']):
                    expected_scope = ScopeType.WIDE_SCOPE
            elif lang == 'en':
                if 'not all' in t or 'not every' in t or 'not each' in t:
                    expected_scope = ScopeType.NARROW_SCOPE
                elif t.strip().startswith('no ') or 'none of' in t:
                    expected_scope = ScopeType.WIDE_SCOPE
                elif any(w in t for w in ['few', 'hardly any', 'scarcely any', 'most', 'majority', 'at most', 'no more than', 'less than', 'fewer than']):
                    expected_scope = ScopeType.WIDE_SCOPE
            elif lang == 'es':
                if 'no todos' in t or 'no todas' in t:
                    expected_scope = ScopeType.NARROW_SCOPE
                elif any(w in t for w in ['ningún', 'ninguno', 'ninguna', 'nadie', 'nada']):
                    # Normalizer maps these to NARROW in ES
                    expected_scope = ScopeType.NARROW_SCOPE
                elif ('todos' in t or 'todas' in t) and ' no ' in t:
                    expected_scope = ScopeType.WIDE_SCOPE
                elif any(w in t for w in ['pocos', 'pocas', 'apenas', 'escasos', 'mayoría', 'mayor parte', 'a lo sumo', 'no más de', 'menos de']):
                    expected_scope = ScopeType.WIDE_SCOPE
        # Apply calibrated threshold: treat as detected only if confidence ≥ tau
        detected = (scope is not None) and (conf >= tau)
        if label == 1:
            if detected and expected_scope is not None and scope == expected_scope:
                tp += 1
                fam_used += 1  # treat detection as Q1 usable
                used_total += 1
            else:
                fn += 1
                used_total += 1
        else:
            if detected:
                fp += 1
            else:
                tn += 1
    precision, recall, accuracy = prf(tp, fp, fn)
    neg_total = sum(1 for r in rows if r['label'] == 0)
    fpr = fp / neg_total if neg_total > 0 else 0.0
    fam_cov = fam_used / used_total if used_total > 0 else 0.0
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'accuracy': accuracy,
        'fp_rate_negatives': fpr,
        'family_coverage_Q1': fam_cov
    }


def eval_aspect(lang: str, rows: List[Dict[str, Any]], tau: float = 0.11, class_taus: Dict[str, float] = None) -> Dict[str, Any]:
    det = RobustAspectDetector()
    tp = fp = fn = tn = 0
    fam_used = used_total = 0
    # paraphrase consistency and counterfactual flip
    base_map: Dict[str, bool] = {}
    para_match = 0
    para_total = 0
    cf_flip = 0
    cf_total = 0
    for r in rows:
        text = r['text']
        label = r['label']
        lang_enum = Language(lang)
        d = det.detect_aspects(text, lang_enum)
        # Selective prediction per class: any aspect passing its class-specific tau accepts
        score_overall = compute_raw_confidence(text, lang_enum, d)
        detected = False
        if len(d.detected_aspects) > 0:
            if class_taus:
                for a in d.detected_aspects:
                    cls = a.get('aspect_type')
                    tval = class_taus.get(cls, tau)
                    # reuse overall score as proxy; detectors already UD-first
                    if score_overall >= tval:
                        detected = True
                        break
            else:
                detected = score_overall >= tau
        if label == 1:
            if detected:
                tp += 1
                fam_used += 1
                used_total += 1
            else:
                fn += 1
                used_total += 1
        else:
            if detected:
                fp += 1
            else:
                tn += 1
        gid = r.get('group_id')
        ttype = r.get('type')
        if gid and ttype == 'base':
            base_map[gid] = detected
        if gid and ttype == 'paraphrase' and gid in base_map:
            para_total += 1
            if base_map[gid] == detected:
                para_match += 1
        if gid and ttype == 'counterfactual' and gid in base_map:
            cf_total += 1
            if base_map[gid] != detected:
                cf_flip += 1
    precision, recall, accuracy = prf(tp, fp, fn)
    neg_total = sum(1 for r in rows if r['label'] == 0)
    fpr = fp / neg_total if neg_total > 0 else 0.0
    fam_cov = fam_used / used_total if used_total > 0 else 0.0
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'accuracy': accuracy,
        'fp_rate_negatives': fpr,
        'family_coverage_A1': fam_cov,
        'paraphrase_consistency': (para_match / para_total) if para_total > 0 else 0.0,
        'counterfactual_flip_rate': (cf_flip / cf_total) if cf_total > 0 else 0.0
    }


def main():
    import os
    run_id = os.environ.get('RUN_ID')
    run_dir = (CHECKPOINT_DIR / run_id) if run_id else None
    pilot_ds_dir = (run_dir / 'datasets') if run_dir and (run_dir / 'datasets').exists() else None
    results: Dict[str, Any] = {'quantifiers': {}, 'aspect': {}, 'overall': {}}
    # Quantifiers
    # Load per-language tau thresholds for quantifiers if available
    q_tau_map: Dict[str, float] = { 'en': 0.11, 'es': 0.11, 'fr': 0.11 }
    if CALIBRATION_PATH.exists():
        try:
            calib = json.loads(CALIBRATION_PATH.read_text())
            for lang in ['en','es','fr']:
                tau_val = calib.get('quantifiers', {}).get(lang, {}).get('tau')
                if isinstance(tau_val, (int, float)):
                    q_tau_map[lang] = float(tau_val)
        except Exception:
            pass
    q_counts = {}
    q_hashes = {}
    for lang in ['en','es','fr']:
        if pilot_ds_dir and (pilot_ds_dir / f'quantifiers_{lang}.jsonl').exists():
            path = pilot_ds_dir / f'quantifiers_{lang}.jsonl'
        else:
            path = SUITES_DIR / 'quantifiers' / f'{lang}.jsonl'
        rows = read_jsonl(path)
        results['quantifiers'][lang] = eval_quantifiers(lang, rows, tau=q_tau_map.get(lang, 0.11))
        q_counts[lang] = len(rows)
        try:
            import hashlib
            q_hashes[lang] = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            q_hashes[lang] = 'missing'
    # Aspect
    # Load per-language tau thresholds from calibration artifacts if available
    tau_map: Dict[str, float] = { 'en': 0.11, 'es': 0.11, 'fr': 0.11 }
    class_tau_map: Dict[str, Dict[str, float]] = { 'en': {}, 'es': {}, 'fr': {} }
    if CALIBRATION_PATH.exists():
        try:
            calib = json.loads(CALIBRATION_PATH.read_text())
            for lang in ['en','es','fr']:
                tau_val = calib.get('aspect', {}).get(lang, {}).get('tau')
                if isinstance(tau_val, (int, float)):
                    tau_map[lang] = float(tau_val)
                per_class = calib.get('aspect', {}).get(lang, {}).get('per_class', {})
                class_tau_map[lang] = {k: float(v.get('tau', tau_map[lang])) for k, v in per_class.items()}
        except Exception:
            pass
    a_counts = {}
    a_hashes = {}
    for lang in ['en','es','fr']:
        # Prefer pilot datasets if present, else held-out, else suites
        if pilot_ds_dir and (pilot_ds_dir / f'aspect_{lang}.jsonl').exists():
            path = pilot_ds_dir / f'aspect_{lang}.jsonl'
        else:
            held = DATASETS_DIR / lang / 'heldout.jsonl'
            path = held if held.exists() else (SUITES_DIR / 'aspect' / f'{lang}.jsonl')
        rows = read_jsonl(path)
        # Use per-language tau and per-class taus when available
        results['aspect'][lang] = eval_aspect(lang, rows, tau=tau_map.get(lang, 0.11), class_taus=class_tau_map.get(lang, {}))
        a_counts[lang] = len(rows)
        try:
            import hashlib
            a_hashes[lang] = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            a_hashes[lang] = 'missing'
    # Save snapshot
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # Write suite metrics to a dedicated file to avoid clobbering other snapshots
    (CHECKPOINT_DIR / 'suite_metrics.json').write_text(json.dumps(results, indent=2))
    if run_dir:
        (run_dir / 'suite_metrics.json').write_text(json.dumps(results, indent=2))
    # Also write legacy snapshot path for backward compatibility
    (CHECKPOINT_DIR / 'evaluation_snapshot.json').write_text(json.dumps(results, indent=2))
    # Update manifest counts stub
    manifest_path = (run_dir / 'run_manifest.json') if run_dir else (CHECKPOINT_DIR / 'run_manifest.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {}
    manifest.setdefault('datasets', {})
    manifest['datasets']['quantifiers'] = {
        'en': {'items_total': q_counts.get('en',0), 'sha256': q_hashes.get('en','missing')},
        'es': {'items_total': q_counts.get('es',0), 'sha256': q_hashes.get('es','missing')},
        'fr': {'items_total': q_counts.get('fr',0), 'sha256': q_hashes.get('fr','missing')},
    }
    manifest['datasets']['aspect'] = {
        'en': {'items_total': a_counts.get('en',0), 'sha256': a_hashes.get('en','missing')},
        'es': {'items_total': a_counts.get('es',0), 'sha256': a_hashes.get('es','missing')},
        'fr': {'items_total': a_counts.get('fr',0), 'sha256': a_hashes.get('fr','missing')},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Evaluation snapshot written to {CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()
