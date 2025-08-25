#!/usr/bin/env python3
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from robust_aspect_mapper import Language
from eil_reasoning_integration import EILReasoningEngine

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'checkpoint'


def _now_run_dir() -> Path:
    run_id = (os.environ.get('RUN_ID') or f"{int(time.time())}-pilot").strip()
    d = OUT_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mk_texts(lang: Language) -> List[Tuple[str, str]]:
    texts: List[Tuple[str, str]] = []
    if lang == Language.EN:
        texts += [
            ("I have just finished.", 'recent_past'),
            ("She had just arrived.", 'recent_past'),
            ("I have been working for three hours.", 'ongoing_for'),
            ("We are studying.", 'ongoing'),
            ("I almost fell.", 'almost_do'),
            ("I stopped working.", 'stop'),
            ("She resumed running.", 'resume'),
        ]
        negatives = [
            "I have been to Paris.",
            "This is almost perfect.",
            "Stop the car!",
            "I just want tea.",
        ]
    elif lang == Language.ES:
        texts += [
            ("Acabo de salir.", 'recent_past'),
            ("Lleva tres horas estudiando.", 'ongoing_for'),
            ("Estoy trabajando.", 'ongoing'),
            ("Casi me caigo.", 'almost_do'),
            ("Dejó de fumar.", 'stop'),
            ("Volvió a intentarlo.", 'resume'),
        ]
        negatives = [
            "He estado en Madrid.",
            "Casi perfecto.",
            "Para el coche!",
            "Solo quiero ir.",
        ]
    else:
        texts += [
            ("Je viens d'arriver.", 'recent_past'),
            ("Elle travaille depuis trois heures.", 'ongoing_for'),
            ("Il est en train de travailler.", 'ongoing'),
            ("J'ai failli tomber.", 'almost_do'),
            ("Elle a cessé de venir.", 'stop'),
            ("Il a recommencé à venir.", 'resume'),
        ]
        negatives = [
            "J'ai été à Paris.",
            "Presque parfait.",
            "Arrête la voiture !",
            "Je veux juste partir.",
        ]
    # Scale up to ~1000 items by cycling
    base = list(texts)
    while len(texts) < 1000:
        texts += base[: max(1, 1000 - len(texts))]
    # Add ~500 negatives
    neg_rows: List[Tuple[str, str]] = []
    idx = 0
    while len(neg_rows) < 500:
        neg_rows.append((negatives[idx % len(negatives)], 'negative'))
        idx += 1
    texts += neg_rows
    return texts


def _goal_for(text: str, lang: Language, t: str) -> Tuple[str, bool]:
    # Construct a goal and whether it's a hard goal
    tl = text.lower()
    if t == 'recent_past':
        # crude lemma heuristic: last token up to punctuation
        lemma = 'finish'
        for tok in [w.strip(".,!?") for w in tl.split()]:
            if tok in ('finished','arrived','left','completed','started','eaten'):
                lemma = tok.rstrip('ed').rstrip('d')
        if lang == Language.ES:
            lemma = 'salir'
        if lang == Language.FR:
            lemma = 'arriver'
        return (f"PAST({lemma})", True)
    if t == 'ongoing_for':
        lemma = 'work' if lang == Language.EN else ('estudiar' if lang == Language.ES else 'travailler')
        return (f"DURING({lemma}, [now−PT3H, now])", True)
    if t == 'ongoing':
        lemma = 'study' if lang == Language.EN else ('trabajar' if lang == Language.ES else 'travailler')
        return (f"ONGOING({lemma})", False)  # not a hard goal; identity allowed
    if t == 'almost_do':
        lemma = 'fall' if lang == Language.EN else ('caer' if lang == Language.ES else 'tomber')
        return (f"¬{lemma} ∧ near({lemma})", True)
    if t == 'stop':
        lemma = 'work' if lang == Language.EN else ('fumar' if lang == Language.ES else 'venir')
        return (f"terminated({lemma})", True)
    if t == 'resume':
        lemma = 'run' if lang == Language.EN else ('intentar' if lang == Language.ES else 'venir')
        return (f"restarted({lemma})", True)
    return ("", False)


def run_pilot() -> Dict[str, Any]:
    engine = EILReasoningEngine()
    report: Dict[str, Any] = {"languages": {}, "overall": {}}
    error_clusters: Dict[str, int] = {}
    traces: List[Dict[str, Any]] = []
    for lang in [Language.EN, Language.ES, Language.FR]:
        rows = _mk_texts(lang)
        successes = 0
        depths: List[int] = []
        family_hits = 0
        hard_goals = 0
        hard_success = 0
        derived_success = 0
        hard_depth_success = 0
        hard_derived_success = 0
        total = len(rows)
        for i, (text, typ) in enumerate(rows):
            goal, is_hard = _goal_for(text, lang, typ)
            if not goal:
                continue
            res = engine.reason_with_aspects_and_quantifiers(text, lang, goal, is_hard_goal=is_hard)
            if res.success:
                successes += 1
                depths.append(res.depth)
                if res.requires_family is not None and res.requires_family in res.families_used:
                    family_hits += 1
                if is_hard:
                    hard_success += 1
                    if res.depth and res.depth >= 1:
                        hard_depth_success += 1
                        hard_derived_success += 1
                if res.depth and res.depth >= 1:
                    derived_success += 1
            else:
                key = f"fail:{typ}:{lang.value}:{'no_rules' if not res.rules_used else res.rules_used[0]}"
                error_clusters[key] = error_clusters.get(key, 0) + 1
                if len(traces) < 10:
                    traces.append({
                        "lang": lang.value,
                        "type": typ,
                        "text": text,
                        "goal": goal,
                        "rules_used": res.rules_used,
                        "families": [f.value for f in res.families_used],
                        "depth": res.depth,
                        "from_facts_only": res.from_facts_only,
                    })
            if is_hard:
                hard_goals += 1
        acc = successes / total if total else 0.0
        med_depth = statistics.median(depths) if depths else 0.0
        fam_cov = family_hits / max(1, hard_goals)
        dpr = derived_success / max(1, successes)
        depth_gt_zero = (sum(1 for d in depths if d and d > 0) / max(1, successes))
        hard_goal_success_rate = hard_success / max(1, hard_goals)
        depth_gt_zero_hard = hard_depth_success / max(1, hard_success)
        dpr_hard = hard_derived_success / max(1, hard_success)
        report["languages"][lang.value] = {
            "items": total,
            "successes": successes,
            "success_rate": acc,
            "median_depth": med_depth,
            "family_coverage_A1": fam_cov,
            "derived_proof_rate": dpr,
            "depth_gt_zero_rate": depth_gt_zero,
            "hard_goal_success_rate": hard_goal_success_rate,
            "derived_proof_rate_hard": dpr_hard,
            "depth_gt_zero_rate_hard": depth_gt_zero_hard,
        }
    # Top 5 error clusters
    top5 = sorted(error_clusters.items(), key=lambda kv: kv[1], reverse=True)[:5]
    report["overall"]["top_error_clusters"] = [{"cluster": k, "count": v} for k, v in top5]
    report["overall"]["example_traces"] = traces
    return report


def main():
    run_dir = _now_run_dir()
    rep = run_pilot()
    (run_dir / 'pilot_report.json').write_text(json.dumps(rep, indent=2))
    # Append traces to evidence logs if present
    ev_path = run_dir / 'evidence_logs.json'
    try:
        data = json.loads(ev_path.read_text()) if ev_path.exists() else {"run_id": run_dir.name, "logs": []}
    except Exception:
        data = {"run_id": run_dir.name, "logs": []}
    if isinstance(data, dict):
        logs = data.get('logs', [])
        for tr in rep.get('overall', {}).get('example_traces', []):
            logs.append({"type": "pilot_trace", "trace": tr, "run_id": run_dir.name})
        data['logs'] = logs
        ev_path.write_text(json.dumps(data, indent=2))
    print(f"Pilot report written to {run_dir / 'pilot_report.json'}")


if __name__ == '__main__':
    import os
    main()


