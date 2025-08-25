#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from robust_aspect_mapper import Language
from eil_reasoning_integration import EILReasoningEngine

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'checkpoint'


def bilingual_pairs() -> List[Tuple[str, str, Language, Language, str, str]]:
    # (en_text, es_text, EN, ES, goal_en, goal_es) minimal slice, expanded by cycling
    seeds: List[Tuple[str, str, Language, Language, str, str]] = []
    seeds.append((
        "I have just finished.",
        "Acabo de terminar.",
        Language.EN,
        Language.ES,
        "PAST(finish)",
        "PAST(terminar)"
    ))
    seeds.append((
        "I have been working for three hours.",
        "Llevo tres horas trabajando.",
        Language.EN,
        Language.ES,
        "DURING(work, [now−PT3H, now])",
        "DURING(trabajar, [now−PT3H, now])"
    ))
    seeds.append((
        "He almost fell.",
        "Casi me caigo.",
        Language.EN,
        Language.ES,
        "¬fall ∧ near(fall)",
        "¬caer ∧ near(caer)"
    ))
    seeds.append((
        "Il est en train de travailler.",
        "He is working.",
        Language.FR,
        Language.EN,
        "ONGOING(travailler)",
        "ONGOING(work)"
    ))
    # Expand to ~200 items by cycling seeds
    out: List[Tuple[str, str, Language, Language, str, str]] = []
    while len(out) < 200:
        out += seeds[: max(1, 200 - len(out))]
    return out[:200]


def evaluate_slice(run_id: str) -> Dict[str, Any]:
    engine = EILReasoningEngine()
    pairs = bilingual_pairs()
    proofs_succ = 0
    total = 0
    prime_preserve = 0
    wsd_stable = 0
    abstain = 0
    traces: List[Dict[str, Any]] = []
    for en_text, es_text, lang1, lang2, goal1, goal2 in pairs:
        for (txt, lang, goal) in [(en_text, lang1, goal1), (es_text, lang2, goal2)]:
            res = engine.reason_with_aspects_and_quantifiers(txt, lang, goal, is_hard_goal=True)
            total += 1
            if res.success:
                proofs_succ += 1
                if res.depth >= 1:
                    prime_preserve += 1
                # proxy: families_used non-empty => synset/WSD stable enough
                if len(res.families_used) > 0:
                    wsd_stable += 1
            else:
                if res.confidence <= 0.11:
                    abstain += 1
            if len(traces) < 10:
                traces.append({
                    "text": txt,
                    "lang": lang.value,
                    "goal": goal,
                    "success": res.success,
                    "depth": res.depth,
                    "families": [f.value for f in res.families_used],
                })
    return {
        "proof_success_rate": proofs_succ / max(1, total),
        "prime_preservation_rate": prime_preserve / max(1, proofs_succ),
        "wsd_stability_rate": wsd_stable / max(1, proofs_succ),
        "abstention_rate": abstain / max(1, total),
        "traces": traces,
    }


def main():
    import os
    run_id = os.environ.get('RUN_ID', 'dev')
    out_dir = OUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_slice(run_id)
    (out_dir / 'translation_sanity.json').write_text(json.dumps(metrics, indent=2))
    # Append illustrative traces to evidence logs
    ev_path = out_dir / 'evidence_logs.json'
    try:
        data = json.loads(ev_path.read_text()) if ev_path.exists() else {"run_id": run_id, "logs": []}
    except Exception:
        data = {"run_id": run_id, "logs": []}
    if isinstance(data, dict):
        logs = data.get('logs', [])
        for tr in metrics.get('traces', [])[:10]:
            logs.append({"type": "translation_trace", "trace": tr, "run_id": run_id})
        data['logs'] = logs
        ev_path.write_text(json.dumps(data, indent=2))
    print(f"Translation sanity written to {out_dir / 'translation_sanity.json'}")


if __name__ == '__main__':
    main()


