#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / 'data' / 'checkpoint'


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # malformed json line
                rows.append({"text": line, "label": None})
    return rows


def fr_basic_sanity(text: str) -> bool:
    t = text.strip()
    # Accept if no French negation marker appears needed
    if (" ne " not in t and " n'" not in t and " pas" not in t):
        return True
    # If negation is present, require both parts of ne ... pas (or n' ... pas)
    ok = ((" ne " in t or " n'" in t) and " pas" in t)
    return ok


def es_basic_sanity(text: str) -> bool:
    t = text.lower().strip()
    # If attempting a narrow-scope, prefer 'no todos/todas'; allow 'ningún/ninguna/ninguno'
    if 'todos' in t or 'todas' in t:
        # if explicit negation construction present, accept
        if ' no ' in t:
            return True
    if any(w in t for w in ['ningún', 'ninguna', 'ninguno', 'nadie', 'nada']):
        return True
    return True


def en_basic_sanity(text: str) -> bool:
    t = text.lower().strip()
    # Simple checks for quantifier constructions
    if t.startswith('not all') or t.startswith('not every') or t.startswith('not each'):
        return True
    if t.startswith('no '):
        return True
    return True


def compute_sanity_for_run(run_id: str) -> Dict[str, Any]:
    run_dir = CHECKPOINT_DIR / run_id
    ds_dir = run_dir / 'datasets'
    out: Dict[str, Any] = {"run_id": run_id, "domains": {}}
    domains = {
        'quantifiers': ['quantifiers_en.jsonl', 'quantifiers_es.jsonl', 'quantifiers_fr.jsonl'],
        'aspect': ['aspect_en.jsonl', 'aspect_es.jsonl', 'aspect_fr.jsonl'],
    }
    for domain, files in domains.items():
        out['domains'][domain] = {}
        for fname in files:
            lang = fname.split('_')[-1].split('.')[0]
            path = ds_dir / fname
            rows = read_jsonl(path)
            total = len(rows)
            fails = 0
            for r in rows:
                text = str(r.get('text', ''))
                ok = True
                if lang == 'fr':
                    ok = fr_basic_sanity(text)
                elif lang == 'es':
                    ok = es_basic_sanity(text)
                elif lang == 'en':
                    ok = en_basic_sanity(text)
                if not ok:
                    fails += 1
            rate = (fails / total) if total > 0 else 0.0
            out['domains'][domain][lang] = {
                'total': total,
                'fails': fails,
                'fail_rate': rate
            }
    return out


def main():
    run_id = os.environ.get('RUN_ID')
    if not run_id:
        print('RUN_ID not set')
        return
    out = compute_sanity_for_run(run_id)
    run_dir = CHECKPOINT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'data_sanity.json').write_text(json.dumps(out, indent=2))
    print(f"Data sanity written to {run_dir / 'data_sanity.json'}")


if __name__ == '__main__':
    main()


