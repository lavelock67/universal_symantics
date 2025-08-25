#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path
import hashlib
import unicodedata
import re
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BUNDLE_DIR = DATA_DIR / "checkpoint"
ASPECT_RESULTS = DATA_DIR / "hotfixed_aspect_mapper_results.json"
REASONING_RESULTS = DATA_DIR / "eil_reasoning_integration_results.json"
SUITE_EVAL_PATH = DATA_DIR / "checkpoint" / "suite_metrics.json"
CALIB_DIR = DATA_DIR / "calibration"


def git(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT)).decode().strip()
    except Exception:
        return "UNKNOWN"


def get_git_info() -> Dict[str, Any]:
    commit = git(["git", "rev-parse", "HEAD"]) or "UNKNOWN"
    branch = git(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "UNKNOWN"
    run_id = f"{int(time.time())}-{commit[:7]}"
    return {"run_id": run_id, "commit_sha": commit, "branch": branch}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def aggregate_aspect_counts(results: Dict[str, Any]) -> Dict[str, Any]:
    out = {"en": {}, "es": {}, "fr": {}}
    tests = results.get("test_cases", [])
    per_lang = {"en": [], "es": [], "fr": []}
    for i, tc in enumerate(tests):
        lang = tc.get("language", "en")
        per_lang.setdefault(lang, []).append({
            "expected": tc.get("expected_aspects", 0),
            "result": results.get("test_cases", [])[i].get("result", {}),
        })
    for lang, items in per_lang.items():
        total = len(items)
        negatives = sum(1 for it in items if it.get("expected", 0) == 0)
        positives = total - negatives
        out[lang] = {
            "items_total": total,
            "positives": positives,
            "negatives": negatives,
            "paraphrase_variants": 0,
            "counterfactual_pairs": 0,
        }
    return out


def compute_aspect_accuracy_per_lang(results: Dict[str, Any]) -> Dict[str, Any]:
    acc = {"en": {"correct": 0, "total": 0}, "es": {"correct": 0, "total": 0}, "fr": {"correct": 0, "total": 0}}
    tests = results.get("test_cases", [])
    for i, tc in enumerate(tests):
        lang = tc.get("language", "en")
        expected = tc.get("expected_aspects", 0)
        res = results.get("test_cases", [])[i].get("result", {})
        detected = len(res.get("detected_aspects", []))
        acc[lang]["total"] += 1
        if detected == expected:
            acc[lang]["correct"] += 1
    per_lang = {}
    for lang, d in acc.items():
        total = d["total"] or 1
        per_lang[lang] = {"accuracy": d["correct"] / total, "total": total, "correct": d["correct"]}
    return per_lang


def export_proof_telemetry(reasoning: Dict[str, Any]) -> None:
    proofs = reasoning.get("proof_analyses", [])
    out = {
        "proofs": proofs,
        "summary": reasoning.get("summary", {}),
        "health_metrics": reasoning.get("health_metrics", {}),
    }
    (BUNDLE_DIR / "proof_telemetry.json").write_text(json.dumps(out, indent=2))


def gather_evidence_logs(aspect: Dict[str, Any]) -> None:
    tests = aspect.get("test_cases", [])
    logs: List[Dict[str, Any]] = []
    # False negatives
    for i, tc in enumerate(tests):
        res = tests[i].get("result", {})
        expected = tc.get("expected_aspects", 0)
        detected = len(res.get("detected_aspects", []))
        if expected > 0 and detected == 0:
            ev = res.get("evidence", {})
            logs.append({"type": "false_negative", "case": tc, "evidence": ev})
            if len(logs) >= 3:
                break
    # False positives
    for i, tc in enumerate(tests):
        res = tests[i].get("result", {})
        expected = tc.get("expected_aspects", 0)
        detected = len(res.get("detected_aspects", []))
        if expected == 0 and detected > 0:
            ev = res.get("evidence", {})
            logs.append({"type": "false_positive", "case": tc, "evidence": ev})
            if sum(1 for l in logs if l["type"] == "false_positive") >= 3:
                break
    # Abstentions
    for i, tc in enumerate(tests):
        res = tests[i].get("result", {})
        expected = tc.get("expected_aspects", 0)
        detected = len(res.get("detected_aspects", []))
        ev = res.get("evidence", {})
        notes = ev.get("notes", "")
        if expected > 0 and detected == 0 and "Abstention" in notes:
            logs.append({"type": "abstention", "case": tc, "evidence": ev})
            if sum(1 for l in logs if l["type"] == "abstention") >= 2:
                break
    # Ambiguous EN quantifier cases (add two with predicate lemma placeholders)
    logs.append({
        "type": "ambiguous_quantifier",
        "case": {"input": "All children aren't playing.", "language": "en"},
        "evidence": {"predicate_lemma": "play", "note": "AMBIG WIDE/NARROW"}
    })
    logs.append({
        "type": "ambiguous_quantifier",
        "case": {"input": "Each child doesn't like vegetables.", "language": "en"},
        "evidence": {"predicate_lemma": "like", "note": "AMBIG WIDE/NARROW"}
    })
    (BUNDLE_DIR / "evidence_logs.json").write_text(json.dumps(logs, indent=2))


def _build_parity_histogram(reasoning: Dict[str, Any]) -> Dict[str, int]:
    """Build a coarse parity histogram from reasoning results.
    Uses proof_analyses and test_cases heuristics to estimate mismatch categories.
    """
    hist = {
        "lemma_mismatch": 0,
        "lang_mismatch": 0,
        "interval_format": 0,
        "interval_value": 0,
        "unicode_minus": 0,
        "clitic_not_stripped": 0,
        "non_xcomp_lemma": 0,
    }
    if not reasoning:
        return hist
    proofs = reasoning.get("proof_analyses", [])
    tests = reasoning.get("test_cases", [])
    for i, p in enumerate(proofs):
        try:
            tc = tests[i] if i < len(tests) else {}
        except Exception:
            tc = {}
        goal = (tc or {}).get("goal", "")
        desc = (tc or {}).get("description", "")
        success = bool(p.get("success", False))
        is_hard = bool(p.get("is_hard_goal", False))
        rules_used = p.get("rules_used", [])
        # Focus on DURING hard-goals where A1 fired and proof failed
        if is_hard and ("DURING(" in goal) and any(str(r).startswith("A1_") for r in rules_used) and not success:
            # unicode minus
            if "now−" in goal:
                hist["unicode_minus"] += 1
            # interval format (quotes missing or spacing)
            if '["now-' not in goal.replace(' ', ''):
                hist["interval_format"] += 1
            # lemma mismatch heuristic: goal lemma not lowercase (unexpected) or contains uppercase accents
            inner = goal[goal.find('(')+1: goal.find(')')] if '(' in goal and ')' in goal else ''
            lemma = inner.split(',')[0].strip().strip('{}').lower() if inner else ''
            if lemma and lemma != lemma.strip():
                hist["lemma_mismatch"] += 1
        # Heuristic: FR resume cases occasionally fail from non-xcomp lemma
        if "resume" in desc.lower() and not success and any(str(r).startswith("A1_") for r in rules_used):
            hist["non_xcomp_lemma"] += 1
    # Add top mismatching lemmas bucket (heuristic from test goals)
    top_lemmas: Dict[str, int] = {}
    tests = reasoning.get("test_cases", [])
    for tc in tests:
        goal = tc.get("goal", "")
        if 'DURING(' in goal:
            inner = goal[goal.find('(')+1: goal.find(')')]
            lemma = inner.split(',')[0].strip().strip('{}').lower()
            top_lemmas[lemma] = top_lemmas.get(lemma, 0) + 1
    # merge a few top into hist-like keys for quick view
    for lemma, count in sorted(top_lemmas.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        hist[f"lemma::{lemma}"] = count
    return hist


def _strip_clitics(lemma: str) -> str:
    return re.sub(r"(se|me|te|nos|os|lo|la|los|las|le|les)$", "", lemma)


def _canon_event_obj(ev_str: str) -> str:
    """Parse {lemma:"...",lang:".."} from a conclusion/goal and return JSON string sorted keys."""
    if not ev_str:
        return json.dumps({"lemma": "", "lang": ""}, sort_keys=True)
    # naive parse
    m_lemma = re.search(r"lemma:\"([^\"]+)\"", ev_str)
    m_lang = re.search(r"lang:\"([^\"]+)\"", ev_str)
    lemma = m_lemma.group(1) if m_lemma else ev_str.strip().strip('{}')
    lang = m_lang.group(1) if m_lang else ""
    lemma = unicodedata.normalize('NFC', lemma.lower())
    lemma = _strip_clitics(lemma)
    return json.dumps({"lemma": lemma, "lang": lang}, sort_keys=True)


def _canon_interval_str(intv_str: str) -> str:
    if not intv_str:
        return json.dumps(["now-PT?","now"])  # open/unknown
    s = intv_str.replace('−', '-').replace(' ', '')
    # enforce quotes around now-PT*
    if not s.startswith('['):
        s = f'[{s}]'
    try:
        # Extract two parts between brackets
        inner = s[s.find('[')+1:s.rfind(']')]
        parts = inner.split(',') if inner else []
        if len(parts) == 2:
            start = parts[0].strip('"')
            end = parts[1].strip('"')
            # Uppercase ISO unit
            if start.startswith('now-pt'):
                start = 'now-' + start[4:].upper()
            if start.startswith('now-PT'):
                # already uppercase
                pass
            return json.dumps([start, end])
    except Exception:
        pass
    return json.dumps(["now-PT?","now"])


def _compute_signature_parity(reasoning: Dict[str, Any]) -> Dict[str, Any]:
    """Compute signature parity for DURING goals where an A1 rule fired.
    Compare canonical JSON of event objects and intervals between goal and proof conclusion.
    """
    total = 0
    mismatches = 0
    results = reasoning.get('results', [])
    for r in results:
        goal = r.get('goal', '')
        if 'DURING(' not in goal:
            continue
        steps = r.get('proof_steps', [])
        # Find the DURING conclusion among steps
        conc = None
        for st in steps:
            c = st.get('conclusion', '')
            if 'DURING(' in c:
                conc = c
                break
        if not conc:
            continue
        total += 1
        # Parse goal event/interval
        g_inner = goal[goal.find('DURING(')+7 : goal.rfind(')')]
        g_ev, g_int = (g_inner.split(',', 1) + [''])[:2]
        c_inner = conc[conc.find('DURING(')+7 : conc.rfind(')')]
        c_ev, c_int = (c_inner.split(',', 1) + [''])[:2]
        g_ev_canon = _canon_event_obj(g_ev.strip())
        c_ev_canon = _canon_event_obj(c_ev.strip())
        g_int_canon = _canon_interval_str(g_int.strip())
        c_int_canon = _canon_interval_str(c_int.strip())
        if g_ev_canon != c_ev_canon or g_int_canon != c_int_canon:
            mismatches += 1
    rate = (mismatches / total) if total > 0 else 0.0
    return {"rate": rate, "passes": rate <= 0.05, "total": total, "mismatches": mismatches}


def write_run_manifest(git_info: Dict[str, Any], aspect: Dict[str, Any], target_dir: Path) -> None:
    # Prefer pilot datasets in target_dir/datasets if present; else fall back to data/suites
    pilot_ds = target_dir / 'datasets'
    qdir = ROOT / 'data' / 'suites' / 'quantifiers'
    q_counts = {"en": "missing", "es": "missing", "fr": "missing"}
    q_hashes = {"en": "missing", "es": "missing", "fr": "missing"}
    try:
        for lang in ['en','es','fr']:
            ppath = pilot_ds / f'quantifiers_{lang}.jsonl'
            if ppath.exists():
                total = sum(1 for _ in open(ppath))
                q_counts[lang] = {"items_total": total}
                q_hashes[lang] = sha256_file(ppath)
            else:
                path = qdir / f'{lang}.jsonl'
                if path.exists():
                    total = sum(1 for _ in open(path))
                    q_counts[lang] = {"items_total": total}
                    q_hashes[lang] = sha256_file(path)
    except Exception:
        pass
    # Aspect dataset counts/hashes
    adir = ROOT / 'data' / 'suites' / 'aspect'
    a_hashes = {"en": "missing", "es": "missing", "fr": "missing"}
    a_counts = {"en": {"items_total": "missing"}, "es": {"items_total": "missing"}, "fr": {"items_total": "missing"}}
    try:
        for lang in ['en','es','fr']:
            ppath = pilot_ds / f'aspect_{lang}.jsonl'
            if ppath.exists():
                total = sum(1 for _ in open(ppath))
                a_hashes[lang] = sha256_file(ppath)
                a_counts[lang] = {"items_total": total}
            else:
                apath = adir / f'{lang}.jsonl'
                if apath.exists():
                    a_hashes[lang] = sha256_file(apath)
                    total = sum(1 for _ in open(apath))
                    a_counts[lang] = {"items_total": total}
    except Exception:
        pass
    # Negatives pool hash
    neg_path = ROOT / 'data' / 'suites' / 'negatives.jsonl'
    neg_hash = sha256_file(neg_path) if neg_path.exists() else "missing"
    # Derive paraphrase/counterfactual counts if suite_metrics is present
    suite_metrics = load_json(SUITE_EVAL_PATH)
    def _suite_counts(domain: str, lang: str) -> Dict[str, Any]:
        counts = {"items_total": (q_counts if domain=="quantifiers" else a_counts).get(lang, {}).get("items_total", "missing")}
        # Try to infer negatives from SUITES files
        src = ROOT / 'data' / 'suites' / domain / f'{lang}.jsonl'
        negs = 0
        paras = 0
        cfs = 0
        if src.exists():
            try:
                with open(src) as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if int(row.get('label', 0)) == 0:
                            negs += 1
                        t = str(row.get('type', ''))
                        if t == 'paraphrase':
                            paras += 1
                        if t == 'counterfactual':
                            cfs += 1
            except Exception:
                pass
        counts.update({"negatives": negs, "paraphrase_variants": paras, "counterfactual_pairs": cfs})
        return counts
    datasets = {
        "quantifiers": {lang: _suite_counts("quantifiers", lang) for lang in ['en','es','fr']},
        "aspect": {lang: _suite_counts("aspect", lang) for lang in ['en','es','fr']}
    }
    manifest = {
        **git_info,
        "datasets": datasets,
        "models_tools": {
            "parser": "none (heuristic UD cues)",
            "wsd": "none",
            "nli": "none",
            "sbert": "none"
        },
        "tool_versions": {
            "python": git(["python3", "--version"]) or "UNKNOWN",
            "generator": "tools/generate_checkpoint_bundle.py",
            "evaluator": "tools/evaluate_suites.py",
            "calibration": "tools/compute_calibration.py"
        },
        "scales_thresholds": {
            "calibration_tau": (CALIB_DIR / 'calibration_artifacts.json').exists(),
            "abstain_policy": "prob<tau",
            "gates": {
                "ece_overall": 0.05,
                "ece_per_class": 0.08,
                "fp_negatives": 0.01
            }
        },
        "ci_image_ids": {
            "evaluator_image": "heuristic_only",
            "dataset_versions": {
                "quantifiers": q_hashes,
                "aspect": a_hashes,
                "negatives": neg_hash
            }
        }
    }
    manifest["run_id"] = git_info["run_id"]
    (target_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    git_info = get_git_info()
    # Allow freezing to an existing run directory via RUN_ID env var
    frozen = os.environ.get('RUN_ID')
    if frozen:
        git_info["run_id"] = frozen
    run_dir = BUNDLE_DIR / git_info["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    aspect = load_json(ASPECT_RESULTS)
    reasoning = load_json(REASONING_RESULTS)

    # Write run manifest (scoped to run_dir)
    write_run_manifest(git_info, aspect, run_dir)

    # Export proof telemetry with run_id
    if reasoning:
        out = {
            **reasoning,
            "run_id": git_info["run_id"],
        }
        (run_dir / "proof_telemetry.json").write_text(json.dumps(out, indent=2))

    # Evidence logs with run_id
    if aspect:
        tests = aspect.get("test_cases", [])
        logs_path = run_dir / "evidence_logs.json"
        gather_evidence_logs(aspect)
        # Move/augment gathered logs into run_dir and add run_id
        existing = load_json(BUNDLE_DIR / "evidence_logs.json")
        logs = []
        if isinstance(existing, list):
            for ev in existing:
                if isinstance(ev, dict):
                    ev["run_id"] = git_info["run_id"]
                    logs.append(ev)
        else:
            # if unexpected structure, wrap
            logs = [existing]
        # Attach canonical parity histogram for auditing
        parity_hist = _build_parity_histogram(reasoning)
        logs.append({
            "type": "signature_parity_histogram",
            "histogram": parity_hist,
            "note": "Fieldwise mismatch counts among A1 hard-goal failures",
            "run_id": git_info["run_id"]
        })
        # Add explicit EN recent-past case studies
        for txt in ["I had just finished.", "She had just left.", "They had just arrived."]:
            logs.append({
                "type": "case_study",
                "language": "en",
                "input": txt,
                "evidence": {"aspect": "RECENT_PAST", "note": "grammatical had just VERB"},
                "run_id": git_info["run_id"]
            })
        # Add FR depuis -> DURING and FR en train de -> ONGOING then DURING traces (before/after)
        logs.append({
            "type": "case_study",
            "language": "fr",
            "input": "Elle travaille depuis trois heures.",
            "evidence": {"aspect": "ONGOING_FOR", "note": "depuis + DURATION → DURING({lemma:'travailler',lang:'fr'},[now-PT3H,now])"},
            "run_id": git_info["run_id"]
        })
        logs.append({
            "type": "case_study",
            "language": "fr",
            "input": "Il est en train de travailler.",
            "evidence": {"aspect": "ONGOING", "note": "être en train de + INF → ONGOING({lemma:'travailler',lang:'fr'}) then DURING via A1"},
            "run_id": git_info["run_id"]
        })
        # ES estar + gerundio → ONGOING
        logs.append({
            "type": "case_study",
            "language": "es",
            "input": "Estoy trabajando.",
            "evidence": {"aspect": "ONGOING", "note": "estar + GERUNDIO → ONGOING({lemma:'trabajar',lang:'es'})"},
            "run_id": git_info["run_id"]
        })
        # Q1 wide-scope exemplar (bound P)
        logs.append({
            "type": "quantifier_case",
            "language": "fr",
            "input": "Tous les enfants ne jouent pas.",
            "evidence": {"scope": "WIDE(∀¬)", "predicate_lemma": "jouer", "note": "Q1_WIDE_TO_ALL_NOT with bound P"},
            "run_id": git_info["run_id"]
        })
        wrapped = {"run_id": git_info["run_id"], "logs": logs}
        logs_path.write_text(json.dumps(wrapped, indent=2))

    # Metrics snapshot: prefer suite-based evaluation if present, else fallback to hotfixed aspect report
    suite_eval = load_json(SUITE_EVAL_PATH)
    # Also read pilot report if present
    pilot_path = BUNDLE_DIR / git_info["run_id"] / 'pilot_report.json'
    pilot = load_json(pilot_path) if pilot_path.exists() else {}
    # Family coverage from reasoning health metrics
    fam_cov = {}
    if reasoning:
        fm = reasoning.get("health_metrics", {}).get("family_coverage", {})
        fam_cov = fm
    # Compute additional CI gates from reasoning proof analyses
    parity_gate = {"rate": 0.0, "passes": True}
    hard_goal_depth_gate = {"violations": 0, "passes": True}
    if reasoning:
        proofs = reasoning.get("proof_analyses", [])
        # Signature parity proxy: A1 fired but success == False on hard goals
        a1_total = 0
        a1_mismatch = 0
        depth_viol = 0
        for p in proofs:
            is_hard = p.get("is_hard_goal", False)
            rules = p.get("rules_used", [])
            success = p.get("success", False)
            depth = int(p.get("depth", 0))
            # Count A1 parity mismatches among hard goals only
            if is_hard and any(str(r).startswith("A1_") for r in rules):
                a1_total += 1
                if not success:
                    a1_mismatch += 1
            # Hard-goal depth gate: any hard-goal success with depth == 0 → violation
            if is_hard and success and depth == 0:
                depth_viol += 1
        parity_cmp = _compute_signature_parity(reasoning)
        parity_gate = {"rate": parity_cmp["rate"], "passes": parity_cmp["passes"], "total": parity_cmp["total"], "mismatches": parity_cmp["mismatches"]}
        hard_goal_depth_gate = {"violations": depth_viol, "passes": depth_viol == 0}
    # Calibration artifacts (load suite artifacts and compute top-bin gate)
    calib_suite = load_json(CALIB_DIR / 'calibration_suite_artifacts.json')
    # Compute top-bin fractions and gate
    top_bin_flags: Dict[str, Any] = {"aspect": {}, "quantifiers": {}}
    for domain in ["aspect", "quantifiers"]:
        for lang in ['en','es','fr']:
            node = calib_suite.get(domain, {}).get(lang, {})
            dev = node.get('dev', {})
            frac = dev.get('top_bin_over95_frac', 0.0)
            top_bin_flags[domain][lang] = {
                'top_bin_over95_frac': frac,
                'passes_top_bin_gate': frac <= 0.20
            }
    # Copy/surface calibration PNGs into run_dir/calibration and list them
    cal_run_dir = run_dir / 'calibration'
    cal_run_dir.mkdir(parents=True, exist_ok=True)
    reliability_pngs = []
    risk_pngs = []
    for p in CALIB_DIR.glob("reliability_*.png"):
        try:
            dest = cal_run_dir / p.name
            if not dest.exists():
                dest.write_bytes(p.read_bytes())
            reliability_pngs.append(str(dest))
        except Exception:
            pass
    for p in CALIB_DIR.glob("risk_coverage_*.png"):
        try:
            dest = cal_run_dir / p.name
            if not dest.exists():
                dest.write_bytes(p.read_bytes())
            risk_pngs.append(str(dest))
        except Exception:
            pass
    calib = {
        "artifacts": calib_suite,
        "top_bin_gate": top_bin_flags,
        "plots": {
            "reliability_pngs": reliability_pngs,
            "risk_coverage_pngs": risk_pngs
        }
    }
    # Build confusion matrices from pilot suite metrics if available
    pilot_conf: Dict[str, Any] = {}
    try:
        pilot_suite = load_json(BUNDLE_DIR / git_info["run_id"] / 'suite_metrics.json')
        def _extract_conf(domain: str) -> Dict[str, Any]:
            node = pilot_suite.get(domain, {}) if isinstance(pilot_suite, dict) else {}
            out: Dict[str, Any] = {}
            for lang in ['en','es','fr']:
                d = node.get(lang, {})
                out[lang] = {k: int(d.get(k, 0) or 0) for k in ['tp','fp','fn','tn']}
            return out
        pilot_conf = {
            'quantifiers': _extract_conf('quantifiers'),
            'aspect': _extract_conf('aspect')
        }
    except Exception:
        pilot_conf = {}
    # NSM primes coverage
    primes_cov_path = BUNDLE_DIR / git_info["run_id"] / 'nsm_primes_coverage.json'
    primes_cov = load_json(primes_cov_path) if primes_cov_path.exists() else {}
    if suite_eval:
        snapshot = {
            "run_id": git_info["run_id"],
            "suite_eval": suite_eval,
            "family_coverage": fam_cov,
            "artifacts": calib,
            "gates": {
                "signature_parity": parity_gate,
                "hard_goal_depth": hard_goal_depth_gate,
            },
            "confusion_matrices": pilot_conf,
            "nsm_primes_coverage": primes_cov
        }
    else:
        per_lang_acc = compute_aspect_accuracy_per_lang(aspect) if aspect else {}
        snapshot = {
            "run_id": git_info["run_id"],
            "quantifiers": {"status": "partial", "family_coverage": fam_cov.get("quantifier", "missing")},
            "aspect": {
                "per_language_accuracy": per_lang_acc,
                "calibration": {
                    "overall_ece": aspect.get("ece_eval", "missing"),
                    "brier": aspect.get("brier_eval", "missing"),
                    "per_class": aspect.get("aspect_metrics", {})
                },
                "artifacts": calib
            },
            "gates": {
                "signature_parity": parity_gate,
                "hard_goal_depth": hard_goal_depth_gate,
            },
            "confusion_matrices": pilot_conf,
            "nsm_primes_coverage": primes_cov
        }
    # Freeze schema keys for snapshot and manifest
    SNAPSHOT_KEYS = {"run_id","suite_eval","family_coverage","artifacts","gates","quantifiers","aspect"}
    MANIFEST_KEYS = {"run_id","datasets","models_tools","tool_versions","scales_thresholds","ci_image_ids","commit_sha","branch"}
    (run_dir / "evaluation_snapshot.json").write_text(json.dumps(snapshot, indent=2))

    # CI validation: ensure all artifacts share the same run_id and required-family compliance
    problems: List[str] = []
    for fname in ["run_manifest.json", "evaluation_snapshot.json", "proof_telemetry.json", "evidence_logs.json"]:
        path = run_dir / fname
        if not path.exists():
            problems.append(f"missing:{fname}")
            continue
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                if data.get("run_id") != git_info["run_id"]:
                    problems.append(f"run_id_mismatch:{fname}")
                # Schema/key drift checks
                if fname == "run_manifest.json":
                    missing = [k for k in MANIFEST_KEYS if k not in data]
                    if missing:
                        problems.append(f"manifest_missing_keys:{','.join(missing)}")
                    # Ensure counts present for each domain/lang
                    ds = data.get("datasets", {})
                    for domain in ["quantifiers","aspect"]:
                        node = ds.get(domain, {})
                        for lang in ['en','es','fr']:
                            if not isinstance(node.get(lang, {}).get("items_total"), int):
                                problems.append(f"counts_missing:{domain}:{lang}")
                if fname == "evaluation_snapshot.json":
                    # Ensure expected keys exist (allow either suite_eval branch or fallback keys)
                    if not any(k in data for k in ["suite_eval","quantifiers","aspect"]):
                        problems.append("snapshot_missing_core_sections")
            else:
                problems.append(f"invalid_json:{fname}")
        except Exception:
            problems.append(f"invalid_json:{fname}")
    # Pilot-based CI gates
    # Quantifier and Aspect per-language thresholds from pilot suite_metrics (source of truth)
    pilot_suite_path = run_dir / 'suite_metrics.json'
    pilot_suite = load_json(pilot_suite_path)
    def _gate_quantifiers_ok() -> bool:
        ok = True
        q = pilot_suite.get('quantifiers', {}) if isinstance(pilot_suite, dict) else {}
        for lang in ['en','es','fr']:
            node = q.get(lang, {})
            # accuracy may be 0.0; treat presence explicitly
            acc = float(node.get('accuracy')) if 'accuracy' in node else 0.0
            # fp_rate_negatives can be 0.0 (valid); don't coerce via `or`
            fpr = float(node.get('fp_rate_negatives')) if 'fp_rate_negatives' in node else 1.0
            fam = float(node.get('family_coverage_Q1')) if 'family_coverage_Q1' in node else 0.0
            if not (acc >= 0.90 and fpr < 0.01 and fam >= 0.80):
                ok = False
        return ok
    def _gate_aspect_ok() -> bool:
        ok = True
        a = pilot_suite.get('aspect', {}) if isinstance(pilot_suite, dict) else {}
        for lang in ['en','es','fr']:
            node = a.get(lang, {})
            acc = float(node.get('accuracy')) if 'accuracy' in node else 0.0
            fpr = float(node.get('fp_rate_negatives')) if 'fp_rate_negatives' in node else 1.0
            fam = float(node.get('family_coverage_A1')) if 'family_coverage_A1' in node else 0.0
            if not (acc >= 0.90 and fpr < 0.01 and fam >= 0.80):
                ok = False
        return ok
    def _gate_reasoning_ok() -> bool:
        if not isinstance(pilot, dict):
            return False
        langs = pilot.get('languages', {})
        # Aggregate across languages (micro-averages)
        dprs = []
        depths = []
        hards = []
        for lang in ['en','es','fr']:
            node = langs.get(lang, {})
            dprs.append(float(node.get('derived_proof_rate', 0.0) or 0.0))
            depths.append(float(node.get('depth_gt_zero_rate', 0.0) or 0.0))
            hards.append(float(node.get('hard_goal_success_rate', 0.0) or 0.0))
        def avg(xs):
            return sum(xs)/max(1,len(xs))
        return (avg(dprs) >= 0.70 and avg(depths) >= 0.80 and avg(hards) >= 0.70)
    # Correlation gate: compare demo (micro) vs pilot averages (prefer hard-only pilot), allow ≤12% diff
    corr_viol = False
    try:
        demo_hm = reasoning.get('health_metrics', {})
        langs = pilot.get('languages', {})
        def avg_lang(key):
            vals = [float(langs.get(l,{}).get(key,0.0) or 0.0) for l in ['en','es','fr']]
            return sum(vals)/max(1,len(vals))
        dpr_avg = avg_lang('derived_proof_rate_hard') if any('derived_proof_rate_hard' in langs.get(l, {}) for l in ['en','es','fr']) else avg_lang('derived_proof_rate')
        depth_avg = avg_lang('depth_gt_zero_rate_hard') if any('depth_gt_zero_rate_hard' in langs.get(l, {}) for l in ['en','es','fr']) else avg_lang('depth_gt_zero_rate')
        hard_avg = avg_lang('hard_goal_success_rate')
        tol = 0.12
        for key, pilot_avg in [
            ('derived_proof_rate', dpr_avg),
            ('depth_gt_zero_rate', depth_avg),
            ('hard_goal_success_rate', hard_avg),
        ]:
            demo_val = float(demo_hm.get(key, 0.0) or 0.0)
            if abs(demo_val - pilot_avg) > tol:
                corr_viol = True
                break
    except Exception:
        pass
    # CI sentinel: calibration plots must exist when data present
    calib_plot_missing = []
    if isinstance(calib_suite, dict):
        # If any domain/lang has dev data, require at least one reliability plot
        if not reliability_pngs:
            calib_plot_missing.append('reliability_plots')
    if calib_plot_missing:
        problems.append('calibration_plots_missing')
    # Required family compliance sentinel – fail if below threshold
    try:
        reqfam_rate = reasoning.get("health_metrics", {}).get("required_family_compliance", 1.0)
    except Exception:
        reqfam_rate = 1.0
    if reqfam_rate < 0.8:
        problems.append("required_family_compliance_below_threshold")
    # Apply pilot gates
    if not _gate_quantifiers_ok():
        problems.append('pilot_gate_quantifiers_failed')
    if not _gate_aspect_ok():
        problems.append('pilot_gate_aspect_failed')
    if not _gate_reasoning_ok():
        problems.append('pilot_gate_reasoning_failed')
    if corr_viol:
        problems.append('correlation_gate_failed')
    # Data sanity gate: fail if >1% fail rate in any domain/lang
    data_sanity_path = run_dir / 'data_sanity.json'
    data_sanity = load_json(data_sanity_path)
    if isinstance(data_sanity, dict):
        doms = data_sanity.get('domains', {})
        for domain in ['quantifiers','aspect']:
            node = doms.get(domain, {})
            for lang in ['en','es','fr']:
                rate = float(node.get(lang, {}).get('fail_rate', 0.0) or 0.0)
                if rate > 0.01:
                    problems.append(f'data_sanity_failed:{domain}:{lang}:{rate:.3f}')
    # NSM primes coverage gate (non-strict for now if file missing)
    primes_cov = load_json(run_dir / 'nsm_primes_coverage.json')
    if isinstance(primes_cov, dict):
        covered = int(primes_cov.get('covered_count', 0) or 0)
        target_total = int(primes_cov.get('target_total', 65) or 65)
        if covered < target_total:
            problems.append(f"nsm_primes_missing:{target_total-covered}")
    # Final decision
    if problems:
        print("CI_VALIDATION: FAIL - " + ", ".join(problems))
    else:
        print(f"Checkpoint bundle written to {run_dir}")


if __name__ == "__main__":
    main()
