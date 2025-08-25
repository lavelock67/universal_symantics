#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
CALIB_DIR = DATA_DIR / 'calibration'
ASPECT_RESULTS = DATA_DIR / 'hotfixed_aspect_mapper_results.json'
CALIB_JSON = CALIB_DIR / 'calibration_artifacts.json'
CALIB_SUITE_JSON = CALIB_DIR / 'calibration_suite_artifacts.json'


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def plot_reliability(reliability: List[Dict[str, Any]], title: str, out_path: Path) -> None:
    bins = [r['bin'] for r in reliability]
    acc = [r['avg_acc'] for r in reliability]
    conf = [r['avg_conf'] for r in reliability]
    width = 0.4
    x = [b for b in bins]
    plt.figure(figsize=(8,5))
    plt.bar([xi - width/2 for xi in x], conf, width=width, alpha=0.6, label='Avg Confidence')
    plt.bar([xi + width/2 for xi in x], acc, width=width, alpha=0.6, label='Avg Accuracy')
    plt.plot([min(x)-1, max(x)+1], [0,1], '--', color='gray', linewidth=1)
    plt.title(title)
    plt.xlabel('Bin')
    plt.ylabel('Value')
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_risk_coverage(rc: List[Dict[str, Any]], title: str, out_path: Path) -> None:
    if not rc:
        return
    taus = [p['tau'] for p in rc]
    cov = [p['coverage'] for p in rc]
    risk = [p['risk'] for p in rc]
    fpr = [p['fpr'] for p in rc]
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(taus, cov, label='Coverage', color='tab:blue')
    ax1.plot(taus, risk, label='Risk (1-acc on accepted)', color='tab:orange')
    ax1.set_xlabel('Threshold τ')
    ax1.set_ylabel('Coverage/Risk')
    ax1.set_ylim(0,1)
    ax2 = ax1.twinx()
    ax2.plot(taus, fpr, label='FPR', color='tab:red')
    ax2.set_ylabel('FPR')
    ax2.set_ylim(0,1)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    aspect = load_json(ASPECT_RESULTS)
    calib = load_json(CALIB_JSON)
    suite = load_json(CALIB_SUITE_JSON)
    # Overall reliability
    rel_eval = aspect.get('reliability_eval', [])
    if rel_eval:
        plot_reliability(rel_eval, 'Reliability (Overall, hold-out)', CALIB_DIR / 'reliability_overall.png')
    # Per-class reliability if present
    aspect_metrics = aspect.get('aspect_metrics', {})
    for key_apl, data in aspect_metrics.items():
        rel = data.get('reliability')
        if rel:
            out_png = CALIB_DIR / f'reliability_{key_apl.replace(":","_")}.png'
            plot_reliability(rel, f'Reliability ({key_apl})', out_png)
    # Risk-coverage (overall fallback)
    rc = aspect.get('risk_coverage', []) or calib.get('risk_coverage', [])
    if rc:
        plot_risk_coverage(rc, 'Risk-Coverage Curve', CALIB_DIR / 'risk_coverage.png')
    # Per-class/per-lang risk-coverage from suite artifacts
    for domain in ('aspect', 'quantifiers'):
        node = suite.get(domain, {}) if isinstance(suite, dict) else {}
        for lang, data in node.items():
            # Per-class
            per_class = data.get('per_class', {}) if domain == 'aspect' else {}
            if per_class:
                for cls, cdata in per_class.items():
                    rc_cls = cdata.get('risk_coverage', [])
                    if rc_cls:
                        out_png = CALIB_DIR / f'risk_coverage_{cls}_{lang}.png'
                        title = f'Risk-Coverage ({domain}:{cls}:{lang})'
                        plot_risk_coverage(rc_cls, title, out_png)
            # Also per-lang overall
            rc_lang = data.get('risk_coverage', [])
            if rc_lang:
                out_png = CALIB_DIR / f'risk_coverage_{domain}_{lang}.png'
                title = f'Risk-Coverage ({domain}:{lang})'
                plot_risk_coverage(rc_lang, title, out_png)
    print(f'Calibration plots written to {CALIB_DIR}')


if __name__ == '__main__':
    main()
