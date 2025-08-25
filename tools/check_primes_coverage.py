#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Set

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'checkpoint'


def current_covered_primes() -> Set[str]:
    from robust_aspect_mapper import AspectType
    from eil_reasoning_integration import EILRuleType
    # Map implemented aspects/rules to NSM-like primitives we expose in EIL
    covered: Set[str] = set()
    # Aspectual/temporal
    mapping_aspect = {
        AspectType.RECENT_PAST.value: 'RECENT_PAST',
        AspectType.ONGOING.value: 'ONGOING',
        AspectType.ONGOING_FOR.value: 'ONGOING_FOR',
        AspectType.ALMOST_DO.value: 'ALMOST_DO',
        AspectType.STOP.value: 'STOP',
        AspectType.RESUME.value: 'RESUME',
        AspectType.DO_AGAIN.value: 'DO_AGAIN',
        getattr(AspectType, 'STILL', None) and AspectType.STILL.value: 'STILL',
        getattr(AspectType, 'NOT_YET', None) and AspectType.NOT_YET.value: 'NOT_YET',
    }
    for k, v in mapping_aspect.items():
        if k:
            covered.add(v)
    # EIL rule predicates that imply temporal/logic primitives
    mapping_rules = {
        EILRuleType.ASPECT_RECENT_PAST.value: 'PAST',
        EILRuleType.ASPECT_ONGOING_FOR.value: 'DURING',
        EILRuleType.ASPECT_ALMOST_DO.value: 'NEAR',
        EILRuleType.ASPECT_STILL.value if hasattr(EILRuleType, 'ASPECT_STILL') else None: 'CONTINUES',
        EILRuleType.ASPECT_NOT_YET.value if hasattr(EILRuleType, 'ASPECT_NOT_YET') else None: 'EXPECT',
        EILRuleType.ASPECT_START.value if hasattr(EILRuleType, 'ASPECT_START') else None: 'BEGUN',
        EILRuleType.ASPECT_FINISH.value if hasattr(EILRuleType, 'ASPECT_FINISH') else None: 'FINISHED',
        EILRuleType.MODAL_ABILITY.value if hasattr(EILRuleType, 'MODAL_ABILITY') else None: 'CAN',
        EILRuleType.MODAL_PERMISSION.value if hasattr(EILRuleType, 'MODAL_PERMISSION') else None: 'MAY',
        EILRuleType.MODAL_OBLIGATION.value if hasattr(EILRuleType, 'MODAL_OBLIGATION') else None: 'MUST',
        EILRuleType.QUANT_NARROW.value: 'NOT',
        EILRuleType.QUANT_WIDE.value: 'ALL',
    }
    for k, v in mapping_rules.items():
        if k:
            covered.add(v)
    return covered


def build_registry() -> List[Dict[str, Any]]:
    # Minimal scaffold; target_total=65; mark known subset as wired
    target = [
        'NOT','ALL','NO',
        'RECENT_PAST','ONGOING','ONGOING_FOR','DURING','PAST',
        'NEAR','STOP','RESUME','DO_AGAIN',
        'STILL','NOT_YET','EXPECT','CONTINUES',
        'BEGUN','FINISHED',
        'CAN','MAY','MUST'
    ]
    covered = current_covered_primes()
    reg: List[Dict[str, Any]] = []
    for name in target:
        reg.append({
            'name': name,
            'wired': name in covered,
            'detector_wired': name in covered,
            'rules_wired': name in covered,
        })
    return reg


def compute_coverage() -> Dict[str, Any]:
    reg = build_registry()
    target_total = 65
    wired_count = sum(1 for r in reg if r.get('wired'))
    return {
        'registry': reg,
        'covered_count': wired_count,
        'target_total': target_total,
        'missing_count': max(0, target_total - wired_count),
        'note': 'Scaffold registry; expand to full 65 with exponents and suites.'
    }


def main():
    run_id = os.environ.get('RUN_ID') or 'dev'
    out = compute_coverage()
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'nsm_primes_coverage.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'covered': out['covered_count'], 'missing': out['missing_count']}, indent=2))


if __name__ == '__main__':
    main()


