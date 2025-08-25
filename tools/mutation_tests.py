#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from robust_aspect_mapper import RobustAspectDetector, Language
from quant_scope_normalizer import QuantifierScopeNormalizer, Language as QLang

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'checkpoint'


def mutate_aspect(text: str, lang: Language) -> List[str]:
    muts: List[str] = []
    tl = text
    # Remove crucial auxiliary/cue
    if lang == Language.EN:
        if 'have just' in tl:
            muts.append(tl.replace('have just', 'have'))
        if 'has just' in tl:
            muts.append(tl.replace('has just', 'has'))
        if 'had just' in tl:
            muts.append(tl.replace('had just', 'had'))
        if ' have been ' in tl:
            muts.append(tl.replace(' have been ', ' have '))
        if ' almost ' in tl:
            muts.append(tl.replace(' almost ', ' '))
    elif lang == Language.ES:
        if 'acabo de ' in tl or 'acaba de ' in tl:
            muts.append(tl.replace(' de ', ' '))
        if 'Lleva ' in tl or 'lleva ' in tl:
            muts.append(tl.replace('Lleva ', '').replace('lleva ', ''))
        if 'Casi ' in tl or 'casi ' in tl:
            muts.append(tl.replace('Casi ', '').replace('casi ', ''))
    elif lang == Language.FR:
        if "en train de" in tl:
            muts.append(tl.replace(' en train de ', ' '))
        if "a failli" in tl or "ai failli" in tl:
            muts.append(tl.replace(' failli ', ' '))
        if "vient d'" in tl or "viens d'" in tl:
            muts.append(tl.replace(" d'", ' '))
    return [m for m in muts if m and m != tl]


def mutate_quant(text: str, lang: str) -> List[str]:
    t = text
    muts: List[str] = []
    if lang == 'en':
        muts.append(t.replace('Not all', 'Only some'))
        muts.append(t.replace('All ', 'Only some '))
    elif lang == 'es':
        muts.append(t.replace('No todos', 'Solo algunos'))
        muts.append(t.replace('Todos ', 'Solo algunos '))
    elif lang == 'fr':
        muts.append(t.replace('Tous ', 'Seulement quelques '))
        muts.append(t.replace('Pas tous', 'Seulement quelques'))
    return [m for m in muts if m and m != t]


def mutation_score() -> Dict[str, Any]:
    det = RobustAspectDetector()
    qn = QuantifierScopeNormalizer()
    # Simple seeds
    aspect_seeds: List[Tuple[str, Language]] = [
        ("I have just finished.", Language.EN),
        ("I have been working for three hours.", Language.EN),
        ("He almost fell.", Language.EN),
        ("Acabo de salir.", Language.ES),
        ("Lleva tres horas estudiando.", Language.ES),
        ("Casi me caigo.", Language.ES),
        ("Il est en train de travailler.", Language.FR),
        ("Je viens d'arriver.", Language.FR),
        ("J'ai failli tomber.", Language.FR),
    ]
    quant_seeds: List[Tuple[str, str]] = [
        ("Not all students study.", 'en'),
        ("No todos los niños juegan.", 'es'),
        ("Tous les enfants ne jouent pas.", 'fr'),
    ]
    total = killed = 0
    details: List[Dict[str, Any]] = []
    # Aspect mutations
    for txt, lang in aspect_seeds:
        muts = mutate_aspect(txt, lang)
        for m in muts:
            total += 1
            d = det.detect_aspects(m, lang)
            alive = len(d.detected_aspects) > 0
            if not alive:
                killed += 1
            details.append({"domain": "aspect", "orig": txt, "mutant": m, "killed": (not alive)})
    # Quant mutations
    for txt, lang in quant_seeds:
        muts = mutate_quant(txt, lang)
        for m in muts:
            total += 1
            qa = qn.normalize_quantifier_scope(m, QLang(lang))
            alive = qa.scope_resolution is not None and (qa.confidence or 0.0) >= 0.11
            if not alive:
                killed += 1
            details.append({"domain": "quantifiers", "orig": txt, "mutant": m, "killed": (not alive)})
    score = killed / total if total else 1.0
    return {"total": total, "killed": killed, "score": score, "details": details}


def main():
    import os
    run_id = os.environ.get('RUN_ID', 'dev')
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rep = mutation_score()
    (run_dir / 'mutation_report.json').write_text(json.dumps(rep, indent=2))
    print(f"Mutation score: {rep['score']:.3f}  ({rep['killed']}/{rep['total']})")


if __name__ == '__main__':
    main()


