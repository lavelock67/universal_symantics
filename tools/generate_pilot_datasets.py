#!/usr/bin/env python3
import json
import random
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / 'data' / 'checkpoint'

random.seed(1234)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def build_pilot_for_lang(lang: str) -> Dict[str, Path]:
    # Build aspect and quantifier rows: 1000 mixed + targeted positives and negatives
    aspect_pos_templates = {
        'en': [
            'I have just finished.',
            'I have been working for three hours.',
            'He almost fell.',
            'I stopped working.',
            'She resumed running.'
        ],
        'es': [
            'Acabo de salir.',
            'Lleva tres horas estudiando.',
            'Casi me caigo.',
            'Dejó de fumar.',
            'Volvió a intentarlo.'
        ],
        'fr': [
            "Je viens d'arriver.",
            'Elle travaille depuis trois heures.',
            "J'ai failli tomber.",
            'Elle a cessé de venir.',
            'Il a recommencé à venir.'
        ]
    }
    quant_pos_templates = {
        'en': ['Not all students study.', 'No student arrived.'],
        'es': ['No todos los niños juegan.', 'Ningún estudiante llegó.'],
        'fr': ['Tous les enfants ne jouent pas.']
    }
    negatives = {
        'en': ['I have been to Paris.', 'This is almost perfect.', 'Stop the car!'],
        'es': ['He estado en Madrid.', 'Casi perfecto.', 'Para el coche!'],
        'fr': ["J'ai été à Paris.", 'Presque parfait.', 'Arrête la voiture !']
    }
    rows_aspect: List[Dict[str, Any]] = []
    rows_quant: List[Dict[str, Any]] = []
    # Targeted positives (~300)
    for _ in range(150):
        rows_aspect.append({'text': random.choice(aspect_pos_templates[lang]), 'language': lang, 'label': 1, 'type': 'aspect_pos'})
        rows_quant.append({'text': random.choice(quant_pos_templates.get(lang, quant_pos_templates['en'])), 'language': lang, 'label': 1, 'type': 'quant_pos'})
    # Hard negatives (~600)
    for _ in range(600):
        rows_aspect.append({'text': random.choice(negatives[lang]), 'language': lang, 'label': 0, 'type': 'neg'})
        rows_quant.append({'text': random.choice(negatives[lang]), 'language': lang, 'label': 0, 'type': 'neg'})
    # Fill up to ~1000 each by cycling
    def fill_to(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        base = list(rows)
        while len(rows) < n:
            rows += base[: max(1, n - len(rows))]
        return rows[:n]
    rows_aspect = fill_to(rows_aspect, 1000)
    rows_quant = fill_to(rows_quant, 1000)
    return {'aspect': rows_aspect, 'quant': rows_quant}


def main():
    import os
    run_id = os.environ.get('RUN_ID', 'pilot-1k')
    out_dir = OUT_BASE / run_id / 'datasets'
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in ['en','es','fr']:
        built = build_pilot_for_lang(lang)
        write_jsonl(out_dir / f'aspect_{lang}.jsonl', built['aspect'])
        write_jsonl(out_dir / f'quantifiers_{lang}.jsonl', built['quant'])
    print(f'Pilot datasets written under {out_dir}')


if __name__ == '__main__':
    main()


