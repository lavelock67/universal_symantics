#!/usr/bin/env python3
import json
import random
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'suites'
DATASETS_DIR = ROOT / 'datasets' / 'aspect'

random.seed(42)

Subjects = {
    'en': ['students', 'children', 'doctors', 'drivers', 'artists', 'runners'],
    'es': ['niños', 'estudiantes', 'doctores', 'conductores', 'artistas'],
    'fr': ['enfants', 'étudiants', 'médecins', 'conducteurs', 'artistes']
}

Predicates = {
    'en': ['study', 'play', 'arrive', 'run', 'read'],
    'es': ['estudiar', 'jugar', 'llegar', 'correr', 'leer'],
    'fr': ['étudier', 'jouer', 'arriver', 'courir', 'lire']
}

AspectVerbs = {
    'en': ['finish', 'work', 'fall', 'run', 'eat'],
    'es': ['salir', 'estudiar', 'trabajar', 'caer', 'leer'],
    'fr': ['travailler', 'venir', 'tomber', 'lire', 'arriver']
}

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def gen_quantifiers(lang: str, n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    subs = Subjects[lang]
    preds = Predicates[lang]
    gid = 0
    # Positives
    for _ in range(n // 2):
        subj = random.choice(subs)
        pred = random.choice(preds)
        if lang == 'en':
            text = f"Not all {subj} {pred}."
        elif lang == 'es':
            text = f"No todos los {subj} {pred}an."
        else:
            text = f"Tous les {subj} ne {pred} pas."
        rows.append({
            'text': text,
            'language': lang,
            'label': 1,
            'predicate': pred,
            'group_id': f"qpos-{lang}-{gid}",
            'type': 'base'
        })
        # paraphrase
        if lang == 'en':
            para = f"Not every {subj} {pred}."
        elif lang == 'es':
            para = f"No todos {pred}an los {subj}."
        else:
            para = f"Pas tous les {subj} {pred}."
        rows.append({
            'text': para,
            'language': lang,
            'label': 1,
            'predicate': pred,
            'group_id': f"qpos-{lang}-{gid}",
            'type': 'paraphrase'
        })
        # counterfactual (flip)
        if lang == 'en':
            cf = f"All {subj} {pred}."
        elif lang == 'es':
            cf = f"Todos los {subj} {pred}an."
        else:
            cf = f"Tous les {subj} {pred}."
        rows.append({
            'text': cf,
            'language': lang,
            'label': 0,
            'predicate': pred,
            'group_id': f"qpos-{lang}-{gid}",
            'type': 'counterfactual'
        })
        gid += 1
    # Negatives
    neg_triggers = {
        'en': ["only", "just", "merely"],
        'es': ["solo", "solamente", "meramente"],
        'fr': ["seulement", "uniquement", "ne ... que"],
    }
    for _ in range(n // 2):
        subj = random.choice(subs)
        pred = random.choice(preds)
        trigger = random.choice(neg_triggers[lang])
        if lang == 'en':
            text = f"{trigger.capitalize()} some {subj} {pred}."
        elif lang == 'es':
            text = f"{trigger.capitalize()} algunos {subj} {pred}an."
        else:
            if trigger == 'ne ... que':
                text = f"{subj.capitalize()} ne {pred} que parfois."
            else:
                text = f"{trigger.capitalize()} quelques {subj} {pred}."
        rows.append({
            'text': text,
            'language': lang,
            'label': 0,
            'predicate': pred,
            'group_id': f"qneg-{lang}-{gid}",
            'type': 'negative'
        })
        gid += 1
    return rows


def gen_aspects(lang: str, n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    verbs = AspectVerbs[lang]
    gid = 0
    # Positives across types
    for _ in range(n // 2):
        v = random.choice(verbs)
        t = random.choice(['recent', 'ongoing_for', 'ongoing', 'almost', 'stop', 'resume'])
        if lang == 'en':
            if t == 'recent':
                text = random.choice([f"I have just {v}ed.", f"She had just {v}ed."])
            elif t == 'ongoing_for':
                text = f"I have been {v}ing for three hours."
            elif t == 'ongoing':
                text = f"I am {v}ing."
            elif t == 'almost':
                text = f"I almost {v}."
            elif t == 'stop':
                text = f"I stopped {v}ing."
            else:
                text = f"I started again."
        elif lang == 'es':
            if t == 'recent':
                text = f"Acabo de {v}."
            elif t == 'ongoing_for':
                text = f"Lleva tres horas {v}ando."
            elif t == 'ongoing':
                text = f"Estoy {v}ando."
            elif t == 'almost':
                text = f"Casi {v}."
            elif t == 'stop':
                text = f"Dejó de {v}."
            else:
                text = f"Volvió a {v}."
        else:
            if t == 'recent':
                text = f"Je viens de {v}."
            elif t == 'ongoing_for' or t == 'ongoing':
                text = f"Il est en train de {v}."
            elif t == 'almost':
                text = f"J'ai failli {v}."
            elif t == 'stop':
                text = f"Elle a cessé de {v}."
            else:
                text = f"Il a recommencé à {v}."
        rows.append({
            'text': text,
            'language': lang,
            'label': 1,
            'group_id': f"apos-{lang}-{gid}",
            'type': t
        })
        # Paraphrase (simple surface)
        rows.append({
            'text': text.replace('.', '!'),
            'language': lang,
            'label': 1,
            'group_id': f"apos-{lang}-{gid}",
            'type': 'paraphrase'
        })
        # Counterfactual (try to flip)
        if lang == 'en' and t == 'recent':
            cf = None
            if 'have just' in text:
                cf = text.replace('have just', 'have never')
            elif 'has just' in text:
                cf = text.replace('has just', 'has never')
            elif 'had just' in text:
                cf = text.replace('had just', 'had never')
            if cf is not None and cf != text:
                rows.append({'text': cf, 'language': lang, 'label': 0, 'group_id': f"apos-{lang}-{gid}", 'type': 'counterfactual'})
        gid += 1
    # Negatives
    negs = {
        'en': ["I have been to Paris.", "I just want to go.", "Stop the car!"],
        'es': ["He estado en París.", "Solo quiero ir.", "Para el coche!"],
        'fr': ["J'ai été à Paris.", "Je veux juste partir.", "Arrête la voiture !"],
    }
    for _ in range(n // 2):
        text = random.choice(negs[lang])
        rows.append({'text': text, 'language': lang, 'label': 0, 'group_id': f"aneg-{lang}-{gid}", 'type': 'negative'})
        gid += 1
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Quantifiers
    for lang in ['en', 'es', 'fr']:
        rows = gen_quantifiers(lang, 120)
        write_jsonl(OUT_DIR / 'quantifiers' / f'{lang}.jsonl', rows)
    # Aspect
    for lang in ['en', 'es', 'fr']:
        rows = gen_aspects(lang, 120)
        write_jsonl(OUT_DIR / 'aspect' / f'{lang}.jsonl', rows)
    # Negatives pool
    neg = []
    for lang in ['en','es','fr']:
        neg += [r for r in gen_aspects(lang, 60) if r['label'] == 0]
    write_jsonl(OUT_DIR / 'negatives.jsonl', neg)
    print(f"Suites generated under {OUT_DIR}")

    # Generate ES/FR held-out datasets as requested
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    def build_es_pos() -> List[Dict[str, Any]]:
        verbs = ['estudiar','trabajar','caer','leer','llegar','intentar','fumar','correr','venir','hacer','ver']
        items: List[Dict[str, Any]] = []
        gid = 0
        # RECENT_PAST: acabar de + INF (with clitics variants)
        for v in verbs[:6]:
            base = f"Acabo de {v}."
            items.append({'text': base, 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'recent'})
            items.append({'text': base.replace('.', '!'), 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'paraphrase'})
            if v == 'intentar':
                items.append({'text': 'Acabo de intentarlo.', 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'recent'})
            gid += 1
        # ONGOING_FOR: llevar + GER + duración (word orders)
        duras = ['tres horas', 'dos semanas', 'cinco minutos']
        gers = ['estudiando','trabajando','leyendo','corriendo','haciendo']
        for d in duras:
            gv = gers[gid % len(gers)]
            items.append({'text': f'Lleva {gv} {d}.', 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'ongoing_for'})
            items.append({'text': f'Lleva {d} {gv}.', 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'paraphrase'})
            gid += 1
        # ONGOING: estar + GER (eventive)
        estar_forms = ['Estoy','Estás','Está','Estamos','Estáis','Están']
        for e in estar_forms[:4]:
            gv = gers[gid % len(gers)]
            items.append({'text': f'{e} {gv}.', 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'ongoing'})
            gid += 1
        # ALMOST: casi / por poco
        items.append({'text': 'Casi me caigo.', 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'almost'})
        items.append({'text': 'Por poco se cae.', 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'almost'})
        gid += 1
        # STOP / RESUME / AGAIN
        items.append({'text': 'Dejó de fumar.', 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'stop'})
        gid += 1
        items.append({'text': 'Volvió a intentarlo.', 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'resume'})
        items.append({'text': 'Lo hizo otra vez.', 'language': 'es', 'label': 1, 'group_id': f'es-pos-{gid}', 'type': 'resume'})
        return items

    def build_es_neg() -> List[Dict[str, Any]]:
        negs = [
            'Llevar un libro.',
            'Casi perfecto.',
            'Dejar el coche.',
            'Estar en Madrid.',
            'Volver a casa.',
            'Voy en el tren.',
        ]
        return [{'text': t, 'language': 'es', 'label': 0, 'group_id': 'es-neg', 'type': 'negative'} for t in negs]

    def build_fr_pos() -> List[Dict[str, Any]]:
        verbs = ['venir','faire','lire','aller','travailler','arriver','courir','voir']
        items: List[Dict[str, Any]] = []
        gid = 0
        # RECENT_PAST: venir de + INF; à l'instant
        items.append({'text': "Je viens d'arriver.", 'language': 'fr', 'label': 1, 'group_id': f'fr-pos-{gid}', 'type': 'recent'})
        gid += 1
        items.append({'text': "Il arrive à l'instant.", 'language': 'fr', 'label': 1, 'group_id': f'fr-pos-{gid}', 'type': 'recent'})
        gid += 1
        # ONGOING: être en train de + INF
        items.append({'text': 'Il est en train de travailler.', 'language': 'fr', 'label': 1, 'group_id': f'fr-pos-{gid}', 'type': 'ongoing'})
        gid += 1
        # ALMOST: faillir + INF
        items.append({'text': 'Il a failli tomber.', 'language': 'fr', 'label': 1, 'group_id': f'fr-pos-{gid}', 'type': 'almost'})
        gid += 1
        # STOP / RESUME
        items.append({'text': 'Elle a cessé de venir.', 'language': 'fr', 'label': 1, 'group_id': f'fr-pos-{gid}', 'type': 'stop'})
        gid += 1
        items.append({'text': 'Il a recommencé à venir.', 'language': 'fr', 'label': 1, 'group_id': f'fr-pos-{gid}', 'type': 'resume'})
        items.append({'text': 'Elle se remet à lire.', 'language': 'fr', 'label': 1, 'group_id': f'fr-pos-{gid}', 'type': 'resume'})
        return items

    def build_fr_neg() -> List[Dict[str, Any]]:
        negs = [
            'Venir de Paris.',
            'Presque parfait.',
            'Cessez !',
            'Il est dans le train.',
            'Il est à Paris.',
        ]
        return [{'text': t, 'language': 'fr', 'label': 0, 'group_id': 'fr-neg', 'type': 'negative'} for t in negs]

    # Assemble heldouts to ~100+75 with simple replication
    es_pos = build_es_pos()
    es_neg = build_es_neg()
    while len(es_pos) < 100:
        es_pos += es_pos[: max(1, 100 - len(es_pos))]
    while len(es_neg) < 75:
        es_neg += es_neg[: max(1, 75 - len(es_neg))]
    fr_pos = build_fr_pos()
    fr_neg = build_fr_neg()
    while len(fr_pos) < 100:
        fr_pos += fr_pos[: max(1, 100 - len(fr_pos))]
    while len(fr_neg) < 75:
        fr_neg += fr_neg[: max(1, 75 - len(fr_neg))]
    es_rows = [{**r, 'group_id': f"es-heldout-{i}", 'type': r.get('type','base')} for i, r in enumerate(es_pos + es_neg)]
    fr_rows = [{**r, 'group_id': f"fr-heldout-{i}", 'type': r.get('type','base')} for i, r in enumerate(fr_pos + fr_neg)]
    write_jsonl(DATASETS_DIR / 'es' / 'heldout.jsonl', es_rows)
    write_jsonl(DATASETS_DIR / 'fr' / 'heldout.jsonl', fr_rows)
    print(f"Held-out datasets written under {DATASETS_DIR}")


if __name__ == '__main__':
    main()
