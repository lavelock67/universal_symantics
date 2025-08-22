#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.sense.linker import CachedBabelNetLinker


def load_spacy(lang: str):
    import spacy
    if lang == 'en':
        return spacy.load('en_core_web_sm')
    if lang == 'es':
        return spacy.load('es_core_news_sm')
    if lang == 'fr':
        return spacy.load('fr_core_news_sm')
    raise ValueError(f'Unsupported lang: {lang}')


def salient_lemma(doc) -> str:
    # Pick first salient content lemma (NOUN > VERB > PROPN > ADJ)
    for pos in ("NOUN", "VERB", "PROPN", "ADJ"):
        for tok in doc:
            if tok.pos_ == pos and tok.lemma_ and tok.is_alpha:
                return tok.lemma_.lower()
    # fallback: first alpha token
    for tok in doc:
        if tok.is_alpha and tok.lemma_:
            return tok.lemma_.lower()
    return ''


def lang_code_for_bn(lang: str) -> str:
    return {'en': 'EN', 'es': 'ES', 'fr': 'FR'}.get(lang, lang.upper())


def main():
    data_path = Path('data/parallel_test_data.json')
    if not data_path.exists():
        print('Parallel data not found')
        return
    parallel = json.loads(data_path.read_text(encoding='utf-8'))

    nlps = {lang: load_spacy(lang) for lang in ['en', 'es', 'fr']}
    linker = CachedBabelNetLinker()

    # Collect salient lemmas per (idx, lang)
    lemmas: Dict[Tuple[int, str], str] = {}
    max_len = min(len(parallel['en']), len(parallel['es']), len(parallel['fr']))
    for idx in range(max_len):
        for lang in ['en', 'es', 'fr']:
            sent = parallel[lang][idx]
            doc = nlps[lang](sent)
            lemmas[(idx, lang)] = salient_lemma(doc)

    # Query BN (cached)
    terms: List[Tuple[str, str]] = []
    for (idx, lang), lemma in lemmas.items():
        if lemma:
            terms.append((lemma, lang_code_for_bn(lang)))
    linked = linker.link_terms(terms)

    # Build per-idx synsets and alignment
    results: List[Dict] = []
    aligned = 0
    for idx in range(max_len):
        syn_en = linked.get((lemmas[(idx, 'en')], 'EN'), []) if lemmas.get((idx, 'en')) else []
        syn_es = linked.get((lemmas[(idx, 'es')], 'ES'), []) if lemmas.get((idx, 'es')) else []
        syn_fr = linked.get((lemmas[(idx, 'fr')], 'FR'), []) if lemmas.get((idx, 'fr')) else []
        inter = set(syn_en) & set(syn_es) & set(syn_fr)
        if inter:
            aligned += 1
        results.append({
            'idx': idx,
            'lemmas': {
                'en': lemmas[(idx, 'en')], 'es': lemmas[(idx, 'es')], 'fr': lemmas[(idx, 'fr')]
            },
            'synsets': {
                'en': syn_en, 'es': syn_es, 'fr': syn_fr
            },
            'intersection': list(inter)
        })

    summary = {
        'total': max_len,
        'aligned_all3': aligned,
        'alignment_rate': aligned / max_len if max_len else 0.0
    }

    out = {'summary': summary, 'results': results}
    out_path = Path('data/babelnet_alignment.json')
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()



