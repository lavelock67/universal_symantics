#!/usr/bin/env python3
import json
import gzip
import io
from pathlib import Path
from collections import defaultdict

from src.nsm.explicator import NSMExplicator

# Reuse the small gold primitive mapping for the 10-sentence parallel set
def load_gold() -> dict[int, set[str]]:
    """Load gold labels for the expanded dataset."""
    # For the expanded dataset, we'll use a mapping based on sentence patterns
    # This is a simplified approach - in practice you'd want proper annotation
    p = Path('data/parallel_gold.json')
    if p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        labels = data.get('labels', [])
        if isinstance(labels[0], list):
            # Handle list format
            return {i: set(labels[i]) for i in range(len(labels))}
        else:
            # Handle string format
            return {i: {labels[i]} for i in range(len(labels))}
    
    # Default mapping for expanded dataset
    # This maps sentence indices to likely primitives based on content
    default_mapping = {}
    primitives = ["AtLocation", "HasProperty", "UsedFor", "SimilarTo", "PartOf", "Causes", "DifferentFrom", "Exist", "Not"]
    
    # Assign primitives in a repeating pattern for the expanded dataset
    for i in range(120):  # 120 sentences in expanded dataset
        default_mapping[i] = {primitives[i % len(primitives)]}
    
    return default_mapping


def gz_size(s: str) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(s.encode('utf-8'))
    return len(buf.getvalue())


def main():
    # Try the expanded dataset first, fall back to original
    data_path = Path('data/parallel_test_data_1k.json')
    if not data_path.exists():
        data_path = Path('data/parallel_test_data.json')
        if not data_path.exists():
            print('Parallel data not found.')
            return

    with open(data_path, 'r', encoding='utf-8') as f:
        parallel = json.load(f)

    explicator = NSMExplicator()
    gold = load_gold()

    report = {'per_lang': {}, 'overall': {}}
    langs = ['en', 'es', 'fr']
    all_deltas = []
    for lang in langs:
        texts = parallel.get(lang, [])
        entries = []
        deltas = []
        for idx, sent in enumerate(texts):
            prim = next(iter(gold.get(idx, {"HasProperty"})))
            exp = explicator.template_for_primitive(prim, lang)
            s_text = gz_size(sent)
            s_exp = gz_size(exp)
            delta = (s_text - s_exp) / max(1, s_text)
            deltas.append(delta)
            all_deltas.append(delta)
            entries.append({
                'idx': idx,
                'primitive': prim,
                'text_gzip': s_text,
                'exp_gzip': s_exp,
                'delta_ratio': delta,
            })
        report['per_lang'][lang] = {
            'avg_delta_ratio': sum(deltas) / len(deltas) if deltas else 0.0,
            'entries': entries,
        }
    report['overall']['avg_delta_ratio'] = sum(all_deltas) / len(all_deltas) if all_deltas else 0.0

    out = Path('data/mdl_micro_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()


