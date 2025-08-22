"""Distant supervision miner to learn primitive detectors from corpora.

This module weakly labels sentences using existing detectors and trains a
lightweight classifier (TF-IDF + LogisticRegression) per primitive.

No new heavy dependencies are introduced beyond scikit-learn already in use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ..table.schema import PeriodicTable
from ..table.bootstrap import augment_with_basics
from ..table.nsm_seed import augment_with_nsm_primes
from ..table.algebra import PrimitiveAlgebra

import click


logger = logging.getLogger(__name__)


@dataclass
class PrimitiveClassifier:
    name: str
    vectorizer: TfidfVectorizer
    model: LogisticRegression


class DistantSupervisionMiner:
    """Train simple per-primitive classifiers using weak labels from detectors."""

    def __init__(self, table: PeriodicTable):
        self.table = table
        self.algebra = PrimitiveAlgebra(table)

    def weak_label(self, sentences: List[str], target_primitives: List[str]) -> Dict[str, List[int]]:
        """Produce weak labels per primitive using current factorization.

        Returns a dict mapping primitive -> binary list aligned with sentences.
        """
        labels: Dict[str, List[int]] = {p: [0] * len(sentences) for p in target_primitives}
        for i, s in enumerate(sentences):
            names = [p.name for p in self.algebra.factor(s)]
            for p in target_primitives:
                if p in names:
                    labels[p][i] = 1
        return labels

    def train_primitive_classifier(
        self, sentences: List[str], labels: List[int]
    ) -> Tuple[TfidfVectorizer, LogisticRegression]:
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, labels, test_size=0.2, random_state=42, stratify=labels if any(labels) else None
        )
        vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
        Xtr = vectorizer.fit_transform(X_train)
        Xte = vectorizer.transform(X_test)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        logger.info("Primitive classifier report: %s", json.dumps(report))
        return vectorizer, model

    def train(self, sentences: List[str], target_primitives: List[str]) -> Dict[str, PrimitiveClassifier]:
        weak = self.weak_label(sentences, target_primitives)
        classifiers: Dict[str, PrimitiveClassifier] = {}
        for name in target_primitives:
            y = weak[name]
            if sum(y) < 5:
                logger.info("Skipping %s (insufficient positives: %d)", name, sum(y))
                continue
            vec, mdl = self.train_primitive_classifier(sentences, y)
            classifiers[name] = PrimitiveClassifier(name=name, vectorizer=vec, model=mdl)
        return classifiers

    @staticmethod
    def save(classifiers: Dict[str, PrimitiveClassifier], path: str) -> None:
        # Save via joblib-compatible pickles
        from pickle import dump
        payload = {
            name: {
                "name": clf.name,
                "vectorizer": clf.vectorizer,
                "model": clf.model,
            }
            for name, clf in classifiers.items()
        }
        with open(path, "wb") as fh:
            dump(payload, fh)

    @staticmethod
    def load(path: str) -> Dict[str, PrimitiveClassifier]:
        from pickle import load
        with open(path, "rb") as fh:
            payload = load(fh)
        result: Dict[str, PrimitiveClassifier] = {}
        for name, obj in payload.items():
            result[name] = PrimitiveClassifier(
                name=obj["name"], vectorizer=obj["vectorizer"], model=obj["model"]
            )
        return result


def _read_corpora(paths: Iterable[str]) -> List[str]:
    lines: List[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            lines.extend([ln.strip() for ln in fh if ln.strip()])
    return lines


@click.command()
@click.option("--input", "-i", required=True, help="Input periodic table JSON file")
@click.option("--augment-basics", is_flag=True, help="Augment with basics")
@click.option("--augment-nsm", is_flag=True, help="Augment with NSM primes")
@click.option("--corpus", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True,
              help="Corpus file(s) for distant supervision")
@click.option("--targets", multiple=True, help="Target primitive names (defaults to keep-set if provided)")
@click.option("--keep-set", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Optional keep-set file to choose targets from")
@click.option("--out", default="models/primitive_detectors.pkl", show_default=True, help="Output model path")
def main(input: str, augment_basics: bool, augment_nsm: bool, corpus: Tuple[str, ...],
         targets: Tuple[str, ...], keep_set: str | None, out: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    table = PeriodicTable.from_dict(json.load(open(input, "r")))
    if augment_basics:
        augment_with_basics(table)
    if augment_nsm:
        augment_with_nsm_primes(table)

    # Determine targets
    target_list: List[str]
    if targets:
        target_list = list(targets)
    elif keep_set:
        content = Path(keep_set).read_text(encoding="utf-8").strip()
        try:
            parsed = json.loads(content)
            target_list = [x for x in parsed if isinstance(x, str)]
        except json.JSONDecodeError:
            target_list = [ln.strip() for ln in content.splitlines() if ln.strip()]
    else:
        target_list = list(table.primitives.keys())

    sentences = _read_corpora(corpus)
    miner = DistantSupervisionMiner(table)
    classifiers = miner.train(sentences, target_list)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    miner.save(classifiers, out)
    logging.info("Saved %d primitive classifiers to %s", len(classifiers), out)


if __name__ == "__main__":
    main()


