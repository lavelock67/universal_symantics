#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests


class SenseLinker:
    def link_terms(self, terms: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[str]]:
        raise NotImplementedError


class CachedBabelNetLinker(SenseLinker):
    def __init__(self, cache_path: str = "data/bn_cache.json", rate_limit_qps: float = 2.0):
        self.base = "https://babelnet.io/v5/getSynsetIds"
        self.key = os.getenv("BN_KEY", "")
        self.cache_file = Path(cache_path)
        self.cache: Dict[str, List[str]] = {}
        self.min_delay = 1.0 / max(0.1, rate_limit_qps)
        self._last_call = 0.0
        if self.cache_file.exists():
            try:
                self.cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _cache_key(self, lemma: str, lang: str) -> str:
        return f"{lang}:::{lemma.lower()}"

    def _save_cache(self) -> None:
        self.cache_file.parent.mkdir(exist_ok=True)
        self.cache_file.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _rate_limit(self):
        now = time.time()
        sleep_for = self.min_delay - (now - self._last_call)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_call = time.time()

    def _query_ids(self, lemma: str, lang: str) -> List[str]:
        if not self.key:
            return []
        self._rate_limit()
        params = {
            "lemma": lemma,
            "searchLang": lang,
            "key": self.key,
        }
        try:
            resp = requests.get(self.base, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # data is a list of objects with id field
                return [str(x.get("id", "")) for x in data if x.get("id")]
            return []
        except Exception:
            return []

    def link_terms(self, terms: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[str]]:
        out: Dict[Tuple[str, str], List[str]] = {}
        for lemma, lang in terms:
            ck = self._cache_key(lemma, lang)
            if ck in self.cache:
                out[(lemma, lang)] = self.cache[ck]
                continue
            ids = self._query_ids(lemma, lang)
            self.cache[ck] = ids
            out[(lemma, lang)] = ids
        self._save_cache()
        return out

# TODO: Integrate BMR (BabelNet Meaning Representation) parsing/generation
# TODO: Add BabelNet synset linking for sense-anchored semantics
# TODO: Align NSM explications with BabelNet synsets in graphs
# TODO: Ingest UDS dataset; mine candidate idea-primes via attributes
# TODO: Score idea-primes with ΔMDL, cross-ling transfer, and stability
# TODO: Add DeepNSM explication model inference; compare to templates
# TODO: Implement NSM+graph joint decoding for generation (conditioned)
# TODO: Calibrate NLI substitutability thresholds per language; cache



