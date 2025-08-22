.PHONY: keep-set eval-filtered delta-smoke

PY=./.venv/bin/python

# Regenerate keep-set from corpora, augmenting with basics + NSM, and write filtered table
keep-set:
	$(PY) -m src.validate.compression \
	  --input primitives.json \
	  --augment-basics --augment-nsm \
	  --build-keep-set \
	  --attr-corpus data/simple_corpus.txt \
	  --attr-corpus data/tatoeba_en.txt \
	  --top-k 40 --min-share 0.005 \
	  --keep-out keep_union.txt \
	  --filtered-out filtered_primitives.json \
	  --eval-corpus data/simple_corpus.txt \
	  --eval-corpus data/tatoeba_en.txt \
	  --domains text --samples 100 \
	  --output compression_report_filtered.txt

# Evaluate corpora using filtered table
eval-filtered:
	$(PY) -m src.validate.compression \
	  --input filtered_primitives.json \
	  --eval-corpus data/simple_corpus.txt \
	  --eval-corpus data/tatoeba_en.txt \
	  --domains text --samples 100 \
	  --output compression_report_filtered.txt

# Run Δ vs raw smoke test
delta-smoke:
	$(PY) -m src.validate.compression \
	  --input filtered_primitives.json \
	  --delta-smoke

# Train distant supervision detectors
.PHONY: train-ds
train-ds:
	$(PY) -m src.mining.distant_supervision \
	  --input primitives.json \
	  --augment-basics --augment-nsm \
	  --corpus data/simple_corpus.txt \
	  --corpus data/tatoeba_en.txt \
	  --keep-set keep_union.txt \
	  --out models/primitive_detectors.pkl

.PHONY: mine-patterns
mine-patterns:
	$(PY) -m src.mining.pattern_miner \
	  --corpus data/simple_corpus.txt \
	  --corpus data/tatoeba_en_10000.txt \
	  --corpus data/tatoeba_fr_10000.txt \
	  --min-support 5 \
	  --top-k 100 \
	  --out candidates.json

.PHONY: validate-candidates
validate-candidates:
	$(PY) -m src.validate.candidate_validator \
	  --base-table primitives.json \
	  --candidates candidates.json \
	  --corpus data/simple_corpus.txt \
	  --corpus data/tatoeba_en_10000.txt \
	  --corpus data/tatoeba_fr_10000.txt \
	  --augment-basics --augment-nsm \
	  --gain-threshold 0 \
	  --delta-threshold 0 \
	  --delta-pairs 200 \
	  --out-keep kept_candidates.json \
	  --merge-into merged_validated_primitives.json


