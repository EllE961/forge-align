# forge-align - common tasks. Every target is a plain python command, so they also work
# on Windows without make (copy the command). Assumes forge-llm is installed editable
# (pip install -e ../forge-llm) and its base checkpoint has already been trained/exported.
PY ?= python

.PHONY: help pairs train-sweep eval-sweep figures gif serve test lint all

help:
	@echo "pairs train-sweep eval-sweep figures gif serve test lint"

pairs:            ## sample 8 completions/task from the base model, score them, write preference pairs
	$(PY) -m align.preference_data --split train --k 8 --out data/pairs_train.jsonl
	$(PY) -m align.preference_data --split eval  --k 8 --out data/pairs_eval.jsonl

train-sweep:       ## DPO at four beta values on the same preference pairs
	$(PY) -m align.dpo configs/dpo-beta0.02.json
	$(PY) -m align.dpo configs/dpo-beta0.1.json
	$(PY) -m align.dpo configs/dpo-beta0.3.json
	$(PY) -m align.dpo configs/dpo-beta1.0.json

eval-sweep:        ## evaluate every run: proxy reward vs. true (held-out) fluency
	$(PY) -m align.evaluate --sweep

figures:            ## README figures -> docs/assets/
	$(PY) scripts/make_figures.py

gif:                 ## base-vs-aligned token-by-token race, with live reward bars -> docs/assets/race.gif
	$(PY) scripts/record_gif.py

serve:              ## playground on http://127.0.0.1:8002
	$(PY) -m align.server --port 8002

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

all: pairs train-sweep eval-sweep figures gif
