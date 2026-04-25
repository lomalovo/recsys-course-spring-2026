SEED       ?= 31312
EPISODES   ?= 30000
DATA_DIR   ?= ./data
TRAIN_LOGS ?= ./data/train_logs
RETRIEVAL_K ?= 500
RERANK_K    ?= 200

VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

.PHONY: setup run collect_data update_model clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --timeout 120 -q
	$(PIP) install -r sim/requirements.txt --timeout 120 -q
	$(PIP) install -r botify/requirements.txt --timeout 120 -q
	cd botify && docker compose down -v --remove-orphans 2>/dev/null || true
	cd botify && docker compose up -d --build --force-recreate --scale recommender=2
	sleep 20

run:
	cd sim && echo "n" | ../$(PYTHON) -m sim.run \
		--episodes $(EPISODES) \
		--config   config/env.yml \
		single --recommender remote --seed $(SEED)
	mkdir -p $(DATA_DIR)
	$(PYTHON) script/dataclient.py --recommender 2 log2local $(DATA_DIR)
	$(PYTHON) analyze_ab.py --data $(DATA_DIR) --output $(DATA_DIR)/ab_result.json

collect_data:
	cd sim && echo "n" | ../$(PYTHON) -m sim.run \
		--episodes $(EPISODES) \
		--config   config/env.yml \
		single --recommender remote --seed $(SEED)
	mkdir -p $(TRAIN_LOGS)
	$(PYTHON) script/dataclient.py --recommender 2 log2local $(TRAIN_LOGS)

update_model:
	$(PIP) install -r requirements-training.txt --timeout 180 -q
	$(PYTHON) script/train_retrieval.py \
		--logs $(TRAIN_LOGS) \
		--tracks botify/data/tracks.json \
		--out-candidates botify/data/retrieval_candidates.jsonl \
		--out-meta botify/data/retrieval_meta.npz \
		--topk $(RETRIEVAL_K) \
		--seed $(SEED)
	$(PYTHON) script/train_quantile_ranker.py \
		--logs $(TRAIN_LOGS) \
		--candidates botify/data/retrieval_candidates.jsonl \
		--meta botify/data/retrieval_meta.npz \
		--output botify/data/learned_i2i.jsonl \
		--topk $(RERANK_K) \
		--seed $(SEED)
	cd botify && docker compose up -d --build --force-recreate --scale recommender=2

clean:
	cd botify && docker compose down -v --remove-orphans 2>/dev/null || true
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
