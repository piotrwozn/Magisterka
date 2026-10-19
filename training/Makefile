# SOR-AI: trening modeli klasyfikacji ESI
#
# Najczęściej:
#   make install      — zależności
#   make prepare      — przygotowanie danych Yale → splity ESI + DATA_AUDIT.md
#   make train-all    — wszystkie modele, zestaw 'triage' (bez tuningu)
#   make tune-all     — tuning Optuną + trening wszystkich modeli (długo: ok. 1 noc)
#   make compare      — porównanie modeli + materiał do ręcznego łączenia
#   make test         — testy
#
# Pojedynczy model: make train MODEL=xgboost   (FEATURES=full, ARGS="--weights balanced")

PYTHON ?= python
MODEL ?= xgboost
FEATURES ?= triage
ARGS ?=

.PHONY: install convert prepare prepare-site train train-all train-full tune tune-all \
        smoke compare eval explain cv demo mlflow test test-fast lint format clean clean-all

install:
	$(PYTHON) -m pip install -r requirements.txt

# --- Dane ---
convert:
	$(PYTHON) scripts/01_convert_rdata.py

prepare:
	$(PYTHON) scripts/02_preprocess.py --split random

# Walidacja zewnętrzna: oddział C w całości jako test
prepare-site:
	$(PYTHON) scripts/02_preprocess.py --split site

# --- Trening ---
train:
	$(PYTHON) scripts/03_train.py --model $(MODEL) --feature-set $(FEATURES) $(ARGS)

train-all:
	$(PYTHON) scripts/03_train.py --model all --feature-set $(FEATURES) $(ARGS)

train-full:
	$(PYTHON) scripts/03_train.py --model all --feature-set full $(ARGS)

tune:
	$(PYTHON) scripts/03_train.py --model $(MODEL) --feature-set $(FEATURES) --tune $(ARGS)

tune-all:
	$(PYTHON) scripts/03_train.py --model all --feature-set $(FEATURES) --tune $(ARGS)

# Szybki test całego pipeline'u na próbce (kilka minut) — nie do raportowania
smoke:
	$(PYTHON) scripts/03_train.py --model all --sample 20000 --eval-sample 10000 --tag smoke

# --- Wyniki ---
compare:
	$(PYTHON) scripts/07_compare_runs.py --feature-set $(FEATURES)

eval:
	$(PYTHON) scripts/04_evaluate.py --model-name $(MODEL)

explain:
	$(PYTHON) scripts/05_explain.py --model-name $(MODEL) --n-patients 5

cv:
	$(PYTHON) scripts/06_cross_validate.py --model $(MODEL) --feature-set $(FEATURES)

demo:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui --backend-store-uri file://$(PWD)/mlruns --port 5000

# --- Jakość ---
test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -q -m "not slow"

lint:
	@which ruff > /dev/null && ruff check src/ tests/ scripts/ app/ || echo "Zainstaluj ruff: pip install ruff"

format:
	@which black > /dev/null && black src/ tests/ scripts/ app/ || echo "Zainstaluj black: pip install black"

# --- Sprzątanie ---
clean:
	@echo "Usuwam logi, wyniki i przygotowane splity (runy w runs/ zostają)…"
	rm -rf logs/training/* logs/experiments/* logs/evaluation/*
	rm -rf results/figures/* results/tables/*
	rm -rf data/processed/esi
	rm -rf mlruns/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf runs/ logs/
	@echo "Usunięto także wszystkie runy (modele, predykcje, leaderboard)."
