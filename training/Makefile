# SOR-AI: trening modeli klasyfikacji ESI — skróty dla Linuksa/macOS (make + bash).
# Na Windows uruchamiaj skrypty wprost: python scripts/run_all.py … (training/README.md).
#
# Najczęściej:
#   make install      — przetestowane wersje zależności (requirements-lock.txt)
#   make check-env    — kontrola środowiska (GPU, RAM, dysk, wersje, dane)
#   make smoke        — 5–10 min: cały pipeline na próbce (test instalacji)
#   make all          — nocny przebieg: tuning + trening 6 modeli, ewaluacja, porównanie (wznawialny)
#   make status       — stan przebiegu z plików-znaczników
#   make test         — testy
#
# Zmienne: PROFILE=standard|thorough|quick|smoke  FEATURES=triage|full|triage,full  MODELS=all|lista
#          RUN_ARGS="--lanes 1 --fail-fast"  (dodatkowe opcje run_all.py)
# Pojedynczy model ręcznie: make train MODEL=xgboost ARGS="--tune --profile quick"

PYTHON ?= python
PROFILE ?= standard
FEATURES ?= triage
MODELS ?= all
MODEL ?= xgboost
ARGS ?=
RUN_ARGS ?=

RUN_ALL = $(PYTHON) scripts/run_all.py --feature-set $(FEATURES) --models $(MODELS) $(RUN_ARGS)

.PHONY: install install-app install-dev check-env all smoke quick thorough status dry-run \
        convert prepare prepare-site train train-all tune compare eval explain cv demo mlflow \
        test test-fast lint format clean clean-all

# --- Instalacja ---
install:
	$(PYTHON) -m pip install -r requirements-lock.txt

install-app: install
	$(PYTHON) -m pip install -r requirements-app.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

# --- Jedno polecenie (scripts/run_all.py) ---
check-env:
	$(PYTHON) scripts/00_check_env.py

all:
	$(RUN_ALL) --profile $(PROFILE)

smoke:
	$(RUN_ALL) --profile smoke

quick:
	$(RUN_ALL) --profile quick

thorough:
	$(RUN_ALL) --profile thorough

status:
	$(RUN_ALL) --profile $(PROFILE) --status

dry-run:
	$(RUN_ALL) --profile $(PROFILE) --dry-run

# --- Kroki pojedynczo ---
convert:
	$(PYTHON) scripts/01_convert_rdata.py

prepare:
	$(PYTHON) scripts/02_preprocess.py --split random

# Walidacja zewnętrzna: oddział B w całości jako test
prepare-site:
	$(PYTHON) scripts/02_preprocess.py --split site

train:
	$(PYTHON) scripts/03_train.py --model $(MODEL) --feature-set $(FEATURES) $(ARGS)

# Wszystkie modele w jednym procesie, parametry domyślne (bez tuningu) — szybki punkt odniesienia
train-all:
	$(PYTHON) scripts/03_train.py --model all --feature-set $(FEATURES) $(ARGS)

tune:
	$(PYTHON) scripts/03_train.py --model $(MODEL) --feature-set $(FEATURES) --tune --profile $(PROFILE) $(ARGS)

# --- Wyniki (tylko VAL; test — raz, na końcu) ---
compare:
	$(PYTHON) scripts/07_compare_runs.py --feature-set $(FEATURES)

eval:
	$(PYTHON) scripts/04_evaluate.py --model-name $(MODEL) --feature-set $(FEATURES)

explain:
	$(PYTHON) scripts/05_explain.py --model-name $(MODEL) --n-patients 5

cv:
	$(PYTHON) scripts/06_cross_validate.py --model $(MODEL) --feature-set $(FEATURES)

demo:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui --backend-store-uri "$$($(PYTHON) -c 'import pathlib; print(pathlib.Path("mlruns").resolve().as_uri())')" --port 5000

# --- Jakość ---
test:
	$(PYTHON) -m pytest tests/ -q

test-fast:
	$(PYTHON) -m pytest tests/ -q -m "not slow"

lint:
	@command -v ruff > /dev/null && ruff check src/ tests/ scripts/ app/ || echo "Zainstaluj: pip install -r requirements-dev.txt"

format:
	@command -v black > /dev/null && black src/ tests/ scripts/ app/ || echo "Zainstaluj: pip install -r requirements-dev.txt"

# --- Sprzątanie ---
# Odcisk danych (DONE.json, nazwy badań Optuny) pomija czas i ścieżkę przygotowania, więc ponowne
# przygotowanie TYCH SAMYCH danych tym samym kodem daje ten sam odcisk — runy i badania pozostają ważne.
clean:
	@echo "Usuwam logi (bez logs/optuna), wyniki results/ i przygotowane splity data/processed/esi."
	@echo "Zostają: runy w runs/ i badania Optuny. Następny trening przygotuje dane od nowa (~3-5 min);"
	@echo "przy tych samych danych i kodzie odcisk się nie zmieni, więc gotowe runy nadal będą pomijane."
	rm -rf logs/training/* logs/experiments/* logs/evaluation/* logs/run_all/*
	rm -rf results/figures/* results/tables/*
	rm -rf data/processed/esi
	rm -rf mlruns/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf runs/ logs/
	@echo "Usunięto także wszystkie runy (modele, predykcje, leaderboard) i badania Optuny."
