# SOR-AI: trening modeli klasyfikacji ESI — skróty dla Linuksa/macOS (make + bash).
# Na Windows uruchamiaj skrypty wprost: python scripts/run_all.py … (training/README.md).
#
# Najczęściej:
#   make install      — przetestowane wersje zależności (requirements-lock.txt)
#   make check-env    — kontrola środowiska (GPU, RAM, dysk, wersje, dane)
#   make smoke        — 5–10 min: cały pipeline na próbce (test instalacji)
#   make all          — nocny przebieg: tuning + trening 6 modeli, ewaluacja, porównanie (wznawialny)
#   make status       — stan przebiegu z plików-znaczników
#   make analyses     — po treningu: analizy do pracy (seedy, leave-site-out, krzywa uczenia; wznawialne)
#   make final-report — raport końcowy do pracy na TEST (raz, dla receptury wybranej na VAL)
#   make explain-global MODEL=ebm — SHAP globalny, zależności, krzywe kształtu EBM (runs/<run>/figures/global)
#   make llm-bakeoff  — porównanie modeli LLM do wyjaśnień (docs/LLM_BAKEOFF.md; Ollama po `ollama signin`)
#   make tfm-setup    — TFM (docs/TFM.md): torch/CUDA, licencja TabPFN, wagi, benchmark GPU (po treningu klasycznym)
#   make tfm-train PROFILE=thorough — trening TabPFN-3.5 + TabICLv2 (run_all --models tfm; wznawialny)
#   make small-data TAG=thorough    — eksperyment A: TFM vs strojony LightGBM przy małym n
#   make tfm-blend TAG=thorough     — eksperyment B: LightGBM + TFM na VAL z bramką → recommendation.json
#   make test         — testy
#
# Zmienne: PROFILE=standard|thorough|quick|smoke  FEATURES=triage|full|triage,full  MODELS=all|lista
#          RUN_ARGS="--lanes 1 --fail-fast"  (dodatkowe opcje run_all.py)
# Pojedynczy model ręcznie: make train MODEL=xgboost ARGS="--tune --profile quick"
# Raport końcowy: make final-report TAG=standard WEIGHTS='{"xgboost":0.4,"catboost":0.3,"ebm":0.3}' \
#                   REPORT_ARGS="--method geometric --main-rule target --target-undertriage 0.15"
# Porównanie LLM: make llm-bakeoff PARTS=exams | RECIPE=runs/final_report__triage__<czas>/recipe.json \
#                   LLM_ARGS="--dry-run";  oceny: make llm-ratings RATINGS="oceny_1.csv oceny_2.csv" KEY=…/key.json
# TFM: opcje skryptów 13/14/15 w TFM_ARGS, np. make tfm-setup TFM_ARGS="--offline-check",
#      make small-data TAG=thorough TFM_ARGS="--plan", make tfm-blend TAG=thorough TFM_ARGS="--plan"

PYTHON ?= python
PROFILE ?= standard
FEATURES ?= triage
MODELS ?= all
MODEL ?= xgboost
ARGS ?=
RUN_ARGS ?=
TAG ?= standard
WEIGHTS ?= optimize
REPORT_ARGS ?=
PARTS ?= exams,esi,tools
RECIPE ?=
LLM_ARGS ?=
RATINGS ?=
KEY ?=
TFM_ARGS ?=

RUN_ALL = $(PYTHON) scripts/run_all.py --feature-set $(FEATURES) --models $(MODELS) $(RUN_ARGS)

.PHONY: install install-app install-dev check-env all smoke quick thorough status dry-run analyses \
        convert prepare prepare-site train train-all tune compare eval explain explain-global cv final-report \
        llm-bakeoff llm-ratings tfm-setup tfm-train small-data tfm-blend demo mlflow test test-fast lint format \
        clean clean-all

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

# Po treningu: 09 seedy → 10 leave-site-out → 11 krzywa uczenia (refity, bez tuningu; wznawialne)
analyses:
	$(RUN_ALL) --profile $(PROFILE) --stage analyses

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

# Wyjaśnienia globalne do pracy (C11): najnowszy run modelu MODEL
explain-global:
	$(PYTHON) scripts/05_explain.py --model-name $(MODEL) --global

# TEST — raz, na końcu: receptura (runy z tagu TAG, wagi, temperatura, reguła) wybrana wcześniej na VAL;
# domyślnie reguła główna target (cel niedotriażowania 15% w vc + siatka); wyniki `make analyses` wczytuje sama
final-report:
	$(PYTHON) scripts/08_final_report.py --feature-set $(FEATURES) --tag $(TAG) --weights '$(WEIGHTS)' $(REPORT_ARGS)

cv:
	$(PYTHON) scripts/06_cross_validate.py --model $(MODEL) --feature-set $(FEATURES)

# --- Porównanie modeli językowych (docs/LLM_BAKEOFF.md): egzaminy NASK + wyjaśnienia ESI + narzędzia ---
# Wznawialne (pamięć podręczna runs/llm_bakeoff/cache); egzaminy działają bez wytrenowanych modeli.
llm-bakeoff:
	$(PYTHON) scripts/12_llm_bakeoff.py run --parts $(PARTS) $(if $(RECIPE),--recipe $(RECIPE),) $(LLM_ARGS)

llm-ratings:
	$(PYTHON) scripts/12_llm_bakeoff.py score-ratings $(foreach f,$(RATINGS),--csv $(f)) $(if $(KEY),--key $(KEY),)

# --- Modele fundamentalne TFM (docs/TFM.md): TabPFN-3.5 + TabICLv2; instalacja PO treningu klasycznym ---
# Najpierw: torch z CUDA z indeksu PyTorch, potem pip install -r requirements-tfm.txt -c requirements-lock.txt
tfm-setup:
	$(PYTHON) scripts/13_tfm_setup.py $(TFM_ARGS)

# Trening TFM (wybór konfiguracji na holdoucie z TRAIN + predykcja VAL/TEST na GPU; wznawialny)
tfm-train:
	$(PYTHON) scripts/run_all.py --feature-set $(FEATURES) --models tfm --profile $(PROFILE) $(RUN_ARGS)

# Eksperyment A (krzywa małych danych); LightGBM liczony zawsze, TFM bez bibliotek/wag → pominięty
small-data:
	$(PYTHON) scripts/14_small_data_curve.py --feature-set $(FEATURES) --tag $(TAG) $(TFM_ARGS)

# Eksperyment B (tylko VAL): bramka i polecenie 08 w runs/analyses/tfm_blend__<cechy>__<tag>/recommendation.json
tfm-blend:
	$(PYTHON) scripts/15_tfm_blend.py --feature-set $(FEATURES) --tag $(TAG) $(TFM_ARGS)

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
