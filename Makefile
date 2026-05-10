# SOR-AI: Makefile dla typowych operacji
# Użycie:
#   make install   — zainstaluj zależności
#   make convert   — krok 1: .RData → parquet
#   make preprocess — krok 2: preprocessing + split
#   make train     — krok 3: trening XGBoost
#   make tune      — trening z tuningiem Optuna
#   make eval      — ewaluacja najnowszego modelu
#   make explain   — wyjaśnienia dla 5 pacjentów
#   make demo      — uruchom Streamlit demo
#   make test      — pytest
#   make clean     — usuń modele, logi, processed data

.PHONY: install convert preprocess train train-all tune eval explain demo test clean lint

install:
	pip install -r requirements.txt

convert:
	python scripts/01_convert_rdata.py

preprocess:
	python scripts/02_preprocess.py

# Pojedyncze modele
train:
	python scripts/03_train.py --model xgboost --feature-set triage_only --weights custom

train-lgbm:
	python scripts/03_train.py --model lightgbm --feature-set triage_only --weights custom

train-rf:
	python scripts/03_train.py --model random_forest --feature-set triage_only --weights custom

train-ebm:
	python scripts/03_train.py --model ebm --feature-set triage_only --weights custom

train-stacking:
	python scripts/03_train.py --model stacking --feature-set triage_only --weights custom

# Wszystkie modele (do porównań)
train-all: train train-lgbm train-rf train-ebm

tune:
	python scripts/03_train.py --model xgboost --tune --n-trials 50 --feature-set triage_only

eval:
	python scripts/04_evaluate.py --model-name xgboost --feature-set triage_only

explain:
	python scripts/05_explain.py --model-name xgboost --n-patients 5

explain-ollama:
	python scripts/05_explain.py --model-name xgboost --n-patients 5 --use-ollama

cv:
	python scripts/06_cross_validate.py --model xgboost --k 10

demo:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui --backend-store-uri file://$(PWD)/mlruns --port 5000

# Pełen pipeline od zera
full-pipeline: convert preprocess train eval explain
	@echo "[+] Pełen pipeline zakończony!"

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -q --tb=line

lint:
	@which ruff > /dev/null && ruff check src/ tests/ scripts/ app/ || echo "Zainstaluj ruff: pip install ruff"

format:
	@which black > /dev/null && black src/ tests/ scripts/ app/ || echo "Zainstaluj black: pip install black"

clean:
	@echo "Usuwam modele, logi, processed data..."
	rm -rf logs/training/* logs/experiments/* logs/evaluation/*
	rm -rf models/*.joblib
	rm -rf results/figures/* results/tables/* results/reports/*
	rm -rf data/processed/*.parquet data/processed/*.csv
	rm -rf mlruns/
	rm -rf .pytest_cache/ __pycache__/ src/**/__pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "[+] Czyszczenie zakończone."

clean-all: clean
	rm -rf data/processed/yale_emmlc_raw.parquet
	rm -rf data/processed/yale_emmlc_processed.parquet
	@echo "[+] Czyszczenie pełne (raz z parquet)."
