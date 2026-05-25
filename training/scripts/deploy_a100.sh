#!/usr/bin/env bash
set -euo pipefail

SERVER="${1:-}"  # user@ip
if [ -z "$SERVER" ]; then
    echo "Uzycie: $0 user@server-ip"
    exit 1
fi

echo "=== Pakowanie projektu ==="
rsync -avz --progress \
    --exclude=.venv --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache \
    --exclude=.idea --exclude='*.so' --exclude='*.pyc' \
    /mnt/c/users/piotr/Magisterka/ "$SERVER":~/Magisterka/

echo "=== Instalacja zależności na serwerze ==="
ssh "$SERVER" bash -s << 'SSHEOF'
    cd ~/Magisterka
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    echo "=== Gotowe ==="
    echo ""
    echo "Odpal XGBoost (uzyje 5 GPU na 5 foldow):"
    echo "  cd ~/Magisterka && source .venv/bin/activate"
    echo "  OMP_NUM_THREADS=1 python3 -m src.models.train --model xgboost --feature-set triage_only --tune --n-trials 300 --weights custom"
    echo ""
    echo "Odpal CatBoost (uzyje 5 GPU na 5 foldow):"
    echo "  cd ~/Magisterka && source .venv/bin/activate"
    echo "  OMP_NUM_THREADS=1 python3 -m src.models.train --model catboost --feature-set triage_only --tune --n-trials 300 --weights custom"
    echo ""
    echo "Odpal w tle (nohup):"
    echo "  nohup bash -c 'source .venv/bin/activate && OMP_NUM_THREADS=1 python3 -m src.models.train --model catboost --feature-set triage_only --tune --n-trials 300 --weights custom' > catboost_tune.log 2>&1 &"
SSHEOF
