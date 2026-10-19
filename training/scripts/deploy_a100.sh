#!/usr/bin/env bash
# Wysyła projekt na serwer GPU, instaluje zależności i podpowiada komendy treningu.
# Użycie: scripts/deploy_a100.sh user@server-ip
set -euo pipefail

SERVER="${1:-}"  # user@ip
if [ -z "$SERVER" ]; then
    echo "Uzycie: $0 user@server-ip"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Pakowanie projektu ==="
rsync -avz --progress \
    --exclude=.venv --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache \
    --exclude=.idea --exclude='*.so' --exclude='*.pyc' \
    --exclude=training/runs --exclude=training/logs --exclude=training/data/processed/esi \
    "$REPO_ROOT/" "$SERVER":~/Magisterka/

echo "=== Instalacja zależności na serwerze ==="
ssh "$SERVER" bash -s << 'SSHEOF'
    cd ~/Magisterka/training
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    echo "=== Gotowe ==="
    echo ""
    echo "Przygotowanie danych + trening wszystkich modeli (GPU wykrywane automatycznie):"
    echo "  cd ~/Magisterka/training && source .venv/bin/activate"
    echo "  python scripts/02_preprocess.py && python scripts/03_train.py --model all"
    echo ""
    echo "Z tuningiem Optuną, w tle:"
    echo "  nohup bash -c 'source .venv/bin/activate && python scripts/03_train.py --model all --tune' > tune_all.log 2>&1 &"
SSHEOF
