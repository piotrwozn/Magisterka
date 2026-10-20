"""
Orkiestrator nocnego przebiegu — jedno polecenie, wznawialne, Windows / Linux / macOS.

    python scripts/run_all.py                         # profil standard, zestaw triage, 6 modeli
    python scripts/run_all.py --profile smoke         # 5–10 min: test instalacji i całego pipeline'u
    python scripts/run_all.py --status                # stan z plików-znaczników, nic nie uruchamia

Kroki (każdy w OSOBNYM procesie — awaria natywna/CUDA/OOM zabija jeden krok, nie całą noc):
    1. preflight  — scripts/00_check_env.py (próba GPU, RAM, dysk, wersje) → preflight.json,
    2. prepare    — scripts/02_preprocess.py --split S, gdy splity nie są gotowe albo mają inny podział,
    3. dla każdego zestawu cech × modelu:
         trening  — scripts/03_train.py --model M --feature-set FS --tune --profile P --tag P
                    (pomijany, gdy istnieje runs/*_M_FS_P/DONE.json z tą samą wersją i odciskiem danych,
                    zwykły run profilu — nie `custom` — z tym samym trybem tuningu, a dla modelu
                    zaplanowanego na GPU: faktycznie trenowany na GPU (run po fallbacku GPU → CPU jest
                    powtarzany, gdy GPU działa); przerwany tuning wznawia Optuna — ponowne
                    uruchomienie kosztuje tylko brakujące próby),
         ewaluacja — scripts/04_evaluate.py --run <run> --split val (wykresy; pomijana, gdy jest
                    runs/<run>/figures/metrics_val.json),
    4. compare    — scripts/07_compare_runs.py --feature-set FS --runs <runy tego przebiegu>,
    5. podsumowanie — runs/run_all_summary__<fs>__<profil>__<czas>.md (TYLKO metryki VAL)
                    + logs/run_all/<czas>/status.json (PID orkiestratora → `--status` wie, czy żyje).

Tory (`--lanes auto`): 2, gdy co najmniej jeden model idzie na GPU (urządzenie 'gpu' z próby GPU
w preflight), ≥ 16 rdzeni fizycznych i ≥ 30 GB RAM (komputer z 32 GB raportuje ~31,8 GB) —
tor GPU (tylko modele na GPU: xgboost → catboost, SOR_AI_N_JOBS=6) i tor CPU (pozostałe modele,
także xgboost/catboost na CPU; pozostałe rdzenie fizyczne). Budżety wątków się nie nakładają
(nadsubskrypcja OpenMP = ~70× wolniej).

Procesy kroków: wyjście czyta osobny wątek, a tor czeka na zakończenie procesu — twarda awaria
kroku nie blokuje toru, nawet gdy proces-wnuk (loky/joblib) trzyma otwarty potok. POSIX: każdy
krok we własnej grupie procesów (po błędzie / przy przerwaniu zabijana cała grupa; Ctrl+C
przekazywany krokom jawnie). Windows: kroki dzielą konsolę, więc Ctrl+C dociera do nich samo;
zabicie obejmuje drzewo procesów (taskkill /T).

Ctrl+C (także SIGTERM / SIGHUP): żadnych nowych kroków, do 120 s na zakończenie bieżących (potem
terminate/kill; drugie Ctrl+C = natychmiast), podsumowanie i polecenie wznowienia (po `--force`:
z `--force` i tylko niedokończonymi modelami). xgboost/catboost z błędem GPU w logu są powtarzane
raz na CPU (SOR_AI_DEVICE_<MODEL>=cpu) — z budżetem wątków toru CPU, gdy ten tor już skończył.

Windows: na czas przebiegu wyłączone dławienie energii procesu (EcoQoS) i tryb QuickEdit konsoli
(kliknięcie w okno zatrzymywałoby wypisywanie, a przez potoki — trening).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────
PROJECT_ROOT: Path = config.PROJECT_ROOT
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

# Kolejność = src.models.MODEL_REGISTRY (bez importu modeli w procesie orkiestratora)
MODEL_ORDER: tuple[str, ...] = ("logreg", "lightgbm", "xgboost", "catboost", "random_forest", "ebm")
MODEL_ALIASES: dict[str, str] = {"lr": "logreg", "lgbm": "lightgbm", "xgb": "xgboost", "cb": "catboost",
                                 "rf": "random_forest"}
FEATURE_SETS: tuple[str, ...] = ("triage", "full")
FEATURE_SET_ALIASES: dict[str, str] = {"triage_only": "triage", "top": "triage"}
GPU_MODELS: tuple[str, ...] = ("xgboost", "catboost")
SPLITS: tuple[str, ...] = ("random", "site")

GPU_LANE_THREADS: int = 6             # wątki CPU toru GPU (przygotowanie danych, ES, predykcja)
LANES_MIN_PHYSICAL_CORES: int = 16
LANES_MIN_RAM_GB: float = 30.0        # nominalne 32 GB system raportuje jako ~31,7–31,9 GB

DONE_JSON: str = "DONE.json"          # = src.models.train.DONE_JSON (ostatni artefakt runa)
EVAL_MARKER: tuple[str, str] = ("figures", "metrics_val.json")
CHILD_GRACE_S: float = 120.0          # Ctrl+C: tyle czekamy na samodzielne zakończenie kroków
CHILD_KILL_S: float = 10.0            # po terminate() — tyle do kill()
READER_JOIN_S: float = 10.0           # po końcu procesu kroku: tyle czekamy na domknięcie jego wyjścia
BOOT_TIME_TOLERANCE_S: float = 120.0  # --status: inny czas startu systemu = był restart
LOG_TAIL_LINES: int = 40

# Zapas na trening finalny + ewaluację ponad limit czasu tuningu (pełne dane; szacunek ±50%)
FINAL_FIT_ALLOWANCE_S: dict[str, int] = {
    "logreg": 180, "lightgbm": 300, "xgboost": 300, "catboost": 600, "random_forest": 900, "ebm": 2700,
}
SAMPLED_FIT_ALLOWANCE_S: int = 60     # profile z próbką danych (smoke, quick)

# Linia logu wyglądająca na błąd GPU (CUDA, OOM karty, NCCL, wymuszone GPU bez karty)
_ERROR_LINE = re.compile(r"error|exception|traceback|nieudany|failed|błąd", re.IGNORECASE)
_GPU_TEXT = re.compile(r"cuda|out of memory|nccl|cublas|cudnn|\bgpu\b|device ordinal|nvml", re.IGNORECASE)
# catboost przy próbie GPU na maszynie bez karty: "Error 35 (…) ignored while obtaining device count" — nie błąd kroku
_IGNORED = re.compile(r"\bignored\b", re.IGNORECASE)

# Stany zadania modelu
PENDING, RUNNING, DONE, SKIPPED = "pending", "running", "done", "skipped"
FAILED, INTERRUPTED, NOT_STARTED = "failed", "interrupted", "not_started"
_OK_STATES = (DONE, SKIPPED)
STATUS_PL: dict[str, str] = {
    PENDING: "oczekuje", RUNNING: "w toku", DONE: "gotowy", SKIPPED: "gotowy wcześniej",
    FAILED: "NIEUDANY", INTERRUPTED: "przerwany", NOT_STARTED: "nieuruchomiony",
}
INTERRUPT_CODES: tuple[int, ...] = (130, -2, 0xC000013A)   # 130 = 03_train.py po Ctrl+C; -SIGINT; Windows


# ─────────────────────────────────────────
# Opcje i zadania
# ─────────────────────────────────────────
@dataclass
class Options:
    profile: str = config.DEFAULT_TUNING_PROFILE
    feature_sets: list[str] = field(default_factory=lambda: ["triage"])
    models: list[str] = field(default_factory=lambda: list(MODEL_ORDER))
    split: str = "random"
    tune: bool = True
    lanes: str = "auto"
    force: bool = False
    fail_fast: bool = False
    dry_run: bool = False
    status: bool = False
    skip_preflight: bool = False
    keep_awake: bool = True
    argv: list[str] = field(default_factory=list)    # do polecenia wznowienia

    @property
    def tag(self) -> str:
        return run_tag(self.profile, self.tune)


@dataclass
class Lane:
    name: str                 # "gpu" | "cpu" | "main"
    models: list[str]
    n_jobs: int


@dataclass
class ModelTask:
    fs: str
    model: str
    lane: str
    n_jobs: int
    status: str = PENDING
    eval_status: str = PENDING
    run_name: str | None = None
    run_dir: str | None = None
    rc: int | None = None
    retried_on_cpu: bool = False
    retry_n_jobs: int | None = None    # wątki powtórki na CPU (budżet toru CPU albo GPU)
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    eta_s: float | None = None
    logs: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def key(self) -> str:
        return f"{self.fs}/{self.model}"


# ─────────────────────────────────────────
# Funkcje czyste (testowane osobno)
# ─────────────────────────────────────────
def parse_models(spec: str) -> list[str]:
    """'all' albo lista po przecinku (aliasy: lr, lgbm, xgb, cb, rf) → modele w kolejności MODEL_ORDER."""
    if spec.strip().lower() in ("", "all"):
        return list(MODEL_ORDER)
    chosen: set[str] = set()
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        name = MODEL_ALIASES.get(name, name)
        if name not in MODEL_ORDER:
            raise ValueError(f"Nieznany model '{raw.strip()}'. Dostępne: {', '.join(MODEL_ORDER)}")
        chosen.add(name)
    return [m for m in MODEL_ORDER if m in chosen]


def parse_feature_sets(spec: str) -> list[str]:
    """'triage', 'full' albo 'triage,full' (aliasy starych nazw) — bez duplikatów, kolejność zachowana."""
    out: list[str] = []
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        name = FEATURE_SET_ALIASES.get(name, name)
        if name not in FEATURE_SETS:
            raise ValueError(f"Nieznany zestaw cech '{raw.strip()}'. Dostępne: {', '.join(FEATURE_SETS)}")
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("Podaj co najmniej jeden zestaw cech")
    return out


def run_tag(profile: str, tune: bool) -> str:
    """
    Tag runów: nazwa profilu (tuning) albo '<profil>-notune' (parametry domyślne) — jak
    `src.models.train.default_run_tag` dla runu bez nadpisań ('<profil>-custom' orkiestrator nie tworzy).
    """
    return profile if tune else f"{profile}-notune"


def data_fingerprint(manifest: Path | dict | None = None) -> str:
    """
    Odcisk przygotowanych danych = `src.data.dataset.data_fingerprint` (bez kluczy zmiennych przy
    każdym przygotowaniu, np. `created`) — ten sam w DONE.json, nazwach badań Optuny i tutaj.
    Import leniwy: moduł orkiestratora (i 00_check_env.py) nie ładuje pandas przy starcie.
    """
    from src.data.dataset import data_fingerprint as _fingerprint

    return _fingerprint(manifest)


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_gpu_device(device: Any) -> bool:
    """'gpu' (catboost) albo 'cuda' / 'cuda:0' (xgboost) = trening na karcie."""
    text = str(device or "").strip().lower()
    return text == "gpu" or text.startswith("cuda")


def done_device(done: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """(faktyczne urządzenie, powód fallbacku) z DONE.json — nowy format `device`, stary `actual_device`."""
    device = done.get("device") if isinstance(done.get("device"), dict) else {}
    return device.get("actual") or done.get("actual_device"), device.get("fallback_reason")


def find_done_run(
    runs_dir: Path,
    model: str,
    feature_set: str,
    tag: str,
    data_version: str,
    fingerprint: str,
    tuned: bool | None = None,
    require_gpu: bool = False,
    since: float | None = None,
    notes: list[str] | None = None,
) -> Path | None:
    """
    Najnowszy UKOŃCZONY run profilu `runs/*_<model>_<fs>_<tag>/DONE.json` zgodny z bieżącymi
    danymi (wersja + odcisk `data_fingerprint`). None = trzeba trenować.

    Odrzuca też runy:
        - niestandardowe (`custom` w DONE.json: nadpisane budżety, próbki, wagi, cel, filtr …)
          — nawet z tym samym tagiem nie zastępują zwykłego runa profilu,
        - z innym trybem tuningu niż `tuned` (None = dowolny),
        - require_gpu=True (model zaplanowany na GPU): wytrenowane na CPU (fallback GPU → CPU) —
          gdy GPU działa, krok jest powtarzany na GPU,
        - since (time.time()): z DONE.json starszym niż ta chwila (szukanie runa właśnie
          zakończonego kroku, nie wcześniejszego).
    notes : lista, do której trafia opis odrzuconych, poza tym zgodnych runów (do logu).
    """
    candidates = sorted(Path(runs_dir).glob(f"*_{model}_{feature_set}_{tag}/{DONE_JSON}"), reverse=True)
    for done_path in candidates:
        done = read_json(done_path)
        if not done:
            continue
        if not (done.get("model") == model and done.get("feature_set") == feature_set
                and (done.get("tag") or "") == tag
                and done.get("data_version") == data_version
                and done.get("data_fingerprint") == fingerprint):
            continue
        name = done_path.parent.name
        if since is not None:
            try:
                if done_path.stat().st_mtime < since:
                    continue
            except OSError:
                continue
        if done.get("custom"):
            if notes is not None:
                overrides = ", ".join(done.get("overrides") or []) or "?"
                notes.append(f"{name}: run niestandardowy (nadpisania: {overrides}) — nie zastępuje runa profilu")
            continue
        if tuned is not None and bool(done.get("tuned")) != tuned:
            if notes is not None:
                notes.append(f"{name}: " + ("bez tuningu" if tuned else "z tuningiem") + " — inny tryb niż ten przebieg")
            continue
        if require_gpu:
            actual, reason = done_device(done)
            if not is_gpu_device(actual):
                if notes is not None:
                    notes.append(f"{name}: wytrenowany na {actual or '?'}"
                                 + (f" (fallback GPU → CPU: {reason})" if reason else "")
                                 + " — GPU teraz działa, trenuję ponownie na GPU")
                continue
        return done_path.parent
    return None


def eval_done(run_dir: Path) -> bool:
    return (Path(run_dir) / EVAL_MARKER[0] / EVAL_MARKER[1]).exists()


def gpu_planned_models(models: Sequence[str], devices: Mapping[str, str], gpu_ok: bool) -> list[str]:
    """
    Modele, które trafią na GPU: xgboost/catboost z urządzeniem 'gpu' (udana próba GPU w preflight
    albo SOR_AI_DEVICE_*=gpu) lub 'auto' (bez preflight — decyzja procesu treningu), gdy GPU
    wygląda na dostępne. Pozostałe (także xgboost/catboost na 'cpu') idą na CPU.
    """
    return [m for m in models if m in GPU_MODELS
            and (devices.get(m) == "gpu" or (devices.get(m) == "auto" and gpu_ok))]


def decide_lanes(
    requested: str,
    gpu_models: Sequence[str],
    physical_cores: int | None,
    ram_gb: float | None,
    models: Sequence[str],
) -> int:
    """
    Liczba torów. 2 tylko wtedy, gdy co najmniej jeden model idzie na GPU (`gpu_models`)
    i co najmniej jeden na CPU — inaczej jeden z torów byłby pusty. Wtedy '1' / '2' wprost,
    a 'auto' → 2 przy ≥ 16 rdzeniach fizycznych i ≥ 30 GB RAM.
    """
    on_gpu = [m for m in models if m in gpu_models]
    on_cpu = [m for m in models if m not in gpu_models]
    if not (on_gpu and on_cpu):
        return 1
    if requested in ("1", "2"):
        return int(requested)
    enough = (physical_cores or 0) >= LANES_MIN_PHYSICAL_CORES and (ram_gb or 0) >= LANES_MIN_RAM_GB
    return 2 if enough else 1


def assign_lanes(
    models: Sequence[str],
    n_lanes: int,
    total_threads: int,
    gpu_models: Sequence[str] = GPU_MODELS,
) -> list[Lane]:
    """
    1 tor: wszystkie modele w kolejności MODEL_ORDER, wszystkie wątki.
    2 tory: GPU (tylko modele z `gpu_models`, xgboost → catboost, min(6, połowa) wątków) i CPU
    (pozostałe w kolejności MODEL_ORDER — także xgboost/catboost na CPU; reszta wątków).
    Budżety rozłączne.
    """
    total = max(1, int(total_threads))
    ordered = [m for m in MODEL_ORDER if m in models]
    gpu = [m for m in GPU_MODELS if m in models and m in gpu_models]
    cpu = [m for m in ordered if m not in gpu]
    if n_lanes < 2 or not gpu or not cpu:
        return [Lane("main", ordered, total)]
    gpu_threads = max(1, min(GPU_LANE_THREADS, total // 2))
    return [Lane("gpu", gpu, gpu_threads), Lane("cpu", cpu, max(1, total - gpu_threads))]


def resolve_devices(env: Mapping[str, str], detected: Mapping[str, str] | None) -> dict[str, str]:
    """
    Urządzenie xgboost/catboost dla procesów treningu: jawne SOR_AI_DEVICE_<LIB> / SOR_AI_DEVICE
    użytkownika > wynik próby GPU z preflight ('gpu'/'cpu') > 'auto' (próbę zrobi proces treningu).
    """
    out: dict[str, str] = {}
    for lib in GPU_MODELS:
        explicit = (env.get(f"SOR_AI_DEVICE_{lib.upper()}") or env.get("SOR_AI_DEVICE") or "").strip().lower()
        if explicit:
            out[lib] = "gpu" if explicit == "cuda" else explicit
        elif detected and detected.get(lib) in ("gpu", "cpu"):
            out[lib] = str(detected[lib])
        else:
            out[lib] = "auto"
    return out


def build_child_env(base: Mapping[str, str], n_jobs: int, devices: Mapping[str, str]) -> dict[str, str]:
    """Środowisko procesu kroku: UTF-8, bez auto-przygotowania danych, wątki toru, urządzenia GPU."""
    env = dict(base)
    env.update({
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",            # wiersze logu na bieżąco w konsoli orkiestratora
        "SOR_AI_NO_AUTO_PREPARE": "1",
        "SOR_AI_N_JOBS": str(int(n_jobs)),
        "OMP_NUM_THREADS": str(int(n_jobs)),
    })
    for lib, device in devices.items():
        env[f"SOR_AI_DEVICE_{lib.upper()}"] = device
    return env


def python_cmd() -> list[str]:
    return [sys.executable, "-X", "utf8"]


def train_command(model: str, feature_set: str, profile: str, tag: str, tune: bool) -> list[str]:
    cmd = [*python_cmd(), str(SCRIPTS_DIR / "03_train.py"), "--model", model, "--feature-set", feature_set,
           "--profile", profile, "--tag", tag]
    return cmd + (["--tune"] if tune else [])


def eval_command(run_dir: Path | str) -> list[str]:
    return [*python_cmd(), str(SCRIPTS_DIR / "04_evaluate.py"), "--run", str(run_dir), "--split", "val"]


def compare_command(feature_set: str, run_names: Sequence[str], out_path: Path) -> list[str]:
    return [*python_cmd(), str(SCRIPTS_DIR / "07_compare_runs.py"), "--feature-set", feature_set,
            "--runs", *run_names, "--out", str(out_path)]


def prepare_command(split: str) -> list[str]:
    return [*python_cmd(), str(SCRIPTS_DIR / "02_preprocess.py"), "--split", split]


def preflight_command(out_dir: Path) -> list[str]:
    return [*python_cmd(), str(SCRIPTS_DIR / "00_check_env.py"), "--out-dir", str(out_dir)]


def log_has_gpu_error(text: str, tail_lines: int = 400) -> bool:
    """Czy koniec logu kroku zawiera linię błędu dotyczącą GPU (CUDA, OOM karty, NCCL, wymuszone GPU)."""
    for line in text.splitlines()[-tail_lines:]:
        if _ERROR_LINE.search(line) and _GPU_TEXT.search(line) and not _IGNORED.search(line):
            return True
    return False


def model_budget(model: str, profile: str, device: str) -> dict[str, Any]:
    """Budżet tuningu modelu z profilu (CPU_FALLBACK_BUDGETS dla xgboost/catboost na CPU)."""
    prof = config.TUNING_PROFILES[profile]
    budget = dict(prof[model])
    if model in GPU_MODELS and device == "cpu":
        budget.update(config.CPU_FALLBACK_BUDGETS.get(profile, {}).get(model, {}))
    return budget


def model_eta_s(model: str, profile: str, device: str, tune: bool) -> float:
    """Górne oszacowanie czasu modelu: limit czasu tuningu z profilu + trening finalny i ewaluacja."""
    sampled = config.TUNING_PROFILES[profile].get("train_sample") is not None
    allowance = SAMPLED_FIT_ALLOWANCE_S if sampled else FINAL_FIT_ALLOWANCE_S.get(model, 600)
    if not tune:
        return float(allowance)
    return float(model_budget(model, profile, device)["timeout_s"] + allowance)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def resume_command(argv: Sequence[str], redo_models: Sequence[str] | None = None) -> str:
    """
    Polecenie wznowienia: to samo bez jednorazowych flag (--dry-run, --status).

    Przebieg z `--force` (redo_models = modele, których trening NIE został ukończony w tym
    przebiegu): `--force` zostaje — wznowienie ma dalej trenować od nowa, a nie przyjąć stare runy —
    ale `--models` zawęża się do redo_models, żeby nie trenować drugi raz modeli już gotowych.
    Pusta lista → bez `--force` (zostały tylko ewaluacje / porównanie, które dokończy zwykłe
    wznowienie). redo_models=None → `--force` bez zmian (np. przerwanie przed planem).
    """
    args = [a for a in argv if a not in ("--dry-run", "--status")]
    if "--force" in args and redo_models is not None:
        if not redo_models:
            args = [a for a in args if a != "--force"]
        else:
            out: list[str] = []
            skip_value = False
            for a in args:
                if skip_value:
                    skip_value = False
                    continue
                if a == "--models":
                    skip_value = True
                    continue
                if a.startswith("--models="):
                    continue
                out.append(a)
            args = [*out, "--models", ",".join(redo_models)]
    return " ".join(["python", "scripts/run_all.py", *(shlex.quote(a) for a in args)])


# ─────────────────────────────────────────
# Procesy: grupy, zabijanie drzewa, czy orkiestrator żyje
# ─────────────────────────────────────────
def popen_group_kwargs() -> dict[str, Any]:
    """
    POSIX: krok we własnej sesji / grupie procesów — zabijany w całości (z procesami loky/joblib),
    a Ctrl+C z terminala do niego nie dociera (orkiestrator przekazuje SIGINT jawnie).
    Windows: wspólna konsola i grupa — CTRL_C_EVENT dociera do kroków sam (bez
    CREATE_NEW_PROCESS_GROUP, który by go wyłączył).
    """
    return {"start_new_session": True} if os.name == "posix" else {}


def kill_process_tree(proc: subprocess.Popen, sig: int | None = None) -> None:
    """
    Zabija proces kroku RAZEM z potomkami (best-effort, nigdy nie rzuca).

    POSIX: sygnał (domyślnie SIGKILL) do całej grupy procesów — działa także po śmierci samego
    kroku, gdy w grupie zostały procesy-wnuki trzymające potok. Windows: `taskkill /T /F`
    (drzewo; tylko gdy proces kroku jeszcze żyje), potem kill().
    """
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL if sig is None else sig)
        except OSError:                 # grupa już pusta / brak uprawnień
            pass
        return
    if proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=15, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.kill()
        except OSError:
            pass


def _boot_time() -> float | None:
    try:
        import psutil

        return float(psutil.boot_time())
    except Exception:  # noqa: BLE001 — brak psutil / platforma bez boot_time
        return None


def _process_create_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 — proces nie istnieje / brak psutil
        return None


def _pid_exists(pid: int) -> bool:
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except ImportError:
        pass
    if os.name == "nt":                  # os.kill(pid, 0) na Windows ZABIJA proces — tylko WinAPI
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong(0)
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259                  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def orchestrator_alive(status: Mapping[str, Any]) -> bool | None:
    """
    Czy orkiestrator ze status.json nadal działa. None = brak danych (status.json bez PID).

    False, gdy: od zapisu był restart (inny czas startu systemu), PID nie istnieje albo należy
    już do innego procesu (inny czas utworzenia procesu).
    """
    pid = status.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return None
    boot, boot_now = status.get("boot_time"), _boot_time()
    if isinstance(boot, (int, float)) and boot_now is not None and abs(float(boot) - boot_now) > BOOT_TIME_TOLERANCE_S:
        return False
    if not _pid_exists(pid):
        return False
    created, created_now = status.get("pid_create_time"), _process_create_time(pid)
    if isinstance(created, (int, float)) and created_now is not None and abs(float(created) - created_now) > 2.0:
        return False
    return True


def _tail(path: Path, n: int = LOG_TAIL_LINES) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def _fmt_metric(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return "—"
    if isinstance(value, (int, float)):
        return "—" if value != value else f"{value:.{digits}f}"   # NaN
    return str(value)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def run_metrics(run_dir: Path | str | None) -> dict[str, Any]:
    """Metryki VAL i opis tuningu runa z leaderboard_row.json + run_info.json (puste, gdy brak)."""
    if not run_dir:
        return {}
    row = read_json(Path(run_dir) / "leaderboard_row.json") or {}
    info = read_json(Path(run_dir) / "run_info.json") or {}
    tuning = info.get("tuning") or {}
    device = info.get("device") or {}
    return {
        **row,
        "n_complete": tuning.get("n_complete"),
        "n_pruned": tuning.get("n_pruned"),
        "n_fail": tuning.get("n_fail"),
        "stopped_reason": tuning.get("stopped_reason"),
        "fallback_reason": device.get("fallback_reason"),
        "best_iteration": info.get("best_iteration"),
        "iteration_cap": info.get("iteration_cap"),
    }


def optuna_progress(storage_dir: Path, model: str, feature_set: str, profile: str,
                    fingerprint: str) -> dict[str, int] | None:
    """
    Postęp badania Optuny modelu (bez importu optuny): liczba prób wg stanu dla badań
    `<model>__<fs>__…__data-<odcisk>__<profil>__…`. None = brak bazy / badania.
    """
    db = Path(storage_dir) / f"optuna_{model}.db"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=5)   # Windows: file:///C:/…
        try:
            rows = con.execute(
                "SELECT s.study_name, t.state, COUNT(t.trial_id) FROM studies s "
                "LEFT JOIN trials t ON t.study_id = s.study_id GROUP BY s.study_id, t.state"
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    counts: dict[str, int] = {}
    found = False
    for name, state, n in rows:
        if not (name.startswith(f"{model}__{feature_set}__") and f"__data-{fingerprint}__{profile}__" in name):
            continue
        found = True
        if state is not None:
            counts[str(state).lower()] = counts.get(str(state).lower(), 0) + int(n)
    return counts if found else None


# ─────────────────────────────────────────
# Orkiestrator
# ─────────────────────────────────────────
Runner = Callable[[list[str], Mapping[str, str], Path, str], int]


class Orchestrator:
    """
    Jeden przebieg `run_all`. `runner` (testy) zastępuje uruchamianie procesów:
    runner(cmd, env, log_path, prefix) → kod wyjścia.
    """

    def __init__(self, opts: Options, runner: Runner | None = None, now: datetime | None = None) -> None:
        self.opts = opts
        self.runner: Runner = runner or self._run_child
        self.started = now or datetime.now()
        self.ts = self.started.strftime("%Y%m%d_%H%M%S")
        self.runs_dir = Path(config.RUNS_DIR)
        self.log_dir = Path(config.LOGS_DIR) / "run_all" / self.ts
        self.stop = threading.Event()
        self.interrupted = False
        self.tasks: list[ModelTask] = []
        self.lanes: list[Lane] = []
        self.devices: dict[str, str] = {}
        self.gpu_models: list[str] = []
        self.preflight: dict[str, Any] = {}
        self.fingerprint = "brak-danych"
        self.prepare_status = PENDING
        self.compare: dict[str, dict[str, Any]] = {}
        self.summaries: list[Path] = []
        self.run_state = "running"         # running | finished | interrupted | failed (status.json)
        self.exit_status: int | None = None
        self._pid_create_time = _process_create_time(os.getpid())
        self._boot_time = _boot_time()
        self._children: set[subprocess.Popen] = set()
        self._lane_busy: dict[str, bool] = {}
        self._lock = threading.Lock()
        self._print_lock = threading.Lock()
        self._file_handler: logging.Handler | None = None

    # ── komunikaty ──
    def say(self, msg: str, level: int = logging.INFO) -> None:
        log.log(level, msg)

    def _echo(self, prefix: str, line: str) -> None:
        with self._print_lock:
            try:
                sys.stdout.write(f"[{prefix}] {line.rstrip()}\n")
                sys.stdout.flush()
            except (OSError, ValueError):
                pass

    # ── procesy ──
    def _run_child(self, cmd: list[str], env: Mapping[str, str], log_path: Path, prefix: str) -> int:
        """
        Uruchamia krok: wyjście do pliku logu (UTF-8) i na konsolę z prefiksem.

        Wyjście czyta osobny wątek (`_pump`), a ten wątek czeka na zakończenie PROCESU — nie na
        koniec potoku. Po twardej awarii kroku potok może trzymać proces-wnuk (loky/joblib);
        wtedy (POSIX) zabijamy całą grupę procesów kroku, a gdy wyjście nadal się nie domyka,
        porzucamy czytnik po READER_JOIN_S (wątek demoniczny) — tor idzie dalej.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "a", encoding="utf-8")   # zamyka go wątek czytający (_pump)
        try:
            fh.write(f"# {datetime.now().isoformat(timespec='seconds')} $ {subprocess.list2cmdline(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, env=dict(env), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                **popen_group_kwargs(),
            )
        except BaseException:
            fh.close()
            raise
        with self._lock:                # przed startem czytnika: Ctrl+C od tej chwili dotrze do kroku
            self._children.add(proc)
        if self.interrupted and os.name == "posix":     # Ctrl+C w trakcie Popen — przekaż od razu
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except OSError:
                pass
        reader = threading.Thread(target=self._pump, args=(proc, fh, prefix), daemon=True,
                                  name=f"wyjście {prefix}")
        reader.start()
        try:
            try:
                rc = proc.wait()
            except BaseException:       # Ctrl+C w wątku głównym (np. preflight) — bez sierot
                kill_process_tree(proc)
                raise
            if rc != 0:
                kill_process_tree(proc)         # POSIX: resztki grupy po awarii / przerwaniu
            reader.join(READER_JOIN_S)
            if reader.is_alive():               # potok trzyma proces potomny, który przeżył krok
                kill_process_tree(proc)
                reader.join(READER_JOIN_S / 2)
            if reader.is_alive():
                # Zamknięcie potoku teraz zablokowałoby się na wątku czytającym — porzucamy go
                self.say(f"[{prefix}] wyjście kroku nadal otwarte {READER_JOIN_S:.0f} s po zakończeniu procesu "
                         f"(trzyma je proces potomny) — pomijam resztę logu i idę dalej", logging.WARNING)
            return rc
        finally:
            with self._lock:
                self._children.discard(proc)

    def _pump(self, proc: subprocess.Popen, fh: Any, prefix: str) -> None:
        """Wątek czytający: wyjście kroku → plik logu + konsola; na końcu zamyka potok i plik."""
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                try:
                    fh.write(line)
                    fh.flush()
                except (OSError, ValueError):
                    pass
                self._echo(prefix, line)
        except (OSError, ValueError):
            pass
        finally:
            for stream in (proc.stdout, fh):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass

    def _live_children(self) -> list[subprocess.Popen]:
        with self._lock:
            return [p for p in self._children if p.poll() is None]

    def _forward_interrupt(self) -> None:
        """
        POSIX: kroki są we własnych grupach procesów, więc Ctrl+C z terminala do nich nie dociera —
        przekaż SIGINT całym grupom (03_train.py kończy się wtedy kodem 130). Windows: kroki
        dzielą konsolę i dostały CTRL_C_EVENT same.
        """
        if os.name != "posix":
            return
        for proc in self._live_children():
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except OSError:
                pass

    def _signal_children(self, kill: bool) -> None:
        for proc in self._live_children():
            if os.name == "posix":
                kill_process_tree(proc, signal.SIGKILL if kill else signal.SIGTERM)
            elif kill:
                kill_process_tree(proc)
            else:
                try:
                    proc.terminate()
                except OSError:
                    pass

    def _install_signal_handlers(self) -> Callable[[], None]:
        """
        SIGTERM / SIGHUP (kill, zamknięcie terminala) → jak Ctrl+C. Kroki we własnych grupach
        procesów nie dostaną tych sygnałów same, więc orkiestrator kończy je łagodnie i zapisuje
        stan. Sygnał ignorowany (np. SIGHUP pod `nohup`) zostaje ignorowany. Zwraca funkcję
        przywracającą poprzednie obsługi (tylko w wątku głównym).
        """
        if threading.current_thread() is not threading.main_thread():
            return lambda: None

        def handler(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt

        previous: dict[int, Any] = {}
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                if signal.getsignal(sig) is signal.SIG_IGN:     # np. `nohup run_all.py &` — zostaw
                    continue
                previous[sig] = signal.signal(sig, handler)
            except (OSError, ValueError):
                pass

        def restore() -> None:
            for sig, old in previous.items():
                try:
                    signal.signal(sig, old)
                except (OSError, ValueError, TypeError):
                    pass

        return restore

    def _wait(self, threads: list[threading.Thread]) -> None:
        """Czeka na wątki; Ctrl+C → łagodne zatrzymanie (patrz _handle_interrupt)."""
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(0.2)
        except KeyboardInterrupt:
            self._handle_interrupt(threads)

    def _handle_interrupt(self, threads: list[threading.Thread]) -> None:
        self.interrupted = True
        self.stop.set()
        self._forward_interrupt()
        self.say(f"Ctrl+C — nie uruchamiam nowych kroków; czekam do {CHILD_GRACE_S:.0f} s na zakończenie "
                 f"bieżących (ponowne Ctrl+C = natychmiast).", logging.WARNING)
        deadline = time.monotonic() + CHILD_GRACE_S
        try:
            while any(t.is_alive() for t in threads) and time.monotonic() < deadline:
                for t in threads:
                    t.join(0.2)
            if any(t.is_alive() for t in threads):
                self.say("Kroki nie zakończyły się same — terminate()", logging.WARNING)
                self._signal_children(kill=False)
                deadline = time.monotonic() + CHILD_KILL_S
                while any(t.is_alive() for t in threads) and time.monotonic() < deadline:
                    for t in threads:
                        t.join(0.2)
        except KeyboardInterrupt:
            pass
        if any(t.is_alive() for t in threads):
            self._signal_children(kill=True)
            for t in threads:
                t.join(5)

    def _run_jobs(self, jobs: list[Callable[[], None]]) -> None:
        def guard(job: Callable[[], None]) -> Callable[[], None]:
            def wrapped() -> None:
                try:
                    job()
                except Exception as exc:  # noqa: BLE001 — błąd orkiestratora w wątku nie może zawiesić nocy
                    log.exception(f"Błąd wewnętrzny orkiestratora: {type(exc).__name__}: {exc}")
            return wrapped

        threads = [threading.Thread(target=guard(job), daemon=True) for job in jobs]
        for t in threads:
            t.start()
        self._wait(threads)

    # ── przygotowanie ──
    def _setup_logging(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "run_all.log"
        get_logger(__name__, log_file=path)
        self._file_handler = next((h for h in log.handlers if getattr(h, "baseFilename", None)
                                   and Path(h.baseFilename).resolve() == path.resolve()), None)

    def _teardown_logging(self) -> None:
        if self._file_handler is not None:
            log.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def _base_env(self, n_jobs: int, devices: Mapping[str, str] | None = None) -> dict[str, str]:
        return build_child_env(os.environ, n_jobs, devices if devices is not None else self.devices)

    def _total_threads(self) -> int:
        from src.utils import hardware

        return hardware.n_jobs()

    def run_preflight(self) -> bool:
        """00_check_env.py w osobnym procesie (kontekst CUDA próby GPU nie zostaje w orkiestratorze)."""
        rc = self.runner(preflight_command(self.log_dir), self._base_env(self._total_threads(), {}),
                         self.log_dir / "preflight.log", "preflight")
        self.preflight = read_json(self.log_dir / "preflight.json") or {}
        if rc != 0:
            self.say(f"Preflight: błędy krytyczne (kod {rc}) — popraw je i uruchom ponownie "
                     f"(szczegóły: {self.log_dir / 'preflight.log'}; obejście: --skip-preflight).", logging.ERROR)
            return False
        return True

    def _gpu_hint(self) -> tuple[bool, dict[str, str] | None]:
        """(czy GPU działa, urządzenia z preflight) — bez preflight: obecność nvidia-smi."""
        facts = self.preflight.get("facts") or {}
        if facts:
            return bool(facts.get("gpu_ok")), facts.get("devices")
        from src.utils.hardware import nvidia_smi_info

        return nvidia_smi_info() is not None, None

    def _resources(self) -> tuple[int | None, float | None]:
        facts = self.preflight.get("facts") or {}
        if facts.get("physical_cores"):
            return facts.get("physical_cores"), facts.get("ram_gb")
        from src.utils import hardware

        summary = hardware.hardware_summary()
        return summary.get("physical_cores"), summary.get("ram_gb")

    def plan(self) -> None:
        """Urządzenia, tory (wg urządzeń modeli), wątki i zadania (z oznaczeniem już ukończonych)."""
        gpu_ok, detected = self._gpu_hint()
        self.devices = resolve_devices(os.environ, detected)
        self.gpu_models = gpu_planned_models(self.opts.models, self.devices, gpu_ok)
        physical, ram = self._resources()
        n_lanes = decide_lanes(self.opts.lanes, self.gpu_models, physical, ram, self.opts.models)
        if self.opts.lanes == "2" and n_lanes < 2:
            self.say("--lanes 2: żaden model nie idzie na GPU (albo wszystkie) — jeden tor", logging.WARNING)
        self.lanes = assign_lanes(self.opts.models, n_lanes, self._total_threads(), self.gpu_models)
        self.tasks = []
        for lane in self.lanes:
            for fs in self.opts.feature_sets:
                for model in lane.models:
                    device = "gpu" if model in self.gpu_models else "cpu"
                    task = ModelTask(fs=fs, model=model, lane=lane.name, n_jobs=lane.n_jobs,
                                     eta_s=model_eta_s(model, self.opts.profile, device, self.opts.tune))
                    self.tasks.append(task)
        self.refresh_done()

    def _requires_gpu(self, model: str) -> bool:
        """Model zaplanowany na GPU (próba GPU udana / wymuszone GPU) — run z CPU nie wystarczy."""
        return self.devices.get(model) == "gpu"

    def refresh_done(self) -> None:
        """Zadania z pasującym DONE.json → SKIPPED (chyba że --force); powód ponownego treningu w logu."""
        from src.utils.config import DATA_VERSION

        self.fingerprint = data_fingerprint()
        for task in self.tasks:
            if task.status not in (PENDING, SKIPPED):
                continue
            notes: list[str] = []
            run_dir = find_done_run(self.runs_dir, task.model, task.fs, self.opts.tag, DATA_VERSION,
                                    self.fingerprint, tuned=self.opts.tune,
                                    require_gpu=self._requires_gpu(task.model), notes=notes)
            if run_dir is not None and not self.opts.force:
                task.status, task.run_dir, task.run_name = SKIPPED, str(run_dir), run_dir.name
                task.eval_status = SKIPPED if eval_done(run_dir) else PENDING
            else:
                task.status, task.run_dir, task.run_name = PENDING, None, None
                if not self.opts.force and run_dir is None:
                    for note in notes[:3]:
                        self.say(f"{task.key}: {note}", logging.WARNING)

    def print_plan(self) -> None:
        o = self.opts
        self.say("═" * 72)
        self.say(f" run_all · profil '{o.profile}' · zestawy {', '.join(o.feature_sets)} · podział '{o.split}' · "
                 f"{'tuning Optuną' if o.tune else 'bez tuningu (parametry domyślne)'} · tag '{o.tag}'")
        self.say("═" * 72)
        self.say("Urządzenia: " + ", ".join(f"{k}={v}" for k, v in self.devices.items())
                 + f" | tory: {len(self.lanes)} — " + "; ".join(
                     f"{lane.name}: {' → '.join(lane.models)} (wątki: {lane.n_jobs})" for lane in self.lanes))
        lane_totals: dict[str, float] = {}
        for task in self.tasks:
            state = ("pomijam — gotowy run " + str(task.run_name)) if task.status == SKIPPED else \
                f"do zrobienia, ETA ≤ {format_duration(task.eta_s)}"
            if task.status == SKIPPED and task.eval_status == PENDING:
                state += " (brak ewaluacji — zostanie dorobiona)"
            self.say(f"  {task.fs:7s} {task.model:14s} [{task.lane}] {state}")
            if task.status == PENDING:
                lane_totals[task.lane] = lane_totals.get(task.lane, 0.0) + float(task.eta_s or 0)
        if lane_totals:
            self.say(f"Szacowany czas (górna granica z limitów profilu, ±50%): "
                     f"{format_duration(max(lane_totals.values()))}"
                     + ("" if len(lane_totals) < 2 else
                        " (" + ", ".join(f"tor {k}: {format_duration(v)}" for k, v in lane_totals.items()) + ")"))
        if not self.opts.dry_run:
            self.say(f"Logi: {self.log_dir}")

    def prepare(self) -> bool:
        """Splity gotowe i z tym samym podziałem → pomiń; inaczej 02_preprocess.py."""
        from src.data.dataset import is_prepared

        manifest = read_json(Path(config.DATA_MANIFEST_JSON)) or {}
        if is_prepared() and manifest.get("split_strategy") == self.opts.split:
            self.prepare_status = SKIPPED
            self.say(f"Dane: gotowe ({manifest.get('data_version')}, podział '{self.opts.split}', "
                     f"odcisk {data_fingerprint()}) — pomijam przygotowanie")
            return True
        if self.opts.dry_run:
            self.prepare_status = PENDING
            self.say(f"Dane: zostaną przygotowane: {subprocess.list2cmdline(prepare_command(self.opts.split))}")
            return True
        self.say(f"Dane: przygotowanie (podział '{self.opts.split}', ~3–5 min, szczyt ~11 GB RAM)…")
        self.prepare_status = RUNNING
        result: dict[str, int] = {}

        def job() -> None:
            result["rc"] = self.runner(prepare_command(self.opts.split), self._base_env(self._total_threads()),
                                       self.log_dir / "prepare.log", "prepare")

        self._run_jobs([job])
        if self.interrupted:
            self.prepare_status = INTERRUPTED
            return False
        if result.get("rc") != 0:
            self.prepare_status = FAILED
            self.say(f"Przygotowanie danych NIEUDANE (kod {result.get('rc')}) — {self.log_dir / 'prepare.log'}",
                     logging.ERROR)
            return False
        self.prepare_status = DONE
        return True

    # ── kroki modelu ──
    def _step_log(self, task: ModelTask, step: str) -> Path:
        path = self.log_dir / f"{task.fs}__{task.model}__{step}.log"
        task.logs.append(str(path))
        return path

    def _prefix(self, task: ModelTask) -> str:
        return task.model if len(self.opts.feature_sets) == 1 else f"{task.model}·{task.fs}"

    def run_task(self, task: ModelTask) -> None:
        from src.utils.config import DATA_VERSION

        t0 = time.time()
        task.started_at = datetime.now().isoformat(timespec="seconds")
        if task.status == PENDING:
            task.status = RUNNING
            self._write_status()
            self.say(f"▶ {task.key}: trening (ETA ≤ {format_duration(task.eta_s)}, wątki: {task.n_jobs}, "
                     f"tor {task.lane})")
            env = self._base_env(task.n_jobs)
            cmd = train_command(task.model, task.fs, self.opts.profile, self.opts.tag, self.opts.tune)
            log_path = self._step_log(task, "train")
            rc = self.runner(cmd, env, log_path, self._prefix(task))
            if (rc != 0 and not self.stop.is_set() and task.model in GPU_MODELS
                    and env.get(f"SOR_AI_DEVICE_{task.model.upper()}") != "cpu"
                    and log_has_gpu_error(_tail(log_path, 400))):
                threads = self._cpu_retry_threads(task)
                self.say(f"{task.key}: błąd GPU w logu — ponawiam RAZ na CPU "
                         f"(SOR_AI_DEVICE_{task.model.upper()}=cpu, budżet zastępczy, wątki: {threads})",
                         logging.WARNING)
                task.retried_on_cpu, task.retry_n_jobs = True, threads
                env = {**self._base_env(threads), f"SOR_AI_DEVICE_{task.model.upper()}": "cpu"}
                log_path = self._step_log(task, "train_cpu")
                rc = self.runner(cmd, env, log_path, self._prefix(task) + "·cpu")
            task.rc = rc
            if rc in INTERRUPT_CODES:       # Ctrl+C dotarło do procesu kroku — nie zaczynaj kolejnych
                self.interrupted = True
                self.stop.set()
            run_dir = None
            if rc == 0:                     # run tego kroku (także po fallbacku na CPU), nie starszy
                run_dir = find_done_run(self.runs_dir, task.model, task.fs, self.opts.tag, DATA_VERSION,
                                        self.fingerprint, tuned=self.opts.tune, since=t0 - 2.0)
            if run_dir is not None:
                task.status, task.run_dir, task.run_name = DONE, str(run_dir), run_dir.name
                actual, _ = done_device(read_json(run_dir / DONE_JSON) or {})
                if self._requires_gpu(task.model) and not is_gpu_device(actual):
                    self.say(f"{task.key}: wytrenowany na CPU mimo planu GPU — następne uruchomienie powtórzy "
                             f"go na GPU, jeśli karta będzie działać", logging.WARNING)
            elif self.interrupted:
                task.status = INTERRUPTED
            else:
                task.status = FAILED
                task.error = (f"kod wyjścia {rc}" if rc != 0 else "kod 0, ale brak DONE.json") \
                    + f" — log: {log_path}"
                self.say(f"✗ {task.key}: NIEUDANY ({task.error})", logging.ERROR)
        if task.status in _OK_STATES and task.run_dir and (self.opts.force or not eval_done(Path(task.run_dir))):
            if self.stop.is_set():
                task.eval_status = NOT_STARTED
            else:
                self._write_status()
                rc = self.runner(eval_command(task.run_dir), self._base_env(task.n_jobs),
                                 self._step_log(task, "eval"), self._prefix(task) + "·eval")
                task.eval_status = DONE if rc == 0 else (INTERRUPTED if self.stop.is_set() else FAILED)
                if task.eval_status == FAILED:
                    self.say(f"✗ {task.key}: ewaluacja nieudana (kod {rc})", logging.ERROR)
        elif task.status in _OK_STATES and task.eval_status == PENDING:
            task.eval_status = SKIPPED
        elif task.status not in _OK_STATES:
            task.eval_status = NOT_STARTED
        task.finished_at = datetime.now().isoformat(timespec="seconds")
        task.duration_s = round(time.time() - t0, 1)
        if task.status == DONE:
            self.say(f"✓ {task.key}: gotowe w {format_duration(task.duration_s)} → {task.run_name}")
        self._write_status()

    def _cpu_retry_threads(self, task: ModelTask) -> int:
        """
        Wątki powtórki na CPU po błędzie GPU (zadanie z toru GPU): budżet toru CPU, gdy ten tor już
        skończył swoje modele; gdy jest zajęty — budżet toru GPU (inaczej wątki by się nakładały),
        z wyraźnym ostrzeżeniem. Jeden tor: bez zmian (wszystkie wątki).
        """
        cpu_lane = next((lane for lane in self.lanes if lane.name == "cpu"), None)
        if task.lane != "gpu" or cpu_lane is None:
            return task.n_jobs
        with self._lock:
            busy = self._lane_busy.get("cpu", True)
        if not busy:
            self.say(f"{task.key}: tor CPU skończył pracę — powtórka na CPU dostaje jego budżet "
                     f"({cpu_lane.n_jobs} wątków)")
            return cpu_lane.n_jobs
        self.say(f"{task.key}: tor CPU jest zajęty — powtórka na CPU działa z budżetem toru GPU "
                 f"({task.n_jobs} wątków zamiast {cpu_lane.n_jobs}), więc będzie wyraźnie wolniejsza; "
                 f"limit czasu tuningu (budżet zastępczy CPU) i tak ją ogranicza", logging.WARNING)
        return task.n_jobs

    def _lane_worker(self, lane: Lane) -> None:
        with self._lock:
            self._lane_busy[lane.name] = True
        try:
            for task in [t for t in self.tasks if t.lane == lane.name]:
                if self.stop.is_set():
                    if task.status == PENDING:
                        task.status, task.eval_status = NOT_STARTED, NOT_STARTED
                    continue
                try:
                    self.run_task(task)
                except Exception as exc:  # noqa: BLE001 — błąd orkiestratora przy jednym modelu nie zatrzymuje toru
                    log.exception(f"{task.key}: błąd wewnętrzny orkiestratora: {type(exc).__name__}: {exc}")
                    task.status, task.error = FAILED, f"błąd orkiestratora: {type(exc).__name__}: {exc}"
                    if task.eval_status == PENDING:
                        task.eval_status = NOT_STARTED
                    self._write_status()
                if task.status == FAILED and self.opts.fail_fast:
                    self.say("--fail-fast: zatrzymuję dalsze kroki", logging.WARNING)
                    self.stop.set()
        finally:
            with self._lock:
                self._lane_busy[lane.name] = False

    def run_compare(self) -> None:
        for fs in self.opts.feature_sets:
            names = [t.run_name for t in self.tasks if t.fs == fs and t.status in _OK_STATES and t.run_name]
            if not names:
                self.compare[fs] = {"status": NOT_STARTED, "note": "brak ukończonych runów"}
                continue
            out = self.runs_dir / f"comparison__{fs}__{self.opts.profile}__{self.ts}.md"
            result: dict[str, int] = {}

            def job(fs: str = fs, names: list[str] = names, out: Path = out) -> None:
                result["rc"] = self.runner(compare_command(fs, names, out), self._base_env(self._total_threads()),
                                           self.log_dir / f"{fs}__compare.log", f"compare·{fs}")

            self._run_jobs([job])
            ok = result.get("rc") == 0
            self.compare[fs] = {"status": DONE if ok else (INTERRUPTED if self.interrupted else FAILED),
                                "report": str(out) if ok and out.exists() else None,
                                "log": str(self.log_dir / f"{fs}__compare.log")}
            if self.interrupted:
                break

    # ── stan i podsumowanie ──
    def _status_payload(self) -> dict[str, Any]:
        return {
            "started_at": self.started.isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "state": self.run_state,
            "exit_code": self.exit_status,
            # --status: czy ten orkiestrator jeszcze działa (PID + czas jego utworzenia + start systemu)
            "pid": os.getpid(),
            "pid_create_time": self._pid_create_time,
            "boot_time": self._boot_time,
            "options": {k: v for k, v in asdict(self.opts).items() if k != "argv"},
            "tag": self.opts.tag,
            "data_version": config.DATA_VERSION,
            "data_fingerprint": self.fingerprint,
            "prepare": self.prepare_status,
            "devices": self.devices,
            "lanes": [asdict(lane) for lane in self.lanes],
            "tasks": [asdict(t) for t in self.tasks],
            "compare": self.compare,
            "interrupted": self.interrupted,
            "summaries": [str(p) for p in self.summaries],
            "log_dir": str(self.log_dir),
        }

    def _write_status(self) -> None:
        from src.utils.fileio import write_text_atomic

        with self._lock:
            payload = json.dumps(self._status_payload(), indent=2, ensure_ascii=False, default=str)
            try:
                write_text_atomic(self.log_dir / "status.json", payload)
            except OSError as exc:
                log.warning(f"Nie zapisano status.json: {exc}")

    def write_summary(self, fs: str) -> Path:
        """runs/run_all_summary__<fs>__<profil>__<czas>.md — tylko metryki VAL."""
        o = self.opts
        ended = datetime.now()
        tasks = [t for t in self.tasks if t.fs == fs]
        facts = self.preflight.get("facts") or {}
        hw = facts.get("hardware") or {}
        smi = hw.get("nvidia_smi") or facts.get("nvidia_smi") or {}
        counts = {s: sum(t.status == s for t in tasks) for s in (DONE, SKIPPED, FAILED, INTERRUPTED, NOT_STARTED)}
        lines = [
            f"# run_all — {fs} · profil {o.profile}",
            "",
            f"- start {self.started:%Y-%m-%d %H:%M}, koniec {ended:%Y-%m-%d %H:%M}, "
            f"czas {format_duration((ended - self.started).total_seconds())}"
            + (" — **PRZERWANO (Ctrl+C)**" if self.interrupted else ""),
            f"- dane `{config.DATA_VERSION}`, odcisk `{self.fingerprint}`, podział `{o.split}`; tag runów `{o.tag}`; "
            + ("tuning Optuną (cel: log loss, CV na TRAIN)" if o.tune else "bez tuningu (parametry domyślne)"),
            "- tory: " + "; ".join(f"{lane.name}: {' → '.join(lane.models)} (wątki: {lane.n_jobs})"
                                   for lane in self.lanes),
            "- urządzenia (żądane): " + ", ".join(f"{k} `{v}`" for k, v in self.devices.items())
            + (f"; GPU {smi.get('name')}, sterownik {smi.get('driver_version')}" if smi else "; brak GPU NVIDIA")
            + (f"; CPU {hw.get('cpu_name')}, {hw.get('physical_cores')} rdzeni, RAM {hw.get('ram_gb')} GB"
               if hw else ""),
            f"- commit `{_git_commit() or '?'}`; logi `{self.log_dir}`",
            f"- stan: gotowe {counts[DONE]}, wcześniej gotowe (pominięte) {counts[SKIPPED]}, "
            f"nieudane {counts[FAILED]}, przerwane {counts[INTERRUPTED]}, nieuruchomione {counts[NOT_STARTED]}",
            "",
            "> **Wyłącznie metryki VAL** — do wyboru modeli, wag łączenia i progów. Zbiór testowy oceniasz "
            "raz, na końcu, dla wybranej receptury (src.ensemble).",
            "",
            "## Modele (VAL)",
            "",
            "| model | stan | urządzenie | próby (ukoń./przyc./błędy) | CV | QWK argmax | QWK koszt | "
            "undertriage (koszt) | QWK vc argmax | QWK vc koszt | czas | run |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for t in tasks:
            m = run_metrics(t.run_dir)
            device = m.get("actual_device") or "—"
            if m.get("device_fallback") or m.get("fallback_reason"):
                device += " (fallback GPU→CPU)"
            if t.retried_on_cpu:
                device += " (powtórka na CPU" + (f", {t.retry_n_jobs} wątków" if t.retry_n_jobs else "") + ")"
            trials = "—" if m.get("n_complete") is None else f"{m['n_complete']}/{m['n_pruned']}/{m['n_fail']}"
            cv = "—" if m.get("cv_score") is None else (
                f"{m.get('cv_metric')} {_fmt_metric(m['cv_score'])}"
                + (f" ± {_fmt_metric(m['cv_std'])}" if m.get("cv_std") is not None else ""))
            status = STATUS_PL.get(t.status, t.status)
            if t.status in _OK_STATES and t.eval_status not in (DONE, SKIPPED, PENDING):
                status += f" (ewaluacja: {STATUS_PL.get(t.eval_status, t.eval_status)})"
            run = f"[`{t.run_name}`]({t.run_name}/report.md)" if t.run_name else "—"
            lines.append(
                f"| {t.model} | {status} | {device} | {trials} | {cv} "
                f"| {_fmt_metric(m.get('val_argmax_quadratic_weighted_kappa'))} "
                f"| {_fmt_metric(m.get('val_cost_quadratic_weighted_kappa'))} "
                f"| {_fmt_metric(m.get('val_cost_undertriage_rate'))} "
                f"| {_fmt_metric(m.get('val_vc_argmax_quadratic_weighted_kappa'))} "
                f"| {_fmt_metric(m.get('val_vc_cost_quadratic_weighted_kappa'))} "
                f"| {format_duration(t.duration_s) if t.status == DONE else '—'} | {run} |"
            )
        lines += [
            "",
            "Kolumny: *CV* = najlepszy wynik tuningu (k-fold na TRAIN, ± odchylenie między foldami); "
            "*koszt* = reguła minimalnego kosztu oczekiwanego (niedotriażowanie 3× droższe); "
            "*vc* = podgrupa z kompletem HR/SBP/RR przy triażu (populacja jak w TOPSOR).",
            "",
        ]
        cmp = self.compare.get(fs) or {}
        lines += ["## Porównanie i łączenie", ""]
        if cmp.get("report"):
            name = Path(cmp["report"]).name
            lines.append(f"- raport porównania (VAL, różnorodność, średnia równowagowa): [`{name}`]({name})")
        else:
            lines.append(f"- porównanie: {cmp.get('status', 'nie uruchomiono')}"
                         + (f" — log `{cmp['log']}`" if cmp.get("log") else ""))
        lines += ["- zestawienie wszystkich runów (VAL): `runs/leaderboard.csv`",
                  "- ręczne łączenie na VAL: `from src.ensemble import latest_runs, aligned_probas, blend, ...` "
                  "(training/README.md)", ""]

        failed = [t for t in tasks if t.status == FAILED or t.eval_status == FAILED]
        if failed:
            lines += ["## Błędy", ""]
            for t in failed:
                log_path = Path(t.logs[-1]) if t.logs else None
                lines += [f"### {t.model} — {STATUS_PL.get(t.status, t.status)}"
                          + (" (ewaluacja nieudana)" if t.eval_status == FAILED else ""),
                          "", f"- {t.error or ''}", f"- log: `{log_path}`" if log_path else "", ""]
                if log_path is not None:
                    lines += ["```text", _tail(log_path), "```", ""]

        lines += ["## Co dalej", ""]
        unfinished = [t for t in tasks if t.status not in _OK_STATES or t.eval_status in (FAILED, NOT_STARTED,
                                                                                            INTERRUPTED)]
        if unfinished:
            lines.append(("- wznowienie (tylko modele niewytrenowane w tym przebiegu --force; Optuna dokończy "
                          "brakujące próby): " if o.force else
                          "- wznowienie (gotowe modele zostaną pominięte, Optuna dokończy brakujące próby): ")
                         + f"`{self._resume()}`")
        lines += [f"- stan bez uruchamiania: `python scripts/run_all.py --status --profile {o.profile}"
                  + (f" --feature-set {fs}" if fs != "triage" else "") + "`"]
        if o.profile in ("smoke", "quick"):
            lines.append("- to był przebieg testowy — właściwy: `python scripts/run_all.py` (profil standard)")
        elif o.profile == "standard":
            lines.append("- dłuższy tuning (≈1,7× próby i czas): `python scripts/run_all.py --profile thorough`")
        path = self.runs_dir / f"run_all_summary__{fs}__{o.profile}__{self.ts}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    # ── całość ──
    def exit_code(self) -> int:
        if self.interrupted:
            return 130
        bad = any(t.status in (FAILED, INTERRUPTED, NOT_STARTED) or t.eval_status == FAILED for t in self.tasks)
        bad = bad or any(c.get("status") == FAILED for c in self.compare.values())
        return 1 if bad else 0

    def _resume(self) -> str:
        """Polecenie wznowienia; po --force — z --force i tylko modelami bez ukończonego treningu."""
        redo = None
        if self.opts.force and self.tasks:
            redo = [m for m in self.opts.models
                    if any(t.model == m and t.status not in _OK_STATES for t in self.tasks)]
        return resume_command(self.opts.argv, redo)

    def run(self) -> int:
        """Cały przebieg; status.json zawsze kończy się stanem (finished / interrupted / failed) i kodem."""
        o = self.opts
        if not o.dry_run:
            self._setup_logging()
        restore_signals = self._install_signal_handlers()
        state, code = "failed", 1
        try:
            code = self._run()
            state = "interrupted" if self.interrupted else "finished"
            return code
        except KeyboardInterrupt:
            state, code, self.interrupted = "interrupted", 130, True
            raise
        finally:
            restore_signals()
            for proc in self._live_children():      # nic nie może przeżyć orkiestratora
                kill_process_tree(proc)
            if not o.dry_run:
                self.run_state, self.exit_status = state, code
                self._write_status()
            self._teardown_logging()

    def _run(self) -> int:
        from src.pipeline.keepawake import KeepAwake, QuickEditGuard
        from src.utils.hardware import disable_power_throttling

        o = self.opts
        if not o.dry_run and disable_power_throttling():
            self.say("Windows: wyłączono dławienie energii (EcoQoS) procesu orkiestratora")
        with KeepAwake(enabled=o.keep_awake and not o.dry_run) as awake, \
                QuickEditGuard(enabled=not o.dry_run) as quickedit:
            if not o.dry_run:
                self.say(f"run_all {self.ts} — blokada uśpienia: {awake.description}",
                         logging.WARNING if awake.host_warning or (o.keep_awake and not awake.active)
                         else logging.INFO)
                if quickedit.active:
                    self.say(f"Konsola: {quickedit.description}")
            if not o.skip_preflight and not o.dry_run:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                if not self.run_preflight():
                    return 2
            if not self.prepare():
                self._write_status()
                self.say(f"Wznowienie: {self._resume()}", logging.WARNING)
                return 130 if self.interrupted else 2
            self.plan()
            self.print_plan()
            if o.dry_run:
                for task in self.tasks:
                    if task.status == PENDING:
                        self.say(f"  $ {subprocess.list2cmdline(train_command(task.model, task.fs, o.profile, o.tag, o.tune))}")
                return 0
            with self._lock:
                self._lane_busy = {lane.name: True for lane in self.lanes}
            self._write_status()
            self._run_jobs([lambda lane=lane: self._lane_worker(lane) for lane in self.lanes])
            for task in self.tasks:     # przerwanie w trakcie: oznacz, co nie ruszyło
                if task.status == PENDING:
                    task.status, task.eval_status = NOT_STARTED, NOT_STARTED
                elif task.status == RUNNING:
                    task.status = INTERRUPTED
            if not self.interrupted:
                self.run_compare()
            for fs in o.feature_sets:
                self.summaries.append(self.write_summary(fs))
            self._write_status()
            self._print_final()
            return self.exit_code()

    def _print_final(self) -> None:
        self.say("═" * 72)
        for t in self.tasks:
            m = run_metrics(t.run_dir)
            self.say(f"  {t.fs:7s} {t.model:14s} {STATUS_PL.get(t.status, t.status):17s} QWK val (argmax/koszt) "
                     f"{_fmt_metric(m.get('val_argmax_quadratic_weighted_kappa'))} / "
                     f"{_fmt_metric(m.get('val_cost_quadratic_weighted_kappa'))}"
                     + (f" → {t.run_name}" if t.run_name else ""))
        for path in self.summaries:
            self.say(f"Podsumowanie: {path}")
        code = self.exit_code()
        if code != 0:
            self.say(f"Wznowienie: {self._resume()}", logging.WARNING)


# ─────────────────────────────────────────
# --status
# ─────────────────────────────────────────
STATE_DEAD = "przerwany (proces nie działa)"


def _latest_status() -> tuple[Path | None, dict[str, Any]]:
    latest = sorted((Path(config.LOGS_DIR) / "run_all").glob("*/status.json"))
    if not latest:
        return None, {}
    return latest[-1], read_json(latest[-1]) or {}


def status_report(opts: Options) -> str:
    """
    Tabela stanu z plików-znaczników (DONE.json, ewaluacja, badania Optuny, ostatni status.json)
    — nic nie uruchamia. Krok „w toku” w status.json martwego orkiestratora (PID nie działa,
    był restart) jest pokazywany jako „przerwany (proces nie działa)”.
    """
    from src.data.dataset import is_prepared

    runs_dir = Path(config.RUNS_DIR)
    fingerprint = data_fingerprint()
    manifest = read_json(Path(config.DATA_MANIFEST_JSON)) or {}
    status_path, status = _latest_status()
    same_run = status.get("tag") == opts.tag
    running = {f"{t.get('fs')}/{t.get('model')}" for t in status.get("tasks", []) if t.get("status") == RUNNING}
    active = status.get("state") in (None, "running")
    alive = orchestrator_alive(status) if status and active else False
    planned = resolve_devices(os.environ, status.get("devices") if same_run else None)

    lines = [
        f"Stan: profil '{opts.profile}', tag '{opts.tag}', dane {config.DATA_VERSION} "
        f"({'gotowe' if is_prepared() else 'NIEGOTOWE'}, podział '{manifest.get('split_strategy')}', "
        f"odcisk {fingerprint})",
        "",
        f"{'zestaw':7s} {'model':14s} {'stan':30s} {'QWK val arg/koszt':18s} {'urządzenie':11s} run / szczegóły",
    ]
    for fs in opts.feature_sets:
        for model in opts.models:
            run_dir = find_done_run(runs_dir, model, fs, opts.tag, config.DATA_VERSION, fingerprint, tuned=opts.tune)
            if run_dir is not None:
                m = run_metrics(run_dir)
                actual, _ = done_device(read_json(run_dir / DONE_JSON) or {})
                state = "gotowy" + ("" if eval_done(run_dir) else " (bez ewaluacji)")
                if planned.get(model) == "gpu" and not is_gpu_device(actual):
                    state = "gotowy na CPU — do powtórki na GPU"
                qwk = (f"{_fmt_metric(m.get('val_argmax_quadratic_weighted_kappa'), 3)} / "
                       f"{_fmt_metric(m.get('val_cost_quadratic_weighted_kappa'), 3)}")
                lines.append(f"{fs:7s} {model:14s} {state:30s} {qwk:18s} {str(actual or '—'):11s} {run_dir.name}")
                continue
            progress = optuna_progress(Path(config.OPTUNA_STORAGE_DIR), model, fs, opts.profile, fingerprint)
            partial = [p for p in runs_dir.glob(f"*_{model}_{fs}_{opts.tag}") if not (p / DONE_JSON).exists()]
            detail = ""
            if progress:
                finished = progress.get("complete", 0) + progress.get("pruned", 0)
                target = config.TUNING_PROFILES[opts.profile][model]["n_trials"]
                state = f"tuning {finished}/{target} prób"
                detail = ", ".join(f"{k} {v}" for k, v in sorted(progress.items()))
            elif partial:
                state, detail = "przerwany", f"niedokończony katalog {sorted(partial)[-1].name}"
            else:
                state = "nierozpoczęty"
            if same_run and active and f"{fs}/{model}" in running:
                if alive is False:
                    detail = f"{state}; {detail}" if detail else state
                    state = STATE_DEAD
                elif alive:
                    state = "w toku — " + state if progress else "w toku"
            lines.append(f"{fs:7s} {model:14s} {state:30s} {'—':18s} {'—':11s} {detail}")
    if status_path is not None:
        line = f"Ostatnie uruchomienie: {status_path.parent} (aktualizacja {status.get('updated_at')})"
        if active and alive is False:
            line += f" — {STATE_DEAD}" + (f"; niedokończone: {', '.join(sorted(running))}" if running else "") \
                + " — wznów tym samym poleceniem"
        elif active and alive:
            line += f" — w toku (PID {status.get('pid')})" + (f": {', '.join(sorted(running))}" if running else "")
        elif active:
            line += " — w toku: " + ", ".join(sorted(running)) if running else ""
        else:
            line += f" — zakończony ({status.get('state')}, kod {status.get('exit_code')})"
        lines += ["", line]
    return "\n".join(lines)


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_all.py",
        description="SOR-AI: jedno polecenie — przygotowanie danych, tuning, trening, ewaluacja i porównanie "
                    "modeli ESI (wznawialne: po awarii uruchom to samo polecenie).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=list(config.TUNING_PROFILES), default=config.DEFAULT_TUNING_PROFILE,
                        help="Budżet: smoke (5–10 min test), quick (~30 min), standard (noc), thorough")
    parser.add_argument("--feature-set", default="triage", help="triage | full | triage,full")
    parser.add_argument("--models", default="all", help="all albo lista po przecinku, np. lightgbm,xgboost")
    parser.add_argument("--split", choices=list(SPLITS), default="random",
                        help="Podział danych: random (stratyfikowany) albo site (oddział B jako test)")
    parser.add_argument("--no-tune", action="store_true", help="Bez tuningu — parametry domyślne z config.py")
    parser.add_argument("--lanes", choices=["auto", "1", "2"], default="auto",
                        help="Tory równoległe: auto = 2, gdy któryś model idzie na GPU, przy ≥ 16 rdzeniach "
                             "fizycznych i ≥ 30 GB RAM (tor GPU: tylko modele na GPU)")
    parser.add_argument("--force", action="store_true",
                        help="Trenuj ponownie także modele z gotowym DONE.json (ukończony tuning Optuny i tak "
                             "zostanie użyty ponownie)")
    parser.add_argument("--fail-fast", action="store_true", help="Pierwszy nieudany model zatrzymuje kolejne")
    parser.add_argument("--dry-run", action="store_true", help="Pokaż plan i polecenia, nic nie uruchamiaj")
    parser.add_argument("--status", action="store_true", help="Pokaż stan z plików-znaczników i zakończ")
    parser.add_argument("--skip-preflight", action="store_true", help="Pomiń kontrolę środowiska")
    parser.add_argument("--no-keep-awake", action="store_true", help="Nie blokuj uśpienia komputera")
    return parser


def options_from_args(argv: Sequence[str]) -> Options:
    args = build_parser().parse_args(list(argv))
    try:
        models = parse_models(args.models)
        feature_sets = parse_feature_sets(args.feature_set)
    except ValueError as exc:
        build_parser().error(str(exc))
    return Options(
        profile=args.profile, feature_sets=feature_sets, models=models, split=args.split,
        tune=not args.no_tune, lanes=args.lanes, force=args.force, fail_fast=args.fail_fast,
        dry_run=args.dry_run, status=args.status, skip_preflight=args.skip_preflight,
        keep_awake=not args.no_keep_awake, argv=list(argv),
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = options_from_args(argv)
    if opts.status:
        print(status_report(opts))
        return 0
    try:
        return Orchestrator(opts).run()
    except KeyboardInterrupt:
        log.warning(f"Przerwano. Wznowienie: {resume_command(argv)}")
        return 130


__all__ = [
    "Lane",
    "ModelTask",
    "Options",
    "Orchestrator",
    "assign_lanes",
    "build_child_env",
    "data_fingerprint",
    "decide_lanes",
    "done_device",
    "find_done_run",
    "gpu_planned_models",
    "is_gpu_device",
    "kill_process_tree",
    "log_has_gpu_error",
    "main",
    "model_eta_s",
    "orchestrator_alive",
    "parse_feature_sets",
    "parse_models",
    "popen_group_kwargs",
    "resolve_devices",
    "resume_command",
    "status_report",
]
