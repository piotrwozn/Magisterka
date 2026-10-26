"""
Krok 13: przygotowanie modeli fundamentalnych (TFM) — TabPFN-3.5 i TabICLv2 — na tej maszynie.

Użycie (z katalogu training/, w venv z requirements-tfm.txt; najlepiej w zwykłej konsoli PowerShell):
    python scripts/13_tfm_setup.py                      # kontrola + pobranie wag + benchmark GPU + szacunki czasu
    python scripts/13_tfm_setup.py --offline-check      # tylko kontrola, BEZ sieci (jak przed treningiem)
    python scripts/13_tfm_setup.py --skip-download      # bez pobierania (wagi muszą już być)
    python scripts/13_tfm_setup.py --skip-benchmark     # bez benchmarku
    python scripts/13_tfm_setup.py --models tabicl      # tylko jeden model
    python scripts/13_tfm_setup.py --device cpu --contexts 500,1000 --test-rows 200   # test mechaniki na CPU

Kroki:
    1. torch i CUDA — wersja, koło z CUDA czy tylko-CPU (torch z PyPI na Windows nie ma CUDA) → dokładne
       polecenia instalacji koła cu130 (sterownik NVIDIA >= 580) albo cu128 (starszy sterownik); karta,
       VRAM całkowita i WOLNA.
    2. Wersje bibliotek: tabpfn >= 9.2.0, tabicl >= 2.2.0 (+ huggingface-hub, safetensors); plik .env
       w bieżącym katalogu z kluczami TABPFN_* (TabPFN czyta go przy imporcie — inna pamięć podręczna wag).
    3. Wagi TabPFN-3.5 — licencja NIEKOMERCYJNA (praca magisterska: dozwolone). Jednorazowa akceptacja na
       https://ux.priorlabs.ai (zakładka Licenses) i klucz API w TABPFN_TOKEN — albo logowanie w przeglądarce,
       które biblioteka otwiera sama w zwykłej konsoli. Plik trafia do pamięci podręcznej TabPFN
       (TABPFN_MODEL_CACHE_DIR; Windows: %APPDATA%\\tabpfn). Błędy rozróżniane: brak sieci / brak klucza
       lub licencji / dostęp Hugging Face.
    4. Checkpoint TabICLv2 (`jingang/TabICL`, plik tabicl-classifier-v2-20260212.ckpt) — hf_hub_download do
       pamięci podręcznej Hugging Face (HF_HOME / HF_HUB_CACHE); licencja z karty modelu.
    5. Kontrola offline (bez sieci, dokładnie jak trening): plik TabPFN w katalogu pamięci podręcznej,
       TabICL przez try_to_load_from_cache / hf_hub_download(local_files_only=True); sha256; konfiguracje
       prawdziwych checkpointów (TabPFN: architektura + InferenceConfig; TabICL: config vs konfiguracja v2
       zakładana w planie pamięci).
    6. Jądro SDPA na GPU: F.scaled_dot_product_attention (fp16) w sdpa_kernel([FLASH_ATTENTION,
       EFFICIENT_ATTENTION]) — zwykłe MHA (TabICL) i kształt TabPFN z GQA (q: 16 głów, k/v: 1 głowa,
       enable_gqa=True). Bez tych jąder SDPA spada do MATH — pamięć O(N²) przy dużym kontekście.
    7. Benchmark na prawdziwych przygotowanych danych (TRAIN, zestaw 'triage'): konteksty 5k / 25k / 50k ×
       2000 wierszy testowych, oba modele przez wrappery projektu (te same ustawienia co trening: bufor KV,
       porcje, drabina OOM) — czas dopasowania, wiersze/s, szczyt VRAM → runs/analyses/tfm_setup/benchmark.json
       + szacunki czasu każdego profilu (wybór konfiguracji + finalna predykcja VAL + TEST) i eksperymentu A
       (krzywa małych danych). Szacunki skalują pomiar liniowo z liczbą estymatorów i interpolują log-log
       po rozmiarze kontekstu — to rząd wielkości, nie gwarancja.

Benchmark na CPU (`--device cpu` albo brak CUDA), z atrapą (SOR_AI_TFM_FAKE) lub z przekierowanymi wagami
= TEST MECHANIKI: wynik trafia do runs/analyses/tfm_setup/benchmark_mechanics.json (benchmark.json z GPU zostaje
nietknięty — run_all.py i 14_small_data_curve.py używają tylko pomiaru GPU z prawdziwymi wagami), a bez CUDA
kod wyjścia to 1 ("TFM nie są gotowe do treningu" — na CPU działa tylko profil smoke).

benchmark.json (kontrakt dla run_all.py / 14_small_data_curve.py): {"kind": "tfm_setup_benchmark", "version": 1,
"created", "device", "gpu", "torch", "libraries", "weights", "data", "results": [{model, n_ctx, n_estimators,
test_rows, status (ok | skipped | oom | failed), fit_s, predict_s, rows_per_s, peak_vram_mb, detail}],
"estimates": {"profiles": {profil: {model: {selection_s, selection_capped_s, final_s, final_bags,
final_capped_s, total_s, eta_upper_s, within_budget}}}, "experiment_a": {model: s, "total_s": s, ...}}}.

Haki testowe (nigdy w zwykłym użyciu): SOR_AI_TABPFN_MODEL_PATH / SOR_AI_TABICL_MODEL_PATH (pliki z losowymi
wagami) — pobieranie tego modelu pominięte, kontrola dotyczy wskazanego pliku; SOR_AI_TFM_FAKE=1 — atrapa
estymatora (benchmark bez torch).

Kody wyjścia: 0 = gotowe do treningu TFM, 1 = coś wymaga poprawy (lista [BŁĄD ] powyżej wyniku) albo brak
CUDA (test mechaniki na CPU się udał, ale TFM nie są gotowe do treningu), 2 = złe argumenty, 130 = Ctrl+C.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 12):
    print(f"[BŁĄD ] Python {sys.version.split()[0]} — wymagany >= 3.12 (zalecany 3.13 x64).")
    sys.exit(1)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Iterable, Mapping, Sequence  # noqa: E402
from datetime import datetime  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Lekki moduł (bez config): --n-jobs musi trafić do SOR_AI_N_JOBS przed importem config
from src.analysis.cli import apply_n_jobs  # noqa: E402

TFM_NAMES: tuple[str, ...] = ("tabpfn", "tabicl")
DISPLAY: dict[str, str] = {"tabpfn": "TabPFN-3.5", "tabicl": "TabICLv2"}
DEFAULT_CONTEXTS: tuple[int, ...] = (5_000, 25_000, 50_000)
DEFAULT_TEST_ROWS: int = 2_000
WARMUP_CONTEXT, WARMUP_TEST_ROWS = 200, 100        # rozgrzewka (CUDA, wczytanie wag) — poza wynikami
SDPA_SEQ_LEN: int = 20_000                          # długość sekwencji próby SDPA na GPU (bez MATH: ~0,1 GB)
SDPA_SEQ_LEN_CPU: int = 512
BENCHMARK_SUBDIR, BENCHMARK_FILE = "tfm_setup", "benchmark.json"
# Test mechaniki (CPU / atrapa / przekierowane wagi) — osobny plik: NIGDY nie nadpisuje pomiaru GPU,
# z którego run_all.py i 14_small_data_curve.py liczą szacunki czasu
MECHANICS_FILE = "benchmark_mechanics.json"
# Szczyt pamięci ≥ tego ułamka VRAM → ostrzeżenie (Windows: Sysmem Fallback zamiast OOM)
VRAM_NEAR_FULL_FRACTION = 0.9
# Czas na wiersz testu rośnie > SLOWDOWN_FACTOR × proporcja kontekstów → podejrzany zjazd (pamięć współdzielona)
SLOWDOWN_FACTOR = 10.0
BENCHMARK_VERSION: int = 1
TABICL_REPO_ID: str = "jingang/TabICL"
DEFAULT_TABICL_CHECKPOINT: str = "tabicl-classifier-v2-20260212.ckpt"
TABPFN_LICENCE_URL: str = "https://ux.priorlabs.ai"
TABPFN_HF_REPO: str = "Prior-Labs/tabpfn_3_5"
# Architektura klasyfikatora TabICLv2 zakładana w planie pamięci (skrypty treningu v2 biblioteki)
TABICL_V2_CONFIG: dict[str, Any] = {
    "max_classes": 10, "num_quantiles": 999, "embed_dim": 128,
    "col_num_blocks": 3, "col_nhead": 8, "col_num_inds": 128, "col_affine": False,
    "col_feature_group": "same", "col_feature_group_size": 3, "col_target_aware": True,
    "col_ssmax": "qassmax-mlp-elementwise",
    "row_num_blocks": 3, "row_nhead": 8, "row_num_cls": 4, "row_rope_base": 100000.0,
    "row_rope_interleaved": False,
    "icl_num_blocks": 12, "icl_nhead": 8, "icl_ssmax": "qassmax-mlp-elementwise",
    "ff_factor": 2, "dropout": 0.0, "activation": "gelu", "norm_first": True,
    "bias_free_ln": False, "zero_init": False, "recompute": False,
}
TABPFN_CONFIG_KEYS: tuple[str, ...] = ("embed_dim", "nlayers", "icl_num_heads", "icl_num_kv_heads_test",
                                      "max_num_classes", "num_buckets", "inference_chunk_cells")
TABPFN_INFERENCE_KEYS: tuple[str, ...] = ("N_ESTIMATORS", "MAX_NUMBER_OF_SAMPLES", "MAX_NUMBER_OF_FEATURES",
                                          "MAX_NUMBER_OF_CLASSES", "MAX_CPU_SAMPLES", "SOFTMAX_TEMPERATURE",
                                          "SUBSAMPLE_SAMPLES", "FEATURE_SUBSAMPLING_METHOD")
# Eksperyment A (krzywa małych danych, TFM_BRIEF §5): domyślne rozmiary, powtórzenia i wiersze ewaluacji
EXP_A_SIZES: tuple[int, ...] = (500, 1_000, 2_000, 5_000, 10_000, 50_000)
EXP_A_REPEATS: tuple[int, ...] = (10, 10, 10, 5, 5, 3)
EXP_A_EVAL_ROWS: int = 20_000
EXP_A_FOREIGN_ROWS: int = 10_000
EXP_A_SITE_VAL_ROWS: int = 16_633                  # VAL oddziału B (gdy danych nie da się odczytać)
EXP_A_N_ESTIMATORS: int = 8                        # domyślne biblioteki ("out of the box")
# Fragmenty komunikatów oznaczające brak sieci (łańcuch wyjątków pobierania)
NETWORK_PATTERNS: tuple[str, ...] = (
    "tunnel connection failed", "name or service not known", "getaddrinfo", "temporary failure in name resolution",
    "network is unreachable", "connection refused", "connection reset", "timed out", "max retries exceeded",
    "failed to establish a new connection", "could not reach the license server", "nodename nor servname",
    "no route to host", "offline mode", "check your connection",
)

STATUS_OK, STATUS_WARN, STATUS_FAIL, STATUS_INFO = "ok", "warn", "fail", "info"
_LABELS = {STATUS_OK: "[ OK  ]", STATUS_WARN: "[UWAGA]", STATUS_FAIL: "[BŁĄD ]", STATUS_INFO: "[INFO ]"}


# ─────────────────────────────────────────
# Wypisywanie (na bieżąco — pobieranie i benchmark trwają)
# ─────────────────────────────────────────
class Console:
    """Wpisy kontroli: wypisywane od razu i zbierane do podsumowania (kod wyjścia)."""

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream
        self.entries: list[dict[str, str]] = []

    def _print(self, text: str = "") -> None:
        print(text, file=self.stream or sys.stdout, flush=True)

    def section(self, title: str) -> None:
        self._print()
        self._print(f"── {title} " + "─" * max(4, 66 - len(title)))

    def add(self, name: str, status: str, message: str, advice: str | Sequence[str] = "") -> None:
        self.entries.append({"name": name, "status": status, "message": message})
        self._print(f"{_LABELS[status]} {name}: {message}")
        lines = [advice] if isinstance(advice, str) else list(advice)
        for line in [x for x in lines if x]:
            self._print(f"          → {line}")

    def info(self, text: str) -> None:
        self._print(f"          {text}")

    @property
    def failures(self) -> list[dict[str, str]]:
        return [e for e in self.entries if e["status"] == STATUS_FAIL]

    @property
    def warnings(self) -> list[dict[str, str]]:
        return [e for e in self.entries if e["status"] == STATUS_WARN]


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _short(exc: BaseException, limit: int = 300) -> str:
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: " + ((lines[0] if lines else "")[:limit])


def _fmt_s(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


# ─────────────────────────────────────────
# 1. torch i CUDA
# ─────────────────────────────────────────
def _driver_major(driver: str | None) -> int | None:
    match = re.match(r"\s*(\d+)", str(driver or ""))
    return int(match.group(1)) if match else None


def torch_install_commands(driver_major: int | None, cpu_only_installed: bool) -> list[str]:
    """Dokładne polecenia instalacji koła torch z CUDA (indeks PyTorch, nie PyPI)."""
    from src.utils.hardware import TORCH_CUDA_INDEX_URL, TORCH_CUDA_INDEX_URL_OLD_DRIVER

    lines = ["pip uninstall -y torch   (usuń koło tylko-CPU — inaczej pip uzna je za wystarczające)"] \
        if cpu_only_installed else []
    if driver_major is None:
        lines += [f"pip install torch --index-url {TORCH_CUDA_INDEX_URL}   (sterownik NVIDIA >= 580)",
                  f"pip install torch --index-url {TORCH_CUDA_INDEX_URL_OLD_DRIVER}   (starszy sterownik)"]
    elif driver_major >= 580:
        lines.append(f"pip install torch --index-url {TORCH_CUDA_INDEX_URL}   (sterownik {driver_major} >= 580)")
    else:
        lines.append(f"pip install torch --index-url {TORCH_CUDA_INDEX_URL_OLD_DRIVER}   (sterownik {driver_major} "
                     f"< 580 — koło cu130 wymaga >= 580; albo zaktualizuj sterownik)")
    lines.append('sprawdzenie: python -c "import torch; print(torch.__version__, torch.cuda.is_available())"')
    return lines


def check_torch(con: Console, *, device: str, torch_info: Callable[[], dict[str, Any]] | None = None,
                smi: Mapping[str, Any] | None | bool = False) -> dict[str, Any]:
    """
    torch + CUDA. device='cpu' (jawne) — brak CUDA tylko ostrzeżeniem (test mechaniki); inaczej błąd.
    Zwraca {'info', 'cuda', 'device' ('cuda' | 'cpu'), 'driver'}.
    """
    from src.utils import hardware

    con.section("1. torch i CUDA")
    info = dict((torch_info or hardware.torch_info)())
    if smi is False:
        smi = hardware.nvidia_smi_info()
    driver = _driver_major((smi or {}).get("driver_version")) if smi else None
    if smi:
        con.add("Karta (nvidia-smi)", STATUS_INFO, f"{smi.get('name')}, sterownik {smi.get('driver_version')}, "
                f"CUDA sterownika {smi.get('cuda_version')}, {smi.get('memory_total_mb')} MB")
    else:
        con.add("Karta (nvidia-smi)", STATUS_INFO, "nvidia-smi niedostępne — brak karty NVIDIA albo sterownika")
    cpu_ok = device == "cpu"
    bad = STATUS_WARN if cpu_ok else STATUS_FAIL
    out = {"info": info, "cuda": False, "device": "cpu", "driver": driver}
    version, build = info.get("version"), info.get("cuda_build")
    if not info.get("installed"):
        con.add("torch", STATUS_FAIL, "nie jest zainstalowany (wymagany przez TabPFN i TabICL)",
                [*torch_install_commands(driver, False),
                 "potem: pip install -r requirements-tfm.txt -c requirements-lock.txt"])
        return out
    if info.get("error") and str(info["error"]).startswith("import torch"):
        con.add("torch", STATUS_FAIL, f"{version}: {info['error']} (uszkodzona instalacja?)",
                torch_install_commands(driver, True))
        return out
    if build is None:
        con.add("torch", bad, f"{version} zbudowany BEZ CUDA (koło tylko-CPU — tak instaluje się torch z PyPI "
                f"na Windows); TFM potrzebują GPU" + ("; --device cpu: tylko test mechaniki" if cpu_ok else ""),
                torch_install_commands(driver, True))
        return out
    if not info.get("cuda_available"):
        reason = info.get("error") or "torch.cuda.is_available() == False"
        con.add("torch", bad, f"{version} (CUDA {build}): {reason} — brak karty NVIDIA albo sterownik za stary "
                f"dla CUDA {build}" + ("; --device cpu: tylko test mechaniki" if cpu_ok else ""),
                ["sterownik NVIDIA >= 580 dla koła cu130 (nvidia-smi: 'Driver Version')",
                 *torch_install_commands(driver, False)])
        return out
    total, free = info.get("total_vram_mb"), info.get("free_vram_mb")
    con.add("torch", STATUS_OK, f"{version} (CUDA {build}) na {info.get('device_name')}: VRAM {float(total or 0) / 1024:.1f} GB, "
            f"wolna {float(free or 0) / 1024:.1f} GB")
    out.update(cuda=True, device="cpu" if cpu_ok else "cuda")
    return out


# ─────────────────────────────────────────
# 2. Wersje bibliotek i ustawienia TabPFN
# ─────────────────────────────────────────
def check_versions(con: Console, models: Sequence[str],
                   version_of: Callable[[str], str | None] | None = None) -> dict[str, bool]:
    """tabpfn >= 9.2.0, tabicl >= 2.2.0 (błąd), huggingface-hub / safetensors (informacja). → {model: gotowy}."""
    from src.pipeline.preflight import TFM_MIN_VERSIONS, _version_at_least

    version_of = version_of or _dist_version
    con.section("2. Biblioteki")
    usable: dict[str, bool] = {}
    for model in models:
        installed, minimum = version_of(model), TFM_MIN_VERSIONS[model]
        if installed is None:
            con.add(model, STATUS_FAIL, f"nie jest zainstalowany (wymagany >= {minimum})",
                    "pip install -r requirements-tfm.txt -c requirements-lock.txt")
            usable[model] = False
        elif not _version_at_least(installed, minimum):
            con.add(model, STATUS_FAIL, f"{installed} < {minimum}",
                    "pip install -r requirements-tfm.txt -c requirements-lock.txt")
            usable[model] = False
        else:
            con.add(model, STATUS_OK, installed)
            usable[model] = True
    extras = {name: version_of(name) for name in ("huggingface-hub", "safetensors")}
    con.add("pozostałe", STATUS_INFO, ", ".join(f"{k} {v or 'brak'}" for k, v in extras.items()))
    return usable


def check_dotenv(con: Console, cwd: Path | None = None) -> list[str]:
    """
    Plik .env w bieżącym katalogu z kluczami TABPFN_* — TabPFN czyta go przy imporcie (pydantic-settings),
    np. TABPFN_MODEL_CACHE_DIR przekierowuje szukanie wag. Zwraca znalezione klucze.
    """
    path = Path(cwd or Path.cwd()) / ".env"
    keys: list[str] = []
    if path.is_file():
        try:
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if line and not line.startswith("#") and line.upper().startswith("TABPFN_") and "=" in line:
                    keys.append(line.split("=", 1)[0].strip())
        except OSError:
            pass
    if keys:
        con.add("Plik .env", STATUS_WARN, f"{path} ustawia {', '.join(keys)} — TabPFN czyta go przy imporcie "
                f"(inna pamięć podręczna wag / ustawienia niż w innym katalogu)",
                "Uruchamiaj trening i ten skrypt z tego samego katalogu (training/) albo usuń te klucze.")
    if os.environ.get("TABPFN_MODEL_CACHE_DIR"):
        con.add("Pamięć podręczna TabPFN", STATUS_INFO, f"TABPFN_MODEL_CACHE_DIR={os.environ['TABPFN_MODEL_CACHE_DIR']}")
    return keys


# ─────────────────────────────────────────
# 3–4. Pobieranie wag
# ─────────────────────────────────────────
def _exception_chain(exc: BaseException | None) -> list[BaseException]:
    seen: list[BaseException] = []
    stack = [exc]
    while stack:
        item = stack.pop()
        if item is None or any(item is s for s in seen):
            continue
        seen.append(item)
        stack += [getattr(item, "__cause__", None), getattr(item, "__context__", None),
                  getattr(item, "original_error", None)]
    return seen


def classify_download_error(exc: BaseException) -> tuple[str, str]:
    """
    Rodzaj błędu pobierania: ('network' | 'licence' | 'gated' | 'other', polski opis z instrukcją).
    TabPFN offline zgłasza mylące „model is gated” (TabPFNHuggingFaceGatedRepoError) — o rodzaju decyduje
    cały łańcuch wyjątków (URLError, ConnectionError, przekroczony czas …).
    """
    chain = _exception_chain(exc)
    names = {type(e).__name__ for e in chain} | {c.__name__ for e in chain for c in type(e).__mro__}
    text = " | ".join(str(e) for e in chain).lower()
    if (names & {"URLError", "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "TimeoutError",
                 "gaierror", "SSLError", "ProxyError", "OfflineModeIsEnabled"}
            or isinstance(exc, (socket.timeout, ConnectionError))
            or any(p in text for p in NETWORK_PATTERNS)):
        return "network", ("brak połączenia z siecią (huggingface.co / priorlabs.ai) — sprawdź internet, proxy "
                           "i zaporę; potem uruchom ponownie (pobrane pliki zostają w pamięci podręcznej)")
    if "TabPFNLicenseError" in names:
        return "licence", (f"brak klucza API albo niezaakceptowana licencja TabPFN — {TABPFN_LICENCE_URL}: zaloguj "
                           f"się, zakładka Licenses → zaakceptuj, klucz z {TABPFN_LICENCE_URL}/account → TABPFN_TOKEN "
                           f"(PowerShell: $env:TABPFN_TOKEN = \"<klucz>\"), albo uruchom w zwykłej konsoli bez "
                           f"--no-browser (przeglądarka)")
    if ("TabPFNHuggingFaceGatedRepoError" in names or "GatedRepoError" in names or "401" in text or "403" in text
            or "gated" in text):
        return "gated", (f"Hugging Face odmawia dostępu (repozytorium z ograniczonym dostępem) — zaakceptuj warunki "
                         f"na https://huggingface.co/{TABPFN_HF_REPO}, potem `hf auth login` albo zmienna HF_TOKEN "
                         f"(token do odczytu)")
    return "other", _short(exc)


def _licence_instructions(con: Console) -> None:
    token = "TABPFN_TOKEN (zmienna)" if os.environ.get("TABPFN_TOKEN") else None
    if token is None:
        with contextlib.suppress(Exception):
            from tabpfn.browser_auth import get_cached_token

            if get_cached_token():
                token = "zapamiętany klucz (~/.cache/tabpfn/auth_token)"
    con.add("Licencja TabPFN-3.5", STATUS_INFO,
            "wagi NIEKOMERCYJNE (praca magisterska — dozwolone); jednorazowa akceptacja licencji"
            + (f"; klucz API: {token}" if token else "; klucz API: brak"),
            [] if token else [f"1. {TABPFN_LICENCE_URL} → zaloguj się (albo załóż konto) → zakładka Licenses → "
                              f"zaakceptuj licencję",
                              f"2. skopiuj klucz API ze strony {TABPFN_LICENCE_URL}/account",
                              '3. PowerShell: $env:TABPFN_TOKEN = "<klucz>"   (na stałe: setx TABPFN_TOKEN "<klucz>" '
                              '+ nowe okno) i uruchom ponownie',
                              "albo: zwykła konsola bez TABPFN_TOKEN — biblioteka otworzy przeglądarkę i zapamięta "
                              "klucz"])


def download_tabpfn(con: Console, version: str, *, no_browser: bool = False,
                    downloader: Callable[..., Any] | None = None) -> bool:
    """Pobranie wag TabPFN (wersja `version`) do pamięci podręcznej TabPFN; True = plik jest."""
    from src.models.tabpfn_model import default_weights_path

    path = default_weights_path(version)
    if path.is_file() and path.stat().st_size > 0:
        con.add("TabPFN: wagi", STATUS_OK, f"już są: {path}")
        return True
    _licence_instructions(con)
    if no_browser:
        os.environ["TABPFN_NO_BROWSER"] = "1"
    con.add("TabPFN: pobieranie", STATUS_INFO, f"{version} → {path} (~0,9 GB; licencja → Hugging Face)")
    try:
        if downloader is None:
            from tabpfn.constants import ModelVersion
            from tabpfn.model_loading import download_model

            result = download_model(path, version=ModelVersion(version), which="classifier")
        else:
            result = downloader(path, version=version)
    except Exception as exc:  # noqa: BLE001 — opis zamiast tracebacku
        result = [exc]
    if result == "ok" and path.is_file():
        con.add("TabPFN: wagi", STATUS_OK, f"pobrane: {path} ({path.stat().st_size / 2**20:,.0f} MB)")
        return True
    errors = list(result) if isinstance(result, (list, tuple)) else []
    if not errors:
        errors = [RuntimeError(f"pobieranie zakończone ('{result}'), ale brak pliku {path}")]
    kind, message = classify_download_error(errors[0])
    con.add("TabPFN: pobieranie", STATUS_FAIL, f"nieudane ({kind}): {message}",
            [f"szczegóły: {_short(e)}" for e in errors[:2]])
    return False


def download_tabicl(con: Console, checkpoint: str, *, downloader: Callable[..., Any] | None = None) -> bool:
    """Pobranie checkpointu TabICL z Hugging Face (`jingang/TabICL`) do pamięci podręcznej HF."""
    try:
        from src.models.tabicl_model import cached_checkpoint
        from src.models.tfm_base import TFMWeightsError

        path = cached_checkpoint(checkpoint)
        con.add("TabICL: checkpoint", STATUS_OK, f"już jest: {path}")
        return True
    except TFMWeightsError:
        pass
    con.add("TabICL: pobieranie", STATUS_INFO, f"{TABICL_REPO_ID}/{checkpoint} (~110 MB) → pamięć podręczna Hugging "
            f"Face")
    try:
        if downloader is None:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(repo_id=TABICL_REPO_ID, filename=checkpoint))
        else:
            path = Path(downloader(repo_id=TABICL_REPO_ID, filename=checkpoint))
    except Exception as exc:  # noqa: BLE001
        kind, message = classify_download_error(exc)
        con.add("TabICL: pobieranie", STATUS_FAIL, f"nieudane ({kind}): {message}", f"szczegóły: {_short(exc)}")
        return False
    con.add("TabICL: checkpoint", STATUS_OK, f"pobrany: {path} ({path.stat().st_size / 2**20:,.0f} MB)")
    if downloader is None:
        with contextlib.suppress(Exception):
            from huggingface_hub import model_info

            card = getattr(model_info(TABICL_REPO_ID, timeout=10), "card_data", None)
            licence = (card or {}).get("license") if isinstance(card, Mapping) else getattr(card, "license", None)
            con.add("TabICL: licencja wag", STATUS_INFO, f"karta modelu {TABICL_REPO_ID}: {licence or 'nie podano'} "
                    f"(kod biblioteki: BSD-3)")
    return True


# ─────────────────────────────────────────
# 5. Kontrola offline + konfiguracje checkpointów
# ─────────────────────────────────────────
def describe_tabpfn_checkpoint(path: Path) -> dict[str, Any]:
    """Nagłówek safetensors (bez wczytywania tensorów): architektura, konfiguracja, InferenceConfig, parametry."""
    out: dict[str, Any] = {"format": path.suffix}
    if path.suffix != ".safetensors":
        return out
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        header = f.metadata() or {}
        n_params = 0
        for key in f.keys():  # noqa: SIM118 — safe_open nie jest iterowalny
            n_params += math.prod(f.get_slice(key).get_shape())
    parsed = {}
    for key, value in header.items():
        try:
            parsed[key] = json.loads(value)
        except (TypeError, ValueError):
            parsed[key] = value
    config = parsed.get("config") if isinstance(parsed.get("config"), dict) else {}
    inference = parsed.get("inference_config") if isinstance(parsed.get("inference_config"), dict) else {}
    out.update({
        "architecture": parsed.get("architecture_name"),
        "n_params": int(n_params),
        "config": {k: config.get(k) for k in TABPFN_CONFIG_KEYS if k in config},
        "inference_config": {k: inference.get(k) for k in TABPFN_INFERENCE_KEYS if k in inference},
    })
    if parsed.get("architecture_name") == "tabpfn_v3_5":
        with contextlib.suppress(Exception):
            import dataclasses

            from tabpfn.architectures import tabpfn_v3_5

            defaults = dataclasses.asdict(tabpfn_v3_5.TabPFNV3p5Config(
                max_num_classes=config.get("max_num_classes", 10), num_buckets=config.get("num_buckets", 5000)))
            out["differs_from_plan"] = {k: {"plan": defaults[k], "checkpoint": config[k]}
                                        for k in defaults if k in config and config[k] != defaults[k]
                                        and not isinstance(defaults[k], (dict, list))}
    return out


def describe_tabicl_checkpoint(path: Path) -> dict[str, Any]:
    """`torch.load(weights_only=True)`: config, liczba parametrów, różnice względem konfiguracji v2."""
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    config = dict(ck.get("config") or {})
    state = ck.get("state_dict") or {}
    return {
        "format": path.suffix,
        "n_params": int(sum(int(t.numel()) for t in state.values())),
        "config": config,
        "differs_from_plan": {k: {"plan": v, "checkpoint": config.get(k)} for k, v in TABICL_V2_CONFIG.items()
                              if config.get(k) != v},
    }


def offline_check(con: Console, models: Sequence[str], *, tabpfn_version: str,
                  tabicl_checkpoint: str) -> dict[str, dict[str, Any]]:
    """Wagi w lokalnej pamięci podręcznej — BEZ sieci, tak jak trening; sha256 i opis konfiguracji."""
    from src.models.tfm_base import weights_identity
    from src.pipeline.preflight import tfm_weights_status

    con.section("5. Kontrola offline (bez sieci, jak trening)")
    out: dict[str, dict[str, Any]] = {}
    for model in models:
        params = {"model_version": tabpfn_version} if model == "tabpfn" else {"checkpoint_version": tabicl_checkpoint}
        try:
            status = tfm_weights_status(model, params)
        except ImportError as exc:
            con.add(f"{model}: wagi", STATUS_FAIL, f"nie sprawdzono — {_short(exc)}",
                    "pip install -r requirements-tfm.txt -c requirements-lock.txt")
            continue
        out[model] = status
        if status.get("override") and str(status["override"]).startswith("SOR_AI_TFM_FAKE"):
            con.add(f"{model}: wagi", STATUS_WARN, "SOR_AI_TFM_FAKE=1 — atrapa estymatora, wagi nie są używane")
            continue
        if not status.get("present"):
            where = status.get("path") or status.get("cache_dir") or "?"
            con.add(f"{model}: wagi", STATUS_FAIL, f"brak w lokalnej pamięci podręcznej ({where})",
                    "uruchom bez --offline-check / --skip-download (z siecią), żeby pobrać")
            continue
        path = Path(str(status["path"]))
        if model == "tabicl" and not status.get("override"):
            with contextlib.suppress(Exception):          # druga droga biblioteki: try_to_load_from_cache
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(TABICL_REPO_ID, tabicl_checkpoint)
                status["try_to_load_from_cache"] = cached if isinstance(cached, str) else None
        ident = weights_identity(path)
        status["sha256"] = ident.get("sha256")
        prefix = f"PRZEKIEROWANIE (tylko testy: {status['override']}) — " if status.get("override") else ""
        con.add(f"{model}: wagi", STATUS_WARN if status.get("override") else STATUS_OK,
                f"{prefix}{path} ({path.stat().st_size / 2**20:,.1f} MB, sha256 {str(ident.get('sha256'))[:16]}…)")
        try:
            desc = describe_tabpfn_checkpoint(path) if model == "tabpfn" else describe_tabicl_checkpoint(path)
        except Exception as exc:  # noqa: BLE001 — uszkodzony plik
            con.add(f"{model}: checkpoint", STATUS_FAIL, f"nie da się odczytać: {_short(exc)}",
                    "usuń plik i pobierz ponownie")
            continue
        status["checkpoint"] = desc
        shown = {k: v for k, v in desc.items() if k not in ("differs_from_plan",) and v not in (None, {}, [])}
        con.add(f"{model}: checkpoint", STATUS_INFO, json.dumps(shown, ensure_ascii=False, default=str)[:600])
        diffs = desc.get("differs_from_plan") or {}
        if diffs:
            text = ", ".join(f"{k}: plan {v['plan']} / plik {v['checkpoint']}" for k, v in list(diffs.items())[:8])
            con.add(f"{model}: konfiguracja", STATUS_INFO if status.get("override") else STATUS_WARN,
                    f"inna niż zakładana w planie pamięci ({len(diffs)} różnic): {text}",
                    "" if status.get("override") else "Szacunki VRAM planu mogą nie obowiązywać — patrz benchmark "
                                                      "(krok 7).")
    return out


# ─────────────────────────────────────────
# 6. Jądro SDPA
# ─────────────────────────────────────────
def sdpa_check(device: str = "cuda", seq_len: int | None = None) -> dict[str, Any]:
    """
    F.scaled_dot_product_attention z jądrami fused (FLASH / EFFICIENT, BEZ MATH): MHA (1, 16, L, 64) i kształt
    TabPFN z GQA (q 16 głów, k/v 1 głowa, enable_gqa=True). fp16 na CUDA, fp32 na CPU (test).
    Returns: {'device', 'seq_len', 'kernels': {nazwa: {'mha', 'gqa'}}, 'fused_ok', 'errors'}.
    """
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    on_gpu = str(device).startswith("cuda")
    seq_len = int(seq_len or (SDPA_SEQ_LEN if on_gpu else SDPA_SEQ_LEN_CPU))
    dtype = torch.float16 if on_gpu else torch.float32
    gen = torch.Generator(device="cpu").manual_seed(0)

    def tensor(heads: int) -> Any:
        return torch.randn(1, heads, seq_len, 64, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    q, k, v, k1, v1 = tensor(16), tensor(16), tensor(16), tensor(1), tensor(1)
    variants = {"flash": [SDPBackend.FLASH_ATTENTION], "efficient": [SDPBackend.EFFICIENT_ATTENTION],
                "fused": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]}
    kernels: dict[str, dict[str, bool]] = {}
    errors: dict[str, str] = {}
    for name, backends in variants.items():
        kernels[name] = {}
        for shape, (kk, vv, gqa) in (("mha", (k, v, False)), ("gqa", (k1, v1, True))):
            try:
                with sdpa_kernel(backends):
                    out = F.scaled_dot_product_attention(q, kk, vv, enable_gqa=gqa)
                if on_gpu:
                    torch.cuda.synchronize()
                kernels[name][shape] = bool(torch.isfinite(out.float()).all().item())
            except Exception as exc:  # noqa: BLE001 — brak jądra = wynik, nie awaria
                kernels[name][shape] = False
                errors[f"{name}/{shape}"] = _short(exc, 200)
    del q, k, v, k1, v1
    if on_gpu:
        torch.cuda.empty_cache()
    return {"device": device, "seq_len": seq_len, "kernels": kernels,
            "fused_ok": bool(kernels["fused"].get("mha") and kernels["fused"].get("gqa")), "errors": errors}


def report_sdpa(con: Console, result: Mapping[str, Any]) -> None:
    kernels = result.get("kernels") or {}
    detail = "; ".join(f"{name}: MHA {'tak' if k.get('mha') else 'nie'}, GQA {'tak' if k.get('gqa') else 'nie'}"
                       for name, k in kernels.items() if name != "fused")
    if result.get("fused_ok"):
        con.add("SDPA (fp16, bez MATH)", STATUS_OK, f"L={result.get('seq_len'):,}: {detail}")
    else:
        con.add("SDPA (fp16, bez MATH)", STATUS_FAIL,
                f"brak jądra FLASH / EFFICIENT dla kształtu TFM (L={result.get('seq_len'):,}): {detail} — SDPA spadłby "
                f"do MATH (pamięć O(N²): przy kontekście 100k dziesiątki GB)",
                ["zaktualizuj torch (koło CUDA z indeksu PyTorch) i sterownik NVIDIA",
                 *[f"{k}: {v}" for k, v in list((result.get("errors") or {}).items())[:3]]])


# ─────────────────────────────────────────
# 7. Benchmark i szacunki czasu
# ─────────────────────────────────────────
def parse_int_list(text: str) -> list[int]:
    values = []
    for raw in str(text).split(","):
        raw = raw.strip().replace("_", "").lower()
        if not raw:
            continue
        mult = 1000 if raw.endswith("k") else 1
        number = int(float(raw[:-1] if mult > 1 else raw) * mult)
        if number <= 0:
            raise ValueError(f"oczekiwano liczb > 0, podano {text!r}")
        values.append(number)
    if not values:
        raise ValueError("pusta lista")
    return sorted(set(values))


def parse_estimators(text: str | None, models: Sequence[str]) -> dict[str, int]:
    """'4' (wszystkie) albo 'tabpfn=4,tabicl=8'; brak → pierwszy kandydat profilu standard."""
    from src.utils import config

    out: dict[str, int] = {}
    for model in models:
        cand = (config.TUNING_PROFILES["standard"][model].get("candidates") or [{}])[0]
        value = cand.get("n_estimators", 8)
        out[model] = 8 if str(value).lower() == "auto" else int(value)
    if text:
        for part in str(text).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                name, value = part.split("=", 1)
                name = {"pfn": "tabpfn", "icl": "tabicl"}.get(name.strip().lower(), name.strip().lower())
                if name not in TFM_NAMES:
                    raise ValueError(f"nieznany model w --n-estimators: {name}")
                out[name] = int(value)
            else:
                out = {m: int(part) for m in models}
    if any(v < 1 for v in out.values()):
        raise ValueError("n_estimators musi być ≥ 1")
    return out


def load_benchmark_data(n_pool: int, test_rows: int, seed: int = 42) -> tuple[Any, Any, Any, dict[str, Any]]:
    """
    Prawdziwe przygotowane dane (TRAIN, zestaw 'triage'): stratyfikowana próbka n_pool + test_rows wierszy
    (pula kontekstu, rozłączne wiersze testowe). Etykiety tylko do kontekstu (benchmark mierzy czas).
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    from src.data.dataset import data_fingerprint, feature_names, get_xy, load_split, manifest
    from src.data.prepare import META_COLS
    from src.utils.config import HOLDOUT_SITE

    df = load_split("train", columns=META_COLS + feature_names("triage"))
    n_total = min(len(df), int(n_pool) + int(test_rows))
    if n_total < len(df):
        df, _ = train_test_split(df, train_size=n_total, random_state=seed, stratify=df["target"])
    X, y = get_xy(df, "triage")
    n_test = min(int(test_rows), max(1, len(X) // 4))
    X_pool, X_test, y_pool, _ = train_test_split(X, y, test_size=n_test, random_state=seed, stratify=y)
    info: dict[str, Any] = {"source": "train", "feature_set": "triage", "n_features": int(X.shape[1]),
                            "pool_rows": int(len(X_pool)), "test_rows": int(len(X_test))}
    with contextlib.suppress(Exception):
        info["data_fingerprint"] = data_fingerprint()
        rows = manifest().get("n_rows") or {}
        info.update({"n_train": rows.get("train"), "n_val": rows.get("val"), "n_test": rows.get("test")})
        sites = load_split("val", columns=["site"])["site"].astype(str)
        info["holdout_site"] = HOLDOUT_SITE
        info["site_val_rows"] = int(np.sum(sites.to_numpy() == str(HOLDOUT_SITE)))
    return X_pool, y_pool, X_test, info


def _context_sample(X_pool: Any, y_pool: Any, n_ctx: int, seed: int) -> tuple[Any, Any]:
    from sklearn.model_selection import train_test_split

    if n_ctx >= len(X_pool):
        return X_pool, y_pool
    X_ctx, _, y_ctx, _ = train_test_split(X_pool, y_pool, train_size=int(n_ctx), random_state=seed, stratify=y_pool)
    return X_ctx, y_ctx


def benchmark_unit(model_name: str, X_ctx: Any, y_ctx: Any, X_test: Any, *, n_estimators: int,
                   n_jobs: int | None, seed: int = 42) -> dict[str, Any]:
    """
    Jeden pomiar przez wrapper projektu (te same ustawienia co trening): dopasowanie worka (bufor KV),
    predykcja `X_test`; czasy i szczyt VRAM z `TFMPredictor.throughput()`. Wyjątki przechodzą dalej.
    """
    from src.models import get_model

    params = {"n_ctx": int(len(X_ctx)), "n_bags": 1, "n_estimators": int(n_estimators), "random_state": int(seed)}
    if n_jobs:
        params["n_jobs"] = int(n_jobs)
    model = get_model(model_name, params=params)
    t0 = time.time()
    try:
        model.fit_estimator(X_ctx, y_ctx)
        model.predict_proba_many([X_test])
        thr = model.model.throughput() if model.model is not None else {}
    finally:
        with contextlib.suppress(Exception):
            model.release()
    # Jeden worek: sumy throughput() = ten worek; wierszy/s liczone w wrapperze z NIEZAOKRĄGLONEGO czasu
    predict_s = float(thr.get("predict_s") or 0.0)
    fit_s = float(thr.get("fit_s_per_bag") or 0.0)
    events = [e.get("kind") for e in getattr(model, "tfm_events", []) or []]
    return {
        "device": str(model.params.get("device")),
        "fit_s": round(fit_s, 3),
        "predict_s": round(predict_s, 3),
        "rows_per_s": thr.get("rows_per_s"),
        "peak_vram_mb": thr.get("peak_vram_mb"),
        "wall_s": round(time.time() - t0, 3),
        "events": [e for e in events if e in ("oom_downgrade", "library_oom_warning", "device_fallback")],
    }


def run_benchmark(con: Console, models: Sequence[str], contexts: Sequence[int], test_rows: int,
                  n_estimators: Mapping[str, int], *, n_jobs: int | None, seed: int = 42,
                  data: tuple[Any, Any, Any, dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict]:
    """Wszystkie (model, kontekst): status ok / skipped (CPU > limit) / oom / failed; wyniki + opis danych."""
    from src.models.tfm_base import TFMDeviceError, TFMOutOfMemoryError
    from src.utils import config

    X_pool, y_pool, X_test, data_info = data if data is not None else load_benchmark_data(
        max(contexts), test_rows, seed)
    con.add("Dane", STATUS_INFO, f"TRAIN 'triage': pula {len(X_pool):,} wierszy, test {len(X_test):,}, "
            f"{data_info.get('n_features')} cech")
    results: list[dict[str, Any]] = []
    for model in models:
        e = int(n_estimators[model])
        warm = min(WARMUP_CONTEXT, len(X_pool))
        try:                                  # rozgrzewka: inicjalizacja CUDA, wczytanie wag (poza wynikami)
            Xw, yw = _context_sample(X_pool, y_pool, warm, seed)
            benchmark_unit(model, Xw, yw, X_test.iloc[:WARMUP_TEST_ROWS], n_estimators=1, n_jobs=n_jobs, seed=seed)
        except Exception as exc:  # noqa: BLE001 — błąd i tak wyjdzie w pomiarach
            con.add(f"{model}: rozgrzewka", STATUS_WARN, _short(exc))
        for n_ctx in contexts:
            row: dict[str, Any] = {"model": model, "n_ctx": int(min(n_ctx, len(X_pool))), "n_estimators": e,
                                   "test_rows": int(len(X_test)), "status": "ok", "detail": None}
            try:
                Xc, yc = _context_sample(X_pool, y_pool, n_ctx, seed)
                row.update(benchmark_unit(model, Xc, yc, X_test, n_estimators=e, n_jobs=n_jobs, seed=seed))
            except TFMDeviceError as exc:
                row.update(status="skipped", detail=f"CPU: kontekst > TFM_CPU_MAX_CONTEXT={config.TFM_CPU_MAX_CONTEXT:,} "
                                                    f"({_short(exc, 120)})")
            except TFMOutOfMemoryError as exc:
                row.update(status="oom", detail=_short(exc))
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 — jeden pomiar nie przerywa reszty
                row.update(status="failed", detail=_short(exc))
            results.append(row)
            if row["status"] == "ok":
                vram = f", szczyt VRAM {row['peak_vram_mb']:,.0f} MB" if row.get("peak_vram_mb") is not None else ""
                con.add(f"{model} n_ctx={row['n_ctx']:,} × {e} est.", STATUS_OK,
                        f"dopasowanie {row['fit_s']:.1f} s, predykcja {row['test_rows']:,} wierszy {row['predict_s']:.1f} s "
                        f"({row['rows_per_s'] or 0:,.0f} wierszy/s){vram} [{row.get('device')}]"
                        + (f"; zdarzenia: {', '.join(row['events'])}" if row.get("events") else ""))
            else:
                status = STATUS_INFO if row["status"] == "skipped" else STATUS_FAIL
                con.add(f"{model} n_ctx={row['n_ctx']:,} × {e} est.", status, f"{row['status']}: {row['detail']}",
                        "brak pamięci GPU — zamknij inne programy na karcie; plan profilu może wymagać mniejszego "
                        "kontekstu" if row["status"] == "oom" else "")
    return results, data_info


def _interp_loglog(points: Sequence[tuple[float, float]], x: float, default_exponent: float) -> float | None:
    """Interpolacja / ekstrapolacja log-log po punktach (x, y > 0); jeden punkt → y·(x/x0)^wykładnik."""
    pts = sorted((float(a), max(float(b), 1e-9)) for a, b in points if a and a > 0)
    if not pts:
        return None
    if len(pts) == 1:
        x0, y0 = pts[0]
        return y0 * (float(x) / x0) ** default_exponent
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        i = next(i for i in range(1, len(pts)) if pts[i][0] >= x)
        (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    if x1 == x0:
        return y0
    slope = (math.log(y1) - math.log(y0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + slope * (math.log(float(x)) - math.log(x0)))


class Curve:
    """Koszt worka z pomiarów jednego modelu: dopasowanie(n_ctx) + wiersze × koszt wiersza(n_ctx), skala × estymatory."""

    def __init__(self, rows: Iterable[Mapping[str, Any]]):
        ok = [r for r in rows if r.get("status") == "ok" and r.get("predict_s") is not None]
        self.n_estimators = int(ok[0]["n_estimators"]) if ok else 1
        self.fit = [(r["n_ctx"], float(r.get("fit_s") or 0.0)) for r in ok]
        self.row = [(r["n_ctx"], float(r["predict_s"]) / max(int(r["test_rows"]), 1)) for r in ok]
        self.measured = sorted({int(r["n_ctx"]) for r in ok})

    def __bool__(self) -> bool:
        return bool(self.row)

    def bag_s(self, n_ctx: int, n_rows: int, n_estimators: int, feature_fraction: float = 1.0) -> float:
        fit = _interp_loglog(self.fit, n_ctx, 1.0) or 0.0
        per_row = _interp_loglog(self.row, n_ctx, 0.0) or 0.0
        scale = float(n_estimators) / float(self.n_estimators) * float(feature_fraction)
        return (fit + float(n_rows) * per_row) * scale


def _n_est(value: Any) -> int:
    return 8 if str(value).lower() == "auto" else int(value)


def profile_estimates(curves: Mapping[str, Curve], data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Szacunek każdego profilu × modelu: wybór konfiguracji (kandydaci etapu 1 na holdoucie + etap worków,
    gdy włączony) i finalna predykcja VAL + TEST najdroższym kandydatem (górna granica) z liczbą worków
    profilu; wartości „capped” = po limicie czasu wyboru i budżecie finalnym (redukcja worków, min. 1).
    """
    from src.utils import config

    n_train = int(data.get("n_train") or 446_421)
    n_val, n_test = int(data.get("n_val") or 55_803), int(data.get("n_test") or 55_803)
    n_features = int(data.get("n_features") or 271)
    out: dict[str, dict[str, Any]] = {}
    for profile, prof in config.TUNING_PROFILES.items():
        out[profile] = {}
        ts, es = prof.get("train_sample"), prof.get("eval_sample")
        n_fit = min(n_train, int(ts)) if ts else n_train
        n_eval = (min(n_val, int(es)) if es else n_val) + (min(n_test, int(es)) if es else n_test)
        for model, curve in curves.items():
            budget = prof.get(model) or {}
            if not curve or not budget.get("candidates"):
                continue
            hold = min(int(budget.get("holdout_rows") or 0), n_fit // 2)
            pool = n_fit - hold
            candidates = list(budget["candidates"])[: int(budget.get("n_trials") or len(budget["candidates"]))]
            selection, first, worst, worst_cfg = 0.0, None, 0.0, None
            for cand in candidates:
                frac = min(1.0, int(cand["max_features"]) / n_features) if cand.get("max_features") else 1.0
                t = curve.bag_s(min(int(cand["n_ctx"]), pool), hold, _n_est(cand.get("n_estimators", 8)), frac)
                selection += t
                first = t if first is None else first
                final_one = curve.bag_s(min(int(cand["n_ctx"]), n_fit), n_eval, _n_est(cand.get("n_estimators", 8)),
                                        frac)
                if final_one >= worst:
                    worst, worst_cfg = final_one, cand
            max_bags = int(budget.get("max_bags") or 1)
            ctx = min(int((worst_cfg or {}).get("n_ctx") or 1), n_fit)
            if budget.get("bags_stage") and worst_cfg is not None:
                n_stage = min(max_bags, math.ceil(pool / max(min(ctx, pool), 1)))
                if n_stage > 1:
                    frac = (min(1.0, int(worst_cfg["max_features"]) / n_features)
                            if worst_cfg.get("max_features") else 1.0)
                    selection += n_stage * curve.bag_s(min(ctx, pool), hold, _n_est(worst_cfg.get("n_estimators", 8)),
                                                       frac)
            bags = min(max_bags, math.ceil(n_fit / max(ctx, 1))) if budget.get("bags_stage") else 1
            final = bags * worst
            timeout, final_budget = float(budget.get("timeout_s") or 0), float(budget.get("final_budget_s") or 0)
            selection_capped = max(min(selection, timeout), first or 0.0) if timeout else selection
            final_capped = final
            if final_budget and final > final_budget:
                kept = max(1, int(final_budget // worst)) if worst > 0 else bags
                final_capped = min(bags, kept) * worst
            out[profile][model] = {
                "selection_s": round(selection, 1), "selection_capped_s": round(selection_capped, 1),
                "final_s": round(final, 1), "final_bags": int(bags), "final_capped_s": round(final_capped, 1),
                "total_s": round(selection_capped + final_capped, 1),
                "eta_upper_s": round(timeout + final_budget, 1),
                "within_budget": bool(selection <= (timeout or math.inf) and final <= (final_budget or math.inf)),
                "worst_candidate": dict(worst_cfg or {}),
            }
    return out


def experiment_a_estimate(curves: Mapping[str, Curve], data: Mapping[str, Any]) -> dict[str, Any]:
    """
    Eksperyment A (krzywa małych danych): jednostki TFM w obu scenariuszach — pooled (ewaluacja na 20 000
    wierszy VAL), site_local (VAL oddziału) i site_local + 10 000 obcych wierszy kontekstu; domyślne
    rozmiary / powtórzenia z TFM_BRIEF §5, 8 estymatorów (domyślne biblioteki) — `tfm_s`; plus część
    LightGBM (`lightgbm_s`: strojony Optuną na podpróbce, domyślny, przeniesiony — formuła planu
    14_small_data_curve.py --plan). `total_s` = obie części.
    """
    site_val = int(data.get("site_val_rows") or EXP_A_SITE_VAL_ROWS)
    out: dict[str, Any] = {"assumptions": {"sizes": list(EXP_A_SIZES), "repeats": list(EXP_A_REPEATS),
                                           "eval_rows": EXP_A_EVAL_ROWS, "site_val_rows": site_val,
                                           "foreign_rows": EXP_A_FOREIGN_ROWS, "n_estimators": EXP_A_N_ESTIMATORS},
                           "note": "jednostki TFM z benchmarku + LightGBM (Optuna na podpróbce, CPU) z formuły planu "
                                   "14_small_data_curve.py --plan"}
    total = 0.0
    for model, curve in curves.items():
        if not curve:
            continue
        seconds = 0.0
        for n, repeats in zip(EXP_A_SIZES, EXP_A_REPEATS, strict=True):
            pooled = curve.bag_s(n, EXP_A_EVAL_ROWS, EXP_A_N_ESTIMATORS)
            local = curve.bag_s(n, site_val, EXP_A_N_ESTIMATORS)
            foreign = curve.bag_s(n + EXP_A_FOREIGN_ROWS, site_val, EXP_A_N_ESTIMATORS)
            seconds += repeats * (pooled + local + foreign)
        out[model] = round(seconds, 1)
        total += seconds
    from src.analysis.small_data import SmallDataConfig, lightgbm_eta_s

    lightgbm = lightgbm_eta_s(SmallDataConfig(sizes=list(EXP_A_SIZES), repeats=list(EXP_A_REPEATS)))
    out["tfm_s"] = round(total, 1)
    out["lightgbm_s"] = round(lightgbm, 1)
    out["total_s"] = round(total + lightgbm, 1)
    return out


def print_estimates(con: Console, estimates: Mapping[str, Any], device: str) -> None:
    con.section("Szacunki czasu (z benchmarku; rząd wielkości)")
    if device != "cuda":
        con.add("Szacunki", STATUS_WARN, f"benchmark na '{device}' — liczby nie dotyczą GPU (tylko mechanika)")
    for profile, models in (estimates.get("profiles") or {}).items():
        for model, e in models.items():
            flag = "" if e["within_budget"] else " — PONAD budżet profilu (wybór przerwany limitem / mniej worków)"
            con.add(f"{profile} / {model}", STATUS_INFO,
                    f"wybór konfiguracji ~{_fmt_s(e['selection_capped_s'])} (bez limitu czasu: "
                    f"{_fmt_s(e['selection_s'])}), finalna predykcja VAL+TEST ~{_fmt_s(e['final_capped_s'])} (worki: "
                    f"{e['final_bags']}; bez budżetu: {_fmt_s(e['final_s'])}); razem ~{_fmt_s(e['total_s'])}, "
                    f"ETA run_all ≤ {_fmt_s(e['eta_upper_s'])}{flag}")
    exp_a = estimates.get("experiment_a") or {}
    if exp_a.get("total_s") is not None:
        parts = ", ".join(f"{m} ~{_fmt_s(exp_a[m])}" for m in TFM_NAMES if m in exp_a)
        con.add("Eksperyment A (14_small_data_curve.py)", STATUS_INFO,
                f"razem ~{_fmt_s(exp_a['total_s'])}: jednostki TFM ~{_fmt_s(exp_a.get('tfm_s', exp_a['total_s']))} "
                f"({parts}) + LightGBM ~{_fmt_s(exp_a.get('lightgbm_s') or 0)}; {exp_a.get('note')}")


def mechanics_only(device: str, fake: bool, weights: Mapping[str, Any]) -> str | None:
    """Powód, dla którego pomiar NIE jest benchmarkiem GPU do szacunków (CPU / atrapa / przekierowanie) albo None."""
    if fake:
        return "atrapa estymatora (SOR_AI_TFM_FAKE)"
    if device != "cuda":
        return f"urządzenie '{device}' (nie GPU)"
    overrides = [str((w or {}).get("override")) for w in (weights or {}).values() if (w or {}).get("override")]
    if overrides:
        return "przekierowane wagi (" + ", ".join(overrides) + ")"
    return None


def write_benchmark(payload: Mapping[str, Any], runs_dir: Path | None = None, *, mechanics: bool = False) -> Path:
    """benchmark.json (GPU, prawdziwe wagi) albo benchmark_mechanics.json (test mechaniki — bez nadpisania GPU)."""
    from src.analysis.common import analyses_root, write_json

    path = analyses_root(runs_dir) / BENCHMARK_SUBDIR / (MECHANICS_FILE if mechanics else BENCHMARK_FILE)
    return write_json(path, dict(payload))


def suspicious_rows(results: Sequence[Mapping[str, Any]], total_vram_mb: float | None) -> list[str]:
    """
    Pomiary podejrzane o przepełnienie VRAM bez OOM (Windows: "CUDA – Sysmem Fallback Policy" kieruje
    nadmiar do współdzielonej pamięci RAM — wielokrotnie wolniej, drabina OOM nie zadziała):
    szczyt ≥ VRAM_NEAR_FULL_FRACTION × VRAM albo czas na wiersz testu rośnie > SLOWDOWN_FACTOR × proporcja
    kontekstów między kolejnymi rozmiarami tego samego modelu.
    """
    notes: list[str] = []
    ok = [r for r in results if r.get("status") == "ok"]
    for r in ok:
        peak = r.get("peak_vram_mb")
        if total_vram_mb and peak is not None and float(peak) >= VRAM_NEAR_FULL_FRACTION * float(total_vram_mb):
            notes.append(f"{r['model']} n_ctx={int(r['n_ctx']):,}: szczyt {float(peak):,.0f} MB ≥ "
                         f"{VRAM_NEAR_FULL_FRACTION:.0%} VRAM ({float(total_vram_mb):,.0f} MB)")
    for model in dict.fromkeys(r["model"] for r in ok):
        rows = sorted((r for r in ok if r["model"] == model and r.get("predict_s")), key=lambda r: int(r["n_ctx"]))
        for a, b in zip(rows, rows[1:], strict=False):
            per_a = float(a["predict_s"]) / max(int(a["test_rows"]), 1)
            per_b = float(b["predict_s"]) / max(int(b["test_rows"]), 1)
            ctx_ratio = int(b["n_ctx"]) / max(int(a["n_ctx"]), 1)
            if per_a > 0 and per_b / per_a > SLOWDOWN_FACTOR * max(ctx_ratio, 1.0):
                notes.append(f"{model}: czas na wiersz {per_b / per_a:.0f}× większy przy n_ctx {int(b['n_ctx']):,} "
                             f"niż {int(a['n_ctx']):,} (kontekst ×{ctx_ratio:.1f})")
    return notes


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="13_tfm_setup.py",
        description="SOR-AI: przygotowanie modeli fundamentalnych TabPFN-3.5 i TabICLv2 — torch/CUDA, wersje, "
                    "pobranie i kontrola wag (offline), jądro SDPA, benchmark GPU i szacunki czasu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", default="tfm", help="tfm (oba), tabpfn, tabicl (aliasy pfn, icl)")
    parser.add_argument("--offline-check", action="store_true",
                        help="Tylko kontrola, bez sieci: torch/CUDA, wersje, wagi w pamięci podręcznej, SDPA")
    parser.add_argument("--skip-download", action="store_true", help="Bez pobierania wag")
    parser.add_argument("--skip-benchmark", action="store_true", help="Bez benchmarku i szacunków czasu")
    parser.add_argument("--no-browser", action="store_true",
                        help="TabPFN: bez logowania w przeglądarce (wymaga TABPFN_TOKEN)")
    parser.add_argument("--tabpfn-version", default=None, help="Wersja wag TabPFN (domyślnie z config: v3.5)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Urządzenie benchmarku; cpu = test mechaniki (brak CUDA tylko ostrzeżeniem, "
                             "konteksty > TFM_CPU_MAX_CONTEXT pomijane)")
    parser.add_argument("--contexts", default=",".join(str(c) for c in DEFAULT_CONTEXTS),
                        help="Rozmiary kontekstu benchmarku, np. 5000,25000,50000 albo 5k,25k")
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS, help="Wiersze testowe na pomiar")
    parser.add_argument("--n-estimators", default=None,
                        help="Estymatory w benchmarku: liczba albo tabpfn=4,tabicl=8 (domyślnie jak profil standard)")
    parser.add_argument("--n-jobs", type=int, default=None, help="Wątki torch na CPU (SOR_AI_N_JOBS)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _parse_models(spec: str) -> list[str]:
    chosen: list[str] = []
    for raw in str(spec).split(","):
        name = raw.strip().lower()
        if not name:
            continue
        names = TFM_NAMES if name in ("tfm", "all") else ({"pfn": "tabpfn", "icl": "tabicl"}.get(name, name),)
        for n in names:
            if n not in TFM_NAMES:
                raise ValueError(f"nieznany model TFM '{raw.strip()}' (tfm, tabpfn, tabicl)")
            if n not in chosen:
                chosen.append(n)
    return [m for m in TFM_NAMES if m in chosen] or list(TFM_NAMES)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    apply_n_jobs(argv)
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(errors="replace")          # konsola Windows bez UTF-8: bez UnicodeEncodeError
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        models = _parse_models(args.models)
        contexts = parse_int_list(args.contexts)
        n_estimators = parse_estimators(args.n_estimators, models)
    except ValueError as exc:
        parser.error(str(exc))
    if args.test_rows < 1:
        parser.error("--test-rows musi być ≥ 1")
    if args.device != "auto":                             # urządzenie wrapperów TFM (device_for czyta na bieżąco)
        for model in models:
            os.environ[f"SOR_AI_DEVICE_{model.upper()}"] = "gpu" if args.device == "cuda" else "cpu"

    from src.utils import config

    tabpfn_version = args.tabpfn_version or str(config.TABPFN_DEFAULT_PARAMS.get("model_version") or "v3.5")
    tabicl_checkpoint = str(config.TABICL_DEFAULT_PARAMS.get("checkpoint_version") or DEFAULT_TABICL_CHECKPOINT)
    offline = bool(args.offline_check)
    con = Console()
    print("SOR-AI — przygotowanie modeli fundamentalnych (TFM): " + ", ".join(DISPLAY[m] for m in models)
          + (" [tylko kontrola offline]" if offline else ""))
    try:
        torch_state = check_torch(con, device=args.device)
        usable = check_versions(con, models)
        check_dotenv(con)
        torch_ok = bool(torch_state["info"].get("installed")) and not str(
            torch_state["info"].get("error") or "").startswith("import torch")
        ready = [m for m in models if usable.get(m) and torch_ok]

        con.section("3–4. Wagi (pobieranie)")
        from src.models.tfm_base import weights_override_active

        if offline or args.skip_download:
            con.add("Pobieranie", STATUS_INFO, "pominięte (" + ("--offline-check" if offline else "--skip-download")
                    + ")")
        elif not ready:
            con.add("Pobieranie", STATUS_INFO, "pominięte — brak gotowej biblioteki (kroki 1–2)")
        else:
            for model in ready:
                override = weights_override_active(model)
                if override:
                    con.add(f"{model}: pobieranie", STATUS_INFO, f"pominięte — aktywne przekierowanie {override}")
                elif model == "tabpfn":
                    download_tabpfn(con, tabpfn_version, no_browser=args.no_browser)
                else:
                    download_tabicl(con, tabicl_checkpoint)
        if ready:
            weights = offline_check(con, ready, tabpfn_version=tabpfn_version, tabicl_checkpoint=tabicl_checkpoint)
        else:
            weights = {}
            con.section("5. Kontrola offline (bez sieci, jak trening)")
            con.add("Wagi", STATUS_INFO, "pominięte — brak gotowej biblioteki (kroki 1–2)")

        con.section("6. Jądro SDPA (atencja na GPU)")
        sdpa: dict[str, Any] | None = None
        if torch_state["cuda"]:
            try:
                sdpa = sdpa_check("cuda")
                report_sdpa(con, sdpa)
            except Exception as exc:  # noqa: BLE001
                con.add("SDPA", STATUS_FAIL, f"próba nieudana: {_short(exc)}")
        else:
            con.add("SDPA", STATUS_INFO, "pominięte — brak CUDA (sprawdzane tylko na GPU)")

        con.section("7. Benchmark (prawdziwe dane, wrappery projektu)")
        bench_ready = [m for m in ready if (weights.get(m) or {}).get("present")]
        fake = bool((weights_override_active(models[0]) or "").startswith("SOR_AI_TFM_FAKE"))
        if fake:                                         # atrapa: bez torch i wag (testy)
            bench_ready = list(models)
        if offline or args.skip_benchmark:
            con.add("Benchmark", STATUS_INFO, "pominięty (" + ("--offline-check" if offline else "--skip-benchmark")
                    + ")")
        elif not bench_ready:
            con.add("Benchmark", STATUS_FAIL if models else STATUS_INFO,
                    "pominięty — brak gotowego modelu (biblioteka / wagi powyżej)")
        elif not torch_state["cuda"] and args.device != "cpu" and not fake:
            con.add("Benchmark", STATUS_INFO, "pominięty — mierzy GPU, a CUDA jest niedostępna "
                    "(test mechaniki na CPU: --device cpu --contexts 500,1000 --test-rows 200)")
        else:
            device = "cuda" if torch_state["cuda"] and args.device != "cpu" else "cpu"
            n_jobs = args.n_jobs or int(config.N_JOBS)
            try:
                results, data_info = run_benchmark(con, bench_ready, contexts, args.test_rows, n_estimators,
                                                   n_jobs=n_jobs, seed=args.seed)
            except (FileNotFoundError, RuntimeError) as exc:
                con.add("Benchmark", STATUS_FAIL, f"brak przygotowanych danych: {_short(exc)}",
                        "python scripts/run_all.py --profile smoke (przygotowanie danych) albo "
                        "python scripts/02_preprocess.py")
                results, data_info = [], {}
            if results:
                curves = {m: Curve([r for r in results if r["model"] == m]) for m in bench_ready}
                estimates = {"profiles": profile_estimates(curves, data_info),
                             "experiment_a": experiment_a_estimate(curves, data_info),
                             "method": "dopasowanie(n_ctx) + wiersze × koszt wiersza(n_ctx), interpolacja log-log, "
                                       "skala liniowa z liczbą estymatorów; najdroższy kandydat profilu"}
                info = torch_state["info"]
                payload = {
                    "kind": "tfm_setup_benchmark", "version": BENCHMARK_VERSION,
                    "created": datetime.now().isoformat(timespec="seconds"), "device": device,
                    "gpu": {"name": info.get("device_name"), "total_vram_mb": info.get("total_vram_mb"),
                            "free_vram_mb": info.get("free_vram_mb")},
                    "torch": {"version": info.get("version"), "cuda_build": info.get("cuda_build")},
                    "libraries": {m: _dist_version(m) for m in bench_ready},
                    "weights": {m: {k: (weights.get(m) or {}).get(k) for k in ("path", "sha256", "override", "label")}
                                for m in bench_ready},
                    "fake": fake, "data": data_info, "n_estimators": dict(n_estimators),
                    "contexts": list(contexts), "results": results, "sdpa": sdpa, "estimates": estimates,
                }
                mechanics = mechanics_only(device, fake, payload["weights"])
                payload["mechanics_only"] = mechanics
                path = write_benchmark(payload, mechanics=mechanics is not None)
                if mechanics:
                    con.add(MECHANICS_FILE, STATUS_INFO, f"{path} — TEST MECHANIKI ({mechanics}); {BENCHMARK_FILE} "
                            f"(pomiar GPU do szacunków run_all / eksperymentu A) NIE został nadpisany")
                else:
                    con.add(BENCHMARK_FILE, STATUS_OK, str(path))
                for note in suspicious_rows(results, info.get("total_vram_mb") if device == "cuda" else None):
                    con.add("Pamięć GPU", STATUS_WARN, note,
                            "możliwe przepełnienie VRAM bez błędu OOM: Windows → Panel sterowania NVIDIA → Zarządzaj "
                            "ustawieniami 3D → Ustawienia programu → python.exe z venv → \"CUDA – Sysmem Fallback "
                            "Policy\" = \"Prefer No Sysmem Fallback\"; zamknij inne programy na karcie")
                print_estimates(con, estimates, device)
                if not any(r["status"] == "ok" for r in results):
                    con.add("Benchmark", STATUS_FAIL, "żaden pomiar się nie udał (szczegóły powyżej)")
    except KeyboardInterrupt:
        print("\nPrzerwano (Ctrl+C) — pobrane pliki zostają w pamięci podręcznej; uruchom ponownie, żeby dokończyć.")
        return 130

    print()
    failures, warnings = con.failures, con.warnings
    if failures:
        print(f"Wynik: {len(failures)} błędów, {len(warnings)} ostrzeżeń — popraw pozycje [BŁĄD ] i uruchom ponownie.")
        return 1
    from src.models.tfm_base import weights_override_active

    fake_hook = bool((weights_override_active(models[0]) or "").startswith("SOR_AI_TFM_FAKE"))
    if not torch_state["cuda"] and not fake_hook:
        print(f"Wynik: tylko test mechaniki ({len(warnings)} ostrzeżeń) — TFM NIE są gotowe do treningu (brak "
              f"CUDA; na CPU działa tylko profil smoke). Zainstaluj koło torch z CUDA (polecenie w kroku 1) i "
              f"uruchom ponownie bez --device cpu.")
        return 1
    print(f"Wynik: OK ({len(warnings)} ostrzeżeń) — TFM gotowe. Dalej: python scripts/00_check_env.py --models tfm "
          f"--profile thorough, potem python scripts/run_all.py --profile thorough --models tfm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
