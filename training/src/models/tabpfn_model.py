"""
TabPFN-3.5 (Prior Labs, `tabpfn==9.2.0`) — tabularny model fundamentalny (in-context learning).

Licencja wag TabPFN-3.5: NIEKOMERCYJNA (praca magisterska — dozwolone); kod biblioteki: Apache-2.0.

Wspólna logika (worki kontekstu, korekta priorów, porcje testu, drabina OOM, artefakt): `tfm_base.py`.
Tu tylko adapter biblioteki:
    - wagi: parametr wrappera `model_version` ("v3.5" domyślnie, "v3.5-fast", "v3") → BEZWZGLĘDNA ścieżka
      pliku w pamięci podręcznej TabPFN (`create_default_for_version`, bez sieci); brak pliku → polecenie
      `python scripts/13_tfm_setup.py` (biblioteka offline zgłosiłaby mylący błąd "model is gated"),
    - estymator: `TabPFNClassifier(model_path=…, device="cuda:0"|"cpu", fit_mode="fit_with_cache", …)`;
      nigdy `n_jobs` (przestarzałe) ani `auto_scale_n_estimators`; tuning_config / eval_metric = None;
      random_state jako zwykły int (z random_state wrappera + numer worka),
    - ustawienia procesu (przywracane w finally): TABPFN_MODEL_CACHE_SIZE=1 na czas predykcji (wagi
      ładowane raz dla wszystkich worków; potem `clear_built_model_cache()`),
      `settings.tabpfn.max_batched_test_rows` = min(32768, predict_batch),
      `settings.tabpfn.max_batched_estimator_rows` = 0 po kroku awaryjnym (jeden bufor naraz),
    - drabina OOM: (a) predict_batch / 2 (do 1024; max_batched_test_rows za nim), (b) keep_cache_on_device=False
      + max_batched_estimator_rows=0, (c) fit_mode="fit_preprocessors" z predict_batch=16384, (d) błąd.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from src.models.tfm_base import (
    MIN_PREDICT_BATCH,
    TFMAdapter,
    TFMTriageModel,
    TFMWeightsError,
    weights_identity,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_VERSIONS: tuple[str, ...] = ("v3.5", "v3.5-fast", "v3")
MAX_BATCHED_TEST_ROWS = 32_768            # domyślne TABPFN_MAX_BATCHED_TEST_ROWS
FALLBACK_PREDICT_BATCH = 16_384           # krok (c): fit_preprocessors
CACHE_SIZE_ENV = "TABPFN_MODEL_CACHE_SIZE"


def default_weights_path(model_version: str) -> Path:
    """Bezwzględna ścieżka pliku wag wersji w pamięci podręcznej TabPFN (bez sieci i bez wczytywania)."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return Path(TabPFNClassifier.create_default_for_version(ModelVersion(model_version)).model_path)


class TabPFNAdapter(TFMAdapter):
    library = "tabpfn"
    display_name = "TabPFN-3.5"
    weights_env = "SOR_AI_TABPFN_MODEL_PATH"

    # ── wagi ──
    def resolve_weights(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if self.fake:
            return super().resolve_weights(params)
        version = str(params.get("model_version") or "v3.5")
        if version not in MODEL_VERSIONS:
            raise ValueError(f"tabpfn: model_version musi być jednym z {MODEL_VERSIONS}, podano {version!r}")
        override = self._override_path()
        if override is not None:
            path = override
            if version not in path.name:
                log.warning(f"TabPFN: plik {path.name} nie zawiera '{version}' w nazwie — biblioteka rozpoznaje "
                            f"wersję architektury po nazwie pliku")
        else:
            path = default_weights_path(version)
            if not path.is_file():
                raise TFMWeightsError(
                    f"Brak wag TabPFN {version}: {path}. Pobierz je raz (akceptacja licencji niekomercyjnej, "
                    f"TABPFN_TOKEN): python scripts/13_tfm_setup.py — potem trening działa offline.")
        ident = weights_identity(path)
        ident.update({"version": version, "override": None})
        return ident

    # ── opcje ──
    def initial_runtime(self, params: Mapping[str, Any], *, n_ctx: int, n_features: int,
                        device: str) -> dict[str, Any]:
        fit_mode = str(params.get("fit_mode") or "fit_with_cache")
        if fit_mode not in ("fit_with_cache", "fit_preprocessors", "low_memory"):
            raise ValueError(f"tabpfn: fit_mode {fit_mode!r} nieobsługiwany (fit_with_cache | fit_preprocessors "
                             f"| low_memory)")
        return {
            "predict_batch": max(1, int(params.get("predict_batch") or MAX_BATCHED_TEST_ROWS)),
            "fit_mode": fit_mode,
            "keep_cache_on_device": bool(params.get("keep_cache_on_device", True)),
            "one_cache_at_a_time": False,
        }

    def estimator_options(self, params: Mapping[str, Any], runtime: Mapping[str, Any], weights: Mapping[str, Any],
                          device: str, seed: int) -> dict[str, Any]:
        n_est = params.get("n_estimators", "auto")
        return {
            "n_estimators": "auto" if str(n_est).lower() == "auto" else int(n_est),
            "model_path": None if weights.get("path") is None else str(weights["path"]),
            "device": "cuda:0" if str(device).startswith(("cuda", "gpu")) else "cpu",
            "fit_mode": runtime["fit_mode"],
            "keep_cache_on_device": bool(runtime["keep_cache_on_device"]),
            "kv_cache_precision": params.get("kv_cache_precision"),
            "memory_saving_mode": params.get("memory_saving_mode", "auto"),
            "inference_precision": params.get("inference_precision", "auto"),
            "random_state": int(seed),
            "ignore_pretraining_limits": False,
            "n_preprocessing_jobs": 1,
        }

    def _make_estimator(self, options: dict[str, Any]) -> Any:
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(**options)

    def describe(self, est: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for attr in ("n_estimators_", "use_autocast_", "fit_mode"):
            value = getattr(est, attr, None)
            if value is not None:
                out[attr] = value
        configs = getattr(est, "configs_", None)
        if configs:
            out["architecture"] = getattr(configs[0], "name", None)
        return out

    # ── ustawienia procesu ──
    def _settings_target(self) -> Any:
        if self.fake:
            from src.models.tfm_fake import FAKE_SETTINGS

            return FAKE_SETTINGS.tabpfn
        from tabpfn.settings import settings

        return settings.tabpfn

    @contextlib.contextmanager
    def process_settings(self) -> Iterator[None]:
        """TABPFN_MODEL_CACHE_SIZE=1 na czas wywołania: model zbudowany raz dla wszystkich worków."""
        set_here = CACHE_SIZE_ENV not in os.environ
        if set_here:
            os.environ[CACHE_SIZE_ENV] = "1"
        try:
            yield
        finally:
            if set_here:
                os.environ.pop(CACHE_SIZE_ENV, None)
                if not self.fake:
                    try:
                        from tabpfn.model_loading import clear_built_model_cache

                        clear_built_model_cache()
                    except Exception:  # noqa: BLE001 — sprzątanie nie może przykryć błędu
                        pass

    @contextlib.contextmanager
    def library_settings(self, runtime: Mapping[str, Any]) -> Iterator[None]:
        target = self._settings_target()
        saved = {name: getattr(target, name) for name in ("max_batched_test_rows", "max_batched_estimator_rows")}
        try:
            target.max_batched_test_rows = int(min(MAX_BATCHED_TEST_ROWS, int(runtime["predict_batch"])))
            if runtime.get("one_cache_at_a_time"):
                target.max_batched_estimator_rows = 0
            yield
        finally:
            for name, value in saved.items():
                setattr(target, name, value)

    # ── OOM ──
    def next_oom_step(self, runtime: Mapping[str, Any], phase: str) -> tuple[dict[str, Any], str, bool] | None:
        rt = dict(runtime)
        batch = int(rt["predict_batch"])
        if phase == "predict" and batch > MIN_PREDICT_BATCH:
            rt["predict_batch"] = max(MIN_PREDICT_BATCH, batch // 2)
            return rt, (f"predict_batch {batch} → {rt['predict_batch']} (max_batched_test_rows "
                        f"{min(MAX_BATCHED_TEST_ROWS, rt['predict_batch'])})"), False
        if rt["fit_mode"] == "fit_with_cache" and (rt["keep_cache_on_device"] or not rt["one_cache_at_a_time"]):
            rt["keep_cache_on_device"] = False
            rt["one_cache_at_a_time"] = True
            return rt, "keep_cache_on_device=False + max_batched_estimator_rows=0 (jeden bufor KV naraz)", True
        if rt["fit_mode"] == "fit_with_cache":
            rt["fit_mode"] = "fit_preprocessors"
            rt["predict_batch"] = FALLBACK_PREDICT_BATCH
            return rt, f"fit_mode=fit_preprocessors, predict_batch={FALLBACK_PREDICT_BATCH}", True
        return None

    def oom_advice(self) -> str:
        return "n_ctx, n_estimators albo predict_batch (ewentualnie model_version='v3.5-fast')"

    def cache_on_device(self, runtime: Mapping[str, Any]) -> bool:
        return bool(runtime.get("keep_cache_on_device", True))

    def reencodes_context(self, runtime: Mapping[str, Any]) -> bool:
        """fit_preprocessors / low_memory: kontekst przechodzi przez transformer przy każdym wywołaniu."""
        return runtime.get("fit_mode") != "fit_with_cache"

    def host_cache_runtime(self, runtime: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
        rt = dict(runtime)
        if rt.get("fit_mode") != "fit_with_cache" or (not rt.get("keep_cache_on_device", True)
                                                       and rt.get("one_cache_at_a_time")):
            return None
        rt["keep_cache_on_device"] = False
        rt["one_cache_at_a_time"] = True
        return rt, "bufory KV w RAM (keep_cache_on_device=False + max_batched_estimator_rows=0)"

    def cache_bytes(self, params: Mapping[str, Any], runtime: Mapping[str, Any], n_ctx: int,
                    n_features: int) -> int | None:
        """Bufor KV v3.5 (plan pamięci): 3840 B/wiersz (int8) lub 6912 B (fp16) + 294 912 B/kolumnę modelu."""
        if runtime.get("fit_mode") != "fit_with_cache":
            return 0
        n_est = params.get("n_estimators", "auto")
        n_est = 8 if str(n_est).lower() == "auto" else int(n_est)
        columns = n_features + math.ceil(n_features / 4) + 1          # + SVD + odcisk (zachowawczo)
        per_row = 6_912 if params.get("kv_cache_precision") == "auto" else 3_840
        return int(n_est * (int(n_ctx) * per_row + 294_912 * columns))


class TabPFNTriageModel(TFMTriageModel):
    """TabPFN-3.5 dla 5-klasowej klasyfikacji ESI (kontekst z TRAIN, predykcja na GPU)."""

    name = "tabpfn"
    display_name = "TabPFN-3.5"
    defaults_name = "TABPFN_DEFAULT_PARAMS"


__all__ = ["MODEL_VERSIONS", "TabPFNAdapter", "TabPFNTriageModel", "default_weights_path"]
