"""
TabICLv2 (Inria, `tabicl==2.2.0`, checkpoint `tabicl-classifier-v2-20260212.ckpt`) — tabularny model
fundamentalny (in-context learning). Kod: BSD-3; licencja wag — karta modelu `jingang/TabICL` na HF.

Wspólna logika (worki kontekstu, korekta priorów, porcje testu, drabina OOM, artefakt): `tfm_base.py`.
Tu tylko adapter biblioteki:
    - wagi: `hf_hub_download(..., local_files_only=True)` (bez sieci) → jawna ścieżka `model_path`
      + `allow_auto_download=False`; `checkpoint_version` zostaje nazwą znaną bibliotece,
    - bufor KV: "kv" (pełny) tylko gdy modelowany rozmiar ≤ TABICL_KV_VRAM_FRACTION × całkowita VRAM
      (na CPU: × budżet RAM hosta), inaczej "repr"; `batch_size` (estymatorów na przebieg) = 1 przy
      n_ctx ≥ 10 000, inaczej 4; `predict_batch="auto"` — z budżetu RAM hosta (wyjście osadzeń kolumn
      fp32: batch_size × (cechy + 4) × 128 × 4 B na wiersz),
    - braki: TabICL imputuje średnią kontekstu — flagi `*_missing` zostają w cechach,
    - drabina OOM: (a) batch_size=1, (b) kv_cache "kv" → "repr" → bez bufora, (c) predict_batch / 2
      (do 1024), (d) błąd. Wątki torch ustawia wrapper (n_jobs TabICL działa tylko w predict).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.models.tfm_base import (
    MIN_PREDICT_BATCH,
    TFMAdapter,
    TFMTriageModel,
    TFMWeightsError,
    weights_identity,
)
from src.utils import config as _config
from src.utils.logger import get_logger

log = get_logger(__name__)

REPO_ID = "jingang/TabICL"
KV_BYTES_PER_ROW_FP16 = 24_576           # 12 warstw × K,V × 8 głów × 64 × 2 B (AMP na CUDA)
COL_BYTES_PER_COLUMN_FP16 = 196_608      # bufor kolumn: 3 × 2 × 128 × 128 × 2 B, kolumny = cechy + 4
REPR_BYTES_PER_ROW = 2_048               # "repr": reprezentacje wierszy (fp32)
COL_EMBED_BYTES_PER_CELL = 128 * 4       # wyjście osadzeń kolumn: 128 × fp32 na (wiersz × kolumnę)
MAX_AUTO_PREDICT_BATCH = 65_536


def cached_checkpoint(checkpoint_version: str) -> Path:
    """Checkpoint z lokalnej pamięci podręcznej HF (bez sieci); brak → TFMWeightsError z poleceniem."""
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=REPO_ID, filename=checkpoint_version, local_files_only=True))
    except ImportError:
        raise
    except Exception as exc:  # noqa: BLE001 — LocalEntryNotFoundError i pochodne (brak w cache)
        raise TFMWeightsError(
            f"Brak checkpointu TabICL '{checkpoint_version}' w lokalnej pamięci podręcznej Hugging Face "
            f"({type(exc).__name__}). Pobierz go raz: python scripts/13_tfm_setup.py — potem trening działa "
            f"offline.") from None


class TabICLAdapter(TFMAdapter):
    library = "tabicl"
    display_name = "TabICLv2"
    weights_env = "SOR_AI_TABICL_MODEL_PATH"

    # ── wagi ──
    def resolve_weights(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if self.fake:
            return super().resolve_weights(params)
        name = str(params.get("checkpoint_version") or "tabicl-classifier-v2-20260212.ckpt")
        override = self._override_path()
        path = override if override is not None else cached_checkpoint(name)
        ident = weights_identity(path)
        ident.update({"checkpoint_version": name, "override": None})
        return ident

    # ── opcje ──
    @staticmethod
    def kv_cache_bytes(n_estimators: int, n_ctx: int, n_features: int, mode: str, device: str) -> int:
        """Modelowany bufor wszystkich estymatorów worka (fp16 pod AMP na CUDA, fp32 na CPU)."""
        scale = 1 if str(device).startswith(("cuda", "gpu")) else 2
        column = COL_BYTES_PER_COLUMN_FP16 * (n_features + 4) * scale
        if mode == "kv":
            return int(n_estimators * (KV_BYTES_PER_ROW_FP16 * scale * n_ctx + column))
        if mode == "repr":
            return int(n_estimators * (REPR_BYTES_PER_ROW * n_ctx + column))
        return 0

    def _auto_kv(self, n_estimators: int, n_ctx: int, n_features: int, device: str) -> str:
        need = self.kv_cache_bytes(n_estimators, n_ctx, n_features, "kv", device)
        if str(device).startswith(("cuda", "gpu")):
            total = None
            if not self.fake:
                from src.utils.hardware import torch_info

                total_mb = torch_info().get("total_vram_mb")
                total = None if not total_mb else int(total_mb) * 2**20
            if total is None:
                return "repr"
            budget = float(_config.TABICL_KV_VRAM_FRACTION) * total
        else:
            budget = float(_config.TABICL_KV_VRAM_FRACTION) * float(_config.TFM_HOST_RAM_BUDGET_GB) * 2**30
        return "kv" if need <= budget else "repr"

    @staticmethod
    def auto_predict_batch(batch_size: int, n_features: int, n_ctx: int, kv_cache: Any) -> int:
        """Wierszy na wywołanie tak, by wyjście osadzeń kolumn mieściło się w TFM_HOST_RAM_BUDGET_GB."""
        budget = float(_config.TFM_HOST_RAM_BUDGET_GB) * 2**30
        per_row = max(1, int(batch_size)) * (int(n_features) + 4) * COL_EMBED_BYTES_PER_CELL
        rows = int(budget // per_row)
        if not kv_cache:                       # bez bufora każde wywołanie koduje też cały kontekst
            rows -= int(n_ctx)
        return int(min(MAX_AUTO_PREDICT_BATCH, max(MIN_PREDICT_BATCH, rows)))

    def initial_runtime(self, params: Mapping[str, Any], *, n_ctx: int, n_features: int,
                        device: str) -> dict[str, Any]:
        n_est = int(params.get("n_estimators") or 8)
        batch = params.get("batch_size", "auto")
        batch_size = (1 if int(n_ctx) >= 10_000 else 4) if str(batch).lower() == "auto" else int(batch)
        batch_size = max(1, min(batch_size, n_est))
        kv = params.get("kv_cache", "auto")
        if isinstance(kv, str) and kv.lower() == "auto":
            kv = self._auto_kv(n_est, int(n_ctx), int(n_features), device)
        elif kv is True or (isinstance(kv, str) and kv.lower() == "kv"):
            kv = "kv"
        elif isinstance(kv, str) and kv.lower() == "repr":
            kv = "repr"
        elif kv in (False, None) or (isinstance(kv, str) and kv.lower() in ("false", "none", "off")):
            kv = False
        else:
            raise ValueError(f"tabicl: kv_cache 'auto' | 'kv' | 'repr' | False, podano {kv!r}")
        pb = params.get("predict_batch", "auto")
        predict_batch = (self.auto_predict_batch(batch_size, n_features, n_ctx, kv)
                         if str(pb).lower() == "auto" else max(1, int(pb)))
        return {"batch_size": batch_size, "kv_cache": kv, "predict_batch": predict_batch}

    def estimator_options(self, params: Mapping[str, Any], runtime: Mapping[str, Any], weights: Mapping[str, Any],
                          device: str, seed: int) -> dict[str, Any]:
        offload_dir = params.get("disk_offload_dir")
        return {
            "n_estimators": int(params.get("n_estimators") or 8),
            "batch_size": int(runtime["batch_size"]),
            "kv_cache": runtime["kv_cache"],
            "model_path": None if weights.get("path") is None else str(weights["path"]),
            "allow_auto_download": False,
            "checkpoint_version": str(params.get("checkpoint_version") or "tabicl-classifier-v2-20260212.ckpt"),
            "device": "cuda" if str(device).startswith(("cuda", "gpu")) else "cpu",
            "use_amp": params.get("use_amp", "auto"),
            "offload_mode": params.get("offload_mode", "auto"),
            "disk_offload_dir": None if offload_dir is None else str(offload_dir),   # TabICL wymaga str
            "random_state": int(seed),
            "n_jobs": None,                   # wątki ustawia wrapper (n_jobs TabICL działa tylko w predict)
            "verbose": False,
        }

    def _make_estimator(self, options: dict[str, Any]) -> Any:
        from tabicl import TabICLClassifier

        return TabICLClassifier(**options)

    def describe(self, est: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        caches = getattr(est, "model_kv_cache_", None)
        if isinstance(caches, dict) and caches:
            try:
                out["kv_cache_mb"] = int(sum(c.cache_size_mb() for c in caches.values()))
            except Exception:  # noqa: BLE001
                pass
        for attr in ("kv_cache", "batch_size", "n_estimators"):
            value = getattr(est, attr, None)
            if value is not None:
                out[attr] = value
        return out

    # ── OOM ──
    def next_oom_step(self, runtime: Mapping[str, Any], phase: str) -> tuple[dict[str, Any], str, bool] | None:
        rt = dict(runtime)
        if int(rt["batch_size"]) > 1:
            old = rt["batch_size"]
            rt["batch_size"] = 1
            return rt, f"batch_size {old} → 1 (jeden estymator na przebieg)", True
        if rt["kv_cache"] == "kv":
            rt["kv_cache"] = "repr"
            return rt, "kv_cache 'kv' → 'repr'", True
        if rt["kv_cache"] == "repr":
            rt["kv_cache"] = False
            return rt, "kv_cache 'repr' → bez bufora", True
        batch = int(rt["predict_batch"])
        if batch > MIN_PREDICT_BATCH:
            rt["predict_batch"] = max(MIN_PREDICT_BATCH, batch // 2)
            return rt, f"predict_batch {batch} → {rt['predict_batch']}", False
        return None

    def oom_advice(self) -> str:
        return "n_ctx, n_estimators, max_features (np. 100) albo predict_batch"

    def reencodes_context(self, runtime: Mapping[str, Any]) -> bool:
        """'repr' / bez bufora: transformer ICL liczony od nowa na całym kontekście przy każdym wywołaniu."""
        return runtime.get("kv_cache") != "kv"

    def host_cache_runtime(self, runtime: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
        """TabICL trzyma bufor na urządzeniu — mniejszy bufor: 'kv' → 'repr' (~12× mniej VRAM na wiersz)."""
        rt = dict(runtime)
        if rt.get("kv_cache") != "kv":
            return None
        rt["kv_cache"] = "repr"
        return rt, "kv_cache 'kv' → 'repr' (mniejszy bufor; kontekst ICL liczony przy każdym wywołaniu)"

    def cache_bytes(self, params: Mapping[str, Any], runtime: Mapping[str, Any], n_ctx: int,
                    n_features: int) -> int | None:
        return self.kv_cache_bytes(int(params.get("n_estimators") or 8), int(n_ctx), int(n_features),
                                   runtime.get("kv_cache") or "", "cuda")


class TabICLTriageModel(TFMTriageModel):
    """TabICLv2 dla 5-klasowej klasyfikacji ESI (kontekst z TRAIN, predykcja na GPU)."""

    name = "tabicl"
    display_name = "TabICLv2"
    defaults_name = "TABICL_DEFAULT_PARAMS"


__all__ = ["REPO_ID", "TabICLAdapter", "TabICLTriageModel", "cached_checkpoint"]
