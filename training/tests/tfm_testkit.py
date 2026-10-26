"""
Pomocnicze dla testów modeli fundamentalnych (TFM): dane w stylu zestawu 'triage', znaczniki pomijania
testów bez torch / tabpfn / tabicl i budowa MAŁYCH checkpointów z LOSOWYMI wagami (offline, bez sieci).

Prawdziwych wag nie da się tu pobrać — testy z prawdziwą biblioteką używają losowo zainicjalizowanej,
maleńkiej sieci przez te same ścieżki ładowania co prawdziwe wagi:
    - TabPFN: plik `.safetensors` z nazwą zawierającą "v3.5" (wersja rozpoznawana z nazwy pliku) →
      `SOR_AI_TABPFN_MODEL_PATH`,
    - TabICL: plik `.ckpt` (`{"config", "state_dict"}`, torch.save) → `SOR_AI_TABICL_MODEL_PATH`.
Predykcje takich sieci są bezwartościowe — testy sprawdzają wyłącznie mechanikę (kształty, zapis, odczyt).
Checkpointy powstają w katalogu tymczasowym testu (nigdy w repozytorium).
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TABPFN = HAS_TORCH and importlib.util.find_spec("tabpfn") is not None
HAS_TABICL = HAS_TORCH and importlib.util.find_spec("tabicl") is not None

requires_tabpfn = pytest.mark.skipif(not HAS_TABPFN, reason="brak torch / tabpfn (requirements-tfm.txt)")
requires_tabicl = pytest.mark.skipif(not HAS_TABICL, reason="brak torch / tabicl (requirements-tfm.txt)")

TABPFN_TINY_NAME = "tabpfn-v3.5-random-tiny.safetensors"   # "v3.5" w nazwie — wersja z nazwy pliku
TABICL_TINY_NAME = "tabicl-tiny-random.ckpt"

# Maleńka architektura TabPFN-3.5 (= fixture upstream tests/test_model_specs.py)
_TABPFN_TINY = dict(embed_dim=16, nlayers=1, icl_num_heads=2, icl_num_kv_heads_test=1, dist_embed_num_heads=2,
                    dist_embed_num_blocks=1, dist_embed_num_inducing_points=4, feat_agg_num_heads=2,
                    feat_agg_num_blocks=1, feat_agg_num_cls_tokens=1)
# Konfiguracja klasyfikatora TabICLv2 (skrypty treningu v2) pomniejszona do ~120 tys. parametrów
_TABICL_V2 = {
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
_TABICL_TINY = {**_TABICL_V2, "embed_dim": 32, "col_num_blocks": 1, "col_nhead": 4, "col_num_inds": 16,
                "row_num_blocks": 1, "row_nhead": 4, "row_num_cls": 2, "icl_num_blocks": 2, "icl_nhead": 4}


def write_tiny_tabpfn(directory: Path | str, seed: int = 0) -> Path:
    """Plik wag TabPFN-3.5 z losową, maleńką siecią (ta sama ścieżka ładowania co prawdziwe wagi)."""
    import torch
    from tabpfn.architectures import tabpfn_v3_5
    from tabpfn.checkpoint import save_as_safetensors
    from tabpfn.constants import ModelVersion
    from tabpfn.inference_config import InferenceConfig, cpu_sample_limit

    path = Path(directory) / TABPFN_TINY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prawdziwy InferenceConfig jest w checkpoincie (niedostępny offline) — najbliższy publiczny odpowiednik:
    # domyślny v2.5 z limitami v3.5 (README: 1 mln wierszy, 20 tys. cech) i limitem CPU wersji
    inference = dataclasses.replace(
        InferenceConfig.get_default("multiclass", ModelVersion.V2_5),
        MAX_NUMBER_OF_SAMPLES=1_000_000, MAX_NUMBER_OF_FEATURES=20_000,
        MAX_CPU_SAMPLES=cpu_sample_limit(ModelVersion.V3_5),
    )
    torch.manual_seed(seed)
    config = tabpfn_v3_5.TabPFNV3p5Config(max_num_classes=10, num_buckets=100, **_TABPFN_TINY)
    model = tabpfn_v3_5.get_architecture(config).eval()
    save_as_safetensors({"state_dict": model.state_dict(), "architecture_name": "tabpfn_v3_5",
                         "config": dataclasses.asdict(config), "inference_config": dataclasses.asdict(inference)},
                        path)
    return path


def write_tiny_tabicl(directory: Path | str, seed: int = 0) -> Path:
    """Checkpoint TabICL (`{"config", "state_dict"}`) z losową, maleńką siecią."""
    import torch
    from tabicl._model.tabicl import TabICL  # moduł prywatny — stabilny w 2.2.0

    path = Path(directory) / TABICL_TINY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = TabICL(**_TABICL_TINY).eval()
    torch.save({"config": dict(_TABICL_TINY), "state_dict": model.state_dict()}, path)
    return path


# ─────────────────────────────────────────
# Dane w stylu zestawu 'triage'
# ─────────────────────────────────────────
CC_CRITICAL_SAMPLE = ("cc_cardiacarrest", "cc_strokealert")


def triage_frame(n: int = 600, seed: int = 0, n_cc_flags: int = 12) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    (X, y, oddziały): parametry życiowe z brakami + flagi `*_missing`, wiek, przyjazd, skale (news2), agregaty
    skarg (cc_count, cc_critical, cc_group_*), flagi CC_CRITICAL i `n_cc_flags` flag skarg o malejącej częstości;
    y = 0–4 (ESI 1 rzadkie, ale ≥ 4 wiersze), oddziały A/B/C.
    """
    rng = np.random.default_rng(seed)
    p = np.array([0.03, 0.27, 0.40, 0.22, 0.08])
    y = rng.choice(5, size=n, p=p)
    y[:4] = 0
    y[4:8] = 4
    rng.shuffle(y)
    frame = {
        "age": rng.integers(18, 95, n).astype(float),
        "arrival_ambulance": (rng.random(n) < 0.5 - 0.08 * y).astype(float),
        "triage_vital_hr": rng.normal(110 - 7 * y, 12, n),
        "triage_vital_sbp": rng.normal(130, 20, n),
        "triage_vital_rr": rng.normal(23 - 1.2 * y, 3, n),
        "triage_vital_o2": np.clip(rng.normal(92 + 1.3 * y, 2.5, n), 70, 100),
        "news2_score": np.clip(rng.poisson(6 - y), 0, 15).astype(float),
    }
    for vital in ("hr", "sbp", "rr"):
        col = f"triage_vital_{vital}"
        missing = rng.random(n) < 0.15
        frame[col] = np.where(missing, np.nan, frame[col])
        frame[f"{vital}_missing"] = missing.astype(float)
    flags = {}
    for i in range(n_cc_flags):
        prevalence = 0.4 / (i + 1)
        flags[f"cc_flag{i:02d}"] = (rng.random(n) < prevalence).astype(float)
    for name in CC_CRITICAL_SAMPLE:
        flags[name] = ((rng.random(n) < 0.01) | (y == 0) & (rng.random(n) < 0.3)).astype(float)
    cc = pd.DataFrame(flags)
    frame["cc_count"] = cc.sum(axis=1).to_numpy()
    frame["cc_critical"] = cc[list(CC_CRITICAL_SAMPLE)].sum(axis=1).to_numpy()
    frame["cc_group_cardiac"] = (cc.iloc[:, :3].sum(axis=1) > 0).astype(float).to_numpy()
    X = pd.concat([pd.DataFrame(frame), cc], axis=1).astype(np.float32)
    site = rng.choice(np.array(["A", "B", "C"]), size=n, p=[0.55, 0.3, 0.15])
    return X, pd.Series(y, name="target"), site


def site_strata(y: pd.Series | np.ndarray, site: np.ndarray) -> np.ndarray:
    """Etykiety ESI × oddział jak `train._site_strata`."""
    return np.char.add(np.char.add(np.asarray(y).astype(int).astype(str), "_"), np.asarray(site).astype(str))
