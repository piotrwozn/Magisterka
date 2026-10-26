"""
Testy scripts/13_tfm_setup.py — przygotowanie modeli fundamentalnych (TFM).

Bez sieci i bez GPU: stan torch / CUDA, wersje bibliotek i pobieranie są podstawiane (monkeypatch), benchmark
działa na syntetycznych przygotowanych danych z atrapą estymatora (SOR_AI_TFM_FAKE=1 — także w venv bez torch),
a z prawdziwymi bibliotekami (tylko venv z torch / tabpfn / tabicl) na maleńkich checkpointach z losowymi
wagami (`tests/tfm_testkit.py`). Brak torch w procesie symulowany przez `sys.modules['torch'] = None`.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

from tests.tfm_testkit import HAS_TABICL, HAS_TABPFN, HAS_TORCH, requires_tabicl, requires_tabpfn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "13_tfm_setup.py"

TORCH_CUDA = {"installed": True, "version": "2.14.1+cu130", "cuda_build": "13.0", "cuda_available": True,
              "device_name": "NVIDIA GeForce RTX 4080 SUPER", "total_vram_mb": 16376, "free_vram_mb": 15000,
              "error": None}
TORCH_NO_GPU = {**TORCH_CUDA, "cuda_available": False, "device_name": None, "total_vram_mb": None,
                "free_vram_mb": None}
TORCH_CPU_WHEEL = {**TORCH_NO_GPU, "version": "2.14.1+cpu", "cuda_build": None}
TORCH_MISSING = {**TORCH_CPU_WHEEL, "installed": False, "version": None, "error": "torch nie jest zainstalowany"}


@pytest.fixture(scope="module")
def setup():
    spec = importlib.util.spec_from_file_location("tfm_setup_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env(monkeypatch):
    """Zmienne, które skrypt ustawia / czyta — przywracane po teście."""
    for var in ("SOR_AI_TFM_FAKE", "SOR_AI_TABPFN_MODEL_PATH", "SOR_AI_TABICL_MODEL_PATH", "SOR_AI_DEVICE",
                "SOR_AI_DEVICE_TABPFN", "SOR_AI_DEVICE_TABICL", "TABPFN_NO_BROWSER", "TABPFN_TOKEN",
                "TABPFN_MODEL_CACHE_DIR", "SOR_AI_N_JOBS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    return monkeypatch


def _status(con, name: str) -> list[str]:
    return [e["status"] for e in con.entries if e["name"] == name]


# ─────────────────────────────────────────
# Funkcje pomocnicze
# ─────────────────────────────────────────
class TestParsing:
    def test_contexts(self, setup):
        assert setup.parse_int_list("5k, 25000,5000,50_000") == [5_000, 25_000, 50_000]
        for bad in ("", "0", "-5", "abc"):
            with pytest.raises(ValueError):
                setup.parse_int_list(bad)

    def test_models(self, setup):
        assert setup._parse_models("tfm") == ["tabpfn", "tabicl"]
        assert setup._parse_models("icl") == ["tabicl"]
        assert setup._parse_models("tabicl,pfn") == ["tabpfn", "tabicl"]
        with pytest.raises(ValueError):
            setup._parse_models("lightgbm")

    def test_estimators_default_from_standard_profile(self, setup):
        from src.utils import config

        expected = {m: config.TUNING_PROFILES["standard"][m]["candidates"][0]["n_estimators"] for m in ("tabpfn",
                                                                                                         "tabicl")}
        assert setup.parse_estimators(None, ["tabpfn", "tabicl"]) == expected
        assert setup.parse_estimators("2", ["tabpfn", "tabicl"]) == {"tabpfn": 2, "tabicl": 2}
        assert setup.parse_estimators("icl=3", ["tabpfn", "tabicl"])["tabicl"] == 3
        for bad in ("x=2", "0"):
            with pytest.raises(ValueError):
                setup.parse_estimators(bad, ["tabpfn"])

    def test_cli_errors_exit_2(self, setup, clean_env):
        for argv in (["--contexts", "abc"], ["--models", "xgboost"], ["--test-rows", "0"]):
            with pytest.raises(SystemExit) as exc:
                setup.main(argv)
            assert exc.value.code == 2


class TestTorchCheck:
    @pytest.mark.parametrize("driver, cpu_only, present, absent", [
        (None, False, ["whl/cu130", "whl/cu128"], ["uninstall"]),
        (581, True, ["pip uninstall -y torch", "whl/cu130"], ["whl/cu128"]),
        (560, False, ["whl/cu128", "560 < 580"], ["whl/cu130"]),
    ])
    def test_install_commands(self, setup, driver, cpu_only, present, absent):
        text = "\n".join(setup.torch_install_commands(driver, cpu_only))
        assert all(p in text for p in present) and not any(a in text for a in absent)
        assert "torch.cuda.is_available()" in text

    def test_cuda_ok(self, setup):
        con = setup.Console()
        out = setup.check_torch(con, device="auto", torch_info=lambda: dict(TORCH_CUDA),
                                smi={"name": "RTX", "driver_version": "581.57", "cuda_version": "13.0"})
        assert out["cuda"] and out["device"] == "cuda" and out["driver"] == 581
        assert _status(con, "torch") == ["ok"]

    def test_cpu_only_wheel_prints_exact_commands(self, setup, capsys):
        con = setup.Console()
        out = setup.check_torch(con, device="auto", torch_info=lambda: dict(TORCH_CPU_WHEEL),
                                smi={"name": "RTX", "driver_version": "581.57"})
        assert not out["cuda"] and _status(con, "torch") == ["fail"]
        printed = capsys.readouterr().out
        assert "BEZ CUDA" in printed and "pip uninstall -y torch" in printed
        assert "pip install torch --index-url https://download.pytorch.org/whl/cu130" in printed

    def test_no_cuda_is_warning_with_explicit_cpu(self, setup):
        con = setup.Console()
        out = setup.check_torch(con, device="cpu", torch_info=lambda: dict(TORCH_NO_GPU), smi=None)
        assert not out["cuda"] and out["device"] == "cpu" and _status(con, "torch") == ["warn"]

    @pytest.mark.parametrize("info", [TORCH_MISSING, {**TORCH_CUDA, "error": "import torch: OSError: DLL load failed"}])
    def test_missing_or_broken_torch(self, setup, info, capsys):
        con = setup.Console()
        setup.check_torch(con, device="cpu", torch_info=lambda: dict(info), smi=None)
        assert _status(con, "torch") == ["fail"]
        assert "pip install torch --index-url" in capsys.readouterr().out


class TestVersionsAndDotenv:
    def test_versions(self, setup):
        con = setup.Console()
        usable = setup.check_versions(con, ["tabpfn", "tabicl"],
                                      version_of={"tabpfn": "9.1.0", "huggingface-hub": "2.3.0"}.get)
        assert usable == {"tabpfn": False, "tabicl": False}
        assert _status(con, "tabpfn") == ["fail"] and _status(con, "tabicl") == ["fail"]
        con = setup.Console()
        assert setup.check_versions(con, ["tabicl"], version_of={"tabicl": "2.2.1"}.get) == {"tabicl": True}

    def test_dotenv_with_tabpfn_keys_warns(self, setup, tmp_path, clean_env):
        (tmp_path / ".env").write_text("# komentarz\nOLLAMA_URL=x\nexport TABPFN_MODEL_CACHE_DIR=/inna\n",
                                       encoding="utf-8")
        con = setup.Console()
        assert setup.check_dotenv(con, tmp_path) == ["TABPFN_MODEL_CACHE_DIR"]
        assert _status(con, "Plik .env") == ["warn"]
        con = setup.Console()
        assert setup.check_dotenv(con, tmp_path / "brak") == [] and con.entries == []


class TestDownloads:
    class TabPFNLicenseError(Exception):
        pass

    class TabPFNHuggingFaceGatedRepoError(Exception):
        pass

    def _gated_offline(self):
        try:
            try:
                raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")
            except urllib.error.URLError as inner:
                raise self.TabPFNHuggingFaceGatedRepoError("model is gated") from inner
        except self.TabPFNHuggingFaceGatedRepoError as exc:
            return exc

    def test_classify(self, setup):
        assert setup.classify_download_error(self._gated_offline())[0] == "network"     # mylące „gated” offline
        assert setup.classify_download_error(ConnectionError("reset"))[0] == "network"
        assert setup.classify_download_error(
            self.TabPFNLicenseError("Could not reach the license server to verify"))[0] == "network"
        kind, text = setup.classify_download_error(self.TabPFNLicenseError("no interactive terminal"))
        assert kind == "licence" and "TABPFN_TOKEN" in text and "ux.priorlabs.ai" in text
        kind, text = setup.classify_download_error(self.TabPFNHuggingFaceGatedRepoError("gated, accept terms"))
        assert kind == "gated" and "HF_TOKEN" in text
        assert setup.classify_download_error(RuntimeError("Failed to download … 403 Client Error"))[0] == "gated"
        assert setup.classify_download_error(ValueError("dziwne"))[0] == "other"

    def test_download_tabpfn_success_and_failure(self, setup, tmp_path, clean_env, monkeypatch):
        from src.models import tabpfn_model

        target = tmp_path / "cache" / "tabpfn-v3.5-20260909.safetensors"
        monkeypatch.setattr(tabpfn_model, "default_weights_path", lambda version: target)

        def fail(path, version):
            return [self.TabPFNLicenseError("no interactive terminal")]

        con = setup.Console()
        assert not setup.download_tabpfn(con, "v3.5", no_browser=True, downloader=fail)
        assert os.environ["TABPFN_NO_BROWSER"] == "1"
        assert _status(con, "Licencja TabPFN-3.5") == ["info"] and _status(con, "TabPFN: pobieranie")[-1] == "fail"

        def ok(path, version):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"w" * 1024)
            return "ok"

        con = setup.Console()
        assert setup.download_tabpfn(con, "v3.5", downloader=ok) and _status(con, "TabPFN: wagi") == ["ok"]
        con = setup.Console()                                            # plik już jest — bez pobierania
        assert setup.download_tabpfn(con, "v3.5", downloader=lambda *a, **k: pytest.fail("pobieranie"))

    def test_download_tabicl(self, setup, tmp_path, monkeypatch):
        from src.models import tabicl_model
        from src.models.tfm_base import TFMWeightsError

        def missing(name):
            raise TFMWeightsError(name)

        monkeypatch.setattr(tabicl_model, "cached_checkpoint", missing)
        ckpt = tmp_path / "tabicl.ckpt"
        ckpt.write_bytes(b"c" * 10)
        seen: list = []
        con = setup.Console()
        assert setup.download_tabicl(con, "tabicl-classifier-v2-20260212.ckpt",
                                     downloader=lambda **kw: seen.append(kw) or str(ckpt))
        assert seen == [{"repo_id": "jingang/TabICL", "filename": "tabicl-classifier-v2-20260212.ckpt"}]

        def offline(**kw):
            raise ConnectionError("Max retries exceeded")

        con = setup.Console()
        assert not setup.download_tabicl(con, "x.ckpt", downloader=offline)
        assert "network" in con.entries[-1]["message"]


# ─────────────────────────────────────────
# Szacunki czasu
# ─────────────────────────────────────────
def _rows(model: str, points: dict[int, tuple[float, float]], n_est: int = 4, test_rows: int = 1000) -> list[dict]:
    return [{"model": model, "n_ctx": c, "n_estimators": n_est, "test_rows": test_rows, "status": "ok",
             "fit_s": fit, "predict_s": pred} for c, (fit, pred) in points.items()]


class TestEstimates:
    def test_interp_loglog(self, setup):
        pts = [(1_000, 1.0), (10_000, 100.0)]                             # y = x² / 1e6
        assert setup._interp_loglog(pts, 5_000, 1.0) == pytest.approx(25.0)
        assert setup._interp_loglog(pts, 100_000, 1.0) == pytest.approx(10_000.0)   # ekstrapolacja
        assert setup._interp_loglog([(1_000, 2.0)], 4_000, 1.0) == pytest.approx(8.0)
        assert setup._interp_loglog([], 10, 1.0) is None

    def test_curve_scales_with_rows_and_estimators(self, setup):
        curve = setup.Curve(_rows("tabpfn", {1_000: (2.0, 1.0), 10_000: (20.0, 4.0)}) +
                            [{"model": "tabpfn", "n_ctx": 50_000, "status": "oom", "n_estimators": 4}])
        assert curve.measured == [1_000, 10_000] and curve.n_estimators == 4
        # 1000 kontekstu: 2 s + 5000 wierszy × 1 ms; 8 estymatorów = × 2
        assert curve.bag_s(1_000, 5_000, 8) == pytest.approx(2 * (2.0 + 5.0))
        assert curve.bag_s(1_000, 5_000, 4, feature_fraction=0.5) == pytest.approx(3.5)
        assert not setup.Curve([{"status": "failed"}])

    def test_profile_estimates(self, setup):
        from src.utils import config

        curves = {m: setup.Curve(_rows(m, {5_000: (5.0, 0.5), 25_000: (30.0, 1.0), 50_000: (70.0, 1.5)}))
                  for m in ("tabpfn", "tabicl")}
        data = {"n_train": 446_421, "n_val": 55_803, "n_test": 55_803, "n_features": 271}
        est = setup.profile_estimates(curves, data)
        assert set(est) == set(config.TUNING_PROFILES)
        for profile, models in est.items():
            for model, e in models.items():
                budget = config.TUNING_PROFILES[profile][model]
                assert e["eta_upper_s"] == budget["timeout_s"] + budget["final_budget_s"]
                assert e["final_capped_s"] <= max(e["final_s"], 0) + 1e-6
                assert e["selection_capped_s"] <= max(e["selection_s"], budget["timeout_s"]) + 1e-6
                assert e["total_s"] == pytest.approx(e["selection_capped_s"] + e["final_capped_s"], abs=0.2)
        assert est["smoke"]["tabpfn"]["final_bags"] == 1 and est["standard"]["tabpfn"]["final_bags"] == 4
        assert est["thorough"]["tabpfn"]["worst_candidate"]["n_ctx"] == 100_000
        assert est["thorough"]["tabpfn"]["final_s"] > est["standard"]["tabpfn"]["final_s"]
        # finalna predykcja smoke: 1 worek × (dopasowanie + 8000 wierszy VAL+TEST) przy 1 estymatorze
        smoke_bag = curves["tabpfn"].bag_s(1_000, 8_000, 1)
        assert est["smoke"]["tabpfn"]["final_s"] == pytest.approx(smoke_bag, abs=0.051)       # zaokrąglenie 0,1 s

    def test_experiment_a_estimate(self, setup):
        curve = setup.Curve(_rows("tabicl", {1_000: (1.0, 0.1), 10_000: (10.0, 0.1)}, n_est=8))
        est = setup.experiment_a_estimate({"tabicl": curve}, {"site_val_rows": 1_000})
        expected = 0.0
        for n, r in zip(setup.EXP_A_SIZES, setup.EXP_A_REPEATS, strict=True):
            expected += r * (curve.bag_s(n, 20_000, 8) + curve.bag_s(n, 1_000, 8) + curve.bag_s(n + 10_000, 1_000, 8))
        assert est["tabicl"] == pytest.approx(expected, rel=1e-3) and est["total_s"] == pytest.approx(expected, rel=1e-3)
        assert est["assumptions"]["site_val_rows"] == 1_000 and "LightGBM" in est["note"]

    def test_report_sdpa(self, setup):
        con = setup.Console()
        setup.report_sdpa(con, {"seq_len": 20_000, "fused_ok": False, "errors": {"fused/gqa": "RuntimeError: x"},
                                "kernels": {"flash": {"mha": False, "gqa": False},
                                            "efficient": {"mha": True, "gqa": False},
                                            "fused": {"mha": True, "gqa": False}}})
        assert con.entries[-1]["status"] == "fail" and "MATH" in con.entries[-1]["message"]
        con = setup.Console()
        setup.report_sdpa(con, {"seq_len": 512, "fused_ok": True, "kernels": {"flash": {"mha": True, "gqa": True}}})
        assert con.entries[-1]["status"] == "ok"


# ─────────────────────────────────────────
# Prawdziwy torch / biblioteki (tylko venv TFM)
# ─────────────────────────────────────────
@pytest.mark.skipif(not HAS_TORCH, reason="brak torch")
def test_sdpa_check_runs_on_cpu(setup):
    result = setup.sdpa_check("cpu", seq_len=128)
    assert result["device"] == "cpu" and result["seq_len"] == 128
    assert set(result["kernels"]) == {"flash", "efficient", "fused"}
    assert result["fused_ok"] is True                  # na CPU: jądro flash (CPU) obsługuje MHA i GQA


@pytest.fixture(scope="module")
def tiny_checkpoints(tmp_path_factory):
    from tests.tfm_testkit import write_tiny_tabicl, write_tiny_tabpfn

    root = tmp_path_factory.mktemp("tfm_setup_ckpt")
    return {"tabpfn": write_tiny_tabpfn(root) if HAS_TABPFN else None,
            "tabicl": write_tiny_tabicl(root) if HAS_TABICL else None}


@requires_tabpfn
def test_describe_tabpfn_checkpoint(setup, tiny_checkpoints):
    desc = setup.describe_tabpfn_checkpoint(tiny_checkpoints["tabpfn"])
    assert desc["architecture"] == "tabpfn_v3_5" and desc["n_params"] > 0
    assert desc["config"]["embed_dim"] == 16 and desc["inference_config"]["MAX_CPU_SAMPLES"] == 5000
    assert desc["differs_from_plan"]["nlayers"] == {"plan": 24, "checkpoint": 1}


@requires_tabicl
def test_describe_tabicl_checkpoint(setup, tiny_checkpoints):
    desc = setup.describe_tabicl_checkpoint(tiny_checkpoints["tabicl"])
    assert desc["n_params"] > 0 and desc["config"]["embed_dim"] == 32
    assert desc["differs_from_plan"]["icl_num_blocks"] == {"plan": 12, "checkpoint": 2}


# ─────────────────────────────────────────
# main(): całe przebiegi
# ─────────────────────────────────────────
def _fake_environment(setup, monkeypatch, info=TORCH_NO_GPU):
    from src.utils import hardware

    monkeypatch.setattr(hardware, "torch_info", lambda: dict(info))
    monkeypatch.setattr(hardware, "nvidia_smi_info", lambda: None)
    monkeypatch.setattr(setup, "_dist_version",
                        lambda name: {"tabpfn": "9.2.0", "tabicl": "2.2.0"}.get(name, "1.0"))


def test_main_fake_benchmark_writes_json(setup, prepared_paths, clean_env, capsys):
    """Atrapa TFM (bez torch): pełny przebieg na CPU → benchmark.json z wynikami i szacunkami, kod 0."""
    from src.utils import config

    _fake_environment(setup, clean_env)
    clean_env.setenv("SOR_AI_TFM_FAKE", "1")
    clean_env.setenv("SOR_AI_DEVICE_TABPFN", "")
    clean_env.setenv("SOR_AI_DEVICE_TABICL", "")
    clean_env.setattr(setup, "download_tabpfn", lambda *a, **k: pytest.fail("bez pobierania"))
    rc = setup.main(["--device", "cpu", "--contexts", "100,300", "--test-rows", "60", "--skip-download"])
    out = capsys.readouterr().out
    assert rc == 0, out
    path = Path(config.RUNS_DIR) / "analyses" / "tfm_setup" / "benchmark.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["kind"] == "tfm_setup_benchmark" and data["version"] == 1 and data["fake"] is True
    assert data["device"] == "cpu" and data["contexts"] == [100, 300]
    assert [(r["model"], r["n_ctx"], r["status"]) for r in data["results"]] == [
        ("tabpfn", 100, "ok"), ("tabpfn", 300, "ok"), ("tabicl", 100, "ok"), ("tabicl", 300, "ok")]
    assert all(r["test_rows"] == 60 and r["fit_s"] >= 0 for r in data["results"])
    assert set(data["estimates"]["profiles"]) == set(config.TUNING_PROFILES)
    assert math.isfinite(data["estimates"]["experiment_a"]["total_s"])
    assert data["data"]["n_features"] > 0 and data["data"]["site_val_rows"] >= 0
    assert "Szacunki czasu" in out and "Eksperyment A" in out and "Wynik: OK" in out
    assert os.environ["SOR_AI_DEVICE_TABPFN"] == "cpu"                # --device cpu → wrappery na CPU


def test_main_cpu_limit_skips_large_contexts(setup, prepared_paths, clean_env, capsys):
    from src.utils import config

    _fake_environment(setup, clean_env)
    clean_env.setenv("SOR_AI_TFM_FAKE", "1")
    clean_env.setattr(config, "TFM_CPU_MAX_CONTEXT", 150)
    rc = setup.main(["--models", "tabpfn", "--device", "cpu", "--contexts", "100,300", "--test-rows", "40",
                     "--skip-download"])
    assert rc == 0, capsys.readouterr().out
    data = json.loads((Path(config.RUNS_DIR) / "analyses" / "tfm_setup" / "benchmark.json").read_text(encoding="utf-8"))
    assert [(r["n_ctx"], r["status"]) for r in data["results"]] == [(100, "ok"), (300, "skipped")]
    assert "TFM_CPU_MAX_CONTEXT" in data["results"][1]["detail"]


def test_main_offline_check_never_downloads_or_benchmarks(setup, prepared_paths, clean_env, capsys):
    from src.utils import config

    _fake_environment(setup, clean_env, info=TORCH_CUDA)
    clean_env.setenv("SOR_AI_TFM_FAKE", "1")
    clean_env.setattr(setup, "download_tabpfn", lambda *a, **k: pytest.fail("offline: bez pobierania"))
    clean_env.setattr(setup, "download_tabicl", lambda *a, **k: pytest.fail("offline: bez pobierania"))
    clean_env.setattr(setup, "run_benchmark", lambda *a, **k: pytest.fail("offline: bez benchmarku"))
    clean_env.setattr(setup, "sdpa_check", lambda device="cuda", seq_len=None: {
        "seq_len": 20_000, "fused_ok": True, "kernels": {"flash": {"mha": True, "gqa": True}}})
    assert setup.main(["--offline-check"]) == 0
    out = capsys.readouterr().out
    assert "tylko kontrola offline" in out and "SDPA (fp16, bez MATH)" in out
    assert not (Path(config.RUNS_DIR) / "analyses" / "tfm_setup" / "benchmark.json").exists()


def test_main_downloads_when_weights_missing_and_skips_redirected(setup, prepared_paths, clean_env, tmp_path, capsys):
    _fake_environment(setup, clean_env, info=TORCH_CUDA)
    ckpt = tmp_path / "tabicl-test.ckpt"
    ckpt.write_bytes(b"c" * 10)
    clean_env.setenv("SOR_AI_TABICL_MODEL_PATH", str(ckpt))
    calls: list[str] = []
    clean_env.setattr(setup, "download_tabpfn", lambda con, version, **k: calls.append(f"tabpfn {version}"))
    clean_env.setattr(setup, "download_tabicl", lambda *a, **k: pytest.fail("przekierowanie — bez pobierania"))
    clean_env.setattr(setup, "offline_check", lambda con, models, **k: {})
    clean_env.setattr(setup, "sdpa_check", lambda device="cuda", seq_len=None: {"fused_ok": True, "seq_len": 1,
                                                                                 "kernels": {}})
    rc = setup.main(["--skip-benchmark", "--no-browser"])
    out = capsys.readouterr().out
    assert calls == ["tabpfn v3.5"] and "aktywne przekierowanie SOR_AI_TABICL_MODEL_PATH" in out
    assert rc == 0, out


def test_main_cpu_only_torch_fails_with_install_commands(setup, prepared_paths, clean_env, capsys):
    """Symulowany brak CUDA (koło tylko-CPU): błąd z dokładnym poleceniem, benchmark GPU pominięty."""
    _fake_environment(setup, clean_env, info=TORCH_CPU_WHEEL)
    clean_env.setenv("SOR_AI_TFM_FAKE", "")
    clean_env.setattr(setup, "offline_check", lambda con, models, **k: {m: {"present": True} for m in models})
    clean_env.setattr(setup, "run_benchmark", lambda *a, **k: pytest.fail("bez CUDA i bez --device cpu"))
    assert setup.main(["--skip-download"]) == 1
    out = capsys.readouterr().out
    assert "BEZ CUDA" in out and "pip uninstall -y torch" in out and "whl/cu130" in out
    assert "Benchmark: pominięty — mierzy GPU" in out and "Wynik: 1 błędów" in out


def test_simulated_missing_torch_subprocess(tmp_path):
    """
    Proces bez działającego torch (pakiet `torch` z PYTHONPATH zgłasza ImportError — jak uszkodzona instalacja,
    a w venv bez torch: brak pakietu): komunikat z poleceniem instalacji, kod 1, bez tracebacku.
    """
    fake = tmp_path / "fake_site" / "torch"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("raise ImportError('symulowany brak torch')\n", encoding="utf-8")
    env = {**os.environ, "SOR_AI_RUNS_DIR": str(tmp_path / "runs"), "SOR_AI_LOGS_DIR": str(tmp_path / "logs"),
           "PYTHONUTF8": "1", "PYTHONPATH": str(tmp_path / "fake_site")}
    for var in ("SOR_AI_TFM_FAKE", "SOR_AI_TABPFN_MODEL_PATH", "SOR_AI_TABICL_MODEL_PATH"):
        env.pop(var, None)
    result = subprocess.run([sys.executable, str(SCRIPT), "--offline-check"], cwd=PROJECT_ROOT, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "Traceback" not in output
    assert "[BŁĄD ] torch:" in output and "pip install torch --index-url" in output


@pytest.mark.skipif(not (HAS_TABPFN and HAS_TABICL), reason="brak torch / tabpfn / tabicl")
def test_main_real_libraries_tiny_checkpoints_on_cpu(setup, prepared_paths, clean_env, tiny_checkpoints, capsys):
    """Prawdziwe TabPFN / TabICL z losowymi, maleńkimi wagami przez przekierowanie — benchmark na CPU."""
    from src.utils import config

    clean_env.setenv("SOR_AI_TABPFN_MODEL_PATH", str(tiny_checkpoints["tabpfn"]))
    clean_env.setenv("SOR_AI_TABICL_MODEL_PATH", str(tiny_checkpoints["tabicl"]))
    clean_env.setenv("SOR_AI_DEVICE_TABPFN", "")
    clean_env.setenv("SOR_AI_DEVICE_TABICL", "")
    rc = setup.main(["--device", "cpu", "--contexts", "150", "--test-rows", "40", "--n-estimators", "1",
                     "--n-jobs", "2"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "PRZEKIEROWANIE" in out and "aktywne przekierowanie" in out
    data = json.loads((Path(config.RUNS_DIR) / "analyses" / "tfm_setup" / "benchmark.json").read_text(encoding="utf-8"))
    assert [(r["model"], r["status"], r["device"]) for r in data["results"]] == [("tabpfn", "ok", "cpu"),
                                                                                ("tabicl", "ok", "cpu")]
    assert data["fake"] is False and data["weights"]["tabpfn"]["sha256"]
    assert data["weights"]["tabicl"]["override"].startswith("SOR_AI_TABICL_MODEL_PATH=")
