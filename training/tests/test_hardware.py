"""
Testy wyboru sprzętu (`src.utils.hardware`).

Bez GPU w CI: prawdziwe próby GPU muszą szybko zwrócić (False, powód) i nigdy nie
rzucić wyjątku; ścieżki z działającym GPU sprawdzane przez podmianę `probe_gpu`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.utils import hardware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_VARS = ("SOR_AI_DEVICE", "SOR_AI_DEVICE_XGBOOST", "SOR_AI_DEVICE_CATBOOST")
_REAL_PROBE = hardware.probe_gpu   # testy podmieniają probe_gpu — cache czyścimy na oryginale


@pytest.fixture
def clean_env(monkeypatch):
    """Bez zmiennych urządzenia/wątków; probe_gpu z czystym cache i plikiem w tmp."""
    for var in (*DEVICE_VARS, "SOR_AI_N_JOBS"):
        monkeypatch.delenv(var, raising=False)
    _REAL_PROBE.cache_clear()
    hardware._announced.clear()
    yield monkeypatch
    _REAL_PROBE.cache_clear()
    hardware._announced.clear()


@pytest.fixture
def probe_file(tmp_path, monkeypatch):
    path = tmp_path / "hardware_probe.json"
    monkeypatch.setattr(hardware, "_probe_cache_path", lambda: path)
    return path


def _fake_probe(monkeypatch, results: dict[str, tuple[bool, str]]) -> list[str]:
    """Podmienia probe_gpu; zwraca listę bibliotek, dla których próba została wywołana."""
    calls: list[str] = []

    def fake(library: str) -> tuple[bool, str]:
        calls.append(library)
        return results[library]

    monkeypatch.setattr(hardware, "probe_gpu", fake)
    return calls


# ─────────────────────────────────────────
# device_for — kolejność zmiennych środowiskowych
# ─────────────────────────────────────────
class TestDeviceFor:
    def test_cpu_forced_globally_skips_probe(self, clean_env):
        calls = _fake_probe(clean_env, {"xgboost": (True, "ok"), "catboost": (True, "ok")})
        clean_env.setenv("SOR_AI_DEVICE", "cpu")
        assert hardware.device_for("xgboost") == "cpu"
        assert hardware.device_for("catboost") == "cpu"
        assert calls == []

    def test_library_variable_beats_global(self, clean_env):
        calls = _fake_probe(clean_env, {"xgboost": (True, "ok"), "catboost": (True, "ok")})
        clean_env.setenv("SOR_AI_DEVICE", "cpu")
        clean_env.setenv("SOR_AI_DEVICE_XGBOOST", "gpu")
        assert hardware.device_for("xgboost") == "gpu"
        assert hardware.device_for("catboost") == "cpu"
        assert calls == ["xgboost"]

    def test_library_cpu_beats_global_gpu(self, clean_env):
        calls = _fake_probe(clean_env, {"xgboost": (True, "ok"), "catboost": (False, "brak")})
        clean_env.setenv("SOR_AI_DEVICE", "gpu")
        clean_env.setenv("SOR_AI_DEVICE_CATBOOST", "cpu")
        assert hardware.device_for("catboost") == "cpu"
        assert hardware.device_for("xgboost") == "gpu"
        assert calls == ["xgboost"]

    def test_auto_uses_probe(self, clean_env):
        _fake_probe(clean_env, {"xgboost": (True, "ok"), "catboost": (False, "CUDA error 35")})
        assert hardware.device_preference("xgboost") == ("auto", "domyślnie")
        assert hardware.device_for("xgboost") == "gpu"
        assert hardware.device_for("catboost") == "cpu"     # auto + porażka → CPU (z ostrzeżeniem)

    def test_forced_gpu_with_failing_probe_raises(self, clean_env):
        _fake_probe(clean_env, {"xgboost": (False, "booster trenował na 'cpu'"), "catboost": (False, "x")})
        clean_env.setenv("SOR_AI_DEVICE_XGBOOST", "gpu")
        with pytest.raises(RuntimeError, match="SOR_AI_DEVICE_XGBOOST=gpu") as err:
            hardware.device_for("xgboost")
        assert "580" in str(err.value)                       # wskazówka o sterowniku
        assert "booster trenował na 'cpu'" in str(err.value)  # powód z próby

    def test_cuda_alias_and_case(self, clean_env):
        _fake_probe(clean_env, {"xgboost": (True, "ok"), "catboost": (True, "ok")})
        clean_env.setenv("SOR_AI_DEVICE", " CUDA ")
        assert hardware.device_preference("catboost") == ("gpu", "SOR_AI_DEVICE")
        assert hardware.device_for("cb") == "gpu"

    def test_invalid_value_raises(self, clean_env):
        clean_env.setenv("SOR_AI_DEVICE", "tpu")
        with pytest.raises(ValueError, match="auto | cpu | gpu"):
            hardware.device_for("xgboost")

    @pytest.mark.parametrize("library", ["lightgbm", "random_forest", "ebm", "logreg"])
    def test_cpu_only_libraries(self, clean_env, library):
        calls = _fake_probe(clean_env, {})
        clean_env.setenv("SOR_AI_DEVICE", "gpu")
        assert hardware.device_for(library) == "cpu"
        assert calls == []


# ─────────────────────────────────────────
# probe_gpu — porażka bez GPU, cache
# ─────────────────────────────────────────
class TestProbeGpu:
    @pytest.mark.parametrize("library", ["xgboost", "catboost"])
    def test_real_probe_never_raises(self, clean_env, probe_file, library):
        t0 = time.time()
        ok, reason = hardware.probe_gpu(library)
        assert isinstance(ok, bool)
        assert isinstance(reason, str) and reason
        assert time.time() - t0 < 120
        if not ok:
            assert library in reason
        # Wynik zapisany do pliku (diagnostyka / preflight)
        data = json.loads(probe_file.read_text(encoding="utf-8"))
        assert data[library]["ok"] is ok

    def test_unknown_library(self, clean_env, probe_file):
        assert hardware.probe_gpu("lightgbm")[0] is False

    def test_xgboost_error_is_reported(self, clean_env, probe_file, monkeypatch):
        import xgboost as xgb

        def boom(*args, **kwargs):
            raise xgb.core.XGBoostError("cudaErrorInsufficientDriver: CUDA driver version is insufficient")

        monkeypatch.setattr(xgb, "train", boom)
        ok, reason = hardware.probe_gpu("xgboost")
        assert ok is False
        assert "XGBoostError" in reason and "cudaErrorInsufficientDriver" in reason

    def test_catboost_zero_devices(self, clean_env, probe_file, monkeypatch):
        import catboost.utils

        monkeypatch.setattr(catboost.utils, "get_gpu_device_count", lambda: 0)
        ok, reason = hardware.probe_gpu("catboost")
        assert ok is False
        assert "get_gpu_device_count() == 0" in reason

    def test_unexpected_exception_is_swallowed(self, clean_env, probe_file, monkeypatch):
        def broken() -> tuple[bool, str]:
            raise RuntimeError("coś poszło nie tak")

        monkeypatch.setitem(hardware._PROBES, "xgboost", broken)
        ok, reason = hardware.probe_gpu("xgboost")
        assert ok is False and "coś poszło nie tak" in reason

    def test_positive_file_cache_reused_negative_rechecked(self, clean_env, probe_file, monkeypatch):
        calls: list[str] = []
        results = {"xgboost": (True, "xgboost trenuje na cuda:0"), "catboost": (False, "brak GPU")}

        for lib in ("xgboost", "catboost"):
            def probe(lib=lib):
                calls.append(lib)
                return results[lib]
            monkeypatch.setitem(hardware._PROBES, lib, probe)

        assert hardware.probe_gpu("xgboost")[0] is True
        assert hardware.probe_gpu("catboost")[0] is False
        assert calls == ["xgboost", "catboost"]

        # Nowy proces (czysty lru_cache): dodatni wynik z pliku, ujemny sprawdzany ponownie
        hardware.probe_gpu.cache_clear()
        ok, reason = hardware.probe_gpu("xgboost")
        assert ok is True and "wynik z" in reason
        assert hardware.probe_gpu("catboost")[0] is False
        assert calls == ["xgboost", "catboost", "catboost"]

        # Inny sterownik / wersja → klucz się nie zgadza → ponowna próba
        hardware.probe_gpu.cache_clear()
        monkeypatch.setattr(hardware, "_probe_cache_key", lambda lib: f"{lib}|inny-sterownik")
        hardware.probe_gpu("xgboost")
        assert calls[-1] == "xgboost"

    def test_in_process_cache(self, clean_env, probe_file, monkeypatch):
        calls: list[int] = []
        monkeypatch.setitem(hardware._PROBES, "catboost", lambda: (calls.append(1), (False, "x"))[1])
        hardware.probe_gpu("catboost")
        hardware.probe_gpu("catboost")
        assert len(calls) == 1


# ─────────────────────────────────────────
# is_gpu_error
# ─────────────────────────────────────────
class TestIsGpuError:
    def test_catboost_cuda_error(self):
        from catboost import CatBoostError

        exc = CatBoostError("catboost/cuda/cuda_lib/cuda_base.h:265: CUDA error 35: "
                            "CUDA driver version is insufficient for CUDA runtime version")
        assert hardware.is_gpu_error(exc)

    def test_xgboost_oom(self):
        from xgboost.core import XGBoostError

        assert hardware.is_gpu_error(XGBoostError("cudaErrorMemoryAllocation: out of memory"))

    @pytest.mark.parametrize("message", ["CUDA error 2: out of memory", "NCCL failure", "GPU is busy"])
    def test_generic_texts(self, message):
        assert hardware.is_gpu_error(RuntimeError(message))

    @pytest.mark.parametrize("exc", [ValueError("bad input shape"), KeyError("age"), MemoryError("cuda"),
                                     KeyboardInterrupt()])
    def test_non_gpu(self, exc):
        assert not hardware.is_gpu_error(exc)


# ─────────────────────────────────────────
# Wątki CPU
# ─────────────────────────────────────────
class TestThreads:
    def test_physical_cores_positive(self):
        n = hardware.physical_cores()
        assert isinstance(n, int) and n >= 1
        assert n <= (os.cpu_count() or n)

    def test_n_jobs_from_env(self, clean_env):
        clean_env.setenv("SOR_AI_N_JOBS", "3")
        assert hardware.n_jobs() == 3

    @pytest.mark.parametrize("value", ["", "0", "-1", "abc"])
    def test_n_jobs_invalid_falls_back_to_physical(self, clean_env, value):
        clean_env.setenv("SOR_AI_N_JOBS", value)
        assert hardware.n_jobs() == hardware.physical_cores()

    def test_physical_cores_without_psutil(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)      # import psutil → ImportError
        hardware.physical_cores.cache_clear()
        try:
            assert hardware.physical_cores() >= 1
        finally:
            hardware.physical_cores.cache_clear()

    def test_physical_cores_last_resort_os_cpu_count(self, monkeypatch):
        import joblib

        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setattr(joblib, "cpu_count", lambda **kw: (_ for _ in ()).throw(OSError("brak")))
        monkeypatch.setattr(os, "cpu_count", lambda: 6)
        hardware.physical_cores.cache_clear()
        try:
            assert hardware.physical_cores() == 6
        finally:
            hardware.physical_cores.cache_clear()

    def test_config_n_jobs_honours_env_set_before_import(self):
        """SOR_AI_N_JOBS ustawione przez proces nadrzędny trafia do config i parametrów modeli."""
        env = {**os.environ, "SOR_AI_N_JOBS": "3"}
        code = (
            "from src.utils import config as c; "
            "print(c.N_JOBS, c.LGBM_DEFAULT_PARAMS['n_jobs'], c.XGB_DEFAULT_PARAMS['n_jobs'], "
            "c.CATBOOST_DEFAULT_PARAMS['thread_count'], c.RF_DEFAULT_PARAMS['n_jobs'], "
            "c.EBM_DEFAULT_PARAMS['n_jobs'])"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["3"] * 6


# ─────────────────────────────────────────
# nvidia-smi i podsumowanie sprzętu
# ─────────────────────────────────────────
class TestNvidiaSmi:
    @pytest.fixture(autouse=True)
    def _clear(self):
        hardware.nvidia_smi_info.cache_clear()
        yield
        hardware.nvidia_smi_info.cache_clear()

    def test_missing_binary(self, monkeypatch):
        monkeypatch.setattr(hardware, "_nvidia_smi_exe", lambda: None)
        assert hardware.nvidia_smi_info() is None

    def test_parses_output(self, monkeypatch):
        monkeypatch.setattr(hardware, "_nvidia_smi_exe", lambda: "nvidia-smi")

        def fake_run(cmd, **kwargs):
            assert kwargs["timeout"] == hardware.NVIDIA_SMI_TIMEOUT_S
            if len(cmd) > 1:
                return subprocess.CompletedProcess(cmd, 0, "NVIDIA GeForce RTX 4080 SUPER, 581.57, 16376\n", "")
            return subprocess.CompletedProcess(cmd, 0, "| NVIDIA-SMI 581.57   Driver Version: 581.57   "
                                                       "CUDA Version: 13.0 |\n", "")

        monkeypatch.setattr(hardware.subprocess, "run", fake_run)
        info = hardware.nvidia_smi_info()
        assert info == {
            "name": "NVIDIA GeForce RTX 4080 SUPER",
            "driver_version": "581.57",
            "cuda_version": "13.0",
            "memory_total_mb": 16376,
            "gpu_count": 1,
        }

    def test_timeout_returns_none(self, monkeypatch):
        monkeypatch.setattr(hardware, "_nvidia_smi_exe", lambda: "nvidia-smi")

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

        monkeypatch.setattr(hardware.subprocess, "run", fake_run)
        assert hardware.nvidia_smi_info() is None

    def test_windows_fallback_path(self, monkeypatch, tmp_path):
        exe = tmp_path / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
        monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
        monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
        monkeypatch.setenv("ProgramFiles", str(tmp_path))
        assert hardware._nvidia_smi_exe() == str(exe)


def test_hardware_summary_is_json_serializable(monkeypatch):
    monkeypatch.setenv("SOR_AI_N_JOBS", "2")
    summary = hardware.hardware_summary()
    for key in ("os", "python", "cpu_name", "physical_cores", "logical_cores", "ram_gb",
                "nvidia_smi", "libraries", "xgboost_cuda_build", "n_jobs", "env"):
        assert key in summary, key
    assert summary["n_jobs"] == 2
    assert summary["env"]["SOR_AI_N_JOBS"] == "2"
    assert summary["libraries"]["xgboost"]
    json.dumps(summary)
    # Kopia — modyfikacja nie psuje cache
    summary["libraries"]["xgboost"] = "zmienione"
    assert hardware.hardware_summary()["libraries"]["xgboost"] != "zmienione"
