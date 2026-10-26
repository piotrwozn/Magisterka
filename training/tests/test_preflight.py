"""
Testy kontroli środowiska (`src.pipeline.preflight`, `scripts/00_check_env.py`).

Próby GPU, opis sprzętu i wersje pakietów są podstawiane (monkeypatch) — testy nie dotykają karty
i nie zależą od maszyny.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.pipeline import preflight
from src.pipeline.preflight import (
    STATUS_FAIL,
    STATUS_INFO,
    STATUS_OK,
    STATUS_WARN,
    PreflightReport,
    check_data,
    check_encoding,
    check_gpu,
    check_packages,
    check_paths,
    check_power,
    check_python,
    check_resources,
    check_tfm,
    run_preflight,
    tfm_weights_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SMI = {"name": "NVIDIA GeForce RTX 4080 SUPER", "driver_version": "581.57", "cuda_version": "13.0",
       "memory_total_mb": 16376, "gpu_count": 1}


def _statuses(report: PreflightReport, name: str) -> list[str]:
    return [c.status for c in report.checks if c.name == name]


def _no_preference(lib: str) -> tuple[str, str]:
    return "auto", "domyślnie"


# ─────────────────────────────────────────
# Python i pakiety
# ─────────────────────────────────────────
class TestPython:
    @pytest.mark.parametrize("version, is_64, expected", [
        ((3, 11, 9), True, STATUS_FAIL),
        ((3, 12, 7), True, STATUS_OK),
        ((3, 13, 16), True, STATUS_OK),
        ((3, 13, 16), False, STATUS_FAIL),
        ((3, 14, 0), True, STATUS_WARN),
    ])
    def test_versions(self, version, is_64, expected):
        report = PreflightReport()
        check_python(report, version_info=version, is_64bit=is_64)
        assert _statuses(report, "Python") == [expected]


class TestPackages:
    @pytest.fixture
    def req_files(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("numpy>=2.0,<3   # komentarz\noptuna>=4.0,<6\n-r inne.txt\n\nxgboost>=3.1\n", encoding="utf-8")
        lock = tmp_path / "requirements-lock.txt"
        lock.write_text('numpy==2.5.4\noptuna==5.0.0\nnvidia-nccl-cu13==2.32.3; platform_system == "Nieistniejacy"\n',
                        encoding="utf-8")
        return req, lock

    @staticmethod
    def _versions(**overrides):
        base = {name: "1.0" for name in preflight.REQUIRED_PACKAGES}
        base.update({"numpy": "2.5.4", "optuna": "5.0.0", "xgboost": "3.4.1", "interpret-core": "0.7.8"})
        base.update(overrides)
        return lambda dist: base.get(dist.lower() if dist != "SQLAlchemy" else "sqlalchemy")

    def test_all_good(self, req_files):
        report = PreflightReport()
        check_packages(report, *req_files, version_of=self._versions())
        assert _statuses(report, "Pakiety") == [STATUS_OK]
        assert _statuses(report, "Wersje vs lock") == [STATUS_OK]      # marker spoza platformy pominięty

    def test_missing_package_is_fatal(self, req_files):
        report = PreflightReport()
        check_packages(report, *req_files, version_of=self._versions(psutil=None))
        assert _statuses(report, "Pakiety") == [STATUS_FAIL]
        assert "psutil" in report.hard_failures[0].message

    def test_out_of_range_is_fatal(self, req_files):
        report = PreflightReport()
        check_packages(report, *req_files, version_of=self._versions(optuna="6.1.0"))
        assert _statuses(report, "Pakiety") == [STATUS_FAIL]
        assert "optuna 6.1.0" in report.hard_failures[0].message

    def test_lock_mismatch_warns(self, req_files):
        report = PreflightReport()
        check_packages(report, *req_files, version_of=self._versions(numpy="2.4.0"))
        assert _statuses(report, "Pakiety") == [STATUS_OK]
        assert _statuses(report, "Wersje vs lock") == [STATUS_WARN]
        assert report.facts["lock_differences"] == ["numpy 2.4.0 (lock 2.5.4)"]
        assert report.ok

    def test_interpret_full_package_accepted(self, req_files):
        report = PreflightReport()
        versions = self._versions(**{"interpret-core": None, "interpret": "0.7.8"})
        check_packages(report, *req_files, version_of=versions)
        assert _statuses(report, "Pakiety") == [STATUS_OK]

    def test_repo_lock_file_parses_and_pins_core(self):
        """requirements-lock.txt w repo: same przypięte wersje, wszystkie wymagane pakiety obecne."""
        reqs = preflight._parse_requirement_lines(preflight.REQUIREMENTS_LOCK)
        names = {preflight._normalize(r.name) for r in reqs}
        for dists in preflight.REQUIRED_PACKAGES.values():
            assert any(preflight._normalize(d) in names for d in dists), dists
        assert all(any(s.operator == "==" for s in r.specifier) for r in reqs)

    def test_repo_requirements_ranges_parse(self):
        reqs = {preflight._normalize(r.name): r for r in preflight._parse_requirement_lines(preflight.REQUIREMENTS_TXT)}
        assert "interpret-core" in reqs and "interpret" not in reqs
        assert reqs["optuna"].specifier.contains("5.0.0") and not reqs["optuna"].specifier.contains("6.0.0")


# ─────────────────────────────────────────
# Zasoby
# ─────────────────────────────────────────
class TestResources:
    @pytest.mark.parametrize("ram, expected", [(8.0, STATUS_FAIL), (15.7, STATUS_WARN), (24.0, STATUS_WARN),
                                               (29.9, STATUS_WARN), (31.8, STATUS_OK),   # 32 GB = ~31,8 GB
                                               (63.8, STATUS_OK), (None, STATUS_WARN)])
    def test_ram(self, ram, expected):
        report = PreflightReport()
        check_resources(report, {"ram_gb": ram, "physical_cores": 24, "logical_cores": 32, "n_jobs": 24}, 500.0)
        assert _statuses(report, "RAM") == [expected]

    @pytest.mark.parametrize("disk, expected", [(5.0, STATUS_FAIL), (15.0, STATUS_WARN), (200.0, STATUS_OK)])
    def test_disk(self, disk, expected):
        report = PreflightReport()
        check_resources(report, {"ram_gb": 64.0, "physical_cores": 24, "logical_cores": 32}, disk)
        assert _statuses(report, "Dysk") == [expected]
        assert report.facts["physical_cores"] == 24


# ─────────────────────────────────────────
# GPU
# ─────────────────────────────────────────
class TestGpu:
    def test_both_gpu_ok(self):
        report = PreflightReport()
        devices = check_gpu(report, SMI, lambda lib: (True, f"{lib} na GPU"), _no_preference, "13.3")
        assert devices == {"xgboost": "gpu", "catboost": "gpu"}
        assert report.facts["gpu_ok"] is True
        assert _statuses(report, "GPU / sterownik") == [STATUS_OK]
        assert report.ok

    def test_probe_failure_falls_back_to_cpu_with_warning(self):
        report = PreflightReport()
        probe = {"xgboost": (False, "trening przebiegł na 'cpu'"), "catboost": (True, "ok")}
        devices = check_gpu(report, SMI, lambda lib: probe[lib], _no_preference)
        assert devices == {"xgboost": "cpu", "catboost": "gpu"}
        check = next(c for c in report.checks if c.name == "GPU: xgboost")
        assert check.status == STATUS_WARN and "580" in check.advice
        assert report.ok

    def test_forced_gpu_failure_is_fatal(self):
        report = PreflightReport()
        check_gpu(report, None, lambda lib: (False, "CUDA error 35"),
                  lambda lib: ("gpu", "SOR_AI_DEVICE_XGBOOST") if lib == "xgboost" else ("auto", "domyślnie"))
        assert [c.name for c in report.hard_failures] == ["GPU: xgboost"]

    def test_forced_cpu_skips_probe(self):
        report = PreflightReport()

        def probe(lib):
            raise AssertionError("próba GPU nie powinna być uruchomiona")

        devices = check_gpu(report, SMI, probe, lambda lib: ("cpu", "SOR_AI_DEVICE"))
        assert devices == {"xgboost": "cpu", "catboost": "cpu"}
        assert report.facts["gpu_ok"] is False

    def test_no_probe_means_auto(self):
        report = PreflightReport()
        assert check_gpu(report, SMI, None, _no_preference) == {"xgboost": "auto", "catboost": "auto"}

    def test_old_driver_warns(self):
        report = PreflightReport()
        check_gpu(report, {**SMI, "driver_version": "560.94"}, lambda lib: (True, "ok"), _no_preference)
        assert _statuses(report, "GPU / sterownik") == [STATUS_WARN]

    def test_no_nvidia_smi_is_info(self):
        report = PreflightReport()
        check_gpu(report, None, lambda lib: (False, "brak GPU"), _no_preference)
        assert _statuses(report, "GPU / sterownik") == [STATUS_INFO]
        assert report.ok

    def test_invalid_env_value_is_fatal(self):
        report = PreflightReport()

        def preference(lib):
            raise ValueError("SOR_AI_DEVICE='tak' — dozwolone wartości: auto | cpu | gpu")

        check_gpu(report, None, lambda lib: (True, "ok"), preference)
        assert len(report.hard_failures) == 2


# ─────────────────────────────────────────
# Dane, ścieżki, kodowanie
# ─────────────────────────────────────────
class TestData:
    @pytest.fixture
    def raw_paths(self, tmp_path, monkeypatch):
        from src.utils import config

        parquet, rdata = tmp_path / "raw" / "yale.parquet", tmp_path / "raw" / "yale.rdata"
        monkeypatch.setattr(config, "RAW_PARQUET", parquet)
        monkeypatch.setattr(config, "RDATA_FILE", rdata)
        return parquet, rdata

    def test_nothing_available_is_fatal(self, isolated_paths, raw_paths):
        report = PreflightReport()
        check_data(report)
        assert _statuses(report, "Dane") == [STATUS_FAIL]

    def test_raw_parquet_ok(self, isolated_paths, raw_paths):
        parquet, _ = raw_paths
        parquet.parent.mkdir(parents=True)
        parquet.write_bytes(b"0" * (preflight.RAW_MIN_BYTES + 1))
        report = PreflightReport()
        check_data(report)
        assert _statuses(report, "Dane") == [STATUS_OK]
        assert report.facts["data"]["prepared"] is False

    def test_tiny_raw_file_is_fatal(self, isolated_paths, raw_paths):
        parquet, _ = raw_paths
        parquet.parent.mkdir(parents=True)
        parquet.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
        report = PreflightReport()
        check_data(report)
        assert _statuses(report, "Dane") == [STATUS_FAIL]

    def test_prepared_without_raw_ok(self, prepared_paths, raw_paths):
        report = PreflightReport()
        check_data(report)
        assert _statuses(report, "Dane") == [STATUS_OK]
        assert report.facts["data"]["prepared"] is True
        assert report.facts["data"]["split_strategy"] == "random"


class TestPaths:
    @pytest.mark.parametrize("path, system, expected", [
        ("/home/user/Magisterka/training", "Linux", STATUS_OK),
        (r"C:\dev\Magisterka\training", "Windows", STATUS_OK),
        (r"C:\Users\Piotr\OneDrive\Dokumenty\Magisterka\training", "Windows", STATUS_WARN),
        (r"C:\Users\Paweł\Magisterka\training", "Windows", STATUS_WARN),
        ("C:\\" + "a" * 120 + "\\training", "Windows", STATUS_WARN),
        ("/home/" + "a" * 120 + "/training", "Linux", STATUS_OK),
    ])
    def test_paths(self, path, system, expected):
        report = PreflightReport()
        check_paths(report, Path(path), system, prefix="/venv")
        assert _statuses(report, "Ścieżki") == [expected]


class TestPower:
    BALANCED = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)"

    @pytest.mark.parametrize("output, overlay, expected", [
        ("Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance)", None, STATUS_OK),
        ("GUID schematu zasilania: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (Wysoka wydajność)", None, STATUS_OK),
        # Ultimate Performance włączany przez -duplicatescheme dostaje NOWY GUID — rozpoznanie po nazwie
        ("Power Scheme GUID: 0f3a2b1c-1111-2222-3333-444455556666  (Ultimate Performance)", None, STATUS_OK),
        (BALANCED, None, STATUS_WARN),
        (BALANCED, "ded574b5-45a0-4f42-8737-46345c09c238", STATUS_OK),        # tryb „Najlepsza wydajność”
        (BALANCED, "961cc777-2547-4f9d-8174-7d86181b8a7a", STATUS_WARN),      # „Najlepsza efektywność”
        (None, None, STATUS_INFO),                                             # powercfg niedostępne
    ])
    def test_power_scheme(self, output, overlay, expected):
        report = PreflightReport()
        check_power(report, "Windows", output, overlay)
        assert _statuses(report, "Zasilanie") == [expected]
        assert report.ok                                     # nigdy błąd krytyczny

    def test_warning_mentions_best_performance_and_console(self):
        report = PreflightReport()
        check_power(report, "Windows", self.BALANCED)
        check = report.warnings[0]
        assert "Najlepsza wydajność" in check.advice and "konsoli" in check.advice

    def test_other_systems_skip(self):
        report = PreflightReport()
        check_power(report, "Linux", None)
        assert report.checks == []


def test_failing_check_is_reported_not_raised(prepared_paths, fake_hardware, monkeypatch):
    """Wywrócona kontrola (np. brak pandas przy sprawdzaniu danych) → wpis w raporcie, nie traceback."""
    def broken(report):
        raise ModuleNotFoundError("No module named 'pandas'")

    monkeypatch.setattr(preflight, "check_data", broken)
    report = run_preflight(probe_gpus=False)
    check = next(c for c in report.checks if c.name == "Dane")
    assert check.status == STATUS_FAIL and "pandas" in check.message
    assert "requirements-lock.txt" in check.advice


def test_encoding_check_records_facts():
    report = PreflightReport()
    check_encoding(report, system="Windows")
    assert set(report.facts["encoding"]) == {"utf8_mode", "preferred", "stdout", "PYTHONUTF8"}
    assert len(report.checks) == 1


# ─────────────────────────────────────────
# Modele fundamentalne (TFM)
# ─────────────────────────────────────────
TORCH_CUDA = {"installed": True, "version": "2.14.1+cu130", "cuda_build": "13.0", "cuda_available": True,
              "device_name": "NVIDIA GeForce RTX 4080 SUPER", "total_vram_mb": 16376, "free_vram_mb": 15000,
              "error": None}
TORCH_CPU_WHEEL = {**TORCH_CUDA, "version": "2.14.1+cpu", "cuda_build": None, "cuda_available": False,
                   "device_name": None, "total_vram_mb": None, "free_vram_mb": None}
TORCH_MISSING = {**TORCH_CPU_WHEEL, "installed": False, "version": None, "error": "torch nie jest zainstalowany"}


class TestTfm:
    @staticmethod
    def run(models=("tabpfn", "tabicl"), profile="standard", *, info=TORCH_CUDA, probe=lambda lib: (True, "torch ok"),
            preference=_no_preference, versions=None, weights=None, fake=False, context=None, disk=None,
            tune=True) -> tuple[PreflightReport, dict]:
        report = PreflightReport()
        report.facts["devices"] = {"xgboost": "auto", "catboost": "auto"}
        versions = {"tabpfn": "9.2.0", "tabicl": "2.2.0"} if versions is None else versions
        present = {"present": True, "path": "/cache/w.safetensors", "cache_dir": "/cache", "size_mb": 835.0,
                   "label": "wagi", "override": None}
        devices = check_tfm(
            report, list(models), profile, tune=tune, torch_info=lambda: dict(info), probe=probe,
            preference=preference, version_of=lambda name: versions.get(name),
            weights=weights or (lambda model: dict(present)), disk_free=disk or (lambda path: 100.0),
            fake=fake, planned_context=context or (lambda model, prof, tune: 1_000 if prof == "smoke" else 50_000))
        return report, devices

    def test_classical_models_skip_check(self):
        report = PreflightReport()
        assert check_tfm(report, ["lightgbm", "xgboost"], "standard", torch_info=lambda: pytest.fail("bez torch")) \
            == {}
        assert report.checks == [] and "tfm" not in report.facts

    def test_all_good_on_gpu(self):
        report, devices = self.run()
        assert devices == {"tabpfn": "gpu", "tabicl": "gpu"}
        assert report.facts["devices"] == {"xgboost": "auto", "catboost": "auto", "tabpfn": "gpu", "tabicl": "gpu"}
        assert report.ok and not report.warnings
        assert _statuses(report, "TFM: VRAM") == [STATUS_OK]
        assert _statuses(report, "TFM: wagi tabpfn") == [STATUS_OK]
        assert report.facts["tfm"]["planned_context"] == {"tabpfn": 50_000, "tabicl": 50_000}

    def test_low_free_vram_warns(self):
        report, _ = self.run(info={**TORCH_CUDA, "free_vram_mb": 6 * 1024})
        check = next(c for c in report.checks if c.name == "TFM: VRAM")
        assert check.status == STATUS_WARN and "6.0 GB" in check.message and "ollama" in check.advice.lower()
        assert report.ok

    @pytest.mark.parametrize("info", [TORCH_CPU_WHEEL, {**TORCH_CUDA, "cuda_available": False}])
    def test_no_cuda_refused_unless_smoke(self, info):
        report, devices = self.run(info=info, profile="standard")
        assert devices == {"tabpfn": "cpu", "tabicl": "cpu"}
        assert {c.name for c in report.hard_failures} == {"TFM: tabpfn", "TFM: tabicl"}
        failure = report.hard_failures[0]
        assert "50,000" in failure.message and "TFM_CPU_MAX_CONTEXT" in failure.message
        assert "cu130" in failure.advice and "cu128" in failure.advice and "--profile smoke" in failure.advice
        report, devices = self.run(info=info, profile="smoke")
        assert devices == {"tabpfn": "cpu", "tabicl": "cpu"} and report.ok
        assert _statuses(report, "TFM: tabpfn") == [STATUS_WARN]

    def test_cpu_only_wheel_message(self):
        report, _ = self.run(info=TORCH_CPU_WHEEL, profile="smoke")
        torch_check = next(c for c in report.checks if c.name == "TFM: torch")
        assert "BEZ CUDA" in torch_check.message and "download.pytorch.org/whl/cu130" in torch_check.advice

    def test_smoke_context_above_cpu_limit_refused(self):
        report, _ = self.run(info=TORCH_CPU_WHEEL, profile="smoke", context=lambda m, p, t: 6_000)
        assert len(report.hard_failures) == 2

    def test_failed_torch_probe_counts_as_no_cuda(self):
        report, devices = self.run(probe=lambda lib: (False, "CUDA error: out of memory"))
        assert devices == {"tabpfn": "cpu", "tabicl": "cpu"} and not report.ok
        assert "out of memory" in report.hard_failures[0].message

    def test_no_probe_uses_cuda_available(self):
        report, devices = self.run(probe=None)
        assert devices == {"tabpfn": "gpu", "tabicl": "gpu"} and report.ok

    def test_torch_missing_is_one_fatal_entry(self):
        report, devices = self.run(info=TORCH_MISSING, profile="smoke",
                                   versions={"tabpfn": "9.2.0"}, weights=lambda m: pytest.fail("bez torch"))
        assert [c.name for c in report.hard_failures] == ["TFM: torch"]
        assert "requirements-tfm.txt" in report.hard_failures[0].advice
        assert devices == {"tabpfn": "cpu", "tabicl": "cpu"}

    def test_forced_cpu_and_forced_gpu(self):
        forced_cpu = lambda lib: ("cpu", f"SOR_AI_DEVICE_{lib.upper()}")  # noqa: E731
        report, devices = self.run(preference=forced_cpu, profile="standard")
        assert devices == {"tabpfn": "cpu", "tabicl": "cpu"} and len(report.hard_failures) == 2
        assert "CPU wymuszone" in report.hard_failures[0].message
        report, _ = self.run(preference=forced_cpu, profile="smoke")
        assert report.ok
        forced_gpu = lambda lib: ("gpu", "SOR_AI_DEVICE")  # noqa: E731
        report, _ = self.run(info=TORCH_CPU_WHEEL, preference=forced_gpu, profile="smoke")
        assert len(report.hard_failures) == 2 and "SOR_AI_DEVICE=gpu" in report.hard_failures[0].message

    def test_invalid_device_variable_is_fatal(self):
        def preference(lib):
            raise ValueError("SOR_AI_DEVICE_TABPFN='tak' — dozwolone wartości: auto | cpu | gpu")

        report, devices = self.run(models=("tabpfn",), preference=preference)
        assert devices == {"tabpfn": "cpu"} and [c.name for c in report.hard_failures] == ["TFM: tabpfn"]

    @pytest.mark.parametrize("versions, fragment", [({"tabpfn": "9.1.0", "tabicl": "2.2.0"}, "9.1.0 < 9.2.0"),
                                                    ({"tabicl": "2.2.0"}, "nie jest zainstalowany")])
    def test_library_versions(self, versions, fragment):
        report, _ = self.run(versions=versions, weights=lambda m: {"present": True, "label": m, "override": None})
        assert [c.name for c in report.hard_failures] == ["TFM: pakiet tabpfn"]
        assert fragment in report.hard_failures[0].message

    def test_missing_weights_point_to_setup_script_and_disk(self):
        missing = {"present": False, "path": "/c/tabpfn-v3.5.safetensors", "cache_dir": "/c", "label": "TabPFN v3.5",
                   "override": None}
        report, _ = self.run(models=("tabpfn",), weights=lambda m: dict(missing), disk=lambda path: 1.0)
        failure = next(c for c in report.checks if c.name == "TFM: wagi tabpfn")
        assert failure.status == STATUS_FAIL and "sprawdzone bez sieci" in failure.message
        assert "13_tfm_setup.py" in failure.advice and "TABPFN_TOKEN" in failure.advice
        assert _statuses(report, "TFM: dysk") == [STATUS_WARN]

    def test_weights_override_warns_or_fails(self):
        override = {"present": True, "path": "/t/x.ckpt", "label": "x", "override": "SOR_AI_TABICL_MODEL_PATH=/t/x.ckpt"}
        report, _ = self.run(models=("tabicl",), weights=lambda m: dict(override))
        assert _statuses(report, "TFM: wagi tabicl") == [STATUS_WARN] and report.ok
        report, _ = self.run(models=("tabicl",), weights=lambda m: {**override, "present": False})
        assert _statuses(report, "TFM: wagi tabicl") == [STATUS_FAIL]

    def test_weights_import_error_is_fatal_with_install_advice(self):
        def broken(model):
            raise ImportError("No module named 'tabpfn'")

        report, _ = self.run(models=("tabpfn",), weights=broken)
        failure = next(c for c in report.checks if c.name == "TFM: wagi tabpfn")
        assert failure.status == STATUS_FAIL and "requirements-tfm.txt" in failure.advice

    def test_fake_mode_skips_libraries_and_weights(self):
        report, devices = self.run(info=TORCH_MISSING, profile="smoke", fake=True, versions={},
                                   weights=lambda m: pytest.fail("atrapa — bez wag"))
        assert devices == {"tabpfn": "cpu", "tabicl": "cpu"} and report.ok
        assert _statuses(report, "TFM: hak testowy") == [STATUS_WARN]

    def test_tfm_weights_status_override_and_cache(self, monkeypatch, tmp_path):
        from src.models import tabicl_model, tabpfn_model
        from src.models.tfm_base import TFMWeightsError

        for var in ("SOR_AI_TFM_FAKE", "SOR_AI_TABPFN_MODEL_PATH", "SOR_AI_TABICL_MODEL_PATH"):
            monkeypatch.delenv(var, raising=False)
        weights = tmp_path / "tabpfn-v3.5-test.safetensors"
        weights.write_bytes(b"x" * 2048)
        monkeypatch.setenv("SOR_AI_TABPFN_MODEL_PATH", str(weights))
        status = tfm_weights_status("tabpfn")
        assert status["present"] and status["override"].startswith("SOR_AI_TABPFN_MODEL_PATH=")
        monkeypatch.setenv("SOR_AI_TABPFN_MODEL_PATH", str(tmp_path / "brak.safetensors"))
        assert not tfm_weights_status("tabpfn")["present"]
        monkeypatch.delenv("SOR_AI_TABPFN_MODEL_PATH")
        seen: list[str] = []
        monkeypatch.setattr(tabpfn_model, "default_weights_path",
                            lambda version: seen.append(version) or tmp_path / "cache" / f"tabpfn-{version}.safetensors")
        status = tfm_weights_status("tabpfn", {"model_version": "v3.5-fast"})
        assert seen == ["v3.5-fast"] and not status["present"] and status["cache_dir"] == str(tmp_path / "cache")
        (tmp_path / "cache").mkdir()
        (tmp_path / "cache" / "tabpfn-v3.5.safetensors").write_bytes(b"x" * 4096)
        assert tfm_weights_status("tabpfn")["present"]

        def no_checkpoint(name):
            raise TFMWeightsError(f"brak {name}")

        monkeypatch.setattr(tabicl_model, "cached_checkpoint", no_checkpoint)
        assert not tfm_weights_status("tabicl")["present"]
        ckpt = tmp_path / "tabicl.ckpt"
        ckpt.write_bytes(b"x" * 100)
        monkeypatch.setattr(tabicl_model, "cached_checkpoint", lambda name: ckpt)
        assert tfm_weights_status("tabicl")["path"] == str(ckpt)
        monkeypatch.setenv("SOR_AI_TFM_FAKE", "1")
        assert tfm_weights_status("tabicl")["present"]
        with pytest.raises(ValueError):
            tfm_weights_status("lightgbm")


def test_run_preflight_with_tfm_models(prepared_paths, fake_hardware, monkeypatch, tmp_path):
    from src.utils import hardware

    monkeypatch.setattr(hardware, "torch_info", lambda: dict(TORCH_CUDA))
    monkeypatch.setattr(preflight, "tfm_weights_status",
                        lambda model, params=None: {"present": True, "path": f"/c/{model}", "label": model,
                                                    "override": None})
    monkeypatch.setattr(preflight, "_dist_version",
                        lambda name: {"tabpfn": "9.2.0", "tabicl": "2.2.0"}.get(name) or "1.0")
    monkeypatch.delenv("SOR_AI_TFM_FAKE", raising=False)
    report = run_preflight(probe_gpus=True, out_dir=tmp_path / "pf", models=["lightgbm", "tabpfn", "tabicl"],
                           profile="thorough")
    data = json.loads((tmp_path / "pf" / "preflight.json").read_text(encoding="utf-8"))
    assert data["facts"]["devices"] == {"xgboost": "gpu", "catboost": "gpu", "tabpfn": "gpu", "tabicl": "gpu"}
    assert data["facts"]["tfm"]["profile"] == "thorough" and data["facts"]["models"][-1] == "tabicl"
    assert {"TFM: torch", "TFM: tabpfn", "TFM: VRAM", "TFM: wagi tabicl"} <= {c["name"] for c in data["checks"]}
    assert data["facts"]["tfm"]["planned_context"]["tabpfn"] == 100_000          # kandydaci profilu thorough
    assert report.facts["devices"]["tabpfn"] == "gpu"


# ─────────────────────────────────────────
# Całość + skrypt
# ─────────────────────────────────────────
@pytest.fixture
def fake_hardware(monkeypatch):
    from src.utils import hardware

    summary = {"ram_gb": 64.0, "physical_cores": 24, "logical_cores": 32, "n_jobs": 24,
               "cpu_name": "Intel(R) Core(TM) i9-14900K", "nvidia_smi": SMI, "xgboost_cuda_build": "13.3"}
    monkeypatch.setattr(hardware, "hardware_summary", lambda: dict(summary))
    monkeypatch.setattr(hardware, "probe_gpu", lambda lib: (True, f"{lib} trenuje na GPU"))
    monkeypatch.setattr(hardware, "device_preference", lambda lib: ("auto", "domyślnie"))
    return summary


def test_run_preflight_writes_json(prepared_paths, fake_hardware, tmp_path):
    report = run_preflight(probe_gpus=True, out_dir=tmp_path / "pf")
    data = json.loads((tmp_path / "pf" / "preflight.json").read_text(encoding="utf-8"))
    assert data["facts"]["devices"] == {"xgboost": "gpu", "catboost": "gpu"}
    assert data["facts"]["gpu_ok"] is True
    assert data["facts"]["physical_cores"] == 24
    assert {c["name"] for c in data["checks"]} >= {"Python", "Pakiety", "RAM", "Dysk", "GPU: xgboost", "Dane",
                                                   "Ścieżki", "Kodowanie"}
    assert data["ok"] == report.ok
    assert "SOR-AI" in report.format()


def test_run_preflight_without_gpu_probe(prepared_paths, fake_hardware, monkeypatch, tmp_path):
    from src.utils import hardware

    monkeypatch.setattr(hardware, "probe_gpu", lambda lib: pytest.fail("próba GPU wyłączona"))
    report = run_preflight(probe_gpus=False)
    assert report.facts["devices"] == {"xgboost": "auto", "catboost": "auto"}
    assert "tfm" not in report.facts                     # bez TFM na liście — kontrola TFM nie rusza


def test_check_env_tfm_fake_subprocess_without_traceback(tmp_path):
    """Prawdziwy 00_check_env.py z --models tfm (atrapa TFM): wpisy TFM, bez tracebacku — w obu venv."""
    env = {**os.environ, "SOR_AI_ESI_DATA_DIR": str(tmp_path / "esi"), "SOR_AI_RUNS_DIR": str(tmp_path / "runs"),
           "SOR_AI_LOGS_DIR": str(tmp_path / "logs"), "PYTHONUTF8": "1", "SOR_AI_TFM_FAKE": "1"}
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "00_check_env.py"), "--no-gpu-probe",
                             "--models", "tfm", "--profile", "smoke", "--out-dir", str(tmp_path / "pf")],
                            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=300)
    output = result.stdout + result.stderr
    assert "Traceback" not in output
    assert "TFM: hak testowy" in output and "TFM: tabpfn" in output
    facts = json.loads((tmp_path / "pf" / "preflight.json").read_text(encoding="utf-8"))["facts"]
    assert facts["devices"]["tabpfn"] in ("cpu", "gpu") and facts["tfm"]["fake"] is True


def _load_check_env():
    spec = importlib.util.spec_from_file_location("check_env_script", PROJECT_ROOT / "scripts" / "00_check_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fatal, code", [(False, 0), (True, 1)])
def test_check_env_exit_code(monkeypatch, tmp_path, capsys, fatal, code):
    module = _load_check_env()

    def fake_run_preflight(probe_gpus=True, out_dir=None, **kwargs):
        report = PreflightReport()
        report.add("RAM", STATUS_FAIL if fatal else STATUS_OK, "test")
        report.facts.update({"gpu_ok": True, "physical_cores": 24, "ram_gb": 31.8,
                             "devices": {"xgboost": "gpu", "catboost": "cpu"}})
        report.write_json(Path(out_dir) / "preflight.json")
        return report

    monkeypatch.setattr(module, "run_preflight", fake_run_preflight)
    assert module.main(["--out-dir", str(tmp_path), "--no-gpu-probe"]) == code
    out = capsys.readouterr().out
    assert "2 tor" in out and "[xgboost]" in out
    assert (tmp_path / "preflight.json").exists()


def test_check_env_passes_models_profile_and_tune(monkeypatch, tmp_path, capsys):
    """--models / --profile / --no-tune trafiają do run_preflight; prognoza torów z tej listy modeli."""
    module = _load_check_env()
    seen: dict = {}

    def fake_run_preflight(probe_gpus=True, out_dir=None, **kwargs):
        seen.update(kwargs, probe_gpus=probe_gpus)
        report = PreflightReport()
        report.add("RAM", STATUS_OK, "test")
        report.facts.update({"gpu_ok": False, "physical_cores": 24, "ram_gb": 64.0,
                             "devices": {"xgboost": "cpu", "catboost": "cpu", "tabpfn": "gpu", "tabicl": "gpu"},
                             "tfm": {"models": ["tabpfn", "tabicl"]}})
        return report

    monkeypatch.setattr(module, "run_preflight", fake_run_preflight)
    assert module.main(["--out-dir", str(tmp_path), "--models", "lgbm,tfm", "--profile", "smoke", "--no-tune"]) == 0
    assert seen == {"models": ["lightgbm", "tabpfn", "tabicl"], "profile": "smoke", "tune": False, "probe_gpus": True}
    out = capsys.readouterr().out
    assert "2 tor" in out and "[tabpfn, tabicl]" in out
    assert "--models lgbm,tfm" in out and "13_tfm_setup.py" in out
    with pytest.raises(SystemExit):
        module.main(["--out-dir", str(tmp_path), "--models", "svm"])


def test_check_env_default_models_are_classical(monkeypatch, tmp_path):
    module = _load_check_env()
    seen: dict = {}

    def fake_run_preflight(probe_gpus=True, out_dir=None, **kwargs):
        seen.update(kwargs)
        report = PreflightReport()
        report.facts.update({"devices": {}, "physical_cores": 4, "ram_gb": 16.0})
        return report

    monkeypatch.setattr(module, "run_preflight", fake_run_preflight)
    assert module.main(["--out-dir", str(tmp_path), "--no-gpu-probe"]) == 0
    assert seen["models"] == ["logreg", "lightgbm", "xgboost", "catboost", "random_forest", "ebm"]


def _check_env_subprocess(tmp_path, blocked: list[str]) -> subprocess.CompletedProcess:
    """00_check_env.py w osobnym procesie z „brakującymi” modułami (None w sys.modules → ImportError)."""
    code = ("import runpy, sys\n"
            f"for name in {blocked!r}:\n"
            "    sys.modules[name] = None\n"
            f"sys.argv = ['00_check_env.py', '--no-gpu-probe', '--out-dir', {str(tmp_path / 'pf')!r}]\n"
            f"runpy.run_path({str(PROJECT_ROOT / 'scripts' / '00_check_env.py')!r}, run_name='__main__')\n")
    env = {**os.environ, "SOR_AI_ESI_DATA_DIR": str(tmp_path / "esi"), "SOR_AI_RUNS_DIR": str(tmp_path / "runs"),
           "SOR_AI_LOGS_DIR": str(tmp_path / "logs"), "PYTHONUTF8": "1"}
    return subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)


def test_check_env_without_pandas_reports_instead_of_traceback(tmp_path):
    result = _check_env_subprocess(tmp_path, ["pandas", "numpy"])
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "Traceback" not in output
    assert "brak pakietu" in output and "pip install -r requirements-lock.txt" in output


def test_check_env_import_failure_lists_missing_packages(tmp_path):
    """Kod kontroli nie importuje się wcale — lista braków z requirements.txt i polecenie instalacji."""
    result = _check_env_subprocess(tmp_path, ["src.pipeline.preflight"])
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "Traceback" not in output
    assert "Nie można uruchomić kontroli" in output and "pip install -r requirements-lock.txt" in output


def test_missing_requirements_parser(tmp_path):
    module = _load_check_env()
    req = tmp_path / "requirements.txt"
    req.write_text("numpy>=2.0,<3  # komentarz\n-r inne.txt\npakiet-ktorego-nie-ma>=1.0\n"
                   "nvidia-nccl-cu13==2.32; platform_system == 'Linux'\n", encoding="utf-8")
    missing = module.missing_requirements(req)
    assert "pakiet-ktorego-nie-ma" in missing and "numpy" not in missing
