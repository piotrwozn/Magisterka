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
    run_preflight,
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


def _load_check_env():
    spec = importlib.util.spec_from_file_location("check_env_script", PROJECT_ROOT / "scripts" / "00_check_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fatal, code", [(False, 0), (True, 1)])
def test_check_env_exit_code(monkeypatch, tmp_path, capsys, fatal, code):
    module = _load_check_env()

    def fake_run_preflight(probe_gpus=True, out_dir=None):
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
