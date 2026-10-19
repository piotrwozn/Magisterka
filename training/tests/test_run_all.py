"""
Testy orkiestratora nocnego przebiegu (`src.pipeline.orchestrator`, `scripts/run_all.py`)
i blokady uśpienia (`src.pipeline.keepawake`).

Większość testów podstawia `runner` (zamiast procesów): udaje 03_train.py (katalog runa
z DONE.json), 04_evaluate.py (figures/metrics_val.json) i 07_compare_runs.py. Jeden test
`slow` uruchamia prawdziwy `run_all.py --profile smoke` w procesach, na syntetycznym Yale
w katalogu tymczasowym (zmienne SOR_AI_ESI_DATA_DIR / SOR_AI_RUNS_DIR / SOR_AI_LOGS_DIR).
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src.pipeline import keepawake, orchestrator
from src.pipeline.keepawake import KeepAwake
from src.pipeline.orchestrator import (
    DONE,
    FAILED,
    INTERRUPTED,
    NOT_STARTED,
    SKIPPED,
    Options,
    Orchestrator,
    assign_lanes,
    build_child_env,
    data_fingerprint,
    decide_lanes,
    find_done_run,
    format_duration,
    log_has_gpu_error,
    model_eta_s,
    optuna_progress,
    parse_feature_sets,
    parse_models,
    resolve_devices,
    resume_command,
    run_tag,
    status_report,
    train_command,
)
from src.utils import config
from src.utils.config import DATA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────
# Funkcje czyste
# ─────────────────────────────────────────
class TestParsing:
    def test_models_all_and_aliases_in_registry_order(self):
        assert parse_models("all") == list(orchestrator.MODEL_ORDER)
        assert parse_models("rf, xgb,lr") == ["logreg", "xgboost", "random_forest"]

    def test_models_unknown(self):
        with pytest.raises(ValueError, match="Nieznany model"):
            parse_models("lightgbm,svm")

    def test_feature_sets(self):
        assert parse_feature_sets("triage") == ["triage"]
        assert parse_feature_sets("triage_only,full,triage") == ["triage", "full"]
        with pytest.raises(ValueError):
            parse_feature_sets("wszystko")

    def test_registry_and_aliases_consistent(self):
        """Lokalne kopie (bez importu modeli w orkiestratorze) zgodne z src.models."""
        from src.models import MODEL_ALIASES, MODEL_REGISTRY

        assert list(orchestrator.MODEL_ORDER) == list(MODEL_REGISTRY)
        for alias, name in orchestrator.MODEL_ALIASES.items():
            assert MODEL_ALIASES.get(alias) == name

    def test_tag(self):
        assert run_tag("standard", True) == "standard"
        assert run_tag("smoke", False) == "smoke-notune"

    def test_cli_options(self):
        opts = orchestrator.options_from_args(["--profile", "smoke", "--models", "lgbm", "--no-tune", "--lanes", "1"])
        assert (opts.profile, opts.models, opts.tune, opts.lanes, opts.tag) == \
            ("smoke", ["lightgbm"], False, "1", "smoke-notune")
        defaults = orchestrator.options_from_args([])
        assert (defaults.profile, defaults.feature_sets, defaults.split) == ("standard", ["triage"], "random")
        with pytest.raises(SystemExit):
            orchestrator.options_from_args(["--models", "svm"])


def test_fingerprint_matches_tuning(tmp_path):
    from src.models.tuning import data_fingerprint as tuning_fingerprint

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"data_version": DATA_VERSION, "created_at": "teraz"}), encoding="utf-8")
    assert data_fingerprint(manifest) == tuning_fingerprint(manifest)
    assert data_fingerprint(tmp_path / "brak.json") == tuning_fingerprint(tmp_path / "brak.json")


def write_done_run(runs_dir: Path, name: str, model: str, fs: str = "triage", tag: str = "standard",
                   fingerprint: str = "abc", data_version: str = DATA_VERSION, tuned: bool = True,
                   metrics: dict | None = None) -> Path:
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    done = {"run_name": name, "model": model, "feature_set": fs, "tag": tag, "profile": tag.split("-")[0],
            "tuned": tuned, "data_version": data_version, "data_fingerprint": fingerprint,
            "actual_device": "cpu", "finished_at": "2026-10-19T12:00:00"}
    row = {"run_name": name, "model": model, "feature_set": fs, "actual_device": "cpu", "cv_metric": "logloss",
           "cv_score": 0.81, "cv_std": 0.004, "val_argmax_quadratic_weighted_kappa": 0.55,
           "val_cost_quadratic_weighted_kappa": 0.5, "val_cost_undertriage_rate": 0.1,
           "val_vc_argmax_quadratic_weighted_kappa": 0.52, "val_vc_cost_quadratic_weighted_kappa": 0.48,
           **(metrics or {})}
    info = {"run_name": name, "model": model, "feature_set": fs, "tag": tag, "data_version": data_version,
            "tuning": {"n_complete": 3, "n_pruned": 1, "n_fail": 0} if tuned else None,
            "device": {"requested": "cpu", "actual": "cpu", "fallback_reason": None}}
    (run_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    (run_dir / "leaderboard_row.json").write_text(json.dumps(row), encoding="utf-8")
    (run_dir / "DONE.json").write_text(json.dumps(done), encoding="utf-8")
    return run_dir


class TestFindDoneRun:
    def test_match_and_newest(self, tmp_path):
        write_done_run(tmp_path, "20261001_000000_lightgbm_triage_standard", "lightgbm")
        write_done_run(tmp_path, "20261002_000000_lightgbm_triage_standard", "lightgbm")
        found = find_done_run(tmp_path, "lightgbm", "triage", "standard", DATA_VERSION, "abc", tuned=True)
        assert found.name == "20261002_000000_lightgbm_triage_standard"

    @pytest.mark.parametrize("kwargs", [
        {"fingerprint": "inny"},                 # dane przygotowane ponownie
        {"data_version": "esi-v1"},              # stara wersja logiki danych
        {"tag": "quick"},                        # inny profil
        {"fs": "full"},                          # inny zestaw cech
        {"tuned": False},                        # run bez tuningu
    ])
    def test_mismatch(self, tmp_path, kwargs):
        name = f"20261001_000000_lightgbm_{kwargs.get('fs', 'triage')}_{kwargs.get('tag', 'standard')}"
        write_done_run(tmp_path, name, "lightgbm", **kwargs)
        assert find_done_run(tmp_path, "lightgbm", "triage", "standard", DATA_VERSION, "abc", tuned=True) is None

    def test_unfinished_run_ignored(self, tmp_path):
        (tmp_path / "20261003_000000_xgboost_triage_standard").mkdir()
        assert find_done_run(tmp_path, "xgboost", "triage", "standard", DATA_VERSION, "abc") is None

    def test_random_forest_name_with_underscore(self, tmp_path):
        write_done_run(tmp_path, "20261001_000000_random_forest_triage_standard", "random_forest")
        assert find_done_run(tmp_path, "random_forest", "triage", "standard", DATA_VERSION, "abc") is not None
        assert find_done_run(tmp_path, "forest", "triage", "standard", DATA_VERSION, "abc") is None


class TestLanes:
    ALL = list(orchestrator.MODEL_ORDER)

    @pytest.mark.parametrize("requested, gpu_ok, cores, ram, expected", [
        ("auto", True, 24, 64.0, 2),
        ("auto", False, 24, 64.0, 1),            # bez GPU — jeden tor
        ("auto", True, 8, 64.0, 1),              # za mało rdzeni
        ("auto", True, 24, 31.8, 1),             # za mało RAM
        ("1", True, 24, 64.0, 1),
        ("2", False, 4, 8.0, 2),                 # jawne żądanie
    ])
    def test_decide(self, requested, gpu_ok, cores, ram, expected):
        assert decide_lanes(requested, gpu_ok, cores, ram, self.ALL) == expected

    def test_single_lane_when_one_side_empty(self):
        assert decide_lanes("2", True, 24, 64.0, ["xgboost", "catboost"]) == 1
        assert decide_lanes("auto", True, 24, 64.0, ["lightgbm", "ebm"]) == 1

    def test_assign_two_lanes(self):
        gpu, cpu = assign_lanes(self.ALL, 2, 24)
        assert (gpu.name, gpu.models, gpu.n_jobs) == ("gpu", ["xgboost", "catboost"], 6)
        assert (cpu.name, cpu.models, cpu.n_jobs) == ("cpu", ["logreg", "lightgbm", "random_forest", "ebm"], 18)

    def test_assign_small_machine_threads_disjoint(self):
        gpu, cpu = assign_lanes(self.ALL, 2, 4)
        assert gpu.n_jobs + cpu.n_jobs == 4 and gpu.n_jobs >= 1 and cpu.n_jobs >= 1

    def test_assign_one_lane(self):
        (main,) = assign_lanes(["ebm", "xgboost", "logreg"], 1, 24)
        assert (main.name, main.models, main.n_jobs) == ("main", ["logreg", "xgboost", "ebm"], 24)


class TestEnvironment:
    def test_devices_precedence(self):
        detected = {"xgboost": "gpu", "catboost": "cpu"}
        assert resolve_devices({}, detected) == {"xgboost": "gpu", "catboost": "cpu"}
        assert resolve_devices({}, None) == {"xgboost": "auto", "catboost": "auto"}
        assert resolve_devices({"SOR_AI_DEVICE": "cpu"}, detected) == {"xgboost": "cpu", "catboost": "cpu"}
        env = {"SOR_AI_DEVICE": "cpu", "SOR_AI_DEVICE_XGBOOST": "CUDA"}
        assert resolve_devices(env, detected) == {"xgboost": "gpu", "catboost": "cpu"}

    def test_child_env(self):
        env = build_child_env({"PATH": "/bin", "SOR_AI_N_JOBS": "99"}, 6, {"xgboost": "gpu", "catboost": "cpu"})
        assert env["PATH"] == "/bin"
        assert env["SOR_AI_N_JOBS"] == env["OMP_NUM_THREADS"] == "6"
        assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
        assert env["SOR_AI_NO_AUTO_PREPARE"] == "1"
        assert env["SOR_AI_DEVICE_XGBOOST"] == "gpu" and env["SOR_AI_DEVICE_CATBOOST"] == "cpu"

    def test_train_command(self):
        cmd = train_command("xgboost", "triage", "standard", "standard", tune=True)
        assert cmd[:3] == [sys.executable, "-X", "utf8"]
        assert cmd[3].endswith("03_train.py")
        assert cmd[4:] == ["--model", "xgboost", "--feature-set", "triage", "--profile", "standard",
                           "--tag", "standard", "--tune"]
        assert "--tune" not in train_command("xgboost", "triage", "smoke", "smoke-notune", tune=False)


class TestHelpers:
    @pytest.mark.parametrize("text, expected", [
        ("12:00 | ERROR | xgboost: TRENING NIEUDANY — CatBoostError: CUDA error 35: driver insufficient", True),
        ("RuntimeError: SOR_AI_DEVICE_XGBOOST=gpu, ale xgboost nie może użyć GPU: …", True),
        ("xgboost.core.XGBoostError: [12:00] out of memory", True),
        ("12:00 | INFO | xgboost: trening na GPU (xgboost 3.4.1 (CUDA 13.3) trenuje na cuda:0)", False),
        ("12:00 | ERROR | lightgbm: TRENING NIEUDANY — ValueError: Input contains NaN", False),
        ("Error 35 (CUDA driver version is insufficient for CUDA runtime version) ignored while obtaining "
         "device count", False),
    ])
    def test_gpu_error_detection(self, text, expected):
        assert log_has_gpu_error("linia\n" + text + "\nkoniec") is expected

    def test_eta(self):
        std = config.TUNING_PROFILES["standard"]
        assert model_eta_s("xgboost", "standard", "gpu", True) == std["xgboost"]["timeout_s"] + 300
        cpu = config.CPU_FALLBACK_BUDGETS["standard"]["xgboost"]["timeout_s"]
        assert model_eta_s("xgboost", "standard", "cpu", True) == cpu + 300
        smoke = config.TUNING_PROFILES["smoke"]["lightgbm"]["timeout_s"]
        assert model_eta_s("lightgbm", "smoke", "cpu", True) == smoke + orchestrator.SAMPLED_FIT_ALLOWANCE_S
        assert model_eta_s("ebm", "standard", "cpu", False) == orchestrator.FINAL_FIT_ALLOWANCE_S["ebm"]

    def test_format_duration(self):
        assert format_duration(None) == "—"
        assert format_duration(42) == "42 s"
        assert format_duration(125) == "2 min 05 s"
        assert format_duration(7 * 3600 + 30 * 60) == "7 h 30 min"

    def test_resume_command_drops_one_off_flags(self):
        cmd = resume_command(["--profile", "thorough", "--force", "--dry-run", "--models", "a b"])
        assert cmd == "python scripts/run_all.py --profile thorough --models 'a b'"


# ─────────────────────────────────────────
# Blokada uśpienia
# ─────────────────────────────────────────
class _FakeKernel32:
    def __init__(self, result: int = 1) -> None:
        self.calls: list[int] = []
        self.result = result

    def SetThreadExecutionState(self, flags: int) -> int:  # noqa: N802 — nazwa z Win32
        self.calls.append(flags)
        return self.result


class _FakeProc:
    def __init__(self, cmd, alive: bool = True, **kwargs) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self._alive = alive
        self.stderr = None
        self.terminated = False

    def poll(self):
        return None if self._alive else 1

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._alive = False


class TestKeepAwake:
    def test_windows_sets_and_clears(self):
        k32 = _FakeKernel32()
        with KeepAwake(platform="win32", kernel32=k32) as awake:
            assert awake.active
        assert k32.calls == [keepawake.ES_CONTINUOUS | keepawake.ES_SYSTEM_REQUIRED, keepawake.ES_CONTINUOUS]

    def test_windows_failure_is_not_fatal(self):
        with KeepAwake(platform="win32", kernel32=_FakeKernel32(result=0)) as awake:
            assert not awake.active and "błąd" in awake.description

    def test_disabled(self):
        k32 = _FakeKernel32()
        with KeepAwake(enabled=False, platform="win32", kernel32=k32) as awake:
            assert not awake.active
        assert k32.calls == []

    def test_linux_without_systemd(self, monkeypatch):
        monkeypatch.setattr(keepawake.shutil, "which", lambda name: None)
        with KeepAwake(platform="linux") as awake:
            assert not awake.active and "systemd-inhibit" in awake.description

    def test_linux_systemd_inhibit(self, monkeypatch):
        procs: list[_FakeProc] = []
        monkeypatch.setattr(keepawake.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(keepawake.time, "sleep", lambda s: None)
        monkeypatch.setattr(keepawake.subprocess, "Popen", lambda cmd, **kw: procs.append(_FakeProc(cmd, **kw))
                            or procs[-1])
        with KeepAwake(platform="linux") as awake:
            assert awake.active
            cmd = procs[0].cmd
            assert cmd[0] == "/usr/bin/systemd-inhibit" and "--what=idle:sleep" in cmd
            assert f"--pid={os.getpid()}" in cmd                   # blokada znika razem z procesem
            assert procs[0].kwargs.get("start_new_session") is True
        assert procs[0].terminated

    def test_helper_dying_immediately(self, monkeypatch):
        monkeypatch.setattr(keepawake.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(keepawake.time, "sleep", lambda s: None)
        monkeypatch.setattr(keepawake.subprocess, "Popen", lambda cmd, **kw: _FakeProc(cmd, alive=False))
        with KeepAwake(platform="linux") as awake:
            assert not awake.active and "nie działa" in awake.description


# ─────────────────────────────────────────
# Orkiestrator z udawanymi procesami
# ─────────────────────────────────────────
class FakeRunner:
    """Udaje kroki: trening → run z DONE.json, ewaluacja → metrics_val.json, porównanie → raport."""

    _run_ids = itertools.count(1)          # unikalne nazwy runów także między instancjami

    def __init__(self, fail: dict[str, list[tuple[int, str]]] | None = None, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, list[str], dict[str, str]]] = []
        self.fail = fail or {}             # model → kolejne (kod, tekst logu) dla treningu
        self.delay = delay
        self._lock = threading.Lock()

    def kinds(self, kind: str) -> list[list[str]]:
        return [cmd for k, cmd, _ in self.calls if k == kind]

    def __call__(self, cmd, env, log_path, prefix) -> int:
        script = Path(cmd[3]).name
        kind = {"03_train.py": "train", "04_evaluate.py": "eval", "07_compare_runs.py": "compare",
                "02_preprocess.py": "prepare", "00_check_env.py": "preflight"}[script]
        with self._lock:
            self.calls.append((kind, list(cmd), dict(env)))
            counter = next(self._run_ids)
        if self.delay:
            time.sleep(self.delay)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "train":
            args = dict(zip(cmd[4::2], cmd[5::2]))
            model, fs, tag = args["--model"], args["--feature-set"], args["--tag"]
            queue = self.fail.get(model)
            if queue:
                code, text = queue.pop(0)
                log_path.write_text(text + "\n", encoding="utf-8")
                return code
            write_done_run(Path(config.RUNS_DIR), f"20261019_{counter:06d}_{model}_{fs}_{tag}", model, fs, tag,
                           fingerprint=data_fingerprint(), tuned="--tune" in cmd)
            log_path.write_text("ok\n", encoding="utf-8")
            return 0
        if kind == "eval":
            figures = Path(cmd[cmd.index("--run") + 1]) / "figures"
            figures.mkdir(parents=True, exist_ok=True)
            (figures / "metrics_val.json").write_text("{}", encoding="utf-8")
            return 0
        if kind == "compare":
            Path(cmd[cmd.index("--out") + 1]).write_text("# porównanie\n", encoding="utf-8")
            return 0
        if kind == "prepare":
            manifest = json.loads(Path(config.DATA_MANIFEST_JSON).read_text(encoding="utf-8"))
            manifest["split_strategy"] = cmd[cmd.index("--split") + 1]
            Path(config.DATA_MANIFEST_JSON).write_text(json.dumps(manifest), encoding="utf-8")
            return 0
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preflight.json").write_text(json.dumps(self.preflight_payload), encoding="utf-8")
        return self.preflight_rc

    preflight_payload: dict = {"facts": {"gpu_ok": True, "devices": {"xgboost": "gpu", "catboost": "gpu"},
                                         "physical_cores": 24, "ram_gb": 64.0}}
    preflight_rc: int = 0


@pytest.fixture
def workspace(prepared_paths, tmp_path, monkeypatch):
    """Gotowe splity (losowy podział) + runs/ i logs/ w tmp; stałe 4 wątki."""
    from src.utils import hardware

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "OPTUNA_STORAGE_DIR", tmp_path / "logs" / "optuna")
    monkeypatch.setattr(hardware, "n_jobs", lambda: 4)
    monkeypatch.setattr(hardware, "nvidia_smi_info", lambda: None)
    for var in ("SOR_AI_DEVICE", "SOR_AI_DEVICE_XGBOOST", "SOR_AI_DEVICE_CATBOOST"):
        monkeypatch.delenv(var, raising=False)
    return prepared_paths


def make_opts(**kwargs) -> Options:
    base = {"profile": "smoke", "models": ["logreg", "lightgbm", "xgboost"], "skip_preflight": True,
            "keep_awake": False, "argv": ["--profile", "smoke"]}
    base.update(kwargs)
    return Options(**base)


def by_model(orch: Orchestrator) -> dict[str, orchestrator.ModelTask]:
    return {t.model: t for t in orch.tasks}


class TestOrchestrator:
    def test_full_run_then_resume_skips(self, workspace):
        runner = FakeRunner()
        orch = Orchestrator(make_opts(), runner=runner)
        assert orch.run() == 0
        assert [c[4:6] for c in runner.kinds("train")] == [["--model", "logreg"], ["--model", "lightgbm"],
                                                           ["--model", "xgboost"]]
        assert len(runner.kinds("eval")) == 3 and len(runner.kinds("compare")) == 1
        assert runner.kinds("prepare") == []                          # splity gotowe, ten sam podział
        assert all(t.status == DONE and t.eval_status == DONE for t in orch.tasks)

        summary = orch.summaries[0]
        assert summary.name.startswith("run_all_summary__triage__smoke__")
        text = summary.read_text(encoding="utf-8")
        assert "Wyłącznie metryki VAL" in text and "QWK vc argmax" in text
        assert "0.5500" in text and "3/1/0" in text                    # QWK val i próby Optuny
        assert "comparison__triage__smoke__" in text
        status = json.loads((orch.log_dir / "status.json").read_text(encoding="utf-8"))
        assert {t["status"] for t in status["tasks"]} == {DONE}

        # Wznowienie: wszystko gotowe → tylko porównanie
        runner2 = FakeRunner()
        orch2 = Orchestrator(make_opts(), runner=runner2)
        assert orch2.run() == 0
        assert runner2.kinds("train") == [] and runner2.kinds("eval") == []
        assert all(t.status == SKIPPED for t in orch2.tasks)
        assert len(runner2.kinds("compare")) == 1

    def test_missing_eval_is_redone_on_resume(self, workspace):
        assert Orchestrator(make_opts(models=["logreg"]), runner=FakeRunner()).run() == 0
        run_dir = next(Path(config.RUNS_DIR).glob("*_logreg_triage_smoke"))
        (run_dir / "figures" / "metrics_val.json").unlink()
        runner = FakeRunner()
        orch = Orchestrator(make_opts(models=["logreg"]), runner=runner)
        assert orch.run() == 0
        assert runner.kinds("train") == [] and len(runner.kinds("eval")) == 1
        assert by_model(orch)["logreg"].eval_status == DONE

    def test_force_retrains(self, workspace):
        assert Orchestrator(make_opts(models=["logreg"]), runner=FakeRunner()).run() == 0
        runner = FakeRunner()
        assert Orchestrator(make_opts(models=["logreg"], force=True), runner=runner).run() == 0
        assert len(runner.kinds("train")) == 1

    def test_new_data_fingerprint_invalidates_done(self, workspace):
        assert Orchestrator(make_opts(models=["logreg"]), runner=FakeRunner()).run() == 0
        manifest = Path(config.DATA_MANIFEST_JSON)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["created_at"] = "ponownie przygotowane"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        runner = FakeRunner()
        assert Orchestrator(make_opts(models=["logreg"]), runner=runner).run() == 0
        assert len(runner.kinds("train")) == 1

    def test_gpu_error_retried_on_cpu(self, workspace):
        runner = FakeRunner(fail={"xgboost": [(1, "12:00 | ERROR | xgboost: TRENING NIEUDANY — "
                                                  "XGBoostError: CUDA error 35: driver version is insufficient")]})
        orch = Orchestrator(make_opts(models=["xgboost"]), runner=runner)
        assert orch.run() == 0
        trains = [env for kind, _, env in runner.calls if kind == "train"]
        assert len(trains) == 2
        assert trains[0]["SOR_AI_DEVICE_XGBOOST"] == "auto"           # bez preflight — decyzja procesu
        assert trains[1]["SOR_AI_DEVICE_XGBOOST"] == "cpu"
        task = by_model(orch)["xgboost"]
        assert task.status == DONE and task.retried_on_cpu
        assert task.logs[1].endswith("triage__xgboost__train_cpu.log")
        assert "powtórka na CPU" in orch.summaries[0].read_text(encoding="utf-8")

    def test_failure_without_gpu_text_not_retried_others_continue(self, workspace):
        runner = FakeRunner(fail={"lightgbm": [(1, "Traceback …\nValueError: zły parametr")]})
        orch = Orchestrator(make_opts(), runner=runner)
        assert orch.run() == 1
        tasks = by_model(orch)
        assert tasks["lightgbm"].status == FAILED and not tasks["lightgbm"].retried_on_cpu
        assert tasks["logreg"].status == DONE and tasks["xgboost"].status == DONE
        assert len(runner.kinds("train")) == 3
        text = orch.summaries[0].read_text(encoding="utf-8")
        assert "## Błędy" in text and "ValueError: zły parametr" in text      # ogon logu
        assert "Wznowienie" in text or "wznowienie" in text

    def test_fail_fast_stops_next_models(self, workspace):
        runner = FakeRunner(fail={"logreg": [(1, "ValueError")]})
        orch = Orchestrator(make_opts(fail_fast=True), runner=runner)
        assert orch.run() == 1
        tasks = by_model(orch)
        assert tasks["logreg"].status == FAILED
        assert tasks["lightgbm"].status == NOT_STARTED and tasks["xgboost"].status == NOT_STARTED
        assert len(runner.kinds("train")) == 1

    def test_child_interrupt_stops_run(self, workspace):
        runner = FakeRunner(fail={"logreg": [(130, "KeyboardInterrupt")]})
        orch = Orchestrator(make_opts(), runner=runner)
        assert orch.run() == 130
        tasks = by_model(orch)
        assert tasks["logreg"].status == INTERRUPTED
        assert tasks["lightgbm"].status == NOT_STARTED
        assert runner.kinds("compare") == []                          # po Ctrl+C bez porównania
        text = orch.summaries[0].read_text(encoding="utf-8")
        assert "PRZERWANO" in text and "python scripts/run_all.py --profile smoke" in text

    def test_keyboard_interrupt_in_main_thread(self, workspace, monkeypatch):
        """Ctrl+C w wątku głównym: bieżący krok kończy się sam, kolejne nie startują, kod 130."""
        runner = FakeRunner(delay=0.5)
        orch = Orchestrator(make_opts(), runner=runner)
        real_wait = orch._wait
        calls = {"n": 0}

        def wait_with_interrupt(threads):
            calls["n"] += 1
            if calls["n"] == 1:                    # pierwsze czekanie = tory modeli
                time.sleep(0.1)
                orch._handle_interrupt(threads)
            else:
                real_wait(threads)

        monkeypatch.setattr(orch, "_wait", wait_with_interrupt)
        assert orch.run() == 130
        tasks = by_model(orch)
        assert tasks["logreg"].status == DONE                         # dokończony w czasie karencji
        assert tasks["lightgbm"].status == NOT_STARTED and tasks["xgboost"].status == NOT_STARTED

    def test_internal_error_marks_model_failed_and_continues(self, workspace):
        class Broken(FakeRunner):
            def __call__(self, cmd, env, log_path, prefix):
                if "--model" in cmd and cmd[cmd.index("--model") + 1] == "logreg":
                    raise OSError("dysk pełny")
                return super().__call__(cmd, env, log_path, prefix)

        orch = Orchestrator(make_opts(models=["logreg", "lightgbm"]), runner=Broken())
        assert orch.run() == 1
        tasks = by_model(orch)
        assert tasks["logreg"].status == FAILED and "dysk pełny" in tasks["logreg"].error
        assert tasks["lightgbm"].status == DONE

    def test_preflight_failure_aborts(self, workspace):
        runner = FakeRunner()
        runner.preflight_rc = 1
        assert Orchestrator(make_opts(skip_preflight=False), runner=runner).run() == 2
        assert runner.kinds("train") == []

    def test_preflight_facts_drive_lanes_and_devices(self, workspace, monkeypatch):
        from src.utils import hardware

        monkeypatch.setattr(hardware, "n_jobs", lambda: 24)
        runner = FakeRunner()
        orch = Orchestrator(make_opts(skip_preflight=False, models=list(orchestrator.MODEL_ORDER)), runner=runner)
        assert orch.run() == 0
        assert [lane.name for lane in orch.lanes] == ["gpu", "cpu"]
        envs = {cmd[5]: env for kind, cmd, env in runner.calls if kind == "train"}
        assert envs["xgboost"]["SOR_AI_N_JOBS"] == "6" and envs["catboost"]["SOR_AI_N_JOBS"] == "6"
        assert envs["lightgbm"]["SOR_AI_N_JOBS"] == "18" and envs["ebm"]["OMP_NUM_THREADS"] == "18"
        assert envs["xgboost"]["SOR_AI_DEVICE_XGBOOST"] == "gpu"
        assert all(env["SOR_AI_NO_AUTO_PREPARE"] == "1" for env in envs.values())
        assert all(t.status == DONE for t in orch.tasks)

    def test_split_change_triggers_prepare(self, workspace):
        runner = FakeRunner()
        orch = Orchestrator(make_opts(models=["logreg"], split="site"), runner=runner)
        assert orch.run() == 0
        assert runner.kinds("prepare")[0][-2:] == ["--split", "site"]
        assert by_model(orch)["logreg"].status == DONE

    def test_prepare_failure_aborts(self, workspace):
        class BrokenPrepare(FakeRunner):
            def __call__(self, cmd, env, log_path, prefix):
                if Path(cmd[3]).name == "02_preprocess.py":
                    self.calls.append(("prepare", list(cmd), dict(env)))
                    return 1
                return super().__call__(cmd, env, log_path, prefix)

        runner = BrokenPrepare()
        orch = Orchestrator(make_opts(models=["logreg"], split="site"), runner=runner)
        assert orch.run() == 2
        assert runner.kinds("train") == []
        assert json.loads((orch.log_dir / "status.json").read_text(encoding="utf-8"))["prepare"] == FAILED

    def test_no_tune_uses_separate_tag(self, workspace):
        runner = FakeRunner()
        assert Orchestrator(make_opts(models=["logreg"], tune=False), runner=runner).run() == 0
        cmd = runner.kinds("train")[0]
        assert "--tune" not in cmd and cmd[cmd.index("--tag") + 1] == "smoke-notune"
        # Run bez tuningu nie zastępuje runa z tuningiem
        runner2 = FakeRunner()
        assert Orchestrator(make_opts(models=["logreg"]), runner=runner2).run() == 0
        assert len(runner2.kinds("train")) == 1

    def test_dry_run_runs_nothing(self, workspace, capsys):
        runner = FakeRunner()
        assert Orchestrator(make_opts(dry_run=True), runner=runner).run() == 0
        assert runner.calls == []
        assert not (Path(config.LOGS_DIR) / "run_all").exists()

    def test_status_report(self, workspace):
        assert Orchestrator(make_opts(models=["logreg"]), runner=FakeRunner()).run() == 0
        text = status_report(make_opts(models=["logreg", "lightgbm"]))
        lines = {line.split()[1]: line for line in text.splitlines() if line.startswith("triage ")}
        assert "gotowy" in lines["logreg"] and "0.550" in lines["logreg"]
        assert "nierozpoczęty" in lines["lightgbm"]
        assert "Ostatnie uruchomienie" in text

    def test_status_shows_partial_run(self, workspace):
        (Path(config.RUNS_DIR) / "20261019_000000_lightgbm_triage_smoke").mkdir(parents=True)
        text = status_report(make_opts(models=["lightgbm"]))
        assert "przerwany" in text


def test_optuna_progress_reads_study_counts(tmp_path):
    import optuna

    fp = "abcdef1234"
    storage = f"sqlite:///{(tmp_path / 'optuna_lightgbm.db').as_posix()}"
    name = f"lightgbm__triage__logloss__w-none__n100__k2__filter0__ss2__data-{fp}__smoke__deadbeef"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=name, storage=storage)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
    other = optuna.create_study(study_name=name.replace("__smoke__", "__quick__"), storage=storage)
    other.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
    assert optuna_progress(tmp_path, "lightgbm", "triage", "smoke", fp) == {"complete": 2}
    assert optuna_progress(tmp_path, "lightgbm", "full", "smoke", fp) is None
    assert optuna_progress(tmp_path, "xgboost", "triage", "smoke", fp) is None


# ─────────────────────────────────────────
# Nadpisania ścieżek zmiennymi (config: PATH ENV OVERRIDES)
# ─────────────────────────────────────────
def test_path_env_overrides_in_subprocess(tmp_path):
    env = {**os.environ, "SOR_AI_ESI_DATA_DIR": str(tmp_path / "esi"), "SOR_AI_RUNS_DIR": str(tmp_path / "runs"),
           "SOR_AI_LOGS_DIR": str(tmp_path / "logs")}
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); from src.utils import config as c; "
            "from src.data import dataset as d; "
            "print(json.dumps({k: str(getattr(c, k)) for k in ('ESI_DATA_DIR', 'TRAIN_PARQUET', 'DATA_MANIFEST_JSON', "
            "'DATA_AUDIT_MD', 'RUNS_DIR', 'LOGS_DIR', 'TRAINING_LOGS_DIR', 'OPTUNA_STORAGE_DIR')} | "
            "{'split_val': str(d.SPLIT_PATHS['val'])}))")
    out = subprocess.run([sys.executable, "-c", code, str(PROJECT_ROOT)], env=env, capture_output=True, text=True,
                         encoding="utf-8", timeout=120, check=True).stdout
    paths = json.loads(out.strip().splitlines()[-1])
    esi, runs, logs = (str((tmp_path / name).resolve()) for name in ("esi", "runs", "logs"))
    assert paths["ESI_DATA_DIR"] == esi
    assert paths["TRAIN_PARQUET"] == str(Path(esi) / "train.parquet")
    assert paths["DATA_MANIFEST_JSON"] == str(Path(esi) / "manifest.json")
    assert paths["DATA_AUDIT_MD"] == str(Path(esi) / "DATA_AUDIT.md")      # nie nadpisuje raportu w repo
    assert paths["split_val"] == str(Path(esi) / "val.parquet")
    assert paths["RUNS_DIR"] == runs
    assert paths["LOGS_DIR"] == logs and paths["TRAINING_LOGS_DIR"] == str(Path(logs) / "training")
    assert paths["OPTUNA_STORAGE_DIR"] == str(Path(logs) / "optuna")


# ─────────────────────────────────────────
# End-to-end: prawdziwe procesy, profil smoke
# ─────────────────────────────────────────
@pytest.mark.slow
def test_run_all_smoke_end_to_end(tmp_path, make_raw):
    audit = PROJECT_ROOT / "DATA_AUDIT.md"
    audit_before = audit.read_bytes() if audit.exists() else None

    raw_path = tmp_path / "raw.parquet"
    make_raw(n=1200, seed=11).to_parquet(raw_path)
    env = {**os.environ, "SOR_AI_ESI_DATA_DIR": str(tmp_path / "esi"), "SOR_AI_RUNS_DIR": str(tmp_path / "runs"),
           "SOR_AI_LOGS_DIR": str(tmp_path / "logs"), "SOR_AI_DEVICE": "cpu", "SOR_AI_N_JOBS": "1",
           "PYTHONUTF8": "1"}
    env.pop("SOR_AI_NO_AUTO_PREPARE", None)

    # Przygotowanie syntetycznego Yale do katalogu tymczasowego (nadpisania ścieżek zmiennymi)
    prep = ("import sys, pandas as pd; sys.path.insert(0, sys.argv[1]); "
            "from src.data.prepare import prepare_dataset; "
            "prepare_dataset(strategy='random', raw=pd.read_parquet(sys.argv[2]))")
    subprocess.run([sys.executable, "-c", prep, str(PROJECT_ROOT), str(raw_path)], env=env, cwd=PROJECT_ROOT,
                   capture_output=True, timeout=600, check=True)
    assert (tmp_path / "esi" / "manifest.json").exists()
    assert (tmp_path / "esi" / "DATA_AUDIT.md").exists()

    cmd = [sys.executable, "scripts/run_all.py", "--profile", "smoke", "--models", "logreg,lightgbm",
           "--skip-preflight", "--no-keep-awake"]
    first = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1800)
    assert first.returncode == 0, first.stdout[-6000:] + first.stderr[-3000:]

    runs = tmp_path / "runs"
    done = sorted(runs.glob("*_triage_smoke/DONE.json"))
    assert {json.loads(p.read_text(encoding="utf-8"))["model"] for p in done} == {"logreg", "lightgbm"}
    for marker in done:
        assert (marker.parent / "figures" / "metrics_val.json").exists()          # ewaluacja na val
        assert json.loads(marker.read_text(encoding="utf-8"))["tag"] == "smoke"
    summaries = list(runs.glob("run_all_summary__triage__smoke__*.md"))
    assert len(summaries) == 1
    summary = summaries[0].read_text(encoding="utf-8")
    assert "logreg" in summary and "lightgbm" in summary and "Wyłącznie metryki VAL" in summary
    assert list(runs.glob("comparison__triage__smoke__*.md"))
    assert (runs / "leaderboard.csv").exists()
    first_logs = sorted((tmp_path / "logs" / "run_all").iterdir())
    assert (first_logs[-1] / "triage__logreg__train.log").exists()
    assert (first_logs[-1] / "status.json").exists()

    # Ponowne uruchomienie: oba modele pominięte (bez treningu i ewaluacji)
    time.sleep(1.1)                                    # inny znacznik czasu katalogu logów
    t0 = time.time()
    second = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=600)
    elapsed = time.time() - t0
    assert second.returncode == 0, second.stdout[-6000:] + second.stderr[-3000:]
    assert "pomijam — gotowy run" in second.stdout
    assert sorted(runs.glob("*_triage_smoke/DONE.json")) == done                 # żadnych nowych runów
    latest_logs = sorted((tmp_path / "logs" / "run_all").iterdir())[-1]
    assert latest_logs != first_logs[-1]
    assert not list(latest_logs.glob("*__train.log")) and not list(latest_logs.glob("*__eval.log"))
    assert elapsed < 300

    # Raport z audytu syntetycznych danych nie nadpisał wersjonowanego DATA_AUDIT.md
    assert (audit.read_bytes() if audit.exists() else None) == audit_before
