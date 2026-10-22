"""Runner tests with fake eval modules: ordering, failure isolation, control verdicts, baseline regression gate, report."""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest

from skillllm_v2.evals import registry
from skillllm_v2.evals import run_all as ra
from skillllm_v2.evals.contract import EvalContext, EvalResult
from skillllm_v2.evals.registry import EvalSpec


def R(name, metric="f1", value=0.8, module="fake", **kw):
    return EvalResult(name, module, metric, value, kw.pop("n", 10), "synthetic-toy", kw.pop("ci", (value - 0.05, value + 0.05) if value is not None else None), **kw)


class Fake:
    """Mutable fake eval module: tests change `.results` between runs."""

    def __init__(self, results=None, boom=None):
        self.results, self.boom, self.calls = results or [], boom, 0

    def __call__(self, ctx):
        self.calls += 1
        if self.boom:
            raise self.boom
        return list(self.results)


def run_cli(tmp_path, specs, *args):
    out = tmp_path / "out"
    code = ra.main(["--root", str(tmp_path), "--out", str(out), "--baseline", str(tmp_path / "baseline.json"), "--no-crosscheck", *args], registry=specs)
    return code, out


def load(out):
    return json.loads((out / "results.json").read_text())


# ------------------------------------------------------------------ registry
def test_registry_order_and_declared_stages():
    assert registry.names()[:3] == ["extraction", "verifier", "calibration"]
    assert {"match", "gap", "e2e"} <= set(registry.names())
    stages = {s for s, _ in registry.PIPELINE_STAGES}
    for spec in registry.REGISTRY:
        assert set(spec.stages) | set(spec.partial_stages) <= stages


def test_registry_load_is_defensive(tmp_path, monkeypatch):
    fn, status, why = registry.load(EvalSpec("ghost", "skillllm_v2.evals.no_such_module_xyz"))
    assert fn is None and status == "not_available"
    broken = tmp_path / "broken_eval_mod.py"
    broken.write_text("import definitely_not_installed_pkg\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    fn, status, why = registry.load(EvalSpec("b", "broken_eval_mod"))
    assert fn is None and status == "import_error" and "definitely_not_installed_pkg" in why
    (tmp_path / "norun_mod.py").write_text("x = 1\n")
    assert registry.load(EvalSpec("n", "norun_mod"))[1] == "no_run_function"
    with pytest.raises(ValueError):
        registry.select(["nope"])


# ------------------------------------------------------------------ isolation + exit codes
def test_failure_isolation_and_missing_module(tmp_path):
    a = Fake([R("a.metric", value=0.9)])
    b = Fake(boom=RuntimeError("kaboom"))
    c = Fake([R("c.metric", value=0.7)])
    specs = [EvalSpec("a", "x", run_fn=a, stages=("extraction",)), EvalSpec("b", "x", run_fn=b, stages=("graph",)),
             EvalSpec("ghost", "skillllm_v2.evals.ghost_evals", stages=("explanation",)), EvalSpec("c", "x", run_fn=c, stages=("job_matching",))]
    code, out = run_cli(tmp_path, specs)
    d = load(out)
    status = {m["name"]: m["status"] for m in d["modules"]}
    assert status == {"a": "ok", "b": "failed", "ghost": "not_available", "c": "ok"}
    assert c.calls == 1  # ran after the failure
    assert "RuntimeError: kaboom" in d["modules"][1]["reason"] and "Traceback" in d["modules"][1]["traceback"]
    assert code == 2  # crashed module -> 2; absent module alone would be 0
    cov = {x["stage"]: x["state"] for x in d["coverage"]}
    assert cov["extraction"] == "covered" and cov["explanation"] == "declared, module unavailable" and cov["gap_detection"] == "not covered"


def test_only_ghost_module_is_not_a_failure(tmp_path):
    code, out = run_cli(tmp_path, [EvalSpec("ghost", "skillllm_v2.evals.ghost_evals")])
    assert code == 0 and load(out)["modules"][0]["status"] == "not_available"


def test_non_evalresult_return_is_isolated(tmp_path):
    code, out = run_cli(tmp_path, [EvalSpec("bad", "x", run_fn=lambda ctx: [{"not": "a result"}])])
    assert code == 2 and load(out)["modules"][0]["status"] == "failed"


def test_only_filter_and_unknown_name(tmp_path):
    a, b = Fake([R("a.m")]), Fake([R("b.m")])
    specs = [EvalSpec("a", "x", run_fn=a), EvalSpec("b", "x", run_fn=b)]
    code, out = run_cli(tmp_path, specs, "--only", "b")
    assert code == 0 and a.calls == 0 and b.calls == 1
    assert ra.main(["--root", str(tmp_path), "--only", "zzz", "--out", str(tmp_path / "o")], registry=specs) == 3


# ------------------------------------------------------------------ controls
@pytest.mark.parametrize("res,expect", [
    (R("m.control.x", value=1.0, details={"control": True, "passed": True}), "pass"),
    (R("m.control.x", value=1.0, details={"control": True, "passed": False}), "fail"),
    (R("m.negative_control", value=0.0, status="failed", reason="did not collapse"), "fail"),
    (R("ctx.neg", metric="x", value=0.1, details={"control_passed": False}), "fail"),
    (R("m.control.y", value=0.1, status="skipped", reason="no data"), "skipped"),
    (R("m.control.z", value=0.1), "unverified"),
])
def test_control_verdicts(res, expect):
    assert ra.is_control(res)
    assert ra.control_verdict(res, {res.eval_name: res})["verdict"] == expect


def test_runner_side_rules_for_bare_value_controls():
    real = R("context.retriever_mrr", metric="mrr", value=0.6)
    rnd_ok = R("context.random_baseline_mrr", metric="mrr (negative control)", value=0.2)
    rnd_bad = R("context.random_baseline_mrr", metric="mrr (negative control)", value=0.7)
    assert ra.control_verdict(rnd_ok, {"context.retriever_mrr": real})["verdict"] == "pass"
    assert ra.control_verdict(rnd_bad, {"context.retriever_mrr": real})["verdict"] == "fail"
    assert not ra.is_control(R("match.baseline.random.ndcg_at_10"))


def test_failed_control_gives_exit_1(tmp_path):
    f = Fake([R("a.m", value=0.9), R("a.control.oracle", metric="ndcg", value=0.5, details={"control": True, "passed": False, "expected": "1.0"})])
    code, out = run_cli(tmp_path, [EvalSpec("a", "x", run_fn=f)])
    d = load(out)
    assert code == 1 and d["integrity"][0]["verdict"] == "fail"
    assert "**FAIL**" in (out / "REPORT.md").read_text()


# ------------------------------------------------------------------ baseline gate
def test_first_full_run_creates_baseline_then_gate_detects_regression(tmp_path):
    f = Fake([R("a.f1", metric="f1", value=0.80), R("a.viol", metric="violation_rate", value=0.0, details={"direction": "lower"}),
              R("a.lat", metric="mean_latency_ms", value=10.0), R("a.info", metric="count", value=12.0)])
    specs = [EvalSpec("a", "x", run_fn=f)]
    code, out = run_cli(tmp_path, specs)
    assert code == 0 and (tmp_path / "baseline.json").exists()
    base = json.loads((tmp_path / "baseline.json").read_text())
    assert base["smoke"] is False and "a.f1|f1" in base["metrics"] and base["metrics"]["a.f1|f1"]["direction"] == "higher"
    assert base["metrics"]["a.info|count"]["direction"] == "none"

    code, out = run_cli(tmp_path, specs)  # identical run
    assert code == 0 and load(out)["regression"]["regressions"] == []

    f.results = [R("a.f1", metric="f1", value=0.70), R("a.viol", metric="violation_rate", value=0.10, details={"direction": "lower"}),
                 R("a.lat", metric="mean_latency_ms", value=20.0), R("a.info", metric="count", value=999.0)]
    code, out = run_cli(tmp_path, specs)
    keys = {x["key"] for x in load(out)["regression"]["regressions"]}
    assert code == 1 and keys == {"a.f1|f1", "a.viol|violation_rate"}  # latency within x3 tolerance; 'none' direction never gated
    assert "Regression gate" in (out / "REPORT.md").read_text()

    f.results = [R("a.f1", metric="f1", value=0.82), R("a.viol", metric="violation_rate", value=0.0, details={"direction": "lower"}),
                 R("a.lat", metric="mean_latency_ms", value=500.0), R("a.info", metric="count", value=1.0)]
    code, out = run_cli(tmp_path, specs)
    assert code == 1 and {x["key"] for x in load(out)["regression"]["regressions"]} == {"a.lat|mean_latency_ms"}

    f.results = [R("a.f1", metric="f1", value=0.60)]  # within --tolerance when loosened
    code, out = run_cli(tmp_path, specs, "--tolerance", "0.5")
    assert code == 0


def test_update_baseline_and_tolerance_boundary(tmp_path):
    f = Fake([R("a.f1", value=0.80)])
    specs = [EvalSpec("a", "x", run_fn=f)]
    run_cli(tmp_path, specs)
    f.results = [R("a.f1", value=0.78)]  # -0.02: inside default 0.03
    assert run_cli(tmp_path, specs)[0] == 0
    f.results = [R("a.f1", value=0.60)]
    assert run_cli(tmp_path, specs)[0] == 1
    assert run_cli(tmp_path, specs, "--update-baseline")[0] == 0
    assert run_cli(tmp_path, specs)[0] == 0  # new normal


def test_failed_metric_in_gated_eval_counts_as_regression(tmp_path):
    f = Fake([R("a.f1", value=0.80)])
    specs = [EvalSpec("a", "x", run_fn=f)]
    run_cli(tmp_path, specs)
    f.results = [R("a.f1", value=None, status="failed", reason="broke")]
    code, out = run_cli(tmp_path, specs)
    assert code == 1


def test_smoke_never_writes_or_updates_baseline_and_skips_mismatched_gate(tmp_path):
    f = Fake([R("a.f1", value=0.8)])
    specs = [EvalSpec("a", "x", run_fn=f)]
    assert run_cli(tmp_path, specs, "--smoke")[0] == 0
    assert not (tmp_path / "baseline.json").exists()  # smoke does not create one
    assert run_cli(tmp_path, specs, "--smoke", "--update-baseline")[0] == 3
    run_cli(tmp_path, specs)  # full baseline
    f.results = [R("a.f1", value=0.1)]
    code, out = run_cli(tmp_path, specs, "--smoke")  # tiny subset numbers must not be gated against full baseline
    assert code == 0 and load(out)["regression"]["skipped_gate"] is True
    assert run_cli(tmp_path, specs, "--only", "a", "--update-baseline")[0] == 3


def test_partial_run_only_compares_selected_modules(tmp_path):
    a, b = Fake([R("a.f1", value=0.8)]), Fake([R("b.f1", value=0.8)])
    specs = [EvalSpec("a", "x", run_fn=a), EvalSpec("b", "x", run_fn=b)]
    run_cli(tmp_path, specs)
    b.results = [R("b.f1", value=0.1)]
    code, out = run_cli(tmp_path, specs, "--only", "a")
    assert code == 0 and load(out)["regression"]["dropped"] == []


# ------------------------------------------------------------------ outputs
def test_outputs_are_strict_json_and_report_has_required_sections(tmp_path):
    f = Fake([R("a.nan", value=float("nan"), ci=(float("nan"), float("nan"))), R("a.ok", value=0.5, caveats=["tiny | pipe\nnewline"], details={"x": (1, 2), "s": {1, 2}})])
    code, out = run_cli(tmp_path, [EvalSpec("a", "x", run_fn=f, stages=("extraction",))])
    text = (out / "results.json").read_text()
    json.loads(text)  # strict parse
    assert "NaN" not in text and "Infinity" not in text
    rep = (out / "REPORT.md").read_text()
    for section in ("## Run metadata", "## Per-module summary", "## Coverage vs the pipeline stages", "## Integrity", "## Oracle cross-check", "## Master table", "## Regression gate"):
        assert section in rep
    assert "| module | eval | metric | value [95% CI] | n | dataset | status | caveats |" in rep
    meta = load(out)["meta"]
    for k in ("git_commit", "seed", "started_utc", "finished_utc", "python", "smoke"):
        assert k in meta
    assert "tiny / pipe newline" in rep  # table-safe caveat text


def test_smoke_output_goes_to_subdir_by_default(tmp_path):
    f = Fake([R("a.f1")])
    code = ra.main(["--root", str(tmp_path), "--smoke", "--no-crosscheck"], registry=[EvalSpec("a", "x", run_fn=f)])
    assert code == 0 and (tmp_path / "reports/skillllm_v2/eval_suite/smoke/REPORT.md").exists()
    assert not (tmp_path / "reports/skillllm_v2/eval_suite/results.json").exists()


# ------------------------------------------------------------------ oracle cross-check
def test_crosscheck_skips_missing_libs_and_agrees_with_scipy():
    checks = {c["name"]: c for c in ra.oracle_crosscheck(1, 300)}
    assert all(c["status"] in ("ok", "skipped") for c in checks.values())
    for name, c in checks.items():
        if c["status"] == "skipped":
            assert c["note"]
    pytest.importorskip("scipy.stats")
    assert checks["stats.kendall_tau_b"]["status"] == "ok" and checks["stats.paired_bootstrap.delta"]["max_abs_diff"] < 1e-12


def test_crosscheck_disagreement_fails_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "oracle_crosscheck", lambda seed, n_boot: [ra._xc("fake.metric", "lib", "ok", 0.5, 1e-9)])
    code = ra.main(["--root", str(tmp_path), "--out", str(tmp_path / "o"), "--baseline", str(tmp_path / "b.json")],
                   registry=[EvalSpec("a", "x", run_fn=Fake([R("a.f1")]))])
    assert code == 1
    assert "DISAGREE" in (tmp_path / "o" / "REPORT.md").read_text() or "FAILED" in (tmp_path / "o" / "REPORT.md").read_text()
