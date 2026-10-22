"""Run the whole evaluation suite and write ``results.json`` + ``REPORT.md``.

    python -m skillllm_v2.evals.run_all [--only match,gap] [--smoke] [--seed 42] [--out DIR]
                                        [--update-baseline] [--baseline PATH] [--tolerance 0.03] [--list]

Exit codes: 0 = clean; 1 = a control failed or a gated metric regressed past tolerance versus the baseline;
2 = an eval module crashed / failed to import (and nothing above); 3 = bad usage.  Modules that simply do not exist yet
are reported as ``not_available`` and never fail the run.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from skillllm_v2.evals.contract import EvalContext, EvalResult
from skillllm_v2.evals.registry import PIPELINE_STAGES, REGISTRY, EvalSpec, load, select

DEFAULT_OUT = Path("reports/skillllm_v2/eval_suite")
DEFAULT_TOLERANCE = 0.03
LATENCY_REL_TOLERANCE = 2.0  # latency regresses only when > (1 + 2.0) x baseline + 5 ms

# ----------------------------------------------------------------------------------------------- helpers
def _clean(o: Any) -> Any:
    """JSON-safe copy: NaN/inf -> None, tuples -> lists, unknown objects -> str."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in (sorted(o, key=str) if isinstance(o, set) else o)]
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    try:
        import numpy as np  # noqa: WPS433
        if isinstance(o, np.generic):
            return _clean(o.item())
        if isinstance(o, np.ndarray):
            return _clean(o.tolist())
    except Exception:  # noqa: BLE001
        pass
    return str(o)


def git_info(root: Path) -> Dict[str, Any]:
    for base in [root.resolve(), *list(root.resolve().parents)[:3]]:
        try:
            head = subprocess.run(["git", "-C", str(base), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True, timeout=10)
            if head.returncode == 0 and head.stdout.strip():
                dirty = subprocess.run(["git", "-C", str(base), "status", "--porcelain"], capture_output=True, text=True, timeout=20)
                return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip()), "repo": str(base)}
        except Exception:  # noqa: BLE001
            continue
    return {"commit": "unknown", "dirty": None, "repo": None}


def default_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "src" / "skillllm_v2").exists():
        return cwd
    return Path(__file__).resolve().parents[3]


# ----------------------------------------------------------------------------------------------- control detection
_CONTROL_RE = re.compile(r"(^|[._\s(])(negative_control|control)([._\s)]|$)|negative control|\bcontrol_", re.I)


def is_control(r: EvalResult) -> bool:
    d = r.details or {}
    if d.get("control") is True or "control_passed" in d:
        return True
    return bool(_CONTROL_RE.search(r.eval_name) or "negative control" in r.metric.lower() or "negative_control" in r.metric.lower())


def _num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _rule_calibration_random(r: EvalResult, by: Dict[str, EvalResult]) -> Optional[Tuple[bool, str]]:
    exp = _num((r.details or {}).get("ece_width_analytic_expectation"))
    if r.value is None or exp is None:
        return None
    return abs(r.value - exp) < 0.05, f"random-confidence ECE within 0.05 of its analytic expectation ({exp:.3f})"


def _rule_calibration_constant(r: EvalResult, by: Dict[str, EvalResult]) -> Optional[Tuple[bool, str]]:
    base = _num((r.details or {}).get("aurc_expected_1-p"))
    aurc = _num((r.details or {}).get("aurc"))
    if r.value is None or aurc is None or base is None:
        return None
    return r.value < 0.05 and abs(aurc - base) < 0.02, "constant base-rate confidence: ECE ~ 0 and AURC = base risk (no selective ability)"


def _rule_zero(limit: float, text: str) -> Callable[[EvalResult, Dict[str, EvalResult]], Optional[Tuple[bool, str]]]:
    def rule(r: EvalResult, by: Dict[str, EvalResult]) -> Optional[Tuple[bool, str]]:
        return None if r.value is None else (r.value <= limit, text)
    return rule


def _rule_below(other: str, text: str) -> Callable[[EvalResult, Dict[str, EvalResult]], Optional[Tuple[bool, str]]]:
    def rule(r: EvalResult, by: Dict[str, EvalResult]) -> Optional[Tuple[bool, str]]:
        o = by.get(other)
        if r.value is None or o is None or o.value is None:
            return None
        return r.value < o.value, text
    return rule


# Runner-side verdicts for controls that report a bare value without a pass flag.  (eval_name regex, rule).
CONTROL_RULES: List[Tuple[str, Callable[[EvalResult, Dict[str, EvalResult]], Optional[Tuple[bool, str]]]]] = [
    (r"^calibration\..*\.control_random_ece$", _rule_calibration_random),
    (r"^calibration\..*\.control_constant_ece$", _rule_calibration_constant),
    (r"^context\.random_baseline_mrr$", _rule_below("context.retriever_mrr", "random ranking scores below the real retriever")),
    (r"^graph\.random_baseline_recall_at_5$", _rule_below("graph.contextual_retriever_recall_at_5", "random course ranking scores below the contextual retriever")),
    (r"audit_false_alarm_control$", _rule_zero(0.05, "removing an unrelated edge triggers (almost) no audit alarm")),
    (r"^link\.negative_control$", _rule_zero(0.02, "random taxonomy ids score ~0 top-1 accuracy")),
]


# Controls whose module encodes the verdict in `status` ("ok" = control held, "failed" = it did not).
STATUS_VERDICT_PATTERNS = [(r"^extract\..*\.negative_control$", "shuffled predictions collapse to ~0 F1 and gold-as-prediction scores 1 (module's own verdict via status)")]


def control_verdict(r: EvalResult, by: Dict[str, EvalResult]) -> Dict[str, Any]:
    """verdict: pass | fail | skipped | unverified (control reported but the suite has no pass criterion for it)."""
    d = r.details or {}
    expected = str(d.get("expected") or "")
    if r.status == "failed":
        verdict, why = "fail", r.reason or "marked failed by the eval"
    elif r.status == "skipped":
        verdict, why = "skipped", r.reason
    elif "passed" in d or "control_passed" in d:
        ok = bool(d.get("passed", d.get("control_passed")))
        verdict, why = ("pass" if ok else "fail"), ""
    else:
        verdict, why = "unverified", "no explicit pass flag; no runner rule"
        for pat, text in STATUS_VERDICT_PATTERNS:
            if re.search(pat, r.eval_name):
                verdict, expected, why = "pass", text, ""
        for pat, rule in CONTROL_RULES:
            if re.search(pat, r.eval_name):
                try:
                    out = rule(r, by)
                except Exception:  # noqa: BLE001
                    out = None
                if out is not None:
                    verdict, expected, why = ("pass" if out[0] else "fail"), out[1], ""
                break
    return {"eval": r.eval_name, "module": r.module, "metric": r.metric, "value": r.value, "verdict": verdict,
            "expected": expected or why or r.metric, "reason": why if verdict in ("fail", "skipped") else ""}


# ----------------------------------------------------------------------------------------------- run
@dataclass
class ModuleRun:
    name: str
    module_path: str
    status: str  # ok | not_available | import_error | no_run_function | failed
    reason: str = ""
    seconds: float = 0.0
    results: List[EvalResult] = field(default_factory=list)
    traceback: str = ""


def run_module(spec: EvalSpec, ctx: EvalContext) -> ModuleRun:
    fn, status, reason = load(spec)
    if fn is None:
        return ModuleRun(spec.name, spec.module_path, status, reason)
    random.seed(ctx.seed)
    t = time.perf_counter()
    try:
        out = fn(ctx)
        if not isinstance(out, (list, tuple)):
            raise TypeError(f"run(ctx) must return a list of EvalResult, got {type(out).__name__}")
        bad = [type(x).__name__ for x in out if not isinstance(x, EvalResult)]
        if bad:
            raise TypeError(f"run(ctx) returned non-EvalResult items: {sorted(set(bad))}")
        return ModuleRun(spec.name, spec.module_path, "ok", "", time.perf_counter() - t, list(out))
    except Exception as exc:  # noqa: BLE001 - isolation: one failing eval must not stop the suite
        tb = traceback.format_exc()
        res = EvalResult(f"{spec.name}.__module_run__", spec.name, "module_run", None, 0, "-", None, {}, [],
                         "failed", f"{type(exc).__name__}: {exc}")
        return ModuleRun(spec.name, spec.module_path, "failed", f"{type(exc).__name__}: {exc}", time.perf_counter() - t, [res], tb)


def run_suite(ctx: EvalContext, specs: Sequence[EvalSpec]) -> List[ModuleRun]:
    return [run_module(s, ctx) for s in specs]


# ----------------------------------------------------------------------------------------------- regression gate
_LOWER = ("violation", "error", "latency", "ece", "brier", "aurc", "loss", "false_alarm", "false_positive", "false_gap", "false_transferable",
          "spurious", "regret", "unsupported", "tie_rate", "clamp_rate", "tie_sensitivity", "max_abs", "abs_error", "fpr", "mismatch")
_HIGHER = ("f1", "recall", "precision", "accuracy", "ndcg", "mrr", "map_at", "hit_at", "hit@", "auroc", "auc", "coverage", "agreement",
           "valid_output", "traceab", "graceful", "identical", "jaccard", "kendall", "spearman", "rho", "linkage", "adherence", "success", "exact")


def metric_direction(r: EvalResult) -> str:
    d = (r.details or {}).get("direction")
    if d in ("higher", "lower", "none"):
        return d
    text = f"{r.eval_name} {r.metric}".lower()
    if any(t in text for t in _LOWER):
        return "lower"
    if any(t in text for t in _HIGHER):
        return "higher"
    return "none"


def is_latency(r: EvalResult) -> bool:
    text = f"{r.eval_name} {r.metric}".lower()
    return "latency" in text or bool((r.details or {}).get("hardware_dependent"))


def metric_key(r: EvalResult) -> str:
    return f"{r.eval_name}|{r.metric}"


def baseline_from(results: Sequence[Tuple[str, EvalResult]], meta: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {}
    for spec, r in results:
        if r.status != "ok" or r.value is None or is_control(r):
            continue
        v = _num(r.value)
        if v is None:
            continue
        metrics[metric_key(r)] = {"value": v, "n": r.n, "direction": metric_direction(r), "latency": is_latency(r), "spec": spec}
    return {"created_utc": meta["finished_utc"], "seed": meta["seed"], "smoke": meta["smoke"], "git_commit": meta["git"]["commit"],
            "tolerance_default": DEFAULT_TOLERANCE, "metrics": metrics}


def compare_baseline(results: Sequence[EvalResult], baseline: Dict[str, Any], tol: float, smoke: bool, seed: int,
                     specs_run: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"compared": 0, "regressions": [], "improvements": 0, "unchanged": 0, "new": 0, "dropped": [], "notes": [],
                           "skipped_gate": False, "tolerance": tol}
    if baseline.get("smoke") != smoke:
        out["skipped_gate"] = True
        out["notes"].append(f"baseline was recorded with smoke={baseline.get('smoke')} but this run has smoke={smoke}; regression gate skipped")
        return out
    if baseline.get("seed") != seed:
        out["notes"].append(f"seed differs from baseline ({baseline.get('seed')} vs {seed}); synthetic datasets differ, interpret deltas with care")
    cur = {metric_key(r): r for r in results if not is_control(r)}
    for key, b in baseline.get("metrics", {}).items():
        if specs_run is not None and b.get("spec") not in specs_run:
            continue  # partial run: other modules were not asked for
        r = cur.get(key)
        if r is None or r.status != "ok" or r.value is None:
            if r is not None and r.status == "failed":
                out["regressions"].append({"key": key, "baseline": b["value"], "current": None, "delta": None, "reason": f"eval failed: {r.reason}"})
            else:
                out["dropped"].append(key)
            continue
        direction = b.get("direction", "none")
        if direction == "none":
            continue
        out["compared"] += 1
        delta = r.value - b["value"]
        if b.get("latency"):
            bad = r.value > b["value"] * (1 + LATENCY_REL_TOLERANCE) + 5.0
        elif direction == "higher":
            bad = delta < -tol
        else:
            bad = delta > tol
        better = (delta > 1e-9) if direction == "higher" else (delta < -1e-9)
        if bad:
            out["regressions"].append({"key": key, "baseline": b["value"], "current": r.value, "delta": delta, "direction": direction,
                                       "reason": f"worse by {abs(delta):.4f} (> tolerance {tol if not b.get('latency') else 'latency x3'})"})
        elif better:
            out["improvements"] += 1
        else:
            out["unchanged"] += 1
    out["new"] = len([k for k in cur if k not in baseline.get("metrics", {}) and cur[k].status == "ok" and cur[k].value is not None])
    return out


# ----------------------------------------------------------------------------------------------- oracle cross-check
def _try_import(name: str) -> Tuple[Any, str]:
    try:
        import importlib
        return importlib.import_module(name), ""
    except Exception as exc:  # noqa: BLE001 - optional oracle libraries are never hard dependencies
        return None, f"{type(exc).__name__}: {str(exc)[:80]}"


def _xc(name: str, library: str, status: str, diff: Optional[float] = None, tol: Optional[float] = None, note: str = "", info: Optional[dict] = None) -> Dict[str, Any]:
    if status == "ok" and diff is not None and tol is not None and diff > tol:
        status = "failed"
    return {"name": name, "library": library, "status": status, "max_abs_diff": diff, "tolerance": tol, "note": note, "info": info or {}}


def oracle_crosscheck(seed: int = 42, n_boot: int = 2000) -> List[Dict[str, Any]]:
    """Recompute a sample of metrics through independent reference libraries (when importable) and record max abs diff.

    Libraries are optional: missing ones give status 'skipped' with the import error.  A diff above tolerance is 'failed'
    and fails the suite.  Nothing is installed.
    """
    from skillllm_v2.eval import ranking_metrics as rm
    from skillllm_v2.evals.contract import bootstrap_ci, paired_bootstrap

    out: List[Dict[str, Any]] = []
    rng = random.Random(f"{seed}:crosscheck")
    # ---- ranking metrics vs ir_measures (+ pytrec_eval underneath)
    irm, why_ir = _try_import("ir_measures")
    pte, why_pt = _try_import("pytrec_eval")
    toy = []
    for q in range(60):
        docs = [f"d{i}" for i in range(40)]
        grades = {d: rng.choices([0, 1, 2, 3], [0.6, 0.2, 0.12, 0.08])[0] for d in docs}
        if q % 12 == 0:  # deliberate zero-relevant scenarios to exercise the drop-vs-count-as-0 policy difference
            grades = {d: 0 for d in docs}
        ranked = rng.sample(docs, 25)
        toy.append((f"q{q}", grades, ranked))
    if irm is None or pte is None:
        why = why_ir if irm is None else f"pytrec_eval: {why_pt}"
        for n in ("ndcg_at_10", "precision_at_5(rel>=2)", "recall_at_10(rel>=2)", "ap_at_10(R<=10 scenarios)"):
            out.append(_xc(f"ranking_metrics.{n}", "ir_measures+pytrec_eval", "skipped", note=f"not importable ({why})"))
    else:
        try:
            qrels = [irm.Qrel(q, d, int(2 ** g - 1)) for q, gr, _ in toy for d, g in gr.items()]  # pre-applied exp gain; linear nDCG on top
            run_ = [irm.ScoredDoc(q, d, float(len(r) - i)) for q, _, r in toy for i, d in enumerate(r)]

            def lib(measure: Any) -> Dict[str, float]:
                return {m.query_id: m.value for m in irm.iter_calc([measure], qrels, run_)}

            specs = [
                ("ndcg_at_10", irm.nDCG @ 10, lambda r, g: rm.ndcg_at_k(r, g, 10)),
                ("precision_at_5(rel>=2)", irm.P(rel=3) @ 5, lambda r, g: rm.precision_at_k(r, g, 5, 2.0)),
                ("recall_at_10(rel>=2)", irm.R(rel=3) @ 10, lambda r, g: rm.recall_at_k(r, g, 10, 2.0)),
            ]
            for name, measure, ours in specs:
                theirs = lib(measure)
                diffs, dropped = [], 0
                for q, gr, r in toy:
                    o = ours(r, gr)
                    if o is None:
                        dropped += 1
                        continue
                    diffs.append(abs(o - theirs.get(q, 0.0)))
                out.append(_xc(f"ranking_metrics.{name}", "ir_measures+pytrec_eval", "ok", max(diffs) if diffs else 0.0, 1e-9,
                               f"{len(diffs)} scenarios compared", {"dropped_zero_relevant_scenarios": dropped,
                                                                  "policy": "ours drops scenarios with no relevant item; trec_eval scores them 0 and keeps them"}))
            theirs_ap = lib(irm.AP(rel=3) @ 10)
            small, big, dropped = [], [], 0
            for q, gr, r in toy:
                o = rm.average_precision_at_k(r, gr, 10, 2.0)
                if o is None:
                    dropped += 1
                    continue
                R = rm.n_relevant(gr, 2.0)
                (small if R <= 10 else big).append(abs(o - theirs_ap.get(q, 0.0)))
            out.append(_xc("ranking_metrics.ap_at_10(R<=10 scenarios)", "ir_measures+pytrec_eval", "ok", max(small) if small else 0.0, 1e-9,
                           f"{len(small)} scenarios with R<=k compared",
                           {"convention": "ours divides by min(R,k); trec_eval/ranx divide by R",
                            "scenarios_with_R_gt_k": len(big), "max_abs_diff_R_gt_k_informational": max(big) if big else None,
                            "dropped_zero_relevant_scenarios": dropped}))
        except Exception as exc:  # noqa: BLE001 - adapter trouble is not a verdict on our code
            out.append(_xc("ranking_metrics.vs_ir_measures", "ir_measures+pytrec_eval", "skipped", note=f"adapter error: {type(exc).__name__}: {str(exc)[:120]}"))

    # ---- statistics vs scipy.stats
    sp, why_sp = _try_import("scipy.stats")
    if sp is None:
        for n in ("bootstrap_ci", "paired_bootstrap", "kendall_tau_b", "spearman_rho"):
            out.append(_xc(f"stats.{n}", "scipy.stats", "skipped", note=f"not importable ({why_sp})"))
    else:
        try:
            import numpy as np
            from skillllm_v2.evals.gap_evals import kendall_tau_b, spearman

            a = [rng.random() for _ in range(80)]
            b = [x * 0.9 + rng.random() * 0.2 for x in a]
            d = np.array(a) - np.array(b)
            pb = paired_bootstrap(a, b, n_boot, seed)
            res = sp.bootstrap((np.array(a), np.array(b)), lambda x, y, axis: np.mean(x - y, axis=axis), paired=True, method="percentile",
                               n_resamples=n_boot, random_state=seed)
            width = float(res.confidence_interval.high - res.confidence_interval.low)
            out.append(_xc("stats.paired_bootstrap.delta", "scipy.stats.bootstrap(paired)", "ok", abs(pb["delta"] - float(d.mean())), 1e-12, "mean difference exact"))
            ci_diff = max(abs(pb["ci_low"] - float(res.confidence_interval.low)), abs(pb["ci_high"] - float(res.confidence_interval.high)))
            out.append(_xc("stats.paired_bootstrap.ci_endpoints", "scipy.stats.bootstrap(paired)", "ok", ci_diff, 0.2 * width + 1e-3,
                           "percentile CI endpoints differ only by Monte-Carlo noise (tolerance = 20% of CI width)", {"ci_width": width}))
            lo, hi = bootstrap_ci(a, n_boot, seed)
            r1 = sp.bootstrap((np.array(a),), np.mean, method="percentile", n_resamples=n_boot, random_state=seed)
            w1 = float(r1.confidence_interval.high - r1.confidence_interval.low)
            out.append(_xc("stats.bootstrap_ci.mean", "scipy.stats.bootstrap", "ok", max(abs(lo - float(r1.confidence_interval.low)), abs(hi - float(r1.confidence_interval.high))),
                           0.2 * w1 + 1e-3, "percentile CI of the mean; Monte-Carlo tolerance", {"ci_width": w1}))
            xs = [rng.randint(0, 5) for _ in range(30)]
            ys = [rng.randint(0, 5) for _ in range(30)]
            out.append(_xc("stats.kendall_tau_b", "scipy.stats.kendalltau", "ok", abs(kendall_tau_b(xs, ys) - float(sp.kendalltau(xs, ys).statistic)), 1e-9, "tied integer data"))
            out.append(_xc("stats.spearman_rho", "scipy.stats.spearmanr", "ok", abs(spearman(xs, ys) - float(sp.spearmanr(xs, ys).statistic)), 1e-9, "tied integer data"))
        except Exception as exc:  # noqa: BLE001
            out.append(_xc("stats.vs_scipy", "scipy.stats", "skipped", note=f"adapter error: {type(exc).__name__}: {str(exc)[:120]}"))

    # ---- Holm vs statsmodels
    sm, why_sm = _try_import("statsmodels.stats.multitest")
    if sm is None:
        out.append(_xc("stats.holm_adjust", "statsmodels.multipletests", "skipped", note=f"not importable ({why_sm})"))
    else:
        try:
            pv = [round(rng.random() ** 2, 4) for _ in range(8)]
            theirs = sm.multipletests(pv, method="holm")[1]
            out.append(_xc("stats.holm_adjust", "statsmodels.multipletests", "ok", max(abs(x - float(y)) for x, y in zip(rm.holm_adjust(pv), theirs)), 1e-9, "8 p-values"))
        except Exception as exc:  # noqa: BLE001
            out.append(_xc("stats.holm_adjust", "statsmodels.multipletests", "skipped", note=f"adapter error: {type(exc).__name__}: {str(exc)[:120]}"))

    # ---- nervaluate (optional; extraction modes are verified in extraction_evals' own tests, no adapter here)
    nv, why_nv = _try_import("nervaluate")
    out.append(_xc("extraction.span_modes", "nervaluate", "skipped",
                   note=("importable but the suite has no adapter for it" if nv is not None else f"not importable ({why_nv})")))
    return out


# ----------------------------------------------------------------------------------------------- coverage
def coverage(specs: Sequence[EvalSpec], mods: Sequence[ModuleRun]) -> List[Dict[str, Any]]:
    by_name = {m.name: m for m in mods}
    rows = []
    for stage, desc in PIPELINE_STAGES:
        direct, partial, state = [], [], "not covered"
        for s in specs:
            m = by_name.get(s.name)
            live = bool(m and m.status == "ok" and any(r.status == "ok" for r in m.results))
            if stage in s.stages:
                direct.append(f"{s.name}{'' if live else ' (unavailable: ' + (m.status if m else 'not run') + ')'}")
                if live:
                    state = "covered"
                elif state != "covered":
                    state = "declared, module unavailable"
            elif stage in s.partial_stages:
                partial.append(f"{s.name}{'' if live else ' (unavailable)'}")
                if state == "not covered" and live:
                    state = "partial"
        rows.append({"stage": stage, "description": desc, "state": state, "evals": direct, "partial": partial})
    return rows


# ----------------------------------------------------------------------------------------------- report
def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return "-"
        return f"{v:.{nd}f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def _cell(s: Any, width: int = 110) -> str:
    t = str(s).replace("|", "/").replace("\n", " ")
    return t if len(t) <= width else t[: width - 1] + "..."


def render_report(run: Dict[str, Any]) -> str:
    meta = run["meta"]
    L: List[str] = []
    L.append("# SkillLLM_v2 evaluation suite report\n")
    verdict = "PASS" if run["exit_code"] == 0 else f"FAIL (exit code {run['exit_code']})"
    L.append(f"**Verdict: {verdict}**" + (" -- smoke run (tiny subsets; numbers are NOT comparable to a full run)" if meta["smoke"] else "") + "\n")
    L.append("## Run metadata\n")
    L.append("| key | value |\n|---|---|")
    for k in ("git_commit", "git_dirty", "seed", "smoke", "started_utc", "finished_utc", "duration_seconds", "python", "platform", "n_boot", "modules_requested"):
        L.append(f"| {k} | {_cell(meta.get(k))} |")
    L.append("")
    integ = run["integrity"]
    cnt = {k: sum(1 for c in integ if c["verdict"] == k) for k in ("pass", "fail", "skipped", "unverified")}
    reg = run["regression"]
    L.append("## Summary\n")
    xcs = run.get("crosscheck") or []
    L.append(f"- Oracle cross-checks: {sum(c['status'] == 'ok' for c in xcs)} agree, {sum(c['status'] == 'failed' for c in xcs)} DISAGREE, {sum(c['status'] == 'skipped' for c in xcs)} skipped (library not importable)")
    L.append(f"- Controls: {cnt['pass']} pass, {cnt['fail']} FAIL, {cnt['unverified']} reported without a verdict, {cnt['skipped']} skipped")
    if reg is None:
        L.append("- Regression gate: not run (" + ("baseline written by this run" if "baseline" in (run.get("baseline_action") or "") and "created" in (run.get("baseline_action") or "") or "updated" in (run.get("baseline_action") or "") else "no baseline") + ")")
    elif reg.get("skipped_gate"):
        L.append(f"- Regression gate: skipped ({'; '.join(reg['notes'])})")
    else:
        L.append(f"- Regression gate vs baseline: {reg['compared']} gated metrics compared, {len(reg['regressions'])} regressed, "
                 f"{reg['improvements']} improved, {reg['unchanged']} unchanged, {reg['new']} new, {len(reg['dropped'])} dropped")
    bad_mods = [m for m in run["modules"] if m["status"] in ("failed", "import_error", "no_run_function")]
    na = [m for m in run["modules"] if m["status"] == "not_available"]
    L.append(f"- Modules: {sum(1 for m in run['modules'] if m['status'] == 'ok')} ran, {len(bad_mods)} crashed/import errors, {len(na)} not available")
    if run.get("baseline_action"):
        L.append(f"- Baseline: {run['baseline_action']}")
    L.append("")
    L.append("## Per-module summary\n")
    L.append("| module | status | evals | ok | skipped | failed | controls (pass/fail/other) | seconds | note |\n|---|---|---|---|---|---|---|---|---|")
    for m in run["modules"]:
        rs = [r for r in run["results"] if r["spec"] == m["name"]]
        c = [x for x in integ if x["spec"] == m["name"]]
        L.append(f"| {m['name']} | {m['status']} | {len(rs)} | {sum(r['status'] == 'ok' for r in rs)} | {sum(r['status'] == 'skipped' for r in rs)} | "
                 f"{sum(r['status'] == 'failed' for r in rs)} | {sum(x['verdict'] == 'pass' for x in c)}/{sum(x['verdict'] == 'fail' for x in c)}/"
                 f"{sum(x['verdict'] in ('unverified', 'skipped') for x in c)} | {_fmt(m['seconds'], 1)} | {_cell(m['reason'], 80)} |")
    L.append("")
    L.append("## Coverage vs the pipeline stages\n")
    L.append("| stage | what | coverage | direct evals | indirect / partial |\n|---|---|---|---|---|")
    for c in run["coverage"]:
        L.append(f"| {c['stage']} | {c['description']} | **{c['state']}** | {', '.join(c['evals']) or '-'} | {', '.join(c['partial']) or '-'} |")
    unc = [c["stage"] for c in run["coverage"] if c["state"] == "not covered"]
    L.append("")
    L.append("Not covered by any eval in this suite: " + (", ".join(unc) if unc else "none") +
             ". (Course ranking quality on labelled scenarios lives in the older `src/skillllm_v2/eval/recommender_benchmark.py`, outside this suite; "
             "the e2e module only checks recommendation precision/traceability on a tiny synthetic set.)\n")
    L.append("## Integrity: controls and negative controls\n")
    L.append("A control is an experiment whose outcome is known in advance (oracle = perfect, shuffled labels = chance, ...). A failing control means the metric or harness cannot be trusted, and fails the run.\n")
    L.append("| module | control | measured | verdict | what must hold |\n|---|---|---|---|---|")
    for c in integ:
        mark = {"pass": "PASS", "fail": "**FAIL**", "skipped": "skipped", "unverified": "no verdict"}[c["verdict"]]
        L.append(f"| {c['spec']} | {_cell(c['eval'], 70)} | {_fmt(c['value'])} | {mark} | {_cell(c['expected'] or c['reason'], 120)} |")
    L.append("")
    L.append("## Oracle cross-check\n")
    xc = run.get("crosscheck") or []
    if not xc:
        L.append("Not run.\n")
    else:
        L.append("Our hand-written metrics/statistics recomputed through independent reference libraries when they are importable (never hard dependencies; missing ones are skipped). "
                 "A difference above tolerance fails the run. Convention differences (AP divisor, dropped zero-relevant scenarios) are recorded in the info column, not counted as failures.\n")
        L.append("| check | library | status | max abs diff | tolerance | note / info |\n|---|---|---|---|---|---|")
        for c in xc:
            info = _cell(c["note"] + (" | " + json.dumps(c["info"]) if c["info"] else ""), 170)
            diff = "-" if c["max_abs_diff"] is None else f"{c['max_abs_diff']:.2e}"
            tol = "-" if c["tolerance"] is None else f"{c['tolerance']:.1e}"
            L.append(f"| {c['name']} | {c['library']} | {c['status'].upper() if c['status'] == 'failed' else c['status']} | {diff} | {tol} | {info} |")
        L.append("")
    L.append("## Master table\n")
    L.append("Value is the point estimate; `[lo, hi]` is a 95% percentile-bootstrap interval over the evaluation units counted in `n` (blank when the eval does not give one). "
             "Full caveats and details are in `results.json`.\n")
    L.append("| module | eval | metric | value [95% CI] | n | dataset | status | caveats |\n|---|---|---|---|---|---|---|---|")
    for r in run["results"]:
        ci = f" [{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}]" if r.get("ci") else ""
        status = r["status"] + (f": {_cell(r['reason'], 60)}" if r["status"] != "ok" and r["reason"] else "")
        L.append(f"| {r['module']} | {_cell(r['eval_name'], 70)} | {_cell(r['metric'], 70)} | {_fmt(r['value'])}{ci} | {r['n']} | {_cell(r['dataset'], 45)} | {status} | "
                 f"{_cell('; '.join(r['caveats']), 100)} |")
    L.append("")
    L.append("## Regression gate\n")
    if reg is None:
        L.append("No baseline was available. Run with `--update-baseline` (full run, not smoke) to record one.\n")
    else:
        for n in reg["notes"]:
            L.append(f"- note: {n}")
        if reg["regressions"]:
            L.append("| metric | baseline | current | delta | reason |\n|---|---|---|---|---|")
            for x in reg["regressions"]:
                L.append(f"| {_cell(x['key'], 100)} | {_fmt(x['baseline'])} | {_fmt(x['current'])} | {_fmt(x['delta'])} | {_cell(x['reason'], 80)} |")
        else:
            L.append("No gated metric regressed past tolerance." if not reg.get("skipped_gate") else "")
        if reg["dropped"]:
            L.append(f"\nMetrics in the baseline that did not produce a value this run ({len(reg['dropped'])}): " + ", ".join(_cell(k, 60) for k in reg["dropped"][:12]) + (" ..." if len(reg["dropped"]) > 12 else ""))
        L.append("")
    L.append("## Failures and skips\n")
    problems = [(m["name"], m["status"], m["reason"]) for m in run["modules"] if m["status"] != "ok"]
    problems += [(r["eval_name"], r["status"], r["reason"]) for r in run["results"] if r["status"] in ("failed", "skipped")]
    if problems:
        L.append("| item | status | reason |\n|---|---|---|")
        for a, b, c in problems:
            L.append(f"| {_cell(a, 70)} | {b} | {_cell(c, 160)} |")
    else:
        L.append("None.")
    L.append("")
    L.append("## How to read this\n")
    L.append("Most datasets are synthetic or small; intervals describe sampling noise over the evaluation units only, not label error or distribution shift. "
             "See `docs/EVAL_SUITE.md` for metric definitions and limitations.\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------- orchestration
def assemble(ctx: EvalContext, specs: Sequence[EvalSpec], mods: Sequence[ModuleRun], started: _dt.datetime, finished: _dt.datetime,
             only: Optional[Sequence[str]]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    flat: List[Tuple[str, EvalResult]] = []
    for m in mods:
        for r in m.results:
            flat.append((m.name, r))
    by_name = {r.eval_name: r for _, r in flat}
    integrity = []
    for spec_name, r in flat:
        c = is_control(r)
        row = {**_clean(r.to_dict()), "spec": spec_name, "is_control": c}
        results.append(row)
        if c:
            integrity.append({**control_verdict(r, by_name), "spec": spec_name})
    git = git_info(ctx.root)
    meta = {
        "git_commit": git["commit"], "git_dirty": git["dirty"], "seed": ctx.seed, "smoke": ctx.smoke, "n_boot": ctx.n_boot,
        "started_utc": started.isoformat(timespec="seconds"), "finished_utc": finished.isoformat(timespec="seconds"),
        "duration_seconds": round((finished - started).total_seconds(), 1), "python": sys.version.split()[0], "platform": platform.platform(),
        "modules_requested": ",".join(only) if only else "all",
    }
    meta["git"] = {"commit": git["commit"]}
    return {"meta": meta,
            "modules": [{"name": m.name, "module_path": m.module_path, "status": m.status, "reason": m.reason, "seconds": round(m.seconds, 2),
                         "n_results": len(m.results), "traceback": m.traceback} for m in mods],
            "results": results, "integrity": integrity, "crosscheck": [], "coverage": coverage(specs, mods), "regression": None, "exit_code": 0}


def decide_exit(run: Dict[str, Any]) -> int:
    control_fail = any(c["verdict"] == "fail" for c in run["integrity"]) or any(c["status"] == "failed" for c in run.get("crosscheck") or [])
    regressed = bool(run["regression"] and run["regression"]["regressions"])
    if control_fail or regressed:
        return 1
    if any(m["status"] in ("failed", "import_error", "no_run_function") for m in run["modules"]):
        return 2
    return 0


def main(argv: Optional[Sequence[str]] = None, registry: Optional[Sequence[EvalSpec]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m skillllm_v2.evals.run_all", description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="", help="comma-separated module names (registry short names or *_evals stems)")
    ap.add_argument("--smoke", action="store_true", help="tiny subsets; writes to <out>/smoke by default and never touches the baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples (default 2000; 300 in smoke)")
    ap.add_argument("--out", default=None, help=f"output directory (default {DEFAULT_OUT})")
    ap.add_argument("--root", default=None, help="repository root (default: cwd if it holds src/skillllm_v2)")
    ap.add_argument("--baseline", default=None, help="baseline file (default <default out>/baseline.json)")
    ap.add_argument("--update-baseline", action="store_true", help="overwrite the baseline with this (full) run")
    ap.add_argument("--no-gate", action="store_true", help="skip the baseline regression gate")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="absolute tolerance for gated metrics")
    ap.add_argument("--no-crosscheck", action="store_true", help="skip the oracle cross-check against ir_measures / scipy / statsmodels")
    ap.add_argument("--list", action="store_true", help="list registered modules and availability, then exit")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else default_root()
    reg = list(registry) if registry is not None else REGISTRY
    if a.list:
        for s in reg:
            fn, st, why = load(s)
            print(f"{s.name:12s} {s.module_path:42s} {st}  {why.splitlines()[0] if why else ''}")
        return 0
    only = [x for x in a.only.split(",") if x.strip()]
    try:
        specs = select(only, reg)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if a.update_baseline and a.smoke:
        print("error: refusing to record a baseline from a --smoke run", file=sys.stderr)
        return 3
    if a.update_baseline and only:
        print("error: refusing to record a baseline from a partial run (--only); run the full suite", file=sys.stderr)
        return 3

    default_out = root / DEFAULT_OUT
    out_dir = Path(a.out) if a.out else (default_out / "smoke" if a.smoke else default_out)
    baseline_path = Path(a.baseline) if a.baseline else default_out / "baseline.json"
    ctx = EvalContext(root=root, seed=a.seed, n_boot=a.n_boot if a.n_boot else (300 if a.smoke else 2000), smoke=a.smoke)

    started = _dt.datetime.now(_dt.timezone.utc)
    mods = run_suite(ctx, specs)
    finished = _dt.datetime.now(_dt.timezone.utc)
    run = assemble(ctx, specs, mods, started, finished, only)

    all_results = [r for m in mods for r in m.results]
    if not a.no_crosscheck:
        try:
            run["crosscheck"] = _clean(oracle_crosscheck(a.seed, ctx.n_boot))
        except Exception as exc:  # noqa: BLE001
            run["crosscheck"] = [{"name": "oracle_crosscheck", "library": "-", "status": "skipped", "max_abs_diff": None, "tolerance": None,
                                  "note": f"cross-check crashed: {type(exc).__name__}: {exc}", "info": {}}]
    baseline_action = ""
    if not a.no_gate:
        if baseline_path.exists() and not a.update_baseline:
            try:
                base = json.loads(baseline_path.read_text(encoding="utf-8"))
                run["regression"] = compare_baseline(all_results, base, a.tolerance, a.smoke, a.seed,
                                                     [s.name for s in specs] if only else None)
                if only:
                    run["regression"]["notes"].append(f"partial run (--only {','.join(only)}): only those modules were compared")
            except (OSError, json.JSONDecodeError) as exc:
                baseline_action = f"baseline unreadable ({exc}); gate skipped"
        elif a.update_baseline or (not baseline_path.exists() and not a.smoke and not only):
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            payload = baseline_from([(m.name, r) for m in mods for r in m.results], run["meta"])
            baseline_path.write_text(json.dumps(_clean(payload), indent=1, sort_keys=True), encoding="utf-8")
            baseline_action = ("baseline updated" if a.update_baseline else "no baseline existed: created from this first full run") + f" ({len(payload['metrics'])} gated metrics) -> {baseline_path}"
        elif not baseline_path.exists():
            baseline_action = "no baseline file; regression gate not run"
    run["baseline_action"] = baseline_action
    run["exit_code"] = decide_exit(run)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(_clean(run), indent=1), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(_clean(run)), encoding="utf-8")
    cnt_fail = sum(1 for c in run["integrity"] if c["verdict"] == "fail")
    print(f"eval suite: {sum(1 for m in mods if m.status == 'ok')}/{len(mods)} modules ran, {len(run['results'])} results, "
          f"{cnt_fail} control failure(s), {len(run['regression']['regressions']) if run['regression'] else 0} regression(s) -> {out_dir}  [exit {run['exit_code']}]")
    for m in mods:
        if m.status not in ("ok",):
            print(f"  - {m.name}: {m.status} {m.reason.splitlines()[0] if m.reason else ''}")
    for c in run["crosscheck"]:
        if c["status"] == "failed":
            print(f"  - CROSS-CHECK DISAGREES: {c['name']} vs {c['library']}: diff {c['max_abs_diff']} > {c['tolerance']}")
    for c in run["integrity"]:
        if c["verdict"] == "fail":
            print(f"  - CONTROL FAILED: {c['eval']} ({c['reason'] or c['expected']})")
    if run["regression"]:
        for x in run["regression"]["regressions"][:20]:
            print(f"  - REGRESSION: {x['key']}: {x['baseline']} -> {x['current']} ({x['reason']})")
    return run["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
