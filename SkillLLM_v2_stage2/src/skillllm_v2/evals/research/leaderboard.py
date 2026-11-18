"""Leaderboard over all research layers + LaTeX tables + auto-filled results blocks of the evaluation chapter.

    PYTHONPATH=src python -m skillllm_v2.evals.research.leaderboard [--chapter] [--smoke]

Reads every ``reports/skillllm_v2/evals/<layer>/results.json`` (missing layers are reported as ``missing``, never
guessed) and writes
  * ``reports/skillllm_v2/evals/LEADERBOARD.md`` and ``leaderboard.json``: per-layer status, headline metrics
    (records with ``details.headline``), H1-H7 verdicts (records with ``metric == "H#_verdict"``), integrity flags;
  * ``docs/thesis/tables/evaluation_{layer_status,headline,hypotheses}.tex`` (booktabs, same style as the thesis tables);
  * with ``--chapter``: the blocks between ``<!-- BEGIN AUTO:<name> -->`` and ``<!-- END AUTO:<name> -->`` in
    ``docs/thesis/evaluation/chapter_evaluation.md`` (prose outside the markers is never touched).
Values are copied from the records, never recomputed. Verdicts are the owning layer's, never re-labelled here.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from skillllm_v2.evals.research.common import EVALS_ROOT, REPO_ROOT, git_info, md_table

LAYERS = [
    ("L01_extraction", "Span extraction (SkillSpan test)", "A1"),
    ("L02_generalization", "Unseen-skill generalization", "A1"),
    ("L03_robustness", "Robustness to perturbations", "A3"),
    ("L04_calibration", "Calibration and selective prediction", "A2"),
    ("L05_wrapper", "Safe-wrapper contribution", "A2"),
    ("L06_linking", "Taxonomy linking / normalization", "A4"),
    ("L07_compaction", "Profile compaction", "A4"),
    ("L08_graph_retrieval", "Graph-augmented retrieval", "A5"),
    ("L09_course_rec", "Course recommendation", "A5"),
    ("L10_significance", "Central significance testing", "A2"),
    ("L11_human_eval", "Human evaluation", "A6"),
    ("L12_systems", "Systems: latency, memory", "A3"),
]
HYPOTHESES = {
    "H1": ("L01_extraction", "Anchor LoRA beats BERT/GLiNER/fair JobBERT by >= 0.10 exact F1, Holm p < 0.05 vs every baseline"),
    "H2": ("L05_wrapper", "Safe wrapper: APR higher than unconstrained post-processing (CI excl. 0) and exact F1 non-inferior (margin -0.005)"),
    "H3": ("L02_generalization", "Anchor seen - unseen exact F1 >= 0.05, CI excludes 0"),
    "H4": ("L04_calibration", "Selective precision at 80%/50% coverage > 100% (CI excl. 0) and AURC better than constant confidence; calibrator fit on dev"),
    "H5": ("L08_graph_retrieval", "Hybrid+graph > hybrid nDCG@10 on multi-gap held-out queries, Holm p < 0.05"),
    "H6": ("L09_course_rec", "Prerequisite-aware ranker: lower violation rate (CI excl. 0), nDCG@10 non-inferior (margin -0.02)"),
    "H7": ("L07_compaction", "Compact profile: >= 50% token reduction, course/job nDCG@10 non-inferior (margin -0.02), gap Jaccard >= 0.9"),
}
TABLES_DIR = REPO_ROOT / "docs" / "thesis" / "tables"
CHAPTER = REPO_ROOT / "docs" / "thesis" / "evaluation" / "chapter_evaluation.md"
PROGRESS = REPO_ROOT / "Progress.md"
MAX_LAYER_ROWS = 30


# ------------------------------------------------------------------------------------------------------------ loading
def load_all(root: Path = EVALS_ROOT, smoke: bool = False) -> Dict[str, Optional[Dict[str, Any]]]:
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for layer, _, _ in LAYERS:
        p = root / layer / ("smoke" if smoke else "") / "results.json"
        if not p.exists():
            out[layer] = None
            continue
        try:
            out[layer] = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            out[layer] = {"meta": {"error": f"unreadable results.json: {e}"}, "records": []}
    return out


def layer_status(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if data is None:
        return {"status": "missing", "counts": {}, "n_records": 0, "utc": "", "smoke": None}
    recs = data.get("records", [])
    c = Counter(r.get("status", "?") for r in recs)
    meta = data.get("meta", {})
    if meta.get("error"):
        st = "failed"
    elif not recs:
        st = "empty"
    elif c.get("measured", 0) == len(recs):
        st = "measured"
    elif c.get("measured", 0) == 0:
        pend = [s for s in ("pending_gpu", "pending_human", "not_available", "failed") if c.get(s)]
        st = pend[0] if len(pend) == 1 else "pending (" + ", ".join(pend) + ")"
    else:
        others = [s for s in ("pending_gpu", "pending_human", "not_available", "failed") if c.get(s)]
        st = "measured + " + "/".join(others)
    return {"status": st, "counts": dict(c), "n_records": len(recs), "utc": meta.get("utc", ""), "smoke": meta.get("smoke"),
            "git": (meta.get("git") or {}).get("commit", "")[:12], "argv": meta.get("argv")}


def fmt_num(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    return f"{v:.{digits}f}"


def fmt_ci(r: Dict[str, Any], digits: int = 4) -> str:
    if r.get("ci_low") is None or r.get("ci_high") is None:
        return "-"
    return f"[{fmt_num(r['ci_low'], digits)}, {fmt_num(r['ci_high'], digits)}]"


def _prov(r: Dict[str, Any]) -> str:
    p = (r.get("provenance") or [])
    if not p:
        return "-"
    first = p[0]
    sha = (first.get("sha256") or "")[:10]
    return f"{first.get('path')}" + (f"@{sha}" if sha else "") + (f" (+{len(p) - 1})" if len(p) > 1 else "")


def headline_rows(all_data: Dict[str, Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for layer, _, _ in LAYERS:
        d = all_data.get(layer)
        for r in (d or {}).get("records", []):
            if (r.get("details") or {}).get("headline"):
                rows.append({"layer": layer, "system": r["system"], "metric": r["metric"], "value": r.get("value"),
                             "ci": fmt_ci(r), "n": r.get("n_units"), "unit": r.get("unit"), "split": r.get("split"),
                             "direction": r.get("direction"), "status": r.get("status"), "reason": r.get("reason", ""),
                             "provenance": _prov(r)})
    return rows


def verdict_rows(all_data: Dict[str, Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    found: Dict[str, List[Dict[str, Any]]] = {h: [] for h in HYPOTHESES}
    for layer, _, _ in LAYERS:
        for r in (all_data.get(layer) or {}).get("records", []):
            m = re.fullmatch(r"(H[1-7])_verdict", r.get("metric", ""))
            if m:
                found[m.group(1)].append({**r, "_layer": layer})
    rows = []
    for h, (owner, rule) in HYPOTHESES.items():
        if not found[h]:
            rows.append({"hypothesis": h, "owner_layer": owner, "verdict": "NOT YET REPORTED", "delta": None, "ci": "-",
                         "p_raw": None, "p_holm": None, "family": "", "status": "missing",
                         "reason": f"no {h}_verdict record in {owner}/results.json yet", "rule": rule, "system": "",
                         "caveat": ""})
            continue
        for r in found[h]:
            d = r.get("details") or {}
            verdict = d.get("verdict") or ("NOT TESTABLE" if r.get("status") != "measured" else "?")
            rows.append({"hypothesis": h, "owner_layer": r["_layer"], "verdict": verdict, "delta": r.get("value"), "ci": fmt_ci(r),
                         "p_raw": d.get("p_raw"), "p_holm": d.get("p_holm"), "family": d.get("family", ""),
                         "status": r.get("status"), "reason": r.get("reason", ""), "rule": d.get("rule") or rule,
                         "system": r.get("system", ""), "caveat": "; ".join(r.get("caveats") or [])[:220]})
    return rows


def integrity_flags(path: Optional[Path] = None) -> List[str]:
    path = path or PROGRESS
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if "| integrity-flag |" in l]


def build(smoke: bool = False, root: Path = EVALS_ROOT) -> Dict[str, Any]:
    data = load_all(root, smoke)
    status = []
    for layer, title, owner in LAYERS:
        s = layer_status(data.get(layer))
        status.append({"layer": layer, "title": title, "owner": owner, **s})
    return {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_info(), "smoke": smoke,
            "layers": status, "headline": headline_rows(data), "hypotheses": verdict_rows(data),
            "integrity_flags": integrity_flags(), "data": data}


# ---------------------------------------------------------------------------------------------------------- markdown
def _status_table(lb: Dict[str, Any]) -> str:
    rows = [{"layer": s["layer"], "title": s["title"], "owner": s["owner"], "status": s["status"], "records": s["n_records"],
             "measured": s["counts"].get("measured", 0), "pending_gpu": s["counts"].get("pending_gpu", 0),
             "pending_human": s["counts"].get("pending_human", 0), "not_available": s["counts"].get("not_available", 0),
             "generated_utc": s["utc"]} for s in lb["layers"]]
    return md_table(rows, ["layer", "title", "owner", "status", "records", "measured", "pending_gpu", "pending_human",
                           "not_available", "generated_utc"])


def _headline_table(lb: Dict[str, Any]) -> str:
    if not lb["headline"]:
        return "_No layer has emitted a headline record yet._"
    rows = [{**r, "value": fmt_num(r["value"]) if r["status"] == "measured" else f"({r['status']})"} for r in lb["headline"]]
    return md_table(rows, ["layer", "system", "metric", "value", "ci", "n", "unit", "split", "provenance"])


def _verdict_table(lb: Dict[str, Any]) -> str:
    rows = [{**r, "delta": fmt_num(r["delta"]), "p_raw": fmt_num(r["p_raw"]) if isinstance(r["p_raw"], (int, float)) else "-",
             "p_holm": fmt_num(r["p_holm"]) if isinstance(r["p_holm"], (int, float)) else "-",
             "note": " ".join(x for x in (r["reason"] or r["family"], r.get("caveat", "")) if x)} for r in lb["hypotheses"]]
    return md_table(rows, ["hypothesis", "owner_layer", "verdict", "delta", "ci", "p_raw", "p_holm", "status", "note"])


def _verdict_statements(lb: Dict[str, Any]) -> str:
    """One bullet per H#_verdict record, in the owning layer's own words (verdict, effect, CI, p, rule, caveat)."""
    out = []
    for r in lb["hypotheses"]:
        eff = (f"primary effect = {fmt_num(r['delta'])} {r['ci']}" if r["delta"] is not None else "no effect reported")
        ps = []
        if isinstance(r["p_raw"], (int, float)):
            ps.append(f"p_raw = {fmt_num(r['p_raw'])}")
        if isinstance(r["p_holm"], (int, float)):
            ps.append(f"p_Holm = {fmt_num(r['p_holm'])}")
        arm = f" [{r['system']}]" if r.get("system") else ""
        extra = f" Reason: {r['reason']}" if r.get("reason") else ""
        cav = f" Caveat recorded by the layer: {r['caveat']}" if r.get("caveat") else ""
        out.append(f"- **{r['hypothesis']}: {r['verdict']}**{arm} (layer {r['owner_layer']}, {r['status']}). {eff}"
                   + (f"; {', '.join(ps)}" if ps else "") + f". Pre-registered rule: {str(r['rule']).rstrip('. ')}.{extra}{cav}")
    return "\n".join(out)


def _layer_table(layer: str, lb: Dict[str, Any]) -> str:
    d = lb["data"].get(layer)
    if d is None:
        return f"_`{layer}/results.json` does not exist yet: no result is reported for this layer._"
    recs = d.get("records", [])
    measured = [r for r in recs if r.get("status") == "measured"]
    pending = Counter(r["status"] for r in recs if r.get("status") != "measured")
    key = sorted(measured, key=lambda r: (not (r.get("details") or {}).get("headline"), not r["metric"].endswith("_verdict")))
    rows = [{"system": r["system"], "metric": r["metric"], "value": fmt_num(r["value"]), "ci": fmt_ci(r), "n": r["n_units"],
             "unit": r["unit"], "split": r["split"]} for r in key[:MAX_LAYER_ROWS]]
    out = [md_table(rows, ["system", "metric", "value", "ci", "n", "unit", "split"]) if rows else "_No measured record._"]
    if len(measured) > MAX_LAYER_ROWS:
        out.append(f"\n_{len(measured) - MAX_LAYER_ROWS} further measured records: see `reports/skillllm_v2/evals/{layer}/REPORT.md`._")
    if pending:
        reasons = Counter((r["status"], r.get("reason", "")[:140]) for r in recs if r.get("status") != "measured")
        out.append("\nNot measured: " + ", ".join(f"{k} {v}" for k, v in pending.items()) + ". Reasons:")
        out += [f"- `{s}` x{n}: {why}" for (s, why), n in reasons.most_common(8)]
    meta = d.get("meta", {})
    out.append(f"\nSource: `reports/skillllm_v2/evals/{layer}/results.json` (generated {meta.get('utc', '?')}, git "
               f"{(meta.get('git') or {}).get('commit', '')[:12]}{' dirty' if (meta.get('git') or {}).get('dirty') else ''}).")
    return "\n".join(out)


def render_md(lb: Dict[str, Any]) -> str:
    g = lb["git"]
    lines = [
        "# SkillLLM_v2 research evaluation — leaderboard", "",
        f"Generated {lb['generated_utc']} from `reports/skillllm_v2/evals/*/results.json` (git {g.get('commit', '')[:12]}"
        f"{' dirty' if g.get('dirty') else ''}). Values are copied from layer records, never recomputed here. "
        "`-` = no value. A missing layer means its results.json does not exist yet; nothing is estimated in its place.",
        "" if not lb["smoke"] else "\n> SMOKE LEADERBOARD — built from `<layer>/smoke/` outputs; not results.\n",
        "## Layer status", "", _status_table(lb), "",
        "## Headline metrics", "", "Records flagged `details.headline` by their owning layer (95% CI; n = independent units).", "",
        _headline_table(lb), "",
        "## Pre-registered hypotheses H1–H7", "",
        "Rules: `docs/thesis/evaluation/hypotheses_preregistration.md`. Verdicts are emitted by the owning layer.", "",
        _verdict_table(lb), "",
        "## Integrity flags raised in Progress.md", "",
    ]
    lines += [f"- {f}" for f in lb["integrity_flags"]] or ["_None._"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------- LaTeX
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "≥": r"$\geq$", "≤": r"$\leq$", "Δ": r"$\Delta$", "−": "-",
        "–": "--", "—": "---"}


def tex(s: Any) -> str:
    return "".join(_TEX.get(ch, ch) for ch in str(s))


def _tabular(caption: str, label: str, spec: str, header: Sequence[str], rows: List[Sequence[Any]], note: str = "") -> str:
    lines = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    if rows:
        lines += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
    else:
        lines.append(rf"\multicolumn{{{len(header)}}}{{c}}{{No records yet.}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\par\smallskip\footnotesize {tex(note)}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_tex(lb: Dict[str, Any]) -> Dict[str, str]:
    gen = f"Generated {lb['generated_utc']} by skillllm_v2.evals.research.leaderboard; values copied from layer records."
    status_rows = [(s["layer"].replace("_", " "), s["title"], s["status"], s["counts"].get("measured", 0),
                    s["counts"].get("pending_gpu", 0) + s["counts"].get("pending_human", 0), s["counts"].get("not_available", 0))
                   for s in lb["layers"]]
    head_rows = [(r["layer"].split("_")[0], r["system"], r["metric"],
                  fmt_num(r["value"]) if r["status"] == "measured" else r["status"], r["ci"], r["n"]) for r in lb["headline"]]
    hyp_rows = [(r["hypothesis"], r["verdict"], fmt_num(r["delta"]), r["ci"],
                 fmt_num(r["p_holm"]) if isinstance(r["p_holm"], (int, float)) else "-", r["status"]) for r in lb["hypotheses"]]
    smoke = " SMOKE RUN, not results." if lb["smoke"] else ""
    return {
        "evaluation_layer_status.tex": _tabular(
            "Status of the twelve evaluation layers. Pending counts are records that await GPU inference or human annotation."
            + smoke, "tab:evaluation-layer-status", "llp{0.16\\textwidth}rrr",
            ["Layer", "Scope", "Status", "Measured", "Pending", "N/A"], status_rows, gen),
        "evaluation_headline.tex": _tabular(
            "Headline evaluation metrics with 95\\% bootstrap confidence intervals over independent units." + smoke,
            "tab:evaluation-headline", "lllrlr", ["Layer", "System", "Metric", "Value", "95\\% CI", "n"], head_rows, gen),
        "evaluation_hypotheses.tex": _tabular(
            "Verdicts for the pre-registered hypotheses H1--H7 under their fixed decision rules." + smoke,
            "tab:evaluation-hypotheses", "llrlrl", ["Hyp.", "Verdict", "$\\Delta$", "95\\% CI", "Holm $p$", "Status"], hyp_rows, gen),
    }


# ---------------------------------------------------------------------------------------------------------- chapter
AUTO_RE = re.compile(r"(<!-- BEGIN AUTO:(?P<name>[\w:.-]+) -->)(?P<body>.*?)(<!-- END AUTO:(?P=name) -->)", re.S)


def chapter_blocks(lb: Dict[str, Any]) -> Dict[str, str]:
    stamp = f"_Auto-generated {lb['generated_utc']} from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._"
    blocks = {"layer_status": _status_table(lb), "headline": _headline_table(lb), "hypotheses": _verdict_table(lb),
              "verdict_statements": _verdict_statements(lb)}
    for layer, _, _ in LAYERS:
        blocks[f"layer:{layer}"] = _layer_table(layer, lb)
    return {k: f"\n{stamp}\n\n{v}\n" for k, v in blocks.items()}


def fill_chapter(text: str, blocks: Dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        name = m.group("name")
        return m.group(1) + blocks.get(name, m.group("body")) + m.group(4)
    return AUTO_RE.sub(sub, text)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chapter", action="store_true", help="refresh the AUTO blocks of chapter_evaluation.md")
    ap.add_argument("--smoke", action="store_true", help="aggregate <layer>/smoke/ outputs; writes to evals/smoke/")
    ap.add_argument("--no-tex", action="store_true")
    a = ap.parse_args(argv)
    lb = build(a.smoke)
    out = EVALS_ROOT / ("smoke" if a.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "LEADERBOARD.md").write_text(render_md(lb))
    (out / "leaderboard.json").write_text(json.dumps({k: v for k, v in lb.items() if k != "data"}, indent=2, default=str) + "\n")
    if not a.no_tex:
        tdir = TABLES_DIR if not a.smoke else out / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        for name, body in render_tex(lb).items():
            (tdir / name).write_text(body)
    if a.chapter and not a.smoke:
        if CHAPTER.exists():
            CHAPTER.write_text(fill_chapter(CHAPTER.read_text(), chapter_blocks(lb)))
        else:
            print(f"warning: {CHAPTER} not found; chapter not refreshed")
    st = Counter(s["status"].split(" ")[0] for s in lb["layers"])
    print(f"leaderboard: layers {dict(st)}; headline rows {len(lb['headline'])}; verdicts "
          + ", ".join(f"{r['hypothesis']}={r['verdict']}" for r in lb["hypotheses"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
