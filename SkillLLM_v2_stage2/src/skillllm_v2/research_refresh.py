from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skillllm_v2.context.cached_context_builder import build_cached_context_packet
from skillllm_v2.context.profile_summarizer import summarize_profile
from skillllm_v2.eval.money_table_generator import generate_money_table
from skillllm_v2.eval.prediction_provenance_check import check_provenance
from skillllm_v2.io_utils import read_json, read_jsonl, write_json, write_jsonl, write_text

REPORT_DIR = Path("reports/skillllm_v2/research_refresh")
ARTIFACT_DIR = Path("artifacts/skillllm_v2/research_refresh")
THESIS_DIR = Path("docs/thesis")

METRIC_FILES = {
    "BERT baseline": Path("artifacts/skillllm_v2/bert_baseline/metrics.json"),
    "GLiNER": Path("artifacts/skillllm_v2/gliner_baseline/metrics.json"),
    "Raw Mistral LoRA": Path("artifacts/skillllm_v2/mistral_lora/metrics.json"),
    "Old Boundary Refiner": Path("artifacts/skillllm_v2/mistral_boundary/metrics.json"),
    "Old Mistral + GLiNER + Taxonomy": Path("artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json"),
    "Old Full CAG/Graph": Path("artifacts/skillllm_v2/full_cag_graph/metrics.json"),
    "SkillLLM_v2 Safe Wrapper": Path("artifacts/skillllm_v2/safe_wrapper/metrics.json"),
}

INSPECT_PATHS = [
    "artifacts/skillllm_v2/mistral_lora/metrics.json",
    "artifacts/skillllm_v2/mistral_boundary/metrics.json",
    "artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json",
    "artifacts/skillllm_v2/full_cag_graph/metrics.json",
    "reports/skillllm_v2/money_table.md",
    "reports/skillllm_v2/thesis_results_summary.md",
    "reports/skillllm_v2/metric_sanity_report.md",
    "reports/skillllm_v2/prediction_provenance_check.md",
]

RESEARCH_SOURCES = [
    {
        "key": "herandi2024skillllm",
        "title": "Skill-LLM: Repurposing General-Purpose LLMs for Skill Extraction",
        "authors": "Amirhossein Herandi, Yitao Li, Zhanlin Liu, Ximin Hu, Xiao Cai",
        "year": "2024",
        "url": "https://arxiv.org/abs/2410.12052",
        "contribution": "Shows that supervised LLM adaptation can improve skill extraction quality.",
        "application": "Supports using a fine-tuned Mistral LoRA as the main extraction anchor.",
        "implementation_effect": "Current: keep the LLM extractor as the anchor and evaluate wrappers separately.",
    },
    {
        "key": "zhang2022skillspan",
        "title": "SkillSpan: Hard and Soft Skill Extraction from English Job Postings",
        "authors": "Mike Zhang, Kristian Nørgaard Jensen, Sif Dam Sonniks, Barbara Plank",
        "year": "2022",
        "url": "https://aclanthology.org/2022.naacl-main.366/",
        "contribution": "Introduces expert span annotations for hard and soft skills in job postings.",
        "application": "Defines the benchmark, labels, and exact boundary pressure used here.",
        "implementation_effect": "Current: use test split for extraction metrics; preserve hard/soft labels.",
    },
    {
        "key": "zaratiana2024gliner",
        "title": "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
        "authors": "Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois",
        "year": "2024",
        "url": "https://aclanthology.org/2024.naacl-long.300/",
        "contribution": "Generalist NER with arbitrary labels using a compact bidirectional encoder.",
        "application": "Useful as a recall booster and baseline, not a replacement for the tuned LLM.",
        "implementation_effect": "Current: add-only auxiliary candidates behind non-regression gates.",
    },
    {
        "key": "dettmers2023qlora",
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": "Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer",
        "year": "2023",
        "url": "https://arxiv.org/abs/2305.14314",
        "contribution": "Memory-efficient LoRA fine-tuning through frozen 4-bit quantized backbones.",
        "application": "Justifies local Mistral adapter training on constrained GPUs.",
        "implementation_effect": "Current: reuse trained adapter; do not retrain during refresh.",
    },
    {
        "key": "edge2024graphrag",
        "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "authors": "Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson",
        "year": "2024",
        "url": "https://arxiv.org/abs/2404.16130",
        "contribution": "Uses graph-based structure for corpus-level reasoning beyond naive chunk retrieval.",
        "application": "Supports downstream graph reasoning after extraction and normalization.",
        "implementation_effect": "Current: keep graph reasoning out of the extraction gate.",
    },
    {
        "key": "anthropic2024contextualretrieval",
        "title": "Contextual Retrieval in AI Systems",
        "authors": "Anthropic",
        "year": "2024",
        "url": "https://www.anthropic.com/engineering/contextual-retrieval",
        "contribution": "Adds document context to chunks to improve retrieval reliability.",
        "application": "Supports contextualized job/course retrieval with evidence.",
        "implementation_effect": "Current: add deterministic contextual retrieval helper; dense/reranker later.",
    },
    {
        "key": "huang2025cag",
        "title": "Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks",
        "authors": "Chan H. Huang et al.",
        "year": "2025",
        "url": "https://dl.acm.org/doi/10.1145/3701716.3715490",
        "contribution": "Frames cache-augmented generation as useful when stable knowledge fits in context.",
        "application": "Use CAG for stable profile/taxonomy/course summaries, not for changing job feeds.",
        "implementation_effect": "Current: add cached context packet; keep job postings in RAG.",
        "note": "Author metadata should be verified before final submission.",
    },
    {
        "key": "dvcDocs",
        "title": "Versioning Data and Models",
        "authors": "DVC Documentation",
        "year": "2026",
        "url": "https://doc.dvc.org/example-scenarios/versioning-data-and-models",
        "contribution": "Documents Git-compatible practices for data/model artifact versioning.",
        "application": "Motivates artifact manifests, stale checks, and metrics provenance.",
        "implementation_effect": "Current: refresh manifest, provenance report, and artifact path tables.",
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return {"cmd": args, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"cmd": args, "returncode": -1, "stdout": "", "stderr": str(exc)}


def generate_run_manifest() -> dict[str, Any]:
    branch = run_command(["git", "branch", "--show-current"])["stdout"] or None
    commit_cmd = run_command(["git", "rev-parse", "HEAD"])
    commit = commit_cmd["stdout"] if commit_cmd["returncode"] == 0 else None
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}

    manifest = {
        "timestamp": now_iso(),
        "branch": branch,
        "commit_hash": commit,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda": torch_info,
        "gpu_info": run_command(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"])["stdout"],
        "active_config_files": sorted(str(path) for path in Path("configs").glob("*.yaml")),
        "artifact_paths_inspected": INSPECT_PATHS,
    }
    write_json(REPORT_DIR / "run_manifest.json", manifest)
    return manifest


def collect_codebase_audit() -> dict[str, Any]:
    modules = sorted(str(path) for path in Path("src/skillllm_v2").rglob("*.py") if "__pycache__" not in str(path))
    scripts = sorted(str(path) for path in Path("scripts").glob("*.sh"))
    configs = sorted(str(path) for path in Path("configs").glob("*.yaml"))
    docs = sorted(str(path) for path in Path("docs").glob("*.md"))
    reports = sorted(str(path) for path in Path("reports/skillllm_v2").glob("*"))
    status_files = sorted(Path("artifacts/skillllm_v2/status").glob("*.json"))
    stages = []
    for path in status_files:
        payload = read_json(path, {})
        stages.append(
            {
                "stage": payload.get("stage") or path.stem.replace(".success", "").replace(".failed", ""),
                "status": payload.get("status") or ("success" if path.name.endswith(".success.json") else "failed"),
                "metrics": payload.get("metrics", {}),
                "path": str(path),
            }
        )

    metrics = {}
    for name, path in METRIC_FILES.items():
        metrics[name] = {"path": str(path), "exists": path.exists(), "metrics": read_json(path, {}) if path.exists() else {}}

    raw = metrics["Raw Mistral LoRA"]["metrics"]
    wrapper_deltas = {}
    for name in ["Old Boundary Refiner", "Old Mistral + GLiNER + Taxonomy", "Old Full CAG/Graph", "SkillLLM_v2 Safe Wrapper"]:
        wrapper_metrics = metrics.get(name, {}).get("metrics", {})
        candidate = wrapper_metrics.get("candidate_metrics", wrapper_metrics)
        selected = wrapper_metrics.get("selected_metrics", candidate)
        if raw and candidate:
            wrapper_deltas[name] = {
                "candidate_exact_f1_delta": _num(candidate.get("exact_span_f1")) - _num(raw.get("exact_span_f1")),
                "selected_exact_f1_delta": _num(selected.get("exact_span_f1")) - _num(raw.get("exact_span_f1")),
                "candidate_soft_f1_delta": _num(candidate.get("soft_skill_f1")) - _num(raw.get("soft_skill_f1")),
                "selected_soft_f1_delta": _num(selected.get("soft_skill_f1")) - _num(raw.get("soft_skill_f1")),
            }

    production_ready = [stage["stage"] for stage in stages if stage["status"] == "success" and not stage["stage"].endswith("_smoke")]
    smoke_only = [stage["stage"] for stage in stages if stage["stage"].endswith("_smoke")]
    missing_requested = [path for path in INSPECT_PATHS if not Path(path).exists()]

    return {
        "generated_at": now_iso(),
        "git": {
            "branch": run_command(["git", "branch", "--show-current"])["stdout"],
            "status_short": run_command(["git", "status", "--short", "--branch"])["stdout"],
            "recent_commits_available": run_command(["git", "log", "--oneline", "-n", "1"])["returncode"] == 0,
        },
        "module_count": len(modules),
        "modules": modules,
        "scripts": scripts,
        "configs": configs,
        "docs": docs,
        "reports": reports,
        "stages": stages,
        "production_ready_stages": production_ready,
        "smoke_only_stages": smoke_only,
        "stale_or_missing": {
            "missing_requested_inspection_files": missing_requested,
            "provenance_issues": check_provenance(),
        },
        "metrics": metrics,
        "wrapper_deltas_vs_raw_mistral": wrapper_deltas,
        "current_best_thesis_claim": (
            "SkillLLM_v2 should use the fine-tuned Mistral LoRA as the extraction anchor; "
            "wrapper components are valuable for normalization, provenance, CAG context, and graph explanations "
            "only when gated so they cannot degrade extraction."
        ),
    }


def write_codebase_audit(audit: dict[str, Any]) -> None:
    write_json(REPORT_DIR / "codebase_audit.json", audit)
    lines = [
        "# SkillLLM_v2 Codebase Audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "## Repository State",
        "",
        f"- Branch: `{audit['git']['branch']}`",
        f"- Recent commit available: `{audit['git']['recent_commits_available']}`",
        "- Git status:",
        "```",
        audit["git"]["status_short"],
        "```",
        "",
        "## Modules and Stages",
        "",
        f"- Python modules under `src/skillllm_v2`: `{audit['module_count']}`",
        f"- Scripts: `{len(audit['scripts'])}`",
        f"- Config files: `{len(audit['configs'])}`",
        f"- Stage status files: `{len(audit['stages'])}`",
        f"- Production-ready success stages: `{len(audit['production_ready_stages'])}`",
        f"- Smoke-only stages: `{len(audit['smoke_only_stages'])}`",
        "",
        "## Real Metrics Inspected",
        "",
        "| System | Exact F1 | Token F1 | Partial F1 | HARD F1 | SOFT F1 | Artifact |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for system, payload in audit["metrics"].items():
        metrics = _display_metrics(payload["metrics"])
        lines.append(
            f"| {system} | {_fmt(metrics.get('exact_span_f1'))} | {_fmt(metrics.get('token_f1'))} | "
            f"{_fmt(metrics.get('partial_span_f1'))} | {_fmt(metrics.get('hard_skill_f1'))} | "
            f"{_fmt(metrics.get('soft_skill_f1'))} | `{payload['path']}` |"
        )
    lines += [
        "",
        "## Wrapper Diagnosis",
        "",
        "| Wrapper | Candidate Exact Delta | Selected Exact Delta | Candidate Soft Delta | Selected Soft Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, delta in audit["wrapper_deltas_vs_raw_mistral"].items():
        lines.append(
            f"| {name} | {_fmt(delta.get('candidate_exact_f1_delta'))} | {_fmt(delta.get('selected_exact_f1_delta'))} | "
            f"{_fmt(delta.get('candidate_soft_f1_delta'))} | {_fmt(delta.get('selected_soft_f1_delta'))} |"
        )
    lines += [
        "",
        "## Answers",
        "",
        "1. Modules exist under `src/skillllm_v2` for baselines, extraction, verification, normalization, graph, matching, training, evaluation, and pipeline orchestration.",
        "2. Pipeline stages are represented by JSON status markers under `artifacts/skillllm_v2/status`.",
        "3. Stages with success markers and full metrics are production-ready for reporting, subject to artifact provenance.",
        "4. Stages ending in `_smoke` are smoke-only gates.",
        "5. Staleness is determined by provenance checks against Raw Mistral prediction timestamps.",
        "6. Full-test metrics are only claimed when they come from persisted `metrics.json` artifacts.",
        "7. Smoke/limited metrics must remain labeled by their own status/config artifacts.",
        "8. Old boundary and full wrapper variants hurt extraction because exact and soft-skill F1 drop below Raw Mistral.",
        "9. Wrapper modules help when they preserve anchor spans and attach normalization/provenance metadata.",
        f"10. Current best thesis claim: {audit['current_best_thesis_claim']}",
    ]
    write_text(REPORT_DIR / "codebase_audit.md", "\n".join(lines) + "\n")


def write_research_audit() -> None:
    write_json(REPORT_DIR / "research_audit.json", {"sources": RESEARCH_SOURCES})
    lines = [
        "# Research Audit",
        "",
        "| Source | Year | Contribution | SkillLLM_v2 application | Implementation effect |",
        "|---|---:|---|---|---|",
    ]
    for source in RESEARCH_SOURCES:
        lines.append(
            f"| [{source['title']}]({source['url']}) | {source['year']} | {source['contribution']} | "
            f"{source['application']} | {source['implementation_effect']} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        "The literature supports a two-part design: a supervised/fine-tuned extractor for boundary-sensitive SkillSpan extraction, then a guarded wrapper for normalization, context, and graph reasoning. GLiNER is useful as a lightweight recall source, but the local metrics show it must be add-only and gated. GraphRAG, Contextual Retrieval, and CAG support downstream job/course reasoning but do not justify changing extraction spans after the Raw Mistral anchor has already produced stronger SkillSpan metrics.",
    ]
    write_text(REPORT_DIR / "research_audit.md", "\n".join(lines) + "\n")
    bib = _bibtex()
    write_text(THESIS_DIR / "references.bib", bib)
    write_text(THESIS_DIR / "skillllm_v2_research_refresh.bib", bib)


def write_metric_sanity_and_provenance(audit: dict[str, Any]) -> None:
    issues = audit["stale_or_missing"]["provenance_issues"]
    lines = [
        "# Metric Sanity Report",
        "",
        "Metrics below are read from local artifacts only. Missing values are not imputed.",
        "",
        "| System | Exact F1 | Token F1 | SOFT F1 | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for system, payload in audit["metrics"].items():
        metrics = _display_metrics(payload["metrics"])
        status = "present" if payload["exists"] else "missing"
        lines.append(f"| {system} | {_fmt(metrics.get('exact_span_f1'))} | {_fmt(metrics.get('token_f1'))} | {_fmt(metrics.get('soft_skill_f1'))} | {status} |")
    write_text("reports/skillllm_v2/metric_sanity_report.md", "\n".join(lines) + "\n")

    provenance = [
        "# Prediction Provenance Check",
        "",
        "Upstream anchor: `artifacts/skillllm_v2/mistral_lora/predictions.jsonl`",
        "",
    ]
    if issues:
        provenance.append("## Issues")
        provenance.extend(f"- {issue}" for issue in issues)
    else:
        provenance.append("No stale or missing downstream artifacts were detected by `prediction_provenance_check`.")
    write_text("reports/skillllm_v2/prediction_provenance_check.md", "\n".join(provenance) + "\n")


def write_context_outputs() -> None:
    profile_path = Path("inputs/demo/user_profile.json")
    profile_payload = read_json(profile_path, {}) if profile_path.exists() else {}
    raw_profile = json.dumps(profile_payload, ensure_ascii=False) if profile_payload else "Demo user profile unavailable."
    predictions = read_jsonl("artifacts/skillllm_v2/mistral_lora/predictions.jsonl")
    extracted = predictions[0].get("predictions", [])[:10] if predictions else []
    summary = summarize_profile(raw_profile, extracted_skills=extracted, user_id=str(profile_payload.get("user_id", "demo_user")))
    packet = build_cached_context_packet(raw_profile, extracted, user_id=summary["user_id"])
    write_json("artifacts/skillllm_v2/context/user_profile_summary.json", summary)
    write_json("artifacts/skillllm_v2/context/cached_context_packet.json", packet)
    md_lines = [
        "# User Profile Summary",
        "",
        f"- User ID: `{summary['user_id']}`",
        f"- Skill facts: `{len(summary['skills'])}`",
        "",
        "| Skill | Label | Confidence | Evidence count |",
        "|---|---|---:|---:|",
    ]
    for skill in summary["skills"]:
        md_lines.append(f"| {skill['surface_form']} | {skill['label']} | {_fmt(skill['confidence'])} | {len(skill['evidence'])} |")
    write_text("artifacts/skillllm_v2/context/user_profile_summary.md", "\n".join(md_lines) + "\n")


def generate_latex(audit: dict[str, Any]) -> None:
    money_rows = _read_money_rows()
    table_lines = ["System & Exact F1 & Token F1 & Partial F1 & HARD F1 & SOFT F1 \\\\", "\\hline"]
    for system, payload in audit["metrics"].items():
        metrics = _display_metrics(payload["metrics"])
        table_lines.append(
            f"{_tex(system)} & {_fmt(metrics.get('exact_span_f1'))} & {_fmt(metrics.get('token_f1'))} & "
            f"{_fmt(metrics.get('partial_span_f1'))} & {_fmt(metrics.get('hard_skill_f1'))} & "
            f"{_fmt(metrics.get('soft_skill_f1'))} \\\\"
        )

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{tikz}}
\title{{SkillLLM\_v2 Research Refresh: Non-Regressive Skill Extraction, Context Compaction, and Graph-Augmented Recommendation}}
\author{{Generated from local SkillLLM\_v2 artifacts}}
\date{{{audit['generated_at']}}}
\begin{{document}}
\maketitle

\section{{Executive Summary}}
This refresh inspected the current Stage-2 codebase and artifacts without retraining expensive models. The verified local artifacts show that Raw Mistral LoRA remains the extraction anchor. Old wrapper variants improve downstream structure only if guarded, because their candidate extraction metrics regress against the anchor.

\section{{Four-Week Update Context}}
The copied Stage-2 repository currently has no Git commits in this worktree, so repository changes were audited from source files, status markers, artifact timestamps, and metric files rather than a commit diff.

\section{{Codebase Audit}}
The audit found {audit['module_count']} Python modules under \texttt{{src/skillllm\_v2}}, {len(audit['scripts'])} shell scripts, and {len(audit['stages'])} status marker files. Full details are in \texttt{{reports/skillllm\_v2/research\_refresh/codebase\_audit.md}}.

\section{{Research Grounding}}
Skill-LLM supports supervised LLM adaptation for skill extraction~\cite{{herandi2024skillllm}}. SkillSpan defines the boundary-sensitive hard/soft skill benchmark~\cite{{zhang2022skillspan}}. GLiNER supports lightweight open-label NER~\cite{{zaratiana2024gliner}}, while QLoRA supports efficient local adapter training~\cite{{dettmers2023qlora}}. GraphRAG, Contextual Retrieval, and CAG motivate downstream context and graph reasoning rather than destructive span rewriting~\cite{{edge2024graphrag,anthropic2024contextualretrieval,huang2025cag}}.

\section{{SkillLLM v1 Architecture Lessons}}
The main lesson carried forward is to keep extraction, normalization, and recommendation as separable stages. Extraction quality must be measured against span boundaries; normalization and graph reasoning should attach evidence and explanations without silently changing the extraction target.

\section{{Current SkillLLM\_v2 Architecture}}
The current architecture includes BERT and GLiNER baselines, a Raw Mistral LoRA extractor, boundary refinement, candidate merging, taxonomy linking, verifier/calibration components, graph matching, and course recommendation. The safe wrapper now treats Raw Mistral as the anchor and gates wrapper outputs against it.

\section{{Current Metrics and Problem Diagnosis}}
\begin{{longtable}}{{lrrrrr}}
\toprule
{chr(10).join(table_lines)}
\bottomrule
\end{{longtable}}

\section{{Wrapper Regression Analysis}}
Old wrapper candidates underperformed Raw Mistral on exact span F1 and soft-skill F1. Therefore, wrapper components must be source-aware and non-destructive.

\section{{Safe Wrapper Design}}
Raw Mistral spans are anchor predictions. Boundary refinement proposes candidate edits but does not destructively replace anchor spans. GLiNER and dictionary candidates are add-only. Taxonomy linking attaches canonical metadata but does not filter extraction. The non-regression gate selects wrapper output only when exact and soft-skill guardrails pass; otherwise it falls back to Raw Mistral extraction.

\section{{Context Compaction and User Profile Summarization}}
The new context path builds evidence-backed user profile summaries and cached context packets. CAG is used for stable profile summaries, taxonomy aliases, and course catalogue summaries. Dynamic job postings remain retrieval targets.

\section{{CAG/RAG/GraphRAG Design}}
CAG caches stable small context, RAG retrieves large and changing job/course text, and graph retrieval explains relational paths from user skills to missing skills, jobs, and courses.

\section{{Experiments and Ablations}}
The current money table is generated from local metrics artifacts. Rows are not invented; missing downstream labels remain NA.

\section{{Results}}
See \texttt{{reports/skillllm\_v2/money\_table.md}} and \texttt{{reports/skillllm\_v2/research\_refresh/codebase\_audit.md}}.

\section{{Error Analysis}}
See \texttt{{reports/skillllm\_v2/research\_refresh/wrapper\_error\_analysis.md}} and \texttt{{wrapper\_regression\_cases.jsonl}}.

\section{{Limitations}}
Graph matching and course recommendation labels are currently unavailable in the inspected artifacts, so nDCG@10 and P@5 remain NA. CAG context helpers are deterministic smoke-ready modules; they do not claim downstream metric gains without labels.

\section{{Next Steps}}
\begin{{enumerate}}
\item Keep Raw Mistral as the extraction anchor.
\item Tune wrapper candidate thresholds on dev data only.
\item Add labeled job/course recommendation evaluation before claiming recommendation gains.
\item Add DVC or equivalent artifact hashing for large model and prediction files.
\end{{enumerate}}

\section{{Reproducibility Checklist}}
Run manifest: \texttt{{reports/skillllm\_v2/research\_refresh/run\_manifest.json}}. Metrics provenance: \texttt{{reports/skillllm\_v2/prediction\_provenance\_check.md}}. Artifact paths are included in the audit and money table.

\section{{Appendix: Commands and Artifacts}}
\begin{{verbatim}}
PYTHONPATH=src pytest tests/skillllm_v2/ -q
bash scripts/run_skillllm_v2_safe_wrapper.sh
python -m skillllm_v2.research_refresh
\end{{verbatim}}

\bibliographystyle{{plain}}
\bibliography{{skillllm_v2_research_refresh}}
\end{{document}}
"""
    write_text(THESIS_DIR / "skillllm_v2_research_refresh.tex", tex.strip() + "\n")


def run_refresh(run_safe: bool = True, compile_latex: bool = True) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    (THESIS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (THESIS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    manifest = generate_run_manifest()
    actions: dict[str, Any] = {"manifest": manifest, "safe_wrapper": None, "wrapper_error_analysis": None, "latex_compile": None}

    try:
        from skillllm_v2.eval.wrapper_error_analysis import run_analysis

        actions["wrapper_error_analysis"] = run_analysis(output_dir=str(REPORT_DIR))
    except Exception as exc:
        actions["wrapper_error_analysis"] = {"error": str(exc)}

    if run_safe:
        try:
            from skillllm_v2.eval.run_safe_wrapper import run_safe_wrapper

            actions["safe_wrapper"] = run_safe_wrapper(use_gliner=False)
        except Exception as exc:
            actions["safe_wrapper"] = {"error": str(exc)}

    try:
        generate_money_table("reports/skillllm_v2")
    except Exception as exc:
        actions["money_table_error"] = str(exc)

    audit = collect_codebase_audit()
    write_codebase_audit(audit)
    write_research_audit()
    write_metric_sanity_and_provenance(audit)
    write_context_outputs()
    generate_latex(audit)

    if compile_latex:
        compiler = None
        if run_command(["which", "xelatex"])["returncode"] == 0:
            compiler = "xelatex"
        elif run_command(["which", "pdflatex"])["returncode"] == 0:
            compiler = "pdflatex"
        if compiler is None:
            actions["latex_compile"] = {"available": False, "error": "No pdflatex or xelatex executable found on PATH."}
        else:
            tex_path = THESIS_DIR / "skillllm_v2_research_refresh.tex"
            first = run_command([compiler, "-interaction=nonstopmode", str(tex_path)])
            second = run_command([compiler, "-interaction=nonstopmode", str(tex_path)])
            actions["latex_compile"] = {"available": True, "compiler": compiler, "first": first, "second": second}
    write_json(REPORT_DIR / "refresh_actions.json", actions)
    return actions


def _read_money_rows() -> list[dict[str, str]]:
    path = Path("reports/skillllm_v2/money_table.csv")
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _display_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if "selected_metrics" in metrics:
        return metrics["selected_metrics"]
    full = (metrics.get("variants") or {}).get("full_cag_graph")
    anchor = metrics.get("anchor_metrics") or {}
    if full:
        view = dict(anchor)
        view["exact_span_f1"] = full.get("selected_exact_span_f1")
        view["soft_skill_f1"] = full.get("selected_soft_skill_f1")
        return view
    return metrics


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def _tex(value: str) -> str:
    return value.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _bibtex() -> str:
    return """@article{herandi2024skillllm,
  title={Skill-LLM: Repurposing General-Purpose LLMs for Skill Extraction},
  author={Herandi, Amirhossein and Li, Yitao and Liu, Zhanlin and Hu, Ximin and Cai, Xiao},
  year={2024},
  journal={arXiv preprint arXiv:2410.12052},
  url={https://arxiv.org/abs/2410.12052}
}

@inproceedings{zhang2022skillspan,
  title={SkillSpan: Hard and Soft Skill Extraction from English Job Postings},
  author={Zhang, Mike and Jensen, Kristian Nørgaard and Sonniks, Sif Dam and Plank, Barbara},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={4962--4984},
  year={2022},
  publisher={Association for Computational Linguistics},
  url={https://aclanthology.org/2022.naacl-main.366/}
}

@inproceedings{zaratiana2024gliner,
  title={GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer},
  author={Zaratiana, Urchade and Tomeh, Nadi and Holat, Pierre and Charnois, Thierry},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2024},
  url={https://aclanthology.org/2024.naacl-long.300/}
}

@inproceedings{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
  url={https://arxiv.org/abs/2305.14314}
}

@article{edge2024graphrag,
  title={From Local to Global: A Graph RAG Approach to Query-Focused Summarization},
  author={Edge, Darren and Trinh, Ha and Cheng, Newman and Bradley, Joshua and Chao, Alex and Mody, Apurva and Truitt, Steven and Larson, Jonathan},
  journal={arXiv preprint arXiv:2404.16130},
  year={2024},
  url={https://arxiv.org/abs/2404.16130}
}

@misc{anthropic2024contextualretrieval,
  title={Contextual Retrieval in AI Systems},
  author={{Anthropic}},
  year={2024},
  url={https://www.anthropic.com/engineering/contextual-retrieval},
  note={Engineering blog}
}

@inproceedings{huang2025cag,
  title={Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks},
  author={Huang, Chan H. and others},
  booktitle={Companion Proceedings of the ACM Web Conference},
  year={2025},
  url={https://dl.acm.org/doi/10.1145/3701716.3715490},
  note={Author metadata should be verified before final thesis submission}
}

@misc{dvcDocs,
  title={Versioning Data and Models},
  author={{DVC Documentation}},
  year={2026},
  url={https://doc.dvc.org/example-scenarios/versioning-data-and-models},
  note={Documentation accessed for reproducible artifact tracking practices}
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the SkillLLM_v2 research refresh audit, safe-wrapper reports, context artifacts, and LaTeX documentation.")
    parser.add_argument("--skip-safe-wrapper", action="store_true")
    parser.add_argument("--no-compile-latex", action="store_true")
    args = parser.parse_args()
    actions = run_refresh(run_safe=not args.skip_safe_wrapper, compile_latex=not args.no_compile_latex)
    fatal_keys = {"safe_wrapper", "wrapper_error_analysis", "money_table_error"}
    if any(key in fatal_keys and isinstance(value, dict) and value.get("error") for key, value in actions.items()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
