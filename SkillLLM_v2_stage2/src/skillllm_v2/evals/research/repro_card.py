"""Evaluation card (ReproEval-style) for the layered research evaluation: what exactly was run, on what, with what.

    PYTHONPATH=src python -m skillllm_v2.evals.research.repro_card [--hash-base-weights] [--no-freeze] [--smoke]

Writes ``reports/skillllm_v2/evals/repro/{EVAL_CARD.md, eval_card.json}`` plus ``repro/prompts/*.txt`` (the extraction prompt
templates with sha256) and ``repro/env_*.txt`` (pip freeze of the CPU and GPU interpreters). Sections: code version (git
commit / branch / dirty files of the outer repository), datasets (sha256 per file), external-dataset manifest, base model
and adapters (config + weight hashes; the 48 GB base weights only with --hash-base-weights), Hugging Face cache revisions,
all ``predictions.jsonl`` hashes, taxonomy, graph and retrieval inputs, prompts, seeds, software environments,
CUDA/driver/torch, hardware, the exact command of every layer (from ``results.json`` meta.argv) and an index of every
intermediate output. Safe to re-run at any time; nothing outside ``repro/`` is written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from skillllm_v2.evals.research.common import EVALS_ROOT, GIT_ROOT, REPO_ROOT, md_table, rel, sha256_file, sha256_tree

OUT = EVALS_ROOT / "repro"
CPU_PY = "/opt/miniconda/bin/python"
GPU_PY = str(GIT_ROOT / "SkillLLM_v2" / ".venv" / "bin" / "python")
BASE_MODEL_DIR = REPO_ROOT / "models" / "mistral-small-3.1-24b-instruct-2503"
BASE_MODEL_HUB_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
HF_CACHES = [Path.home() / ".cache" / "huggingface" / "hub"] + sorted(Path("/usr/data/models/skillllm-v2-stage2").glob("hf_cache_*/hub"))
HF_RELEVANT = re.compile(r"mistral|gliner|jobbert|bert|minilm|mpnet|e5|bge|esco|roberta|deberta|skill|escoxlm", re.I)
LAYER_IDS = ["L01_extraction", "L02_generalization", "L03_robustness", "L04_calibration", "L05_wrapper", "L06_linking",
             "L07_compaction", "L08_graph_retrieval", "L09_course_rec", "L10_significance", "L11_human_eval", "L12_systems"]
SMALL_MODEL_FILES = ("config.json", "params.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
                     "tekken.json", "special_tokens_map.json", "chat_template.json", "model.safetensors.index.json",
                     "preprocessor_config.json", "processor_config.json", "SYSTEM_PROMPT.txt")


def _run(cmd: List[str], timeout: int = 120) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001 - recorded, not fatal
        return f"<unavailable: {type(e).__name__}>"


def _files(root: Path, pattern: str = "*") -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    out = []
    for f in sorted(q for q in root.rglob(pattern) if q.is_file()):
        out.append({"path": rel(f), "bytes": f.stat().st_size, "sha256": sha256_file(f)})
    return out


# ------------------------------------------------------------------------------------------------------------ sections
def section_git() -> Dict[str, Any]:
    g = lambda *a: _run(["git", "-C", str(GIT_ROOT), *a])  # noqa: E731
    porcelain = g("status", "--porcelain", "--", str(REPO_ROOT)).splitlines()
    return {"repository": str(GIT_ROOT), "commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "commit_date": g("log", "-1", "--format=%cI"), "dirty": bool(porcelain), "n_dirty_entries": len(porcelain),
            "dirty_entries_sample": porcelain[:40],
            "note": "Stage-2 is tracked by the outer repository; its nested .git is empty and is never used."}


def section_datasets() -> Dict[str, Any]:
    out = {"skillspan_hf": _files(REPO_ROOT / "data/source/skillspan_hf"),
           "skillllm_v2_derived": _files(REPO_ROOT / "data/skillllm_v2"),
           "course_benchmark_inputs": _files(REPO_ROOT / "inputs/courses"),
           "eval_inputs": _files(REPO_ROOT / "inputs/evals", "*.json*"),
           "dataset_info": _files(REPO_ROOT / "data", "dataset_info.json")}
    man = REPO_ROOT / "data/external/MANIFEST.jsonl"
    entries = []
    if man.exists():
        for line in man.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries.append({k: e.get(k) for k in ("name", "url", "version_or_commit", "license", "downloaded_utc", "downloaded_by",
                                                      "purpose")} | {"n_files": len(e.get("files") or []),
                                                                     "files": e.get("files") or []})
    out["external_manifest"] = {"path": rel(man), "sha256": sha256_file(man) if man.exists() else None, "entries": entries}
    return out


def section_models(hash_base: bool) -> Dict[str, Any]:
    base = {"hub_id_assumed": BASE_MODEL_HUB_ID, "local_dir": str(BASE_MODEL_DIR), "resolved": str(BASE_MODEL_DIR.resolve()),
            "revision": "not recorded locally (plain snapshot directory, no HF cache metadata)",
            "files": [{"path": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)}
                      for f in sorted(BASE_MODEL_DIR.iterdir()) if f.is_file() and f.name in SMALL_MODEL_FILES]
            if BASE_MODEL_DIR.exists() else [],
            "weights": []}
    if BASE_MODEL_DIR.exists():
        for f in sorted(BASE_MODEL_DIR.glob("*.safetensors")):
            base["weights"].append({"path": f.name, "bytes": f.stat().st_size,
                                    "sha256": sha256_file(f) if hash_base else "not hashed (run with --hash-base-weights)"})
    adapters = []
    for cfg in sorted((REPO_ROOT / "artifacts/skillllm_v2").glob("*/*/adapter_config.json")):
        d = cfg.parent
        entry = {"dir": rel(d), "files": [], "checkpoints": sorted(p.name for p in d.glob("checkpoint-*"))}
        try:
            entry["base_model_name_or_path"] = json.loads(cfg.read_text()).get("base_model_name_or_path")
        except json.JSONDecodeError:
            pass
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix in (".safetensors", ".json", ".yaml", ".yml", ".jinja"):
                entry["files"].append({"path": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)})
        adapters.append(entry)
    other = []
    for d in sorted((REPO_ROOT / "artifacts/skillllm_v2").glob("*/model")):
        if not (d / "adapter_config.json").exists():
            other.append({"dir": rel(d), "files": [{"path": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)}
                                                   for f in sorted(d.iterdir()) if f.is_file() and f.stat().st_size < 50_000_000]})
    return {"base_model": base, "adapters": adapters, "other_model_dirs": other, "hf_cache": section_hf_cache()}


def section_hf_cache() -> List[Dict[str, Any]]:
    out = []
    for hub in HF_CACHES:
        if not hub.exists():
            continue
        for repo in sorted(hub.glob("models--*")) + sorted(hub.glob("datasets--*")):
            rid = repo.name.split("--", 1)[1].replace("--", "/")
            if not HF_RELEVANT.search(rid) and "skillllm-v2-stage2" not in str(hub):
                continue
            snaps = sorted(p.name for p in (repo / "snapshots").glob("*")) if (repo / "snapshots").exists() else []
            refs = {}
            for r in (repo / "refs").glob("*") if (repo / "refs").exists() else []:
                try:
                    refs[r.name] = r.read_text().strip()
                except OSError:
                    pass
            out.append({"cache": str(hub), "repo": rid, "kind": repo.name.split("--")[0], "snapshots": snaps, "refs": refs})
    return out


def section_predictions() -> List[Dict[str, Any]]:
    roots = [REPO_ROOT / "artifacts/skillllm_v2"]
    out = []
    for root in roots:
        for f in sorted(root.rglob("predictions*.jsonl")):
            try:
                n = sum(1 for _ in open(f, encoding="utf-8"))
            except OSError:
                n = None
            out.append({"path": rel(f), "rows": n, "bytes": f.stat().st_size, "sha256": sha256_file(f),
                        "mtime_utc": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")})
    return out


def section_taxonomy_graph_index() -> Dict[str, Any]:
    tax = REPO_ROOT / "data/skillllm_v2/taxonomy/skill_aliases.jsonl"
    tax_info: Dict[str, Any] = {"path": rel(tax)}
    if tax.exists():
        rows = [json.loads(l) for l in tax.read_text(encoding="utf-8").splitlines() if l.strip()]
        tax_info.update({"sha256": sha256_file(tax), "rows": len(rows),
                         "concepts": len({r.get("canonical_skill_id") for r in rows}),
                         "taxonomies": sorted({str(r.get("taxonomy")) for r in rows}),
                         "alias_sources": dict(sorted(_count(r.get("source") for r in rows).items()))})
    ext = [p for p in (REPO_ROOT / "data/external").glob("*") if p.is_dir() and re.search("esco|onet|lightcast|taxonomy", p.name, re.I)]
    graph_inputs = [REPO_ROOT / p for p in ("inputs/courses/course_catalog_large.jsonl", "inputs/courses/course_catalog.jsonl",
                                            "data/skillllm_v2/taxonomy/skill_aliases.jsonl", "artifacts/skillllm_v2/smoke/graph.json")]
    return {
        "taxonomy": tax_info,
        "external_taxonomies": [{"dir": rel(p), "n_files": sum(1 for _ in p.rglob("*") if _.is_file())} for p in ext],
        "graph_snapshot": {
            "note": "The course/skill graph is rebuilt in memory from the catalogue + taxonomy on every run (no persisted "
                    "production snapshot); its identity is the hash of these inputs plus the code commit.",
            "inputs": [{"path": rel(p), "sha256": sha256_file(p)} for p in graph_inputs if p.exists()]},
        "retrieval_index": {
            "note": "BM25 (k1=1.5, b=0.75) and the dense index (sentence-transformers/all-MiniLM-L6-v2, CPU) are rebuilt "
                    "from the catalogue text at run time; no index file is persisted. Identity = catalogue hash + model "
                    "revision in the HF cache section.",
            "inputs": [{"path": rel(p), "sha256": sha256_file(p)} for p in graph_inputs[:1] if p.exists()]},
    }


def _count(it) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in it:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def section_prompts(out_dir: Path) -> List[Dict[str, Any]]:
    pdir = out_dir / "prompts"
    pdir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        from skillllm_v2.extract import mistral_extractor as mx
        for version, (system, user) in mx.PROMPT_VERSIONS.items():
            for part, text in (("system", system), ("user_template", user)):
                f = pdir / f"extraction_{version}_{part}.txt"
                f.write_text(text, encoding="utf-8")
                rows.append({"name": f"extraction/{version}/{part}", "path": rel(f), "sha256": sha256_file(f),
                             "default": version == mx.DEFAULT_PROMPT_VERSION,
                             "source": "src/skillllm_v2/extract/mistral_extractor.py::PROMPT_VERSIONS"})
    except Exception as e:  # noqa: BLE001
        rows.append({"name": "extraction", "error": f"{type(e).__name__}: {e}"})
    for src, name in ((BASE_MODEL_DIR / "chat_template.json", "base_model_chat_template.json"),
                      (BASE_MODEL_DIR / "SYSTEM_PROMPT.txt", "base_model_default_SYSTEM_PROMPT.txt"),
                      (REPO_ROOT / "artifacts/skillllm_v2/mistral_lora/model/chat_template.jinja", "anchor_adapter_chat_template.jinja")):
        if src.exists():
            dst = pdir / name
            shutil.copyfile(src, dst)
            rows.append({"name": name, "path": rel(dst), "sha256": sha256_file(dst), "source": rel(src)})
    return rows


def section_seeds(layers: Dict[str, Any]) -> Dict[str, Any]:
    seeds = {l: (d or {}).get("meta", {}).get("seed") for l, d in layers.items() if d}
    train = {}
    for cfg in sorted((REPO_ROOT / "artifacts/skillllm_v2").glob("*/*/*train_config*.y*ml")):
        m = re.search(r"^seed:\s*(\d+)", cfg.read_text(encoding="utf-8"), re.M)
        train[rel(cfg)] = int(m.group(1)) if m else None
    return {"layer_seeds": seeds, "training_config_seeds": train, "bootstrap_convention": "seed 42, 2,000 resamples (10,000 headline)",
            "recommender_benchmark_seed": 20260101, "recommender_split_salt": "skillllm-recsys-loop-v1"}


def section_env(out_dir: Path, freeze: bool) -> Dict[str, Any]:
    envs = {}
    for name, py in (("cpu_miniconda", CPU_PY), ("gpu_venv", GPU_PY)):
        info: Dict[str, Any] = {"python": py, "exists": Path(py).exists()}
        if Path(py).exists():
            info["version"] = _run([py, "-c", "import sys; print(sys.version.split()[0])"])
            if freeze:
                f = out_dir / f"env_{name}.txt"
                f.write_text(_run([py, "-m", "pip", "freeze", "--all"], timeout=300) + "\n")
                info["pip_freeze"] = rel(f)
                info["pip_freeze_sha256"] = sha256_file(f)
            info["key_packages"] = _run([py, "-c", (
                "import importlib.metadata as m\n"
                "for p in ['numpy','scipy','scikit-learn','torch','transformers','peft','bitsandbytes','accelerate',"
                "'sentence-transformers','gliner','networkx','matplotlib','pytest']:\n"
                "    try: print(p, m.version(p))\n"
                "    except Exception: pass")])
        envs[name] = info
    torch_info = _run([GPU_PY, "-c", "import torch;print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), "
                                     "torch.cuda.is_available())"], timeout=180) if Path(GPU_PY).exists() else ""
    envs["torch_cuda"] = torch_info
    envs["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,pci.bus_id", "--format=csv,noheader"])
    envs["nvcc"] = _run(["bash", "-lc", "nvcc --version 2>/dev/null | tail -1"])
    return envs


def section_hardware() -> Dict[str, Any]:
    cpu = _run(["bash", "-c", "lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Socket'"])
    mem = ""
    try:
        mem = [l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")][0]
    except Exception:  # noqa: BLE001
        pass
    return {"host": platform.node(), "platform": platform.platform(), "cpu": cpu, "memory": mem,
            "gpus": _run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"]),
            "disk_usr_data": _run(["bash", "-c", "df -h /usr/data | tail -1"])}


def _module_command(argv: Optional[List[str]]) -> str:
    if not argv:
        return ""
    first = str(argv[0])
    m = re.search(r"src/(skillllm_v2/.+)\.py$", first)
    if m:
        mod = m.group(1).replace("/", ".")
        return "PYTHONPATH=src " + CPU_PY + " -m " + mod + (" " + " ".join(map(str, argv[1:])) if len(argv) > 1 else "")
    return " ".join(map(str, argv))


def load_layers(smoke: bool = False) -> Dict[str, Optional[Dict[str, Any]]]:
    out = {}
    for l in LAYER_IDS:
        p = EVALS_ROOT / l / ("smoke" if smoke else "") / "results.json"
        try:
            out[l] = json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            out[l] = None
    return out


def section_commands(layers: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for l, d in layers.items():
        if not d:
            rows.append({"layer": l, "command": "", "utc": "", "git": "", "status": "missing"})
            continue
        meta = d.get("meta", {})
        rows.append({"layer": l, "command": _module_command(meta.get("argv")), "utc": meta.get("utc", ""),
                     "git": (meta.get("git") or {}).get("commit", "")[:12] + (" dirty" if (meta.get("git") or {}).get("dirty") else ""),
                     "status": "present", "python": meta.get("python", ""), "host": meta.get("host", "")})
    rows.append({"layer": "(all)", "command": "bash scripts/evals/run_research_evals.sh [--smoke]", "status": "runner"})
    return rows


def section_outputs() -> Dict[str, Any]:
    idx = []
    for root in (EVALS_ROOT, REPO_ROOT / "inputs/evals/research", REPO_ROOT / "artifacts/skillllm_v2/evals_gpu"):
        if not root.exists():
            continue
        for f in sorted(q for q in root.rglob("*") if q.is_file()):
            if OUT in f.parents:
                continue
            st = f.stat()
            idx.append({"path": rel(f), "bytes": st.st_size, "sha256": sha256_file(f) if st.st_size < 2_000_000_000 else "skipped (>2GB)"})
    done = [rel(p) for p in (REPO_ROOT / "artifacts/skillllm_v2/evals_gpu").glob("*/DONE.json")] \
        if (REPO_ROOT / "artifacts/skillllm_v2/evals_gpu").exists() else []
    return {"n_files": len(idx), "files": idx, "gpu_jobs_done": sorted(done)}


# ---------------------------------------------------------------------------------------------------------- render
def render(card: Dict[str, Any]) -> str:
    g = card["git"]
    L = ["# Evaluation card — SkillLLM_v2 research evaluation (proposed name: SkillGraphLM)", "",
         f"Generated {card['generated_utc']} by `python -m skillllm_v2.evals.research.repro_card` (re-runnable). "
         "Machine-readable twin: `eval_card.json` (complete lists; this page abbreviates long lists).", "",
         "## 1. Code version", "",
         f"- Repository `{g['repository']}`, branch `{g['branch']}`, commit `{g['commit']}` ({g['commit_date']}).",
         f"- Working tree dirty: **{g['dirty']}** ({g['n_dirty_entries']} entries under SkillLLM_v2_stage2). The evaluation code is "
         "uncommitted, so the commit alone does not identify it. The file hashes below and in `intermediate_outputs` do.", ""]
    ds = card["datasets"]
    L += ["## 2. Datasets", "", "SkillSpan (HF export, CC-BY-4.0, Zhang et al. 2022):", "",
          md_table([{"file": d["path"], "bytes": d["bytes"], "sha256": d["sha256"]} for d in ds["skillspan_hf"]], ["file", "bytes", "sha256"]), "",
          f"Derived SkillLLM_v2 data: {len(ds['skillllm_v2_derived'])} files; course benchmark inputs: "
          f"{len(ds['course_benchmark_inputs'])} files; eval inputs: {len(ds['eval_inputs'])} files (hashes in JSON).", "",
          md_table([{"file": d["path"], "sha256": d["sha256"][:16]} for d in ds["skillllm_v2_derived"] + ds["course_benchmark_inputs"]],
                   ["file", "sha256"]), "",
          f"External datasets (`{ds['external_manifest']['path']}`, sha256 `{ds['external_manifest']['sha256']}`):", ""]
    ext = ds["external_manifest"]["entries"]
    L += [md_table([{k: e.get(k) for k in ("name", "version_or_commit", "license", "downloaded_by", "n_files")} for e in ext],
                   ["name", "version_or_commit", "license", "downloaded_by", "n_files"]) if ext else "_No external dataset recorded yet._", ""]
    m = card["models"]
    L += ["## 3. Models and adapters", "",
          f"- Base model: `{m['base_model']['hub_id_assumed']}` (local `{m['base_model']['local_dir']}` -> "
          f"`{m['base_model']['resolved']}`); revision: {m['base_model']['revision']}.",
          f"- Base config/tokenizer hashes: " + ", ".join(f"{f['path']}={f['sha256'][:12]}" for f in m["base_model"]["files"]),
          f"- Base weights: {len(m['base_model']['weights'])} safetensors files, "
          + ("hashed." if m["base_model"]["weights"] and not str(m["base_model"]["weights"][0]["sha256"]).startswith("not")
             else "not hashed in this run (pass --hash-base-weights; ~96 GB)."), "",
          md_table([{"adapter": a["dir"], "weights_sha256": next((f["sha256"][:16] for f in a["files"] if f["path"].endswith(".safetensors")), "-"),
                     "config_sha256": next((f["sha256"][:16] for f in a["files"] if f["path"] == "adapter_config.json"), "-"),
                     "checkpoints": ", ".join(a["checkpoints"]) or "-"} for a in m["adapters"]],
                   ["adapter", "weights_sha256", "config_sha256", "checkpoints"]), "",
          "Hugging Face cache revisions used by the evaluation (models downloaded by agents are listed with their snapshot):", "",
          md_table([{"repo": h["repo"], "snapshots": ", ".join(s[:12] for s in h["snapshots"]) or "-", "cache": h["cache"]} for h in m["hf_cache"]],
                   ["repo", "snapshots", "cache"]) if m["hf_cache"] else "_none found_", ""]
    L += ["## 4. Predictions", "", md_table([{"file": p["path"], "rows": p["rows"], "sha256": p["sha256"][:16], "mtime_utc": p["mtime_utc"]}
                                              for p in card["predictions"]], ["file", "rows", "sha256", "mtime_utc"]), ""]
    t = card["taxonomy_graph_index"]
    L += ["## 5. Taxonomy, graph, retrieval index", "",
          f"- Taxonomy `{t['taxonomy'].get('path')}` sha256 `{t['taxonomy'].get('sha256')}`: {t['taxonomy'].get('rows')} alias rows, "
          f"{t['taxonomy'].get('concepts')} concepts, alias sources {t['taxonomy'].get('alias_sources')}.",
          f"- External taxonomies: {t['external_taxonomies'] or 'none downloaded yet'}.",
          f"- Graph: {t['graph_snapshot']['note']} Inputs: " + ", ".join(f"`{i['path']}`={i['sha256'][:12]}" for i in t["graph_snapshot"]["inputs"]),
          f"- Retrieval index: {t['retrieval_index']['note']}", ""]
    L += ["## 6. Prompts", "", md_table([{"name": p.get("name"), "file": p.get("path", p.get("error")), "sha256": (p.get("sha256") or "")[:16],
                                           "default": p.get("default", "")} for p in card["prompts"]], ["name", "file", "sha256", "default"]),
          "", "The shipped anchor and loop-v1/v2 adapters were trained and evaluated with the `original` prompt; `short` "
          "applies only to the loop-v3 adapter trained with it.", ""]
    s = card["seeds"]
    L += ["## 7. Seeds", "", f"- Layer seeds: {s['layer_seeds']}", f"- Training config seeds: {s['training_config_seeds']}",
          f"- Bootstrap: {s['bootstrap_convention']}; recommender benchmark seed {s['recommender_benchmark_seed']}, split salt "
          f"`{s['recommender_split_salt']}`.", ""]
    e = card["environment"]
    L += ["## 8. Software environment", ""]
    for name in ("cpu_miniconda", "gpu_venv"):
        info = e.get(name, {})
        L += [f"- **{name}** `{info.get('python')}` Python {info.get('version', '?')}; pip freeze `{info.get('pip_freeze', 'not captured')}`"
              f" (sha256 {str(info.get('pip_freeze_sha256', ''))[:12]}). Key packages: "
              + "; ".join((info.get("key_packages") or "").splitlines())]
    L += [f"- torch / CUDA / cuDNN / cuda_available (GPU venv; availability as seen by the card process, which the CPU launcher runs without GPUs): `{e.get('torch_cuda')}`", f"- nvidia-smi: `{e.get('nvidia_smi')}`",
          f"- nvcc: `{e.get('nvcc') or 'not on PATH'}`", ""]
    h = card["hardware"]
    L += ["## 9. Hardware", "", f"- Host `{h['host']}`, {h['platform']}", f"- CPU: {' | '.join(h['cpu'].splitlines())}",
          f"- {h['memory']}", f"- GPUs: {' | '.join(h['gpus'].splitlines())}", f"- /usr/data: {h['disk_usr_data']}", ""]
    L += ["## 10. Exact commands per layer (from results.json meta.argv)", "",
          md_table(card["commands"], ["layer", "command", "utc", "git", "status"]), "",
          "Heavy jobs run through `/home/923873155/server-guardrails/bin/guardrails run cpu|--ram 16 gpu0|gpu1 -- <cmd>`; "
          "GPU jobs (anchor re-inference, fair BERT-family baselines, robustness) are listed with their logs in `Progress.md`.", ""]
    o = card["intermediate_outputs"]
    L += ["## 11. Intermediate outputs", "", f"{o['n_files']} files under `reports/skillllm_v2/evals/`, `inputs/evals/research/` and "
          f"`artifacts/skillllm_v2/evals_gpu/`, each with sha256 in `eval_card.json` (`intermediate_outputs.files`). "
          f"GPU jobs with DONE.json: {o['gpu_jobs_done'] or 'none yet'}.", ""]
    L += ["## 12. Known reproducibility limits", "",
          "- Stored anchor offsets were produced by an older `schema.find_offsets` (first plain-substring match); the current "
          "code matches word boundaries first, so re-running the anchor cannot reproduce stored offsets byte-for-byte "
          "(Progress.md integrity-flag, A3). Model-output identity and offset effects are reported separately by L01/L12.",
          "- One training run per adapter; two same-seed 80-epoch runs differed by 0.005 exact F1 (run-1 adapter overwritten).",
          "- Base-model revision is not recorded in the local copy; the config/tokenizer hashes above identify it.",
          "- The course catalogue, scenarios and recommender labels are synthetic and project-authored.", ""]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hash-base-weights", action="store_true", help="sha256 the ~96 GB base weights (slow; use guardrails)")
    ap.add_argument("--no-freeze", action="store_true", help="skip pip freeze (faster)")
    ap.add_argument("--smoke", action="store_true", help="write to repro/smoke/ (skips freeze)")
    a = ap.parse_args(argv)
    out = OUT / ("smoke" if a.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    layers = load_layers()
    card = {"schema": "skillllm_v2-eval-card/1.0", "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": sys.argv, "git": section_git(), "datasets": section_datasets(), "models": section_models(a.hash_base_weights),
            "predictions": section_predictions(), "taxonomy_graph_index": section_taxonomy_graph_index(),
            "prompts": section_prompts(out), "seeds": section_seeds(layers),
            "environment": section_env(out, freeze=not (a.no_freeze or a.smoke)), "hardware": section_hardware(),
            "commands": section_commands(layers), "intermediate_outputs": section_outputs()}
    (out / "eval_card.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    (out / "EVAL_CARD.md").write_text(render(card) + "\n")
    print(f"eval card: {rel(out / 'EVAL_CARD.md')}; predictions {len(card['predictions'])}, adapters {len(card['models']['adapters'])}, "
          f"outputs indexed {card['intermediate_outputs']['n_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
