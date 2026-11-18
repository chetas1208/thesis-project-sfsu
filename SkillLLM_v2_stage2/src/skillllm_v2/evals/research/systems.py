"""L12_systems: model sizes, memory, latency / throughput, accuracy-vs-cost, and the anchor re-run reproduction check.

Everything is read from files: safetensors headers (parameter counts without loading weights), on-disk sizes, the
DONE.json + per-doc rows of A3's GPU/CPU jobs (artifacts/skillllm_v2/evals_gpu/*), stored predictions (exact F1 on the
974-doc test subset via extraction_evals.doc_counts). Batch size 1 everywhere; the GPU-sharing / host-load state recorded
by the jobs is reported next to every latency (the host is shared: other projects keep the 20-core CPU busy, which slows
launch-bound 4-bit decoding).

Reproduction (J1): the anchor was re-run on the 974 test docs with the original decoding. Two effects are separated:
  (a) model-output identity: per doc, the ordered list of (surface_form, raw label) emitted by the model vs the stored file;
  (b) offset resolution: the stored file was aligned with the pre-1f8bd68 ``find_offsets`` (first plain substring match);
      the J1 surfaces are re-resolved with that legacy function and compared with the stored (start, end, label) sets.
J1 is scored with BOTH the current offset code (as written by the extractor today) and the legacy one.

  python -m skillllm_v2.evals.research.systems [--smoke] [--seed 42] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.evals.extraction_evals import METRICS, doc_counts
from skillllm_v2.evals.research import stats
from skillllm_v2.evals.research.common import REPO_ROOT, md_table, record, rel, write_layer

LAYER = "L12_systems"
GPU = REPO_ROOT / "artifacts/skillllm_v2/evals_gpu"
ART = REPO_ROOT / "artifacts/skillllm_v2"
OUT_DIR = REPO_ROOT / "reports/skillllm_v2/evals" / LAYER
BASE_MODEL = REPO_ROOT / "models/mistral-small-3.1-24b-instruct-2503"
ADAPTER = ART / "mistral_lora/model/adapter_model.safetensors"
DS_TEST = "SkillSpan(HF) test, 974 gold-bearing docs"


# ----------------------------------------------------------------------------- helpers
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else None


def safetensors_header(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def count_params(paths: Iterable[Path]) -> Dict[str, int]:
    """Parameter counts by top-level group from safetensors headers (no weights loaded)."""
    out: Dict[str, int] = {}
    for p in paths:
        for name, meta in safetensors_header(p).items():
            if name == "__metadata__":
                continue
            k = int(np.prod(meta["shape"])) if meta["shape"] else 1
            group = ("vision_tower" if "vision_tower" in name else "multi_modal_projector" if "multi_modal_projector" in name
                     else "language_model")
            out[group] = out.get(group, 0) + k
    out["total"] = sum(v for k, v in out.items())
    return out


def dir_size(path: Path, pattern: str = "*") -> int:
    path = Path(os.path.realpath(path))
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob(pattern) if f.is_file())


def pct(values: Sequence[float]) -> Dict[str, Optional[float]]:
    a = np.asarray([v for v in values if v is not None], float)
    if a.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "sum": float(a.sum())}


def boot_pct_ci(values: Sequence[float], q: float, n_boot: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    a = np.asarray(values, float)
    if a.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    s = np.array([np.percentile(a[rng.integers(0, a.size, a.size)], q) for _ in range(n_boot)])
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


# ----------------------------------------------------------------------------- legacy offsets (reproduction)
def legacy_find_offsets(text: str, surface: str, used: set) -> Tuple[Optional[int], Optional[int]]:
    """schema.find_offsets as of commit 3e86a1b (used when the stored anchor predictions were produced)."""
    if not surface:
        return None, None
    matches = [(m.start(), m.end()) for m in re.finditer(re.escape(surface), text)]
    if not matches:
        lt, ls = text.lower(), surface.lower()
        matches = [(m.start(), m.end()) for m in re.finditer(re.escape(ls), lt)]
    for s, e in matches:
        if (s, e) not in used:
            return s, e
    return matches[0] if matches else (None, None)


def relocate_legacy(text: str, preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-resolve offsets of the model's spans in emitted order with the legacy rule (same 'used' bookkeeping as
    MistralExtractor._skills_from_payload)."""
    used: set = set()
    out = []
    for p in preds:
        s, e = legacy_find_offsets(text, p["surface_form"], used)
        if s is not None and e is not None:
            used.add((s, e))
        out.append({**p, "start": s, "end": e})
    return out


def output_signature(preds: List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    return [(p["surface_form"], (p.get("metadata") or {}).get("raw_label", p.get("label"))) for p in preds]


def span_set(preds: List[Dict[str, Any]]) -> set:
    return {(p.get("start"), p.get("end"), p.get("label")) for p in preds}


def reproduction(n_boot: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
    recs: List[Dict[str, Any]] = []
    j1_path = GPU / "anchor_test_logprobs/predictions.jsonl"
    stored_path = ART / "mistral_lora/predictions.jsonl"
    if not (GPU / "anchor_test_logprobs/DONE.json").exists():
        recs.append(record(LAYER, "anchor", "rerun_output_identity_rate", None, n_units=974, unit="document", split="test",
                           dataset=DS_TEST, direction="higher", status="pending_gpu", reason="J1 (anchor_test_logprobs) not finished"))
        return recs, {}
    docs = {d["id"]: d for d in load_skillspan_split(REPO_ROOT / "data/source/skillspan_hf", "test") if d["gold_spans"]}
    stored = {r["id"]: r["predictions"] for r in read_jsonl(stored_path)}
    j1 = {r["id"]: r for r in read_jsonl(j1_path)}
    ids = [i for i in docs if i in stored and i in j1]
    out_same = off_same_legacy = off_same_current = 0
    differing_output, differing_offsets, parse_errors = [], [], 0
    rows_cur, rows_leg, rows_sto = [], [], []
    for i in ids:
        text, gold = docs[i]["text"], docs[i]["gold_spans"]
        new = j1[i]["predictions"]
        parse_errors += j1[i].get("status", "ok") != "ok"
        leg = relocate_legacy(text, new)
        if output_signature(new) == output_signature(stored[i]):
            out_same += 1
        else:
            differing_output.append(i)
        if span_set(leg) == span_set(stored[i]):
            off_same_legacy += 1
        else:
            differing_offsets.append(i)
        off_same_current += span_set(new) == span_set(stored[i])
        rows_cur.append(doc_counts(gold, new, text))
        rows_leg.append(doc_counts(gold, leg, text))
        rows_sto.append(doc_counts(gold, stored[i], text))
    n = len(ids)
    C, L, S = (np.array(x, float) for x in (rows_cur, rows_leg, rows_sto))
    f1 = METRICS["exact_span_f1"]
    res = {"n_docs": n, "output_identical_docs": out_same, "output_identity_rate": out_same / n,
           "legacy_offset_spanset_identical_docs": off_same_legacy, "current_offset_spanset_identical_docs": off_same_current,
           "differing_output_ids": differing_output[:50], "differing_offset_ids_legacy": differing_offsets[:50],
           "parse_errors": parse_errors,
           "exact_f1_stored": float(f1(S.sum(0))), "exact_f1_rerun_legacy_offsets": float(f1(L.sum(0))),
           "exact_f1_rerun_current_offsets": float(f1(C.sum(0)))}
    for name, rows in (("current", C), ("legacy", L), ("stored", S)):
        res[f"metrics_{name}"] = {k: float(fn(rows.sum(0))) for k, fn in METRICS.items()}
    pb = stats.paired_bootstrap(C, S, f1, n_boot=n_boot, seed=seed)
    ar = stats.approximate_randomization(C, S, f1, n_shuffles=10000, seed=seed)
    res["delta_current_minus_stored"] = {k: pb[k] for k in ("delta", "ci_low", "ci_high", "p", "n_units_differ")} | {"p_ar": ar["p"]}
    pb2 = stats.paired_bootstrap(L, S, f1, n_boot=n_boot, seed=seed)
    res["delta_legacy_minus_stored"] = {k: pb2[k] for k in ("delta", "ci_low", "ci_high", "p", "n_units_differ")}
    prov = [stored_path, j1_path, GPU / "anchor_test_logprobs/DONE.json"]
    cav = ["stored predictions remain the official anchor artifact (coordinator ruling); the re-run quantifies reproducibility"]
    recs.append(record(LAYER, "anchor", "rerun_output_identity_rate", res["output_identity_rate"], n_units=n, unit="document",
                       split="test", dataset=DS_TEST, direction="higher", provenance=prov, caveats=cav,
                       details={"definition": "share of docs whose ordered (surface_form, raw label) list from the J1 re-run equals the stored file",
                                "differing_ids": differing_output[:50], "parse_errors": parse_errors}))
    recs.append(record(LAYER, "anchor", "rerun_spanset_identity_rate_legacy_offsets", off_same_legacy / n, n_units=n,
                       unit="document", split="test", dataset=DS_TEST, direction="higher", provenance=prov, caveats=cav,
                       details={"definition": "share of docs whose J1 (start,end,label) set, re-resolved with the legacy find_offsets, equals the stored set"}))
    recs.append(record(LAYER, "anchor", "rerun_spanset_identity_rate_current_offsets", off_same_current / n, n_units=n,
                       unit="document", split="test", dataset=DS_TEST, direction="higher", provenance=prov, caveats=cav,
                       details={"definition": "share of docs whose J1 (start,end,label) set as written by today's extractor equals the stored set"}))
    for name, rows in (("rerun_current_offsets", C), ("rerun_legacy_offsets", L), ("stored", S)):
        ci = stats.bootstrap_ci(rows, f1, n_boot=n_boot, seed=seed)
        recs.append(record(LAYER, "anchor", f"exact_span_f1_{name}", ci["value"], n_units=n, unit="document", split="test",
                           dataset=DS_TEST, direction="higher", ci=(ci["ci_low"], ci["ci_high"]), ci_method="document bootstrap",
                           provenance=prov, caveats=cav))
    recs.append(record(LAYER, "anchor", "exact_span_f1_delta_rerun_current_minus_stored", pb["delta"], n_units=n, unit="document",
                       split="test", dataset=DS_TEST, direction="none", ci=(pb["ci_low"], pb["ci_high"]),
                       ci_method="paired document bootstrap", provenance=prov,
                       details={"p_boot": pb["p"], "p_ar": ar["p"], "n_units_differ": pb["n_units_differ"],
                                "interpretation": "offset-code drift + any decoding nondeterminism"}))
    out = GPU / "anchor_test_logprobs/REPRODUCTION.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    return recs, res


# ----------------------------------------------------------------------------- systems inventory
def exact_f1_974(pred_path: Path, n_boot: int, seed: int) -> Optional[Dict[str, Any]]:
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
    if not pred_path.exists():
        return None
    docs = [d for d in load_skillspan_split(REPO_ROOT / "data/source/skillspan_hf", "test") if d["gold_spans"]]
    preds = {r["id"]: r["predictions"] for r in read_jsonl(pred_path)}
    rows = np.array([doc_counts(d["gold_spans"], preds.get(d["id"], []), d["text"]) for d in docs], float)
    ci = stats.bootstrap_ci(rows, METRICS["exact_span_f1"], n_boot=n_boot, seed=seed)
    return {"value": ci["value"], "ci": (ci["ci_low"], ci["ci_high"]), "n": len(docs), "missing": sum(d["id"] not in preds for d in docs)}


def anchor_jobs() -> Dict[str, Dict[str, Any]]:
    jobs = {}
    for job in ("anchor_test_logprobs", "anchor_dev_logprobs", "anchor_test_nospan", "robustness_anchor", "xdata_anchor"):
        done = load_json(GPU / job / "DONE.json")
        if done is None:
            continue
        rows = read_jsonl(GPU / job / "predictions.jsonl")
        jobs[job] = {"done": done, "rows": rows}
    return jobs


def cpu_rss_gb(system: str, done: Dict[str, Any]) -> Optional[float]:
    """Peak process RSS (GB): from the test974 DONE.json if the job recorded it, else from the dedicated fresh-process
    measurement (scripts/evals/gpu_cpu_rss.sh -> evals_gpu/smoke/rss_<system>/DONE.json, 100 docs)."""
    v = done.get("process_peak_rss_gb_so_far")
    if v:
        return v
    d = load_json(GPU / "smoke" / f"rss_{system}" / "DONE.json")
    return d.get("process_peak_rss_gb_so_far") if d else None


def latency_block(rows: List[Dict[str, Any]], key: str = "latency_seconds") -> Dict[str, Any]:
    lat = [r[key] for r in rows if r.get(key) is not None]
    per_span = [r[key] / len(r["predictions"]) for r in rows if r.get(key) is not None and r["predictions"]]
    total_spans = sum(len(r["predictions"]) for r in rows)
    gen = [r["n_gen_tokens"] for r in rows if r.get("n_gen_tokens")]
    tps = [r["n_gen_tokens"] / r[key] for r in rows if r.get("n_gen_tokens") and r.get(key)]
    snaps = [r["gpu_snapshot"] for r in rows if r.get("gpu_snapshot")]
    return {"per_doc": pct(lat), "per_predicted_span_doclevel": pct(per_span),
            "seconds_per_predicted_span_aggregate": (sum(lat) / total_spans) if total_spans else None,
            "throughput_docs_per_hour": 3600 / float(np.mean(lat)) if lat else None,
            "gen_tokens": pct(gen), "gen_tokens_per_second": pct(tps),
            "prompt_tokens": pct([r.get("n_prompt_tokens") for r in rows if r.get("n_prompt_tokens")]),
            "gpu_sampled_docs": len(snaps),
            "gpu_sampled_docs_shared": sum(1 for s in snaps if s.get("other_compute_pids")),
            "host_loadavg_1m": pct([s.get("host_loadavg_1m") for s in snaps if s.get("host_loadavg_1m") is not None]),
            "gpu_util_pct": pct([s.get("util_pct") for s in snaps if s.get("util_pct") is not None])}


def run(n_boot: int, seed: int, smoke: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {}
    # --- parameters / disk
    shards = sorted(BASE_MODEL.glob("model-*.safetensors"))
    base = count_params(shards) if shards else {}
    lora = count_params([ADAPTER]) if ADAPTER.exists() else {}
    info["params"] = {"base": base, "lora": lora}
    if base:
        recs.append(record(LAYER, "anchor_base", "n_params", base["total"], n_units=1, unit="model", split="n/a",
                           dataset="model files", direction="none", provenance=[BASE_MODEL / "model.safetensors.index.json"],
                           details={"by_group": base, "source": "safetensors headers of the 10 HF shards (bf16 checkpoint)",
                                    "note": "Mistral-Small-3.1 is multimodal; only the language model is used for extraction"}))
        recs.append(record(LAYER, "anchor_base", "n_params_language_model", base.get("language_model"), n_units=1, unit="model",
                           split="n/a", dataset="model files", direction="none", provenance=[BASE_MODEL / "model.safetensors.index.json"]))
    if lora:
        recs.append(record(LAYER, "anchor_lora", "n_params_trainable", lora["total"], n_units=1, unit="model", split="n/a",
                           dataset="adapter files", direction="none", provenance=[ADAPTER],
                           details={"fraction_of_base": lora["total"] / base["total"] if base else None,
                                    "adapter_config": load_json(ADAPTER.parent / "adapter_config.json")}))
    disk = {
        "base_model_hf_shards": sum(p.stat().st_size for p in shards),
        "base_model_dir_total": dir_size(BASE_MODEL),
        "lora_adapter": ADAPTER.stat().st_size if ADAPTER.exists() else None,
        "span_verifier_joblib": (ART / "span_verifier/model/verifier_model.joblib").stat().st_size
        if (ART / "span_verifier/model/verifier_model.joblib").exists() else None,
        "token_lookup_table": (ART / "bert_baseline/model/model.json").stat().st_size,
    }
    sel = load_json(GPU / "bert_fair/SELECTED.json")
    if sel:
        disk["bert_fair_model_pt"] = (REPO_ROOT / sel["selected"] / "model.pt").stat().st_size
    gl = sorted(Path("/usr/data/models/skillllm-v2-stage2/hf_cache_a3/hub").glob("models--urchade--gliner_medium-v2.1"))
    if gl:
        disk["gliner_medium_v2.1_cache"] = dir_size(gl[0] / "blobs")
    info["disk_bytes"] = disk
    for k, v in disk.items():
        if v is not None:
            recs.append(record(LAYER, k, "disk_gb", v / 1e9, n_units=1, unit="artifact", split="n/a", dataset="files",
                               direction="lower", details={"bytes": v, "note": ("dir also holds a 48 GB consolidated.safetensors "
                                                                               "(Mistral-native format) not read by the HF loader")
                                                                      if k == "base_model_dir_total" else ""}))
    vjob = ART / "span_verifier/model/verifier_model.joblib"
    recs.append(record(LAYER, "span_verifier", "n_params", None, n_units=1, unit="model", split="n/a", dataset="files",
                       direction="none", status="not_available",
                       reason=("the safe wrapper's SpanVerifier (src/skillllm_v2/verify/span_verifier.py) is rule-based "
                               "(confidence threshold + generic-phrase / length rules); the TF-IDF+LR joblib in "
                               f"{rel(vjob)} is a separate artifact, its size is reported as disk_gb")))
    recs.append(record(LAYER, "taxonomy_reranker", "n_params", None, n_units=1, unit="model", split="n/a", dataset="files",
                       direction="none", status="not_available",
                       reason="never trained: artifacts/skillllm_v2/taxonomy_reranker/metrics.json training_mode=retrieval_only (placeholder README)"))
    # --- anchor GPU jobs
    jobs = anchor_jobs()
    info["anchor_jobs"] = {}
    for job, d in jobs.items():
        segs = d["done"].get("segments", [])
        lb = latency_block(d["rows"])
        lb["peak_alloc_gb"] = max((s.get("peak_alloc_gb") or 0) for s in segs) if segs else None
        lb["peak_reserved_gb"] = max((s.get("peak_reserved_gb") or 0) for s in segs) if segs else None
        lb["load_seconds"] = [s.get("load_seconds") for s in segs if s.get("load_seconds")]
        lb["alloc_after_load_gb"] = [s.get("alloc_after_load_gb") for s in segs if s.get("alloc_after_load_gb")]
        lb["status_counts"] = d["done"]["summary"].get("status_counts")
        lb["process_peak_rss_gb"] = d["done"].get("process_peak_rss_gb")
        lb["gpu"] = d["done"].get("segments", [{}])[0].get("gpu_start", {}).get("name")
        info["anchor_jobs"][job] = lb
    j1 = info["anchor_jobs"].get("anchor_test_logprobs")
    if j1:
        rows = jobs["anchor_test_logprobs"]["rows"]
        lat = [r["latency_seconds"] for r in rows]
        lo, hi = boot_pct_ci(lat, 95, 1000 if not smoke else 100, seed)
        prov = [GPU / "anchor_test_logprobs/predictions.jsonl", GPU / "anchor_test_logprobs/DONE.json"]
        _ld = j1['host_loadavg_1m'].get('p50')
        cav = [(f"shared host: median 1-min load average during the run {_ld:.1f} on {os.cpu_count()} cores; " if _ld is not None else
                f"shared host ({os.cpu_count()} cores): load average was NOT recorded in the J1 snapshots (recorded 35.3 / 35.2 / 34.9 median in the later J3/J4/J6 runs, ~38 in the coordinator's Progress.md notes); ") +
               "bitsandbytes 4-bit decoding is CPU-launch-bound, so latency depends on host load",
               "batch size 1, greedy, max_new_tokens 512, logprob recording on (output_scores), one RTX 3090"]
        recs.append(record(LAYER, "anchor", "latency_p95", j1["per_doc"]["p95"], n_units=len(lat), unit="document", split="test",
                           dataset=DS_TEST, direction="lower", ci=(lo, hi), ci_method="bootstrap of the 95th percentile over docs (1000)",
                           provenance=prov, caveats=cav,
                           details={"headline": True, "unit_of_value": "seconds per document", "p50": j1["per_doc"]["p50"],
                                    "p99": j1["per_doc"]["p99"], "mean": j1["per_doc"]["mean"],
                                    "per_predicted_span": j1["per_predicted_span_doclevel"],
                                    "throughput_docs_per_hour": j1["throughput_docs_per_hour"],
                                    "gen_tokens_per_second": j1["gen_tokens_per_second"], "gpu_sampled_docs_shared": j1["gpu_sampled_docs_shared"]}))
        recs.append(record(LAYER, "anchor", "peak_vram_gb", j1["peak_reserved_gb"], n_units=len(lat), unit="document", split="test",
                           dataset=DS_TEST, direction="lower", provenance=prov, caveats=cav[1:],
                           details={"headline": True, "definition": "torch.cuda.max_memory_reserved over the J1 run (1e9 bytes); "
                                    "nvidia-smi adds ~0.3-0.5 GB CUDA context", "peak_allocated_gb": j1["peak_alloc_gb"],
                                    "alloc_after_load_gb": j1["alloc_after_load_gb"]}))
        recs.append(record(LAYER, "anchor", "cpu_peak_rss_gb", None, n_units=len(lat), unit="document", split="test", dataset=DS_TEST,
                           direction="lower", status="not_available",
                           reason="host process RSS was not recorded by the anchor GPU jobs (gpu_anchor_infer.py logs CUDA memory and "
                                  "nvidia-smi snapshots only); not re-measured to avoid a second 4-bit model load"))
        recs.append(record(LAYER, "anchor", "latency_p50", j1["per_doc"]["p50"], n_units=len(lat), unit="document", split="test",
                           dataset=DS_TEST, direction="lower", provenance=prov, caveats=cav))
        recs.append(record(LAYER, "anchor", "latency_p99", j1["per_doc"]["p99"], n_units=len(lat), unit="document", split="test",
                           dataset=DS_TEST, direction="lower", provenance=prov, caveats=cav))
        recs.append(record(LAYER, "anchor", "seconds_per_predicted_span", j1["seconds_per_predicted_span_aggregate"], n_units=len(lat),
                           unit="document", split="test", dataset=DS_TEST, direction="lower", provenance=prov, caveats=cav))
        recs.append(record(LAYER, "anchor", "throughput_docs_per_hour_per_gpu", j1["throughput_docs_per_hour"], n_units=len(lat),
                           unit="document", split="test", dataset=DS_TEST, direction="higher", provenance=prov, caveats=cav))
        xs = np.array([r.get("n_gen_tokens") or 0 for r in rows], float)
        if xs.any():
            X = np.c_[np.ones_like(xs), xs]
            y = np.array(lat, float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            info["latency_model"] = {"intercept_s": float(coef[0]), "seconds_per_generated_token": float(coef[1]), "r2": float(r2)}
            recs.append(record(LAYER, "anchor", "decode_seconds_per_generated_token", float(coef[1]), n_units=len(lat),
                               unit="document", split="test", dataset=DS_TEST, direction="lower", provenance=prov, caveats=cav,
                               details={"model": "OLS latency = a + b * n_generated_tokens over J1 docs", "intercept_s": float(coef[0]),
                                        "r2": float(r2), "mean_generated_tokens": float(xs.mean())}))
    else:
        for m in ("latency_p95", "peak_vram_gb"):
            recs.append(record(LAYER, "anchor", m, None, n_units=974, unit="document", split="test", dataset=DS_TEST,
                               direction="lower", status="pending_gpu", reason="J1 not finished", details={"headline": True}))
    tpath = ART / "qlora_loop_v4_eval/timing_segments.jsonl"
    if tpath.exists():
        seg = read_jsonl(tpath)[0]
        recs.append(record(LAYER, "qlora_v4_adapter_prior_run", "mean_seconds_per_doc", seg["gen_seconds"] / seg["docs_generated"],
                           n_units=seg["docs_generated"], unit="document", split="test", dataset=DS_TEST, direction="lower",
                           provenance=[tpath], details={"peak_alloc_gb": seg["peak_alloc_gb"], "peak_reserved_gb": seg["peak_reserved_gb"],
                                                         "note": "same base + an r=16 adapter, same decoding path with logprobs, earlier quieter host"}))
    # --- CPU systems (and the wrapper stages) on the same 974 docs
    cpu_sys = {}
    for system in ("token_lookup", "gliner_zeroshot", "bert_fair", "safe_wrapper"):
        done = load_json(GPU / f"test974_{system}" / "DONE.json")
        if done is None:
            continue
        rows = read_jsonl(GPU / f"test974_{system}" / "predictions.jsonl")
        cpu_sys[system] = {"done": done, "rows": rows, "lat": latency_block(rows)}
        prov = [GPU / f"test974_{system}" / "predictions.jsonl", GPU / f"test974_{system}" / "DONE.json"]
        recs.append(record(LAYER, system, "latency_p95", cpu_sys[system]["lat"]["per_doc"]["p95"], n_units=len(rows), unit="document",
                           split="test", dataset=DS_TEST, direction="lower", provenance=prov,
                           caveats=["CPU, batch 1, guardrails-pinned threads on a shared host"] + (
                               ["wrapper latency EXCLUDES the anchor LLM call"] if system == "safe_wrapper" else []),
                           details={"device": done.get("device", "cpu"), "p50": cpu_sys[system]["lat"]["per_doc"]["p50"],
                                    "p99": cpu_sys[system]["lat"]["per_doc"]["p99"],
                                    "peak_rss_gb": cpu_rss_gb(system, done), "peak_rss_source": "test974 job" if done.get("process_peak_rss_gb_so_far") else "fresh-process measurement, 100 docs",
                                    "stage_latency_seconds": done.get("stage_latency_seconds"),
                                    "n_params": done.get("n_params")}))
        if system == "safe_wrapper" and done.get("stage_latency_seconds"):
            for stage, v in done["stage_latency_seconds"].items():
                recs.append(record(LAYER, f"safe_wrapper:{stage}", "stage_latency_mean_s", v["mean"], n_units=v["n"], unit="document",
                                   split="test", dataset=DS_TEST, direction="lower", provenance=prov,
                                   details={"p50": v["p50"], "p95": v["p95"], "p99": v["p99"], "sum_s": v["sum"]}))
    info["cpu_systems"] = {k: {"lat": v["lat"], "done": {**{kk: v["done"].get(kk) for kk in ("n_params", "device", "model", "process_peak_rss_gb_so_far", "load_seconds")}, "process_peak_rss_gb_so_far": cpu_rss_gb(k, v["done"])}}
                           for k, v in cpu_sys.items()}
    # --- reproduction of the CPU baselines against stored artifacts
    for system, stored in (("token_lookup", ART / "bert_baseline/predictions.jsonl"), ("gliner_zeroshot", ART / "gliner_baseline/predictions.jsonl"),
                           ("safe_wrapper", ART / "safe_wrapper/predictions.jsonl")):
        if system not in cpu_sys:
            continue
        st = {r["id"]: r["predictions"] for r in read_jsonl(stored)}
        rows = cpu_sys[system]["rows"]
        same = sum(span_set(r["predictions"]) == span_set(st.get(r["id"], [])) for r in rows)
        recs.append(record(LAYER, system, "rerun_spanset_identity_rate", same / len(rows), n_units=len(rows), unit="document",
                           split="test", dataset=DS_TEST, direction="higher", provenance=[stored, GPU / f"test974_{system}" / "predictions.jsonl"],
                           details={"definition": "share of docs whose re-run (start,end,label) set equals the stored artifact"}))
    # --- J5 training cost
    for p in sorted((GPU / "bert_fair/runs").glob("*/DONE.json")):
        d = json.loads(p.read_text())
        recs.append(record(LAYER, f"bert_fair:{p.parent.name}", "train_seconds", d["train_seconds"], n_units=1, unit="run",
                           split="train", dataset="SkillSpan train (4800) + dev selection", direction="lower", provenance=[p],
                           details={"peak_alloc_gb_train": d.get("peak_alloc_gb"), "epochs_run": d["epochs_run"], "best_epoch": d["best_epoch"],
                                    "n_params": d["n_params"], "test_predict_seconds_3569_docs_batch64": d["test_predict_seconds"]}))
    # --- accuracy vs cost
    pareto = []
    f1_paths = {"anchor": ART / "mistral_lora/predictions.jsonl", "safe_wrapper": ART / "safe_wrapper/predictions.jsonl",
                "gliner_zeroshot": ART / "gliner_baseline/predictions.jsonl", "token_lookup": ART / "bert_baseline/predictions.jsonl",
                "bert_fair": GPU / "bert_fair/predictions.jsonl"}
    for system, path in f1_paths.items():
        f = exact_f1_974(path, n_boot, seed) if path.exists() else None
        if f is None:
            continue
        if system == "anchor" and j1:
            p95, mem, memkind = j1["per_doc"]["p95"], j1["peak_reserved_gb"], "GPU VRAM (reserved)"
        elif system == "safe_wrapper" and j1 and "safe_wrapper" in cpu_sys:
            lat_a = {r["id"]: r["latency_seconds"] for r in jobs["anchor_test_logprobs"]["rows"]}
            tot = [lat_a[r["id"]] + r["latency_seconds"] for r in cpu_sys["safe_wrapper"]["rows"] if r["id"] in lat_a]
            p95, mem, memkind = pct(tot)["p95"], j1["peak_reserved_gb"], "GPU VRAM (anchor) + CPU wrapper"
        elif system in cpu_sys:
            p95 = cpu_sys[system]["lat"]["per_doc"]["p95"]
            mem, memkind = cpu_rss_gb(system, cpu_sys[system]["done"]), "CPU RSS (process)"
        else:
            p95, mem, memkind = None, None, None
        pareto.append({"system": system, "exact_f1": f["value"], "exact_f1_ci": f["ci"], "latency_p95_s": p95, "memory_gb": mem,
                       "memory_kind": memkind})
    info["pareto"] = pareto
    return recs, info


def plot_pareto(pareto: List[Dict[str, Any]], out: Path) -> List[Path]:
    pts = [p for p in pareto if p["latency_p95_s"] is not None]
    if not pts:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for xkey, xlabel, fname in (("latency_p95_s", "p95 latency per document (s, log scale, batch 1)", "pareto_f1_vs_latency.png"),
                                ("memory_gb", "peak memory (GB): VRAM for GPU systems, process RSS for CPU systems", "pareto_f1_vs_memory.png")):
        q = [p for p in pts if p.get(xkey)]
        if not q:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for k, p in enumerate(q):
            gpu = "VRAM" in (p["memory_kind"] or "")
            lo, hi = p["exact_f1_ci"]
            ax.errorbar(p[xkey], p["exact_f1"], yerr=[[p["exact_f1"] - lo], [hi - p["exact_f1"]]], fmt="o" if gpu else "s", capsize=3)
            ax.annotate(p["system"], (p[xkey], p["exact_f1"]), textcoords="offset points", xytext=(-60 if k % 2 else 5, -14 if k % 2 else 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("exact-span F1, 974-doc SkillSpan test subset (95% CI)")
        ax.set_title("L12 accuracy vs cost (circles: GPU, squares: CPU)")
        fig.tight_layout()
        p_out = out / fname
        fig.savefig(p_out, dpi=150)
        plt.close(fig)
        paths.append(p_out)
    return paths


def report(recs: List[Dict[str, Any]], info: Dict[str, Any], repro: Dict[str, Any], figs: List[Path]) -> str:
    def val(system: str, metric: str) -> Optional[Dict[str, Any]]:
        return next((r for r in recs if r["system"] == system and r["metric"] == metric), None)
    L = ["# L12_systems — size, memory, latency, accuracy-vs-cost, reproduction", ""]
    a, v = val("anchor", "latency_p95"), val("anchor", "peak_vram_gb")
    if a and a["status"] == "measured":
        L += [f"**Headline:** anchor p95 latency **{a['value']:.2f} s/doc** [95% CI {a['ci_low']:.2f}, {a['ci_high']:.2f}] "
              f"(p50 {a['details']['p50']:.2f}, p99 {a['details']['p99']:.2f}; n = {a['n_units']} test docs, batch 1, one RTX 3090); "
              f"peak VRAM **{v['value']:.2f} GB** (reserved; allocated {v['details']['peak_allocated_gb']:.2f} GB).", ""]
    else:
        L += ["**Headline:** pending (J1 not finished).", ""]
    lm = info.get("latency_model")
    if lm:
        L += [f"Latency is dominated by output length: OLS over J1 docs gives {lm['intercept_s']:.2f} s fixed (prompt + overhead) + "
              f"{lm['seconds_per_generated_token']:.3f} s per generated token (R² = {lm['r2']:.3f}).", ""]
    p = info.get("params", {})
    if p.get("base"):
        L += ["## Parameters and disk", "",
              f"- Base Mistral-Small-3.1-24B (bf16 checkpoint, loaded in 4-bit NF4): {p['base']['total']:,} parameters "
              f"(language model {p['base'].get('language_model', 0):,}; vision tower {p['base'].get('vision_tower', 0):,}; projector "
              f"{p['base'].get('multi_modal_projector', 0):,}).",
              f"- LoRA adapter (r=16, alpha=32, all attention+MLP projections of the language model): {p['lora'].get('total', 0):,} trainable "
              f"parameters ({100 * p['lora'].get('total', 0) / max(1, p['base']['total']):.2f}% of base).",
              "- Disk (GB): " + ", ".join(f"{k} {v / 1e9:.3f}" for k, v in info.get("disk_bytes", {}).items() if v), ""]
    jobs = info.get("anchor_jobs", {})
    if jobs:
        tab = [{"job": k, "docs": v["per_doc"]["n"], "p50_s": v["per_doc"]["p50"], "p95_s": v["per_doc"]["p95"], "p99_s": v["per_doc"]["p99"],
                "gen_tok/s": v["gen_tokens_per_second"]["p50"], "peak_alloc_GB": v["peak_alloc_gb"], "peak_reserved_GB": v["peak_reserved_gb"],
                "load_avg_p50": v["host_loadavg_1m"].get("p50"), "shared_snaps": f"{v['gpu_sampled_docs_shared']}/{v['gpu_sampled_docs']}",
                "parse_err": (v["status_counts"] or {}).get("parse_error", 0)} for k, v in jobs.items()]
        L += ["## Anchor GPU jobs (batch 1, greedy, 4-bit NF4, RTX 3090)", "",
              md_table(tab, ["job", "docs", "p50_s", "p95_s", "p99_s", "gen_tok/s", "peak_alloc_GB", "peak_reserved_GB", "load_avg_p50",
                             "shared_snaps", "parse_err"]), "",
              "`shared_snaps` = nvidia-smi snapshots (every 25 docs) that showed another compute process on the same GPU.", ""]
    cs = info.get("cpu_systems", {})
    if cs:
        tab = [{"system": k, "p50_s": v["lat"]["per_doc"]["p50"], "p95_s": v["lat"]["per_doc"]["p95"], "p99_s": v["lat"]["per_doc"]["p99"],
                "peak_rss_GB": v["done"].get("process_peak_rss_gb_so_far"), "n_params": v["done"].get("n_params")} for k, v in cs.items()]
        L += ["## CPU systems on the same 974 docs (batch 1)", "", md_table(tab, ["system", "p50_s", "p95_s", "p99_s", "peak_rss_GB", "n_params"]), ""]
        st = [r for r in recs if r["system"].startswith("safe_wrapper:")]
        if st:
            L += ["Safe-wrapper stage wall-clock (mean s/doc, CPU, excluding the anchor call): " +
                  ", ".join(f"{r['system'].split(':')[1]} {r['value']:.4f}" for r in st), ""]
    if info.get("pareto"):
        L += ["## Accuracy vs cost", "", md_table([{**x, "exact_f1_ci": f"[{x['exact_f1_ci'][0]:.3f}, {x['exact_f1_ci'][1]:.3f}]"}
                                                   for x in info["pareto"]],
                                                  ["system", "exact_f1", "exact_f1_ci", "latency_p95_s", "memory_gb", "memory_kind"]), ""]
        L += [f"![pareto]({f.name})" for f in figs] + [""]
    if repro:
        L += ["## Anchor re-run reproduction (J1)", "",
              f"- Model output identical (ordered surface + raw label) on {repro['output_identical_docs']}/{repro['n_docs']} docs "
              f"({100 * repro['output_identity_rate']:.2f}%); parse errors {repro['parse_errors']}.",
              f"- Re-resolving the re-run surfaces with the legacy find_offsets reproduces the stored span sets on "
              f"{repro['legacy_offset_spanset_identical_docs']}/{repro['n_docs']} docs; with today's offset code "
              f"{repro['current_offset_spanset_identical_docs']}/{repro['n_docs']}.",
              f"- Exact F1: stored {repro['exact_f1_stored']:.6f}; re-run + legacy offsets {repro['exact_f1_rerun_legacy_offsets']:.6f}; "
              f"re-run + current offsets {repro['exact_f1_rerun_current_offsets']:.6f} (Δ current − stored "
              f"{repro['delta_current_minus_stored']['delta']:+.4f} [{repro['delta_current_minus_stored']['ci_low']:+.4f}, "
              f"{repro['delta_current_minus_stored']['ci_high']:+.4f}], AR p = {repro['delta_current_minus_stored']['p_ar']:.4f}).",
              "- The stored predictions remain the official anchor artifact; both numbers are quotable with this explanation.", ""]
    pend = [r for r in recs if r["status"] != "measured"]
    if pend:
        L += ["## Pending / not available", ""] + [f"- {r['system']} / {r['metric']}: {r['status']} — {r['reason']}" for r in pend] + [""]
    L += ["## Caveats", "",
          "- Latency was measured on a shared host (other projects' CPU jobs; load average 35-54 on 20 cores during the J1 anchor run and the CPU baseline runs, 1-min load: median 35 in J3/J4/J6, 41-54 at the end of the test974 CPU jobs, not recorded in J1/J2 snapshots; CPU latencies are therefore inflated by contention, GPU latency by launch-bound decoding). 4-bit decoding with "
          "bitsandbytes is CPU-launch-bound, so absolute latencies are upper-bound-ish for this hardware; an earlier quieter run of "
          "the same decoding path measured ~4.6 s/doc (qlora_loop_v4_eval timing).",
          "- Batch size 1, no KV-cache reuse, no vLLM/TGI: these are research-prototype costs, not an optimised deployment.",
          "- Memory for CPU systems is process RSS, not VRAM; the Pareto memory plot mixes the two kinds (marked).", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    n_boot = 200 if args.smoke else args.n_boot
    out = OUT_DIR / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    rrecs, repro = reproduction(n_boot, args.seed)
    recs, info = run(n_boot, args.seed, args.smoke)
    recs = rrecs + recs
    figs = plot_pareto(info.get("pareto", []), out)
    (out / "systems_inventory.json").write_text(json.dumps({k: v for k, v in info.items()}, indent=2, default=str) + "\n")
    write_layer(LAYER, recs, report(recs, info, repro, figs), seed=args.seed, smoke=args.smoke, extra_meta={"n_boot": n_boot})
    print(f"wrote {out} ({len(recs)} records)")


if __name__ == "__main__":
    main()
