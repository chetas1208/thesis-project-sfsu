"""Resumable anchor (Mistral-Small-3.1-24B + LoRA) inference for the research evaluation (A3 GPU jobs J1-J4, J6).

Thin driver around the SAME code path as the anchor's original test run (smoke_runner.eval_raw_mistral_lora /
scripts/eval_adapter_test.py): config ``configs/skillllm_v2_training_80ep_2gpu.yaml`` (model path, 4-bit NF4, bf16 compute,
max_new_tokens=512), ``MistralExtractor(allow_fallback=False, prompt_version="original")``, one document at a time, greedy
decoding (``do_sample=False``). Nothing in the decoding is changed; ``--logprobs`` only adds ``output_scores`` (the
generated sequence is identical under greedy decoding) and stores per-span sum/mean/min token log-prob in
``metadata.logprob`` exactly like ``QLORA_LOGPROBS=1``.

What this adds over eval_adapter_test.py (which hard-codes the gold-bearing test split and writes into reports/):
  * any SkillSpan split + subset (gold | nogold | all) or a custom ``--input`` jsonl of ``{"id","text",...}`` rows,
  * deterministic round-robin sharding (``--shard k --num-shards n``) + a ``--merge`` step,
  * per-doc wall latency (CUDA-synchronised), prompt / generated token counts, per-doc peak CUDA memory, the raw model
    output, a parse-error status instead of a crash, periodic GPU-sharing snapshots (nvidia-smi),
  * ``DONE.json`` (counts, timings, memory, sha256 of outputs + adapter, exact command, git commit, library versions).

Outputs: artifacts/skillllm_v2/evals_gpu/<job>/predictions.jsonl (+ DONE.json), or .../<job>/shards/shard<k>of<n>/ when
sharded. ``--smoke`` writes to artifacts/skillllm_v2/evals_gpu/smoke/<job>/ (never a result).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_ROOT = REPO_ROOT.parent
DEFAULT_CONFIG = "configs/skillllm_v2_training_80ep_2gpu.yaml"  # same config the anchor eval used
DEFAULT_ADAPTER = "artifacts/skillllm_v2/mistral_lora/model"
OUT_ROOT = Path("artifacts/skillllm_v2/evals_gpu")
GPU_SAMPLE_EVERY = 25


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(GIT_ROOT), *args], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty_stage2": bool(run("status", "--porcelain", "--", str(REPO_ROOT)))}


def lib_versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node()}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = "missing"
    return out


def _peak_rss_gb() -> Optional[float]:
    """Peak resident set size of this process (Linux ru_maxrss is in KiB); for a merge step it is the merger's RSS."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9
    except Exception:  # noqa: BLE001
        return None


def physical_gpu() -> Optional[str]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return vis.split(",")[0] if vis else None


def gpu_snapshot(own_pid: int) -> Dict[str, Any]:
    """Other compute processes + utilisation on our physical GPU (CUDA_DEVICE_ORDER=PCI_BUS_ID => same index as nvidia-smi)."""
    idx = physical_gpu()
    snap: Dict[str, Any] = {"utc": utc(), "gpu_index": idx}
    try:
        snap["host_loadavg_1m"], snap["host_ncpu"] = os.getloadavg()[0], os.cpu_count()  # CPU contention matters: 4-bit decoding is launch-bound
    except OSError:
        pass
    if idx is None:
        return snap
    try:
        apps = subprocess.run(["nvidia-smi", "-i", idx, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        pids = [a.split(",")[0].strip() for a in apps if a.strip()]
        snap["other_compute_pids"] = [p for p in pids if p != str(own_pid)]
        q = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=name,utilization.gpu,memory.used,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        name, util, mem, power, temp = [x.strip() for x in q.split(",")]
        snap.update({"name": name, "util_pct": float(util), "mem_used_mib": float(mem), "power_w": float(power), "temp_c": float(temp)})
    except Exception as exc:  # noqa: BLE001
        snap["error"] = str(exc)[:200]
    return snap


# ----------------------------------------------------------------------------- documents
def load_docs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.input:
        docs = []
        with open(args.input, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    docs.append({"id": str(row["id"]), "text": row["text"], "meta": row.get("meta", {})})
    else:
        from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
        all_docs = load_skillspan_split(Path("data/source/skillspan_hf"), args.split, limit=None)
        if args.subset == "gold":
            all_docs = [d for d in all_docs if d["gold_spans"]]  # == smoke_runner._docs filter (anchor's 974-doc test subset)
        elif args.subset == "nogold":
            all_docs = [d for d in all_docs if not d["gold_spans"]]
        docs = [{"id": d["id"], "text": d["text"], "meta": {"source": d.get("source")}} for d in all_docs]
    ids = [d["id"] for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate doc ids in input")
    if args.limit:
        docs = docs[: args.limit]
    return docs


def shard_docs(docs: List[Dict[str, Any]], shard: int, n: int) -> List[Dict[str, Any]]:
    return [d for i, d in enumerate(docs) if i % n == shard]


def job_dir(args: argparse.Namespace) -> Path:
    base = OUT_ROOT / ("smoke" if args.smoke else "") / args.job
    return base


def run_dir(args: argparse.Namespace) -> Path:
    base = job_dir(args)
    return base / "shards" / f"shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else base


def read_done_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail from an interrupted write; the doc is redone
                done[row["id"]] = row
    return done


def percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None, "sum": None}
    import numpy as np
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "sum": float(a.sum())}


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    lat = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
    n_spans = sum(len(r["predictions"]) for r in rows)
    per_span = [r["latency_seconds"] / len(r["predictions"]) for r in rows if r["predictions"] and r.get("latency_seconds")]
    gen_tok = [r["n_gen_tokens"] for r in rows if r.get("n_gen_tokens") is not None]
    tok_s = [r["n_gen_tokens"] / r["gen_seconds"] for r in rows if r.get("n_gen_tokens") and r.get("gen_seconds")]
    peaks = [r["peak_alloc_gb_doc"] for r in rows if r.get("peak_alloc_gb_doc") is not None]
    statuses: Dict[str, int] = {}
    for r in rows:
        statuses[r.get("status", "ok")] = statuses.get(r.get("status", "ok"), 0) + 1
    shared = sum(1 for r in rows if (r.get("gpu_snapshot") or {}).get("other_compute_pids"))
    sampled = sum(1 for r in rows if r.get("gpu_snapshot"))
    return {"n_docs": len(rows), "status_counts": statuses, "n_predicted_spans": n_spans,
            "docs_with_predictions": sum(1 for r in rows if r["predictions"]),
            "latency_seconds_per_doc": percentiles(lat), "latency_seconds_per_predicted_span": percentiles(per_span),
            "n_gen_tokens": percentiles([float(x) for x in gen_tok]), "gen_tokens_per_second": percentiles(tok_s),
            "peak_alloc_gb_doc_max": max(peaks) if peaks else None,
            "gpu_snapshots": {"sampled_docs": sampled, "sampled_docs_with_other_compute_procs": shared}}


# ----------------------------------------------------------------------------- inference
def infer(args: argparse.Namespace) -> None:
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    docs = load_docs(args)
    if args.num_shards > 1:
        docs = shard_docs(docs, args.shard, args.num_shards)
    out = run_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    pred_path, done_path = out / "predictions.jsonl", out / "DONE.json"
    if done_path.exists() and not args.force:
        print(f"{done_path} exists; nothing to do", flush=True)
        return
    done = read_done_rows(pred_path)
    todo = [d for d in docs if d["id"] not in done]
    print(f"{utc()} job={args.job} docs={len(docs)} already_done={len(done)} todo={len(todo)} gpu={physical_gpu()}", flush=True)
    seg: Dict[str, Any] = {"start_utc": utc(), "docs_before": len(done), "pid": os.getpid(), "gpu_start": gpu_snapshot(os.getpid())}
    if todo:
        import torch
        from skillllm_v2.extract import MistralExtractor
        from skillllm_v2.extract.mistral_extractor import parse_skill_json  # noqa: F401  (documented failure path)
        from skillllm_v2.pipeline.smoke_runner import _cfg

        config = _cfg(args.config)
        model_cfg = config.get("model", {})
        extractor = MistralExtractor(
            model_path=model_cfg.get("model_path"), adapter_path=args.adapter,
            use_4bit=bool(model_cfg.get("use_4bit", True)), max_new_tokens=int(model_cfg.get("max_new_tokens", 512)),
            allow_fallback=False, prompt_version="original", record_logprobs=args.logprobs,
        )
        if not Path(args.adapter, "adapter_config.json").exists():
            raise FileNotFoundError(f"adapter missing: {args.adapter}")
        t_load = time.perf_counter()
        extractor.load()
        if not extractor.is_model_loaded():
            raise RuntimeError(f"model not loaded: {extractor.load_error}")
        seg["load_seconds"] = time.perf_counter() - t_load
        seg["alloc_after_load_gb"] = torch.cuda.memory_allocated() / 1e9
        seg["decoding"] = {"do_sample": False, "max_new_tokens": extractor.max_new_tokens, "use_4bit": extractor.use_4bit,
                           "prompt_version": extractor.prompt_version, "record_logprobs": extractor.record_logprobs,
                           "model_path": extractor.model_path, "adapter_path": args.adapter, "batch_size": 1}
        print(f"model loaded in {seg['load_seconds']:.0f}s alloc={seg['alloc_after_load_gb']:.1f} GB", flush=True)

        captured: Dict[str, Any] = {}
        orig_lp, orig_plain = extractor._generate_with_logprobs, extractor._generate

        def cap_lp(text: str):  # noqa: ANN202 - capture raw output + token count, behaviour unchanged
            raw, info = orig_lp(text)
            captured["raw"], captured["n_gen"] = raw, len(info["ids"])
            return raw, info

        def cap_plain(text: str):  # noqa: ANN202
            raw = orig_plain(text)
            captured["raw"], captured["n_gen"] = raw, None
            return raw

        extractor._generate_with_logprobs, extractor._generate = cap_lp, cap_plain
        tok = extractor._tokenizer
        torch.cuda.reset_peak_memory_stats()
        job_peak_alloc, job_peak_reserved = 0.0, 0.0
        t0 = time.perf_counter()
        with pred_path.open("a", encoding="utf-8") as fh:
            for i, doc in enumerate(todo):
                captured.clear()
                prompt = tok.apply_chat_template(extractor.build_messages(doc["text"]), tokenize=False, add_generation_prompt=True)
                n_prompt = len(tok(prompt)["input_ids"])
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t_doc = time.perf_counter()
                status, error = "ok", None
                try:
                    preds = extractor.extract(doc["text"])
                except ValueError as exc:  # unparseable JSON from the model -> no spans, logged (original path would crash)
                    preds, status, error = [], "parse_error", str(exc)[:300]
                torch.cuda.synchronize()
                latency = time.perf_counter() - t_doc
                pa, pr = torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9
                job_peak_alloc, job_peak_reserved = max(job_peak_alloc, pa), max(job_peak_reserved, pr)
                row = {"id": doc["id"], "text": doc["text"], "predictions": preds, "status": status, "error": error,
                       "gen_seconds": latency, "latency_seconds": latency, "n_prompt_tokens": n_prompt,
                       "n_gen_tokens": captured.get("n_gen"), "raw_output": captured.get("raw"),
                       "peak_alloc_gb_doc": pa, "peak_reserved_gb_doc": pr, "meta": doc.get("meta", {}),
                       "gpu_snapshot": gpu_snapshot(os.getpid()) if i % GPU_SAMPLE_EVERY == 0 else None}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done[doc["id"]] = row
                if i % 25 == 0:
                    el = time.perf_counter() - t0
                    print(f"{utc()} [{i+1}/{len(todo)}] {el:.0f}s elapsed {el/(i+1):.2f}s/doc status={status}", flush=True)
        seg["gen_wall_seconds"] = time.perf_counter() - t0
        seg["docs_generated"] = len(todo)
        seg["peak_alloc_gb"], seg["peak_reserved_gb"] = job_peak_alloc, job_peak_reserved
    seg["end_utc"] = utc()
    seg["gpu_end"] = gpu_snapshot(os.getpid())
    with (out / "timing_segments.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(seg) + "\n")
    rows = [done[d["id"]] for d in docs if d["id"] in done]
    if len(rows) != len(docs):
        raise RuntimeError(f"incomplete: {len(rows)}/{len(docs)}")
    # rewrite in input order (dedup) so the file is canonical, then mark DONE
    tmp = pred_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(pred_path)
    write_done(args, out, rows, [json.loads(x) for x in (out / "timing_segments.jsonl").read_text().splitlines() if x.strip()])
    if args.num_shards > 1:
        merge(args)


def write_done(args: argparse.Namespace, out: Path, rows: List[Dict[str, Any]], segments: List[Dict[str, Any]],
               extra: Optional[Dict[str, Any]] = None) -> None:
    adapter_st = Path(args.adapter) / "adapter_model.safetensors"
    payload = {
        "job": args.job, "status": "complete", "completed_utc": utc(), "smoke": bool(args.smoke),
        "input": ({"input": args.input, "input_sha256": sha256_file(Path(args.input))} if args.input
                  else {"split": args.split, "subset": args.subset, "dataset": "data/source/skillspan_hf",
                        "split_sha256": sha256_file(Path("data/source/skillspan_hf") / f"{args.split}.json")}),
        "shard": {"shard": args.shard, "num_shards": args.num_shards}, "limit": args.limit,
        "summary": summarise(rows),
        "segments": segments,
        "outputs": {"predictions.jsonl": sha256_file(out / "predictions.jsonl")},
        "adapter": {"path": args.adapter, "adapter_model.safetensors_sha256": sha256_file(adapter_st) if adapter_st.exists() else None},
        "config": {"path": args.config, "sha256": sha256_file(Path(args.config))},
        "command": " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTORCH_CUDA_ALLOC_CONF", "PYTHONPATH")},
        "versions": lib_versions(), "git": git_info(),
        "process_peak_rss_gb": _peak_rss_gb(), **(extra or {}),
    }
    (out / "DONE.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{utc()} wrote {out / 'DONE.json'}", flush=True)


def merge(args: argparse.Namespace) -> None:
    """Merge shard outputs into <job>/predictions.jsonl in input order once every shard has a DONE.json."""
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    base = job_dir(args)
    base.mkdir(parents=True, exist_ok=True)
    with open(base / ".merge.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (base / "DONE.json").exists() and not args.force:
            print("merged DONE.json exists", flush=True)
            return
        shard_dirs = [base / "shards" / f"shard{k}of{args.num_shards}" for k in range(args.num_shards)]
        missing = [str(d) for d in shard_dirs if not (d / "DONE.json").exists()]
        if missing:
            print(f"merge deferred; shards not done: {missing}", flush=True)
            return
        docs = load_docs(args)
        by_id: Dict[str, Dict[str, Any]] = {}
        segments: List[Dict[str, Any]] = []
        for k, d in enumerate(shard_dirs):
            by_id.update(read_done_rows(d / "predictions.jsonl"))
            segments += [{**json.loads(x), "shard": k} for x in (d / "timing_segments.jsonl").read_text().splitlines() if x.strip()]
        rows = [by_id[d["id"]] for d in docs]
        with (base / "predictions.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        shard_done = {f"shard{k}": json.loads((d / "DONE.json").read_text())["outputs"] for k, d in enumerate(shard_dirs)}
        a = argparse.Namespace(**{**vars(args), "shard": None})
        write_done(a, base, rows, segments, {"merged_from_shards": shard_done})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--subset", default="gold", choices=["gold", "nogold", "all"])
    ap.add_argument("--input", help="custom jsonl with id/text(/meta) rows (overrides --split/--subset)")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--logprobs", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge", action="store_true", help="only merge finished shards")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.limit and not args.smoke:
        ap.error("--limit is only allowed with --smoke (partial runs are never results)")
    if args.merge:
        merge(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
