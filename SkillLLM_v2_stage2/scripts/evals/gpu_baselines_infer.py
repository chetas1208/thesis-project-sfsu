"""Run the non-LLM systems on arbitrary inputs (perturbed docs, cross-dataset docs, the test split) for L03/L12 (A3).

Systems (each writes artifacts/skillllm_v2/evals_gpu/<job>_<system>/{predictions.jsonl, DONE.json}):
  gliner_zeroshot  GLiNERExtractor() exactly as the stored GLiNER baseline / wrapper use it (urchade/gliner_medium-v2.1,
                   labels "hard skill"/"soft skill", threshold 0.35, zero-shot)
  token_lookup     the repository's "BERT baseline" = token-lookup baseline (memorised BIO tag per lower-cased train
                   token; BertBaseline.load() + the same _labels_to_spans), NOT a neural BERT
  bert_fair        the dev-selected fair BERT-family tagger from scripts/evals/gpu_bert_fair.py (bert_fair/SELECTED.json)
  safe_wrapper     recomposition of the stored safe wrapper's accepted ``full_cag_graph`` variant
                   (skillllm_v2.eval.run_safe_wrapper): anchor + GLiNER + dictionary -> CandidateMerger -> BoundaryRefiner
                   -> SpanVerifier -> ConfidenceCalibrator -> TaxonomyLinker, applied to ANCHOR predictions given by
                   --anchor-predictions (no LLM inference here). Per-stage wall time is recorded per doc.
Input rows: {"id","text","tokens","token_offsets"} (perturbed / cross-dataset docs) or --split test --subset gold|all.
Per-doc latency is recorded for L12 (batch size 1, CPU unless --device cuda).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
OUT_ROOT = REPO_ROOT / "artifacts/skillllm_v2/evals_gpu"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.input:
        rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
        return [{"id": r["id"], "text": r["text"], "tokens": r["tokens"], "offsets": [tuple(o) for o in r["token_offsets"]]} for r in rows]
    from skillllm_v2.train.build_verifier_dataset import _tokens_to_text, load_skillspan_split
    docs = load_skillspan_split(REPO_ROOT / "data/source/skillspan_hf", args.split, limit=None)
    if args.subset == "gold":
        docs = [d for d in docs if d["gold_spans"]]
    return [{"id": d["id"], "text": d["text"], "tokens": d["tokens"], "offsets": _tokens_to_text(d["tokens"])[1]} for d in docs]


def pct(values: List[float]) -> Dict[str, Any]:
    import numpy as np
    if not values:
        return {"n": 0}
    a = np.asarray(values, float)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "sum": float(a.sum())}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT.parent), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def write_out(job: str, system: str, rows: List[Dict[str, Any]], extra: Dict[str, Any], args: argparse.Namespace) -> None:
    out = OUT_ROOT / ("smoke" if args.smoke else "") / f"{job}_{system}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lat = [r["latency_seconds"] for r in rows]
    per_span = [r["latency_seconds"] / len(r["predictions"]) for r in rows if r["predictions"]]
    done = {"job": job, "system": system, "status": "complete", "completed_utc": utc(), "n_docs": len(rows),
            "n_predicted_spans": sum(len(r["predictions"]) for r in rows), "latency_seconds_per_doc": pct(lat),
            "latency_seconds_per_predicted_span": pct(per_span),
            "input": {"input": args.input, "sha256": sha256_file(Path(args.input))} if args.input else {"split": args.split, "subset": args.subset},
            "outputs": {"predictions.jsonl": sha256_file(out / "predictions.jsonl")},
            "command": " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "OMP_NUM_THREADS")},
            "host_loadavg_1m_end": os.getloadavg()[0], "host_ncpu": os.cpu_count(), "git_commit": git_commit(),
            "process_peak_rss_gb_so_far": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024 / 1e9,
            **extra}
    (out / "DONE.json").write_text(json.dumps(done, indent=2) + "\n")
    print(f"{utc()} {system}: {len(rows)} docs -> {out}", flush=True)


def run_gliner(docs, args):  # noqa: ANN001, ANN201
    from skillllm_v2.extract import GLiNERExtractor
    ex = GLiNERExtractor(device=args.device)
    t0 = time.perf_counter()
    ex.load()
    load_s = time.perf_counter() - t0
    if not ex.is_model_loaded():
        raise RuntimeError("GLiNER not loaded (would silently fall back)")
    rows = []
    for d in docs:
        t = time.perf_counter()
        preds = ex.extract(d["text"])
        rows.append({"id": d["id"], "text": d["text"], "predictions": preds, "latency_seconds": time.perf_counter() - t})
    n_params = sum(p.numel() for p in ex._model.parameters()) if hasattr(ex._model, "parameters") else None
    return rows, {"model": ex.model_name, "threshold": ex.threshold, "device": ex.device, "load_seconds": load_s, "n_params": n_params}


def run_token_lookup(docs, args):  # noqa: ANN001, ANN201
    from skillllm_v2.baselines.bert_baseline import BertBaseline, _labels_to_spans
    m = BertBaseline()
    m.load()
    rows = []
    for d in docs:
        t = time.perf_counter()
        labels = [m.token_label.get(tok.lower(), "O") for tok in d["tokens"]]
        preds = _labels_to_spans(d["text"], d["tokens"], labels)
        rows.append({"id": d["id"], "text": d["text"], "predictions": preds, "latency_seconds": time.perf_counter() - t})
    return rows, {"model": "artifacts/skillllm_v2/bert_baseline/model/model.json (token-lookup baseline, memorised BIO tags)",
                  "model_sha256": sha256_file(REPO_ROOT / "artifacts/skillllm_v2/bert_baseline/model/model.json"),
                  "n_entries": len(m.token_label)}


def run_bert_fair(docs, args):  # noqa: ANN001, ANN201
    import importlib.util
    import torch
    from transformers import AutoTokenizer
    spec = importlib.util.spec_from_file_location("gpu_bert_fair", REPO_ROOT / "scripts/evals/gpu_bert_fair.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    run = Path(args.bert_run) if args.bert_run else REPO_ROOT / json.loads((OUT_ROOT / "bert_fair/SELECTED.json").read_text())["selected"]
    meta = json.loads((run / "DONE.json").read_text())
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(meta["model"])
    model = bf.TwoHeadTagger.build(meta["model"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.to(device)
    rows = []
    for d in docs:  # one doc at a time (batch size 1) so per-doc latency is comparable with the anchor
        t = time.perf_counter()
        feats, _ = bf.encode(tok, [d], with_labels=False)
        pred = bf.predict(model, feats, [d], 1, device, tok.pad_token_id)[0]
        pred["latency_seconds"] = time.perf_counter() - t
        rows.append(pred)
    return rows, {"run": str(run.relative_to(REPO_ROOT)), "model": meta["model"], "seed": meta["seed"], "device": str(device),
                  "n_params": meta.get("n_params"), "model_sha256": meta["outputs"]["model.pt"]}


def run_safe_wrapper(docs, args):  # noqa: ANN001, ANN201
    from skillllm_v2.extract import CandidateMerger, DictionaryExtractor, GLiNERExtractor
    from skillllm_v2.normalize import TaxonomyLinker
    from skillllm_v2.verify import BoundaryRefiner, ConfidenceCalibrator, SpanVerifier
    anchor = {}
    for line in Path(args.anchor_predictions).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            anchor[r["id"]] = r["predictions"]
    gliner = GLiNERExtractor(device=args.device)
    gliner.load()
    if not gliner.is_model_loaded():
        raise RuntimeError("GLiNER not loaded")
    dictionary, merger, refiner = DictionaryExtractor(), CandidateMerger(), BoundaryRefiner()
    verifier, calibrator, linker = SpanVerifier(), ConfidenceCalibrator(), TaxonomyLinker()
    rows, missing = [], 0
    for d in docs:
        if d["id"] not in anchor:
            missing += 1
            continue
        text, st = d["text"], {}
        t = time.perf_counter(); g = gliner.extract(text); st["gliner"] = time.perf_counter() - t
        t = time.perf_counter(); dc = dictionary.extract(text); st["dictionary"] = time.perf_counter() - t
        t = time.perf_counter(); merged = merger.merge(list(anchor[d["id"]]) + list(g) + list(dc)); st["merger"] = time.perf_counter() - t
        t = time.perf_counter(); refined = refiner.refine(text, merged); st["boundary_refiner"] = time.perf_counter() - t
        t = time.perf_counter(); verified = verifier.filter(refined); st["verifier"] = time.perf_counter() - t
        t = time.perf_counter(); calibrated = calibrator.calibrate(verified); st["calibrator"] = time.perf_counter() - t
        t = time.perf_counter(); linked = linker.link_candidates(calibrated); st["linker"] = time.perf_counter() - t
        rows.append({"id": d["id"], "text": text, "predictions": linked, "latency_seconds": sum(st.values()), "stage_seconds": st})
    stages = {k: pct([r["stage_seconds"][k] for r in rows]) for k in (rows[0]["stage_seconds"] if rows else {})}
    return rows, {"variant": "full_cag_graph (recomposed from skillllm_v2.eval.run_safe_wrapper; gate not applied)",
                  "anchor_predictions": args.anchor_predictions, "anchor_sha256": sha256_file(Path(args.anchor_predictions)),
                  "docs_without_anchor": missing, "stage_latency_seconds": stages,
                  "note": "latency excludes the anchor LLM call (anchor predictions are inputs)"}


RUNNERS = {"gliner_zeroshot": run_gliner, "token_lookup": run_token_lookup, "bert_fair": run_bert_fair, "safe_wrapper": run_safe_wrapper}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--input")
    ap.add_argument("--split", default="test")
    ap.add_argument("--subset", default="gold", choices=["gold", "all"])
    ap.add_argument("--systems", nargs="+", default=list(RUNNERS), choices=list(RUNNERS))
    ap.add_argument("--anchor-predictions")
    ap.add_argument("--bert-run")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.limit and not args.smoke:
        ap.error("--limit only with --smoke")
    os.chdir(REPO_ROOT)
    docs = load_inputs(args)[: args.limit] if args.limit else load_inputs(args)
    for system in args.systems:
        out = OUT_ROOT / ("smoke" if args.smoke else "") / f"{args.job}_{system}" / "DONE.json"
        if out.exists() and not args.force:
            print(f"skip {system}: {out} exists", flush=True)
            continue
        if system == "safe_wrapper" and not args.anchor_predictions:
            print("skip safe_wrapper: --anchor-predictions not given", flush=True)
            continue
        rows, extra = RUNNERS[system](docs, args)
        write_out(args.job, system, rows, extra, args)


if __name__ == "__main__":
    main()
