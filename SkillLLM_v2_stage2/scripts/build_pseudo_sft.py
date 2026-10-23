"""Experiment 2, step 2 (CPU): majority-vote the three adapters' chunk predictions into pseudo-labelled SFT rows.

Rule (pre-declared, same as scripts/ensemble_eval.py): keep a span (start, end, label) iff >= 2 of the 3 systems (anchor, v1, v2) predict it.
Chunk inclusion: >= 1 voted span -> positive; no adapter predicted any span -> confident negative; otherwise (only 1-vote spans) -> uncertain,
excluded. A chunk where any adapter's output was not parseable JSON is excluded. Confident negatives are subsampled so they are <= 25% of the
pool. Output rows live in artifacts (private text); counts-only stats go to reports/skillllm_v2/qlora_loop_v4_pseudo_counts.json.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

POOL = Path("artifacts/skillllm_v2/qlora_loop_v4_distill/pseudo_pool")
ADAPTERS = ["anchor", "v1", "v2"]
NEG_MAX_SHARE = 0.25


def load_preds(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def vote_spans(per_system: list[list[dict]], min_votes: int = 2) -> tuple[list[dict], Counter, int]:
    """Exact (start, end, label) voting. Returns (kept spans sorted, vote histogram over distinct spans, #overlap conflicts dropped)."""
    votes: dict[tuple, list[dict]] = {}
    for spans in per_system:
        seen = set()
        for sp in spans:
            if sp.get("start") is None or sp.get("end") is None:
                continue
            key = (sp["start"], sp["end"], sp.get("label"))
            if key in seen:
                continue
            seen.add(key)
            votes.setdefault(key, []).append(sp)
    hist = Counter(len(v) for v in votes.values())
    kept = [(k, v) for k, v in votes.items() if len(v) >= min_votes]
    kept.sort(key=lambda kv: (-len(kv[1]), -(kv[0][1] - kv[0][0]), kv[0][0]))
    chosen: list[tuple] = []
    conflicts = 0
    for key, sps in kept:
        if any(key[0] < c[0][1] and c[0][0] < key[1] for c in chosen):
            conflicts += 1
            continue
        chosen.append((key, sps))
    rows = [
        {"surface_form": sps[0]["surface_form"], "label": key[2], "start": key[0], "end": key[1], "votes": len(sps)}
        for key, sps in sorted(chosen, key=lambda kv: (kv[0][0], kv[0][1]))
    ]
    return rows, hist, conflicts


def main() -> None:
    chunks = {c["id"]: c for c in (json.loads(l) for l in (POOL / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    preds = {a: load_preds(POOL / f"preds_{a}.jsonl") for a in ADAPTERS}
    ids = [i for i in chunks if all(i in preds[a] for a in ADAPTERS)]
    stats: Counter = Counter()
    hist_total: Counter = Counter()
    per_adapter_spans = {a: 0 for a in ADAPTERS}
    positives, negatives = [], []
    for i in ids:
        if not all(preds[a][i]["parse_ok"] for a in ADAPTERS):
            stats["excluded_parse_failure"] += 1
            continue
        per_sys = [preds[a][i]["predictions"] for a in ADAPTERS]
        for a, spans in zip(ADAPTERS, per_sys):
            per_adapter_spans[a] += len(spans)
        kept, hist, conflicts = vote_spans(per_sys)
        hist_total.update(hist)
        stats["overlap_conflicts_dropped"] += conflicts
        any_pred = any(per_sys)
        row = {"id": f"pseudo-{i}", "input": chunks[i]["text"], "output": {"skills": kept}}
        if kept:
            positives.append(row)
            if hist.get(1, 0):
                stats["positive_chunks_with_dropped_1vote_spans"] += 1  # recall-noise proxy: real skills may be missing from the target
        elif not any_pred:
            negatives.append(row)
        else:
            stats["excluded_uncertain_only_1vote"] += 1
    rng = random.Random(20261022)
    max_neg = int(len(positives) * NEG_MAX_SHARE / (1 - NEG_MAX_SHARE))
    neg_used = rng.sample(negatives, min(len(negatives), max_neg))
    pool = positives + neg_used
    rng.shuffle(pool)
    out = POOL / "pseudo_train.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in pool:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels = Counter(s["label"] for r in positives for s in r["output"]["skills"])
    votes_kept = Counter(s["votes"] for r in positives for s in r["output"]["skills"])
    summary = {
        "chunks_in_pool": len(chunks),
        "chunks_with_all_three_adapters": len(ids),
        **{k: v for k, v in stats.items()},
        "positive_chunks": len(positives),
        "confident_negative_chunks_available": len(negatives),
        "confident_negative_chunks_used": len(neg_used),
        "pseudo_pool_rows": len(pool),
        "pseudo_pool_positive_share": round(len(positives) / max(1, len(pool)), 4),
        "spans_predicted_per_adapter": per_adapter_spans,
        "distinct_span_vote_histogram": {str(k): v for k, v in sorted(hist_total.items())},
        "kept_spans": sum(labels.values()),
        "kept_spans_by_label": dict(labels),
        "kept_spans_by_votes": {str(k): v for k, v in sorted(votes_kept.items())},
        "neg_max_share": NEG_MAX_SHARE,
        "note": "counts only; chunk text is private (artifacts dir, outside git)",
    }
    Path("reports/skillllm_v2/qlora_loop_v4_pseudo_counts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
