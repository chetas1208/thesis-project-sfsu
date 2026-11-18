"""Tests for the pure helpers of A3's GPU driver scripts (sharding, resumability, summaries, BERT decoding)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/evals" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def infer():
    return load("gpu_anchor_infer")


def test_shards_partition_docs(infer):
    docs = [{"id": f"d{i}"} for i in range(11)]
    shards = [infer.shard_docs(docs, k, 3) for k in range(3)]
    ids = sorted(d["id"] for s in shards for d in s)
    assert ids == sorted(d["id"] for d in docs) and len({d["id"] for s in shards for d in s}) == 11
    assert abs(len(shards[0]) - len(shards[2])) <= 1


def test_read_done_rows_tolerates_truncated_tail(infer, tmp_path):
    p = tmp_path / "predictions.jsonl"
    p.write_text(json.dumps({"id": "a", "predictions": []}) + "\n" + '{"id": "b", "predic')
    done = infer.read_done_rows(p)
    assert list(done) == ["a"]
    assert infer.read_done_rows(tmp_path / "missing.jsonl") == {}


def test_summarise_counts_and_percentiles(infer):
    rows = [{"id": "a", "predictions": [{}, {}], "latency_seconds": 2.0, "gen_seconds": 2.0, "n_gen_tokens": 10, "status": "ok",
             "peak_alloc_gb_doc": 16.0, "gpu_snapshot": {"other_compute_pids": []}},
            {"id": "b", "predictions": [], "latency_seconds": 4.0, "gen_seconds": 4.0, "n_gen_tokens": 4, "status": "parse_error",
             "peak_alloc_gb_doc": 16.5, "gpu_snapshot": None}]
    s = infer.summarise(rows)
    assert s["n_docs"] == 2 and s["status_counts"] == {"ok": 1, "parse_error": 1}
    assert s["n_predicted_spans"] == 2 and s["latency_seconds_per_doc"]["mean"] == 3.0
    assert s["latency_seconds_per_predicted_span"]["n"] == 1 and s["peak_alloc_gb_doc_max"] == 16.5
    assert s["gpu_snapshots"] == {"sampled_docs": 1, "sampled_docs_with_other_compute_procs": 0}


def test_bert_fair_decoding_matches_gold_convention():
    bf = load("gpu_bert_fair")
    text, offs = bf.word_offsets(["Strong", "Python", "and", "team", "work"])
    assert text == "Strong Python and team work"
    spans = bf.spans_from_word_tags(text, offs, ["O", "B", "O", "B", "I"], "HARD_SKILL", [1, 0.9, 1, 0.8, 0.6])
    assert [(s["surface_form"], s["start"], s["end"]) for s in spans] == [("Python", 7, 13), ("team work", 18, 27)]
    assert spans[1]["confidence"] == pytest.approx(0.7)
    # an I without a preceding B opens a span, exactly like the gold builder
    spans = bf.spans_from_word_tags(text, offs, ["O", "I", "I", "O", "O"], "SOFT_SKILL")
    assert [s["surface_form"] for s in spans] == ["Python and"]
    assert bf.norm_tag("B-SKILL") == "B" and bf.norm_tag("I-KNOWLEDGE") == "I" and bf.norm_tag("O") == "O"
