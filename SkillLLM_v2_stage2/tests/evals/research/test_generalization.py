"""Tests for skillllm_v2.evals.research.generalization (tiny synthetic fixtures)."""

import numpy as np
import pytest

from skillllm_v2.evals.research import generalization as G


@pytest.mark.parametrize("raw,norm", [
    ("Skills", "skill"), ("processes", "process"), ("abilities", "ability"), ("approaches", "approach"),
    ("C++", "c++"), ("C#", "c#"), (".NET", ".net"), ("analysis", "analysis"), ("status", "status"),
    (" (Python) ", "python"), ("teamwork.", "teamwork"), ("problem  solving skills", "problem solving skill"),
    ("CSS", "css"), ("AWS", "aws"), ("is", "is"),
])
def test_norm_surface(raw, norm):
    assert G.norm_surface(raw) == norm


def test_stratum_edges():
    assert [G.stratum(c) for c in (0, 1, 2, 5, 6)] == ["0", "1", "2-5", "2-5", ">5"]


def _bucketer():
    train = [("Python", "HARD_SKILL"), ("problem solving", "SOFT_SKILL"), ("REST APIs", "HARD_SKILL"),
             ("Python", "HARD_SKILL")]
    alias = {"problem solving": {"alias:ps"}, "rest api": {"alias:rest"}, "rest apis": {"alias:rest"}}
    return G.Bucketer(train, alias, {})


def test_bucketer_s1_to_s4():
    B = _bucketer()
    assert B.bucket("Python") == ("S1", "")
    assert B.bucket("python") == ("S2", "")
    # documented limitation of the plural rule (D-A1-06): "apis" ends in "is" and is kept, so only the alias table
    # links "REST API" to the train surface "REST APIs"
    assert G.norm_surface("REST APIs") == "rest apis"
    assert B.bucket("REST API") == ("S3", "alias")
    assert B.bucket("problem-solving") == ("S3", "alias")  # norm keeps '-', normalize_text maps it to the alias
    assert B.bucket("Rust") == ("S4", "")
    assert B.freq("python") == "2-5" and B.freq("Rust") == "0"


def test_doc_rows_buckets_and_tp_alignment():
    B = _bucketer()
    text = "Python and Rust and problem-solving"
    gold = [{"start": 0, "end": 6, "label": "HARD_SKILL"}, {"start": 11, "end": 15, "label": "HARD_SKILL"},
            {"start": 20, "end": 35, "label": "SOFT_SKILL"}]
    pred = [{"start": 0, "end": 6, "label": "HARD_SKILL"}, {"start": 11, "end": 15, "label": "SOFT_SKILL"},
            {"start": 0, "end": 6, "label": "HARD_SKILL"}]
    row = G.doc_rows(gold, pred, text, B, {})
    s1 = G.OFF_BUCKET + G.BUCKETS.index("S1") * G.COLS_PER
    s3 = G.OFF_BUCKET + G.BUCKETS.index("S3") * G.COLS_PER
    s4 = G.OFF_BUCKET + G.BUCKETS.index("S4") * G.COLS_PER
    assert row[s1:s1 + 4].tolist() == [1, 1, 1, 1]  # Python: gold hit, one de-duplicated pred, TP
    assert row[s4:s4 + 4].tolist() == [1, 0, 1, 0]  # Rust: wrong label -> no TP on either side
    assert row[s3:s3 + 4].tolist() == [1, 0, 0, 0]
    S = row[None, :]
    p, r, f = G.prf_from(S, G.bucket_offsets(["S1", "S2"]))
    assert float(r[0]) == 1.0 and float(p[0]) == 1.0
    assert float(G.STATS["unseen_recall"](S)[0]) == 0.0


def test_overlap_stats_jaccard():
    train = [("Python", "HARD_SKILL"), ("Java", "HARD_SKILL")]
    test = [("python", "HARD_SKILL"), ("Go", "HARD_SKILL")]
    ov = G.overlap_stats(train, test)
    assert ov["mention_share_exact_seen"] == 0.0
    assert ov["mention_share_normalized_seen"] == 0.5
    assert ov["type_jaccard_normalized"] == pytest.approx(1 / 3)


def test_p_two_sided():
    assert G._p_two_sided(np.array([1.0] * 99)) == pytest.approx(2 / 100)
    assert G._p_two_sided(np.array([-1.0, 1.0] * 50)) == 1.0


def test_len_band_and_length_rows():
    assert [G.len_band(s) for s in ("Python", "machine learning", "build REST APIs fast")] == ["1", "2", "3+"]
    B = _bucketer()
    text = "Python and Rust"
    gold = [{"start": 0, "end": 6, "label": "HARD_SKILL"}, {"start": 11, "end": 15, "label": "HARD_SKILL"}]
    row = G.length_rows(gold, [{"start": 0, "end": 6, "label": "HARD_SKILL"}], text, B, {})
    # seen/1-word: n=1, tp=1 ; unseen/1-word: n=1, tp=0
    assert row[0:2].tolist() == [1, 1]
    assert row[len(G.LEN_BANDS) * 2: len(G.LEN_BANDS) * 2 + 2].tolist() == [1, 0]
