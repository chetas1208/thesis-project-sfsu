"""Tests for the prompt_version option, extra (pseudo-label) training rows and per-span logprob capture."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from skillllm_v2.extract import mistral_extractor as me
from skillllm_v2.schema import SkillCandidate
from skillllm_v2.train import loop_trainer as lt
from skillllm_v2.train.build_mistral_sft_data import build_mistral_sft_data


# ---------------- prompt_version ----------------
def test_default_prompt_is_the_original_and_unchanged() -> None:
    msgs = me.build_prompt_messages("hello world")
    assert msgs == [
        {"role": "system", "content": me.SYSTEM_PROMPT},
        {"role": "user", "content": me.USER_PROMPT_TEMPLATE.format(text="hello world")},
    ]
    assert me.MistralExtractor().build_messages("hello world") == msgs
    assert me.MistralExtractor().prompt_version == "original"


def test_short_prompt_keeps_schema_and_labels_and_is_much_shorter() -> None:
    short = me.build_prompt_messages("hello world", "short")
    full = me.build_prompt_messages("hello world")
    joined = "\n".join(m["content"] for m in short)
    assert "HARD_SKILL" in joined and "SOFT_SKILL" in joined and '"skills"' in joined and '"text"' in joined and '"label"' in joined
    assert joined.endswith("hello world")
    assert sum(len(m["content"]) for m in short) < 0.25 * sum(len(m["content"]) for m in full)
    assert me.MistralExtractor(prompt_version="short").build_messages("x") == me.build_prompt_messages("x", "short")


def test_unknown_prompt_version_rejected() -> None:
    with pytest.raises(ValueError):
        me.build_prompt_messages("x", "nope")
    with pytest.raises(ValueError):
        me.MistralExtractor(prompt_version="nope")


def test_sft_builder_prompt_version_tag(tmp_path: Path) -> None:
    default = build_mistral_sft_data("data/source/skillspan_hf", tmp_path / "a", limit=2)
    short = build_mistral_sft_data("data/source/skillspan_hf", tmp_path / "b", limit=2, prompt_version="short")
    import json

    assert "prompt_version" not in json.loads(Path(default["train"]).read_text().splitlines()[0])
    assert json.loads(Path(short["train"]).read_text().splitlines()[0])["prompt_version"] == "short"
    with pytest.raises(ValueError):
        build_mistral_sft_data("data/source/skillspan_hf", tmp_path / "c", limit=2, prompt_version="nope")


class FakeTok:
    """Whitespace 'tokenizer' with Mistral-like [INST] wrapping, enough for the token-breakdown helper."""

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return "<s> " + " ".join(m["content"] for m in messages) + " [/INST]"

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


def test_measure_example_tokens_short_is_smaller_and_default_is_original() -> None:
    rows = [{"id": "a", "input": "python and sql", "output": {"skills": [{"surface_form": "python", "label": "HARD_SKILL"}]}}]
    full = lt.measure_example_tokens(FakeTok(), rows)
    short = lt.measure_example_tokens(FakeTok(), rows, "short")
    assert full["prompt_version"] == "original" and short["prompt_version"] == "short"
    assert short["fixed_prompt_tokens"] < full["fixed_prompt_tokens"]
    assert short["mean_total_tokens"] < full["mean_total_tokens"]
    assert full["fixed_prompt_share"] > short["fixed_prompt_share"]


# ---------------- extra (pseudo-label) rows in the loop ----------------
def _row(doc_id: str, with_skill: bool) -> dict[str, Any]:
    spans = [{"surface_form": "x", "start": 0, "end": 4, "label": "HARD_SKILL"}] if with_skill else []
    return {"id": doc_id, "input": f"text {doc_id}", "output": {"skills": spans}}


def test_pick_extra_rows_cycles_without_replacement() -> None:
    extra = [_row(f"ps{i}", True) for i in range(10)]
    seen: set[str] = set()
    got: list[str] = []
    for r in range(2):
        rows, seen, wraps = lt.pick_extra_rows(extra, seen, 5, r, 1)
        got += [x["id"] for x in rows]
        assert wraps == 0
    assert len(set(got)) == 10  # two rounds of 5 cover all 10 exactly once
    rows, seen, wraps = lt.pick_extra_rows(extra, seen, 5, 2, 1)
    assert wraps == 1 and len(rows) == 5
    assert lt.pick_extra_rows(extra, set(), 0, 0, 1)[0] == [] and lt.pick_extra_rows([], set(), 5, 0, 1)[0] == []


class _Backend:
    def __init__(self) -> None:
        self.mixes: list[list[str]] = []

    def train_round(self, rows, lr, round_idx, budget, seed):
        self.mixes.append([r["id"] for r in rows])
        budget.charge_tokens(10)
        return {"tokens": 10, "examples": len(rows)}

    def predict(self, rows, budget):
        return [[dict(s, label=s.get("label")) for s in r["output"]["skills"]] for r in rows]

    def save_adapter(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.bin").write_text("x")


def test_run_loop_mixes_extra_rows_by_extra_frac(tmp_path: Path) -> None:
    train = [_row(f"p{i:03d}", True) for i in range(40)] + [_row(f"e{i:03d}", False) for i in range(60)]
    dev = [_row(f"d{i:03d}", True) for i in range(20)]
    extra = [_row(f"pseudo-{i:03d}", i % 4 != 0) for i in range(80)]
    cfg = {
        "seed": 1,
        "output": {"dir": str(tmp_path / "out")},
        "loop": {"max_rounds": 2, "examples_per_round": 20, "extra_frac": 0.5, "patience": 5},
        "mining": {"probe_size": 8, "recheck_max": 4},
        "eval": {"dev_size": 10},
        "budget": {"max_gpu_hours": 10},
    }
    backend = _Backend()
    summary = lt.run_loop(cfg, backend, train, dev, extra_rows=extra)
    assert summary["completed_rounds"] == 2
    for mix in backend.mixes:
        assert len(mix) == 20
        assert sum(i.startswith("pseudo-") for i in mix) == 10  # 1:1 gold:pseudo
    ids = [i for m in backend.mixes for i in m if i.startswith("pseudo-")]
    assert len(ids) == len(set(ids))  # no pseudo doc repeated across the 2 rounds
    state = lt.LoopState.load(Path(cfg["output"]["dir"]) / "loop_state.json")
    assert state.extra_seen_ids and state.history[0]["mix"]["extra_examples"] == 10
    # probes/mining never touch pseudo rows
    assert not any(k.startswith("pseudo-") for k in state.hard_pool)


def test_run_loop_without_extra_is_unchanged(tmp_path: Path) -> None:
    train = [_row(f"p{i:03d}", True) for i in range(40)] + [_row(f"e{i:03d}", False) for i in range(60)]
    dev = [_row(f"d{i:03d}", True) for i in range(20)]
    cfg = {"seed": 1, "output": {"dir": str(tmp_path / "out")}, "loop": {"max_rounds": 1, "examples_per_round": 20, "extra_frac": 0.5}, "mining": {"probe_size": 8, "recheck_max": 0}, "eval": {"dev_size": 10}, "budget": {"max_gpu_hours": 10}}
    backend = _Backend()
    lt.run_loop(cfg, backend, train, dev)  # extra_frac set but no extra rows -> ignored
    assert len(backend.mixes[0]) == 20 and not any(i.startswith("pseudo-") for i in backend.mixes[0])


# ---------------- per-span logprobs ----------------
RAW = '{"skills":[{"text":"Python","label":"HARD_SKILL"},{"text":"team \\"player\\"","label":"SOFT_SKILL"}]}'


class CharTok:
    """One token per `size` characters; decode returns the corresponding prefix of RAW."""

    def __init__(self, raw: str, size: int = 4) -> None:
        self.raw, self.size = raw, size

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.raw[: len(ids) * self.size]


def test_span_value_char_ranges_locates_values_with_escapes() -> None:
    ranges = me.span_value_char_ranges(RAW)
    assert [v for _, _, v in ranges] == ["Python", 'team "player"']
    s, e, _ = ranges[0]
    assert RAW[s:e] == "Python"


def test_span_logprob_stats_uses_only_overlapping_tokens() -> None:
    tok = CharTok(RAW, 4)
    n_tok = -(-len(RAW) // 4)
    ids = list(range(n_tok))
    ends = me.token_char_ends(tok, ids)
    assert ends[-1] == len(RAW) and ends == sorted(ends)
    lps = [-0.1 * (i + 1) for i in ids]
    s, e, _ = me.span_value_char_ranges(RAW)[0]
    stats = me.span_token_logprob_stats(lps, ends, s, e)
    picked = [i for i in ids if i * 4 < e and (i + 1) * 4 > s]
    assert stats["n_tokens"] == len(picked)
    assert stats["sum"] == pytest.approx(sum(lps[i] for i in picked))
    assert stats["min"] == min(lps[i] for i in picked)
    assert stats["mean"] == pytest.approx(stats["sum"] / stats["n_tokens"])


def test_attach_span_logprobs_aligned_and_misaligned() -> None:
    tok = CharTok(RAW, 4)
    n_tok = -(-len(RAW) // 4)
    info = {"ids": list(range(n_tok)), "logprobs": [-0.5] * n_tok}
    skills = [
        SkillCandidate("Python", "HARD_SKILL", 0, 6, "mistral", 0.86, "Python"),
        SkillCandidate('team "player"', "SOFT_SKILL", None, None, "mistral", 0.86, 'team "player"'),
    ]
    me.attach_span_logprobs(skills, RAW, info, tok)
    assert skills[0].metadata["logprob"]["n_tokens"] >= 2
    assert skills[0].metadata["logprob"]["mean"] == pytest.approx(-0.5)
    assert skills[1].metadata["logprob"] is not None
    # a skill list that disagrees with the raw output must NOT get guessed numbers
    wrong = [SkillCandidate("Java", "HARD_SKILL", None, None, "mistral", 0.86, "Java")]
    me.attach_span_logprobs(wrong, RAW, info, tok)
    assert wrong[0].metadata["logprob"] is None
