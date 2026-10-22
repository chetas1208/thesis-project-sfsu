from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from skillllm_v2.train import loop_trainer as lt


def span(start: int, end: int, label: str = "HARD_SKILL") -> dict[str, Any]:
    return {"surface_form": "x", "start": start, "end": end, "label": label}


def row(doc_id: str, spans: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"id": doc_id, "input": f"text {doc_id}", "output": {"skills": spans or []}}


def make_rows(n_pos: int = 30, n_empty: int = 70) -> list[dict[str, Any]]:
    return [row(f"p{i:03d}", [span(0, 4)]) for i in range(n_pos)] + [row(f"e{i:03d}") for i in range(n_empty)]


# ---------------- hard-example mining ----------------
def test_classify_errors_categories() -> None:
    gold = [span(0, 5), span(10, 15), span(20, 25, "SOFT_SKILL"), span(30, 35)]
    pred = [span(0, 5), span(10, 18), span(20, 25, "HARD_SKILL"), span(50, 55), span(None, None)]
    pred[4] = {"surface_form": "ghost", "start": None, "end": None, "label": "HARD_SKILL"}
    counts = lt.classify_errors(gold, pred)
    assert counts == {"exact": 1, "boundary": 1, "label": 1, "missed": 1, "spurious": 2}


def test_difficulty_and_mining_orders_hardest_first() -> None:
    rows = [row("a", [span(0, 3)]), row("b", [span(0, 3), span(5, 9)]), row("c", [span(0, 3)])]
    preds = [[span(0, 3)], [], [span(0, 3), span(10, 12)]]  # a perfect, b misses 2, c one FP
    mined = lt.mine_hard_examples(rows, preds)
    assert [m["id"] for m in mined] == ["b", "c"]
    assert mined[0]["score"] == 2.0 and mined[0]["errors"]["missed"] == 2
    assert mined[1]["errors"]["spurious"] == 1
    assert [m["id"] for m in lt.mine_hard_examples(rows, preds, top_k=1)] == ["b"]
    # boundary errors are weighted lower than a clean miss
    boundary_only = lt.difficulty_score({"boundary": 1})
    assert 0 < boundary_only < lt.difficulty_score({"missed": 1})


def test_hard_pool_retires_fixed_docs_and_caps_size() -> None:
    pool = {"a": {"score": 2.0, "errors": {}, "round": 0, "streak": 1}, "z": {"score": 1.0, "errors": {}, "round": 0, "streak": 1}}
    mined = [{"id": "a", "score": 3.0, "errors": {"missed": 3}}, {"id": "n", "score": 1.0, "errors": {"spurious": 1}}]
    new_pool, info = lt.update_hard_pool(pool, mined, probed_ids=["a", "n", "z"], round_idx=1, max_size=10)
    assert set(new_pool) == {"a", "n"}  # z probed and no longer wrong -> retired
    assert new_pool["a"]["streak"] == 2 and new_pool["a"]["score"] == 3.0
    assert info["fixed"] == 1
    # un-probed docs stay untouched
    kept, _ = lt.update_hard_pool(pool, [], probed_ids=["n"], round_idx=1)
    assert set(kept) == {"a", "z"}
    capped, info = lt.update_hard_pool({}, [{"id": str(i), "score": float(i), "errors": {}} for i in range(1, 6)], [str(i) for i in range(1, 6)], 0, max_size=2)
    assert set(capped) == {"5", "4"} and info["dropped"] == 3


def test_round_mix_upsamples_hard_and_tracks_seen() -> None:
    rows = make_rows()
    pool = {"p001": {"score": 3.0}, "p002": {"score": 1.0}, "gone": {"score": 2.0}}
    mix, seen, stats = lt.build_round_mix(rows, pool, set(), 0, 20, hard_frac=0.5, max_repeat=3, empty_frac=0.2, seed=1)
    assert len(mix) == 20
    assert stats["hard_examples"] == 4 and stats["hard_slots_available"] == 4  # 3 + 1 slots; unknown id ignored
    ids = [r["id"] for r in mix]
    assert ids.count("p001") >= 3
    assert stats["fresh_empty"] == round(16 * 0.2)
    assert len(seen) == 16
    again, seen2, _ = lt.build_round_mix(rows, pool, set(), 0, 20, 0.5, 3, 0.2, 1)
    assert [r["id"] for r in again] == ids and seen2 == seen  # deterministic
    nxt, seen3, _ = lt.build_round_mix(rows, {}, seen, 1, 20, 0.0, 3, 0.2, 1)
    assert not ({r["id"] for r in nxt} & seen)  # fresh docs are unseen ones
    assert seen < seen3


def test_round_mix_wraps_when_pool_exhausted() -> None:
    rows = make_rows(n_pos=10, n_empty=5)
    _, seen, _ = lt.build_round_mix(rows, {}, set(), 0, 8, 0.0, 1, 0.0, 3)
    _, seen2, stats = lt.build_round_mix(rows, {}, seen, 1, 8, 0.0, 1, 0.0, 3)
    assert stats["seen_wraps"] == 1 and stats["examples"] == 8


def test_probe_rechecks_hard_pool_and_dev_subset_is_fixed() -> None:
    rows = make_rows()
    pool = {"p001": {"score": 3.0}, "p002": {"score": 1.0}}
    probe = lt.select_probe_subset(rows, pool, 1, size=10, empty_frac=0.3, recheck_max=1, seed=5)
    ids = [r["id"] for r in probe]
    assert ids[0] == "p001" and len(ids) == len(set(ids)) == 10
    assert sum(1 for i in ids if i.startswith("e")) == round(9 * 0.3)
    dev_a = lt.select_dev_subset(rows, 12, seed=9)
    dev_b = lt.select_dev_subset(rows, 12, seed=9)
    assert [r["id"] for r in dev_a] == [r["id"] for r in dev_b]
    assert all(r["output"]["skills"] for r in dev_a)  # gold-bearing only by default


def test_curriculum_and_lr_schedules() -> None:
    assert lt.empty_frac_for_round(0, 5, 0.1, 0.3) == pytest.approx(0.1)
    assert lt.empty_frac_for_round(4, 5, 0.1, 0.3) == pytest.approx(0.3)
    assert lt.lr_for_round(1e-4, 0, 0.85) == pytest.approx(1e-4)
    assert lt.lr_for_round(1e-4, 50, 0.85) == pytest.approx(0.25e-4)
    scales = [lt.lr_scale(s, 20, 0.1, 0.1) for s in range(20)]
    assert scales[0] < scales[2] and scales[-1] < scales[5] and scales[-1] >= 0.1 - 1e-9
    batches = lt.make_batches([5, 90, 7, 80, 6, 85, 8, 95], micro_batch_size=2, seed=1, bucket_mult=4)
    assert sorted(i for b in batches for i in b) == list(range(8)) and all(len(b) <= 2 for b in batches)


# ---------------- budget / early stop ----------------
def test_budget_caps_and_pause_not_billed() -> None:
    b = lt.BudgetTracker(max_gpu_hours=1.0, max_train_tokens=1000, max_charge_seconds=60, eval_reserve_frac=0.1)
    b.charge_time(10_000)  # a 3 h SIGSTOP pause is billed as only one 60 s event
    assert b.gpu_seconds == 60
    assert b.exhausted() is None
    b.gpu_seconds = 3300  # 0.9167 h >= 90% of cap: no more training, eval still allowed
    assert b.train_exhausted() == "budget_gpu_hours" and b.exhausted() is None
    b.gpu_seconds = 3600
    assert b.exhausted() == "budget_gpu_hours"
    t = lt.BudgetTracker(max_gpu_hours=10, max_train_tokens=1000)
    t.charge_tokens(1001)
    assert t.exhausted() == "budget_tokens" and t.train_exhausted() == "budget_tokens"
    n = lt.BudgetTracker(max_gpu_hours=1.0, max_train_tokens=1000)
    n.gpu_seconds = 1800
    assert n.can_start_round(0.6, 0) == "budget_gpu_hours_next_round"
    assert n.can_start_round(0.4, 0) is None
    n.train_tokens = 700
    assert n.can_start_round(0, 400) == "budget_tokens_next_round"
    r = lt.BudgetTracker()
    r.restore(n.to_dict())
    assert r.gpu_seconds == 1800 and r.train_tokens == 700


def test_plateau_stopper_patience_and_best() -> None:
    s = lt.PlateauStopper(patience=2, min_delta=0.01)
    assert s.update(0, 0.50) is True and not s.should_stop
    assert s.update(1, 0.505) is True and s.since_improve == 1  # new best but below min_delta
    assert not s.should_stop
    assert s.update(2, 0.49) is False and s.should_stop
    assert s.best_round == 1 and s.best == pytest.approx(0.505)
    assert s.update(3, 0.60) is True and s.since_improve == 0 and not s.should_stop


# ---------------- state / resume / loop driver ----------------
def test_state_roundtrip_is_atomic_and_resume_path(tmp_path: Path) -> None:
    path = tmp_path / "loop_state.json"
    assert lt.LoopState.load(path).completed_rounds == 0
    assert lt.resume_adapter_path(tmp_path, lt.LoopState()) is None
    state = lt.LoopState(completed_rounds=2, best_round=1, best_f1=0.7, seen_ids=["a"], hard_pool={"a": {"score": 1.0}})
    state.save(path)
    assert not path.with_suffix(".json.tmp").exists()
    loaded = lt.LoopState.load(path)
    assert (loaded.completed_rounds, loaded.best_round, loaded.seen_ids) == (2, 1, ["a"])
    with pytest.raises(FileNotFoundError):
        lt.resume_adapter_path(tmp_path, loaded)
    adapter = lt.round_dir(tmp_path, 1) / "adapter"
    adapter.mkdir(parents=True)
    assert lt.resume_adapter_path(tmp_path, loaded) == adapter


class FakeBackend:
    """Deterministic stand-in: dev F1 follows a scripted curve, probe predictions miss every 2nd doc."""

    def __init__(self, curve: list[float], tokens_per_round: int = 100, hours_per_round: float = 0.0, solve_ids: set[str] | None = None) -> None:
        self.curve = curve
        self.tokens_per_round = tokens_per_round
        self.hours_per_round = hours_per_round
        self.round = 0
        self.trained_rounds: list[int] = []
        self.mixes: list[list[str]] = []
        self.dev_calls = 0
        self.solved = solve_ids or set()

    def train_round(self, rows, lr, round_idx, budget, seed):
        self.trained_rounds.append(round_idx)
        self.mixes.append([r["id"] for r in rows])
        budget.charge_tokens(self.tokens_per_round)
        budget.gpu_seconds += self.hours_per_round * 3600
        self.round = round_idx
        return {"tokens": self.tokens_per_round, "examples": len(rows)}

    def predict(self, rows, budget):
        out = []
        score = self.curve[min(self.round, len(self.curve) - 1)]
        for i, r in enumerate(rows):
            gold = r["output"]["skills"]
            if r["id"].startswith("dev"):
                # perfect on the first `score` fraction of dev docs, empty prediction elsewhere
                out.append(list(gold) if (i / max(1, len(rows))) < score else [])
            else:
                out.append(list(gold) if (i % 2 == 0 or r["id"] in self.solved) else [])
        return out

    def save_adapter(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.bin").write_text(f"round={self.round}")


def loop_config(tmp_path: Path, **loop: Any) -> dict[str, Any]:
    return {
        "seed": 3,
        "loop": {"max_rounds": 5, "examples_per_round": 20, "hard_frac": 0.5, "max_repeat": 2, "patience": 2, "min_delta": 0.001, **loop},
        "mining": {"probe_size": 20, "probe_empty_frac": 0.0, "recheck_max": 5, "hard_pool_max": 50},
        "eval": {"dev_size": 10},
        "train": {"learning_rate": 1e-4},
        "budget": {"max_gpu_hours": 5.0, "max_train_tokens": 10_000},
        "output": {"dir": str(tmp_path / "out")},
    }


def loop_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row(f"p{i:03d}", [span(0, 4), span(6, 9)]) for i in range(60)] + [row(f"e{i:03d}") for i in range(40)]
    dev = [row(f"dev{i:02d}", [span(0, 4)]) for i in range(20)]
    return train, dev


def test_loop_keeps_best_adapter_and_early_stops(tmp_path: Path) -> None:
    cfg = loop_config(tmp_path)
    train, dev = loop_data()
    backend = FakeBackend([0.4, 0.8, 0.7, 0.6, 0.5])
    summary = lt.run_loop(cfg, backend, train, dev)
    out = Path(cfg["output"]["dir"])
    assert summary["stop_reason"] == "plateau"
    assert summary["completed_rounds"] == 4  # best at round 1, then 2 rounds without improvement
    assert summary["best_round"] == 1
    assert (out / "best" / "adapter_model.bin").read_text() == "round=1"
    assert (out / "rounds" / "round_03" / "adapter" / "adapter_model.bin").exists()
    # hard examples were mined from round 0's probe and upsampled into round 1
    assert (out / "rounds" / "round_00" / "hard_examples.jsonl").read_text().strip()
    assert summary["hard_pool_size"] > 0
    round1 = lt.LoopState.load(out / "loop_state.json").history[1]["mix"]
    assert round1["hard_examples"] > 0
    assert summary["dev_curve"][1] > summary["dev_curve"][0]


def test_loop_stops_on_token_budget(tmp_path: Path) -> None:
    cfg = loop_config(tmp_path)
    cfg["budget"]["max_train_tokens"] = 250
    train, dev = loop_data()
    backend = FakeBackend([0.1, 0.2, 0.3, 0.4, 0.5], tokens_per_round=100)
    summary = lt.run_loop(cfg, backend, train, dev)
    assert summary["stop_reason"].startswith("budget_tokens")
    assert summary["train_tokens"] <= 300 and len(backend.trained_rounds) <= 3
    assert summary["completed_rounds"] == len(backend.trained_rounds)


def test_loop_refuses_round_that_would_exceed_gpu_hours(tmp_path: Path) -> None:
    cfg = loop_config(tmp_path)
    cfg["budget"]["max_gpu_hours"] = 1.0
    train, dev = loop_data()
    backend = FakeBackend([0.1, 0.2, 0.3, 0.4, 0.5], hours_per_round=0.45)
    # rounds' wall time is ~0 in the fake, so seed the estimate through the config prior
    cfg["budget"]["est_round_hours"] = 0.45
    summary = lt.run_loop(cfg, backend, train, dev)
    assert summary["stop_reason"] in {"budget_gpu_hours_next_round", "budget_gpu_hours"}
    assert summary["gpu_hours"] <= 1.0 + 1e-6
    assert len(backend.trained_rounds) == 2


def test_loop_resumes_from_last_completed_round(tmp_path: Path) -> None:
    cfg = loop_config(tmp_path, max_rounds=2, patience=10)
    train, dev = loop_data()
    first = FakeBackend([0.1, 0.2, 0.3, 0.4, 0.5])
    s1 = lt.run_loop(cfg, first, train, dev)
    assert s1["completed_rounds"] == 2 and s1["stop_reason"] == "max_rounds"
    out = Path(cfg["output"]["dir"])
    state1 = lt.LoopState.load(out / "loop_state.json")
    assert lt.resume_adapter_path(out, state1) == out / "rounds" / "round_01" / "adapter"

    # raise max_rounds: a fresh process continues at round 2 and does not redo rounds 0-1
    cfg2 = loop_config(tmp_path, max_rounds=4, patience=10)
    second = FakeBackend([0.1, 0.2, 0.3, 0.4, 0.5])
    s2 = lt.run_loop(cfg2, second, train, dev)
    assert second.trained_rounds == [2, 3]
    assert s2["completed_rounds"] == 4 and len(s2["dev_curve"]) == 4
    # seen docs / budget carried over; round 2 mix must not repeat round 0-1 fresh docs wholesale
    assert s2["train_tokens"] == 400
    # finished by plateau/budget stays finished
    cfg3 = loop_config(tmp_path, max_rounds=4, patience=10)
    third = FakeBackend([0.5])
    lt.run_loop(cfg3, third, train, dev)
    assert third.trained_rounds == []


def test_crashed_round_is_redone_not_half_committed(tmp_path: Path) -> None:
    cfg = loop_config(tmp_path, max_rounds=3, patience=10)
    train, dev = loop_data()

    class Boom(FakeBackend):
        def train_round(self, rows, lr, round_idx, budget, seed):
            if round_idx == 1:
                raise RuntimeError("cuda died")
            return super().train_round(rows, lr, round_idx, budget, seed)

    with pytest.raises(RuntimeError):
        lt.run_loop(cfg, Boom([0.1, 0.2, 0.3]), train, dev)
    out = Path(cfg["output"]["dir"])
    state = lt.LoopState.load(out / "loop_state.json")
    assert state.completed_rounds == 1 and not state.finished
    retry = FakeBackend([0.1, 0.2, 0.3])
    summary = lt.run_loop(cfg, retry, train, dev)
    assert retry.trained_rounds == [1, 2] and summary["completed_rounds"] == 3


def test_apply_overrides_and_terminal_logic() -> None:
    cfg = lt.apply_overrides({"loop": {"max_rounds": 8}}, ["loop.max_rounds=3", "budget.max_gpu_hours=1.5"])
    assert cfg["loop"]["max_rounds"] == 3 and cfg["budget"]["max_gpu_hours"] == 1.5
    done = lt.LoopState(completed_rounds=3, finished=True, stop_reason="plateau")
    assert lt.is_terminal(done, cfg)
    assert not lt.is_terminal(done, lt.apply_overrides(cfg, ["loop.force_continue=true"]))
    capped = lt.LoopState(completed_rounds=3, finished=True, stop_reason="max_rounds")
    assert lt.is_terminal(capped, cfg) and not lt.is_terminal(capped, lt.apply_overrides(cfg, ["loop.max_rounds=5"]))
