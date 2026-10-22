"""Low-compute, quantized, iterative ("loop") QLoRA trainer for the SkillSpan extractor.

Instead of one brute-force 80-epoch run, training is split into short rounds
(a fraction of an epoch each). After every round the loop

1. evaluates the adapter on a FIXED held-out dev subset (model selection only),
2. probes the adapter on a rotating slice of the TRAIN split and mines hard examples
   (missed spans, spurious spans, boundary errors, label errors),
3. upsamples those hard examples into the next round's SFT mix (curriculum),
4. keeps the best adapter by dev exact-span F1, stops on plateau, and stops when a
   hard GPU-hours / token budget is reached.

Mining uses train-split probes, never the dev subset, so dev F1 stays an honest
selection signal. State is committed once per round (``loop_state.json``), so a
killed run resumes from the last completed round.

Everything above ``QLoraBackend`` is pure Python (no torch import) and unit-tested.
GPU work lives in ``QLoraBackend`` and imports torch/peft/bitsandbytes lazily.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from skillllm_v2.config import load_yaml
from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.io_utils import read_jsonl, write_json, write_jsonl

SELECTION_METRIC = "exact_span_f1"
DEFAULT_ERROR_WEIGHTS = {"missed": 1.0, "spurious": 1.0, "boundary": 0.75, "label": 0.75}


# --------------------------------------------------------------------------------------
# Hard-example mining (pure)
# --------------------------------------------------------------------------------------
def _key(span: dict[str, Any]) -> tuple[Any, Any, Any]:
    return span.get("start"), span.get("end"), span.get("label")


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("start") is None or b.get("start") is None or a.get("end") is None or b.get("end") is None:
        return False
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def classify_errors(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> dict[str, int]:
    """Split one document's exact-match errors into missed / spurious / boundary / label counts.

    Exact matches (start, end, label) are correct. A leftover gold span that overlaps a
    leftover prediction is a ``label`` error when the offsets agree and a ``boundary``
    error otherwise; leftover gold spans with no overlapping prediction are ``missed``
    (false negatives); leftover predictions (including unaligned ones) are ``spurious``
    (false positives).
    """
    gold_keys = {_key(item) for item in gold}
    pred_keys = {_key(item) for item in pred}
    exact = len(gold_keys & pred_keys)
    left_gold = [item for item in gold if _key(item) not in pred_keys]
    left_pred = [item for item in pred if _key(item) not in gold_keys]
    counts = {"exact": exact, "missed": 0, "spurious": 0, "boundary": 0, "label": 0}
    used_pred: set[int] = set()
    for g in left_gold:
        match = None
        for index, p in enumerate(left_pred):
            if index not in used_pred and _overlaps(g, p):
                match = index
                break
        if match is None:
            counts["missed"] += 1
            continue
        used_pred.add(match)
        p = left_pred[match]
        if (g.get("start"), g.get("end")) == (p.get("start"), p.get("end")):
            counts["label"] += 1
        else:
            counts["boundary"] += 1
    counts["spurious"] = len(left_pred) - len(used_pred)
    return counts


def difficulty_score(counts: dict[str, int], weights: dict[str, float] | None = None) -> float:
    weights = {**DEFAULT_ERROR_WEIGHTS, **(weights or {})}
    return float(sum(weights[name] * counts.get(name, 0) for name in DEFAULT_ERROR_WEIGHTS))


def mine_hard_examples(
    rows: list[dict[str, Any]],
    predictions: list[list[dict[str, Any]]],
    weights: dict[str, float] | None = None,
    min_score: float = 0.5,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Return hard documents (score >= min_score), hardest first. ``rows`` are SFT rows."""
    mined: list[dict[str, Any]] = []
    for row, pred in zip(rows, predictions):
        counts = classify_errors(row.get("output", {}).get("skills", []), pred)
        score = difficulty_score(counts, weights)
        if score >= min_score:
            mined.append({"id": row["id"], "score": score, "errors": counts})
    mined.sort(key=lambda item: (-item["score"], item["id"]))
    return mined[:top_k] if top_k else mined


def update_hard_pool(
    pool: dict[str, dict[str, Any]],
    mined: list[dict[str, Any]],
    probed_ids: list[str],
    round_idx: int,
    max_size: int = 400,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Merge mined docs into the pool. Probed docs that are no longer wrong are retired."""
    new_pool = copy.deepcopy(pool)
    mined_by_id = {item["id"]: item for item in mined}
    fixed = 0
    for doc_id in probed_ids:
        if doc_id in mined_by_id:
            item = mined_by_id[doc_id]
            streak = new_pool.get(doc_id, {}).get("streak", 0) + 1
            new_pool[doc_id] = {"score": item["score"], "errors": item["errors"], "round": round_idx, "streak": streak}
        elif doc_id in new_pool:
            del new_pool[doc_id]
            fixed += 1
    dropped = 0
    if len(new_pool) > max_size:
        keep = sorted(new_pool.items(), key=lambda kv: (-kv[1]["score"], kv[0]))[:max_size]
        dropped = len(new_pool) - len(keep)
        new_pool = dict(keep)
    return new_pool, {"added_or_kept": len(mined_by_id), "fixed": fixed, "dropped": dropped, "size": len(new_pool)}


def repeat_count(entry: dict[str, Any], max_repeat: int) -> int:
    return max(1, min(int(max_repeat), int(round(entry.get("score", 1.0)))))


# --------------------------------------------------------------------------------------
# Data selection / curriculum (pure)
# --------------------------------------------------------------------------------------
def _has_skills(row: dict[str, Any]) -> bool:
    return bool(row.get("output", {}).get("skills"))


def subsample(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if not limit or limit >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    positives = [row for row in rows if _has_skills(row)]
    empties = [row for row in rows if not _has_skills(row)]
    n_pos = min(len(positives), max(1, int(round(limit * 0.7))))
    chosen = rng.sample(positives, n_pos) + rng.sample(empties, min(len(empties), limit - n_pos))
    return chosen


def select_dev_subset(rows: list[dict[str, Any]], size: int, seed: int, only_with_gold: bool = True, empty_frac: float = 0.0) -> list[dict[str, Any]]:
    """Fixed dev subset (same ids every round). Defaults to gold-bearing docs, like the wrapper tuning set."""
    rng = random.Random(seed + 17)
    positives = sorted((row for row in rows if _has_skills(row)), key=lambda r: r["id"])
    empties = sorted((row for row in rows if not _has_skills(row)), key=lambda r: r["id"])
    n_empty = 0 if only_with_gold and empty_frac <= 0 else int(round(size * empty_frac))
    chosen = rng.sample(positives, min(len(positives), size - n_empty)) + rng.sample(empties, min(len(empties), n_empty))
    return sorted(chosen, key=lambda r: r["id"])


def select_probe_subset(
    train_rows: list[dict[str, Any]],
    hard_pool: dict[str, dict[str, Any]],
    round_idx: int,
    size: int,
    empty_frac: float,
    recheck_max: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Rotating train probe: re-check current hard docs, fill the rest with fresh docs (some empty = FP bait)."""
    rng = random.Random(seed * 7919 + round_idx)
    by_id = {row["id"]: row for row in train_rows}
    recheck_ids = sorted(hard_pool, key=lambda i: (-hard_pool[i]["score"], i))[:recheck_max]
    probe = [by_id[i] for i in recheck_ids if i in by_id]
    taken = {row["id"] for row in probe}
    remaining = max(0, size - len(probe))
    n_empty = int(round(remaining * empty_frac))
    positives = [row for row in train_rows if _has_skills(row) and row["id"] not in taken]
    empties = [row for row in train_rows if not _has_skills(row) and row["id"] not in taken]
    probe += rng.sample(positives, min(len(positives), remaining - n_empty))
    probe += rng.sample(empties, min(len(empties), n_empty))
    return probe


def empty_frac_for_round(round_idx: int, max_rounds: int, start: float, end: float) -> float:
    """Curriculum: begin positive-heavy, ramp in empty (hard-negative) docs as training proceeds."""
    if max_rounds <= 1:
        return end
    t = min(1.0, max(0.0, round_idx / (max_rounds - 1)))
    return start + (end - start) * t


def build_round_mix(
    train_rows: list[dict[str, Any]],
    hard_pool: dict[str, dict[str, Any]],
    seen_ids: set[str],
    round_idx: int,
    n_examples: int,
    hard_frac: float,
    max_repeat: int,
    empty_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Compose one round: upsampled hard examples + unseen fresh docs. Deterministic per (seed, round)."""
    rng = random.Random(seed * 1000 + round_idx)
    by_id = {row["id"]: row for row in train_rows}
    slots: list[str] = []
    for doc_id, entry in sorted(hard_pool.items()):
        if doc_id in by_id:
            slots.extend([doc_id] * repeat_count(entry, max_repeat))
    hard_n = min(int(round(n_examples * hard_frac)), len(slots))
    hard_ids = rng.sample(slots, hard_n) if hard_n else []
    fresh_n = n_examples - hard_n
    empty_n = int(round(fresh_n * empty_frac))
    pos_n = fresh_n - empty_n

    new_seen = set(seen_ids)
    wraps = 0
    fresh_ids: list[str] = []
    for want, want_pos in ((pos_n, True), (empty_n, False)):
        pool_ids = sorted(r["id"] for r in train_rows if _has_skills(r) == want_pos)
        unseen = [i for i in pool_ids if i not in new_seen]
        if len(unseen) < want:
            wraps += 1
            new_seen -= set(pool_ids)
            unseen = pool_ids
        pick = rng.sample(unseen, min(want, len(unseen)))
        fresh_ids.extend(pick)
        new_seen.update(pick)

    mix_ids = hard_ids + fresh_ids
    rng.shuffle(mix_ids)
    rows = [by_id[i] for i in mix_ids]
    stats = {
        "round": round_idx,
        "examples": len(rows),
        "hard_slots_available": len(slots),
        "hard_examples": hard_n,
        "hard_unique": len(set(hard_ids)),
        "fresh_positive": len([i for i in fresh_ids if _has_skills(by_id[i])]),
        "fresh_empty": len([i for i in fresh_ids if not _has_skills(by_id[i])]),
        "empty_frac": round(empty_frac, 4),
        "seen_wraps": wraps,
    }
    return rows, new_seen, stats


def make_batches(lengths: list[int], micro_batch_size: int, seed: int, bucket_mult: int = 16) -> list[list[int]]:
    """Length-bucketed micro-batches (less padding): shuffle, sort inside buckets, shuffle batches."""
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    bucket = max(1, micro_batch_size * bucket_mult)
    batches: list[list[int]] = []
    for start in range(0, len(order), bucket):
        chunk = sorted(order[start : start + bucket], key=lambda i: lengths[i])
        batches.extend(chunk[i : i + micro_batch_size] for i in range(0, len(chunk), micro_batch_size))
    rng.shuffle(batches)
    return batches


def lr_scale(step: int, total_steps: int, warmup_ratio: float, min_ratio: float) -> float:
    warmup = max(1, int(total_steps * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def lr_for_round(base_lr: float, round_idx: int, decay: float, floor_ratio: float = 0.25) -> float:
    return base_lr * max(floor_ratio, decay**round_idx)


# --------------------------------------------------------------------------------------
# Budget / early stopping / state (pure)
# --------------------------------------------------------------------------------------
@dataclass
class BudgetTracker:
    """Hard cap on GPU-hours and trained tokens. Survives resume via ``to_dict``/``from_dict``.

    ``max_charge_seconds`` clamps what one event (a step / generate batch) may bill, so a
    guardrails SIGSTOP pause is not charged as GPU time. ``eval_reserve_frac`` keeps a slice
    of the GPU-hour cap for the post-training dev eval of the final round.
    """

    max_gpu_hours: float = 5.0
    max_train_tokens: int = 2_500_000
    num_gpus: int = 1
    max_charge_seconds: float = 120.0
    eval_reserve_frac: float = 0.1
    gpu_seconds: float = 0.0
    train_tokens: int = 0

    @property
    def gpu_hours(self) -> float:
        return self.gpu_seconds / 3600.0

    def charge_time(self, seconds: float) -> None:
        self.gpu_seconds += min(max(0.0, seconds), self.max_charge_seconds) * self.num_gpus

    def charge_tokens(self, tokens: int) -> None:
        self.train_tokens += int(tokens)

    def exhausted(self) -> str | None:
        if self.gpu_hours >= self.max_gpu_hours:
            return "budget_gpu_hours"
        if self.train_tokens >= self.max_train_tokens:
            return "budget_tokens"
        return None

    def train_exhausted(self) -> str | None:
        """Stop training a bit early so the round's dev eval still fits under the hard cap."""
        if self.gpu_hours >= self.max_gpu_hours * (1.0 - self.eval_reserve_frac):
            return "budget_gpu_hours"
        if self.train_tokens >= self.max_train_tokens:
            return "budget_tokens"
        return None

    def can_start_round(self, est_round_hours: float, est_round_tokens: int) -> str | None:
        if self.train_exhausted():
            return self.train_exhausted()
        if est_round_hours > 0 and self.gpu_hours + est_round_hours > self.max_gpu_hours:
            return "budget_gpu_hours_next_round"
        if est_round_tokens > 0 and self.train_tokens + est_round_tokens > self.max_train_tokens:
            return "budget_tokens_next_round"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"gpu_seconds": self.gpu_seconds, "train_tokens": self.train_tokens}

    def restore(self, payload: dict[str, Any]) -> None:
        self.gpu_seconds = float(payload.get("gpu_seconds", 0.0))
        self.train_tokens = int(payload.get("train_tokens", 0))


@dataclass
class PlateauStopper:
    """Early stop when dev F1 fails to beat the best by ``min_delta`` for ``patience`` rounds."""

    patience: int = 2
    min_delta: float = 0.002
    best: float = -1.0
    best_round: int = -1
    since_improve: int = 0

    def update(self, round_idx: int, value: float) -> bool:
        """Record a round. Returns True when it is the new best (adapter should be saved as best)."""
        is_best = value > self.best
        if value > self.best + self.min_delta:
            self.since_improve = 0
        else:
            self.since_improve += 1
        if is_best:
            self.best = value
            self.best_round = round_idx
        return is_best

    @property
    def should_stop(self) -> bool:
        return self.since_improve >= self.patience


@dataclass
class LoopState:
    completed_rounds: int = 0
    best_round: int = -1
    best_f1: float = -1.0
    since_improve: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    hard_pool: dict[str, dict[str, Any]] = field(default_factory=dict)
    seen_ids: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    stop_reason: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "LoopState":
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def save(self, path: str | Path) -> None:
        """Atomic write: a crash can never leave a half-written state file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = file_path.with_suffix(file_path.suffix + ".tmp")
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, file_path)


def is_terminal(state: LoopState, config: dict[str, Any]) -> bool:
    """A finished run stays finished (plateau / budget) unless loop.force_continue is set.

    A run that only ended at ``max_rounds`` continues automatically when max_rounds is raised.
    """
    if not state.finished:
        return False
    if bool(config.get("loop", {}).get("force_continue", False)):
        return False
    if state.stop_reason == "max_rounds":
        return state.completed_rounds >= int(config.get("loop", {}).get("max_rounds", 8))
    return True


def round_dir(out_dir: str | Path, round_idx: int) -> Path:
    return Path(out_dir) / "rounds" / f"round_{round_idx:02d}"


def resume_adapter_path(out_dir: str | Path, state: LoopState) -> Path | None:
    """Adapter of the last COMPLETED round (None for a fresh run). A missing dir is a hard error."""
    if state.completed_rounds <= 0:
        return None
    path = round_dir(out_dir, state.completed_rounds - 1) / "adapter"
    if not path.exists():
        raise FileNotFoundError(f"loop_state.json reports {state.completed_rounds} completed rounds but {path} is missing")
    return path


# --------------------------------------------------------------------------------------
# Loop driver (backend-agnostic, unit-tested with a fake backend)
# --------------------------------------------------------------------------------------
class Backend(Protocol):
    def train_round(self, rows: list[dict[str, Any]], lr: float, round_idx: int, budget: BudgetTracker, seed: int) -> dict[str, Any]: ...

    def predict(self, rows: list[dict[str, Any]], budget: BudgetTracker | None) -> list[list[dict[str, Any]]]: ...

    def save_adapter(self, path: Path) -> None: ...


def _log(message: str) -> None:
    print(f"[loop {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _gold(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [row.get("output", {}).get("skills", []) for row in rows]


def run_loop(config: dict[str, Any], backend: Backend, train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    out_dir = Path(config.get("output", {}).get("dir", "artifacts/skillllm_v2/qlora_loop"))
    loop_cfg = config.get("loop", {})
    mining_cfg = config.get("mining", {})
    eval_cfg = config.get("eval", {})
    train_cfg = config.get("train", {})
    budget_cfg = config.get("budget", {})

    max_rounds = int(loop_cfg.get("max_rounds", 8))
    n_examples = int(loop_cfg.get("examples_per_round", 640))
    state_path = out_dir / "loop_state.json"
    state = LoopState.load(state_path)

    budget = BudgetTracker(
        max_gpu_hours=float(budget_cfg.get("max_gpu_hours", 5.0)),
        max_train_tokens=int(budget_cfg.get("max_train_tokens", 2_500_000)),
        num_gpus=int(budget_cfg.get("num_gpus", 1)),
        max_charge_seconds=float(budget_cfg.get("max_charge_seconds_per_event", 120.0)),
        eval_reserve_frac=float(budget_cfg.get("eval_reserve_frac", 0.1)),
    )
    budget.restore(state.budget)
    stopper = PlateauStopper(int(loop_cfg.get("patience", 2)), float(loop_cfg.get("min_delta", 0.002)), state.best_f1, state.best_round, state.since_improve)

    dev_subset = select_dev_subset(
        dev_rows,
        int(eval_cfg.get("dev_size", 160)),
        seed,
        bool(eval_cfg.get("dev_only_with_gold", True)),
        float(eval_cfg.get("dev_empty_frac", 0.0)),
    )
    seen_ids = set(state.seen_ids)
    weights = mining_cfg.get("error_weights")
    _log(f"resume: completed_rounds={state.completed_rounds} best_round={state.best_round} best_f1={state.best_f1:.4f} gpu_hours={budget.gpu_hours:.3f}")
    _log(f"dev subset={len(dev_subset)} docs, train pool={len(train_rows)} docs, hard pool={len(state.hard_pool)}")

    if is_terminal(state, config):
        _log(f"already finished ({state.stop_reason}); nothing to do")
        return _finalize(out_dir, state, budget, config)
    state.finished = False

    if bool(loop_cfg.get("baseline_eval", False)) and state.completed_rounds == 0 and not state.history:
        t0 = time.perf_counter()
        preds = backend.predict(dev_subset, budget)
        metrics = evaluate_extraction(_gold(dev_subset), preds)
        state.history.append({"round": -1, "baseline": True, "dev": metrics, "seconds": time.perf_counter() - t0})
        _log(f"baseline dev {SELECTION_METRIC}={metrics[SELECTION_METRIC]:.4f}")

    stop_reason: str | None = None
    for round_idx in range(state.completed_rounds, max_rounds):
        recent = [h["seconds"] for h in state.history if not h.get("baseline") and h.get("seconds")]
        est_hours = (sum(recent) / len(recent) / 3600.0 * budget.num_gpus) if recent else float(budget_cfg.get("est_round_hours", 0.0))
        est_tokens = int(max((h.get("train", {}).get("tokens", 0) for h in state.history), default=0))
        reason = budget.can_start_round(est_hours, est_tokens)
        if reason:
            stop_reason = reason
            break

        started = time.perf_counter()
        empty_frac = empty_frac_for_round(round_idx, max_rounds, float(loop_cfg.get("empty_frac_start", 0.10)), float(loop_cfg.get("empty_frac_end", 0.25)))
        mix_rows, seen_ids, mix_stats = build_round_mix(
            train_rows, state.hard_pool, seen_ids, round_idx, n_examples,
            float(loop_cfg.get("hard_frac", 0.35)), int(loop_cfg.get("max_repeat", 3)), empty_frac, seed,
        )
        rdir = round_dir(out_dir, round_idx)
        if rdir.exists():
            shutil.rmtree(rdir)  # partial leftovers of a crashed attempt
        rdir.mkdir(parents=True)
        write_json(rdir / "mix_manifest.json", {**mix_stats, "ids": [row["id"] for row in mix_rows]})
        lr = lr_for_round(float(train_cfg.get("learning_rate", 1.5e-4)), round_idx, float(train_cfg.get("lr_decay_per_round", 0.85)))
        _log(f"round {round_idx}: lr={lr:.2e} mix={mix_stats}")

        train_info = backend.train_round(mix_rows, lr, round_idx, budget, seed)
        backend.save_adapter(rdir / "adapter")

        dev_preds = backend.predict(dev_subset, budget)
        dev_metrics = evaluate_extraction(_gold(dev_subset), dev_preds)
        write_jsonl(rdir / "dev_predictions.jsonl", ({"id": r["id"], "gold": _gold([r])[0], "predictions": p} for r, p in zip(dev_subset, dev_preds)))

        mining_info: dict[str, Any] = {"skipped": True}
        if not budget.exhausted() and not train_info.get("budget_stop"):
            probe = select_probe_subset(
                train_rows, state.hard_pool, round_idx + 1,
                int(mining_cfg.get("probe_size", 192)), float(mining_cfg.get("probe_empty_frac", 0.3)),
                int(mining_cfg.get("recheck_max", 48)), seed,
            )
            probe_preds = backend.predict(probe, budget)
            mined = mine_hard_examples(probe, probe_preds, weights, float(mining_cfg.get("min_score", 0.5)), mining_cfg.get("top_k"))
            state.hard_pool, pool_info = update_hard_pool(state.hard_pool, mined, [row["id"] for row in probe], round_idx, int(mining_cfg.get("hard_pool_max", 400)))
            write_jsonl(rdir / "hard_examples.jsonl", mined)
            totals = {name: sum(item["errors"][name] for item in mined) for name in ("missed", "spurious", "boundary", "label")}
            mining_info = {"probe_docs": len(probe), "mined": len(mined), "error_totals": totals, **pool_info}

        is_best = stopper.update(round_idx, dev_metrics[SELECTION_METRIC])
        if is_best:
            best_dir = out_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(rdir / "adapter", best_dir)
            write_json(best_dir / "best_info.json", {"round": round_idx, "dev": dev_metrics})

        entry = {
            "round": round_idx,
            "lr": lr,
            "seconds": time.perf_counter() - started,
            "mix": mix_stats,
            "train": train_info,
            "dev": dev_metrics,
            "mining": mining_info,
            "is_best": is_best,
            "gpu_hours_total": budget.gpu_hours,
            "train_tokens_total": budget.train_tokens,
        }
        state.history.append(entry)
        write_json(rdir / "round_summary.json", entry)
        # Commit point: only now does the round count as completed for resume.
        state.completed_rounds = round_idx + 1
        state.best_round, state.best_f1, state.since_improve = stopper.best_round, stopper.best, stopper.since_improve
        state.seen_ids = sorted(seen_ids)
        state.budget = budget.to_dict()
        state.save(state_path)
        _log(f"round {round_idx} done: dev {SELECTION_METRIC}={dev_metrics[SELECTION_METRIC]:.4f} best={stopper.best:.4f}@{stopper.best_round} gpu_h={budget.gpu_hours:.3f} tokens={budget.train_tokens}")

        reason = budget.exhausted() or ("budget_stop_mid_round" if train_info.get("budget_stop") else None)
        if reason:
            stop_reason = reason
            break
        if stopper.should_stop:
            stop_reason = "plateau"
            break
    else:
        stop_reason = "max_rounds"

    state.finished = True
    state.stop_reason = stop_reason or "max_rounds"
    state.budget = budget.to_dict()
    state.save(state_path)
    _log(f"stopped: {state.stop_reason}")
    return _finalize(out_dir, state, budget, config)


def _finalize(out_dir: Path, state: LoopState, budget: BudgetTracker, config: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "training_mode": "qlora_loop",
        "stop_reason": state.stop_reason,
        "completed_rounds": state.completed_rounds,
        "best_round": state.best_round,
        "best_dev_exact_span_f1": state.best_f1,
        "best_adapter": str(out_dir / "best") if (out_dir / "best").exists() else None,
        "gpu_hours": budget.gpu_hours,
        "train_tokens": budget.train_tokens,
        "budget": {"max_gpu_hours": budget.max_gpu_hours, "max_train_tokens": budget.max_train_tokens},
        "dev_curve": [h["dev"][SELECTION_METRIC] for h in state.history if "dev" in h],
        "hard_pool_size": len(state.hard_pool),
        "run_id": config.get("run_id"),
    }
    write_json(out_dir / "loop_summary.json", summary)
    return summary


# --------------------------------------------------------------------------------------
# GPU backend (lazy heavy imports)
# --------------------------------------------------------------------------------------
class QLoraBackend:
    """4-bit NF4 (double-quant, bf16 compute) base + small LoRA, custom loop with completion-only loss."""

    def __init__(self, config: dict[str, Any], resume_adapter: Path | None) -> None:
        try:
            import torch
            import transformers
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            from transformers import AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("QLoRA loop training requires torch, transformers, peft, and bitsandbytes.") from exc
        from skillllm_v2.train.mistral_lora_trainer import _ensure_chat_template, _load_model, _load_tokenizer

        self.torch = torch
        self.config = config
        model_cfg = config.get("model", {})
        lora_cfg = config.get("lora", {})
        self.train_cfg = config.get("train", {})
        self.eval_cfg = config.get("eval", {})
        self.use_4bit = bool(model_cfg.get("use_4bit", True))
        self.device_index = int(model_cfg.get("device_index", 0))
        self.cuda = torch.cuda.is_available() and bool(model_cfg.get("use_cuda", True))
        self.device = torch.device(f"cuda:{self.device_index}") if self.cuda else torch.device("cpu")
        torch.manual_seed(int(config.get("seed", 42)))

        if self.cuda:
            torch.cuda.set_device(self.device_index)
            free_gb = torch.cuda.mem_get_info(self.device_index)[0] / 1e9
            need = float(model_cfg.get("min_free_gb", 19.0))
            _log(f"cuda:{self.device_index} free={free_gb:.1f} GB (need >= {need:.1f})")
            if free_gb < need:
                raise RuntimeError(f"Refusing to load: only {free_gb:.1f} GB free on cuda:{self.device_index} (< min_free_gb={need}). Another process owns this GPU.")

        model_path = model_cfg["model_path"]
        self.tokenizer = _load_tokenizer(AutoTokenizer, model_path)
        _ensure_chat_template(self.tokenizer, model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        dtype = torch.bfloat16 if (self.cuda and torch.cuda.is_bf16_supported()) or not self.cuda else torch.float16
        kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True, "attn_implementation": model_cfg.get("attn_implementation", "sdpa")}
        if self.use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        kwargs["device_map"] = {"": self.device_index} if self.cuda else {"": "cpu"}
        try:
            model = _load_model(transformers, model_path, {**kwargs, "dtype": dtype})
        except RuntimeError:
            model = _load_model(transformers, model_path, {**kwargs, "torch_dtype": dtype})

        if bool(self.train_cfg.get("gradient_checkpointing", True)):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        if bool(model_cfg.get("upcast_norms", False)):
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(self.train_cfg.get("gradient_checkpointing", True)))

        init_adapter = resume_adapter or model_cfg.get("init_adapter_path")
        if init_adapter and Path(init_adapter).exists():
            _log(f"loading trainable adapter from {init_adapter}")
            model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
        else:
            lora = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=lora_cfg.get("target_modules", r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"),
            )
            model = get_peft_model(model, lora)
        self.model = model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _log(f"trainable LoRA params: {trainable / 1e6:.2f}M")
        if self.cuda:
            _log(f"model loaded; cuda allocated={torch.cuda.memory_allocated(self.device_index) / 1e9:.1f} GB")

    # -- encoding ------------------------------------------------------------------
    def _prompt_text(self, text: str) -> str:
        from skillllm_v2.extract.mistral_extractor import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode_example(self, row: dict[str, Any]) -> tuple[list[int], list[int]]:
        """(input_ids, labels) with the loss on assistant tokens only; prompt tokenized exactly as at inference."""
        from skillllm_v2.train.mistral_lora_trainer import _target_json

        prompt_ids = self.tokenizer(self._prompt_text(row["input"]))["input_ids"]
        target_ids = self.tokenizer(_target_json(row), add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids

    def _collate(self, encoded: list[tuple[list[int], list[int]]]) -> dict[str, Any]:
        torch = self.torch
        width = max(len(ids) for ids, _ in encoded)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(encoded), width), pad, dtype=torch.long)
        labels = torch.full((len(encoded), width), -100, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for i, (ids, lab) in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            labels[i, : len(lab)] = torch.tensor(lab)
            mask[i, : len(ids)] = 1
        return {"input_ids": input_ids.to(self.device), "labels": labels.to(self.device), "attention_mask": mask.to(self.device)}

    # -- training ------------------------------------------------------------------
    def _make_optimizer(self, params: list[Any], lr: float) -> Any:
        torch = self.torch
        weight_decay = float(self.train_cfg.get("weight_decay", 0.0))
        name = str(self.train_cfg.get("optimizer", "paged_adamw_8bit"))
        if name.startswith("paged_adamw_8bit") and self.cuda:
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def train_round(self, rows: list[dict[str, Any]], lr: float, round_idx: int, budget: BudgetTracker, seed: int) -> dict[str, Any]:
        torch = self.torch
        max_len = int(self.train_cfg.get("max_seq_length", 768))
        micro = int(self.train_cfg.get("micro_batch_size", 2))
        accum = int(self.train_cfg.get("grad_accum", 8))
        encoded, dropped = [], 0
        for row in rows:
            ids, labels = self.encode_example(row)
            if len(ids) > max_len:
                dropped += 1  # never truncate: a cut-off target would teach broken JSON
                continue
            encoded.append((ids, labels))
        batches = make_batches([len(ids) for ids, _ in encoded], micro, seed * 31 + round_idx)
        total_steps = max(1, math.ceil(len(batches) / accum))
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self._make_optimizer(params, lr)
        min_ratio = float(self.train_cfg.get("min_lr_ratio", 0.1))
        warmup = float(self.train_cfg.get("warmup_ratio", 0.1))
        max_norm = float(self.train_cfg.get("max_grad_norm", 1.0))
        autocast_dtype = torch.bfloat16 if (self.cuda and torch.cuda.is_bf16_supported()) or not self.cuda else torch.float16

        self.model.train()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device_index)
        losses: list[float] = []
        tokens = 0
        charged_tokens = 0
        oom = 0
        step = 0
        budget_stop = None
        t_step = time.perf_counter()
        for b_index, batch_ids in enumerate(batches):
            batch = self._collate([encoded[i] for i in batch_ids])
            try:
                with torch.autocast(device_type="cuda" if self.cuda else "cpu", dtype=autocast_dtype):
                    loss = self.model(**batch, use_cache=False).loss
                (loss / accum).backward()
                losses.append(float(loss.detach()))
                tokens += int(batch["attention_mask"].sum())
            except torch.cuda.OutOfMemoryError:
                oom += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                _log(f"OOM at batch {b_index} (len={batch['input_ids'].shape[1]}); skipping")
                if oom > int(self.train_cfg.get("max_oom", 3)):
                    raise
                continue
            at_boundary = (b_index + 1) % accum == 0 or b_index == len(batches) - 1
            if not at_boundary:
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr * lr_scale(step, total_steps, warmup, min_ratio)
            torch.nn.utils.clip_grad_norm_(params, max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            now = time.perf_counter()
            budget.charge_time(now - t_step)
            t_step = now
            budget.charge_tokens(tokens - charged_tokens)
            charged_tokens = tokens
            if step % int(self.train_cfg.get("log_every_steps", 5)) == 0 or step == total_steps:
                recent = losses[-accum * 5 :]
                _log(f"  step {step}/{total_steps} loss={sum(recent) / len(recent):.4f} tokens={tokens} gpu_h={budget.gpu_hours:.3f}")
            budget_stop = budget.train_exhausted()
            if budget_stop:
                _log(f"budget reached during round ({budget_stop}); ending round early")
                break
        budget.charge_tokens(tokens - charged_tokens)
        del optimizer
        peak = torch.cuda.max_memory_allocated(self.device_index) / 1e9 if self.cuda else 0.0
        if self.cuda:
            torch.cuda.empty_cache()
        return {
            "examples": len(encoded),
            "dropped_too_long": dropped,
            "optimizer_steps": step,
            "tokens": tokens,
            "mean_loss": sum(losses) / max(1, len(losses)),
            "last_loss": losses[-1] if losses else None,
            "oom_skips": oom,
            "peak_alloc_gb": round(peak, 2),
            "budget_stop": budget_stop,
        }

    # -- inference -----------------------------------------------------------------
    def predict(self, rows: list[dict[str, Any]], budget: BudgetTracker | None) -> list[list[dict[str, Any]]]:
        from skillllm_v2.extract.mistral_extractor import MistralExtractor, parse_skill_json

        torch = self.torch
        helper = MistralExtractor(allow_fallback=True)  # only used for its payload->span alignment
        batch_size = int(self.eval_cfg.get("batch_size", 8))
        max_new = int(self.eval_cfg.get("max_new_tokens", 384))
        prompts = [self._prompt_text(row["input"]) for row in rows]
        order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
        outputs: list[list[dict[str, Any]]] = [[] for _ in rows]
        self.model.eval()
        previous_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        parse_failures = 0
        try:
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                t0 = time.perf_counter()
                inputs = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad(), torch.autocast(device_type="cuda" if self.cuda else "cpu", dtype=torch.bfloat16 if self.cuda else torch.float32, enabled=self.cuda):
                    generated = self.model.generate(
                        **inputs, max_new_tokens=max_new, do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id
                    )
                for local, i in enumerate(idx):
                    text = self.tokenizer.decode(generated[local][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
                    try:
                        payload = parse_skill_json(text)
                        outputs[i] = [skill.to_dict() for skill in helper._skills_from_payload(rows[i]["input"], payload)]
                    except Exception:
                        parse_failures += 1
                if budget is not None:
                    budget.charge_time(time.perf_counter() - t0)
        finally:
            self.tokenizer.padding_side = previous_side
        if parse_failures:
            _log(f"  generation: {parse_failures}/{len(rows)} outputs were not parseable JSON")
        return outputs

    def save_adapter(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))


# --------------------------------------------------------------------------------------
# Config / CLI
# --------------------------------------------------------------------------------------
def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (values parsed as YAML scalars)."""
    import yaml

    config = copy.deepcopy(config)
    for item in overrides:
        dotted, _, raw = item.partition("=")
        node = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return config


SMOKE_OVERRIDES = [
    "data.limit_train=20",
    "data.limit_dev=20",
    "loop.max_rounds=1",
    "loop.examples_per_round=12",
    "train.grad_accum=2",
    "train.max_seq_length=768",
    "mining.probe_size=8",
    "mining.recheck_max=0",
    "eval.dev_size=8",
    "eval.batch_size=4",
    "eval.max_new_tokens=160",
    "budget.max_gpu_hours=0.5",
    "output.dir=artifacts/skillllm_v2/qlora_loop_smoke",
]


def plan(config: dict[str, Any], train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    loop_cfg = config.get("loop", {})
    n = int(loop_cfg.get("examples_per_round", 640))
    seed = int(config.get("seed", 42))
    seen: set[str] = set()
    rounds = []
    for r in range(int(loop_cfg.get("max_rounds", 8))):
        ef = empty_frac_for_round(r, int(loop_cfg.get("max_rounds", 8)), float(loop_cfg.get("empty_frac_start", 0.1)), float(loop_cfg.get("empty_frac_end", 0.25)))
        _, seen, stats = build_round_mix(train_rows, {}, seen, r, n, 0.0, 1, ef, seed)
        rounds.append(stats)
    return {
        "train_docs": len(train_rows),
        "train_positive_docs": sum(_has_skills(r) for r in train_rows),
        "dev_docs": len(dev_rows),
        "dev_subset": len(select_dev_subset(dev_rows, int(config.get("eval", {}).get("dev_size", 160)), seed)),
        "examples_per_round": n,
        "epoch_fraction_per_round": round(n / max(1, len(train_rows)), 3),
        "max_rounds": int(loop_cfg.get("max_rounds", 8)),
        "budget": config.get("budget", {}),
        "output_dir": config.get("output", {}).get("dir"),
        "round0_mix_no_hard_pool": rounds[0] if rounds else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-compute QLoRA loop trainer (hard-example mining, budget cap, resumable).")
    parser.add_argument("--config", default="configs/skillllm_v2_loop_qlora.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="override, e.g. loop.max_rounds=4")
    parser.add_argument("--smoke", action="store_true", help="tiny 1-round run (20 docs) for plumbing checks")
    parser.add_argument("--plan", action="store_true", help="print the data/round plan and exit (no model load)")
    args = parser.parse_args()

    config = apply_overrides(load_yaml(args.config), (SMOKE_OVERRIDES if args.smoke else []) + args.overrides)
    seed = int(config.get("seed", 42))
    data_cfg = config.get("data", {})
    train_rows = subsample(read_jsonl(data_cfg["train_file"]), data_cfg.get("limit_train"), seed)
    dev_rows = subsample(read_jsonl(data_cfg["dev_file"]), data_cfg.get("limit_dev"), seed)
    if not train_rows or not dev_rows:
        raise FileNotFoundError(f"Missing SFT data: {data_cfg.get('train_file')} / {data_cfg.get('dev_file')} (run scripts/build_mistral_sft_data.sh)")
    if args.plan:
        print(json.dumps(plan(config, train_rows, dev_rows), indent=2))
        return

    out_dir = Path(config.get("output", {}).get("dir", "artifacts/skillllm_v2/qlora_loop"))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "resolved_config.json", config)
    state = LoopState.load(out_dir / "loop_state.json")
    if is_terminal(state, config):
        _log(f"run already finished ({state.stop_reason}); see {out_dir / 'loop_summary.json'}")
        return
    backend = QLoraBackend(config, resume_adapter_path(out_dir, state))
    summary = run_loop(config, backend, train_rows, dev_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
