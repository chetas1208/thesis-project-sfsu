"""J5: a properly trained BERT-family token-classification baseline for SkillSpan (fair baseline for H1).

Why: the repository's "BERT baseline" (src/skillllm_v2/baselines/bert_baseline.py) is not a neural model - it memorises
the most frequent BIO tag of each lower-cased training token and tags test tokens by lookup (training_mode
"token_classification_memorized_baseline"), scoring exact F1 0.059. Comparing the anchor against it is a straw man.

Model (follows the SkillSpan paper's setup as closely as the repo allows, Zhang et al., NAACL 2022): a pre-trained
encoder (jjzha/jobbert-base-cased = the paper's JobBERT, or bert-base-cased) with TWO BIO token-classification heads, one
for SkillSpan "knowledge" tags (-> HARD_SKILL) and one for "skill" tags (-> SOFT_SKILL), so overlapping/nested spans of
the two types are representable (multi-task, shared encoder). Labels on the first sub-token of each word (others -100).
AdamW lr 5e-5, weight decay 0.01, linear schedule with 10% warm-up, batch 16, max 20 epochs, early stopping on DEV exact
span F1 (patience 5), the best-dev checkpoint is kept. Training data: SkillSpan train (4,800 sentences), dev for model
selection only (3,174 sentences), test never used for any decision.

Spans are decoded from predicted word tags with the SAME function that builds gold spans from gold tags
(build_verifier_dataset._spans_from_tags), on the SAME text/offsets (_tokens_to_text), so a perfect tagger reproduces gold
exactly. Predictions are written for all 3,569 test sentences (the 974-doc gold subset is a filter of them) in the common
``{"id","text","predictions":[...]}`` format.

Usage (GPU job, via scripts/evals/gpu_queue.sh):
  python scripts/evals/gpu_bert_fair.py --model jjzha/jobbert-base-cased --seeds 42 43 44
  python scripts/evals/gpu_bert_fair.py --select            # CPU: pick the dev-best run -> bert_fair/predictions.jsonl
  python scripts/evals/gpu_bert_fair.py --predict-input X.jsonl --run <run_dir> --out Y.jsonl   # tokens given per row
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
OUT = REPO_ROOT / "artifacts/skillllm_v2/evals_gpu/bert_fair"
TAGS = ["O", "B", "I"]
HEADS = (("tags_knowledge", "HARD_SKILL"), ("tags_skill", "SOFT_SKILL"))
MAX_LEN = 256


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(model: str) -> str:
    return model.replace("/", "__")


def load_split(split: str) -> List[Dict[str, Any]]:
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
    return load_skillspan_split(REPO_ROOT / "data/source/skillspan_hf", split, limit=None)


def norm_tag(t: str) -> str:
    return "B" if t.startswith("B") else ("I" if t.startswith("I") else "O")


def spans_from_word_tags(text: str, offsets: Sequence[Tuple[int, int]], tags: Sequence[str], label: str,
                         probs: Sequence[float] | None = None) -> List[Dict[str, Any]]:
    """Gold-identical span decoding (build_verifier_dataset._spans_from_tags) + mean tag probability as confidence."""
    from skillllm_v2.train.build_verifier_dataset import _spans_from_tags
    spans = _spans_from_tags(text, list(offsets), list(tags), label)
    out = []
    for sp in spans:
        idx = [i for i, (s, e) in enumerate(offsets) if s >= sp["start"] and e <= sp["end"]]
        conf = float(sum(probs[i] for i in idx) / len(idx)) if probs is not None and idx else None
        out.append({"surface_form": sp["surface_form"], "label": label, "start": sp["start"], "end": sp["end"],
                    "source": "bert_fair", "confidence": conf, "evidence": sp["surface_form"],
                    "canonical_skill_id": None, "canonical_skill_name": None, "metadata": {}})
    return out


def word_offsets(tokens: Sequence[str]) -> Tuple[str, List[Tuple[int, int]]]:
    from skillllm_v2.train.build_verifier_dataset import _tokens_to_text
    return _tokens_to_text(list(tokens))


class TwoHeadTagger:  # thin namespace; torch imported lazily so --select runs without CUDA
    @staticmethod
    def build(model_name: str):
        import torch
        from transformers import AutoModel

        class _M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = AutoModel.from_pretrained(model_name)
                h = self.encoder.config.hidden_size
                self.drop = torch.nn.Dropout(0.1)
                self.heads = torch.nn.ModuleList([torch.nn.Linear(h, len(TAGS)) for _ in HEADS])

            def forward(self, input_ids, attention_mask):  # noqa: ANN001, ANN202
                hs = self.drop(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
                return [head(hs) for head in self.heads]
        return _M()


def encode(tokenizer, docs: List[Dict[str, Any]], with_labels: bool) -> List[Dict[str, Any]]:  # noqa: ANN001
    feats = []
    n_trunc_words = 0
    for d in docs:
        enc = tokenizer(d["tokens"], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
        wids = enc.word_ids()
        first = {}
        for pos, w in enumerate(wids):
            if w is not None and w not in first:
                first[w] = pos
        n_trunc_words += len(d["tokens"]) - len(first)
        f = {"input_ids": enc["input_ids"], "first": first, "n_words": len(d["tokens"])}
        if with_labels:
            labs = []
            for key, _ in HEADS:
                lab = [-100] * len(enc["input_ids"])
                tags = d["gold_token_tags"][key]
                for w, pos in first.items():
                    lab[pos] = TAGS.index(norm_tag(tags[w]))
                labs.append(lab)
            f["labels"] = labs
        feats.append(f)
    return feats, n_trunc_words


def batches(feats: List[Dict[str, Any]], bs: int, shuffle: bool, rng: random.Random, pad_id: int, device):  # noqa: ANN001
    import torch
    order = list(range(len(feats)))
    if shuffle:
        rng.shuffle(order)
    for lo in range(0, len(order), bs):
        idx = order[lo:lo + bs]
        L = max(len(feats[i]["input_ids"]) for i in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        labs = [torch.full((len(idx), L), -100, dtype=torch.long) for _ in HEADS] if "labels" in feats[idx[0]] else None
        for r, i in enumerate(idx):
            n = len(feats[i]["input_ids"])
            ids[r, :n] = torch.tensor(feats[i]["input_ids"])
            att[r, :n] = 1
            if labs is not None:
                for h in range(len(HEADS)):
                    labs[h][r, :n] = torch.tensor(feats[i]["labels"][h])
        yield idx, ids.to(device), att.to(device), (None if labs is None else [x.to(device) for x in labs])


def predict(model, feats, docs, bs: int, device, pad_id: int) -> List[Dict[str, Any]]:  # noqa: ANN001
    """Word tags (first sub-token argmax) -> spans; words lost to truncation are tagged O (counted in the log)."""
    import torch
    model.eval()
    out: List[Dict[str, Any]] = [None] * len(docs)  # type: ignore[list-item]
    with torch.no_grad():
        for idx, ids, att, _ in batches(feats, bs, False, random.Random(0), pad_id, device):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(ids, att)
            probs = [torch.softmax(lg.float(), dim=-1).cpu() for lg in logits]
            for r, i in enumerate(idx):
                d, f = docs[i], feats[i]
                text, offs = d["text"], d["offsets"]
                preds = []
                for h, (_, label) in enumerate(HEADS):
                    tags, conf = [], []
                    for w in range(f["n_words"]):
                        pos = f["first"].get(w)
                        if pos is None:
                            tags.append("O"); conf.append(1.0)
                        else:
                            p = probs[h][r, pos]
                            k = int(p.argmax())
                            tags.append(TAGS[k]); conf.append(float(p[k]))
                    preds += spans_from_word_tags(text, offs, tags, label, conf)
                out[i] = {"id": d["id"], "text": text, "predictions": preds}
    return out


def exact_f1(gold_docs: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    from skillllm_v2.eval.extraction_eval import evaluate_extraction
    m = evaluate_extraction([d["gold_spans"] for d in gold_docs], [r["predictions"] for r in pred_rows])
    return {k: m[k] for k in ("exact_span_precision", "exact_span_recall", "exact_span_f1", "hard_skill_f1", "soft_skill_f1")}


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def train_one(model_name: str, seed: int, args: argparse.Namespace, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    run = OUT / "runs" / f"{slug(model_name)}_s{seed}"
    if (run / "DONE.json").exists():
        return json.loads((run / "DONE.json").read_text())
    run.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = TwoHeadTagger.build(model_name).to(device)
    feats = {s: encode(tok, data[s], with_labels=(s == "train")) for s in data}
    trunc = {s: feats[s][1] for s in feats}
    feats = {s: feats[s][0] for s in feats}
    steps_per_epoch = math.ceil(len(feats["train"]) / args.batch_size)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps_per_epoch * args.max_epochs), steps_per_epoch * args.max_epochs)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    rng = random.Random(seed)
    best, best_state, best_epoch, bad, log = -1.0, None, -1, 0, []
    t_train = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0, tot = time.perf_counter(), 0.0
        for _, ids, att, labs in batches(feats["train"], args.batch_size, True, rng, tok.pad_token_id, device):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(ids, att)
                loss = sum(ce(lg.float().reshape(-1, len(TAGS)), lb.reshape(-1)) for lg, lb in zip(logits, labs))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss)
        dev_rows = predict(model, feats["dev"], data["dev"], 64, device, tok.pad_token_id)
        dm = exact_f1(data["dev"], dev_rows)
        entry = {"epoch": epoch, "train_loss": tot / steps_per_epoch, "dev": dm, "epoch_seconds": time.perf_counter() - t0}
        log.append(entry)
        print(f"{utc()} {model_name} s{seed} ep{epoch} loss={entry['train_loss']:.4f} dev_exact_f1={dm['exact_span_f1']:.4f}", flush=True)
        if dm["exact_span_f1"] > best:
            best, best_epoch, bad = dm["exact_span_f1"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= args.patience:
                break
    train_seconds = time.perf_counter() - t_train
    model.load_state_dict(best_state)
    torch.save(best_state, run / "model.pt")
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    t0 = time.perf_counter()
    test_rows = predict(model, feats["test"], data["test"], 64, device, tok.pad_token_id)
    test_seconds = time.perf_counter() - t0
    dev_rows = predict(model, feats["dev"], data["dev"], 64, device, tok.pad_token_id)
    write_jsonl(run / "predictions.jsonl", test_rows)
    write_jsonl(run / "dev_predictions.jsonl", dev_rows)
    (run / "train_log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    n_params = sum(p.numel() for p in model.parameters())
    done = {"model": model_name, "seed": seed, "status": "complete", "completed_utc": utc(), "best_epoch": best_epoch,
            "epochs_run": len(log), "best_dev": log[best_epoch - 1]["dev"], "selection": "best dev exact_span_f1 (patience %d)" % args.patience,
            "hyperparameters": {"lr": args.lr, "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience,
                                "weight_decay": 0.01, "warmup_frac": 0.1, "max_len": MAX_LEN, "dropout": 0.1, "precision": "bf16 autocast",
                                "grad_clip": 1.0, "heads": [h for h, _ in HEADS]},
            "truncated_words": trunc, "n_params": n_params, "train_seconds": train_seconds, "test_predict_seconds": test_seconds,
            "test_docs": len(test_rows), "peak_alloc_gb": peak,
            "outputs": {f: sha256_file(run / f) for f in ("predictions.jsonl", "dev_predictions.jsonl", "model.pt")},
            "hf_model_revision": _hf_revision(model_name), "command": " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME")}, "versions": _versions(), "git": _git()}
    (run / "DONE.json").write_text(json.dumps(done, indent=2) + "\n")
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return done


def _hf_revision(model_name: str) -> str | None:
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_name:
                return sorted(r.commit_hash for r in repo.revisions)[0]
    except Exception:  # noqa: BLE001
        return None
    return None


def _versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for n in ("torch", "transformers"):
        try:
            out[n] = __import__(n).__version__
        except Exception:  # noqa: BLE001
            out[n] = "missing"
    return out


def _git() -> Dict[str, Any]:
    try:
        c = subprocess.run(["git", "-C", str(REPO_ROOT.parent), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        c = ""
    return {"commit": c}


def prepare_data() -> Dict[str, List[Dict[str, Any]]]:
    data = {}
    for s in ("train", "dev", "test"):
        docs = load_split(s)
        for d in docs:
            d["text"], d["offsets"] = word_offsets(d["tokens"])
        data[s] = docs
    return data


def select() -> None:
    """Pick the single dev-best run (exact span F1 on dev) as THE fair baseline; test is never looked at."""
    runs = sorted((OUT / "runs").glob("*/DONE.json"))
    if not runs:
        raise SystemExit("no finished runs")
    cands = [json.loads(p.read_text()) | {"dir": str(p.parent.relative_to(REPO_ROOT))} for p in runs]
    best = max(cands, key=lambda c: c["best_dev"]["exact_span_f1"])
    src = REPO_ROOT / best["dir"]
    shutil.copyfile(src / "predictions.jsonl", OUT / "predictions.jsonl")
    shutil.copyfile(src / "dev_predictions.jsonl", OUT / "dev_predictions.jsonl")
    summary = {"selected": best["dir"], "model": best["model"], "seed": best["seed"], "selection_rule": "max dev exact_span_f1 over all runs",
               "runs": [{k: c[k] for k in ("dir", "model", "seed", "best_epoch", "epochs_run", "best_dev")} for c in cands],
               "outputs": {"predictions.jsonl": sha256_file(OUT / "predictions.jsonl")}, "completed_utc": utc()}
    (OUT / "SELECTED.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def predict_file(args: argparse.Namespace) -> None:
    """Tag a custom jsonl (rows: id, text, tokens, token_offsets) with a trained run; offsets come from the row."""
    import torch
    from transformers import AutoTokenizer
    run = Path(args.run)
    meta = json.loads((run / "DONE.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(meta["model"])
    model = TwoHeadTagger.build(meta["model"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.to(device)
    docs = []
    for line in open(args.predict_input, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            docs.append({"id": r["id"], "text": r["text"], "tokens": r["tokens"], "offsets": [tuple(x) for x in r["token_offsets"]]})
    feats, _ = encode(tok, docs, with_labels=False)
    t0 = time.perf_counter()
    rows = predict(model, feats, docs, 64, device, tok.pad_token_id)
    el = time.perf_counter() - t0
    for r in rows:
        for p in r["predictions"]:
            p["metadata"] = {"run": str(run.relative_to(REPO_ROOT)) if run.is_absolute() else str(run)}
    write_jsonl(Path(args.out), rows)
    print(f"tagged {len(rows)} docs in {el:.1f}s on {device}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="jjzha/jobbert-base-cased")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--predict-input")
    ap.add_argument("--run")
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch, writes under bert_fair/smoke")
    args = ap.parse_args()
    os.chdir(REPO_ROOT)
    if args.select:
        return select()
    if args.predict_input:
        return predict_file(args)
    global OUT
    data = prepare_data()
    if args.smoke:
        OUT = OUT / "smoke"
        data = {s: v[:200] for s, v in data.items()}
        args.max_epochs, args.seeds = 1, args.seeds[:1]
    OUT.mkdir(parents=True, exist_ok=True)
    results = [train_one(args.model, s, args, data) for s in args.seeds]
    (OUT / f"DONE_{slug(args.model)}.json").write_text(json.dumps(
        {"model": args.model, "seeds": args.seeds, "completed_utc": utc(),
         "runs": [{"seed": r["seed"], "best_epoch": r["best_epoch"], "best_dev": r["best_dev"]} for r in results]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
