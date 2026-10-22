"""Experiment 2, step 0 (CPU only): build the unlabeled in-domain chunk pool for pseudo-labelling.

Input  : SkillLLM_v2/Extract_jobDesc/data/job_descriptions.jsonl (local/private; text is NEVER written to docs/reports/git).
Steps  : (1) drop postings overlapping SkillSpan train/dev/test text, (2) sample up to N postings, (3) split into SkillSpan-sized
         chunks, (4) drop chunks equal to a SkillSpan dev/test sentence, (5) sample <= K chunks per posting.
Output : artifacts/skillllm_v2/qlora_loop_v4_distill/pseudo_pool/chunks.jsonl (private working file, ignored by reports) and
         reports/skillllm_v2/qlora_loop_v4_corpus_counts.json (COUNTS ONLY).

Overlap rule (pre-declared): a SkillSpan sentence with >= 8 normalised (lower-case alphanumeric) tokens overlaps a posting if >= 50% of
its 8-gram shingles occur in the posting (100% = normalised substring). A posting with any overlapping sentence is removed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

JOBS = "/home/923873155/Sem 4 Research/SkillLLM_v2/Extract_jobDesc/data/job_descriptions.jsonl"
NGRAM = 8
_TOK = re.compile(r"[a-z0-9]+")


def norm_tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def shingles(tokens: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def split_chunks(text: str, min_words: int = 4, max_chars: int = 300) -> list[str]:
    chunks: list[str] = []
    for line in re.split(r"[\r\n]+", text):
        line = re.sub(r"^[\s\-\*•·●>#]+", "", line).strip()
        if not line:
            continue
        parts = [line] if len(line) <= max_chars else re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9(])", line)
        for part in parts:
            part = part.strip()
            if len(part.split()) >= min_words and len(part) <= max_chars and re.search(r"[A-Za-z]", part):
                chunks.append(part)
    return chunks


def overlapping(posting_shingles: set, sentence_shingles: list[set], threshold: float = 0.5) -> bool:
    for sh in sentence_shingles:
        if sh and len(sh & posting_shingles) / len(sh) >= threshold:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=JOBS)
    ap.add_argument("--n-postings", type=int, default=600)
    ap.add_argument("--chunks-per-posting", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out-dir", default="artifacts/skillllm_v2/qlora_loop_v4_distill/pseudo_pool")
    ap.add_argument("--counts-out", default="reports/skillllm_v2/qlora_loop_v4_corpus_counts.json")
    args = ap.parse_args()

    sentences: dict[str, list[list[str]]] = {}
    exact: set[str] = set()
    for split in ("train", "dev", "test"):
        docs = load_skillspan_split(Path("data/source/skillspan_hf"), split)
        sentences[split] = [norm_tokens(d["text"]) for d in docs]
        if split in ("dev", "test"):
            exact.update(" ".join(t) for t in sentences[split])
    long_sent = [t for split in sentences.values() for t in split if len(t) >= NGRAM]
    # index sentence shingles by first shingle -> cheap candidate lookup via an inverted index of all shingles
    sent_sh = [shingles(t) for t in long_sent]
    inv: dict[tuple, list[int]] = {}
    for i, sh in enumerate(sent_sh):
        for g in sh:
            inv.setdefault(g, []).append(i)

    with open(args.jobs, encoding="utf-8") as fh:  # iterate on "\n" only (str.splitlines would split inside strings)
        postings = [json.loads(line) for line in fh if line.strip()]
    kept, removed = [], 0
    for idx, post in enumerate(postings):
        toks = norm_tokens(post.get("job_description_text") or "")
        psh = shingles(toks)
        hit: dict[int, int] = {}
        for g in psh:
            for i in inv.get(g, ()):
                hit[i] = hit.get(i, 0) + 1
        if any(c / len(sent_sh[i]) >= 0.5 for i, c in hit.items()):
            removed += 1
        else:
            kept.append(idx)

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(kept, min(args.n_postings, len(kept))))
    chunks_out, dropped_exact, total_chunks = [], 0, 0
    for idx in chosen:
        cand = split_chunks(postings[idx].get("job_description_text") or "")
        total_chunks += len(cand)
        ok = []
        for c in cand:
            if " ".join(norm_tokens(c)) in exact:
                dropped_exact += 1
            else:
                ok.append(c)
        for k, c in enumerate(rng.sample(ok, min(args.chunks_per_posting, len(ok)))):
            chunks_out.append({"id": f"pool-{idx:04d}-{k}", "text": c})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for row in chunks_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {
        "postings_total": len(postings),
        "skillspan_sentences_ge8_tokens": len(long_sent),
        "overlap_rule": f">= 50% of a SkillSpan (train/dev/test) sentence's {NGRAM}-gram shingles (sentences with >= {NGRAM} normalised tokens) occur in the posting",
        "postings_removed_for_overlap": removed,
        "postings_surviving": len(kept),
        "postings_sampled": len(chosen),
        "chunks_candidate_in_sampled_postings": total_chunks,
        "chunks_dropped_equal_to_skillspan_dev_test_sentence": dropped_exact,
        "chunks_selected": len(chunks_out),
        "chunks_per_posting_cap": args.chunks_per_posting,
        "seed": args.seed,
        "note": "counts only; posting text is private and is not stored in reports/docs/git",
    }
    Path(args.counts_out).write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
