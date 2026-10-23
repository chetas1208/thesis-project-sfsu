#!/usr/bin/env python
"""Fail if text from the private job-posting corpus (or a flagged company name) appears in the repository.

    PYTHONPATH=src python scripts/check_no_private_text.py [--corpus PATH] [--root REPO] [--min-run 20] [--shingle 8]

Method
  * Every posting of the private corpus (``job_description_text``) is lower-cased and tokenised (``\\w+``).
    All overlapping ``--shingle``-token windows (default 8-grams) are indexed.
  * Every file returned by ``git ls-files -co --exclude-standard`` (tracked + untracked-but-not-ignored) is tokenised the same
    way.  Matching 8-grams that continue on the same diagonal (consecutive in both the file and a posting) are chained; a chain
    that spans >= ``--min-run`` consecutive tokens (default 20) is a leak.
  * Company names are matched as whole lower-case tokens.  They are stored here ONLY as SHA-256 digests so that this script does
    not itself contain them; ``--extra-name NAME`` adds more (compared in plain form for that run only).

The matched text is NEVER printed: the report lists file paths and counts only.  Exit status 1 on any leak (or unreadable
corpus unless ``--allow-missing-corpus``), 0 when clean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT.parent / "SkillLLM_v2/Extract_jobDesc/data/job_descriptions.jsonl"
TOKEN_RE = re.compile(r"\w+")
MAX_FILE_BYTES = 64 * 1024 * 1024

# sha256(lower-case name) of the company names that appeared in the removed provenance records
NAME_DIGESTS = {
    "665c1f3b43d9b0e44dbac4613a56666ba9b541b12b0ef33c2e755f1c1b3c992a",
    "7fc379543f07e198f873ec87727c837be6fa0c6ba527f8b80340e7cf76814620",
    "3cb882789f1a7e5703b1839f76b6e83b4990fe267e9164e37675881029f06e19",
    "4f8317ade42d96bf88a3f2fa531f6445d92432aa1e217e0a40881f532f15f287",
    "ff67070ca54b498c7a051f8721b05575ad56d60a271eff29b78cf1a29eb02646",
    "48272715346d6e3b46741a21e02212ef35debe1c3b8693fb2acb1077f32c09df",
    "d4690300ba7d6a12b9b5b77a7a13edfbfd7eb0b5aaa8c5672eb052abb9672b6e",
}


def _sha(s: str) -> str:
    return hashlib.sha256(s.lower().encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_index(texts, shingle: int = 8) -> dict[int, list[tuple[int, int]]]:
    """Map hash of each ``shingle``-token window -> [(posting_id, start_position), ...]."""
    index: dict[int, list[tuple[int, int]]] = {}
    for pid, text in enumerate(texts):
        toks = tokenize(text)
        for i in range(len(toks) - shingle + 1):
            index.setdefault(hash(tuple(toks[i:i + shingle])), []).append((pid, i))
    return index


def longest_run(tokens: list[str], index: dict[int, list[tuple[int, int]]], shingle: int = 8) -> int:
    """Longest stretch (in tokens) of ``tokens`` that is also a contiguous stretch of one indexed posting (0 if none)."""
    best = 0
    diag: dict[tuple[int, int], tuple[int, int]] = {}  # (posting, p - i) -> (last i, chain length in shingles)
    for i in range(len(tokens) - shingle + 1):
        hits = index.get(hash(tuple(tokens[i:i + shingle])))
        if not hits:
            continue
        for pid, p in hits:
            key = (pid, p - i)
            last, n = diag.get(key, (-2, 0))
            n = n + 1 if last == i - 1 else 1
            diag[key] = (i, n)
            if n + shingle - 1 > best:
                best = n + shingle - 1
    return best


def name_hits(tokens: list[str], digests: set[str], plain: set[str], lengths: set[int]) -> int:
    n = 0
    for t in tokens:
        if t in plain or (len(t) in lengths and _sha(t) in digests):
            n += 1
    return n


def iter_repo_files(root: Path):
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root, capture_output=True, check=True).stdout
    for rel in out.split(b"\0"):
        if rel:
            yield rel.decode("utf-8", "surrogateescape")


def read_text(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:8192]:
        return None  # binary
    return raw.decode("utf-8", "ignore")


def scan(root: Path, corpus_texts, *, shingle: int = 8, min_run: int = 20, name_digests: set[str] | None = None,
         plain_names: set[str] | None = None, skip: set[Path] | None = None, public_texts=()) -> dict:
    """Scan ``root`` (a git work tree). Returns {'files_scanned', 'run_leaks': {path: longest_run}, 'name_leaks': {path: count}}.

    ``public_texts``: texts that are public by design (e.g. the SkillSpan dataset, which was itself built from job postings and
    legitimately overlaps the private corpus). Shingles that also occur there are removed from the private index, so only overlap
    that is NOT explained by public data counts as a leak."""
    index = build_index(corpus_texts, shingle)
    if public_texts:
        for h in set(build_index(public_texts, shingle)):
            index.pop(h, None)
    digests = set(NAME_DIGESTS if name_digests is None else name_digests)
    plain = {n.lower() for n in (plain_names or set())}
    lengths = {len(n) for n in plain}
    # digest-only names: lengths unknown, so hash every token of plausible company-name length
    lengths |= set(range(3, 16)) if digests else set()
    skip = {p.resolve() for p in (skip or set())}
    res = {"files_scanned": 0, "run_leaks": {}, "name_leaks": {}}
    for rel in iter_repo_files(root):
        path = (root / rel)
        if path.resolve() in skip:
            continue
        text = read_text(path)
        if text is None:
            continue
        res["files_scanned"] += 1
        toks = tokenize(text)
        run = longest_run(toks, index, shingle)
        if run >= min_run:
            res["run_leaks"][rel] = run
        nh = name_hits(toks, digests, plain, lengths)
        if nh:
            res["name_leaks"][rel] = nh
    return res


def load_corpus(path: Path) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    t = json.loads(line).get("job_description_text") or ""
                except json.JSONDecodeError:
                    continue
                if t:
                    texts.append(t)
    return texts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--root", default=None, help="git work tree to scan (default: the enclosing repository root)")
    ap.add_argument("--min-run", type=int, default=20)
    ap.add_argument("--shingle", type=int, default=8)
    ap.add_argument("--extra-name", action="append", default=[])
    ap.add_argument("--allow-missing-corpus", action="store_true")
    ap.add_argument("--public", action="append", default=None,
                    help="file whose text is public by design (repeatable). Default: the checked-in SkillSpan train/dev/test export. "
                         "Pass --no-public to disable.")
    ap.add_argument("--no-public", action="store_true")
    a = ap.parse_args(argv)

    corpus = Path(a.corpus)
    if a.root:
        root = Path(a.root).resolve()
    else:  # the stage2 dir may carry its own (empty) .git; the real repository is the highest ancestor with one
        root = ROOT
        for cand in (ROOT.parent, ROOT.parent.parent):
            if (cand / ".git").exists():
                root = cand
    texts: list[str] = []
    if corpus.exists():
        texts = load_corpus(corpus)
    elif not a.allow_missing_corpus:
        print(f"corpus not found: {corpus}", file=sys.stderr)
        return 1
    public_paths = [] if a.no_public else [Path(x) for x in (a.public or [str(ROOT / "data/source/skillspan_hf" / f"{n}.json") for n in ("train", "dev", "test")])]
    public_texts = [pp.read_text(encoding="utf-8", errors="ignore") for pp in public_paths if pp.exists()]
    res = scan(root, texts, shingle=a.shingle, min_run=a.min_run, plain_names=set(a.extra_name), skip={corpus}, public_texts=public_texts)
    print(f"root={root} postings={len(texts)} public_files={len(public_texts)} files_scanned={res['files_scanned']} shingle={a.shingle} min_run={a.min_run}")
    for rel, run in sorted(res["run_leaks"].items()):
        print(f"LEAK run>={a.min_run} tokens (longest {run}): {rel}")
    for rel, n in sorted(res["name_leaks"].items()):
        print(f"LEAK flagged name x{n}: {rel}")
    bad = len(res["run_leaks"]) + len(res["name_leaks"])
    print("RESULT: " + (f"FAIL ({bad} file(s))" if bad else "clean"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
