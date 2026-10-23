"""Unit tests for scripts/check_no_private_text.py on synthetic data (no private corpus needed)."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_no_private_text", REPO / "scripts" / "check_no_private_text.py")
cnpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cnpt)

POSTING = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo "
           "sierra tango uniform victor whiskey xray yankee zulu one two three four five six seven eight nine ten")


def _repo(tmp_path: Path, files: dict[str, str], ignored: tuple[str, ...] = ()) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / ".gitignore").write_text("\n".join(ignored) + "\n", encoding="utf-8")
    return tmp_path


def test_longest_run_detects_contiguous_copy_and_ignores_short_overlap():
    idx = cnpt.build_index([POSTING], 8)
    toks = cnpt.tokenize("intro words " + " ".join(POSTING.split()[3:30]) + " outro words here")
    assert cnpt.longest_run(toks, idx, 8) == 27
    short = cnpt.tokenize("zzz " + " ".join(POSTING.split()[3:15]) + " yyy")
    assert cnpt.longest_run(short, idx, 8) == 12
    assert cnpt.longest_run(cnpt.tokenize("completely unrelated sentence about something else entirely, nothing shared"), idx, 8) == 0
    # a reordered copy is not one contiguous run
    words = POSTING.split()
    assert cnpt.longest_run(words[10:20] + words[0:10], idx, 8) < 20


def test_scan_flags_leaks_by_path_and_never_reports_text(tmp_path, capsys):
    root = _repo(tmp_path, {
        "leak.md": "Some notes. " + POSTING + " More notes.",
        "ok.md": "Nothing to see: " + " ".join(POSTING.split()[:10]),
        "named.txt": "We worked with Foocorp last year.",
        "ignored.txt": POSTING,
    }, ignored=("ignored.txt",))
    res = cnpt.scan(root, [POSTING], min_run=20, name_digests=set(), plain_names={"foocorp"})
    assert set(res["run_leaks"]) == {"leak.md"}
    assert set(res["name_leaks"]) == {"named.txt"}
    assert res["files_scanned"] >= 3 and "ignored.txt" not in res["run_leaks"]


def test_name_digests_match_without_plaintext():
    digest = cnpt._sha("Foocorp")
    toks = cnpt.tokenize("Foocorp and foocorp and Barcorp")
    assert cnpt.name_hits(toks, {digest}, set(), set(range(3, 16))) == 2


def test_main_exit_codes(tmp_path, capsys):
    import json
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"job_description_text": POSTING}) + "\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo, {"a.md": "clean text only"})
    assert cnpt.main(["--corpus", str(corpus), "--root", str(repo)]) == 0
    (repo / "b.md").write_text(POSTING, encoding="utf-8")
    assert cnpt.main(["--corpus", str(corpus), "--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "b.md" in out and "alpha bravo" not in out
    assert cnpt.main(["--corpus", str(tmp_path / "missing.jsonl"), "--root", str(repo)]) == 1


def test_public_text_overlap_is_not_a_leak(tmp_path):
    import importlib.util, subprocess
    spec = importlib.util.spec_from_file_location("cnp", Path(__file__).resolve().parents[2] / "scripts" / "check_no_private_text.py")
    cnp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cnp)
    sent = " ".join(f"w{i}" for i in range(30))
    (tmp_path / "pub.txt").write_text(sent)
    (tmp_path / "priv.txt").write_text(" ".join(f"z{i}" for i in range(30)))
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text(sent + " " + " ".join(f"z{i}" for i in range(30)))
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    corpus = [sent + " " + " ".join(f"z{i}" for i in range(30))]
    leak = cnp.scan(tmp_path, corpus, name_digests=set(), skip={tmp_path / "pub.txt", tmp_path / "priv.txt"})
    assert "a.txt" in leak["run_leaks"]                        # both parts overlap the corpus
    ok = cnp.scan(tmp_path, [sent], name_digests=set(), skip={tmp_path / "pub.txt", tmp_path / "priv.txt"}, public_texts=[sent])
    assert not ok["run_leaks"]                                 # fully explained by public data
    partial = cnp.scan(tmp_path, corpus, name_digests=set(), skip={tmp_path / "pub.txt", tmp_path / "priv.txt"}, public_texts=[sent])
    assert "a.txt" in partial["run_leaks"]                     # the private half still counts
