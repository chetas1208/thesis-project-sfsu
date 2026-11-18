"""Deterministic, label-preserving input perturbations with exact span-offset remapping (L03_robustness).

Representation: a document is ``tokens`` + ``seps`` (len(tokens)+1 separator strings: before the first token, between
tokens, after the last) + gold spans as TOKEN ranges ``(i, j, label)`` (half-open). Text = seps[0] + tok0 + seps[1] + ...
+ tok[n-1] + seps[n]. Every perturbation edits tokens / separators / inserts or deletes whole tokens and re-indexes the
spans, so character offsets are recomputed, never guessed; ``validate`` asserts text[start:end] == the span's tokens
joined by its internal separators for every span. SkillSpan sentences are already word-tokenised, and gold spans are
token-aligned, which makes this exact.

Families (one variant each unless noted; see Decisions.md D-A3-04 for the rationale and validity caveats):
  identity            unchanged copy (determinism control: anchor output must equal the clean run)
  whitespace          separators OUTSIDE spans -> double space / tab / newline (p=0.3), leading/trailing whitespace
  bullets_prose       bullet-like line -> prose (drop leading marker, add final period) else prose -> bullet ("• " prefix,
                      newline before, final period dropped)
  punct_detok         realistic punctuation spacing: ",.;:!?)" glued to the previous token, "(" to the next (outside spans)
  punct_strip         delete punctuation tokens outside spans
  case_lower / case_upper / case_title   whole text (spans included; gold surface changes, still the same skill)
  typo_outside        one realistic typo (swap/delete/duplicate/keyboard-neighbour) in ~15% of eligible context words
  typo_inside         one typo in one word of each gold span with p=0.5 (>=1 span); gold surface changes, still a skill
  boilerplate         a synthetic, skill-free company/application sentence inserted at the start or end
  distractor          a synthetic sentence with a skill-like NAME used as a non-skill (person/company/place)
  alias_sub           gold span surface replaced by another alias of the same concept (repo taxonomy multi-alias groups +
                      ESCO preferred/alt labels when available); spans updated
  paraphrase_context  rule-based synonym substitution of context words only (outside spans)
  reorder_clauses     (a) sections: segments between sentence-internal ; : . ! ? tokens / mid-text bullet markers are
                      shuffled (seeded, != identity; delimiters, leading bullets and the final full stop stay in place);
                      (b) if impossible, coordination swap "A and B" -> "B and A" for two gold spans A, B;
                      else not_available (SkillSpan HF has no comma tokens, so most sentences have one section)
Inapplicable cases return ``None`` with a reason (reported as not_available, never silently dropped).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FAMILIES = ("identity", "whitespace", "bullets_prose", "punct_detok", "punct_strip", "case_lower", "case_upper",
            "case_title", "typo_outside", "typo_inside", "boilerplate", "distractor", "alias_sub", "paraphrase_context",
            "reorder_clauses")
# families whose gold surface changes inside spans (still the same skill, but exact string differs)
SURFACE_CHANGING = {"case_lower", "case_upper", "case_title", "typo_inside", "alias_sub"}

PLACEHOLDER = re.compile(r"^<[A-Z_]+>$")
PUNCT_CLOSE = {",", ".", ";", ":", "!", "?", ")", "]"}
PUNCT_OPEN = {"(", "["}
PUNCT_ALL = PUNCT_CLOSE | PUNCT_OPEN | {"-", "/", "&", "'", '"'}
BULLET_MARKERS = {"-", "*", "•", "·", "o", "–", "—", "+", "\x95"}
# SkillSpan (HF) sentences contain no comma tokens: sections are split at standalone sentence-internal punctuation
SECTION_DELIMS = {",", ";", ":", ".", "!", "?", "|"}
COORDINATORS = {"and", "or", "&", "/"}

# Synthetic text written for this evaluation (no real job postings). Checked to contain no skill mentions.
BOILERPLATE = [
    "Our offices are located close to public transport and have free parking .",
    "Apply today through our careers page and we will get back to you within two weeks .",
    "The position is full time and based in our main office .",
    "We offer a pension scheme , a canteen and twenty five days of holiday .",
    "Please include your earliest possible start date in your application .",
    "The company was founded more than thirty years ago and has grown steadily since .",
    "Interviews will be held on a rolling basis until the position is filled .",
    "Questions about the application process can be sent by email .",
    "Relocation support is available for candidates moving from abroad .",
    "The salary will be agreed upon individually .",
]
# Skill-like words used as names (person / company / place / product brand), i.e. NOT skill mentions in context.
# Validity risk (documented): an annotator could still read some of these as skills; see the human validity sample.
DISTRACTORS = [
    "Please send your application to Ruby Hansen at Agile Dairy Ltd .",
    "The team meets every Friday at the Java Lounge cafe next to the office .",
    "Our sister company Python Peak Outdoor sells hiking equipment .",
    "Lunch is provided by Spark Avenue Catering .",
    "The office is on Swift Street , opposite the Oracle Bakery .",
    "Contact our receptionist , Grace Excel , if you have questions about parking .",
    "Staff discounts apply at Scala Fashion and Rust Hardware stores .",
    "The building is managed by Docker Brothers Property Services .",
]
# Context-only paraphrases (lower-case keys; case of the first letter is preserved).
PARAPHRASE = {
    "experience": "background", "strong": "solid", "excellent": "outstanding", "good": "solid", "ability": "capability",
    "required": "needed", "requirements": "prerequisites", "candidate": "applicant", "candidates": "applicants",
    "role": "position", "job": "position", "responsible": "accountable", "ensure": "make sure", "including": "such as",
    "looking": "searching", "preferred": "desirable", "plus": "bonus", "team": "group", "company": "firm",
    "opportunity": "chance", "years": "yrs", "must": "should", "help": "assist", "work": "operate", "working": "operating",
    "develop": "build", "developing": "building", "great": "excellent", "new": "fresh", "big": "large", "large": "big",
    "within": "inside", "across": "throughout", "provide": "offer", "join": "become part of", "currently": "presently",
    "position": "role", "skills": "abilities", "knowledge": "understanding", "also": "additionally", "many": "numerous",
}
KEYBOARD = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "adwx", "d": "sfec", "f": "dgrv", "g": "fhtb", "h": "gjyn", "j": "hkum", "k": "jli", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


@dataclass
class TDoc:
    id: str
    tokens: List[str]
    seps: List[str]
    spans: List[Tuple[int, int, str]]  # token ranges, half-open
    meta: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "TDoc":
        return TDoc(self.id, list(self.tokens), list(self.seps), list(self.spans), dict(self.meta))

    # ------------------------------------------------------------------ rendering
    def render(self) -> Tuple[str, List[Tuple[int, int]]]:
        parts, offs, pos = [self.seps[0]], [], len(self.seps[0])
        for k, tok in enumerate(self.tokens):
            offs.append((pos, pos + len(tok)))
            parts.append(tok)
            pos += len(tok)
            parts.append(self.seps[k + 1])
            pos += len(self.seps[k + 1])
        return "".join(parts), offs

    def char_spans(self) -> List[Dict[str, Any]]:
        text, offs = self.render()
        out = []
        for i, j, label in self.spans:
            s, e = offs[i][0], offs[j - 1][1]
            out.append({"surface_form": text[s:e], "label": label, "start": s, "end": e})
        return sorted(out, key=lambda x: (x["start"], x["end"], x["label"]))

    def in_span(self) -> List[bool]:
        flags = [False] * len(self.tokens)
        for i, j, _ in self.spans:
            for k in range(i, j):
                flags[k] = True
        return flags

    def internal_sep(self) -> List[bool]:
        """internal_sep[k] is True if separator k (between token k-1 and k) lies inside some span."""
        flags = [False] * len(self.seps)
        for i, j, _ in self.spans:
            for k in range(i + 1, j):
                flags[k] = True
        return flags


def from_skillspan(doc: Dict[str, Any]) -> TDoc:
    """SkillSpan doc (load_skillspan_split row: tokens joined by single spaces) -> TDoc with token-range gold spans."""
    tokens = list(doc["tokens"])
    seps = [""] + [" "] * (len(tokens) - 1) + [""] if tokens else [""]
    t = TDoc(str(doc["id"]), tokens, seps, [], {"source": doc.get("source")})
    text, offs = t.render()
    if text != doc["text"]:
        raise ValueError(f"{doc['id']}: text does not match token join")
    start_idx = {s: k for k, (s, _) in enumerate(offs)}
    end_idx = {e: k for k, (_, e) in enumerate(offs)}
    spans = []
    for sp in doc["gold_spans"]:
        if sp["start"] not in start_idx or sp["end"] not in end_idx:
            raise ValueError(f"{doc['id']}: gold span not token aligned: {sp}")
        spans.append((start_idx[sp["start"]], end_idx[sp["end"]] + 1, sp["label"]))
    t.spans = sorted(spans)
    return t


def validate(t: TDoc, base: Optional[TDoc] = None, surface_may_change: bool = False) -> None:
    text, offs = t.render()
    assert len(t.seps) == len(t.tokens) + 1, "separator count"
    for i, j, label in t.spans:
        assert 0 <= i < j <= len(t.tokens), f"span range {i},{j}"
        s, e = offs[i][0], offs[j - 1][1]
        expect = "".join(t.tokens[k] + (t.seps[k + 1] if k < j - 1 else "") for k in range(i, j))
        assert text[s:e] == expect, "offset remap"
    if base is not None:
        assert sorted(l for *_, l in t.spans) == sorted(l for *_, l in base.spans), "label multiset changed"
        if not surface_may_change:
            a = sorted((x["surface_form"], x["label"]) for x in base.char_spans())
            b = sorted((x["surface_form"], x["label"]) for x in t.char_spans())
            assert a == b, "span surfaces changed"


# ------------------------------------------------------------------ token edit primitives
def _insert_tokens(t: TDoc, at: int, new_tokens: Sequence[str], new_seps: Sequence[str]) -> TDoc:
    """Insert tokens before token index ``at`` (0..n). ``new_seps`` = separators AFTER each new token. Spans shift."""
    n = len(new_tokens)
    if len(new_seps) != n or n == 0:
        raise ValueError("need one separator per new token")
    for i, j, _ in t.spans:
        if i < at < j:
            raise ValueError("insertion inside a span")
    out = t.copy()
    out.tokens = t.tokens[:at] + list(new_tokens) + t.tokens[at:]
    if at == len(t.tokens) and t.tokens:
        # append: a space joins the old last token to the block; new_seps[-1] becomes the trailing separator
        out.seps = t.seps[:at] + [" "] + list(new_seps)
    else:
        out.seps = t.seps[: at + 1] + list(new_seps) + t.seps[at + 1:]
    out.spans = [(i + n if i >= at else i, j + n if j > at else j, l) for i, j, l in t.spans]
    return out


def _delete_token(t: TDoc, k: int) -> TDoc:
    if t.in_span()[k]:
        raise ValueError("cannot delete a span token")
    out = t.copy()
    out.tokens = t.tokens[:k] + t.tokens[k + 1:]
    # the separators around the deleted token merge: keep the one before it; for the last token keep the trailing one
    if k == len(t.tokens) - 1:
        out.seps = t.seps[:k] + [t.seps[-1]]
    else:
        out.seps = t.seps[: k + 1] + t.seps[k + 2:]
    out.spans = [(i - 1 if i > k else i, j - 1 if j > k else j, l) for i, j, l in t.spans]
    return out


def _replace_token_range(t: TDoc, i: int, j: int, new_tokens: Sequence[str]) -> TDoc:
    """Replace tokens [i, j) by ``new_tokens`` joined with single spaces; spans equal to [i, j) are resized, spans that
    partially overlap the range make the edit invalid (ValueError)."""
    if not new_tokens:
        raise ValueError("replacement must be non-empty")
    d = len(new_tokens) - (j - i)
    spans = []
    for a, b, l in t.spans:
        if (a, b) == (i, j):
            spans.append((i, i + len(new_tokens), l))
        elif b <= i:
            spans.append((a, b, l))
        elif a >= j:
            spans.append((a + d, b + d, l))
        elif a <= i and b >= j:  # span strictly contains the range
            spans.append((a, b + d, l))
        else:
            raise ValueError("partial overlap")
    out = t.copy()
    out.tokens = t.tokens[:i] + list(new_tokens) + t.tokens[j:]
    out.seps = t.seps[: i + 1] + [" "] * (len(new_tokens) - 1) + t.seps[j:]
    out.spans = spans
    return out


# ------------------------------------------------------------------ helpers
def doc_rng(doc_id: str, family: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{seed}|{family}|{doc_id}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def is_word(tok: str) -> bool:
    return tok.isalpha() and not PLACEHOLDER.match(tok)


def typo(word: str, rng: random.Random) -> str:
    """One realistic typo on a word of length >= 4 (first letter kept): adjacent swap, deletion, duplication, keyboard."""
    ops = ["swap", "delete", "duplicate", "keyboard"]
    for _ in range(10):
        op = rng.choice(ops)
        k = rng.randrange(1, len(word) - 1)
        if op == "swap":
            new = word[:k] + word[k + 1] + word[k] + word[k + 2:]
        elif op == "delete":
            new = word[:k] + word[k + 1:]
        elif op == "duplicate":
            new = word[:k] + word[k] + word[k:]
        else:
            c = word[k]
            near = KEYBOARD.get(c.lower())
            if not near:
                continue
            r = rng.choice(near)
            new = word[:k] + (r.upper() if c.isupper() else r) + word[k + 1:]
        if new != word:
            return new
    return word[:-1] + word[-1] * 2


def _match_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def norm_surface(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


class AliasTable:
    """concept -> set of alias surfaces. Sources: repo taxonomy groups with >= 2 aliases + (optional) ESCO labels."""

    def __init__(self, groups: Dict[str, List[str]], sources: List[Dict[str, Any]]):
        self.groups = groups
        self.sources = sources
        self.index: Dict[str, List[str]] = {}
        for cid, aliases in groups.items():
            for a in aliases:
                self.index.setdefault(norm_surface(a), []).append(cid)

    @classmethod
    def load(cls, repo_root: Path, esco_path: Optional[Path] = None, esco_min_rows: int = 10000) -> "AliasTable":
        groups: Dict[str, List[str]] = {}
        sources = []
        tax = repo_root / "data/skillllm_v2/taxonomy/skill_aliases.jsonl"
        if tax.exists():
            tmp: Dict[str, set] = {}
            for line in tax.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("source") != "manual_curated":  # skip aliases mined from data (some from anchor TEST output)
                        continue
                    tmp.setdefault("repo:" + r["canonical_skill_id"], set()).add(r["alias"].strip())
            groups.update({k: sorted(v) for k, v in tmp.items() if len(v) > 1})
            sources.append({"path": str(tax.relative_to(repo_root)), "groups_used": sum(1 for k in groups if k.startswith("repo:"))})
        if esco_path is not None and esco_path.exists():
            rows = [json.loads(x) for x in esco_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            if len(rows) >= esco_min_rows:
                n = 0
                for r in rows:
                    labels = [r.get("preferred_label", "")] + list(r.get("alt_labels") or [])
                    labels = sorted({l.strip() for l in labels if l and "(" not in l and len(l.split()) <= 5})
                    if len(labels) > 1:
                        groups["esco:" + r["uri"].rsplit("/", 1)[-1]] = labels
                        n += 1
                sources.append({"path": str(esco_path), "rows": len(rows), "groups_used": n})
            else:
                sources.append({"path": str(esco_path), "rows": len(rows), "groups_used": 0,
                                "skipped": f"incomplete download (< {esco_min_rows} rows)"})
        return cls(groups, sources)

    def alternatives(self, surface: str) -> List[str]:
        key = norm_surface(surface)
        alts = set()
        for cid in self.index.get(key, []):
            for a in self.groups[cid]:
                if norm_surface(a) == key:
                    continue
                if cid.startswith("esco:") and not esco_alias_ok(surface, a):
                    continue  # ESCO alt labels include narrower concepts / other tools; keep true aliases only
                alts.add(a)
        return sorted(alts)


GENERIC_EXTRA = {"programming", "language", "languages", "software", "development", "skills", "skill", "management",
                 "system", "systems", "framework", "database", "technology", "technologies", "tool", "tools", "platform",
                 "computer", "js"}


def _words(s: str) -> List[str]:
    return [w for w in re.split(r"[^0-9a-z#+]+", s.lower()) if w]


def esco_alias_ok(orig: str, alt: str) -> bool:
    """True alias relations only: acronym <-> expansion, spelling variant, or the same head with generic extra words."""
    wo, wa = _words(orig), _words(alt)
    if not wo or not wa:
        return False
    so, sa = "".join(wo), "".join(wa)
    if so == sa:  # React.js / ReactJS, Node.js / NodeJS
        return True
    if len(wo) >= 2 and len(wa) == 1 and sa == "".join(w[0] for w in wo):  # Information Technology -> IT
        return True
    if len(wa) >= 2 and len(wo) == 1 and so == "".join(w[0] for w in wa):
        return True
    o, a = set(wo), set(wa)
    if o < a and (a - o) <= GENERIC_EXTRA:  # TypeScript -> TypeScript programming language
        return True
    if a < o and (o - a) <= GENERIC_EXTRA:  # communication skills -> communication
        return True
    return False


# ------------------------------------------------------------------ families
Result = Tuple[Optional[TDoc], str]  # (perturbed doc or None, description / not_available reason)


def p_identity(t: TDoc, rng: random.Random, **_: Any) -> Result:
    return t.copy(), "unchanged copy"


def p_whitespace(t: TDoc, rng: random.Random, **_: Any) -> Result:
    out = t.copy()
    internal = t.internal_sep()
    changed = 0
    eligible = [k for k in range(1, len(t.seps) - 1) if not internal[k]]
    for k in eligible:
        if rng.random() < 0.3:
            out.seps[k] = rng.choice(["  ", "\t", " \n", "   "])
            changed += 1
    if eligible and not changed:  # at least one inter-token change when possible
        out.seps[rng.choice(eligible)] = rng.choice(["  ", "\t", " \n", "   "])
        changed = 1
    out.seps[0] = rng.choice(["\n", "  ", "\n\t"]) + t.seps[0]
    out.seps[-1] = t.seps[-1] + rng.choice(["\n", "  ", " \n\n"])
    return out, f"{changed} inter-token separators changed + leading/trailing whitespace"


def p_bullets_prose(t: TDoc, rng: random.Random, **_: Any) -> Result:
    if not t.tokens:
        return None, "empty document"
    flags = t.in_span()
    if t.tokens[0] in BULLET_MARKERS and not flags[0] and len(t.tokens) > 1:
        out = t
        while len(out.tokens) > 1 and out.tokens[0] in BULLET_MARKERS and not out.in_span()[0]:
            out = _delete_token(out, 0)  # all leading markers ("* •")
        out.seps[0] = ""
        if out.tokens[-1] not in PUNCT_CLOSE:
            out = _insert_tokens(out, len(out.tokens), ["."], [out.seps[-1]])  # SkillSpan spacing "word ."
        return out, "bullet -> prose (leading marker removed, final period ensured)"
    out = t.copy()
    if out.tokens[-1] == "." and not flags[-1] and len(out.tokens) > 1:
        out = _delete_token(out, len(out.tokens) - 1)
    out = _insert_tokens(out, 0, [rng.choice(["•", "-", "*"])], [" "])
    out.seps[0] = "\n"
    return out, "prose -> bullet (marker prefix, newline, final period dropped)"


def p_punct_detok(t: TDoc, rng: random.Random, **_: Any) -> Result:
    out = t.copy()
    internal = t.internal_sep()
    changed = 0
    for k, tok in enumerate(t.tokens):
        if tok in PUNCT_CLOSE and k > 0 and not internal[k] and out.seps[k] != "":
            out.seps[k] = ""
            changed += 1
        if tok in PUNCT_OPEN and k + 1 < len(t.tokens) and not internal[k + 1] and out.seps[k + 1] != "":
            out.seps[k + 1] = ""
            changed += 1
    if not changed:
        return None, "no punctuation token outside spans"
    return out, f"{changed} punctuation separators removed"


def p_punct_strip(t: TDoc, rng: random.Random, **_: Any) -> Result:
    flags = t.in_span()
    idx = [k for k, tok in enumerate(t.tokens) if tok in PUNCT_ALL and not flags[k]]
    if not idx or len(idx) == len(t.tokens):
        return None, "no removable punctuation token outside spans"
    out = t
    for k in sorted(idx, reverse=True):
        out = _delete_token(out, k)
    return out, f"{len(idx)} punctuation tokens removed"


def _case(t: TDoc, fn) -> TDoc:  # noqa: ANN001
    out = t.copy()
    out.tokens = [tok if PLACEHOLDER.match(tok) else fn(tok) for tok in t.tokens]
    return out


def p_case_lower(t: TDoc, rng: random.Random, **_: Any) -> Result:
    out = _case(t, str.lower)
    return (out, "all tokens lower-cased") if out.tokens != t.tokens else (None, "already lower-case")


def p_case_upper(t: TDoc, rng: random.Random, **_: Any) -> Result:
    out = _case(t, str.upper)
    return (out, "all tokens upper-cased") if out.tokens != t.tokens else (None, "already upper-case")


def p_case_title(t: TDoc, rng: random.Random, **_: Any) -> Result:
    out = _case(t, lambda s: s[:1].upper() + s[1:])
    return (out, "first letter of every token upper-cased (rest unchanged)") if out.tokens != t.tokens else (None, "already title-case")


def p_typo_outside(t: TDoc, rng: random.Random, **_: Any) -> Result:
    flags = t.in_span()
    elig = [k for k, tok in enumerate(t.tokens) if not flags[k] and is_word(tok) and len(tok) >= 4]
    if not elig:
        return None, "no context word of length >= 4 outside spans"
    n = max(1, round(0.15 * len(elig)))
    pick = sorted(rng.sample(elig, n))
    out = t.copy()
    for k in pick:
        out.tokens[k] = typo(t.tokens[k], rng)
    return out, f"{n} context words with one typo each"


def p_typo_inside(t: TDoc, rng: random.Random, **_: Any) -> Result:
    out = t.copy()
    done_tokens: set = set()
    cands = []
    for i, j, _ in t.spans:
        elig = [k for k in range(i, j) if is_word(t.tokens[k]) and len(t.tokens[k]) >= 4]
        if elig:
            cands.append(elig)
    if not cands:
        return None, "no span word of length >= 4"
    chosen = [c for c in cands if rng.random() < 0.5] or [rng.choice(cands)]
    for elig in chosen:
        free = [k for k in elig if k not in done_tokens]
        if not free:
            continue
        k = rng.choice(free)
        out.tokens[k] = typo(t.tokens[k], rng)
        done_tokens.add(k)
    return out, f"{len(done_tokens)} span words with one typo each"


def _insert_sentence(t: TDoc, sentence: str, rng: random.Random) -> Tuple[TDoc, str]:
    """Insert a synthetic sentence on its own line (postings are line-structured) at the start or the end."""
    toks = sentence.split()
    if rng.random() < 0.5 or not t.tokens:
        out = _insert_tokens(t, 0, toks, [" "] * (len(toks) - 1) + ["\n"])
        return out, "start"
    n = len(t.tokens)
    out = _insert_tokens(t, n, toks, [" "] * (len(toks) - 1) + [t.seps[-1]])
    out.seps[n] = "\n"
    return out, "end"


def p_boilerplate(t: TDoc, rng: random.Random, **_: Any) -> Result:
    sent = rng.choice(BOILERPLATE)
    out, where = _insert_sentence(t, sent, rng)
    return out, f"boilerplate inserted at {where}: {sent!r}"


def p_distractor(t: TDoc, rng: random.Random, **_: Any) -> Result:
    sent = rng.choice(DISTRACTORS)
    out, where = _insert_sentence(t, sent, rng)
    return out, f"distractor inserted at {where}: {sent!r}"


def p_alias_sub(t: TDoc, rng: random.Random, aliases: Optional[AliasTable] = None, **_: Any) -> Result:
    if aliases is None:
        return None, "no alias table"
    text, offs = t.render()
    opts = []
    for i, j, label in t.spans:
        surface = text[offs[i][0]:offs[j - 1][1]]
        alts = aliases.alternatives(surface)
        if alts:
            opts.append((i, j, label, surface, alts))
    if not opts:
        return None, "no gold span with a taxonomy alias"
    out = t
    descr = []
    for i, j, label, surface, alts in sorted(opts, key=lambda x: -x[0]):  # right-to-left keeps earlier indices valid
        new = rng.choice(alts)
        try:
            out = _replace_token_range(out, i, j, new.split())
        except ValueError:
            continue  # span partially overlaps another span: skip this substitution
        descr.append(f"{surface!r}->{new!r}")
    if not descr:
        return None, "alias spans overlap other spans"
    return out, "; ".join(reversed(descr))


def p_paraphrase_context(t: TDoc, rng: random.Random, **_: Any) -> Result:
    flags = t.in_span()
    hits = [k for k, tok in enumerate(t.tokens) if not flags[k] and tok.lower() in PARAPHRASE]
    if not hits:
        return None, "no paraphrasable context word"
    out = t
    descr = []
    for k in sorted(hits, reverse=True):
        src = t.tokens[k]
        repl = _match_case(src, PARAPHRASE[src.lower()])
        out = _replace_token_range(out, k, k + 1, repl.split())
        descr.append(f"{src}->{repl}")
    return out, "; ".join(reversed(descr))


def _permute_pieces(t: TDoc, pieces: List[Tuple[int, int]], perm: List[int]) -> TDoc:
    """Reorder contiguous token pieces (covering 0..n in order) as ``perm``. Separators inside a piece are kept; the
    separator at each slot boundary stays with the SLOT (original layout). Every span must lie inside one piece."""
    for i, j, _ in t.spans:
        if not any(a <= i and j <= b for a, b in pieces):
            raise ValueError("span crosses a piece boundary")
    new_tokens: List[str] = []
    new_seps: List[str] = [t.seps[0]]
    new_spans: List[Tuple[int, int, str]] = []
    for slot, p in enumerate(perm):
        a, b = pieces[p]
        if slot > 0:
            new_seps[-1] = t.seps[pieces[slot][0]]
        base = len(new_tokens)
        for k in range(a, b):
            new_tokens.append(t.tokens[k])
            new_seps.append(t.seps[k + 1] if k < b - 1 else "")
        new_spans += [(base + i - a, base + j - a, l) for i, j, l in t.spans if a <= i and j <= b]
    new_seps[-1] = t.seps[-1]
    return TDoc(t.id, new_tokens, new_seps, sorted(new_spans), dict(t.meta))


def _runs(flags: List[bool], lo: int, hi: int) -> List[Tuple[int, int, bool]]:
    out: List[Tuple[int, int, bool]] = []
    k = lo
    while k < hi:
        m = k
        while m < hi and flags[m] == flags[k]:
            m += 1
        out.append((k, m, flags[k]))
        k = m
    return out


def p_reorder_clauses(t: TDoc, rng: random.Random, **_: Any) -> Result:
    """Mode 1 (sections): runs between sentence-internal ; : . ! ? tokens or mid-text bullet markers are shuffled.
    Mode 2 (coordination_swap, only if mode 1 is impossible): "A and/or/&// B" where A ends and B starts a gold span ->
    "B and A". Otherwise not_available. Leading bullet markers and the final full stop never move."""
    flags = t.in_span()
    n = len(t.tokens)
    end = n - 1 if n and t.tokens[-1] in {".", "!", "?"} and not flags[-1] else n
    start = 0
    while start < end and t.tokens[start] in BULLET_MARKERS and not flags[start]:
        start += 1
    is_delim = [(tok in SECTION_DELIMS or tok in BULLET_MARKERS) and not flags[k] for k, tok in enumerate(t.tokens)]
    runs = _runs(is_delim, start, end)
    clause_runs = [r for r in runs if not r[2]]
    pieces = ([(0, start)] if start > 0 else []) + [(a, b) for a, b, _ in runs] + ([(end, n)] if end < n else [])
    crossing = any(not any(a <= i and j <= b for a, b in pieces) for i, j, _ in t.spans)
    if len(clause_runs) >= 2 and not crossing and len({tuple(t.tokens[a:b]) for a, b, _ in clause_runs}) >= 2:
        movable = [pieces.index((a, b)) for a, b, _ in clause_runs]
        texts = {p: tuple(t.tokens[pieces[p][0]:pieces[p][1]]) for p in movable}
        new_order = list(movable)
        for _ in range(20):
            rng.shuffle(new_order)
            if [texts[p] for p in new_order] != [texts[p] for p in movable]:
                break
        perm = list(range(len(pieces)))
        for slot, p in zip(movable, new_order):
            perm[slot] = p
        return _permute_pieces(t, pieces, perm), f"sections: {len(clause_runs)} segments reordered"
    cands = []
    for k in range(1, n - 1):
        if t.tokens[k].lower() not in COORDINATORS or flags[k]:
            continue
        ends = [i for i, j, _ in t.spans if j == k]
        starts = [j for i, j, _ in t.spans if i == k + 1]
        if not ends or not starts:
            continue
        ia, jb = min(ends), max(starts)
        ok = all((i >= ia and j <= k) or (i >= k + 1 and j <= jb) or j <= ia or i >= jb for i, j, _ in t.spans)
        if ok and t.tokens[ia:k] != t.tokens[k + 1:jb]:
            cands.append((ia, k, jb))
    if not cands:
        return None, "single section and no 'A and/or B' coordination of two gold spans"
    ia, k, jb = rng.choice(cands)
    pieces = [(a, b) for a, b in [(0, ia), (ia, k), (k, k + 1), (k + 1, jb), (jb, n)] if b > a]
    ai, bi = pieces.index((ia, k)), pieces.index((k + 1, jb))
    perm = list(range(len(pieces)))
    perm[ai], perm[bi] = bi, ai
    out = _permute_pieces(t, pieces, perm)
    return out, f"coordination_swap: {' '.join(t.tokens[ia:k])!r} <-> {' '.join(t.tokens[k + 1:jb])!r}"


PERTURB = {
    "identity": p_identity, "whitespace": p_whitespace, "bullets_prose": p_bullets_prose, "punct_detok": p_punct_detok,
    "punct_strip": p_punct_strip, "case_lower": p_case_lower, "case_upper": p_case_upper, "case_title": p_case_title,
    "typo_outside": p_typo_outside, "typo_inside": p_typo_inside, "boilerplate": p_boilerplate, "distractor": p_distractor,
    "alias_sub": p_alias_sub, "paraphrase_context": p_paraphrase_context, "reorder_clauses": p_reorder_clauses,
}
assert tuple(PERTURB) == FAMILIES


def perturb(t: TDoc, family: str, seed: int = 42, aliases: Optional[AliasTable] = None) -> Result:
    rng = doc_rng(t.id, family, seed)
    out, note = PERTURB[family](t, rng, aliases=aliases)
    if out is None:
        return None, note
    out.id = f"{t.id}::{family}"
    validate(out, t, surface_may_change=family in SURFACE_CHANGING)
    return out, note


def to_row(base: TDoc, out: TDoc, family: str, note: str) -> Dict[str, Any]:
    text, offs = out.render()
    btext, _ = base.render()
    return {"id": out.id, "text": text, "tokens": out.tokens, "token_offsets": [list(o) for o in offs],
            "gold_spans": out.char_spans(),
            "meta": {"base_id": base.id, "family": family, "note": note, "source": base.meta.get("source"),
                     "base_text": btext, "n_gold": len(out.spans)}}


def stratified_sample(docs: Sequence[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Proportional allocation over (source x gold-span-count bucket) strata, seeded, largest-remainder rounding."""
    import numpy as np

    def bucket(k: int) -> str:
        return "1" if k == 1 else ("2-3" if k <= 3 else "4+")
    strata: Dict[Tuple[str, str], List[int]] = {}
    for idx, d in enumerate(docs):
        strata.setdefault((str(d.get("source")), bucket(len(d["gold_spans"]))), []).append(idx)
    keys = sorted(strata)
    total = len(docs)
    quotas = {k: n * len(strata[k]) / total for k in keys}
    alloc = {k: int(quotas[k]) for k in keys}
    rest = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quotas[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for k in keys:
        pool = strata[k]
        chosen += sorted(rng.choice(pool, size=min(alloc[k], len(pool)), replace=False).tolist())
    return [docs[i] for i in sorted(chosen)]


def build(docs: Iterable[Dict[str, Any]], families: Sequence[str] = FAMILIES, seed: int = 42,
          aliases: Optional[AliasTable] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """-> (perturbed rows, not_available rows {base_id, family, reason})."""
    rows, na = [], []
    for d in docs:
        base = from_skillspan(d)
        for fam in families:
            out, note = perturb(base, fam, seed, aliases)
            if out is None:
                na.append({"base_id": base.id, "family": fam, "reason": note})
            else:
                rows.append(to_row(base, out, fam, note))
    return rows, na


def main() -> None:  # pragma: no cover - CLI
    import argparse
    import os
    import sys

    from skillllm_v2.evals.research.common import REPO_ROOT, sha256_file
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO_ROOT / "inputs/evals/research/L03_robustness"))
    ap.add_argument("--esco", default=str(REPO_ROOT / "data/external/esco/v1.2.1/skills_en.jsonl"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mark-ready", action="store_true")
    args = ap.parse_args()
    os.chdir(REPO_ROOT)
    test = [d for d in load_skillspan_split(Path("data/source/skillspan_hf"), "test") if d["gold_spans"]]
    n = 10 if args.smoke else args.n_docs
    sample = stratified_sample(test, n, args.seed)
    aliases = AliasTable.load(REPO_ROOT, Path(args.esco))
    rows, na = build(sample, FAMILIES, args.seed, aliases)
    out = Path(args.out) / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "perturbed_docs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "not_available.jsonl", "w", encoding="utf-8") as fh:
        for r in na:
            fh.write(json.dumps(r) + "\n")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["meta"]["family"]] = counts.get(r["meta"]["family"], 0) + 1
    manifest = {"seed": args.seed, "n_base_docs": len(sample), "base_ids": [d["id"] for d in sample],
                "families": list(FAMILIES), "rows_per_family": counts, "n_rows": len(rows), "n_not_available": len(na),
                "alias_sources": aliases.sources, "alias_groups": len(aliases.groups),
                "perturbed_docs_sha256": sha256_file(out / "perturbed_docs.jsonl"),
                "split_sha256": sha256_file(Path("data/source/skillspan_hf/test.json")), "argv": sys.argv}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "base_ids"}, indent=2))
    if args.mark_ready and not args.smoke:
        (out / "READY").write_text(manifest["perturbed_docs_sha256"] + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
