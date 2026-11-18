"""L07 profile compaction: does the compact structured profile keep what downstream decisions need? (H7)

Seeded synthetic user histories with generator-authored truth (held skills + evidence sentences, goals, gaps and
planted contradictions) are turned into representations and fed to the repository's real decision components:

* ``full``           the whole history text; its profile = every extracted skill fact (no compaction).
* ``compact@<b>``     production ``build_cached_context_packet`` (-> ``compact_context`` -> ``summarize_profile``);
                      ``@shipped`` = ``budget_tokens=None`` (what ``pipeline/inference.py`` ships), ``@f`` = budget of
                      ``f x`` the proxy-token count of the full history (``context.budget.count_tokens``).
* ``minimal``         JSON list of skill names only (no evidence, goals, confidence).
* ``recent@<f>``      naive baseline: most recent history lines that fit the same proxy budget.
* ``truth``           oracle profile (true held skills) - upper reference.

Downstream: ``match.job_matcher.JobMatcher`` over ``match_evals.make_jobs`` (authored job grades on TRUE skills),
``match.gap_detector.GapDetector.detect`` and ``match.course_recommender.CourseRecommender`` over
``inputs/courses/course_catalog_large.jsonl`` (authored course relevance from TRUE gaps). Tokens are counted with the
Mistral tokenizer of the anchor model. H7 rule: see Decisions D-A4-01. Unit of resampling = profile.

Run: ``python -m skillllm_v2.evals.research.compaction [--smoke] [--seed 42] [--n-boot 2000] [--n-profiles 120]``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.evals.research.common import EVALS_ROOT, REPO_ROOT, md_table, record, rel, write_layer

LAYER = "L07_compaction"
GEN_VERSION = "l07-histories-v1"
INPUT_DIR = REPO_ROOT / "inputs" / "evals" / "research" / LAYER
TOKENIZER = REPO_ROOT / "artifacts" / "skillllm_v2" / "mistral_lora" / "model" / "tokenizer.json"
CATALOG = REPO_ROOT / "inputs" / "courses" / "course_catalog_large.jsonl"
UNIVERSE = REPO_ROOT / "inputs" / "evals" / "skill_universe.json"
TIMESTAMP = "2026-01-01T00:00:00+00:00"
BUDGET_FRACTIONS = (4.0, 2.0, 1.0, 0.5, 0.25, 0.1)  # 4.0 and 2.0 are exploratory (added after the smoke run; D-A4-03)
RECENT_FRACTIONS = (0.5, 0.25)
PRIMARY = "compact@0.5"
SHIPPED = "compact@shipped"
MARGIN = -0.02

ROLE_TITLES = {
    "ml_engineer": "Machine Learning Engineer", "data_analyst": "Data Analyst", "data_engineer": "Data Engineer",
    "backend_dev": "Backend Developer", "frontend_dev": "Frontend Developer", "devops_engineer": "DevOps Engineer",
    "cloud_architect": "Cloud Architect", "security_analyst": "Security Analyst", "product_manager": "Product Manager",
    "nlp_scientist": "NLP Scientist", "mobile_dev": "Mobile Developer", "qa_engineer": "QA Engineer",
    "bi_developer": "BI Developer", "project_manager": "Project Manager",
}
SOFT = ["communication", "leadership", "Teamwork", "Problem Solving", "Time Management", "Mentoring", "Negotiation",
        "Stakeholder Management", "Agile"]
EXTRA_HARD = ["Pandas", "NumPy", "TensorFlow", "Scikit-Learn", "Jenkins", "MongoDB", "Redis", "Go", "Rust", "Scrum",
              "GraphQL", "Node.js", "Django", "Flask", "Snowflake", "dbt", "Hadoop", "Ansible", "Prometheus", "Grafana",
              "Helm", "Microsoft Azure", "Google Cloud Platform", "MySQL", "Data Visualization", "Deep Learning",
              "Computer Vision"]
# surface variants used by the simulated writer (alias-resolved by the simulated extractor to the canonical name)
VARIANTS = {"Python": ["python3"], "JavaScript": ["JS"], "PostgreSQL": ["Postgres"], "Kubernetes": ["k8s"],
            "Machine Learning": ["ML"], "AWS": ["Amazon Web Services"], "NLP": ["Natural Language Processing"],
            "Power BI": ["PowerBI"], "Excel": ["Microsoft Excel"], "Statistics": ["statistical analysis"],
            "REST APIs": ["RESTful APIs"], "Test Automation": ["automated testing"]}
# authored synonyms between universe skill names and catalogue skill names (truth side only)
CATALOG_SYNONYMS = {"aws": {"amazon web services"}, "airflow": {"apache airflow"}, "kafka": {"apache kafka"},
                    "nlp": {"natural language processing"}, "networking": {"computer networking"},
                    "scikit learn": {"scikit learn"}}
ORGS = ["Northwind Labs", "Bluefield Analytics", "Harbor Systems", "Cedar Logistics", "Oakridge Health", "Pinecrest Retail",
        "Summit Fintech", "Lakeside Media", "Granite Insurance", "Riverbend Energy"]
CITIES = ["Lisbon", "Leeds", "Porto", "Ghent", "Tampere", "Graz", "Bilbao", "Utrecht"]
TEAMS = ["platform", "analytics", "mobile", "security", "data", "web"]
THINGS = ["dashboard", "data pipeline", "internal tool", "reporting service", "prototype", "test harness", "API"]
HELD_T = [
    "{y}: at {org} I used {s} daily as part of my role.",
    "{y}: built an internal {thing} with {s} at {org}.",
    "{y}: completed a course and applied {s} in a capstone {thing}.",
    "{y}: I am comfortable using {s} and reviewed teammates' work in it.",
    "{y}: maintained a production {thing} at {org} that depended on {s}.",
]
HELD2_T = "{y}: led a small project at {org} that relied on {s} and {s2}."
DISTRACT_T = "{y}: the {team} team at {org} adopted {s}, but I did not work with it."
ASPIRE_T = "{y}: I would like to learn {s} next."
RETRACT_A = "{y}: at {org} I used {s} for reporting tasks."
RETRACT_B = "{y}: I have not used {s} since {y0} and no longer list it as a skill."
PROF_A = "{y}: rated myself an expert in {s} during an annual review."
PROF_B = "{y}: an external assessment placed me at beginner level in {s}."
GOAL_T = "{y}: my career goal is to work as a {title}."
GOAL_OLD_T = "{y}: my career goal was to become a {title}."
FILLER_T = ["{y}: relocated to {city}.", "{y}: took a three-month break to travel.",
            "{y}: attended a regional technology meetup.", "{y}: volunteered at a local community event.",
            "{y}: moved to a hybrid working arrangement."]
LENGTHS = {"short": (3, 6), "medium": (8, 15), "long": (20, 40)}
HELD_N = {"short": (3, 5), "medium": (6, 9), "long": (8, 14)}


def _norm(s: str) -> str:
    from skillllm_v2.schema import normalize_text
    return normalize_text(s)


def load_universe() -> Dict[str, Any]:
    return json.loads(UNIVERSE.read_text(encoding="utf-8"))


def skill_label(name: str) -> str:
    return "SOFT_SKILL" if name in SOFT else "HARD_SKILL"


# ============================================================================================ generator
def generate_history(pid: str, length: str, seed: int, uni: Dict[str, Any], confidence_mode: str = "informative") -> Dict[str, Any]:
    """One synthetic history with authored truth. Deterministic in (pid, length, seed)."""
    rng = random.Random(f"{GEN_VERSION}:{seed}:{pid}")
    roles = sorted(uni["roles"])
    home = rng.choice(roles)
    target = home if rng.random() < 0.25 else rng.choice([r for r in roles if r != home])
    pool = sorted({s for r in uni["roles"].values() for s in r["required"] + r["preferred"]} | set(EXTRA_HARD) | set(SOFT))
    t_spec = uni["roles"][target]
    h_spec = uni["roles"][home]
    lo, hi = HELD_N[length]
    n_held = rng.randint(lo, hi)
    cand = [s for s in h_spec["required"] if rng.random() < 0.85] + [s for s in h_spec["preferred"] if rng.random() < 0.5]
    cand += [s for s in t_spec["required"] + t_spec["preferred"] if rng.random() < 0.3]
    cand += rng.sample(pool, 6)
    cand += rng.sample(SOFT, 2)
    held = list(dict.fromkeys(cand))[:n_held]
    if len(held) < n_held:
        held += [s for s in rng.sample(pool, len(pool)) if s not in held][: n_held - len(held)]
    not_held_pool = [s for s in pool if s not in held]
    gaps_req = [s for s in t_spec["required"] if s not in held]
    gaps_pref = [s for s in t_spec["preferred"] if s not in held]
    n_lo, n_hi = LENGTHS[length]
    n_events = rng.randint(n_lo, n_hi)
    year0 = rng.randint(2008, 2014)

    # contradictions
    n_contra = {"short": 1, "medium": rng.randint(1, 2), "long": rng.randint(2, 3)}[length]
    ctypes = rng.sample(["retraction", "label_conflict", "goal_change", "proficiency"], n_contra)
    events: List[Dict[str, Any]] = []  # each: {"order": float, "tpl_kind", "text_parts", "mentions": [(skill, kind)]}
    contradictions: List[Dict[str, Any]] = []
    org = lambda: rng.choice(ORGS)  # noqa: E731

    def add(order: float, text: str, mentions: List[Tuple[str, str, Optional[str]]], kind: str, **extra) -> Dict[str, Any]:
        ev = {"order": order, "text": text, "mentions": mentions, "kind": kind, **extra}
        events.append(ev)
        return ev

    retracted: List[str] = []
    label_conflict_skill = None
    if "retraction" in ctypes and not_held_pool:
        s = rng.choice([x for x in not_held_pool if x not in SOFT] or not_held_pool)
        not_held_pool.remove(s)
        retracted.append(s)
        a = rng.uniform(0.0, 0.4)
        b = rng.uniform(0.6, 1.0)
        ya = year0 + int(a * 10)
        e1 = add(a, RETRACT_A.format(y=ya, org=org(), s=s), [(s, "retract_claim", None)], "retract_claim")
        e2 = add(b, RETRACT_B.format(y=year0 + int(b * 10) + 1, s=s, y0=ya), [(s, "retract_statement", None)], "retract_statement")
        contradictions.append({"type": "retraction", "key": s, "s1": e1["text"], "s2": e2["text"]})
    if "label_conflict" in ctypes:
        s = next((x for x in held if x in SOFT), None) or rng.choice(held)
        label_conflict_skill = s
        a, b = rng.uniform(0, 1), rng.uniform(0, 1)
        e1 = add(a, HELD_T[0].format(y=year0 + int(a * 10), org=org(), s=s), [(s, "held", "HARD_SKILL")], "held")
        e2 = add(b, HELD_T[3].format(y=year0 + int(b * 10), s=s), [(s, "held", "SOFT_SKILL")], "held")
        contradictions.append({"type": "label_conflict", "key": s, "s1": e1["text"], "s2": e2["text"]})
    if "proficiency" in ctypes:
        s = rng.choice([x for x in held if x != label_conflict_skill] or held)
        a, b = rng.uniform(0, 0.5), rng.uniform(0.5, 1)
        e1 = add(a, PROF_A.format(y=year0 + int(a * 10), s=s), [(s, "held", None)], "held")
        e2 = add(b, PROF_B.format(y=year0 + int(b * 10) + 1, s=s), [(s, "held", None)], "held")
        contradictions.append({"type": "proficiency", "key": s, "s1": e1["text"], "s2": e2["text"]})
    goal_title = ROLE_TITLES[target]
    goal_history = [goal_title]
    if "goal_change" in ctypes:
        old = rng.choice([r for r in roles if r != target])
        a = rng.uniform(0.0, 0.5)
        e1 = add(a, GOAL_OLD_T.format(y=year0 + int(a * 10), title=ROLE_TITLES[old]), [], "goal_old", goal=ROLE_TITLES[old])
        goal_history = [ROLE_TITLES[old], goal_title]
    b = rng.uniform(0.85, 1.0)
    eg = add(b, GOAL_T.format(y=year0 + int(b * 10) + 1, title=goal_title), [], "goal", goal=goal_title)
    if "goal_change" in ctypes:
        contradictions.append({"type": "goal_change", "key": goal_title, "s1": e1["text"], "s2": eg["text"]})

    # every held skill gets >= 1 evidence sentence; extra mentions for 'important' skills (home-role required)
    for s in held:
        if s == label_conflict_skill:
            continue
        a = rng.uniform(0, 1)
        add(a, rng.choice(HELD_T).format(y=year0 + int(a * 10), org=org(), s=s, thing=rng.choice(THINGS)), [(s, "held", None)], "held")
    # noise: distractors / aspirations (aspirations draw from true gaps when possible)
    n_dis = {"short": rng.randint(0, 1), "medium": rng.randint(1, 3), "long": rng.randint(2, 5)}[length]
    n_asp = {"short": rng.randint(0, 1), "medium": rng.randint(0, 2), "long": rng.randint(1, 3)}[length]
    for _ in range(n_dis):
        cands = [x for x in not_held_pool if x not in retracted]
        if not cands:
            break
        s = rng.choice(cands)
        a = rng.uniform(0, 1)
        add(a, DISTRACT_T.format(y=year0 + int(a * 10), team=rng.choice(TEAMS), org=org(), s=s), [(s, "distractor", None)], "distractor")
    asp_pool = [x for x in gaps_req + gaps_pref if x not in retracted] or [x for x in not_held_pool if x not in retracted]
    for s in rng.sample(asp_pool, min(n_asp, len(asp_pool))):
        a = rng.uniform(0.5, 1)
        add(a, ASPIRE_T.format(y=year0 + int(a * 10) + 1, s=s), [(s, "aspiration", None)], "aspiration")
    # fill to n_events with extra held evidence (pairs) and fillers
    while len(events) < n_events:
        a = rng.uniform(0, 1)
        r = rng.random()
        if r < 0.55 and len(held) >= 2:
            s, s2 = rng.sample(held, 2)
            add(a, HELD2_T.format(y=year0 + int(a * 10), org=org(), s=s, s2=s2), [(s, "held", None), (s2, "held", None)], "held")
        elif r < 0.8:
            s = rng.choice(held)
            add(a, rng.choice(HELD_T).format(y=year0 + int(a * 10), org=org(), s=s, thing=rng.choice(THINGS)), [(s, "held", None)], "held")
        else:
            add(a, rng.choice(FILLER_T).format(y=year0 + int(a * 10), city=rng.choice(CITIES)), [], "filler")
    events.sort(key=lambda e: (e["order"], e["text"]))

    # surface variants + simulated extraction
    lines, extractions, evidence = [], [], defaultdict(list)
    offset = 0
    for ev in events:
        text = ev["text"]
        for s, kind, forced_label in ev["mentions"]:
            if s in VARIANTS and rng.random() < 0.35 and text.count(s) == 1:
                text = text.replace(s, rng.choice(VARIANTS[s]), 1)
        ev["final_text"] = text
        for s, kind, forced_label in ev["mentions"]:
            surf = next((v for v in [s] + VARIANTS.get(s, []) if v in text), s)
            i = text.find(surf)
            conf = _confidence(rng, kind, confidence_mode)
            extractions.append({
                "surface_form": surf, "label": forced_label or skill_label(s), "start": offset + i, "end": offset + i + len(surf),
                "confidence": round(conf, 4), "source": "sim_extractor", "evidence": text,
                "canonical_skill_id": _norm(s), "canonical_skill_name": s, "_kind": kind})
            if kind == "held":
                evidence[s].append(text)
        lines.append(text)
        offset += len(text) + 1
    for c in contradictions:  # contradiction sentences after variant substitution
        for k in ("s1", "s2"):
            for ev in events:
                if ev["text"] == c[k]:
                    c[k] = ev["final_text"]
                    break
    truth = {"held": sorted(held), "evidence": {k: sorted(set(v)) for k, v in evidence.items()},
             "retracted": retracted, "goal": goal_title, "goal_history": goal_history, "target_role": target,
             "home_role": home, "gaps_required": gaps_req, "gaps_preferred": gaps_pref, "contradictions": contradictions,
             "non_held_mentions": sorted({m[0] for ev in events for m in ev["mentions"] if m[1] in ("distractor", "aspiration", "retract_claim", "retract_statement")})}
    return {"profile_id": pid, "length_class": length, "confidence_mode": confidence_mode, "text": "\n".join(lines),
            "lines": lines, "line_kinds": [ev["kind"] for ev in events], "extractions": extractions, "truth": truth}


def _confidence(rng: random.Random, kind: str, mode: str) -> float:
    if mode == "uninformative":
        return rng.uniform(0.5, 0.99)
    lo, hi = {"held": (0.75, 0.99), "retract_claim": (0.75, 0.99), "retract_statement": (0.5, 0.8),
              "distractor": (0.3, 0.6), "aspiration": (0.35, 0.65)}[kind]
    return rng.uniform(lo, hi)


def generate_all(n: int, seed: int, confidence_mode: str = "informative") -> List[Dict[str, Any]]:
    uni = load_universe()
    classes = list(LENGTHS)
    return [generate_history(f"p{i:03d}", classes[i % 3], seed, uni, confidence_mode) for i in range(n)]


# ============================================================================================ tokens / representations
class MistralTokens:
    def __init__(self, path: Path = TOKENIZER) -> None:
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(path))
        # the stored tokenizer.json carries truncation max_length=512 (training setting); counting must not truncate
        self.tok.no_truncation()
        self.tok.no_padding()

    def __call__(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)


def proxy_tokens(text: str) -> int:
    from skillllm_v2.context.budget import count_tokens
    return count_tokens(text)


def _public_extractions(h: Dict[str, Any], lines_keep: Optional[set] = None) -> List[Dict[str, Any]]:
    out = []
    if lines_keep is None:
        return [{k: v for k, v in e.items() if not k.startswith("_")} for e in h["extractions"]]
    starts, off = [], 0
    for ln in h["lines"]:
        starts.append((off, off + len(ln)))
        off += len(ln) + 1
    for e in h["extractions"]:
        li = next(i for i, (a, b) in enumerate(starts) if a <= e["start"] < b)
        if li in lines_keep:
            out.append({k: v for k, v in e.items() if not k.startswith("_")})
    return out


def facts_from_summary(skills: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"name": s.get("canonical_skill_name") or s.get("surface_form"), "id": s.get("canonical_skill_id"),
             "confidence": float(s.get("confidence", 1.0)), "evidence": [e.get("text", "") for e in s.get("evidence", [])],
             "label": s.get("label", "HARD_SKILL")} for s in skills]


def build_representations(h: Dict[str, Any], ntok: Callable[[str], int]) -> Dict[str, Dict[str, Any]]:
    """name -> {serial, tokens, facts, goals, conflict_notes, evidence_blob}."""
    from skillllm_v2.context.cached_context_builder import build_cached_context_packet
    from skillllm_v2.context.profile_summarizer import summarize_profile
    reps: Dict[str, Dict[str, Any]] = {}
    ex = _public_extractions(h)
    goal = h["truth"]["goal"]
    full_summary = summarize_profile(h["text"], extracted_skills=ex, user_id=h["profile_id"], timestamp=TIMESTAMP)
    full_facts = facts_from_summary(full_summary["skills"])
    reps["full"] = {"serial": h["text"], "facts": full_facts, "goal_texts": [h["text"]], "notes": full_summary["conflict_notes"],
                    "evidence": [h["text"]], "budget": None}
    ptok = proxy_tokens(h["text"])
    for frac in (None,) + BUDGET_FRACTIONS:
        budget = None if frac is None else max(1, int(math.floor(frac * ptok)))
        pkt = build_cached_context_packet(h["text"], ex, job_target=goal, user_id=h["profile_id"], budget_tokens=budget,
                                          timestamp=TIMESTAMP)
        stable = pkt["payload"]["stable_cag_context"]
        prof = stable["profile_summary"]
        facts = facts_from_summary(prof.get("skills", []))
        name = SHIPPED if frac is None else f"compact@{frac:g}"
        reps[name] = {"serial": json.dumps(pkt, sort_keys=True, ensure_ascii=False), "facts": facts,
                      "goal_texts": list(prof.get("target_roles", [])), "notes": list(pkt["payload"].get("conflict_notes", [])),
                      "evidence": [t for f in facts for t in f["evidence"]], "budget": budget,
                      "budget_met": pkt["payload"].get("budget", {}).get("met", True),
                      "kept": pkt["payload"].get("budget", {}).get("kept"), "total": pkt["payload"].get("budget", {}).get("total")}
    names = sorted({f["name"] for f in full_facts})
    reps["minimal"] = {"serial": json.dumps(names, ensure_ascii=False), "facts": [{"name": n, "id": _norm(n), "confidence": 1.0, "evidence": [], "label": None} for n in names],
                       "goal_texts": [], "notes": [], "evidence": [], "budget": None}
    for frac in RECENT_FRACTIONS:
        budget = max(1, int(math.floor(frac * ptok)))
        keep, used = set(), 0
        for i in range(len(h["lines"]) - 1, -1, -1):
            c = proxy_tokens(h["lines"][i]) + 1
            if used + c > budget:
                break
            keep.add(i)
            used += c
        text = "\n".join(h["lines"][i] for i in sorted(keep))
        exk = _public_extractions(h, keep)
        summ = summarize_profile(text, extracted_skills=exk, user_id=h["profile_id"], timestamp=TIMESTAMP)
        reps[f"recent@{frac:g}"] = {"serial": text, "facts": facts_from_summary(summ["skills"]), "goal_texts": [text],
                                   "notes": summ["conflict_notes"], "evidence": [text], "budget": budget}
    reps["truth"] = {"serial": json.dumps(h["truth"]["held"]), "facts": [{"name": n, "id": _norm(n), "confidence": 1.0, "evidence": [], "label": None} for n in h["truth"]["held"]],
                     "goal_texts": [goal], "notes": [], "evidence": [], "budget": None, "reference_only": True}
    for r in reps.values():
        r["tokens"] = ntok(r["serial"])
        r["proxy_tokens"] = proxy_tokens(r["serial"])
    return reps


# ============================================================================================ downstream
class Downstream:
    def __init__(self, seed: int) -> None:
        from skillllm_v2.evals.match_evals import Universe, make_jobs
        from skillllm_v2.match.course_recommender import CourseRecommender
        from skillllm_v2.match.gap_detector import GapDetector
        from skillllm_v2.match.job_matcher import JobMatcher
        u = load_universe()
        adj: Dict[str, set] = {}
        for group in u["adjacency"]:
            for s in group:
                adj.setdefault(s, set()).update(x for x in group if x != s)
        self.uni = Universe(u["roles"], adj, u["aliases"], list(u["distractor_skills"]))
        self.jobs = make_jobs(self.uni, random.Random(f"{seed}:l07-jobs"), 3)
        self.matcher = JobMatcher()
        self.gaps = GapDetector()
        self.rec = CourseRecommender()
        self.catalog = [json.loads(l) for l in CATALOG.read_text(encoding="utf-8").splitlines() if l.strip()]

    @staticmethod
    def profile(h: Dict[str, Any], facts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {"user_id": h["profile_id"], "target_roles": [h["truth"]["goal"]],
                "skills": [{"surface_form": f["name"], "canonical_skill_name": f["name"], "canonical_skill_id": f["id"] or _norm(f["name"]),
                            "confidence": f["confidence"]} for f in facts]}

    def target_job(self, h: Dict[str, Any]) -> Dict[str, Any]:
        spec = self.uni.roles[h["truth"]["target_role"]]
        return {"job_id": f"target-{h['truth']['target_role']}", "title": h["truth"]["goal"],
                "required_skills": list(spec["required"]), "preferred_skills": list(spec["preferred"])}

    def job_scores(self, prof: Dict[str, Any]) -> Dict[str, float]:
        from skillllm_v2.evals.match_evals import clean_job
        return {j["job_id"]: float(self.matcher.score(prof, clean_job(j))["match_score"]) for j in self.jobs}

    def job_grades(self, h: Dict[str, Any]) -> Dict[str, float]:
        from skillllm_v2.evals.match_evals import grade_pair
        return {j["job_id"]: float(grade_pair(self.uni, h["truth"]["held"], j)) for j in self.jobs}

    def gap_set(self, prof: Dict[str, Any], h: Dict[str, Any]) -> Tuple[set, List[Dict[str, Any]]]:
        gaps = self.gaps.detect(prof, self.target_job(h))
        return {_norm(g["skill"]) for g in gaps}, gaps

    def course_scores(self, prof: Dict[str, Any], h: Dict[str, Any], gaps: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[str]]:
        recs = self.rec.recommend(gaps, self.catalog, user_profile=prof, target_job=self.target_job(h))
        ranked = [r["course_id"] for r in recs]
        scores = {r["course_id"]: float(r["score"]) for r in recs}
        for c in self.catalog:
            scores.setdefault(c["course_id"], -1.0)
        return scores, ranked

    def course_grades(self, h: Dict[str, Any]) -> Dict[str, float]:
        held = {_norm(s) for s in h["truth"]["held"]}
        held |= {x for s in list(held) for x in CATALOG_SYNONYMS.get(s, set())}
        req = {_norm(s) for s in h["truth"]["gaps_required"]}
        pref = {_norm(s) for s in h["truth"]["gaps_preferred"]}

        def names(s: str) -> set:
            return {s} | CATALOG_SYNONYMS.get(s, set())

        req_n = set().union(*[names(s) for s in req]) if req else set()
        pref_n = set().union(*[names(s) for s in pref]) if pref else set()
        out = {}
        for c in self.catalog:
            taught = {_norm(s) for s in c.get("teaches_skills", [])}
            g = 2.0 if taught & req_n else 1.0 if taught & pref_n else 0.0
            if g > 0 and all(_norm(p) in held for p in c.get("prerequisites", [])):
                g += 1.0
            out[c["course_id"]] = g
        return out


def kendall(a: Dict[str, float], b: Dict[str, float]) -> float:
    from scipy.stats import kendalltau
    keys = sorted(a)
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    if np.allclose(x, y):
        return 1.0
    t = kendalltau(x, y).statistic
    return 0.0 if t is None or not np.isfinite(t) else float(t)


def rank_ids(scores: Dict[str, float]) -> List[str]:
    return sorted(scores, key=lambda k: (-scores[k], k))


def top_overlap(a: Sequence[str], b: Sequence[str], k: int = 5) -> float:
    """|top-k(a) & top-k(b)| / max(|top-k(a)|, |top-k(b)|); both lists empty -> 1.0 (the recommender may return < k)."""
    den = max(len(a[:k]), len(b[:k]))
    return 1.0 if den == 0 else len(set(a[:k]) & set(b[:k])) / den


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


# ============================================================================================ per-profile scoring
def _present(sentence: str, rep: Dict[str, Any]) -> bool:
    """A sentence is retained if it occurs verbatim in a raw-text representation, or an evidence snippet of the
    representation is a substring of it with length >= min(30, len(sentence))."""
    if any(sentence in t for t in rep["evidence"] if len(t) >= len(sentence)):
        return True
    return any(e and e in sentence and len(e) >= min(30, len(sentence)) for e in rep["evidence"])


def score_profile(h: Dict[str, Any], reps: Dict[str, Dict[str, Any]], ds: Downstream) -> Dict[str, Dict[str, float]]:
    from skillllm_v2.eval.ranking_metrics import ndcg_at_k
    truth = h["truth"]
    t_held = {_norm(s) for s in truth["held"]}
    jg, cg = ds.job_grades(h), ds.course_grades(h)
    t_gap = {_norm(s) for s in truth["gaps_required"] + truth["gaps_preferred"]}
    out: Dict[str, Dict[str, float]] = {}
    base: Dict[str, Any] = {}
    for name in ["full"] + [n for n in reps if n != "full"]:
        r = reps[name]
        prof = ds.profile(h, r["facts"])
        held = {_norm(f["name"]) for f in r["facts"]}
        js = ds.job_scores(prof)
        gset, gaps = ds.gap_set(prof, h)
        cs, cranked = ds.course_scores(prof, h, gaps)
        jr = rank_ids(js)
        m: Dict[str, float] = {
            "tokens": r["tokens"], "proxy_tokens": r["proxy_tokens"],
            "skill_precision": len(held & t_held) / len(held) if held else float("nan"),  # undefined when nothing kept
            "skill_recall": len(held & t_held) / len(t_held) if t_held else 1.0,
            "job_ndcg10": ndcg_at_k(jr, jg, 10) if ndcg_at_k(jr, jg, 10) is not None else float("nan"),
            "course_ndcg10": (lambda v: float("nan") if v is None else v)(ndcg_at_k(cranked, cg, 10)),
            "gap_precision_truth": len(gset & t_gap) / len(gset) if gset else float("nan"),
            "gap_recall_truth": len(gset & t_gap) / len(t_gap) if t_gap else 1.0,
            "goal_retained": float(any(truth["goal"].lower() in g.lower() for g in r["goal_texts"])),
            "evidence_retention": (sum(1 for s in truth["held"] if any(_present(x, r) for x in truth["evidence"].get(s, [])))
                                   / max(1, len(truth["held"]))),
        }
        # contradiction preservation
        pres = []
        for c in truth["contradictions"]:
            key = _norm(c["key"])
            if c["type"] == "label_conflict":
                ok = any(key in _norm(n) for n in r["notes"]) or (_present(c["s1"], r) and _present(c["s2"], r))
            elif c["type"] == "goal_change":
                ok = all(any(t.lower() in g.lower() for g in r["goal_texts"]) for t in truth["goal_history"])
            else:
                ok = _present(c["s1"], r) and _present(c["s2"], r)
            pres.append(float(ok))
            m[f"contradiction_{c['type']}"] = float(ok)
        m["contradiction_preservation"] = float(np.mean(pres)) if pres else float("nan")
        if name == "full":
            base = {"held": held, "js": js, "jr": jr, "gset": gset, "cs": cs, "cr": cranked, "tokens": r["tokens"]}
        m["skill_recall_vs_full"] = len(held & base["held"]) / len(base["held"]) if base["held"] else 1.0
        m["job_kendall_vs_full"] = kendall(js, base["js"])
        m["job_top5_overlap_vs_full"] = top_overlap(jr, base["jr"])
        m["course_kendall_vs_full"] = kendall(cs, base["cs"])
        m["course_top5_overlap_vs_full"] = top_overlap(cranked, base["cr"])
        m["gap_jaccard_vs_full"] = jaccard(gset, base["gset"])
        m["token_ratio_vs_full"] = r["tokens"] / base["tokens"] if base["tokens"] else float("nan")
        m["token_reduction_vs_full"] = 1.0 - m["token_ratio_vs_full"]
        if r.get("budget") is not None and name.startswith("compact@"):
            m["budget_met"] = float(bool(r.get("budget_met", True)))
            m["skills_kept_share"] = (r.get("kept") or 0) / r["total"] if r.get("total") else float("nan")
        out[name] = m
    return out


# ============================================================================================ main
def run(args: argparse.Namespace) -> Path:
    from skillllm_v2.evals.research import stats as S
    seed, n_boot = args.seed, args.n_boot
    n = 9 if args.smoke else args.n_profiles
    ntok = MistralTokens()
    ds = Downstream(seed)
    arms: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    hist_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ("informative", "uninformative"):
        hs = generate_all(n, seed, arm)
        hist_by_arm[arm] = hs
        arms[arm] = {h["profile_id"]: score_profile(h, build_representations(h, ntok), ds) for h in hs}
    if not args.smoke:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        for arm, hs in hist_by_arm.items():
            with open(INPUT_DIR / f"histories_{arm}.jsonl", "w", encoding="utf-8") as fh:
                for h in hs:
                    fh.write(json.dumps(h, ensure_ascii=False, sort_keys=True) + "\n")
    hist_files = [INPUT_DIR / f"histories_{a}.jsonl" for a in hist_by_arm] if not args.smoke else []
    prov = [TOKENIZER, CATALOG, UNIVERSE, REPO_ROOT / "src/skillllm_v2/context/context_compactor.py",
            REPO_ROOT / "src/skillllm_v2/context/cached_context_builder.py"] + hist_files
    dataset = f"{GEN_VERSION} (seed={seed}, n={n}; synthetic, generator-authored truth)"
    records: List[Dict[str, Any]] = []
    per_unit: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    systems = list(next(iter(arms["informative"].values())).keys())
    metrics = ["tokens", "token_reduction_vs_full", "skill_precision", "skill_recall", "skill_recall_vs_full",
               "goal_retained", "gap_precision_truth", "gap_recall_truth", "evidence_retention", "contradiction_preservation",
               "job_ndcg10", "course_ndcg10", "job_kendall_vs_full", "job_top5_overlap_vs_full", "course_kendall_vs_full",
               "course_top5_overlap_vs_full", "gap_jaccard_vs_full"]
    for arm, res in arms.items():
        pids = sorted(res)
        summary[arm] = {}
        for sysname in systems:
            summary[arm][sysname] = {}
            for met in metrics + ["contradiction_retraction", "contradiction_label_conflict", "contradiction_goal_change",
                                  "contradiction_proficiency", "budget_met"]:
                vals = [(p, res[p][sysname].get(met)) for p in pids]
                vals = [(p, v) for p, v in vals if v is not None and np.isfinite(v)]
                if not vals:
                    continue
                ci = S.bootstrap_mean_ci([v for _, v in vals], n_boot=n_boot, seed=seed)
                summary[arm][sysname][met] = ci["value"]
                summary[arm][sysname][met + "_ci"] = (ci["ci_low"], ci["ci_high"])
                if met in metrics or met.startswith("contradiction_") or met == "budget_met":
                    head = arm == "informative" and sysname == PRIMARY and met in ("token_reduction_vs_full", "course_ndcg10", "job_ndcg10", "gap_jaccard_vs_full")
                    records.append(record(LAYER, f"{sysname}[{arm}]", met, ci["value"], n_units=len(vals), unit="profile",
                                          split="synthetic", dataset=dataset, direction=_direction(met),
                                          ci=(ci["ci_low"], ci["ci_high"]), ci_method=f"profile percentile bootstrap, n_boot={n_boot}, seed={seed}",
                                          provenance=prov, caveats=["Synthetic histories; truth authored by the generator.",
                                                                    "Simulated oracle-span extractor; confidence rule = arm."],
                                          details={"arm": arm, **({"headline": True} if head else {})}))
                if arm == "informative":
                    for p, v in vals:
                        per_unit.append({"layer": LAYER, "family": "H7", "system": sysname, "metric": met, "unit_id": p, "value": v})
    # ---- H7: paired non-inferiority vs full
    h7: Dict[str, Dict[str, Any]] = {}
    comps = []
    for arm, res in arms.items():
        pids = sorted(res)
        for sysname in [s for s in systems if s not in ("full",)]:
            row: Dict[str, Any] = {"arm": arm, "system": sysname}
            for met in ("job_ndcg10", "course_ndcg10"):
                pairs = [(res[p][sysname][met], res[p]["full"][met]) for p in pids
                         if np.isfinite(res[p][sysname][met]) and np.isfinite(res[p]["full"][met])]
                x = np.array([a for a, _ in pairs])[:, None]
                y = np.array([b for _, b in pairs])[:, None]
                pb = S.paired_bootstrap(x, y, S.mean_stat(len(pairs)), n_boot=n_boot, seed=seed, keep_samples=True)
                ni = S.non_inferiority(pb["ci_low"], MARGIN, pb["delta"], pb["samples"])
                row[met] = {"delta": pb["delta"], "ci": (pb["ci_low"], pb["ci_high"]), "p_two_sided": pb["p"],
                            "p_noninf": ni.get("p_one_sided"), "non_inferior": ni["non_inferior"], "n": len(pairs),
                            "n_units_differ": pb["n_units_differ"]}
                comps.append({"arm": arm, "system": sysname, "metric": met, "p": ni.get("p_one_sided"),
                              "family": f"H7[{arm}][{sysname}]"})
            tr = [res[p][sysname]["token_reduction_vs_full"] for p in pids]
            gj = [res[p][sysname]["gap_jaccard_vs_full"] for p in pids]
            row["token_reduction"] = float(np.mean(tr))
            row["token_reduction_ci"] = (lambda c: (c["ci_low"], c["ci_high"]))(S.bootstrap_mean_ci(tr, n_boot=n_boot, seed=seed))
            row["gap_jaccard"] = float(np.mean(gj))
            row["gap_jaccard_ci"] = (lambda c: (c["ci_low"], c["ci_high"]))(S.bootstrap_mean_ci(gj, n_boot=n_boot, seed=seed))
            h7[f"{arm}|{sysname}"] = row
    S.adjust_families(comps, p_key="p", family_key="family")
    for c in comps:
        h7[f"{c['arm']}|{c['system']}"][c["metric"]]["p_holm_noninf"] = c.get("p_holm")
    rule = ("token reduction (mean, Mistral tokens) >= 0.50 AND job nDCG@10 and course nDCG@10 non-inferior to full "
            "(lower 95% CI of delta > -0.02, Holm over the two) AND mean gap Jaccard vs full >= 0.9")
    for arm in arms:
        for sysname in (PRIMARY, SHIPPED):
            row = h7[f"{arm}|{sysname}"]
            crit = {"token": row["token_reduction"] >= 0.5, "job": row["job_ndcg10"]["non_inferior"],
                    "course": row["course_ndcg10"]["non_inferior"], "gap": row["gap_jaccard"] >= 0.9}
            verdict = "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED"
            row["criteria"], row["verdict"] = crit, verdict
            cd = row["course_ndcg10"]
            records.append(record(LAYER, f"{sysname}[{arm}]", "H7_verdict", cd["delta"], n_units=cd["n"], unit="profile",
                                  split="synthetic", dataset=dataset, direction="higher", ci=cd["ci"],
                                  ci_method="paired profile bootstrap", provenance=prov,
                                  details={"verdict": verdict, "rule": rule, "p_raw": cd["p_noninf"], "p_holm": cd.get("p_holm_noninf"),
                                           "family": f"H7[{arm}][{sysname}]", "primary_effect": "course nDCG@10 delta (rep - full)",
                                           "criteria": crit, "token_reduction": row["token_reduction"],
                                           "token_reduction_ci": row["token_reduction_ci"], "job_ndcg10": row["job_ndcg10"],
                                           "course_ndcg10": row["course_ndcg10"], "gap_jaccard": row["gap_jaccard"],
                                           "gap_jaccard_ci": row["gap_jaccard_ci"],
                                           "headline_verdict": arm == "informative" and sysname == PRIMARY}))
    for key, row in h7.items():
        arm, sysname = key.split("|")
        for met in ("job_ndcg10", "course_ndcg10"):
            d = row[met]
            records.append(record(LAYER, f"{sysname} - full[{arm}]", f"delta_{met}", d["delta"], n_units=d["n"], unit="profile",
                                  split="synthetic", dataset=dataset, direction="higher", ci=d["ci"],
                                  ci_method="paired profile bootstrap", provenance=prov,
                                  details={"p_two_sided": d["p_two_sided"], "p_noninf_one_sided": d["p_noninf"],
                                           "p_holm_noninf": d.get("p_holm_noninf"), "non_inferior_margin": MARGIN,
                                           "non_inferior": d["non_inferior"], "n_units_differ": d["n_units_differ"]}))
    out_dir = EVALS_ROOT / LAYER / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        with open(out_dir / "per_unit_scores.jsonl", "w", encoding="utf-8") as fh:
            for r in per_unit:
                fh.write(json.dumps(r) + "\n")
    curve = _figures(summary, out_dir)
    lengths = {h["profile_id"]: h["length_class"] for h in hist_by_arm["informative"]}
    by_len = []
    for sysname in [SHIPPED, "compact@4", "compact@2", "compact@1", PRIMARY, "recent@0.5", "minimal"]:
        for lc in LENGTHS:
            pids = [p for p in sorted(arms["informative"]) if lengths[p] == lc]
            vals = lambda met: float(np.nanmean([arms["informative"][p][sysname].get(met, float("nan")) for p in pids]))  # noqa: E731
            by_len.append({"representation": sysname, "length": lc, "n": len(pids), "full tokens": float(np.mean([arms["informative"][p]["full"]["tokens"] for p in pids])),
                           "token red.": vals("token_reduction_vs_full"), "skill R vs full": vals("skill_recall_vs_full"),
                           "job tau": vals("job_kendall_vs_full"), "gap J": vals("gap_jaccard_vs_full"),
                           "budget met": vals("budget_met") if sysname.startswith("compact@") and sysname != SHIPPED else None})
    with open(out_dir / "by_length_class.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(by_len[0].keys()))
        w.writeheader()
        w.writerows(by_len)
    report = _report(summary, h7, n, rule, curve, arms, by_len)
    return write_layer(LAYER, records, report, seed=seed, smoke=args.smoke,
                       extra_meta={"generator": GEN_VERSION, "n_profiles": n, "n_boot": n_boot})


def _direction(met: str) -> str:
    if met in ("tokens",):
        return "lower"
    return "higher"


def _figures(summary: Dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for arm, sysd in summary.items():
        for sysname, m in sysd.items():
            if sysname == "truth":
                continue
            rows.append({"arm": arm, "system": sysname, "token_ratio": 1 - m.get("token_reduction_vs_full", 0.0),
                         "job_kendall": m.get("job_kendall_vs_full"), "course_kendall": m.get("course_kendall_vs_full"),
                         "job_top5": m.get("job_top5_overlap_vs_full"), "course_top5": m.get("course_top5_overlap_vs_full"),
                         "gap_jaccard": m.get("gap_jaccard_vs_full"), "job_ndcg10": m.get("job_ndcg10"),
                         "course_ndcg10": m.get("course_ndcg10"), "skill_recall": m.get("skill_recall"),
                         "skill_precision": m.get("skill_precision")})
    with open(out_dir / "compression_vs_agreement.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, arm in zip(axes, ("informative", "uninformative")):
        pts = [r for r in rows if r["arm"] == arm]
        for fam, marker in (("compact@", "o"), ("recent@", "s")):
            sel = sorted([r for r in pts if r["system"].startswith(fam) and r["system"] != SHIPPED], key=lambda r: r["token_ratio"])
            for key, ls in (("course_kendall", "-"), ("job_kendall", "--"), ("gap_jaccard", ":")):
                ax.plot([r["token_ratio"] for r in sel], [r[key] for r in sel], ls, marker=marker, ms=4,
                        label=f"{fam.rstrip('@')} {key}")
        for r in pts:
            if r["system"] in (SHIPPED, "minimal"):
                ax.scatter([r["token_ratio"]], [r["course_kendall"]], marker="x", color="k")
                ax.annotate(r["system"], (r["token_ratio"], r["course_kendall"]), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("tokens(rep) / tokens(full history)  [Mistral tokenizer, log]")
        ax.set_title(f"confidence arm: {arm}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("agreement with full-history decision")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "compression_vs_agreement.png", dpi=150)
    plt.close(fig)
    return rows


def _fmt_ci(v: Any) -> str:
    if not v or v[0] is None:
        return "-"
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


def _report(summary, h7, n, rule, curve, arms, by_len=()) -> str:
    inf = summary["informative"]
    order = ["full", SHIPPED, "compact@4", "compact@2", "compact@1", "compact@0.5", "compact@0.25", "compact@0.1", "recent@0.5", "recent@0.25", "minimal", "truth"]
    t1 = []
    for s in order:
        if s not in inf:
            continue
        m = inf[s]
        t1.append({"representation": s, "tokens": m.get("tokens"), "reduction": m.get("token_reduction_vs_full"),
                   "skill P": m.get("skill_precision"), "skill R": m.get("skill_recall"), "goal": m.get("goal_retained"),
                   "gap R (truth)": m.get("gap_recall_truth"), "evidence": m.get("evidence_retention"),
                   "contradictions": m.get("contradiction_preservation")})
    t2 = []
    for s in order:
        if s not in inf:
            continue
        m = inf[s]
        t2.append({"representation": s, "job nDCG@10": m.get("job_ndcg10"), "course nDCG@10": m.get("course_ndcg10"),
                   "job tau": m.get("job_kendall_vs_full"), "job top5": m.get("job_top5_overlap_vs_full"),
                   "course tau": m.get("course_kendall_vs_full"), "course top5": m.get("course_top5_overlap_vs_full"),
                   "gap Jaccard": m.get("gap_jaccard_vs_full")})
    t3 = []
    for key, row in h7.items():
        arm, s = key.split("|")
        if s not in (PRIMARY, SHIPPED, "compact@4", "compact@2", "compact@1", "recent@0.5", "minimal"):
            continue
        t3.append({"arm": arm, "representation": s, "token red.": row["token_reduction"],
                   "Δ job nDCG": row["job_ndcg10"]["delta"], "job CI": _fmt_ci(row["job_ndcg10"]["ci"]),
                   "Δ course nDCG": row["course_ndcg10"]["delta"], "course CI": _fmt_ci(row["course_ndcg10"]["ci"]),
                   "gap J": row["gap_jaccard"], "verdict": row.get("verdict", "(not an H7 config)")})
    primary = h7[f"informative|{PRIMARY}"]
    lines = [
        "# L07 — Profile compaction (H7)",
        "",
        f"**H7 (primary, `{PRIMARY}`, informative-confidence arm): {primary['verdict']}** — criteria {primary['criteria']}.",
        f"Shipped default (`{SHIPPED}`, budget None): {h7[f'informative|{SHIPPED}']['verdict']} — criteria {h7[f'informative|{SHIPPED}']['criteria']}.",
        f"Sensitivity arm (uninformative confidence): primary {h7[f'uninformative|{PRIMARY}']['verdict']} {h7[f'uninformative|{PRIMARY}']['criteria']}.",
        "",
        f"Rule (pre-registered, Decisions D-A4-01): {rule}.",
        "",
        f"Data: {n} seeded synthetic histories (short/medium/long thirds), truth authored by the generator "
        f"(`research/compaction.py`, `{GEN_VERSION}`), inputs in `inputs/evals/research/L07_compaction/`. Tokens = Mistral "
        "tokenizer (`artifacts/skillllm_v2/mistral_lora/model/tokenizer.json`) on the exact serialisation.",
        "",
        "## Retention vs generator truth (informative arm; means over profiles)",
        "",
        md_table(t1, list(t1[0].keys()), fmt="{:.3f}"),
        "",
        "## Downstream decisions (informative arm)",
        "",
        md_table(t2, list(t2[0].keys()), fmt="{:.3f}"),
        "",
        "## Non-inferiority vs full history (paired profile bootstrap)",
        "",
        md_table(t3, list(t3[0].keys()), fmt="{:.3f}"),
        "",
        "Figure: `compression_vs_agreement.png` (+ `.csv`): agreement with the full-history decision vs token ratio.",
        "",
        "## By history length (informative arm)",
        "",
        md_table(list(by_len), list(by_len[0].keys()) if by_len else [], fmt="{:.3f}"),
        "",
        "## Contradiction preservation by type (informative arm)",
        "",
        md_table([{"representation": s, **{t: inf[s].get(f"contradiction_{t}") for t in ("retraction", "label_conflict", "goal_change", "proficiency")}}
                  for s in order if s in inf], ["representation", "retraction", "label_conflict", "goal_change", "proficiency"], fmt="{:.3f}"),
        "",
        "## Reading the result",
        "",
        "- `full` keeps every extracted mention, including distractors/aspirations/retracted skills, so its skill precision "
        "vs truth is < 1; a compactor that drops low-confidence mentions can be *closer to truth* while *disagreeing* with "
        "full. Agreement-with-full and accuracy-vs-truth are reported separately for that reason.",
        "- The shipped packet (`budget_tokens=None`) is lossless JSON and is LARGER than the raw history; it can only fail "
        "the token criterion.",
        "- `goal_retained` for compact is a pass-through of the caller's `job_target` (the system does not extract goals from "
        "history text); goal *history* (goal-change contradictions) is not kept.",
        "- `minimal` exposes the same skill set as `full`, so its downstream decisions equal full by construction; it loses "
        "goals, evidence and contradictions.",
        "- Course nDCG@10 RISES when skills are dropped: with fewer held skills every target skill becomes a gap, so courses "
        "for the true gaps that `full` hides (aspiration mentions read as held skills) get recommended again. Course-nDCG "
        "non-inferiority is therefore satisfied trivially by an empty profile and is not sufficient on its own; the job "
        "criterion and gap agreement are what expose the information loss.",
        "- The JSON packet costs ~35-45 proxy tokens per skill fact versus ~12-15 for the raw sentence, plus a fixed skeleton "
        "(~200 Mistral tokens incl. a 64-hex content hash); at budgets below ~4x the raw history it keeps few or no skills, "
        "and for short histories the 0.5x budget is below the skeleton itself (`budget met` < 1).",
        "",
        "## Caveats",
        "",
        "- Synthetic histories and authored relevance: absolute nDCG is optimistic (ranker and grades both reason about skill "
        "coverage); paired deltas and agreement are the informative quantities.",
        "- Extraction is simulated (oracle spans, authored confidence). The informative-confidence arm favours "
        "confidence-ranked compaction; the uninformative arm is the stress test.",
        "- Proxy-token budgets (the system's unit) differ from Mistral tokens; the token criterion is measured in Mistral tokens.",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-profiles", type=int, default=120)
    args = ap.parse_args(argv)
    print(f"wrote {run(args)}")


if __name__ == "__main__":
    main()
