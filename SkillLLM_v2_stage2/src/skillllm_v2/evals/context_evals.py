"""Context-compaction and context-retrieval evals (module "context").

What is measured, and what kind of code measures it
---------------------------------------------------
Everything here is deterministic code run offline on SYNTHETIC profiles (seeded generator below; every result is
marked ``synthetic``).  There are no human labels and no LLM calls.

* The system under test is the production path ``build_cached_context_packet`` -> ``compact_context`` ->
  ``summarize_profile`` + ``build_skill_memory`` and ``ContextRetriever``.
* IMPORTANT, found while reading the module: production ``compact_context`` has NO token budget and NO importance
  ranking; it is a lossless re-packaging (and it stores every skill twice: ``profile_summary.skills`` and
  ``skill_memory.skills_by_label``).  Budget-dependent metrics therefore cannot be measured on production
  directly.  To still exercise "fidelity vs token savings" we define a REFERENCE budgeted compactor here
  (``reference_compact``: rank skills by confidence * (1 + ln #evidence), emit one verifiable evidence snippet
  each, stop at the budget).  It is eval-side scaffolding that shows what a budget-aware compactor could achieve and
  validates the metrics; it is NOT production code, and production budget adherence is reported separately.
* Gold "importance" comes from the generator's latent variable (high-importance skills get more mentions and higher
  confidence on average, with ~20% "quiet" important skills that have a single mention).  That makes the ranker
  partly informed by the same signals the generator used (circular by construction); the negative control (random
  drop at identical budget) is what makes the comparison meaningful, not the absolute recall.
* Token counts use a deterministic proxy (``\\w+|[^\\w\\s]`` regex units); real BPE counts are typically ~1.2-1.5x
  larger, so absolute budgets are proxy-tokens.  Ratios and the retention curve are unaffected by that scale.
"""

from __future__ import annotations

import json
import math
import random
import re
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from unittest import mock

from .contract import EvalContext, EvalResult, bootstrap_ci, paired_bootstrap

MODULE = "context"
DATASET = "synthetic_profiles_v1 (seeded generator, evals/context_evals.py; fixtures in inputs/evals/)"

# --------------------------------------------------------------------------------------------------------------
# Synthetic vocabulary
# --------------------------------------------------------------------------------------------------------------

# (canonical_id, canonical_name, surface_variants)
HARD_SKILLS: List[Tuple[str, str, List[str]]] = [
    ("python", "Python", ["Python"]), ("sql", "SQL", ["SQL"]), ("java", "Java", ["Java"]),
    ("javascript", "JavaScript", ["JavaScript", "JS"]), ("typescript", "TypeScript", ["TypeScript"]),
    ("golang", "Go", ["Golang"]), ("rust", "Rust", ["Rust"]), ("cpp", "C++", ["C++"]),
    ("kubernetes", "Kubernetes", ["Kubernetes", "k8s"]), ("docker", "Docker", ["Docker"]),
    ("aws", "Amazon Web Services", ["AWS"]), ("gcp", "Google Cloud Platform", ["GCP"]),
    ("azure", "Microsoft Azure", ["Azure"]), ("terraform", "Terraform", ["Terraform"]),
    ("ansible", "Ansible", ["Ansible"]), ("jenkins", "Jenkins", ["Jenkins"]),
    ("git", "Git", ["Git"]), ("linux", "Linux", ["Linux"]), ("bash", "Bash", ["Bash"]),
    ("postgresql", "PostgreSQL", ["PostgreSQL", "Postgres"]), ("mysql", "MySQL", ["MySQL"]),
    ("mongodb", "MongoDB", ["MongoDB"]), ("redis", "Redis", ["Redis"]), ("kafka", "Apache Kafka", ["Kafka"]),
    ("spark", "Apache Spark", ["Spark", "PySpark"]), ("airflow", "Apache Airflow", ["Airflow"]),
    ("dbt", "dbt", ["dbt"]), ("snowflake", "Snowflake", ["Snowflake"]), ("tableau", "Tableau", ["Tableau"]),
    ("powerbi", "Power BI", ["Power BI"]), ("excel", "Microsoft Excel", ["Excel"]),
    ("pandas", "pandas", ["pandas"]), ("numpy", "NumPy", ["NumPy"]), ("scikitlearn", "scikit-learn", ["scikit-learn", "sklearn"]),
    ("pytorch", "PyTorch", ["PyTorch"]), ("tensorflow", "TensorFlow", ["TensorFlow"]), ("keras", "Keras", ["Keras"]),
    ("nlp", "Natural Language Processing", ["NLP"]), ("cv", "Computer Vision", ["Computer Vision"]),
    ("ml", "Machine Learning", ["Machine Learning", "ML"]), ("dl", "Deep Learning", ["Deep Learning"]),
    ("statistics", "Statistics", ["Statistics"]), ("abtesting", "A/B Testing", ["A/B testing"]),
    ("react", "React", ["React"]), ("vue", "Vue.js", ["Vue"]), ("angular", "Angular", ["Angular"]),
    ("nodejs", "Node.js", ["Node.js"]), ("django", "Django", ["Django"]), ("flask", "Flask", ["Flask"]),
    ("fastapi", "FastAPI", ["FastAPI"]), ("spring", "Spring Boot", ["Spring Boot"]), ("graphql", "GraphQL", ["GraphQL"]),
    ("rest", "REST APIs", ["REST APIs"]), ("grpc", "gRPC", ["gRPC"]), ("rabbitmq", "RabbitMQ", ["RabbitMQ"]),
    ("elasticsearch", "Elasticsearch", ["Elasticsearch"]), ("prometheus", "Prometheus", ["Prometheus"]),
    ("grafana", "Grafana", ["Grafana"]), ("datadog", "Datadog", ["Datadog"]), ("cicd", "CI/CD", ["CI/CD"]),
    ("selenium", "Selenium", ["Selenium"]), ("pytest", "pytest", ["pytest"]), ("junit", "JUnit", ["JUnit"]),
    ("cybersecurity", "Cybersecurity", ["Cybersecurity"]), ("oauth", "OAuth", ["OAuth"]),
    ("networking", "Computer Networking", ["TCP/IP"]), ("swift", "Swift", ["Swift"]), ("kotlin", "Kotlin", ["Kotlin"]),
    ("android", "Android Development", ["Android"]), ("ios", "iOS Development", ["iOS"]),
    ("figma", "Figma", ["Figma"]), ("photoshop", "Adobe Photoshop", ["Photoshop"]), ("jira", "Jira", ["Jira"]),
    ("scrum", "Scrum", ["Scrum"]), ("kanban", "Kanban", ["Kanban"]), ("r", "R", ["R language"]),
    ("matlab", "MATLAB", ["MATLAB"]), ("sas", "SAS", ["SAS"]), ("hadoop", "Apache Hadoop", ["Hadoop"]),
    ("hive", "Apache Hive", ["Hive"]), ("bigquery", "BigQuery", ["BigQuery"]), ("redshift", "Amazon Redshift", ["Redshift"]),
    ("databricks", "Databricks", ["Databricks"]), ("mlflow", "MLflow", ["MLflow"]), ("kubeflow", "Kubeflow", ["Kubeflow"]),
    ("huggingface", "Hugging Face", ["Hugging Face"]), ("langchain", "LangChain", ["LangChain"]),
    ("opencv", "OpenCV", ["OpenCV"]), ("nginx", "NGINX", ["NGINX"]), ("helm", "Helm", ["Helm"]),
    ("istio", "Istio", ["Istio"]), ("vault", "HashiCorp Vault", ["Vault"]), ("sqlalchemy", "SQLAlchemy", ["SQLAlchemy"]),
    ("celery", "Celery", ["Celery"]), ("pydantic", "Pydantic", ["Pydantic"]), ("webpack", "Webpack", ["Webpack"]),
    ("nextjs", "Next.js", ["Next.js"]), ("tailwind", "Tailwind CSS", ["Tailwind"]), ("sass", "Sass", ["Sass"]),
    ("unity", "Unity", ["Unity"]), ("unreal", "Unreal Engine", ["Unreal"]), ("blender", "Blender", ["Blender"]),
    ("solidity", "Solidity", ["Solidity"]), ("ruby", "Ruby", ["Ruby"]), ("rails", "Ruby on Rails", ["Rails"]),
    ("php", "PHP", ["PHP"]), ("laravel", "Laravel", ["Laravel"]), ("scala", "Scala", ["Scala"]),
]
SOFT_SKILLS: List[Tuple[str, str, List[str]]] = [
    ("communication", "Communication", ["communication"]), ("leadership", "Leadership", ["leadership"]),
    ("teamwork", "Teamwork", ["teamwork"]), ("problem_solving", "Problem Solving", ["problem solving"]),
    ("mentoring", "Mentoring", ["mentoring"]), ("negotiation", "Negotiation", ["negotiation"]),
    ("presentation", "Presentation Skills", ["presentation skills"]), ("time_management", "Time Management", ["time management"]),
    ("adaptability", "Adaptability", ["adaptability"]), ("stakeholder_mgmt", "Stakeholder Management", ["stakeholder management"]),
]
DOMAINS = ["churn forecasting", "fraud alerts", "supply chain planning", "clinical trial reporting", "ad attribution",
           "payment reconciliation", "customer onboarding", "inventory tracking", "energy load balancing", "route optimization",
           "credit scoring", "support ticket triage", "recommendation feeds", "log anomaly detection", "genomics intake",
           "ride pricing", "warehouse scheduling", "claims processing", "document search", "sensor telemetry"]
VERBS = ["Built", "Shipped", "Maintained", "Designed", "Migrated", "Automated", "Optimized", "Led delivery of"]
FILLER = ["Passionate about continuous improvement and collaboration across teams.",
          "Available for relocation and open to hybrid arrangements.",
          "Seeking a challenging role with growth opportunities.",
          "References are available upon request.",
          "Enjoys hiking, chess and volunteering at community events.",
          "Holds a bachelor degree and several professional certificates.",
          "Native speaker of two languages with strong writing ability."]
NOISE_WORDS = ["team", "project", "data", "systems", "tools", "work"]  # spurious low-confidence extractions
SIZES = (10, 25, 60, 95)  # number of true skills per profile
BUDGETS = (128, 256, 512, 1024, 2048)  # proxy tokens

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Deterministic token proxy: word and punctuation units."""
    return len(_TOKEN_RE.findall(text))


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------------------------------------------
# Deterministic synthetic profile generator
# --------------------------------------------------------------------------------------------------------------


def make_profile(seed: int, n_skills: int, noise: float = 0.3, idx: int = 0) -> Dict[str, Any]:
    """One synthetic profile.  Returns text, extractions (with true offsets), gold importance and injected conflicts.

    Ground truth is fully known: ``gold_important`` (latent high importance), ``date_conflicts`` (skills given two
    contradictory date ranges), ``label_conflicts`` (skills extracted with both HARD and SOFT labels), ``noise_surfaces``.
    """
    rng = random.Random(f"ctxprof:{seed}:{n_skills}:{idx}")
    n_soft = min(len(SOFT_SKILLS), max(1, n_skills // 8))
    chosen = rng.sample(HARD_SKILLS, n_skills - n_soft) + rng.sample(SOFT_SKILLS, n_soft)
    rng.shuffle(chosen)
    n_high = max(2, round(0.2 * n_skills))
    tiers: Dict[str, str] = {}
    for i, (cid, _, _) in enumerate(chosen):
        tiers[cid] = "high" if i < n_high else ("mid" if i < n_high + round(0.4 * n_skills) else "low")

    mentions: List[Tuple[str, str]] = []  # (sentence, skill id) -- sentence may carry two skills
    sentences: List[str] = []
    extractions: List[Dict[str, Any]] = []
    quiet: List[str] = []
    topics = DOMAINS[:]
    rng.shuffle(topics)
    topic_i = 0

    def topic() -> str:
        nonlocal topic_i
        t = topics[topic_i % len(topics)]
        topic_i += 1
        # make topics unique across the profile even when the pool wraps around
        return t if topic_i <= len(topics) else f"{t} phase {topic_i // len(topics)}"

    plan: List[Tuple[Tuple[str, str, List[str]], int, float]] = []
    for item in chosen:
        cid = item[0]
        t = tiers[cid]
        if t == "high":
            if rng.random() < 0.2:
                quiet.append(cid)
                n_m, base = 1, rng.uniform(0.6, 0.8)
            else:
                n_m, base = rng.randint(2, 5), rng.uniform(0.8, 0.98)
        elif t == "mid":
            n_m, base = rng.randint(1, 2), rng.uniform(0.6, 0.9)
        else:
            n_m, base = 1, rng.uniform(0.35, 0.7)
        plan.append((item, n_m, base))

    # sentences: each mention becomes one sentence; occasionally a second skill shares the sentence
    pending = [(item, base) for item, n_m, base in plan for _ in range(n_m)]
    rng.shuffle(pending)
    k = 0
    sources = ["mistral", "bert", "gliner", "dictionary"]
    while k < len(pending):
        (cid, cname, variants), base = pending[k]
        surf = rng.choice(variants)
        tp = topic()
        verb = rng.choice(VERBS)
        pair = None
        if k + 1 < len(pending) and rng.random() < 0.25 and pending[k + 1][0][0] != cid:
            pair = pending[k + 1]
        if pair:
            (cid2, cname2, variants2), base2 = pair
            surf2 = rng.choice(variants2)
            sentence = f"{verb} {tp} workflows using {surf} and {surf2}."
            k += 2
        else:
            sentence = f"{verb} {tp} workflows using {surf}."
            k += 1
        sentences.append(sentence)
        mentions.append((sentence, cid))
        # record extraction specs now, offsets resolved after text assembly
        extractions.append({"_sentence": sentence, "_surface": surf, "canonical_skill_id": cid, "canonical_skill_name": cname,
                            "label": "SOFT_SKILL" if cid in {s[0] for s in SOFT_SKILLS} else "HARD_SKILL",
                            "confidence": max(0.05, min(0.99, base + rng.uniform(-0.05, 0.05))),
                            "source": rng.choice(sources)})
        if pair:
            extractions.append({"_sentence": sentence, "_surface": surf2, "canonical_skill_id": cid2, "canonical_skill_name": cname2,
                                "label": "SOFT_SKILL" if cid2 in {s[0] for s in SOFT_SKILLS} else "HARD_SKILL",
                                "confidence": max(0.05, min(0.99, base2 + rng.uniform(-0.05, 0.05))),
                                "source": rng.choice(sources)})

    # contradictory dates for ~15% of skills
    date_conflicts: List[str] = []
    ids = [c[0] for c in chosen]
    for cid in rng.sample(ids, max(1, round(0.15 * len(ids)))):
        name = next(c for c in chosen if c[0] == cid)[2][0]
        sentences.append(f"Used {name} from 2018 to 2021.")
        sentences.append(f"Worked with {name} between 2013 and 2015, before any commercial exposure.")
        date_conflicts.append(cid)

    # noise sentences (no skills)
    n_filler = int(round(noise * len(sentences)))
    for _ in range(n_filler):
        sentences.append(rng.choice(FILLER))
    # shuffle sentences deterministically while keeping the exact strings
    rng.shuffle(sentences)
    text = " ".join(sentences)

    # resolve offsets
    out: List[Dict[str, Any]] = []
    used: set = set()
    for e in extractions:
        sent_start = text.find(e["_sentence"])
        assert sent_start >= 0
        rel = e["_sentence"].find(e["_surface"])
        start = sent_start + rel
        end = start + len(e["_surface"])
        if (start, end) in used:  # same sentence twice cannot happen, but keep offsets unique
            continue
        used.add((start, end))
        rec = {"surface_form": e["_surface"], "label": e["label"], "start": start, "end": end, "source": e["source"],
               "confidence": round(e["confidence"], 3), "evidence": e["_sentence"].rstrip("."),
               "canonical_skill_id": e["canonical_skill_id"], "canonical_skill_name": e["canonical_skill_name"]}
        out.append(rec)

    # a few extractions without explicit evidence: the summarizer must fall back to a source snippet
    for rec in out[:: max(2, len(out) // 6)]:
        rec["evidence"] = ""

    # label conflicts: same surface extracted again with the other label
    label_conflicts: List[Dict[str, Any]] = []
    hard_recs = [r for r in out if r["label"] == "HARD_SKILL"]
    for j, rec in enumerate(rng.sample(hard_recs, min(len(hard_recs), max(2, n_skills // 10)))):
        dup = dict(rec)
        dup["label"] = "SOFT_SKILL"
        dup["source"] = "gliner"
        if j % 2 == 0:  # subtype "same_id": identical canonical id (merged by the summarizer)
            subtype = "same_id"
        else:  # subtype "diff_id": different canonical id, same surface
            dup["canonical_skill_id"] = rec["canonical_skill_id"] + "_alt"
            subtype = "diff_id"
        out.append(dup)
        label_conflicts.append({"surface": rec["surface_form"], "subtype": subtype})

    # spurious low-confidence noise extractions on common words (no canonical id)
    noise_surfaces: List[str] = []
    for w in NOISE_WORDS:
        for m in re.finditer(rf"\b{w}\b", text):
            if rng.random() < noise:
                out.append({"surface_form": w, "label": "HARD_SKILL", "start": m.start(), "end": m.end(), "source": "bert",
                            "confidence": round(rng.uniform(0.1, 0.45), 3), "evidence": "", "canonical_skill_id": None,
                            "canonical_skill_name": None})
                noise_surfaces.append(w)
                break

    rng.shuffle(out)
    gold_important = sorted(cid for cid, t in tiers.items() if t == "high")
    return {
        "profile_id": f"synthetic-{n_skills}-{idx}", "dataset": "synthetic", "n_skills": n_skills, "text": text,
        "extractions": out, "gold_important": gold_important, "quiet_important": sorted(quiet),
        "gold_skills": sorted(tiers), "tiers": tiers, "date_conflicts": sorted(date_conflicts),
        "label_conflicts": label_conflicts, "noise_surfaces": sorted(set(noise_surfaces)),
        "job_target": "data engineer",
    }


def make_profiles(ctx: EvalContext) -> List[Dict[str, Any]]:
    per_size = 2 if ctx.smoke else 10
    sizes = SIZES[:2] if ctx.smoke else SIZES
    return [make_profile(ctx.seed, n, noise=0.3, idx=i) for n in sizes for i in range(per_size)]


# --------------------------------------------------------------------------------------------------------------
# Production packet helpers
# --------------------------------------------------------------------------------------------------------------

_FIXED_TS = "2026-01-01T00:00:00+00:00"


@contextmanager
def fixed_clock(ts: str = _FIXED_TS) -> Iterator[None]:
    with mock.patch("skillllm_v2.context.profile_summarizer.utc_now_iso", lambda: ts):
        yield


@contextmanager
def ticking_clock() -> Iterator[None]:
    """A clock that advances one second per call, i.e. what the real clock does across a second boundary."""
    state = {"n": 0}

    def tick() -> str:
        state["n"] += 1
        return f"2026-01-01T00:{state['n'] // 60:02d}:{state['n'] % 60:02d}+00:00"

    with mock.patch("skillllm_v2.context.profile_summarizer.utc_now_iso", tick):
        yield


def build_production_packet(profile: Dict[str, Any], extractions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    from skillllm_v2.context.cached_context_builder import build_cached_context_packet

    return build_cached_context_packet(profile["text"], extractions if extractions is not None else profile["extractions"],
                                       job_target=profile["job_target"], user_id=profile["profile_id"])


def strip_timestamps(obj: Any) -> Any:
    """Remove the only wall-clock-dependent fields (generated_at, evidence timestamp, content_hash)."""
    if isinstance(obj, dict):
        return {k: strip_timestamps(v) for k, v in obj.items() if k not in {"generated_at", "timestamp", "content_hash"}}
    if isinstance(obj, list):
        return [strip_timestamps(v) for v in obj]
    return obj


def packet_skills(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    return packet["payload"]["stable_cag_context"]["profile_summary"]["skills"]


# --------------------------------------------------------------------------------------------------------------
# Reference budgeted compactor + negative control (eval-side scaffolding, see module docstring)
# --------------------------------------------------------------------------------------------------------------

SNIPPET_CHARS = 120


def _snippet(evidence: str, surface: str) -> str:
    """A substring of `evidence` (<= SNIPPET_CHARS) centred on the surface form when it occurs."""
    if len(evidence) <= SNIPPET_CHARS:
        return evidence
    i = evidence.lower().find(surface.lower())
    if i < 0:
        return evidence[:SNIPPET_CHARS]
    lo = max(0, min(i - SNIPPET_CHARS // 4, len(evidence) - SNIPPET_CHARS))
    return evidence[lo:lo + SNIPPET_CHARS]


def _entry(skill: Dict[str, Any]) -> Dict[str, Any]:
    ev = sorted(skill.get("evidence", []), key=lambda e: (-float(e.get("confidence", 0.0)), -len(e.get("text", "")), e.get("text", "")))
    best = ev[0]["text"] if ev else ""
    return {"id": skill.get("canonical_skill_id") or skill["surface_form"].strip().lower(), "skill": skill["surface_form"],
            "conf": round(float(skill.get("confidence", 0.0)), 2), "n_ev": len(skill.get("evidence", [])),
            "evidence": _snippet(best, skill["surface_form"])}


def importance_score(skill: Dict[str, Any]) -> float:
    return float(skill.get("confidence", 0.0)) * (1.0 + math.log(max(1, len(skill.get("evidence", [])))))


def reference_compact(packet: Dict[str, Any], budget: int, order: str = "importance", rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Keep skills (ranked by `order`) while the serialized compact packet stays within `budget` proxy-tokens."""
    skills = list(packet_skills(packet))
    if order == "importance":
        skills.sort(key=lambda s: (-importance_score(s), s["surface_form"].lower()))
    elif order == "random":
        assert rng is not None
        rng.shuffle(skills)
    elif order == "oracle":  # reserved for callers that pre-sort
        pass
    else:
        raise ValueError(order)
    return _fill(skills, budget, packet)


def _fill(skills: Sequence[Dict[str, Any]], budget: int, packet: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"packet_type": "reference_compact", "user_id": packet["payload"]["stable_cag_context"]["profile_summary"].get("user_id"), "skills": []}
    used = count_tokens(_json(out))
    for sk in skills:
        e = _entry(sk)
        cost = count_tokens(_json(e)) + 1
        if used + cost > budget:
            continue  # greedy knapsack: later (cheaper) entries may still fit
        out["skills"].append(e)
        used += cost
    return out


def compact_token_count(compact: Dict[str, Any]) -> int:
    return count_tokens(_json(compact))


# --------------------------------------------------------------------------------------------------------------
# Verification predicates (deterministic)
# --------------------------------------------------------------------------------------------------------------


def evidence_verifiable(snippet: str, source: str) -> bool:
    return bool(snippet) and snippet in source


def grounded_in(strings: Sequence[str], source: str, allowed: Sequence[str] = ()) -> List[str]:
    """Strings that are neither a substring of the source text (case-insensitive) nor one of `allowed` verbatim."""
    low = source.lower()
    allowed_set = set(allowed)
    return [s for s in strings if s and s.lower() not in low and s not in allowed_set]


def audit_production_packet(packet: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """Per-skill evidence preservation + hallucination audit of a production packet against the source text."""
    src = profile["text"]
    inputs = {e["surface_form"] for e in profile["extractions"]} | {e.get("canonical_skill_name") or "" for e in profile["extractions"]}
    skills = packet_skills(packet)
    has_verifiable = 0
    ev_total = ev_ok = 0
    ungrounded_surfaces: List[str] = []
    fabricated_by_compaction: List[str] = []
    for s in skills:
        texts = [e["text"] for e in s["evidence"]]
        ok = [evidence_verifiable(t, src) for t in texts]
        ev_total += len(ok)
        ev_ok += sum(ok)
        has_verifiable += int(any(ok))
        ungrounded_surfaces.extend(grounded_in([s["surface_form"]], src))
        fabricated_by_compaction.extend(grounded_in([s["surface_form"]], src, allowed=sorted(inputs)))
        fabricated_by_compaction.extend(grounded_in(texts, src, allowed=sorted(inputs)))
    return {"n_skills": len(skills), "skills_with_verifiable_evidence": has_verifiable, "evidence_total": ev_total,
            "evidence_substring_ok": ev_ok, "ungrounded_surfaces": ungrounded_surfaces,
            "fabricated_by_compaction": fabricated_by_compaction}


def audit_compact(compact: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    src = profile["text"]
    n = len(compact["skills"])
    ev_ok = sum(1 for e in compact["skills"] if evidence_verifiable(e["evidence"], src))
    bad_skill = [e["skill"] for e in compact["skills"] if e["skill"].lower() not in src.lower()]
    return {"n": n, "evidence_ok": ev_ok, "ungrounded_skills": bad_skill}


def retained_ids(compact: Dict[str, Any]) -> set:
    return {e["id"] for e in compact["skills"]}


def packet_ids(packet: Dict[str, Any]) -> set:
    return {s.get("canonical_skill_id") or s["surface_form"].strip().lower() for s in packet_skills(packet)}


def recall(retained: set, gold: Sequence[str]) -> float:
    return len(retained & set(gold)) / len(gold) if gold else 1.0


# --------------------------------------------------------------------------------------------------------------
# Result helpers
# --------------------------------------------------------------------------------------------------------------


def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v) if v else float("nan")


def _res(name: str, metric: str, vals: Sequence[float], ctx: EvalContext, details: Optional[Dict[str, Any]] = None,
         caveats: Optional[List[str]] = None, dataset: str = DATASET) -> EvalResult:
    if not vals:
        return EvalResult(name, MODULE, metric, None, 0, dataset, status="skipped", reason="no units", details=details or {}, caveats=caveats or [])
    ci = bootstrap_ci(vals, n_boot=ctx.n_boot, seed=ctx.seed) if len(vals) > 1 else None
    return EvalResult(name, MODULE, metric, _mean(vals), len(vals), dataset, ci=ci, details=details or {}, caveats=caveats or [])


# --------------------------------------------------------------------------------------------------------------
# Context compaction evals
# --------------------------------------------------------------------------------------------------------------


def eval_production_packet(ctx: EvalContext, profiles: List[Dict[str, Any]]) -> List[EvalResult]:
    res: List[EvalResult] = []
    ratio_raw, ratio_naive, ratio_chars, expansion_dup = [], [], [], []
    ret_imp, ret_all = [], []
    ev_pres, ev_sub, dup_ev, no_dup_skill = [], [], [], []
    halluc_src, halluc_comp = [], []
    by_size: Dict[int, Dict[str, List[float]]] = {}
    with fixed_clock():
        for p in profiles:
            packet = build_production_packet(p)
            ser = _json(packet)
            raw_tok = count_tokens(p["text"])
            naive_tok = count_tokens(p["text"]) + count_tokens(_json(p["extractions"]))
            pk_tok = count_tokens(ser)
            ratio_raw.append(pk_tok / raw_tok)
            ratio_naive.append(pk_tok / naive_tok)
            ratio_chars.append(len(ser) / len(p["text"]))
            ids = packet_ids(packet)
            ret_imp.append(recall(ids, p["gold_important"]))
            ret_all.append(recall(ids, p["gold_skills"]))
            a = audit_production_packet(packet, p)
            ev_pres.append(a["skills_with_verifiable_evidence"] / max(1, a["n_skills"]))
            ev_sub.append(a["evidence_substring_ok"] / max(1, a["evidence_total"]))
            halluc_src.append(len(a["ungrounded_surfaces"]) / max(1, a["n_skills"]))
            halluc_comp.append(len(a["fabricated_by_compaction"]) / max(1, a["n_skills"]))
            skills = packet_skills(packet)
            texts = [(s["surface_form"], e["text"]) for s in skills for e in s["evidence"]]
            dup_ev.append(1.0 - len(set(texts)) / max(1, len(texts)))
            # skills stored twice (profile_summary.skills and skill_memory.skills_by_label)
            mem = packet["payload"]["stable_cag_context"]["skill_memory"]["skills_by_label"]
            n_mem = sum(len(v) for v in mem.values())
            expansion_dup.append(1.0 + n_mem / max(1, len(skills)))
            b = by_size.setdefault(p["n_skills"], {"ratio_raw": [], "ret_imp": []})
            b["ratio_raw"].append(pk_tok / raw_tok)
            b["ret_imp"].append(ret_imp[-1])

    size_detail = {str(k): {"mean_packet_over_raw_tokens": round(_mean(v["ratio_raw"]), 3), "n_profiles": len(v["ratio_raw"])}
                   for k, v in sorted(by_size.items())}
    res.append(_res("context.packet_over_raw_tokens", "packet_tokens / raw_profile_tokens (>1 = expansion, <1 = savings)", ratio_raw, ctx,
                    details={"by_profile_size": size_detail, "token_proxy": "regex word+punct units"},
                    caveats=["production compact_context does no pruning: it re-packages, so values >1 mean the 'compacted' packet is LARGER than the raw text"]))
    res.append(_res("context.packet_over_naive_tokens", "packet_tokens / (raw_text + extraction_json) tokens", ratio_naive, ctx,
                    details={"note": "naive = what a prompt would carry without the packet"}))
    res.append(_res("context.packet_over_raw_chars", "packet_chars / raw_profile_chars", ratio_chars, ctx))
    res.append(_res("context.skill_storage_multiplicity", "copies of each skill stored in the packet (profile_summary.skills + skill_memory.skills_by_label; 2.0 = every skill stored twice)",
                    expansion_dup, ctx, caveats=["packet-size inefficiency: profile_summary.skills duplicated in skill_memory.skills_by_label"]))
    res.append(_res("context.production_recall_important", "recall of gold-important skills in production packet", ret_imp, ctx,
                    caveats=["no skill is ever dropped by production compaction, so recall is bounded only by extraction input quality; gold comes from the generator"]))
    res.append(_res("context.production_recall_all", "recall of all gold skills in production packet", ret_all, ctx))
    res.append(_res("context.evidence_preservation_rate", "share of packet skills with >=1 evidence snippet that is a substring of source", ev_pres, ctx,
                    caveats=["includes skills whose evidence is the fallback snippet built from offsets; noise extractions use surface as evidence"]))
    res.append(_res("context.evidence_substring_rate", "share of all packet evidence strings that are exact substrings of source", ev_sub, ctx))
    res.append(_res("context.ungrounded_skill_rate", "packet skill surfaces not found in source text (inherited from extraction input), per skill", halluc_src, ctx))
    res.append(_res("context.compaction_fabrication_rate", "packet strings found in neither the source nor the extraction input (introduced by compaction), per skill", halluc_comp, ctx,
                    details={"claim": "0.0 means compaction never invents a skill or evidence string"}))
    res.append(_res("context.duplicate_evidence_rate", "share of repeated (skill, evidence text) pairs kept in the packet", dup_ev, ctx,
                    caveats=["summarize_profile appends one Evidence per extraction without de-duplication, so repeated evidence inflates the packet"]))
    return res


def eval_determinism(ctx: EvalContext, profiles: List[Dict[str, Any]]) -> List[EvalResult]:
    same_clock, tick_raw, tick_mod, hash_tick, order_inv = [], [], [], [], []
    for p in profiles:
        with fixed_clock():
            a = _json(build_production_packet(p))
            b = _json(build_production_packet(p))
        same_clock.append(float(a == b))
        with ticking_clock():
            pa = build_production_packet(p)
            pb = build_production_packet(p)
        tick_raw.append(float(_json(pa) == _json(pb)))
        hash_tick.append(float(pa["content_hash"] == pb["content_hash"]))
        tick_mod.append(float(_json(strip_timestamps(pa)) == _json(strip_timestamps(pb))))
        rng = random.Random(f"perm:{ctx.seed}:{p['profile_id']}")
        shuffled = list(p["extractions"])
        rng.shuffle(shuffled)
        with fixed_clock():
            c = build_production_packet(p, shuffled)
            d = build_production_packet(p)
        order_inv.append(float(_json(strip_timestamps(c)) == _json(strip_timestamps(d))))
    return [
        _res("context.determinism_same_clock", "byte-identical packet for identical input within one clock tick", same_clock, ctx),
        _res("context.determinism_across_clock_ticks", "byte-identical packet (incl. content_hash) when the wall clock advances between calls", tick_raw, ctx,
             details={"content_hash_equal_rate": _mean(hash_tick)},
             caveats=["FAILS by design of the module: summarize_profile stamps generated_at/Evidence.timestamp with utc_now_iso(), which feeds content_hash, so the 'stable' cache key changes every second"]),
        _res("context.determinism_modulo_timestamps", "byte-identical packet after removing generated_at/timestamp/content_hash", tick_mod, ctx),
        _res("context.order_invariance_modulo_timestamps", "packet unchanged when the extraction list is permuted (timestamps stripped)", order_inv, ctx,
             caveats=["evidence order and source_offsets order follow the input order, so a permuted extraction list changes the bytes"]),
    ]


def eval_conflicts(ctx: EvalContext, profiles: List[Dict[str, Any]]) -> List[EvalResult]:
    det_all, det_same, det_diff, date_found = [], [], [], []
    n_by = {"same_id": 0, "diff_id": 0}
    with fixed_clock():
        for p in profiles:
            packet = build_production_packet(p)
            notes = " ".join(packet["payload"]["conflict_notes"]).lower()
            for c in p["label_conflicts"]:
                hit = float(re.sub(r"[\W_]+", " ", c["surface"].lower()).strip() in notes)
                det_all.append(hit)
                (det_same if c["subtype"] == "same_id" else det_diff).append(hit)
                n_by[c["subtype"]] += 1
            # date contradictions: is any date or skill with a date conflict surfaced in notes?
            for cid in p["date_conflicts"]:
                date_found.append(float(("2013" in notes or "2018" in notes or "date" in notes or "contradict" in notes)))
    return [
        _res("context.label_conflict_detection", "injected HARD/SOFT label conflicts surfaced in conflict_notes", det_all, ctx,
             details={"same_canonical_id": _mean(det_same) if det_same else None, "different_canonical_id": _mean(det_diff) if det_diff else None,
                      "n_by_subtype": n_by},
             caveats=["_find_conflicts runs on already-merged facts keyed by canonical id or normalized surface, so conflicts between extractions that merge into one fact are silently lost"]),
        _res("context.date_conflict_detection", "injected contradictory date ranges surfaced in conflict_notes", date_found, ctx,
             caveats=["capability absent: the summarizer never parses dates, conflict_notes only covers label conflicts; 0.0 is a documented gap, not a regression"]),
    ]


def eval_budgeted(ctx: EvalContext, profiles: List[Dict[str, Any]]) -> List[EvalResult]:
    res: List[EvalResult] = []
    n_draw = 3 if ctx.smoke else 8
    curve: Dict[str, Dict[str, Any]] = {}
    per_budget_imp: Dict[int, List[float]] = {b: [] for b in BUDGETS}
    per_budget_rand: Dict[int, List[float]] = {b: [] for b in BUDGETS}
    per_budget_all: Dict[int, List[float]] = {b: [] for b in BUDGETS}
    per_budget_oracle: Dict[int, List[float]] = {b: [] for b in BUDGETS}
    ratio_ref: Dict[int, List[float]] = {b: [] for b in BUDGETS}
    adhere_ref: List[float] = []
    adhere_rand: List[float] = []
    adhere_prod: List[float] = []
    ev_ok_ref: List[float] = []
    halluc_ref: List[float] = []
    quiet_missed: List[float] = []
    with fixed_clock():
        for p in profiles:
            packet = build_production_packet(p)
            prod_tok = count_tokens(_json(packet))
            raw_tok = count_tokens(p["text"])
            gold = p["gold_important"]
            for b in BUDGETS:
                adhere_prod.append(float(prod_tok <= b))
                ref = reference_compact(packet, b)
                tok = compact_token_count(ref)
                adhere_ref.append(float(tok <= b))
                per_budget_imp[b].append(recall(retained_ids(ref), gold))
                per_budget_all[b].append(recall(retained_ids(ref), p["gold_skills"]))
                ratio_ref[b].append(tok / raw_tok)
                a = audit_compact(ref, p)
                if a["n"]:
                    ev_ok_ref.append(a["evidence_ok"] / a["n"])
                    halluc_ref.append(len(a["ungrounded_skills"]) / a["n"])
                # random-drop control: mean over n_draw seeded shuffles
                rr = []
                for d in range(n_draw):
                    rng = random.Random(f"rnd:{ctx.seed}:{p['profile_id']}:{b}:{d}")
                    rc = reference_compact(packet, b, order="random", rng=rng)
                    adhere_rand.append(float(compact_token_count(rc) <= b))
                    rr.append(recall(retained_ids(rc), gold))
                per_budget_rand[b].append(_mean(rr))
                # oracle: gold-important first, then the rest by importance -- upper bound for any ranker at this budget
                skills = sorted(packet_skills(packet), key=lambda s: (0 if (s.get("canonical_skill_id") in set(gold)) else 1, -importance_score(s), s["surface_form"].lower()))
                oc = _fill(skills, b, packet)
                per_budget_oracle[b].append(recall(retained_ids(oc), gold))
                if b == 512 and p["quiet_important"]:
                    quiet_missed.append(1.0 - recall(retained_ids(ref), p["quiet_important"]))

    for b in BUDGETS:
        pb = paired_bootstrap(per_budget_imp[b], per_budget_rand[b], n_boot=ctx.n_boot, seed=ctx.seed)
        curve[str(b)] = {"importance_ranked": round(_mean(per_budget_imp[b]), 4), "random_drop": round(_mean(per_budget_rand[b]), 4),
                         "oracle_upper_bound": round(_mean(per_budget_oracle[b]), 4), "all_skills_recall": round(_mean(per_budget_all[b]), 4),
                         "delta_vs_random": round(pb["delta"], 4), "delta_ci": [round(pb["ci_low"], 4), round(pb["ci_high"], 4)], "p": pb["p"],
                         "ref_tokens_over_raw_tokens": round(_mean(ratio_ref[b]), 3)}
    res.append(EvalResult("context.retention_vs_budget_curve", MODULE, "recall of gold-important skills per budget (reference compactor vs random drop vs oracle)",
                          _mean([_mean(per_budget_imp[b]) for b in BUDGETS]), len(profiles), DATASET, details={"curve": curve, "budgets_proxy_tokens": list(BUDGETS)},
                          caveats=["reference compactor is eval-side scaffolding, not production code",
                                   "gold importance is the generator's latent tier; ranker uses confidence and evidence count which the generator correlated with it (circular), the control is the random-drop comparison"]))
    for b in (256, 512, 1024):
        pb = paired_bootstrap(per_budget_imp[b], per_budget_rand[b], n_boot=ctx.n_boot, seed=ctx.seed)
        res.append(_res(f"context.retention_important_at_{b}", f"recall of gold-important skills at {b} proxy-tokens (reference compactor)", per_budget_imp[b], ctx,
                        details={"random_drop_mean": _mean(per_budget_rand[b]), "paired_delta_vs_random": pb["delta"], "delta_ci": [pb["ci_low"], pb["ci_high"]], "p": pb["p"],
                                 "oracle_mean": _mean(per_budget_oracle[b])}))
    # negative control verdict
    wins = [1.0 if a > r else 0.0 for b in (256, 512, 1024) for a, r in zip(per_budget_imp[b], per_budget_rand[b])]
    deltas_mid = paired_bootstrap([per_budget_imp[b][i] for b in (256, 512) for i in range(len(profiles))],
                                  [per_budget_rand[b][i] for b in (256, 512) for i in range(len(profiles))], n_boot=ctx.n_boot, seed=ctx.seed)
    ok = deltas_mid["ci_low"] > 0
    res.append(EvalResult("context.negative_control_random_drop", MODULE, "importance-ranked minus random-drop recall (tight budgets 256,512), paired over profiles",
                          deltas_mid["delta"], len(profiles), DATASET, ci=(deltas_mid["ci_low"], deltas_mid["ci_high"]),
                          details={"p": deltas_mid["p"], "control_passed": ok, "profile_win_rate": _mean(wins)},
                          status="ok" if ok else "failed", reason="" if ok else "random drop is not worse than importance ranking: the retention metric lacks discriminative power",
                          caveats=["a passing control shows the metric can tell a ranked compactor from a random one; it does not validate the generator's notion of importance"]))
    res.append(_res("context.reference_budget_adherence", "reference compactor output tokens <= budget (all budgets/profiles)", adhere_ref, ctx,
                    details={"random_drop_adherence": _mean(adhere_rand)}))
    res.append(_res("context.production_budget_adherence", "production packet tokens <= budget (budgets 128..2048 proxy-tokens)", adhere_prod, ctx,
                    details={"budgets": list(BUDGETS)},
                    caveats=["production compact_context accepts no budget parameter; this is the share of (profile, budget) cells that happen to fit, i.e. a measured absence of a feature"]))
    res.append(_res("context.reference_evidence_preservation", "retained reference-compactor skills whose evidence snippet is a substring of source", ev_ok_ref, ctx))
    res.append(_res("context.reference_hallucination_rate", "retained reference-compactor skill surfaces missing from the source", halluc_ref, ctx))
    res.append(_res("context.quiet_important_miss_at_512", "share of single-mention important skills dropped at 512 tokens (failure mode of confidence/count ranking)", quiet_missed, ctx,
                    caveats=["only profiles that contain 'quiet' important skills"]))
    return res


def eval_audit_sensitivity(ctx: EvalContext, profiles: List[Dict[str, Any]]) -> List[EvalResult]:
    """Does the hallucination/evidence audit itself notice corruption?  Inject fabricated skills/evidence into a packet."""
    caught_skill, caught_ev, clean_flags = [], [], []
    with fixed_clock():
        for p in profiles:
            packet = build_production_packet(p)
            a0 = audit_production_packet(packet, p)
            clean_flags.append(float(bool(a0["fabricated_by_compaction"])))
            # fabricated skill
            bad = json.loads(_json(packet))
            sk = packet_skills(bad)
            sk.append({"surface_form": "Quantum Basketweaving", "label": "HARD_SKILL", "canonical_skill_id": None, "canonical_skill_name": None,
                       "confidence": 0.9, "evidence": [{"source": "x", "text": "Quantum Basketweaving at CERN", "confidence": 0.9, "timestamp": None}], "metadata": {}})
            a1 = audit_production_packet(bad, p)
            caught_skill.append(float(bool(a1["fabricated_by_compaction"])))
            # corrupted evidence: alter one character in the first evidence string
            bad2 = json.loads(_json(packet))
            for s in packet_skills(bad2):
                if s["evidence"] and len(s["evidence"][0]["text"]) > 5:
                    t = s["evidence"][0]["text"]
                    s["evidence"][0]["text"] = t[:3] + "#" + t[4:]
                    break
            a2 = audit_production_packet(bad2, p)
            caught_ev.append(float(a2["evidence_substring_ok"] < a0["evidence_substring_ok"]))
    return [
        _res("context.audit_sensitivity_fabricated_skill", "injected fabricated skill+evidence flagged by the hallucination audit", caught_skill, ctx),
        _res("context.audit_sensitivity_corrupted_evidence", "single-character corruption of an evidence snippet flagged by the substring audit", caught_ev, ctx),
        _res("context.audit_false_positive_rate", "clean production packets flagged as fabricating strings", clean_flags, ctx),
    ]


# --------------------------------------------------------------------------------------------------------------
# ContextRetriever: recall@k / MRR vs trivial keyword baselines
# --------------------------------------------------------------------------------------------------------------

_STOP_FREE = re.compile(r"[\W_]+")


def _norm(s: str) -> str:
    return _STOP_FREE.sub(" ", s.lower()).strip()


def make_queries(p: Dict[str, Any], rng: random.Random, per_type: int = 6) -> List[Dict[str, Any]]:
    """Queries with known gold skill ids.  Types: name, alias, context, natural."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for e in p["extractions"]:
        if e["canonical_skill_id"] and e["canonical_skill_id"] in p["tiers"]:
            by_id.setdefault(e["canonical_skill_id"], e)
    ids = sorted(by_id)
    rng.shuffle(ids)
    out: List[Dict[str, Any]] = []
    for cid in ids[:per_type]:
        e = by_id[cid]
        out.append({"type": "name", "query": e["surface_form"], "gold": [cid]})
    alias_ids = [c for c in ids if _norm(by_id[c]["canonical_skill_name"]) != _norm(by_id[c]["surface_form"])]
    for cid in alias_ids[:per_type]:
        out.append({"type": "alias", "query": by_id[cid]["canonical_skill_name"], "gold": [cid]})
    for cid in ids[per_type:2 * per_type]:
        e = by_id[cid]
        out.append({"type": "natural", "query": f"what experience does the candidate have with {e['surface_form']}", "gold": [cid]})
    # context queries: the topic phrase of a sentence, gold = all skills extracted from that sentence
    sent_to_ids: Dict[str, set] = {}
    for e in p["extractions"]:
        ev = e["evidence"]
        if ev and e["canonical_skill_id"] in p["tiers"]:
            sent_to_ids.setdefault(ev, set()).add(e["canonical_skill_id"])
    sents = sorted(sent_to_ids)
    rng.shuffle(sents)
    for s in sents[:per_type]:
        m = re.search(r"(?:Built|Shipped|Maintained|Designed|Migrated|Automated|Optimized|Led delivery of) (.+?) workflows", s)
        if m:
            out.append({"type": "context", "query": m.group(1), "gold": sorted(sent_to_ids[s])})
    return out


def _rank_ids_context_retriever(packet: Dict[str, Any], query: str, k: int) -> List[str]:
    from skillllm_v2.context.context_retriever import ContextRetriever

    hits = ContextRetriever(packet).retrieve_skills(query, top_k=k)
    return [h.get("canonical_skill_id") or h["surface_form"].strip().lower() for h in hits]


def _rank_ids_surface_baseline(packet: Dict[str, Any], query: str, k: int) -> List[str]:
    """Trivial keyword baseline: token overlap with the surface form only (no canonical name, no evidence)."""
    terms = set(_norm(query).split())
    scored = []
    for s in packet_skills(packet):
        sc = len(terms & set(_norm(s["surface_form"]).split()))
        if sc:
            scored.append((sc, s.get("canonical_skill_id") or s["surface_form"].strip().lower()))
    scored.sort(key=lambda t: -t[0])
    return [i for _, i in scored[:k]]


def _rank_ids_random(packet: Dict[str, Any], rng: random.Random, k: int) -> List[str]:
    ids = [s.get("canonical_skill_id") or s["surface_form"].strip().lower() for s in packet_skills(packet)]
    rng.shuffle(ids)
    return ids[:k]


def rr_and_recall(ranked: Sequence[str], gold: Sequence[str], k: int) -> Tuple[float, float, float]:
    """(reciprocal rank of first gold, recall@k, hit@1)."""
    g = set(gold)
    rr = 0.0
    for i, x in enumerate(ranked, 1):
        if x in g:
            rr = 1.0 / i
            break
    return rr, len(set(ranked[:k]) & g) / len(g), float(bool(ranked) and ranked[0] in g)


def eval_context_retriever(ctx: EvalContext, profiles: List[Dict[str, Any]]) -> List[EvalResult]:
    k = 5
    types = ["name", "alias", "natural", "context"]
    agg: Dict[str, Dict[str, List[float]]] = {m: {} for m in ("retriever", "surface_baseline", "random")}
    per_profile: Dict[str, List[Tuple[float, float, float]]] = {m: [] for m in agg}
    with fixed_clock():
        for p in profiles:
            packet = build_production_packet(p)
            rng = random.Random(f"q:{ctx.seed}:{p['profile_id']}")
            qs = make_queries(p, rng)
            vals: Dict[str, Dict[str, List[Tuple[float, float, float]]]] = {m: {t: [] for t in types} for m in agg}
            for q in qs:
                r_ret = rr_and_recall(_rank_ids_context_retriever(packet, q["query"], k), q["gold"], k)
                r_base = rr_and_recall(_rank_ids_surface_baseline(packet, q["query"], k), q["gold"], k)
                rrng = random.Random(f"r:{ctx.seed}:{p['profile_id']}:{q['query']}")
                r_rand = rr_and_recall(_rank_ids_random(packet, rrng, k), q["gold"], k)
                vals["retriever"][q["type"]].append(r_ret)
                vals["surface_baseline"][q["type"]].append(r_base)
                vals["random"][q["type"]].append(r_rand)
            for m in agg:
                allq = [x for t in types for x in vals[m][t]]
                per_profile[m].append(tuple(_mean([x[i] for x in allq]) for i in range(3)))  # type: ignore
                for t in types:
                    if vals[m][t]:
                        for i, key in enumerate(("mrr", "recall", "hit1")):
                            agg[m].setdefault(f"{t}.{key}", []).append(_mean([x[i] for x in vals[m][t]]))
    res: List[EvalResult] = []
    mrr = {m: [x[0] for x in per_profile[m]] for m in agg}
    rec = {m: [x[1] for x in per_profile[m]] for m in agg}
    hit = {m: [x[2] for x in per_profile[m]] for m in agg}
    by_type = {m: {kk: round(_mean(v), 4) for kk, v in agg[m].items()} for m in agg}
    pb_mrr = paired_bootstrap(mrr["retriever"], mrr["surface_baseline"], n_boot=ctx.n_boot, seed=ctx.seed)
    pb_rec = paired_bootstrap(rec["retriever"], rec["surface_baseline"], n_boot=ctx.n_boot, seed=ctx.seed)
    caveats = ["queries are synthetic with template-derived gold (name/alias/natural/context); 'context' queries use topic phrases unique to a sentence, so they flatter any lexical method",
               "the surface baseline is token overlap on surface_form only; ContextRetriever additionally indexes canonical names and evidence text, so the comparison isolates those two fields",
               "unit of analysis = profile (mean over that profile's queries)"]
    res.append(_res("context.retriever_mrr", "MRR of ContextRetriever.retrieve_skills (top-5)", mrr["retriever"], ctx,
                    details={"surface_baseline_mean": _mean(mrr["surface_baseline"]), "random_mean": _mean(mrr["random"]), "paired_vs_baseline": pb_mrr, "by_query_type": by_type}, caveats=caveats))
    res.append(_res("context.retriever_recall_at_5", "recall@5 of ContextRetriever.retrieve_skills", rec["retriever"], ctx,
                    details={"surface_baseline_mean": _mean(rec["surface_baseline"]), "random_mean": _mean(rec["random"]), "paired_vs_baseline": pb_rec}, caveats=caveats))
    res.append(_res("context.retriever_hit_at_1", "hit@1 of ContextRetriever.retrieve_skills", hit["retriever"], ctx, details={"surface_baseline_mean": _mean(hit["surface_baseline"])}, caveats=caveats))
    res.append(_res("context.surface_baseline_mrr", "MRR of trivial surface-form keyword baseline", mrr["surface_baseline"], ctx, caveats=caveats))
    res.append(_res("context.random_baseline_mrr", "MRR of random ranking (negative control)", mrr["random"], ctx, caveats=["negative control: any real retriever must beat this"]))
    return res


# --------------------------------------------------------------------------------------------------------------
# Checked-in fixtures (inputs/evals/): regenerated deterministically, a test asserts they have not drifted
# --------------------------------------------------------------------------------------------------------------


def fixture_payload(seed: int = 42) -> Dict[str, Any]:
    profiles = []
    for n in (10, 25):
        p = make_profile(seed, n, noise=0.3, idx=0)
        qs = make_queries(p, random.Random(f"q:{seed}:{p['profile_id']}"), per_type=3)
        profiles.append({**p, "queries": qs})
    return {"dataset": "synthetic", "generator": "skillllm_v2.evals.context_evals.make_profile", "seed": seed,
            "note": "SYNTHETIC profiles: all text, skills, offsets and gold labels are generated, not real user data", "profiles": profiles}


def write_fixtures(root: Any, seed: int = 42) -> str:
    from pathlib import Path

    path = Path(root) / "inputs" / "evals" / "context_profiles_synthetic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture_payload(seed), indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------------------


def skipped_results() -> List[EvalResult]:
    return [
        EvalResult("context.importance_vs_human_labels", MODULE, "agreement between compaction ranking and human-labelled skill importance", None, 0, "none", status="skipped",
                   reason="no human-labelled importance exists offline; gold importance is the synthetic generator's latent tier only"),
        EvalResult("context.downstream_answer_fidelity", MODULE, "same downstream answers from compacted vs full context", None, 0, "none", status="skipped",
                   reason="requires an LLM to answer from each context; no network/model calls allowed"),
        EvalResult("context.real_profile_compaction", MODULE, "compression and retention on real user profiles", None, 0, "none", status="skipped",
                   reason="only one 15-word demo profile is checked in and it has no gold importance; real profiles are not available"),
    ]


def run(ctx: EvalContext) -> List[EvalResult]:
    profiles = make_profiles(ctx)
    results: List[EvalResult] = []
    for fn in (eval_production_packet, eval_determinism, eval_conflicts, eval_budgeted, eval_audit_sensitivity, eval_context_retriever):
        try:
            results.extend(fn(ctx, profiles))
        except Exception as exc:  # an eval crash must not hide the other results
            results.append(EvalResult(f"context.{fn.__name__}", MODULE, fn.__name__, None, 0, DATASET, status="failed", reason=f"{type(exc).__name__}: {exc}"))
    results.extend(skipped_results())
    return results
