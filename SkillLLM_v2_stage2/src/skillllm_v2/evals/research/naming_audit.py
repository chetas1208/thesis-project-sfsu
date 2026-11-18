"""Naming-collision audit for the proposed research name "SkillGraphLM" and its variants (owner: A6).

Queries public, key-less search APIs (arXiv, OpenAlex, Semantic Scholar, DBLP, Crossref, ACL Anthology bibliography, GitHub via
an authenticated ``gh`` CLI if present, PyPI, Hugging Face models/datasets/spaces, npm) and writes one record per
(term, source) with the date, hit count, the hits that match the term exactly after normalisation, and a *screening* risk.
General-web, Google-Scholar and trademark searches cannot be scripted reliably (no public API, bot blocking); they are run by
hand and merged from ``--manual JSON`` (same record shape). Web content is untrusted data: only titles, names, URLs and short
descriptions are stored, nothing is executed.

    PYTHONPATH=src python -m skillllm_v2.evals.research.naming_audit [--manual reports/.../manual_queries.json]
                                                                      [--acl-bib PATH] [--out reports/skillllm_v2/evals/naming_audit]

Screening rule (fixed before running; final risk is adjudicated in REPORT.md and may be raised, never silently lowered):
  * exact  = the normalised term (lower-case, alphanumerics only) equals a normalised token / token pair / token triple of a hit
             title or name;  substring = the normalised term occurs inside the normalised hit text.
  * scholarly source (arXiv, OpenAlex, S2, DBLP, Crossref, ACL): exact in a title -> high; substring only -> low; nothing -> none.
  * package registry (PyPI, npm) exact project name taken -> medium (high if the description is in the skills/jobs/education
    domain); not taken -> none.
  * code / model hubs (GitHub, Hugging Face): exact name -> high if the hit is in the skills/jobs/education/LLM domain or has
    >= 50 stars/likes, otherwise medium; substring only -> low; nothing -> none.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skillllm_v2.evals.research.common import EVALS_ROOT

TERMS = ["SkillGraphLM", "SkillGraph-LM", "SkillGraph LM", "SkillGraph", "Skill-LLM", "SkillLLM", "SkillGPT"]
TITLE = "SkillGraphLM: Non-Regressive Skill Extraction and Graph-Augmented Learning-Path Intelligence"
DOMAIN_WORDS = ("skill", "job", "career", "resume", "cv ", "course", "learning path", "education", "esco", "labor", "labour",
                "recruit", "talent", "curricul", "occupation", "workforce", "hiring", "competenc")
UA = "SkillLLM_v2-naming-audit/1.0 (academic research; contact via repository owner)"
RANK = {"unknown": -1, "none": 0, "low": 1, "medium": 2, "high": 3}
OUT_DEFAULT = EVALS_ROOT / "naming_audit"
ACL_BIB_URL = "https://aclanthology.org/anthology.bib.gz"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def match_kind(term: str, text: str) -> str:
    """'exact' if the normalised term equals a normalised 1-3 token window of ``text``; 'substring'; or ''."""
    t = norm(term)
    if not t or not text:
        return ""
    toks = [norm(x) for x in re.split(r"[\s/_:;,()\[\]{}|\"'.!?]+", text) if norm(x)]
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            if "".join(toks[i:i + n]) == t:
                return "exact"
    return "substring" if t in norm(text) else ""


def in_domain(text: str) -> bool:
    low = (text or "").lower()
    return any(w in low for w in DOMAIN_WORDS)


def http_get(url: str, *, accept: str = "application/json", retries: int = 4, timeout: int = 30) -> Tuple[int, bytes]:
    last: Tuple[int, bytes] = (0, b"")
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed public https endpoints
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            last = (e.code, e.read() if hasattr(e, "read") else b"")
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1) ** 2)
                continue
            return last
        except Exception as e:  # noqa: BLE001 - network failures are recorded, not fatal
            last = (0, str(e).encode())
            time.sleep(2 * (attempt + 1))
    return last


def _hit(title: str, url: str, extra: str = "", **kw: Any) -> Dict[str, Any]:
    return {"title": (title or "")[:240], "url": url, "extra": (extra or "")[:240], **kw}


# ----------------------------------------------------------------------------------------------------------------- sources
def src_arxiv(term: str) -> Dict[str, Any]:
    q = f'all:"{term}"' if " " in term or "-" in term else f"all:{term}"
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": 25})
    code, body = http_get(url, accept="application/atom+xml")
    if code != 200:
        return {"status": f"http_{code}", "url": url}
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    root = ET.fromstring(body)
    total = int(root.findtext("o:totalResults", default="0", namespaces=ns))
    hits = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", default="", namespaces=ns) or "").split())
        summ = " ".join((e.findtext("a:summary", default="", namespaces=ns) or "").split())
        hits.append(_hit(title, e.findtext("a:id", default="", namespaces=ns), summ[:200], kind="paper",
                         title_match=match_kind(term, title), abstract_match=match_kind(term, summ)))
    return {"status": "ok", "url": url, "hit_count": total, "hits": hits}


def src_openalex(term: str) -> Dict[str, Any]:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"search": f'"{term}"', "per_page": 25,
                                                                         "mailto": "research-audit@example.org"})
    code, body = http_get(url)
    if code != 200:
        return {"status": f"http_{code}", "url": url}
    d = json.loads(body)
    hits = []
    for w in d.get("results", []):
        title = w.get("display_name") or ""
        hits.append(_hit(title, w.get("doi") or w.get("id") or "", str(w.get("publication_year")), kind="paper",
                         title_match=match_kind(term, title)))
    return {"status": "ok", "url": url, "hit_count": int(d.get("meta", {}).get("count", 0)), "hits": hits}


def src_semantic_scholar(term: str) -> Dict[str, Any]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {"query": term, "limit": 25, "fields": "title,year,url,venue"})
    code, body = http_get(url, retries=6)
    if code != 200:
        return {"status": f"http_{code}", "url": url}
    d = json.loads(body)
    hits = [_hit(p.get("title") or "", p.get("url") or "", f"{p.get('year')} {p.get('venue') or ''}", kind="paper",
                 title_match=match_kind(term, p.get("title") or "")) for p in d.get("data", []) or []]
    return {"status": "ok", "url": url, "hit_count": int(d.get("total", 0)), "hits": hits}


def src_dblp(term: str) -> Dict[str, Any]:
    url = "https://dblp.org/search/publ/api?" + urllib.parse.urlencode({"q": term, "format": "json", "h": 25})
    code, body = http_get(url)
    if code not in (200, 202):
        return {"status": f"http_{code}", "url": url}
    if body.lstrip()[:1] != b"{":
        return {"status": f"blocked: dblp returned an HTML bot challenge (HTTP {code}), no API answer", "url": url}
    d = json.loads(body).get("result", {}).get("hits", {})
    hits = []
    for h in d.get("hit", []) or []:
        info = h.get("info", {})
        title = info.get("title") or ""
        hits.append(_hit(title, info.get("ee") or info.get("url") or "", f"{info.get('year')} {info.get('venue') or ''}",
                         kind="paper", title_match=match_kind(term, title)))
    return {"status": "ok", "url": url, "hit_count": int(d.get("@total", 0)), "hits": hits}


def src_crossref(term: str) -> Dict[str, Any]:
    """Crossref ranks fuzzily, so its total is not meaningful; we keep the top 20 and count title matches only."""
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": term, "rows": 20,
                                                                         "select": "DOI,title,issued"})
    code, body = http_get(url)
    if code != 200:
        return {"status": f"http_{code}", "url": url}
    items = json.loads(body).get("message", {}).get("items", [])
    hits = []
    for it in items:
        title = " ".join(it.get("title") or [])
        hits.append(_hit(title, f"https://doi.org/{it.get('DOI')}", "", kind="paper", title_match=match_kind(term, title)))
    hc = sum(1 for h in hits if h["title_match"])
    return {"status": "ok", "url": url, "hit_count": hc, "hits": hits,
            "note": "hit_count = top-20 relevance results whose title contains the term (Crossref totals are fuzzy)"}


_ACL_CACHE: Dict[str, Any] = {}


def _acl_entries(bib_path: Optional[Path], cache_dir: Path) -> Tuple[List[Tuple[str, str, str]], str]:
    if "entries" in _ACL_CACHE:
        return _ACL_CACHE["entries"], _ACL_CACHE["src"]
    path = bib_path
    if path is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "anthology.bib.gz"
        if not path.exists():
            req = urllib.request.Request(ACL_BIB_URL, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as fh:  # noqa: S310
                shutil.copyfileobj(resp, fh)
    opener = gzip.open if str(path).endswith(".gz") else open
    entries: List[Tuple[str, str, str]] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        cur_title, cur_url, cur_key = "", "", ""
        for line in fh:
            s = line.strip()
            if s.startswith("@"):
                if cur_key:
                    entries.append((cur_key, cur_title, cur_url))
                cur_key, cur_title, cur_url = s.split("{", 1)[-1].rstrip(","), "", ""
            elif s.startswith("title"):
                cur_title = s.split("=", 1)[-1].strip().strip(",").strip('"{}')
            elif s.startswith("url"):
                cur_url = s.split("=", 1)[-1].strip().strip(",").strip('"{}')
        if cur_key:
            entries.append((cur_key, cur_title, cur_url))
    _ACL_CACHE["entries"], _ACL_CACHE["src"] = entries, str(path)
    return entries, str(path)


def src_acl(term: str, bib_path: Optional[Path], cache_dir: Path) -> Dict[str, Any]:
    try:
        entries, src = _acl_entries(bib_path, cache_dir)
    except Exception as e:  # noqa: BLE001
        return {"status": f"error: {e}"[:200], "url": ACL_BIB_URL}
    hits = []
    for key, title, url in entries:
        mk = match_kind(term, title)
        if mk:
            hits.append(_hit(title, url or f"https://aclanthology.org/{key}", key, kind="paper", title_match=mk))
    return {"status": "ok", "url": ACL_BIB_URL, "hit_count": len(hits), "hits": hits[:50],
            "note": f"title search over {len(entries)} ACL Anthology bibliography entries ({Path(src).name})"}


def src_github(term: str) -> Dict[str, Any]:
    if not shutil.which("gh"):
        return {"status": "skipped: gh CLI not available"}
    q = f'"{term}"' if " " in term else term
    endpoint = "search/repositories?" + urllib.parse.urlencode({"q": f"{q} in:name,description,readme", "per_page": 30})
    try:
        out = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return {"status": f"error: {e}"[:200]}
    if out.returncode != 0:
        return {"status": f"error: {out.stderr.strip()[:200]}"}
    d = json.loads(out.stdout)
    hits = []
    for r in d.get("items", []):
        desc = r.get("description") or ""
        hits.append(_hit(r.get("full_name", ""), r.get("html_url", ""), desc, kind="repo", stars=r.get("stargazers_count", 0),
                         title_match=match_kind(term, r.get("name", "")), desc_match=match_kind(term, desc),
                         domain=in_domain(desc + " " + r.get("full_name", "")), updated=r.get("updated_at")))
    time.sleep(2.5)  # search API: 30 req/min authenticated
    return {"status": "ok", "url": "https://github.com/search?type=repositories&q=" + urllib.parse.quote(q),
            "hit_count": int(d.get("total_count", 0)), "hits": hits}


def src_pypi(term: str) -> Dict[str, Any]:
    name = re.sub(r"[-_. ]+", "-", term).lower()  # PEP 503 normalisation
    url = f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json"
    code, body = http_get(url, retries=2)
    if code == 404:
        return {"status": "ok", "url": f"https://pypi.org/project/{name}/", "hit_count": 0, "hits": [],
                "note": f"project name {name!r} not registered"}
    if code != 200:
        return {"status": f"http_{code}", "url": url}
    info = json.loads(body).get("info", {})
    summ = info.get("summary") or ""
    return {"status": "ok", "url": f"https://pypi.org/project/{name}/", "hit_count": 1,
            "hits": [_hit(info.get("name", name), info.get("project_url") or f"https://pypi.org/project/{name}/", summ,
                          kind="package", title_match="exact", domain=in_domain(summ + " " + (info.get("keywords") or "")),
                          version=info.get("version"))],
            "note": f"project name {name!r} is registered"}


def src_npm(term: str) -> Dict[str, Any]:
    name = re.sub(r"\s+", "-", term).lower()
    code, body = http_get(f"https://registry.npmjs.org/{urllib.parse.quote(name)}", retries=2)
    exact = []
    if code == 200:
        d = json.loads(body)
        desc = d.get("description") or ""
        exact.append(_hit(name, f"https://www.npmjs.com/package/{name}", desc, kind="package", title_match="exact",
                          domain=in_domain(desc)))
    surl = "https://registry.npmjs.org/-/v1/search?" + urllib.parse.urlencode({"text": term, "size": 20})
    c2, b2 = http_get(surl, retries=2)
    total, others = 0, []
    if c2 == 200:
        d2 = json.loads(b2)
        total = int(d2.get("total", 0))
        for o in d2.get("objects", []):
            p = o.get("package", {})
            mk = match_kind(term, p.get("name", ""))
            if mk and p.get("name", "").lower() != name:
                others.append(_hit(p.get("name", ""), p.get("links", {}).get("npm", ""), p.get("description") or "",
                                   kind="package", title_match=mk, domain=in_domain(p.get("description") or "")))
    return {"status": "ok", "url": f"https://www.npmjs.com/package/{name}", "hit_count": len(exact) + len(others),
            "hits": exact + others, "note": f"exact name {'registered' if exact else 'free'}; npm full-text total {total}"}


def src_hf(term: str, kind: str) -> Dict[str, Any]:
    q = norm(term) if kind != "datasets" else term.replace(" ", "")
    url = f"https://huggingface.co/api/{kind}?" + urllib.parse.urlencode({"search": q, "limit": 200})
    code, body = http_get(url)
    if code != 200:
        return {"status": f"http_{code}", "url": url}
    items = json.loads(body)
    hits = []
    for it in items:
        rid = it.get("id") or it.get("modelId") or ""
        hits.append(_hit(rid, f"https://huggingface.co/{'' if kind == 'models' else kind + '/'}{rid}", "",
                         kind=kind[:-1], title_match=match_kind(term, rid.split("/")[-1]), likes=it.get("likes", 0),
                         domain=in_domain(rid)))
    return {"status": "ok", "url": f"https://huggingface.co/{kind}?search={urllib.parse.quote(q)}",
            "hit_count": len(items), "hits": hits}


SOURCES = {
    "arxiv": src_arxiv, "openalex": src_openalex, "semantic_scholar": src_semantic_scholar, "dblp": src_dblp,
    "crossref": src_crossref, "github": src_github, "pypi": src_pypi, "npm": src_npm,
    "hf_models": lambda t: src_hf(t, "models"), "hf_datasets": lambda t: src_hf(t, "datasets"),
    "hf_spaces": lambda t: src_hf(t, "spaces"),
}
SCHOLARLY = {"arxiv", "openalex", "semantic_scholar", "dblp", "crossref", "acl_anthology"}
REGISTRY = {"pypi", "npm"}
HUBS = {"github", "hf_models", "hf_datasets", "hf_spaces"}


def screen(source: str, res: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
    """Return (risk, relevant_hits, rationale) under the fixed screening rule in the module docstring."""
    if res.get("status") != "ok":
        return "unknown", [], f"query failed: {res.get('status')}"
    hits = res.get("hits", [])
    exact = [h for h in hits if h.get("title_match") == "exact" or h.get("desc_match") == "exact"]
    sub = [h for h in hits if h not in exact and (h.get("title_match") or h.get("desc_match") or h.get("abstract_match"))]
    if source in SCHOLARLY:
        title_exact = [h for h in hits if h.get("title_match") == "exact"]
        if title_exact:
            return "high", title_exact, f"{len(title_exact)} paper title(s) use the exact name"
        abstract_exact = [h for h in hits if h.get("abstract_match") == "exact"]
        if abstract_exact:
            return "medium", abstract_exact, f"{len(abstract_exact)} abstract(s) use the exact name"
        subs = [h for h in hits if h.get("title_match") == "substring"]
        return ("low", subs, f"{len(subs)} title(s) contain it as a substring") if subs else ("none", [], "no matching title")
    if source in REGISTRY:
        if exact:
            dom = any(h.get("domain") for h in exact)
            return ("high" if dom else "medium"), exact, "package name registered" + (" in the same domain" if dom else "")
        return "none", [], "package name free"
    if source in HUBS:
        if exact:
            strong = [h for h in exact if h.get("domain") or (h.get("stars") or 0) >= 50 or (h.get("likes") or 0) >= 50]
            if strong:
                return "high", strong, f"{len(strong)} same-name item(s) in the domain or with >= 50 stars/likes"
            return "medium", exact, f"{len(exact)} same-name item(s), unrelated domain, < 50 stars/likes"
        return ("low", sub[:10], f"{len(sub)} partial-name item(s)") if sub else ("none", [], "no same-name item")
    return "unknown", [], "unscreened source"


def run(terms: List[str], manual: Optional[Path], acl_bib: Optional[Path], cache_dir: Path,
        include_title: bool = True) -> Dict[str, Any]:
    date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for term in terms + ([TITLE] if include_title else []):
        is_title = term == TITLE
        sources = ["arxiv", "openalex", "semantic_scholar", "dblp", "crossref", "acl_anthology"] if is_title else \
            list(SOURCES) + ["acl_anthology"]
        for source in sources:
            try:
                if source == "acl_anthology":
                    res = src_acl(term, acl_bib, cache_dir)
                else:
                    res = SOURCES[source](term)
            except Exception as e:  # noqa: BLE001 - one broken source must not stop the audit
                res = {"status": f"error: {type(e).__name__}: {e}"[:200]}
            if is_title:
                # whole-title check: a paper with (nearly) the same title is a collision; partial word overlap is not
                for h in res.get("hits", []) or []:
                    h["title_match"] = "exact" if norm(h.get("title", "")) == norm(TITLE) else (
                        "substring" if "skillgraphlm" in norm(h.get("title", "")) else "")
                    h.pop("abstract_match", None)
            risk, relevant, why = screen(source, res)
            records.append({"date_utc": date, "term": term, "source": source, "query_url": res.get("url", ""),
                            "status": res.get("status"), "hit_count": res.get("hit_count"),
                            "relevant_hits": relevant[:15], "screening_risk": risk, "rationale": why,
                            "note": res.get("note", "")})
            if source == "semantic_scholar":
                time.sleep(4)
    manual_records = json.loads(manual.read_text()) if manual and manual.exists() else []
    return {"meta": {"date_utc": date, "terms": terms, "title": TITLE, "screening_rule": __doc__.split("Screening rule")[1],
                     "manual_source": str(manual) if manual else None},
            "records": records, "manual_records": manual_records}


def summarise(audit: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in audit["records"] + audit.get("manual_records", []):
        t = r["term"]
        s = out.setdefault(t, {"max_risk": "none", "by_source": {}, "unknown_sources": []})
        risk = r.get("final_risk") or r.get("screening_risk", "unknown")
        if risk == "unknown":
            s["unknown_sources"].append(r["source"])
        s["by_source"][r["source"]] = risk
        if RANK[risk] > RANK[s["max_risk"]]:
            s["max_risk"] = risk
    return out


def finalize(out: Path) -> Dict[str, Any]:
    """Merge automated evidence (naming_audit_raw.json), manual web/trademark queries (manual_queries.json), the lighter
    audit of alternative names (alternatives/naming_audit_raw.json) and the human adjudication (assessment.json) into
    naming_audit.json.  The adjudicated risk may be higher than the screening risk, never silently lower: a lower final
    risk needs an explicit 'override_reason'."""
    raw = json.loads((out / "naming_audit_raw.json").read_text())
    manual = json.loads((out / "manual_queries.json").read_text()) if (out / "manual_queries.json").exists() else []
    alt_path = out / "alternatives" / "naming_audit_raw.json"
    alt = json.loads(alt_path.read_text()) if alt_path.exists() else None
    alt_manual_path = out / "alternatives" / "manual_queries.json"
    if alt is not None:
        alt["manual_records"] = json.loads(alt_manual_path.read_text()) if alt_manual_path.exists() else []
    assessment = json.loads((out / "assessment.json").read_text())
    raw["manual_records"] = manual
    screening = summarise(raw)
    terms = {}
    for t, a in assessment["terms"].items():
        scr = screening.get(t, {}).get("max_risk", "unknown")
        fin = a["final_risk"]
        if RANK.get(fin, -1) < RANK.get(scr, -1) and not a.get("override_reason"):
            raise ValueError(f"{t}: final risk {fin} below screening risk {scr} without override_reason")
        terms[t] = {"screening_max_risk": scr, "screening_by_source": screening.get(t, {}).get("by_source", {}),
                    "unreachable_sources": screening.get(t, {}).get("unknown_sources", []), **a}
    alts = {}
    if alt:
        alt_scr = summarise(alt)
        for t, a in assessment.get("alternatives", {}).items():
            alts[t] = {"screening_max_risk": alt_scr.get(t, {}).get("max_risk", "unknown"),
                       "screening_by_source": alt_scr.get(t, {}).get("by_source", {}), **a}
    final = {"meta": {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "automated_run_utc": raw["meta"]["date_utc"], "title": TITLE,
                      "screening_rule": raw["meta"]["screening_rule"], "decision_rule": assessment.get("decision_rule"),
                      "sources_unreachable": assessment.get("sources_unreachable", [])},
             "verdict": assessment.get("verdict"), "terms": terms, "alternatives": alts,
             "records": raw["records"], "manual_records": manual,
             "alternative_records": (alt or {}).get("records", [])}
    (out / "naming_audit.json").write_text(json.dumps(final, indent=2) + "\n")
    return final

def render_report(out: Path) -> str:
    """REPORT.md from naming_audit.json (written by ``--finalize``): verdict, per-term risk by source, query records, alternatives."""
    f = json.loads((out / "naming_audit.json").read_text())
    v = f.get("verdict") or {}
    L = ["# Naming audit: \"SkillGraphLM\" and variants", "",
         f"Generated {f['meta']['generated_utc']} (automated API run {f['meta']['automated_run_utc']}; manual web searches "
         "dated per record). **Nothing in the repository has been renamed**; the decision is the author's. All web content is "
         "untrusted data: only titles, names and URLs are recorded.", "",
         f"## Verdict: **{v.get('recommendation', 'n/a')}**", "", v.get("summary", ""), ""]
    if v.get("caveats"):
        L += ["Caveats and required actions:", ""] + [f"- {c}" for c in v["caveats"]] + [""]
    L += ["Decision rule (fixed before adjudication): " + str(f["meta"].get("decision_rule")), "",
          "## Risk by term", "", "| term | screening max | final (adjudicated) | why |", "|---|---|---|---|"]
    for t, a in f["terms"].items():
        L.append(f"| {t} | {a['screening_max_risk']} | **{a['final_risk']}** | {a.get('why', '')} |")
    srcs = sorted({s for a in f["terms"].values() for s in a["screening_by_source"]})
    L += ["", "## Screening risk per source (automated API queries and manual web searches)", "",
          "| term | " + " | ".join(srcs) + " |", "|---|" + "---|" * len(srcs)]
    for t, a in f["terms"].items():
        L.append(f"| {t[:40]} | " + " | ".join(a["screening_by_source"].get(s, "") for s in srcs) + " |")
    if f["meta"].get("sources_unreachable"):
        L += ["", "Sources that could not be queried: " + "; ".join(f["meta"]["sources_unreachable"]) + "."]
    L += ["", "## Manual web-search records (one per query)", "",
          "| date (UTC) | by | term | source | query | hits | screening | final | collision evidence |", "|---|---|---|---|---|---|---|---|---|"]
    for r in f["manual_records"]:
        urls = "; ".join(f"[{h['title'][:60]}]({h['url']})" for h in r.get("relevant_hits", [])[:4])
        L.append(f"| {r['date_utc'][:16]} | {r.get('recorded_by', '')} | {r['term'][:30]} | {r['source']} | {r['query'][:70]} | "
                 f"{r.get('hit_count')} | {r.get('screening_risk')} | {r.get('final_risk', '')} | {r['rationale'][:300]} {urls} |")
    L += ["", "## Automated API/registry records with relevant hits (screening rule in the module docstring)", ""]
    for r in f["records"]:
        if r["screening_risk"] in ("high", "medium", "unknown"):
            hits = "; ".join(f"[{h['title'][:50]}]({h['url']})" for h in r["relevant_hits"][:4])
            L.append(f"- `{r['term'][:30]}` / {r['source']}: {r['screening_risk']} (hits {r['hit_count']}; {r['rationale'][:120]}) {hits}")
    if f.get("alternatives"):
        L += ["", "## Alternative names (lighter audit: same automated sources plus manual web search)", "",
              "| alternative | screening max | final | note |", "|---|---|---|---|"]
        for t, a in f["alternatives"].items():
            L.append(f"| {t} | {a['screening_max_risk']} | **{a['final_risk']}** | {a.get('why', '')} |")
        L += ["", "Alternatives are only candidates; none is endorsed over keeping the current name unless the author wants a "
              "name with no near-duplicates in the skill-graph field. Trademark clearance is not covered by any of this."]
    L += ["", "## Limits", "", "- Trademark registries (USPTO, EUIPO, WIPO) are JS applications and could not be queried; the only trademark "
          "record is an unverified aggregator listing. Run a formal search before commercial use.",
          "- Google Scholar and DBLP are bot-protected from this host; OpenAlex, Semantic Scholar (when not rate-limited), Crossref, "
          "arXiv and the ACL Anthology bibliography were used instead.",
          "- Search engines are US-only and index only part of the web; absence of a hit is evidence, not proof."]
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--manual", type=Path, default=OUT_DEFAULT / "manual_queries.json")
    ap.add_argument("--acl-bib", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=Path("/usr/data/models/skillllm-v2-stage2/a6_scratch/naming_cache"))
    ap.add_argument("--terms", nargs="*", default=TERMS)
    ap.add_argument("--no-title", action="store_true", help="do not query the proposed thesis title")
    ap.add_argument("--retry-unknown", action="store_true",
                    help="re-query only records whose screening risk is 'unknown' in an existing naming_audit_raw.json")
    ap.add_argument("--retry-wait", type=float, default=45.0, help="seconds between retried queries (rate limits)")
    ap.add_argument("--finalize", action="store_true", help="merge evidence + assessment.json into naming_audit.json and write REPORT.md")
    a = ap.parse_args(argv)
    if a.finalize:
        f = finalize(a.out)
        (a.out / "REPORT.md").write_text(render_report(a.out))
        for t, v in f["terms"].items():
            print(f"{t:<40} screening={v['screening_max_risk']:<7} final={v['final_risk']}")
        return 0
    if a.retry_unknown:
        raw = a.out / "naming_audit_raw.json"
        audit = json.loads(raw.read_text())
        for r in audit["records"]:
            if r["screening_risk"] != "unknown" or r["source"] not in SOURCES:
                continue
            time.sleep(a.retry_wait)
            try:
                res = SOURCES[r["source"]](r["term"])
            except Exception as e:  # noqa: BLE001
                res = {"status": f"error: {type(e).__name__}: {e}"[:200]}
            if r["term"] == TITLE:
                for h in res.get("hits", []) or []:
                    h["title_match"] = "exact" if norm(h.get("title", "")) == norm(TITLE) else (
                        "substring" if "skillgraphlm" in norm(h.get("title", "")) else "")
                    h.pop("abstract_match", None)
            risk, relevant, why = screen(r["source"], res)
            r.update({"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "query_url": res.get("url", r["query_url"]),
                      "status": res.get("status"), "hit_count": res.get("hit_count"), "relevant_hits": relevant[:15],
                      "screening_risk": risk, "rationale": why + " (retry)", "note": res.get("note", "")})
        audit["manual_records"] = json.loads(a.manual.read_text()) if a.manual and a.manual.exists() else []
        audit["summary"] = summarise(audit)
        raw.write_text(json.dumps(audit, indent=2) + "\n")
        for t, s_ in audit["summary"].items():
            print(f"{t:<40} max_screening_risk={s_['max_risk']:<7} {s_['by_source']}")
        return 0
    audit = run(a.terms, a.manual, a.acl_bib, a.cache_dir, include_title=not a.no_title)
    audit["summary"] = summarise(audit)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "naming_audit_raw.json").write_text(json.dumps(audit, indent=2) + "\n")
    for t, s in audit["summary"].items():
        print(f"{t:<40} max_screening_risk={s['max_risk']:<7} {s['by_source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
