"""L05_wrapper — what does each safe-wrapper component add to (or take from) the Raw Mistral LoRA anchor?

Ladder (cumulative, replayed on the STORED anchor predictions in the real pipeline order merge -> refine -> verify ->
calibrate -> link): R0 raw Mistral, R1 +boundary selector/refiner, R2 +GLiNER add-only merge, R3 +dictionary,
R4 +taxonomy link, R5 +verifier, R6 +confidence calibrator (= replayed safe wrapper), R7 the stored, gated safe-wrapper
output. Plus single-component and leave-one-out variants (order-dependence), the stored wrapper variants, the legacy
ungated artefacts (H2 comparators), a same-anchor "policy-off" counterfactual (secondary) and a leakage ablation that
removes the 76 alias-table rows mined from TEST-split anchor predictions.

Per rung vs anchor, per document: TP gained (= recovered FN), TP destroyed, FP added, FP removed, verifier / calibrator
false rejections, merger gold drops, soft-skill recall loss, exact / token F1; Anchor Preservation Rate and Net Wrapper
Gain (definitions: Decisions.md D-A2-02). Document-cluster bootstrap CIs; paired bootstrap + approximate randomization
for deltas. H2 verdict.

    python -m skillllm_v2.evals.research.wrapper_contribution [--smoke] [--seed 42] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from skillllm_v2.evals.research import stats as S
from skillllm_v2.evals.research.common import EVALS_ROOT, REPO_ROOT, md_table, record, rel, write_layer

LAYER = "L05_wrapper"
DATASET = "SkillSpan test (HF jjzha/skillspan), 974 docs with >=1 gold span (2,265 gold spans)"
SPLIT = "test"
P = {
    "gold": "data/source/skillspan_hf/test.json",
    "train": "data/source/skillspan_hf/train.json",
    "dev": "data/source/skillspan_hf/dev.json",
    "anchor": "artifacts/skillllm_v2/mistral_lora/predictions.jsonl",
    "anchor_metrics": "artifacts/skillllm_v2/mistral_lora/metrics.json",
    "gliner": "artifacts/skillllm_v2/gliner_baseline/predictions.jsonl",
    "alias_table": "data/skillllm_v2/taxonomy/skill_aliases.jsonl",
    "merger_cfg": "artifacts/skillllm_v2/candidate_merger/tuned_config.yaml",
    "boundary_cfg": "artifacts/skillllm_v2/boundary_refiner/tuned_config.yaml",
    "calibrator_cfg": "artifacts/skillllm_v2/confidence_calibrator/tuned_config.yaml",
    "dev_preds": "artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl",
    "safe_final": "artifacts/skillllm_v2/safe_wrapper/predictions.jsonl",
    "safe_br": "artifacts/skillllm_v2/safe_wrapper/boundary_refiner/candidate_predictions.jsonl",
    "safe_mgt": "artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/candidate_predictions.jsonl",
    "safe_full": "artifacts/skillllm_v2/safe_wrapper/full_cag_graph/candidate_predictions.jsonl",
    "gate": "artifacts/skillllm_v2/safe_wrapper/gate_results.json",
    "legacy_full": "artifacts/skillllm_v2/full_cag_graph/predictions.jsonl",
    "legacy_mgt": "artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl",
    "legacy_boundary": "artifacts/skillllm_v2/mistral_boundary/predictions.jsonl",
    "prefix_analysis": "reports/skillllm_v2/wrapper_error_analysis.json",
    "existing_sig": "reports/skillllm_v2/significance_f1.json",
}
CODE = ["src/skillllm_v2/extract/candidate_merger.py", "src/skillllm_v2/verify/boundary_refiner.py",
        "src/skillllm_v2/verify/boundary_selector.py", "src/skillllm_v2/verify/span_verifier.py",
        "src/skillllm_v2/verify/confidence_calibrator.py", "src/skillllm_v2/extract/dictionary_extractor.py",
        "src/skillllm_v2/normalize/taxonomy_linker.py", "src/skillllm_v2/policy/source_aware_policy.py"]
DERIVED_DIR = REPO_ROOT / "inputs" / "evals" / "research" / LAYER
NI_MARGIN = -0.005
LABELS = ("HARD_SKILL", "SOFT_SKILL")

# per-document row layout
COLS = ["tp", "fp", "fn", "ttp", "tfp", "tfn", "htp", "hfp", "hfn", "stp", "sfp", "sfn",
        "anchor_tp", "kept_tp", "gained_tp", "destroyed_tp", "anchor_fp", "added_fp", "removed_fp", "n_gold",
        "soft_anchor_tp", "soft_kept_tp", "soft_gold", "n_pred", "changed",
        "verifier_rejected", "verifier_false_rej", "calib_rejected", "calib_false_rej",
        "merger_gold_drop_anchor", "merger_gold_drop_new", "merger_dedupe_anchor_drop"]
C = {c: i for i, c in enumerate(COLS)}


# ----------------------------------------------------------------------------- span helpers
def key(sp: Dict[str, Any]) -> Tuple[Any, Any, str]:
    return (sp.get("start"), sp.get("end"), sp.get("label"))


def sources(sp: Dict[str, Any]) -> Set[str]:
    s = (sp.get("metadata") or {}).get("sources")
    return set(s) if s else {sp.get("source", "?")}


def read_preds(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row.get("predictions", row.get("linked_skills", []))
    return out


# ----------------------------------------------------------------------------- components
class _NoPolicyMerger:
    """CandidateMerger with the add-only (anchor-protection) filter switched off: group/merge/subsumption only."""

    def __init__(self, base: Any) -> None:
        self.base = base

    def merge(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from skillllm_v2.schema import validate_skill
        b = self.base
        valid = []
        for c in candidates:
            validate_skill(c)
            valid.append(dict(c))
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for c in valid:
            groups[(b._canonical_key(c), c["label"])].append(c)
        merged = [b._merge_group(g) for g in groups.values()]
        merged.sort(key=lambda it: (it["start"] if it["start"] is not None else 10**9, it["end"] or 10**9, -it["confidence"]))
        return b._drop_subsumed(merged)


PREFIX_LIKE_EDITS = ("trim_leading_filler", "trim_trailing_filler", "split_conjunction")


def destructive_selector_config(edits: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Policy-off boundary selector: the given edit types (default: all) are allowed and any applicable edit beats
    'original'. ``edits=PREFIX_LIKE_EDITS`` approximates the pre-fix refiner that replaced spans by filler-trimmed / split
    versions (reports/skillllm_v2/wrapper_error_analysis.md; that code is not in git history, so this is a construct)."""
    from skillllm_v2.verify.boundary_selector import DEFAULT_CONFIG
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    allowed = list(edits) if edits is not None else [k for k in cfg["edit_type_scores"] if k != "original"]
    cfg["edit_type_scores"] = {k: (0.0 if k == "original" else (1.0 if k in allowed else -1.0)) for k in cfg["edit_type_scores"]}
    cfg["allowed_edit_types"] = allowed
    cfg["max_trim_tokens"] = 99
    cfg["label_thresholds"] = {"HARD_SKILL": {"min_margin_to_replace": 0.0}, "SOFT_SKILL": {"min_margin_to_replace": 0.0}}
    cfg["default_min_margin_to_replace"] = 0.0
    return cfg


def build_components(alias_path: Optional[str] = None, policy_off: Sequence[str] = (),
                     edits: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Wrapper components exactly as run_safe_wrapper builds them (default configs + on-disk tuned YAMLs, cwd = repo
    root), optionally with a different alias table or with the source-aware policy switched off per component."""
    from skillllm_v2.extract import CandidateMerger
    from skillllm_v2.normalize import TaxonomyLinker
    from skillllm_v2.verify import BoundaryRefiner, ConfidenceCalibrator, SpanVerifier
    from skillllm_v2.verify.boundary_selector import BoundarySelector
    ap = [alias_path] if alias_path else None
    merger = CandidateMerger(alias_paths=ap)
    comps = {
        "merger": _NoPolicyMerger(merger) if "merger" in policy_off else merger,
        "refiner": BoundaryRefiner(alias_paths=ap),
        "selector": BoundarySelector(config=destructive_selector_config(edits), alias_paths=ap) if "refiner" in policy_off
        else BoundarySelector(alias_paths=ap),
        "verifier": SpanVerifier(protect_anchor_sources=frozenset()) if "verifier" in policy_off else SpanVerifier(),
        "calibrator": ConfidenceCalibrator(protect_anchor_sources=[]) if "calibrator" in policy_off else ConfidenceCalibrator(),
        "linker": TaxonomyLinker(alias_table_paths=ap) if ap else TaxonomyLinker(),
    }
    return comps


def run_chain(text: str, anchor: List[Dict[str, Any]], gl: List[Dict[str, Any]], dic: List[Dict[str, Any]],
              comps: Dict[str, Any], enabled: Set[str], gold_keys: Set[Tuple]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Run the enabled stages in pipeline order; return (final candidates, per-doc stage diagnostics)."""
    diag = Counter()
    cands = list(anchor)
    if "gliner" in enabled or "dictionary" in enabled or "merger" in enabled:
        inp = list(anchor) + (list(gl) if "gliner" in enabled else []) + (list(dic) if "dictionary" in enabled else [])
        out = comps["merger"].merge(inp)
        out_keys = {key(c) for c in out}
        anchor_keys = {key(c) for c in anchor}
        for k in {key(c) for c in inp if key(c) in gold_keys} - out_keys:
            if k in anchor_keys:
                diag["merger_gold_drop_anchor"] += 1
            else:
                diag["merger_gold_drop_new"] += 1
        diag["merger_dedupe_anchor_drop"] += len(anchor_keys - out_keys)
        cands = out
    if "refine" in enabled:
        cands = comps["refiner"].refine(text, cands, selector=comps["selector"])
    if "verify" in enabled:
        v = comps["verifier"]
        kept = []
        for c in cands:
            if v.verify(c)["is_valid_skill"]:
                kept.append(c)
            else:
                diag["verifier_rejected"] += 1
                diag["verifier_false_rej"] += int(key(c) in gold_keys)
        cands = v.filter(cands)
        assert len(cands) == len(kept)
    if "calibrate" in enabled:
        cal = comps["calibrator"]
        for c in cands:
            if float(c["confidence"]) < cal.threshold_for(c):
                diag["calib_rejected"] += 1
                diag["calib_false_rej"] += int(key(c) in gold_keys)
        cands = cal.calibrate(cands)
    if "link" in enabled:
        before = [key(c) for c in cands]
        cands = comps["linker"].link_candidates(cands)
        assert [key(c) for c in cands] == before, "taxonomy linking must not change spans"
    return cands, dict(diag)


# ----------------------------------------------------------------------------- per-document accounting
def doc_row(gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]], anchor: Sequence[Dict[str, Any]],
            text: str, diag: Optional[Dict[str, int]] = None) -> np.ndarray:
    from skillllm_v2.evals import extraction_evals as ev
    base = ev.doc_counts(gold, pred, text)
    r = np.zeros(len(COLS))
    r[C["tp"]], r[C["fp"]], r[C["fn"]] = base[ev.C_TP], base[ev.C_FP], base[ev.C_FN]
    r[C["ttp"]], r[C["tfp"]], r[C["tfn"]] = base[ev.C_TTP], base[ev.C_TFP], base[ev.C_TFN]
    r[C["htp"]], r[C["hfp"]], r[C["hfn"]] = base[ev.C_HTP], base[ev.C_HFP], base[ev.C_HFN]
    r[C["stp"]], r[C["sfp"]], r[C["sfn"]] = base[ev.C_STP], base[ev.C_SFP], base[ev.C_SFN]
    g = {key(x) for x in gold}
    a, s = {key(x) for x in anchor}, {key(x) for x in pred}
    at, af, st, sf = a & g, a - g, s & g, s - g
    r[C["anchor_tp"]], r[C["kept_tp"]] = len(at), len(at & st)
    r[C["gained_tp"]], r[C["destroyed_tp"]] = len(st - at), len(at - st)
    r[C["anchor_fp"]], r[C["added_fp"]], r[C["removed_fp"]] = len(af), len(sf - af), len(af - sf)
    r[C["n_gold"]] = len(g)
    gs = {k for k in g if k[2] == "SOFT_SKILL"}
    r[C["soft_anchor_tp"]], r[C["soft_kept_tp"]], r[C["soft_gold"]] = len(at & gs), len(at & st & gs), len(gs)
    r[C["n_pred"]] = len(s)
    r[C["changed"]] = float(a != s)
    for k, v in (diag or {}).items():
        r[C[k]] = v
    return r


def _cols(*names: str) -> List[int]:
    return [C[n] for n in names]


def stat_f1(rows_sum: np.ndarray, t: str = "tp", f: str = "fp", n: str = "fn") -> np.ndarray:
    return S.micro_f1(rows_sum, C[t], C[f], C[n])


STATS = {
    "exact_span_f1": lambda s: S.micro_f1(s, C["tp"], C["fp"], C["fn"]),
    "exact_span_precision": lambda s: S.micro_precision(s, C["tp"], C["fp"]),
    "exact_span_recall": lambda s: S.micro_recall(s, C["tp"], C["fn"]),
    "token_f1": lambda s: S.micro_f1(s, C["ttp"], C["tfp"], C["tfn"]),
    "hard_skill_f1": lambda s: S.micro_f1(s, C["htp"], C["hfp"], C["hfn"]),
    "soft_skill_f1": lambda s: S.micro_f1(s, C["stp"], C["sfp"], C["sfn"]),
    "soft_skill_recall": lambda s: S.micro_recall(s, C["stp"], C["sfn"]),
    "apr": S.ratio_stat(C["kept_tp"], C["anchor_tp"]),
    "soft_apr": S.ratio_stat(C["soft_kept_tp"], C["soft_anchor_tp"]),
    "nwg_per_1000_gold": lambda s: 1000 * S._div(np.asarray(s)[..., C["gained_tp"]] - np.asarray(s)[..., C["destroyed_tp"]]
                                                 - np.asarray(s)[..., C["added_fp"]] + np.asarray(s)[..., C["removed_fp"]],
                                                 np.asarray(s)[..., C["n_gold"]]),
}
for _m in ("gained_tp", "destroyed_tp", "added_fp", "removed_fp"):
    STATS[f"{_m}_per_1000_gold"] = (lambda m: lambda s: 1000 * S._div(np.asarray(s)[..., C[m]], np.asarray(s)[..., C["n_gold"]]))(_m)


# ----------------------------------------------------------------------------- systems
LADDER = [  # (system id, enabled stages, description)
    ("R0_raw_mistral", set(), "Raw Mistral LoRA anchor (stored predictions)"),
    ("R1_boundary", {"refine"}, "+ boundary selector/refiner"),
    ("R2_gliner_addonly", {"refine", "gliner"}, "+ GLiNER candidates via add-only CandidateMerger"),
    ("R3_dictionary", {"refine", "gliner", "dictionary"}, "+ dictionary (alias) candidates"),
    ("R4_taxonomy", {"refine", "gliner", "dictionary", "link"}, "+ taxonomy linking (span-neutral by construction)"),
    ("R5_verifier", {"refine", "gliner", "dictionary", "link", "verify"}, "+ SpanVerifier"),
    ("R6_calibration", {"refine", "gliner", "dictionary", "link", "verify", "calibrate"},
     "+ ConfidenceCalibrator (= replayed safe wrapper, full chain)"),
]
FULL = LADDER[-1][1]
FULL_NOLINK = FULL - {"link"}  # linking is span-neutral (asserted on R4-R6), so non-ladder variants skip it for speed
SINGLE = [
    ("alone_refiner", {"refine"}, "boundary refiner alone on anchor"),
    ("alone_merger_dedupe", {"merger"}, "CandidateMerger on anchor only (no new sources): grouping/subsumption"),
    ("alone_gliner", {"gliner"}, "merge(anchor + GLiNER)"),
    ("alone_dictionary", {"dictionary"}, "merge(anchor + dictionary)"),
    ("alone_verifier", {"verify"}, "SpanVerifier alone on anchor"),
    ("alone_calibrator", {"calibrate"}, "ConfidenceCalibrator alone on anchor"),
]
LOO = [(f"loo_minus_{c}", FULL_NOLINK - {c}, f"full chain without {c}")
       for c in ("refine", "gliner", "dictionary", "verify", "calibrate")]
POLICY_OFF = [
    ("policyoff_full", FULL_NOLINK, ("merger", "refiner", "verifier", "calibrator"),
     "full chain, source-aware protection OFF everywhere, every boundary edit type forced (upper bound on harm)"),
    ("policyoff_prefix_like", FULL_NOLINK, ("merger", "refiner", "verifier", "calibrator"),
     "protection OFF, only filler-trim / conjunction-split edits forced (approximation of the pre-fix wrapper)"),
    ("policyoff_refiner_only", FULL_NOLINK, ("refiner",), "full chain, only the boundary selector unconstrained"),
    ("policyoff_merger_only", FULL_NOLINK, ("merger",), "full chain, only the merger's add-only filter off"),
    ("policyoff_verifier_only", FULL_NOLINK, ("verifier",), "full chain, only the verifier's anchor protection off"),
    ("policyoff_calibrator_only", FULL_NOLINK, ("calibrator",), "full chain, only the calibrator's anchor protection off"),
]
STORED = [
    ("R7_safe_wrapper_stored", "safe_final", "stored gated safe-wrapper output (final; = full_cag_graph variant, gate 'accept')"),
    ("stored_boundary_refiner", "safe_br", "stored safe-wrapper variant boundary_refiner (refine+verify+calibrate)"),
    ("stored_mistral_gliner_taxonomy", "safe_mgt", "stored safe-wrapper variant mistral_gliner_taxonomy (merge+link)"),
    ("legacy_full_cag_graph", "legacy_full", "LEGACY ungated full pipeline, Mistral regenerated live (H2 comparator)"),
    ("legacy_mistral_gliner_taxonomy", "legacy_mgt", "LEGACY ungated merge+link stage, Mistral regenerated live (H2 comparator)"),
    ("legacy_mistral_boundary", "legacy_boundary", "LEGACY boundary stage artefact (predates realign_word_boundary)"),
]


# ----------------------------------------------------------------------------- integrity audit
def tuning_audit(root: Path) -> Dict[str, Any]:
    """What each wrapper component was configured / tuned on (rule 2 of the contract)."""
    import yaml
    out: Dict[str, Any] = {}
    exists = {k: (root / P[k]).exists() for k in ("merger_cfg", "boundary_cfg", "calibrator_cfg", "dev_preds")}
    from skillllm_v2.extract.candidate_merger import DEFAULT_CONFIG as MDEF
    from skillllm_v2.train import tune_candidate_merger as tcm
    from skillllm_v2.train import tune_boundary_refiner as tbr
    from skillllm_v2.verify.boundary_selector import DEFAULT_CONFIG as BDEF
    mcfg = yaml.safe_load((root / P["merger_cfg"]).read_text())["candidate_merger"] if exists["merger_cfg"] else None
    bcfg = yaml.safe_load((root / P["boundary_cfg"]).read_text())["boundary_selector"] if exists["boundary_cfg"] else None
    merger_in_grid = bool(mcfg) and mcfg["min_confidence_new_span"]["gliner"] in tcm.GLINER_THRESHOLD_GRID \
        and mcfg["min_confidence_new_span"]["dictionary"] in tcm.DICTIONARY_THRESHOLD_GRID
    grid_sets = [sorted(x) for x in tbr.ALLOWED_EDIT_TYPE_GRID]
    boundary_in_grid = bool(bcfg) and sorted(bcfg["allowed_edit_types"]) in grid_sets
    boundary_eq_default = bool(bcfg) and json.dumps(bcfg, sort_keys=True) == json.dumps(BDEF, sort_keys=True)
    aliases = [json.loads(l) for l in (root / P["alias_table"]).read_text().splitlines() if l.strip()]
    src_counts = Counter(a.get("source") for a in aliases)
    # default dictionary strings: do they occur as gold spans in train/dev or only in test?
    from skillllm_v2.extract.dictionary_extractor import _default_aliases
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
    surf = {}
    for split in ("train", "dev", "test"):
        docs = load_skillspan_split(root / "data/source/skillspan_hf", split)
        surf[split] = Counter(" ".join(g["surface_form"].lower().split()) for d in docs for g in d["gold_spans"])
    dict_terms = {}
    for term in _default_aliases():
        t = term.lower()
        dict_terms[term] = {s: int(surf[s].get(t, 0)) for s in ("train", "dev", "test")}
    test_only = sorted(t for t, c in dict_terms.items() if c["test"] > 0 and c["train"] == 0 and c["dev"] == 0)
    out["components"] = [
        {"component": "Raw Mistral LoRA anchor", "configured_on": "SkillSpan train (SFT, 80 epochs); final checkpoint",
         "test_contact": "none found", "flag": False},
        {"component": "CandidateMerger thresholds", "configured_on":
            f"hand-set YAML {P['merger_cfg']} (gliner {mcfg and mcfg['min_confidence_new_span']['gliner']}, default "
            f"{mcfg and mcfg['default_min_confidence_new_span']}; library default gliner {MDEF['min_confidence_new_span']['gliner']}); "
            f"values inside tuner grid: {merger_in_grid}; dev predictions file present: {exists['dev_preds']}",
         "test_contact": "not tuner output; no dev tuning evidence on disk", "flag": True},
        {"component": "BoundarySelector config", "configured_on":
            f"YAML identical to DEFAULT_CONFIG: {boundary_eq_default}; allowed_edit_types in tuner grid: {boundary_in_grid}",
         "test_contact": "realign_word_boundary / anchor protection were introduced after the TEST-split error analysis "
                         "(reports/skillllm_v2/wrapper_error_analysis.md, split=test)", "flag": True},
        {"component": "SpanVerifier", "configured_on": "rule list GENERIC_REJECTS + min_confidence 0.35 (no fitting); the "
                     "trained TF-IDF/LogReg verifier is never loaded", "test_contact": "anchor protection designed after test error analysis",
         "flag": True},
        {"component": "ConfidenceCalibrator", "configured_on": f"library defaults (tuned YAML present: {exists['calibrator_cfg']})",
         "test_contact": "none (defaults)", "flag": False},
        {"component": "Alias table (merger evidence gate, boundary alias bonus, taxonomy_normalized, linker)",
         "configured_on": dict(src_counts), "test_contact":
            f"{src_counts.get('mistral_high_confidence', 0)} rows mined from TEST-split anchor predictions", "flag": True},
        {"component": "DictionaryExtractor default list", "configured_on": f"{len(dict_terms)} hard-coded strings",
         "test_contact": f"strings that occur as gold spans only in TEST (never train/dev): {test_only}", "flag": bool(test_only)},
        {"component": "Non-regression gate", "configured_on": "evaluated on the 974-doc TEST subset; decides which output ships",
         "test_contact": "selection on test (the final system choice is test-informed)", "flag": True},
        {"component": "GLiNER (zero-shot urchade/gliner_medium-v2.1, thr 0.35)", "configured_on": "library default",
         "test_contact": "none", "flag": False},
    ]
    out["dictionary_terms_gold_counts"] = dict_terms
    out["alias_source_counts"] = dict(src_counts)
    out["files_present"] = exists
    return out


def write_filtered_alias_table(root: Path) -> Path:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    out = DERIVED_DIR / "skill_aliases_without_test_mined.jsonl"
    rows = [l for l in (root / P["alias_table"]).read_text().splitlines() if l.strip()]
    keep = [l for l in rows if json.loads(l).get("source") != "mistral_high_confidence"]
    out.write_text("\n".join(keep) + "\n")
    return out


# ----------------------------------------------------------------------------- main computation
def compute(root: Path, smoke: bool, seed: int, n_boot: int, n_boot_head: int, log=print) -> Dict[str, Any]:
    from skillllm_v2.extract import DictionaryExtractor
    from skillllm_v2.evals import extraction_evals as ev
    from skillllm_v2.evals.contract import EvalContext
    t0 = time.time()
    gold_all = ev.load_gold(EvalContext(root=root))
    anchor = read_preds(root / P["anchor"])
    ids = [i for i in anchor if i in gold_all]
    if smoke:
        ids = ids[:60]
    gl = read_preds(root / P["gliner"])
    dic_ex = DictionaryExtractor()
    texts = {i: gold_all[i]["text"] for i in ids}
    gold = {i: gold_all[i]["gold_spans"] for i in ids}
    gkeys = {i: {key(x) for x in gold[i]} for i in ids}
    dic = {i: dic_ex.extract(texts[i]) for i in ids}
    log(f"loaded {len(ids)} docs in {time.time() - t0:.1f}s")

    systems: Dict[str, Dict[str, Any]] = {}
    preds: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def add(name: str, desc: str, kind: str, pr: Dict[str, List[Dict[str, Any]]], diags: Optional[Dict[str, Dict[str, int]]] = None):
        rows = np.array([doc_row(gold[i], pr.get(i, []), anchor[i], texts[i], (diags or {}).get(i)) for i in ids])
        n_sp = sum(len(pr.get(i, [])) for i in ids)
        n_linked = sum(1 for i in ids for sp in pr.get(i, []) if sp.get("canonical_skill_name"))
        systems[name] = {"desc": desc, "kind": kind, "rows": rows,
                         "linking_coverage": (n_linked / n_sp) if n_sp and n_linked else None}
        preds[name] = pr

    def replay(enabled: Set[str], comps: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, int]]]:
        pr, dg = {}, {}
        for i in ids:
            pr[i], dg[i] = run_chain(texts[i], anchor[i], gl.get(i, []), dic[i], comps, enabled, gkeys[i])
        return pr, dg

    safe = build_components()
    for name, enabled, desc in LADDER:
        pr, dg = replay(enabled, safe) if enabled else ({i: anchor[i] for i in ids}, {})
        add(name, desc, "ladder", pr, dg)
        log(f"{name} done {time.time() - t0:.1f}s")
    for name, enabled, desc in SINGLE + LOO:
        pr, dg = replay(enabled, safe)
        add(name, desc, "single" if name.startswith("alone") else "leave_one_out", pr, dg)
    for name, enabled, off, desc in POLICY_OFF:
        pr, dg = replay(enabled, build_components(policy_off=off, edits=PREFIX_LIKE_EDITS if name == "policyoff_prefix_like" else None))
        add(name, desc, "policy_off_counterfactual", pr, dg)
    log(f"policy-off done {time.time() - t0:.1f}s")
    filt = write_filtered_alias_table(root)
    pr, dg = replay(FULL_NOLINK, build_components(alias_path=str(filt)))
    add("leakage_no_test_mined_aliases", "full safe chain with the 76 test-mined alias rows removed", "leakage_ablation", pr, dg)
    for name, k, desc in STORED:
        sp = read_preds(root / P[k])
        add(name, desc, "stored" if not name.startswith("legacy") else "legacy", {i: sp.get(i, []) for i in ids})
    log(f"all systems {time.time() - t0:.1f}s")

    # replay vs stored reproduction
    def same_docs(a: str, b: str) -> Tuple[int, List[str]]:
        diff = [i for i in ids if sorted(map(str, (key(x) for x in preds[a][i]))) != sorted(map(str, (key(x) for x in preds[b][i])))]
        return len(ids) - len(diff), diff
    repro = {}
    n_same, diff = same_docs("R6_calibration", "R7_safe_wrapper_stored")
    repro["R6_calibration_vs_R7_safe_wrapper_stored"] = {"docs_identical": n_same, "n_docs": len(ids), "differing_doc_ids": diff}
    # stored boundary_refiner = refine+verify+calibrate on anchor
    br_pr, _ = replay({"refine", "verify", "calibrate"}, safe)
    preds["_replay_br"] = br_pr
    n_same, diff = same_docs("_replay_br", "stored_boundary_refiner")
    repro["replay_refine_verify_calibrate_vs_stored_boundary_refiner"] = {"docs_identical": n_same, "n_docs": len(ids), "differing_doc_ids": diff}
    mg_pr, _ = replay({"gliner", "dictionary", "link"}, safe)
    preds["_replay_mgt"] = mg_pr
    n_same, diff = same_docs("_replay_mgt", "stored_mistral_gliner_taxonomy")
    repro["replay_merge_link_vs_stored_mistral_gliner_taxonomy"] = {"docs_identical": n_same, "n_docs": len(ids), "differing_doc_ids": diff}
    return {"ids": ids, "systems": systems, "preds": preds, "repro": repro, "gold": gold, "texts": texts, "anchor": anchor,
            "audit": tuning_audit(root), "filtered_alias": filt, "elapsed": time.time() - t0}


def edit_breakdown(res: Dict[str, Any], system: str) -> Dict[str, Dict[str, int]]:
    """Boundary edits applied in a system, by edit type: how many turned a non-gold span into gold and vice versa."""
    out: Dict[str, Counter] = defaultdict(Counter)
    for i in res["ids"]:
        g = {key(x) for x in res["gold"][i]}
        for sp in res["preds"][system][i]:
            be = (sp.get("metadata") or {}).get("boundary_edit")
            if not be or be.get("edit_type") == "original":
                continue
            et = be["edit_type"]
            orig = (be["original_start"], be["original_end"], sp["label"])
            out[et]["n"] += 1
            out[et]["orig_was_gold"] += int(orig in g)
            out[et]["new_is_gold"] += int(key(sp) in g)
    return {k: dict(v) for k, v in out.items()}


def source_breakdown(res: Dict[str, Any], system: str) -> Dict[str, Dict[str, int]]:
    """Spans in a system that the anchor did not predict, by source combination: gold vs not."""
    out: Dict[str, Counter] = defaultdict(Counter)
    for i in res["ids"]:
        g = {key(x) for x in res["gold"][i]}
        a = {key(x) for x in res["anchor"][i]}
        for sp in res["preds"][system][i]:
            if key(sp) in a:
                continue
            sk = "+".join(sorted(sources(sp)))
            be = (sp.get("metadata") or {}).get("boundary_edit") or {}
            origin = "boundary_edit_of_anchor" if be.get("edit_type", "original") != "original" and "mistral" in sources(sp) else sk
            out[origin]["n"] += 1
            out[origin]["gold"] += int(key(sp) in g)
    return {k: dict(v) for k, v in out.items()}


def analyse(res: Dict[str, Any], seed: int, n_boot: int, n_boot_head: int, smoke: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = REPO_ROOT
    ids = res["ids"]
    n = len(ids)
    sysd = res["systems"]
    A = sysd["R0_raw_mistral"]["rows"]
    prov_common = [P["gold"], P["anchor"]]
    prov_by_kind = {
        "ladder": prov_common + [P["gliner"], P["alias_table"], P["merger_cfg"], P["boundary_cfg"]] + CODE,
        "single": prov_common + [P["gliner"], P["alias_table"], P["merger_cfg"], P["boundary_cfg"]] + CODE,
        "leave_one_out": prov_common + [P["gliner"], P["alias_table"], P["merger_cfg"], P["boundary_cfg"]] + CODE,
        "policy_off_counterfactual": prov_common + [P["gliner"], P["alias_table"]] + CODE,
        "leakage_ablation": prov_common + [P["gliner"], rel(res["filtered_alias"])] + CODE,
    }
    base_cav = ["Single training run of the anchor and one test split: CIs reflect document sampling only.",
                "SOFT_SKILL = SkillSpan 'skill' component (tags_skill), HARD_SKILL = 'knowledge' (tags_knowledge).",
                "974-doc subset with >=1 gold span: false positives on skill-free sentences are not measured here."]
    replay_cav = ("Replay uses the stored zero-shot GLiNER predictions as the GLiNER source (the wrapper never persisted "
                  "its own GLiNER calls); see replay_reproduction records for the replay-vs-stored mismatch.")
    records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    H = {"seed": seed}

    def rec(system, metric, value, *, ci=None, ci_method="", direction="higher", details=None, caveats=(), kind="ladder", n_units=n, unit="document"):
        prov = prov_by_kind.get(kind, prov_common + [P.get({"R7_safe_wrapper_stored": "safe_final", "stored_boundary_refiner": "safe_br",
                                                             "stored_mistral_gliner_taxonomy": "safe_mgt", "legacy_full_cag_graph": "legacy_full",
                                                             "legacy_mistral_gliner_taxonomy": "legacy_mgt",
                                                             "legacy_mistral_boundary": "legacy_boundary"}.get(system, "anchor"), P["anchor"])])
        cav = list(base_cav) + list(caveats) + ([replay_cav] if kind in ("ladder", "single", "leave_one_out", "policy_off_counterfactual", "leakage_ablation") and system != "R0_raw_mistral" else [])
        records.append(record(LAYER, system, metric, value, n_units=n_units, unit=unit, split=SPLIT, dataset=DATASET,
                              direction=direction, ci=ci, ci_method=ci_method, provenance=prov, caveats=cav, details=details or {}))

    level_metrics = ("exact_span_f1", "exact_span_precision", "exact_span_recall", "token_f1", "hard_skill_f1",
                     "soft_skill_f1", "soft_skill_recall", "apr", "soft_apr", "nwg_per_1000_gold",
                     "gained_tp_per_1000_gold", "destroyed_tp_per_1000_gold", "added_fp_per_1000_gold", "removed_fp_per_1000_gold")
    lower_better = {"destroyed_tp_per_1000_gold", "added_fp_per_1000_gold"}
    for k, (name, s) in enumerate(sysd.items()):
        rows = s["rows"]
        tot = rows.sum(axis=0)
        kind = s["kind"]
        row: Dict[str, Any] = {"system": name, "kind": kind, "description": s["desc"]}
        nb = n_boot_head if kind in ("ladder", "stored", "legacy") else n_boot
        # one set of document draws per system, shared with the anchor (paired) and across all metrics
        bs, ba = S.bootstrap_sums(rows, A, n_boot=nb, seed=seed + 31 * k)
        for m in level_metrics:
            b = S.summarize_samples(STATS[m](bs), float(STATS[m](tot)))
            row[m] = b["value"]
            row[m + "_ci"] = (b["ci_low"], b["ci_high"])
            det = {"description": s["desc"], "kind": kind, "n_boot": nb,
                   "headline": bool(name == "R7_safe_wrapper_stored" and m in ("exact_span_f1", "apr"))}
            if m in ("apr", "soft_apr"):
                det["definition"] = "pooled |anchor TP ∩ stage TP| / |anchor TP| (Decisions D-A2-02)"
            if m == "nwg_per_1000_gold":
                det["definition"] = "1000*[(TP gained - TP destroyed) - (FP added - FP removed)] / gold spans (D-A2-02)"
            rec(name, m, b["value"], ci=(b["ci_low"], b["ci_high"]), ci_method=f"document-cluster percentile bootstrap ({nb})",
                direction="lower" if m in lower_better else "higher", details=det, kind=kind)
        counts = {c: int(tot[C[c]]) for c in ("tp", "fp", "fn", "gained_tp", "destroyed_tp", "added_fp", "removed_fp", "anchor_tp",
                                               "kept_tp", "n_gold", "n_pred", "verifier_rejected", "verifier_false_rej",
                                               "calib_rejected", "calib_false_rej", "merger_gold_drop_anchor", "merger_gold_drop_new",
                                               "merger_dedupe_anchor_drop", "soft_anchor_tp", "soft_kept_tp")}
        counts["docs_changed"] = int(tot[C["changed"]])
        row.update(counts)
        row["linking_coverage"] = s.get("linking_coverage")
        if s.get("linking_coverage") is not None:
            rec(name, "linking_coverage", s["linking_coverage"], direction="none", kind=kind, unit="span", n_units=counts["n_pred"],
                details={"definition": "share of output spans with a canonical_skill_name (coverage, NOT linking accuracy; see L06)"},
                caveats=["Coverage only: no gold canonical ids here; BM25 fallback links almost any string."])
        for c in ("gained_tp", "destroyed_tp", "added_fp", "removed_fp", "verifier_false_rej", "calib_false_rej",
                  "merger_gold_drop_new", "merger_gold_drop_anchor", "docs_changed"):
            rec(name, {"gained_tp": "tp_gained (= recovered FN)", "destroyed_tp": "anchor_tp_destroyed", "added_fp": "fp_added",
                       "removed_fp": "fp_removed", "verifier_false_rej": "verifier_false_rejections",
                       "calib_false_rej": "calibrator_false_rejections", "merger_gold_drop_new": "merger_dropped_gold_new_candidates",
                       "merger_gold_drop_anchor": "merger_dropped_gold_anchor_spans", "docs_changed": "n_docs_changed_vs_anchor"}[c],
                float(counts[c]), direction="none", details={"count_total": counts[c], "unit_note": "pooled count over documents"}, kind=kind)
        # paired deltas vs anchor
        if name != "R0_raw_mistral":
            for j, m in enumerate(("exact_span_f1", "token_f1")):
                obs = float(STATS[m](tot) - STATS[m](A.sum(axis=0)))
                b = S.summarize_samples(STATS[m](bs) - STATS[m](ba), obs, paired=True)
                ar = S.approximate_randomization(rows, A, STATS[m], 10000 if not smoke else 500, seed + 1000 + 7 * k + j)
                ni = S.non_inferiority(b["ci_low"], NI_MARGIN, obs) if m == "exact_span_f1" else None
                row[f"delta_{m}"] = obs
                row[f"delta_{m}_ci"] = (b["ci_low"], b["ci_high"])
                row[f"p_boot_{m}"], row[f"p_ar_{m}"] = b["p"], ar["p"]
                if ni is not None:
                    row["non_inferior_-0.005"] = ni["non_inferior"]
                rec(name, f"delta_{m}_vs_anchor", obs, ci=(b["ci_low"], b["ci_high"]),
                    ci_method=f"paired document bootstrap ({nb})", direction="higher",
                    details={"p_boot": b["p"], "p_shift": b["p_shift"], "p_ar": ar["p"], "ar_method": ar["method"],
                             "mde_80": S.mde(b["se"]), "n_units_differ": int(ar["n_units_differ"]),
                             "non_inferiority": ni, "comparison": f"{name} - R0_raw_mistral"}, kind=kind)
            obs = float(STATS["soft_skill_recall"](A.sum(axis=0)) - STATS["soft_skill_recall"](tot))
            loss = S.summarize_samples(STATS["soft_skill_recall"](ba) - STATS["soft_skill_recall"](bs), obs, paired=True)
            row["soft_recall_loss"] = obs
            row["soft_recall_loss_ci"] = (loss["ci_low"], loss["ci_high"])
            rec(name, "soft_skill_recall_loss_vs_anchor", obs, ci=(loss["ci_low"], loss["ci_high"]),
                ci_method=f"paired document bootstrap ({nb})", direction="lower",
                details={"definition": "recall_SOFT(anchor) - recall_SOFT(stage); positive = loss", "p_boot": loss["p"]}, kind=kind)
        summary_rows.append(row)

    # ---------------- H2
    def apr_rows(name: str) -> np.ndarray:
        return sysd[name]["rows"][:, [C["kept_tp"], C["anchor_tp"]]]
    h2 = {"comparisons": []}
    treat = "R7_safe_wrapper_stored"
    for j, comp in enumerate(("legacy_full_cag_graph", "legacy_mistral_gliner_taxonomy")):
        b = S.paired_bootstrap(apr_rows(treat), apr_rows(comp), S.ratio_stat(0, 1), n_boot_head, seed + 3000 + j)
        ar = S.approximate_randomization(apr_rows(treat), apr_rows(comp), S.ratio_stat(0, 1), 10000 if not smoke else 500, seed + 3100 + j)
        h2["comparisons"].append({"comparator": comp, "delta_apr": b["delta"], "ci": [b["ci_low"], b["ci_high"]], "p_boot": b["p"],
                                  "p_ar": ar["p"], "apr_treatment": b["value_a"], "apr_comparator": b["value_b"],
                                  "n_units_differ": b["n_units_differ"]})
    holm = S.holm([c["p_boot"] for c in h2["comparisons"]])
    for c, hp in zip(h2["comparisons"], holm):
        c["p_holm"] = hp
        c["ci_excludes_0_positive"] = bool(c["ci"][0] is not None and c["ci"][0] > 0)
    f1cmp = S.compare_counts(sysd[treat]["rows"], A, STATS["exact_span_f1"], n_boot_head, 10000 if not smoke else 500, seed + 3200,
                             margin=NI_MARGIN)
    cond_i = all(c["ci_excludes_0_positive"] for c in h2["comparisons"])
    cond_ii = bool(f1cmp["non_inferiority"]["non_inferior"])
    verdict = "SUPPORTED" if (cond_i and cond_ii) else "NOT SUPPORTED"
    # secondary: same-anchor policy-off counterfactual
    sec = S.paired_bootstrap(apr_rows("R6_calibration"), apr_rows("policyoff_prefix_like"), S.ratio_stat(0, 1), n_boot_head, seed + 3300)
    sec_f1 = S.paired_bootstrap(sysd["R6_calibration"]["rows"], sysd["policyoff_prefix_like"]["rows"], STATS["exact_span_f1"], n_boot_head, seed + 3301)
    sec_ub = S.paired_bootstrap(apr_rows("R6_calibration"), apr_rows("policyoff_full"), S.ratio_stat(0, 1), n_boot_head, seed + 3302)
    h2.update({"treatment": treat, "condition_i_apr_higher_vs_both_legacy": cond_i, "condition_ii_f1_noninferior": cond_ii,
               "f1_delta_vs_anchor": f1cmp["delta"], "f1_ci": [f1cmp["ci_low"], f1cmp["ci_high"]], "f1_p_boot": f1cmp["p"],
               "f1_p_ar": f1cmp["p_ar"], "verdict": verdict,
               "secondary_policy_off": {"delta_apr_safe_replay_minus_policyoff": sec["delta"], "ci": [sec["ci_low"], sec["ci_high"]],
                                        "p": sec["p"], "apr_policyoff": sec["value_b"], "apr_safe_replay": sec["value_a"],
                                        "delta_f1": sec_f1["delta"], "delta_f1_ci": [sec_f1["ci_low"], sec_f1["ci_high"]],
                                        "comparator": "policyoff_prefix_like",
                                        "upper_bound_policyoff_full": {"delta_apr": sec_ub["delta"], "ci": [sec_ub["ci_low"], sec_ub["ci_high"]],
                                                                       "apr_policyoff_full": sec_ub["value_b"]},
                                        "label": "SECONDARY, post-hoc construct (not pre-registered)"}})
    rule = ("H2: APR(safe) - APR(legacy) 95% CI lower bound > 0 against BOTH legacy full_cag_graph and legacy "
            "mistral_gliner_taxonomy, AND exact-F1(safe) - exact-F1(anchor) lower 95% CI bound > -0.005 (D-A2-02).")
    primary = min(h2["comparisons"], key=lambda c: c["delta_apr"])  # the weaker of the two contrasts
    rec(treat, "H2_verdict", primary["delta_apr"], ci=tuple(primary["ci"]), ci_method=f"paired document bootstrap ({n_boot_head})",
        direction="higher", kind="stored",
        details={"verdict": verdict, "rule": rule, "p_raw": primary["p_boot"], "p_holm": primary["p_holm"], "family": "H2",
                 "primary_effect": f"delta APR vs {primary['comparator']} (the weaker of the two contrasts)", **h2},
        caveats=["Legacy comparators regenerated Mistral live (not the stored anchor) and already used source-aware "
                 "components, so ΔAPR mixes regeneration non-determinism with ungated post-processing."])
    H["h2"] = h2

    # ---------------- reproduction + audit records
    for k2, v in res["repro"].items():
        rec("replay_check", f"replay_reproduction:{k2}", float(v["docs_identical"]), direction="higher", kind="ladder",
            details={**v, "differing_doc_ids": v["differing_doc_ids"][:50]})
    audit = res["audit"]
    rec("tuning_audit", "components_flagged_for_test_contact", float(sum(c["flag"] for c in audit["components"])), direction="lower",
        kind="ladder", n_units=len(audit["components"]), unit="component",
        details=audit, caveats=["Audit of configuration provenance; see Progress.md integrity-flag lines."])
    H["summary_rows"] = summary_rows
    H["edits"] = {s: edit_breakdown(res, s) for s in ("R1_boundary", "R6_calibration", "R7_safe_wrapper_stored", "policyoff_full",
                                                       "policyoff_prefix_like", "legacy_full_cag_graph")}
    H["sources"] = {s: source_breakdown(res, s) for s in ("R3_dictionary", "R6_calibration", "R7_safe_wrapper_stored",
                                                           "legacy_full_cag_graph", "legacy_mistral_gliner_taxonomy", "policyoff_full",
                                                           "policyoff_prefix_like")}
    return records, H


# ----------------------------------------------------------------------------- outputs
def fmt_ci(ci: Tuple[Optional[float], Optional[float]], d: int = 4) -> str:
    if not ci or ci[0] is None:
        return "-"
    return f"[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]" if ci[0] < 0 or ci[1] < 0 else f"[{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def write_outputs(out_dir: Path, res: Dict[str, Any], H: Dict[str, Any], records: List[Dict[str, Any]], smoke: bool) -> None:
    lad = out_dir / "ladder"
    lad.mkdir(parents=True, exist_ok=True)
    rows = H["summary_rows"]
    keys = ["system", "kind", "description", "exact_span_f1", "exact_span_f1_ci", "delta_exact_span_f1", "delta_exact_span_f1_ci",
            "p_boot_exact_span_f1", "p_ar_exact_span_f1", "non_inferior_-0.005", "token_f1", "delta_token_f1", "apr", "apr_ci",
            "soft_apr", "nwg_per_1000_gold", "nwg_per_1000_gold_ci", "gained_tp", "destroyed_tp", "added_fp", "removed_fp",
            "verifier_rejected", "verifier_false_rej", "calib_rejected", "calib_false_rej", "merger_gold_drop_new",
            "merger_gold_drop_anchor", "merger_dedupe_anchor_drop", "soft_skill_recall", "soft_recall_loss", "soft_recall_loss_ci",
            "hard_skill_f1", "soft_skill_f1", "docs_changed", "tp", "fp", "fn", "n_pred"]
    with open(lad / "ladder_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([json.dumps(r.get(k)) if isinstance(r.get(k), (tuple, list)) else r.get(k) for k in keys])
    (lad / "ladder_summary.json").write_text(json.dumps({"rows": rows, "h2": H["h2"], "edits": H["edits"], "sources": H["sources"],
                                                         "replay_reproduction": res["repro"]}, indent=1, default=float))
    # per-document long table (counts only, no text)
    with open(lad / "per_doc_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit_id", "system"] + COLS)
        for name, s in res["systems"].items():
            for i, r in zip(res["ids"], s["rows"]):
                w.writerow([i, name] + [int(x) for x in r])
    # per-unit scores for L10
    with open(out_dir / "per_unit_scores.jsonl", "w") as fh:
        for name, s in res["systems"].items():
            fam = "H2_wrapper_vs_anchor" if s["kind"] in ("ladder", "stored", "legacy") else f"L05_{s['kind']}"
            extra = {} if name == "R0_raw_mistral" else {"reference": "R0_raw_mistral"}
            for i, r in zip(res["ids"], s["rows"]):
                for metric, cols in (("exact_span", ("tp", "fp", "fn")), ("token", ("ttp", "tfp", "tfn"))):
                    fh.write(json.dumps({"layer": LAYER, "family": fam, "system": name, "metric": metric, "label": "ALL",
                                         "unit_id": i, "tp": int(r[C[cols[0]]]), "fp": int(r[C[cols[1]]]), "fn": int(r[C[cols[2]]]), **extra}) + "\n")
                fh.write(json.dumps({"layer": LAYER, "family": fam, "system": name, "metric": "exact_span", "label": "HARD_SKILL",
                                     "unit_id": i, "tp": int(r[C["htp"]]), "fp": int(r[C["hfp"]]), "fn": int(r[C["hfn"]]), **extra}) + "\n")
                fh.write(json.dumps({"layer": LAYER, "family": fam, "system": name, "metric": "exact_span", "label": "SOFT_SKILL",
                                     "unit_id": i, "tp": int(r[C["stp"]]), "fp": int(r[C["sfp"]]), "fn": int(r[C["sfn"]]), **extra}) + "\n")
    # compact replayed predictions (offsets/labels/sources only) for audit
    pdir = DERIVED_DIR / ("replay_predictions_smoke" if smoke else "replay_predictions")
    pdir.mkdir(parents=True, exist_ok=True)
    for name in res["systems"]:
        if res["systems"][name]["kind"] in ("stored", "legacy") or name == "R0_raw_mistral":
            continue
        with open(pdir / f"{name}.jsonl", "w") as fh:
            for i in res["ids"]:
                fh.write(json.dumps({"id": i, "spans": [[sp.get("start"), sp.get("end"), sp.get("label"), sorted(sources(sp)),
                                                          ((sp.get("metadata") or {}).get("boundary_edit") or {}).get("edit_type", "original")]
                                                         for sp in res["preds"][name][i]]}) + "\n")
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def ladder_fig(order: List[Dict[str, Any]], path: Path, title: str) -> None:
            names = [r["system"] for r in order]
            d = np.array([r["delta_exact_span_f1"] for r in order])
            lo = np.array([r["delta_exact_span_f1_ci"][0] for r in order])
            hi = np.array([r["delta_exact_span_f1_ci"][1] for r in order])
            fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(names) + 1.8), sharey=True)
            y = np.arange(len(names))[::-1].astype(float)
            axes[0].errorbar(d, y, xerr=[d - lo, hi - d], fmt="o", color="#1f5f99", ecolor="#7aa6c9", capsize=3)
            axes[0].axvline(0, color="k", lw=0.8)
            axes[0].axvline(NI_MARGIN, color="#b03a2e", lw=0.8, ls="--", label="non-inferiority margin -0.005")
            axes[0].set_yticks(y)
            axes[0].set_yticklabels(names, fontsize=8)
            axes[0].set_xlabel("Δ exact-span F1 vs raw anchor (95% paired document bootstrap CI)")
            axes[0].legend(fontsize=7, loc="lower left")
            g = np.array([r["gained_tp"] for r in order])
            de = np.array([r["destroyed_tp"] for r in order])
            ad = np.array([r["added_fp"] for r in order])
            rm = np.array([r["removed_fp"] for r in order])
            axes[1].barh(y + 0.18, g, height=0.34, color="#2e8b57", label="TP gained")
            axes[1].barh(y + 0.18, -de, height=0.34, color="#c0392b", label="anchor TP destroyed")
            axes[1].barh(y - 0.18, rm, height=0.34, color="#7fbf7f", label="FP removed")
            axes[1].barh(y - 0.18, -ad, height=0.34, color="#e6a0a0", label="FP added")
            axes[1].axvline(0, color="k", lw=0.8)
            axes[1].set_xlabel("span count vs anchor (right = good, left = bad)")
            axes[1].legend(fontsize=7, loc="lower left")
            fig.suptitle(title, fontsize=9)
            fig.tight_layout()
            fig.savefig(path, dpi=130)
            plt.close(fig)

        main = [r for r in rows if r["system"] != "R0_raw_mistral" and r["kind"] in ("ladder", "stored", "legacy", "leakage_ablation")]
        ladder_fig(main, out_dir / "wrapper_ladder.png", f"Safe-wrapper ladder, stored artefacts and legacy comparators ({len(res['ids'])} SkillSpan test docs)")
        cf = [r for r in rows if r["kind"] == "policy_off_counterfactual"] + [r for r in rows if r["system"] == "R6_calibration"]
        ladder_fig(cf, out_dir / "wrapper_policy_off.png", "Post-hoc counterfactual: source-aware protection switched off (SECONDARY)")
        with open(out_dir / "wrapper_ladder.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["system", "kind", "delta_exact_f1", "ci_low", "ci_high", "tp_gained", "anchor_tp_destroyed", "fp_added", "fp_removed", "apr"])
            for r in main + cf:
                w.writerow([r["system"], r["kind"], r["delta_exact_span_f1"], r["delta_exact_span_f1_ci"][0], r["delta_exact_span_f1_ci"][1],
                            r["gained_tp"], r["destroyed_tp"], r["added_fp"], r["removed_fp"], r["apr"]])
    except Exception as exc:  # noqa: BLE001
        print(f"figure failed: {exc}")


def historical_line() -> str:
    """Aggregate counts of the destroyed pre-fix wrapper (only aggregates survive; run-1 anchor, not the stored one)."""
    p = REPO_ROOT / P["prefix_analysis"]
    if not p.exists():
        return "- Historical pre-fix wrapper: analysis file missing."
    d = json.loads(p.read_text())
    st = d["stages"]["full_cag_graph"]
    c = st["counts"]
    apr = c["kept"] / (c["kept"] + c["removed"]) if (c["kept"] + c["removed"]) else float("nan")
    return (f"- Historical pre-fix full wrapper (`{P['prefix_analysis']}`, vs the run-1 anchor F1 "
            f"{d['raw_mistral_metrics']['exact_span_f1']:.4f}; aggregates only, no predictions on disk): exact F1 "
            f"{st['metrics']['exact_span_f1']:.4f} (Δ {st['deltas_vs_raw_mistral']['exact_span_f1']:+.4f}); anchor TPs kept {c['kept']}, "
            f"destroyed {c['removed']} → implied APR {apr:.4f}; boundary-modified {c['boundary_modified']}, verifier rejections "
            f"{c['rejected_by_verifier']}, GLiNER additions {c['added_by_gliner']}, dictionary additions {c['added_by_dictionary']}. "
            "Use it to judge how close the policy-off constructs are to the real pre-fix behaviour.")


def report_md(res: Dict[str, Any], H: Dict[str, Any], seed: int, n_boot: int, n_boot_head: int) -> str:
    rows = {r["system"]: r for r in H["summary_rows"]}
    h2 = H["h2"]
    L = ["# L05_wrapper — contribution of each safe-wrapper component vs the Raw Mistral LoRA anchor", "",
         f"Data: {DATASET}; n = {len(res['ids'])} documents. Seed {seed}; bootstrap {n_boot} (headline F1/APR/H2: {n_boot_head}); "
         "approximate randomization 10,000 shuffles (exact enumeration when ≤16 documents differ). Definitions: Decisions.md D-A2-02.", "",
         "## Key findings (computed)", ""]
    sw, md_, al = rows["R7_safe_wrapper_stored"], rows["alone_merger_dedupe"], rows.get("policyoff_merger_only", {})
    singles = sorted((rows[n] for n, *_ in SINGLE), key=lambda r: -abs(r["delta_exact_span_f1"]))
    L += [f"- Shipped safe wrapper vs raw anchor: exact F1 {sw['exact_span_f1']:.4f} vs {rows['R0_raw_mistral']['exact_span_f1']:.4f}, "
          f"Δ {sw['delta_exact_span_f1']:+.4f} {fmt_ci(sw['delta_exact_span_f1_ci'])} (AR p {sw['p_ar_exact_span_f1']:.3f}); only "
          f"{sw['docs_changed']} of {len(res['ids'])} documents change. Token F1 Δ {sw['delta_token_f1']:+.4f}.",
          f"- Anchor preservation: APR {sw['apr']:.4f} ({sw['destroyed_tp']} anchor TPs lost). The CandidateMerger's grouping / "
          f"subsumption step is not anchor-protected: on the anchor alone it removes {md_['destroyed_tp']} TPs and {md_['removed_fp']} FPs.",
          f"- Largest single-component effect: `{singles[0]['system']}` Δ {singles[0]['delta_exact_span_f1']:+.4f} "
          f"{fmt_ci(singles[0]['delta_exact_span_f1_ci'])} (AR p {singles[0]['p_ar_exact_span_f1']:.3f}, {singles[0]['docs_changed']} docs).",
          f"- Add-only merger: {rows['R6_calibration']['merger_gold_drop_new']} gold-correct GLiNER/dictionary candidates are rejected; "
          f"switching the policy off would add {al.get('gained_tp', 0)} TPs but {al.get('added_fp', 0)} FPs (Δ F1 {al.get('delta_exact_span_f1', float('nan')):+.4f}).",
          "- New (non-anchor) spans in the shipped output by origin (n / gold): " + "; ".join(
              f"{k} {v.get('n', 0)} / {v.get('gold', 0)}" for k, v in sorted(H["sources"].get("R7_safe_wrapper_stored", {}).items()))
          + ". The measurable gain comes from the verifier removing offset-less spans and from `realign_word_boundary` "
          "repairing anchor offsets, i.e. from cleaning the anchor's offset reconstruction rather than from new skills.", "",
          "## H2 verdict", "",
          f"**{h2['verdict']}.** Rule: {('APR(safe) − APR(legacy) CI > 0 vs BOTH legacy artefacts AND F1(safe) − F1(anchor) lower CI > −0.005')}.", ""]
    L.append(md_table([{"comparator": c["comparator"], "APR safe": c["apr_treatment"], "APR comparator": c["apr_comparator"],
                        "ΔAPR": c["delta_apr"], "95% CI": fmt_ci(tuple(c["ci"])), "p boot": c["p_boot"], "p AR": c["p_ar"],
                        "p Holm": c["p_holm"]} for c in h2["comparisons"]],
                      ["comparator", "APR safe", "APR comparator", "ΔAPR", "95% CI", "p boot", "p AR", "p Holm"]))
    L += ["", f"- Condition (i) ΔAPR > 0 vs both legacy: **{h2['condition_i_apr_higher_vs_both_legacy']}**.",
          f"- Condition (ii) non-inferiority: Δ exact F1 (safe − anchor) = {h2['f1_delta_vs_anchor']:+.5f}, 95% CI {fmt_ci(tuple(h2['f1_ci']), 5)} "
          f"(p boot {h2['f1_p_boot']:.4f}, p AR {h2['f1_p_ar']:.4f}); margin −0.005 → **{h2['condition_ii_f1_noninferior']}**.",
          f"- Secondary (post-hoc, same anchor): APR safe replay {h2['secondary_policy_off']['apr_safe_replay']:.4f} vs pre-fix-like policy-off "
          f"counterfactual {h2['secondary_policy_off']['apr_policyoff']:.4f}, Δ {h2['secondary_policy_off']['delta_apr_safe_replay_minus_policyoff']:+.4f} "
          f"{fmt_ci(tuple(h2['secondary_policy_off']['ci']))}; ΔF1 {h2['secondary_policy_off']['delta_f1']:+.4f} {fmt_ci(tuple(h2['secondary_policy_off']['delta_f1_ci']))}; "
          f"upper bound (all edits forced) APR {h2['secondary_policy_off']['upper_bound_policyoff_full']['apr_policyoff_full']:.4f}.",
          historical_line(),
          "- Caveat: the legacy artefacts regenerated Mistral live and already contained the source-aware components (their metadata "
          "shows `accepted_anchor`), so their APR loss mixes generation non-determinism with ungated post-processing. The truly "
          "destructive pre-fix pipeline (exact F1 0.56–0.57 in `wrapper_error_analysis.md`) has no predictions on disk; the "
          "policy-off counterfactual is A2's approximation of it.", "",
          "## Ladder (cumulative; replayed on the stored anchor)", ""]
    cols = ["system", "exact F1", "Δ vs anchor [95% CI]", "p AR", "APR", "NWG/1000", "TP+", "TP−", "FP+", "FP−", "docs Δ", "soft-recall loss"]

    def line(name):
        r = rows[name]
        if name == "R0_raw_mistral":
            return {"system": name, "exact F1": r["exact_span_f1"], "Δ vs anchor [95% CI]": "-", "p AR": None, "APR": r["apr"],
                    "NWG/1000": r["nwg_per_1000_gold"], "TP+": 0, "TP−": 0, "FP+": 0, "FP−": 0, "docs Δ": 0, "soft-recall loss": None}
        return {"system": name, "exact F1": r["exact_span_f1"],
                "Δ vs anchor [95% CI]": f"{r['delta_exact_span_f1']:+.4f} {fmt_ci(r['delta_exact_span_f1_ci'])}",
                "p AR": r["p_ar_exact_span_f1"], "APR": r["apr"], "NWG/1000": r["nwg_per_1000_gold"], "TP+": r["gained_tp"],
                "TP−": r["destroyed_tp"], "FP+": r["added_fp"], "FP−": r["removed_fp"], "docs Δ": r["docs_changed"],
                "soft-recall loss": r["soft_recall_loss"]}
    L.append(md_table([line(n) for n, *_ in LADDER] + [line("R7_safe_wrapper_stored")], cols))
    L += ["", "Stage diagnostics (pooled counts): verifier rejections / of which gold-correct; calibrator rejections / gold-correct; "
          "merger: gold-correct GLiNER/dictionary candidates dropped by the add-only policy, gold-correct anchor spans dropped, "
          "anchor spans removed by grouping/subsumption.", ""]
    L.append(md_table([{"system": n, "verifier rej": rows[n]["verifier_rejected"], "verifier false rej": rows[n]["verifier_false_rej"],
                        "calib rej": rows[n]["calib_rejected"], "calib false rej": rows[n]["calib_false_rej"],
                        "merger gold drop (new)": rows[n]["merger_gold_drop_new"], "merger gold drop (anchor)": rows[n]["merger_gold_drop_anchor"],
                        "merger anchor dedupe": rows[n]["merger_dedupe_anchor_drop"]} for n, *_ in LADDER[1:]],
                      ["system", "verifier rej", "verifier false rej", "calib rej", "calib false rej", "merger gold drop (new)",
                       "merger gold drop (anchor)", "merger anchor dedupe"]))
    L += ["", "## Order-dependence: single components and leave-one-out", ""]
    L.append(md_table([line(n) for n, *_ in SINGLE + LOO], cols))
    L += ["", "## Stored artefacts, legacy comparators, policy-off counterfactual, leakage ablation", ""]
    L.append(md_table([line(n) for n in ("stored_boundary_refiner", "stored_mistral_gliner_taxonomy", "legacy_full_cag_graph",
                                          "legacy_mistral_gliner_taxonomy", "legacy_mistral_boundary") + tuple(n for n, *_ in POLICY_OFF)
                       + ("leakage_no_test_mined_aliases",)], cols))
    lk = rows["leakage_no_test_mined_aliases"]
    r6 = rows["R6_calibration"]
    L += ["", f"Leakage ablation: removing the {res['audit']['alias_source_counts'].get('mistral_high_confidence', 0)} alias rows mined from "
          f"test-split anchor predictions changes the replayed safe wrapper's exact F1 from {r6['exact_span_f1']:.5f} to "
          f"{lk['exact_span_f1']:.5f} (each vs anchor: {r6['delta_exact_span_f1']:+.5f} and {lk['delta_exact_span_f1']:+.5f}).", "",
          "## Where the changes come from", "", "Boundary edits applied (by type): n / original span was gold / edited span is gold.", ""]
    for s, e in H["edits"].items():
        L.append(f"- `{s}`: " + ("; ".join(f"{k}: {v.get('n', 0)} / {v.get('orig_was_gold', 0)} / {v.get('new_is_gold', 0)}" for k, v in sorted(e.items())) or "none"))
    L += ["", "Spans not in the anchor, by origin: n / gold.", ""]
    for s, e in H["sources"].items():
        L.append(f"- `{s}`: " + ("; ".join(f"{k}: {v.get('n', 0)} / {v.get('gold', 0)}" for k, v in sorted(e.items())) or "none"))
    L += ["", "## Replay reproduction", ""]
    for k, v in res["repro"].items():
        L.append(f"- {k}: {v['docs_identical']}/{v['n_docs']} documents identical (differing: {', '.join(v['differing_doc_ids'][:10]) or 'none'}).")
    L += ["", "## Tuning / test-contact audit (contract rule 2)", ""]
    L.append(md_table([{"component": c["component"], "configured on": json.dumps(c["configured_on"]) if isinstance(c["configured_on"], dict) else c["configured_on"],
                        "test contact": c["test_contact"], "flag": "FLAG" if c["flag"] else ""} for c in res["audit"]["components"]],
                      ["component", "configured on", "test contact", "flag"]))
    L += ["", "## Reading guide / caveats", "",
          "- All wrapper effects are a handful of documents; read Δ with `docs Δ` and the AR p-value (exact when ≤16 documents differ).",
          "- TP gained ≡ recovered anchor FN (set semantics, fixed gold).",
          "- Taxonomy linking never changes spans (asserted); its contribution is linking coverage, evaluated in L06.",
          "- The replayed GLiNER source is the stored zero-shot GLiNER output, not the wrapper's own (unpersisted) live calls.",
          "- Single anchor training run, single split; no training-seed variance in any CI.",
          "", "Files: `ladder/ladder_summary.{csv,json}`, `ladder/per_doc_counts.csv`, `per_unit_scores.jsonl`, `wrapper_ladder.{png,csv}`, `wrapper_policy_off.png`, "
          f"replayed predictions `{rel(DERIVED_DIR)}/replay_predictions/`."]
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-head", type=int, default=10000)
    args = ap.parse_args(argv)
    import os
    os.chdir(REPO_ROOT)  # components resolve their default config/alias paths relative to the repo root
    n_boot = min(args.n_boot, 200) if args.smoke else args.n_boot
    n_head = min(args.n_boot_head, 300) if args.smoke else args.n_boot_head
    res = compute(REPO_ROOT, args.smoke, args.seed, n_boot, n_head)
    records, H = analyse(res, args.seed, n_boot, n_head, args.smoke)
    out = write_layer(LAYER, records, report_md(res, H, args.seed, n_boot, n_head), seed=args.seed, smoke=args.smoke,
                      extra_meta={"n_boot": n_boot, "n_boot_headline": n_head, "elapsed_compute_s": res["elapsed"]})
    write_outputs(out, res, H, records, args.smoke)
    print(f"wrote {out} ({len(records)} records); H2 {H['h2']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
