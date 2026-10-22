"""Registry of evaluation modules in the suite, in run order, with defensive loading.

An eval module is any importable module exposing ``run(ctx: EvalContext) -> list[EvalResult]``.  Modules written by other
people may not exist yet; ``load`` never raises -- it returns ``(run_fn | None, status, reason)`` so the runner can mark the
module ``not_available`` (module file absent) or ``import_error`` (module present but broken) and carry on.

To add an eval: create ``src/skillllm_v2/evals/<name>_evals.py`` with ``run(ctx)``, then append an ``EvalSpec`` below
(declare which pipeline stages it covers so the coverage section of the report stays honest).
"""

from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

# Pipeline stages in execution order (conceptual; mirrors docs/SKILLLM_V2_PIPELINE.md and wrapper.SkillLLMv2Pipeline).
PIPELINE_STAGES: List[Tuple[str, str]] = [
    ("extraction", "candidate extraction (Mistral / GLiNER / dictionary)"),
    ("candidate_merging", "candidate merging across extractors"),
    ("boundary_refinement", "span boundary refinement"),
    ("span_verification", "span verifier (accept / reject / correct)"),
    ("confidence_calibration", "confidence calibration + thresholds"),
    ("taxonomy_linking", "taxonomy / canonical skill linking"),
    ("context_building", "cached context / profile summarisation (CAG)"),
    ("graph", "skill-course knowledge graph"),
    ("job_matching", "job <-> candidate matching"),
    ("gap_detection", "skill-gap detection and weighting"),
    ("course_recommendation", "course retrieval / ranking / path planning"),
    ("explanation", "evidence-backed explanations"),
    ("end_to_end", "whole pipeline: text in, final JSON out"),
]


@dataclass(frozen=True)
class EvalSpec:
    name: str  # short registry name, used by --only
    module_path: str  # dotted import path
    description: str = ""
    stages: Tuple[str, ...] = ()  # stages this module evaluates directly
    partial_stages: Tuple[str, ...] = ()  # stages it touches only indirectly / on synthetic data
    run_fn: Optional[Callable] = field(default=None, compare=False)  # injected callable (tests); bypasses import


REGISTRY: List[EvalSpec] = [
    EvalSpec("extraction", "skillllm_v2.evals.extraction_evals", "span extraction quality, merger/refiner effects",
             ("extraction", "candidate_merging", "boundary_refinement")),
    EvalSpec("verifier", "skillllm_v2.evals.verifier_evals", "span verifier accept/reject quality", ("span_verification",)),
    EvalSpec("calibration", "skillllm_v2.evals.calibration_evals", "confidence calibration (ECE, reliability)", ("confidence_calibration",)),
    EvalSpec("linking", "skillllm_v2.evals.linking_evals", "taxonomy linking accuracy", ("taxonomy_linking",)),
    EvalSpec("context", "skillllm_v2.evals.context_evals", "context compaction / cached context fidelity", ("context_building",)),
    EvalSpec("graph", "skillllm_v2.evals.graph_evals", "knowledge-graph integrity and retrieval", ("graph",)),
    EvalSpec("match", "skillllm_v2.evals.match_evals", "job matching ranking quality, robustness, invariance, properties",
             ("job_matching",)),
    EvalSpec("gap", "skillllm_v2.evals.gap_evals", "gap detection P/R/F1, severity rank correlation, monotonicity properties",
             ("gap_detection",)),
    EvalSpec("explanation", "skillllm_v2.evals.explanation_evals", "explanation faithfulness / completeness", ("explanation",)),
    EvalSpec("e2e", "skillllm_v2.evals.e2e_evals", "end-to-end traceability, schema, latency, determinism, degradation",
             ("end_to_end",), ("course_recommendation", "explanation")),
]


def names() -> List[str]:
    return [s.name for s in REGISTRY]


def select(only: Optional[Sequence[str]] = None, registry: Optional[Sequence[EvalSpec]] = None) -> List[EvalSpec]:
    """Registry order is preserved.  `only` entries match the short name, the module path, or a `*_evals` file stem."""
    reg = list(registry if registry is not None else REGISTRY)
    if not only:
        return reg
    wanted = {o.strip() for o in only if o.strip()}
    out = [s for s in reg if s.name in wanted or s.module_path in wanted or s.module_path.rsplit(".", 1)[-1] in wanted]
    unknown = wanted - {x for s in out for x in (s.name, s.module_path, s.module_path.rsplit(".", 1)[-1])}
    if unknown:
        raise ValueError(f"unknown eval module(s): {sorted(unknown)}; known: {[s.name for s in reg]}")
    return out


def load(spec: EvalSpec) -> Tuple[Optional[Callable], str, str]:
    """Returns (run_fn, status, reason); status in {"ok", "not_available", "import_error", "no_run_function"}.  Never raises."""
    if spec.run_fn is not None:
        return spec.run_fn, "ok", ""
    try:
        mod = importlib.import_module(spec.module_path)
    except ModuleNotFoundError as exc:
        if exc.name and (spec.module_path == exc.name or spec.module_path.startswith(exc.name + ".")):
            return None, "not_available", f"module {spec.module_path} does not exist (yet)"
        return None, "import_error", f"missing dependency: {exc}"
    except Exception as exc:  # noqa: BLE001 - a broken sibling must not stop the suite
        return None, "import_error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
    fn = getattr(mod, "run", None)
    if not callable(fn):
        return None, "no_run_function", f"{spec.module_path} has no callable run(ctx)"
    return fn, "ok", ""
