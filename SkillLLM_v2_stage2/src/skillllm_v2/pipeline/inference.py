"""inference.py -- the single end-to-end SkillLLM_v2 inference pipeline.

Wires every inference-relevant module into twelve individually testable
stages, each returning a typed artifact (see ``inference_types``) with
provenance, timing and a shared :class:`PipelineTrace`::

     1 extract    Mistral anchor (lazy, optional) + optional GLiNER + dictionary
     2 merge      CandidateMerger (source-aware policy: anchor-protected, add-only)
     3 refine     BoundaryRefiner + BoundarySelector
     4 verify     SpanVerifier
     5 calibrate  ConfidenceCalibrator
     6 link       taxonomy link (attach-only; never removes a skill)
     7 gate       safe-wrapper non-regression gate: fall back to the UNTOUCHED anchor when floors fail
     8 profile    profile summary, skill memory, compaction, context retrieval, contextual retrieval
     9 graph      graph build + SkillGraphRetriever + skill-path explainer
    10 match      JobMatcher + GapDetector
    11 courses    course recommendation THROUGH CourseRecommenderPort ONLY
    12 explain    final evidence-backed explanation

Graceful degradation: with no GPU / no Mistral weights the anchor is either a
deterministic stub (``anchor.mode: stub|auto`` + ``allow_stub``) or absent
(``anchor.mode: none``); both are flagged in every affected candidate's
``metadata.extractor_mode`` and in ``trace.stub_components`` -- model output is
never silently faked.

CLI::

    python -m skillllm_v2.pipeline.inference --profile P.json --job J.jsonl --out out.json [--no-mistral]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from skillllm_v2.config import deep_merge, load_config, repo_root_from
from skillllm_v2.context.cached_context_builder import build_cached_context_packet
from skillllm_v2.context.context_retriever import ContextRetriever
from skillllm_v2.extract import CandidateMerger, DictionaryExtractor, GLiNERExtractor, MistralExtractor
from skillllm_v2.graph.graph_builder import SkillGraphBuilder
from skillllm_v2.graph.graph_retriever import SkillGraphRetriever
from skillllm_v2.graph.schema import EdgeType, NodeType
from skillllm_v2.graph.skill_graph_explainer import explain_skill_path
from skillllm_v2.io_utils import read_json, read_jsonl, write_json
from skillllm_v2.match.explanation_generator import generate_recommendation_explanation
from skillllm_v2.match.gap_detector import GapDetector
from skillllm_v2.match.job_matcher import JobMatcher
from skillllm_v2.normalize import AliasResolver, build_taxonomy_linker
from skillllm_v2.pipeline.course_port import (
    CourseRecommendationRequest,
    CourseRecommenderPort,
    NullCourseRecommender,
    build_course_recommender,
)
from skillllm_v2.pipeline.inference_types import (
    JOB_EXTRACT_STAGE,
    PIPELINE_VERSION,
    REQUIRED_STAGES,
    STAGE_ORDER,
    AnchorMode,
    CandidateArtifact,
    CourseArtifact,
    ExplanationArtifact,
    ExtractArtifact,
    ExtractionResult,
    GateArtifact,
    GraphArtifact,
    MatchArtifact,
    PipelineResult,
    PipelineTrace,
    ProfileArtifact,
    StageRecord,
)
from skillllm_v2.policy import ANCHOR_SOURCES, is_anchor_candidate
from skillllm_v2.retrieval.contextual_retrieval import ContextualRetriever
from skillllm_v2.schema import graph_key, normalize_text, validate_skill
from skillllm_v2.verify import BoundaryRefiner, BoundarySelector, ConfidenceCalibrator, LearnedSpanVerifier, SpanVerifier

DEFAULT_CONFIG_PATH = "configs/skillllm_v2_inference.yaml"
#: Component switches accepted by ``skip=`` in addition to stage names.
COMPONENT_SWITCHES: tuple[str, ...] = ("mistral", "gliner", "dictionary")

DEFAULT_INFERENCE: dict[str, Any] = {
    "stages": {name: True for name in STAGE_ORDER},
    # When an optional downstream/post-processing stage raises: record it as failed, pass data through, keep going.
    "continue_on_error": True,
    "anchor": {
        # auto: real Mistral if weights load, else deterministic stub (allow_stub) else no anchor.
        # mistral: real model or raise. stub: deterministic stub (flagged). none: no anchor (dictionary/GLiNER only).
        "mode": "auto",
        # The deterministic stub is OPT-IN: auto never substitutes it silently. Set anchor.mode=stub / --no-mistral
        # (or allow_stub: true) to use it; it is then flagged in the trace and on every candidate.
        "allow_stub": False,
        # Mistral's extractor assigns a CONSTANT confidence (0.86), so calibration thresholds are not meaningful for
        # anchor spans. "constant" records that; "hook" requires confidence_hook=...; "logprob" is NOT IMPLEMENTED.
        "confidence": "constant",
    },
    "verify": {
        # Optional trained TF-IDF/LogReg span verifier (never evaluated as an inference component): off by default.
        "learned": {"enabled": False, "model_path": "artifacts/skillllm_v2/span_verifier/model/verifier_model.joblib", "threshold": 0.5},
    },
    "gliner": {"enabled": "auto", "device": "cpu", "threshold": 0.35, "allow_fallback": False},
    "dictionary": {"enabled": True},
    "taxonomy": {
        "attach_min_confidence": 0.0,  # 0.0 == attach every link (parity with the evaluated wrapper)
        "alias_table_paths": ["data/skillllm_v2/taxonomy/skill_aliases.jsonl"],
        "esco_path": None,
        "onet_path": None,
        "lightcast_path": None,
    },
    "paths": {
        "merger_tuned_config": "artifacts/skillllm_v2/candidate_merger/tuned_config.yaml",
        "boundary_tuned_config": "artifacts/skillllm_v2/boundary_refiner/tuned_config.yaml",
        "calibrator_tuned_config": "artifacts/skillllm_v2/confidence_calibrator/tuned_config.yaml",
    },
    "gate": {
        # auto | wrapper (force accept) | anchor (force fallback)
        "mode": "auto",
        "status_path": "artifacts/skillllm_v2/safe_wrapper/non_regression_status.json",
        "guardrails_path": "configs/skillllm_v2_wrapper_guardrails.yaml",
        # decision used when the anchor is a real model but no offline gate evidence exists
        "on_missing_status": "fallback_to_anchor",
        # inference-time structural guard (not tuned): min share of anchor (start, end, label) spans the wrapper must keep
        "min_anchor_retention": 0.8,
        # which candidates stages 8-12 consume: "enriched" (wrapper-linked; documented gate behaviour) or "selected"
        "downstream_from": "enriched",
    },
    "profile": {"user_id": "demo_user", "top_k_skills": 5, "top_k_jobs": 5},
    "matching": {"uncertain_confidence_below": 0.5},
    # backend: local (default; nothing changes) | auto | neo4j | mirror -- see docs/NEO4J_GRAPH.md. Only the catalogue graph
    # is ever read from / written to Neo4j; the user's own skills and the target job stay in the local in-memory graph.
    "graph": {"max_hops": 4, "backend": "local"},
    "courses": {"port": "auto", "catalog_path": "inputs/courses/course_catalog.jsonl", "top_k": 5},
}


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


# --------------------------------------------------------------------------- stub anchor
class DeterministicStubExtractor:
    """Deterministic alias-matching stand-in for the Mistral anchor.

    Emits ``source="mistral"`` candidates so the source-aware policy treats them
    like the anchor, but EVERY candidate is tagged
    ``metadata.extractor_mode="deterministic_stub"`` / ``metadata.stub=True``,
    and the pipeline records the stub in ``trace.stub_components``. It is not
    model output and must never be reported as such.
    """

    mode = "stub"

    def __init__(self, aliases: list[str] | None = None) -> None:
        self._inner = MistralExtractor(model_path=None, fallback_aliases=aliases, allow_fallback=True)

    def is_model_loaded(self) -> bool:
        return False

    def load(self) -> None:
        return None

    def extract(self, text: str) -> list[dict[str, Any]]:
        out = self._inner.extract(text)
        for candidate in out:
            candidate["metadata"] = dict(candidate.get("metadata") or {})
            candidate["metadata"]["extractor_mode"] = "deterministic_stub"
            candidate["metadata"]["stub"] = True
        return out


def _tag(candidates: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    for candidate in candidates:
        candidate["metadata"] = dict(candidate.get("metadata") or {})
        candidate["metadata"].setdefault("extractor_mode", mode)
    return candidates


def _drop_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (candidate.get("surface_form"), candidate.get("start"), candidate.get("end"), candidate.get("label"))


def _span_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (candidate.get("start"), candidate.get("end"), candidate.get("label"))


def _skill_names(items: Iterable[Any] | None) -> list[str]:
    names: list[str] = []
    for item in items or []:
        if isinstance(item, str):
            name = item
        elif isinstance(item, Mapping):
            name = item.get("canonical_skill_name") or item.get("skill_name") or item.get("name") or item.get("skill") or item.get("surface_form") or ""
        else:
            name = str(item)
        name = str(name).strip()
        if name:
            names.append(name)
    return names


# --------------------------------------------------------------------------- pipeline
class SkillLLMV2Pipeline:
    """Config-driven end-to-end pipeline. See the module docstring for the stage list.

    Every collaborator can be injected (``mistral``, ``gliner``, ``dictionary``, ``merger``,
    ``refiner``, ``verifier``, ``calibrator``, ``linker``, ``job_matcher``, ``gap_detector``,
    ``course_port``) -- that is the supported way to swap a component.

    A failing extractor (Mistral anchor, GLiNER or dictionary) never blocks the run: the failure is recorded on the
    ``extract`` stage record (``status="degraded"``, error text in ``provenance["extractor_failures"]`` and the notes, a
    warning on the trace) and the remaining extractors are used. If every extractor that was tried fails, the extraction
    is empty and the stage record is ``status="failed"``. ``strict_extractors=True`` (or ``continue_on_error: false``)
    makes an extractor exception propagate instead; an explicit ``anchor.mode: mistral`` that cannot load always raises.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | str | Path | None = None,
        *,
        root: str | Path | None = None,
        mistral: Any | None = None,
        gliner: Any | None = None,
        dictionary: Any | None = None,
        merger: CandidateMerger | None = None,
        refiner: BoundaryRefiner | None = None,
        selector: BoundarySelector | None = None,
        verifier: SpanVerifier | None = None,
        calibrator: ConfidenceCalibrator | None = None,
        linker: Any | None = None,
        job_matcher: JobMatcher | None = None,
        gap_detector: GapDetector | None = None,
        course_port: CourseRecommenderPort | None = None,
        learned_verifier: LearnedSpanVerifier | None = None,
        confidence_hook: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        strict_extractors: bool | None = None,
    ) -> None:
        self.root = Path(root).resolve() if root else repo_root_from(__file__)
        self.config_source: str | None = None
        user_cfg: dict[str, Any]
        if config is None:
            default = self.root / DEFAULT_CONFIG_PATH
            if default.exists():
                user_cfg = load_config(default)
                self.config_source = str(default)
            else:
                user_cfg = {}
        elif isinstance(config, (str, Path)):
            user_cfg = load_config(self.root / config if not Path(config).is_absolute() else config)
            self.config_source = str(config)
        else:
            user_cfg = dict(config)
            self.config_source = "<dict>"
        self.base_config = user_cfg
        self.cfg: dict[str, Any] = deep_merge(DEFAULT_INFERENCE, user_cfg.get("inference", {}) or {})
        unknown = set(self.cfg["stages"]) - set(STAGE_ORDER)
        if unknown:
            raise ValueError(f"Unknown stage names in inference.stages: {sorted(unknown)}")
        self.config_hash = hashlib.sha256(json.dumps({"base": user_cfg, "inference": self.cfg}, sort_keys=True, default=str).encode()).hexdigest()[:16]

        alias_paths = [str(self._abs(p)) for p in self.cfg["taxonomy"]["alias_table_paths"]]
        tax_cfg = user_cfg.get("taxonomy", {}) or {}
        self.linker = linker or build_taxonomy_linker(
            taxonomy_paths=[str(self._abs(p)) for p in tax_cfg.get("paths", [])],
            alias_table_paths=alias_paths,
            fuzzy_threshold=float(tax_cfg.get("fuzzy_threshold", 0.88)),
            esco_path=self._opt_abs(self.cfg["taxonomy"].get("esco_path")),
            onet_path=self._opt_abs(self.cfg["taxonomy"].get("onet_path")),
            lightcast_path=self._opt_abs(self.cfg["taxonomy"].get("lightcast_path")),
        )
        paths = self.cfg["paths"]
        self._merger_injected = merger is not None
        self._alias_paths = alias_paths
        self._no_anchor_merger: CandidateMerger | None = None
        self.merger = merger or CandidateMerger(alias_paths=alias_paths, tuned_config_path=str(self._abs(paths["merger_tuned_config"])))
        self.selector = selector or BoundarySelector(alias_paths=alias_paths, tuned_config_path=str(self._abs(paths["boundary_tuned_config"])))
        self.refiner = refiner or BoundaryRefiner(alias_paths=alias_paths)
        self.verifier = verifier or SpanVerifier()
        cal = user_cfg.get("calibration", {}) or {}
        self.calibrator = calibrator or ConfidenceCalibrator(
            hard_skill_threshold=cal.get("hard_skill_threshold"),
            soft_skill_threshold=cal.get("soft_skill_threshold"),
            dictionary_only_threshold=cal.get("dictionary_only_threshold"),
            multi_source_threshold=cal.get("multi_source_threshold"),
            tuned_config_path=str(self._abs(paths["calibrator_tuned_config"])),
        )
        learned_cfg = self.cfg["verify"]["learned"]
        self.learned_verifier = learned_verifier
        if self.learned_verifier is None and learned_cfg.get("enabled"):
            self.learned_verifier = LearnedSpanVerifier(self._abs(learned_cfg["model_path"]), float(learned_cfg["threshold"]))
        self.confidence_hook = confidence_hook
        # strict_extractors=True: an extractor exception propagates (default: follow continue_on_error -> degrade and continue)
        self._strict_override = strict_extractors

        # lazily resolved extractors (no model weights are touched at construction time)
        self._mistral_injected = mistral
        self._gliner_injected = gliner
        self._dictionary_injected = dictionary
        self._anchor: Any | None = None
        self._anchor_mode: AnchorMode | None = None
        self._anchor_note: str = ""
        self._gliner: Any | None = None
        self._gliner_failed: str | None = None
        self._dictionary: Any | None = None

        port_cfg = self.cfg["courses"]
        self.course_port: CourseRecommenderPort = course_port or build_course_recommender(
            str(port_cfg.get("port", "auto")), self._opt_abs(port_cfg.get("catalog_path"))
        )
        self._build_alias_maps()
        # match / gap compare skills by shared canonical key + this pipeline's alias table (canonical name OR surface meet)
        self.job_matcher = job_matcher or JobMatcher(alias_resolver=self.alias_resolver)
        self.gap_detector = gap_detector or GapDetector(alias_resolver=self.alias_resolver)

    # ------------------------------------------------------------------ construction helpers
    @classmethod
    def from_config(cls, path: str | Path | None = None, **kwargs: Any) -> "SkillLLMV2Pipeline":
        return cls(path, **kwargs)

    def _abs(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    def _opt_abs(self, path: str | Path | None) -> str | None:
        return str(self._abs(path)) if path else None

    def _build_alias_maps(self) -> None:
        """Exact-alias canonicalization (wires ``AliasResolver``) shared by profile, graph, match and courses."""
        aliases: dict[str, str] = {}
        ids: dict[str, str] = {}
        for entry in getattr(self.linker, "entries", []):
            name = entry["canonical_skill_name"]
            for alias in list(entry["aliases"]) + [name]:
                norm = normalize_text(alias)
                aliases.setdefault(norm, name)
                ids.setdefault(norm, entry["canonical_skill_id"])
        self.alias_resolver = AliasResolver(aliases)
        self._alias_ids = ids

    def canonical(self, name: str) -> tuple[str, str, str]:
        """``(canonical_name, canonical_id, key)`` using exact alias evidence only (no fuzzy guessing)."""
        resolved = self.alias_resolver.resolve(name)
        key = normalize_text(resolved)
        canonical_id = self._alias_ids.get(normalize_text(name)) or self._alias_ids.get(key) or key
        return resolved, canonical_id, key

    # ------------------------------------------------------------------ extractor resolution
    def _anchor_cfg(self) -> dict[str, Any]:
        return self.cfg["anchor"]

    def _build_mistral(self, allow_fallback: bool) -> MistralExtractor:
        model_cfg = self.base_config.get("model", {}) or {}
        model_path = model_cfg.get("model_path")
        adapter = model_cfg.get("adapter_path")
        trained = self._abs("artifacts/skillllm_v2/mistral_lora/model")
        if (trained / "adapter_config.json").exists():
            adapter = str(trained)
        return MistralExtractor(
            model_path=self._opt_abs(model_path),
            adapter_path=self._opt_abs(adapter) if adapter and not Path(adapter).is_absolute() else adapter,
            use_4bit=bool(model_cfg.get("use_4bit", True)),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 512)),
            allow_fallback=allow_fallback,
        )

    def resolve_anchor(self, *, disabled: bool = False) -> tuple[Any | None, AnchorMode, str]:
        """Return ``(extractor | None, mode, note)``. Loads model weights at most once, and only for ``mistral``/``auto``."""
        if disabled:
            return None, "none", "anchor disabled for this run (skip=mistral)"
        if self._anchor_mode is not None:
            return self._anchor, self._anchor_mode, self._anchor_note
        cfg = self._anchor_cfg()
        mode = str(cfg.get("mode", "auto")).lower()
        if self._mistral_injected is not None:
            injected = self._mistral_injected
            if hasattr(injected, "load"):
                injected.load()
            loaded = injected.is_model_loaded() if hasattr(injected, "is_model_loaded") else True
            if isinstance(injected, DeterministicStubExtractor) or not loaded:
                self._set_anchor(injected, "stub", "injected anchor extractor has no model loaded: deterministic stub output")
            else:
                self._set_anchor(injected, "model", "injected anchor extractor")
        elif mode == "none":
            self._set_anchor(None, "none", "anchor.mode=none: no anchor (dictionary/GLiNER candidates only)")
        elif mode == "stub":
            self._set_anchor(DeterministicStubExtractor(), "stub", "anchor.mode=stub: deterministic stub anchor (NOT model output)")
        elif mode == "mistral":
            extractor = self._build_mistral(allow_fallback=False)
            extractor.load()  # raises when the model cannot be loaded: explicit request, no silent fallback
            self._set_anchor(extractor, "model", "Mistral LoRA anchor")
        elif mode == "auto":
            extractor = self._build_mistral(allow_fallback=True)
            extractor.load()
            if extractor.is_model_loaded():
                self._set_anchor(extractor, "model", "Mistral LoRA anchor")
            else:
                reason = extractor.load_error or "model weights/path unavailable"
                if cfg.get("allow_stub", True):
                    self._set_anchor(DeterministicStubExtractor(), "stub", f"Mistral not loadable ({reason}); using deterministic stub anchor (NOT model output)")
                else:
                    self._set_anchor(None, "none", f"Mistral not loadable ({reason}); allow_stub=false: no anchor")
        else:
            raise ValueError(f"Unknown anchor.mode: {mode!r} (expected auto | mistral | stub | none)")
        return self._anchor, self._anchor_mode, self._anchor_note  # type: ignore[return-value]

    def _set_anchor(self, extractor: Any | None, mode: AnchorMode, note: str) -> None:
        self._anchor, self._anchor_mode, self._anchor_note = extractor, mode, note

    def _gliner_extractor(self) -> Any | None:
        if self._gliner is not None:
            return self._gliner
        if self._gliner_injected is not None:
            self._gliner = self._gliner_injected
            return self._gliner
        legacy = (self.base_config.get("extractors", {}) or {}).get("gliner", {}) or {}
        cfg = deep_merge(legacy, self.cfg["gliner"])
        self._gliner = GLiNERExtractor(
            model_name=cfg.get("model_name"),
            allow_fallback=bool(cfg.get("allow_fallback", False)),
            threshold=float(cfg.get("threshold", 0.35)),
            device=str(cfg.get("device", "cpu")),
        )
        return self._gliner

    def _gliner_enabled(self) -> bool | str:
        legacy = ((self.base_config.get("extractors", {}) or {}).get("gliner", {}) or {}).get("enabled")
        if legacy is False:  # an explicit legacy `extractors.gliner.enabled: false` still switches GLiNER off
            return False
        return self.cfg["gliner"].get("enabled", "auto")

    def _dictionary_extractor(self) -> Any:
        if self._dictionary is None:
            if self._dictionary_injected is not None:
                self._dictionary = self._dictionary_injected
            else:
                legacy = (self.base_config.get("extractors", {}) or {}).get("dictionary", {}) or {}
                cfg = deep_merge(legacy, self.cfg["dictionary"])
                self._dictionary = DictionaryExtractor(
                    alias_paths=[str(self._abs(p)) for p in cfg.get("alias_paths", [])] or None,
                    fuzzy_threshold=float(cfg.get("fuzzy_threshold", 0.92)),
                )
        return self._dictionary

    # ------------------------------------------------------------------ plumbing
    def _new_trace(self) -> PipelineTrace:
        return PipelineTrace(run_id=uuid.uuid4().hex[:12], config_source=self.config_source, config_hash=self.config_hash)

    def _enabled_stages(self, stages: Iterable[str] | None, skip: Iterable[str]) -> tuple[set[str], set[str]]:
        """Return ``(enabled stage names, skipped component switches)``."""
        skip_set = set(skip or ())
        unknown = skip_set - set(STAGE_ORDER) - set(COMPONENT_SWITCHES)
        if unknown:
            raise ValueError(f"Unknown skip entries: {sorted(unknown)} (stages: {list(STAGE_ORDER)}; components: {list(COMPONENT_SWITCHES)})")
        if stages is None:
            enabled = {name for name in STAGE_ORDER if self.cfg["stages"].get(name, True)}
        else:
            enabled = set(stages)
            bad = enabled - set(STAGE_ORDER)
            if bad:
                raise ValueError(f"Unknown stages: {sorted(bad)} (valid: {list(STAGE_ORDER)})")
        enabled -= skip_set
        missing = REQUIRED_STAGES - enabled
        if missing:
            raise ValueError(f"Stage(s) {sorted(missing)} cannot be skipped")
        return enabled, skip_set & set(COMPONENT_SWITCHES)

    @staticmethod
    def _skipped(trace: PipelineTrace | None, name: str, reason: str, component: str = "") -> StageRecord:
        record = StageRecord(name=name, status="skipped", component=component, notes=[reason])
        if trace is not None:
            trace.add(record)
        return record

    class _Timer:
        """Context manager: times a stage, records failures, appends the record to the trace."""

        def __init__(self, name: str, component: str, trace: PipelineTrace | None, input_count: int | None = None) -> None:
            self.record = StageRecord(name=name, component=component, input_count=input_count)
            self.trace = trace
            self._t0 = 0.0

        def __enter__(self) -> StageRecord:
            self._t0 = time.perf_counter()
            return self.record

        def __exit__(self, exc_type, exc, tb) -> bool:
            self.record.duration_seconds = time.perf_counter() - self._t0
            if exc is not None:
                self.record.status = "failed"
                self.record.error = f"{exc_type.__name__}: {exc}"
            if self.trace is not None:
                self.trace.add(self.record)
            return False

    def _check(self, stage: str, candidates: list[dict[str, Any]]) -> None:
        for candidate in candidates:
            try:
                validate_skill(candidate)
            except Exception as exc:
                raise PipelineStageError(stage, f"invalid candidate {candidate.get('surface_form')!r}: {exc}") from exc

    # ================================================================== stage 1
    def stage_extract(
        self,
        text: str,
        *,
        include_anchor: bool = True,
        include_gliner: bool = True,
        include_dictionary: bool = True,
        trace: PipelineTrace | None = None,
    ) -> ExtractArtifact:
        strict = self._strict_extractors()
        with self._Timer("extract", "anchor+gliner+dictionary", trace) as record:
            attempted: list[str] = []
            failures: dict[str, str] = {}
            anchor_extractor: Any | None = None
            anchor_mode: AnchorMode = "none"
            try:
                anchor_extractor, anchor_mode, anchor_note = self.resolve_anchor(disabled=not include_anchor)
            except Exception as exc:
                # an explicit anchor.mode=mistral request must fail loudly (documented); otherwise a missing/broken anchor degrades
                explicit = str(self._anchor_cfg().get("mode", "auto")).lower() == "mistral" and self._mistral_injected is None
                if strict or explicit:
                    raise
                self._note_extractor_failure(record, trace, failures, "mistral", exc, "anchor could not be resolved")
                anchor_extractor, anchor_mode, anchor_note = None, "none", f"anchor unavailable ({failures['mistral']}): no anchor candidates"
                attempted.append("mistral")
            record.notes.append(anchor_note)
            anchor: list[dict[str, Any]] = []
            if anchor_extractor is not None:
                attempted.append("mistral")
                try:
                    anchor = _tag(list(anchor_extractor.extract(text)), "model" if anchor_mode == "model" else "deterministic_stub")
                except Exception as exc:
                    if strict:
                        raise
                    self._note_extractor_failure(record, trace, failures, "mistral", exc, "anchor extractor raised")
                    anchor, anchor_extractor, anchor_mode = [], None, "none"
                if anchor_mode == "stub":
                    if trace is not None:
                        trace.flag_stub("anchor:deterministic_stub")
                    record.status = "degraded"
            conf_mode = str(self._anchor_cfg().get("confidence", "constant"))
            if anchor and self.confidence_hook is not None:
                anchor = list(self.confidence_hook(text, anchor))
                conf_note = "anchor confidence from confidence_hook"
            elif conf_mode == "logprob":
                conf_note = "anchor confidence: token-logprob confidence is NOT IMPLEMENTED (no hook supplied); constant extractor confidence kept"
                if trace is not None:
                    trace.warn(conf_note)
            else:
                conf_note = "anchor confidence is a constant assigned by the extractor (0.86): calibration is not informative for anchor spans"
            if anchor:
                record.notes.append(conf_note)
            if trace is not None:
                trace.anchor_mode = anchor_mode
                if anchor_mode == "none" and include_anchor:
                    trace.degraded = True

            gliner: list[dict[str, Any]] = []
            gliner_enabled = self._gliner_enabled()
            if include_gliner and gliner_enabled not in (False, "false", "off"):
                gliner = self._run_gliner(text, gliner_enabled, record, trace, attempted=attempted, failures=failures, strict=strict)
            elif include_gliner:
                record.notes.append("gliner disabled by config")

            dictionary: list[dict[str, Any]] = []
            dict_enabled = self.cfg["dictionary"].get("enabled", True)
            if include_dictionary and dict_enabled:
                attempted.append("dictionary")
                try:
                    dictionary = _tag(list(self._dictionary_extractor().extract(text)), "dictionary_rules")
                except Exception as exc:
                    if strict:
                        raise
                    self._note_extractor_failure(record, trace, failures, "dictionary", exc, "dictionary extractor raised")
            elif include_dictionary:
                record.notes.append("dictionary disabled by config")

            candidates = anchor + gliner + dictionary
            self._check("extract", candidates)
            record.output_count = len(candidates)
            record.provenance.update({
                "anchor_mode": anchor_mode,
                "anchor_component": type(anchor_extractor).__name__ if anchor_extractor is not None else None,
                "per_source": {"anchor": len(anchor), "gliner": len(gliner), "dictionary": len(dictionary)},
                "anchor_sources": sorted(ANCHOR_SOURCES),
                "anchor_confidence": "hook" if self.confidence_hook is not None else "constant",
                "extractor_failures": dict(failures),
            })
            if failures and set(attempted) <= set(failures):
                # every extractor that was tried raised: return an EMPTY extraction flagged failed (never fabricated output)
                record.status = "failed"
                record.error = "; ".join(f"{name}: {err}" for name, err in failures.items())
                record.notes.append("all attempted extractors failed: empty extraction")
                if trace is not None:
                    trace.degraded = True
                    trace.warn(f"all extractors failed ({record.error}); extraction is empty")
        return ExtractArtifact(candidates, anchor, gliner, dictionary, anchor_mode, record)

    def _strict_extractors(self) -> bool:
        """``True``: an extractor exception propagates. Default follows ``continue_on_error`` (True -> degrade and continue)."""
        if self._strict_override is not None:
            return bool(self._strict_override)
        return not self.cfg["continue_on_error"]

    @staticmethod
    def _note_extractor_failure(record: StageRecord, trace: PipelineTrace | None, failures: dict[str, str], name: str, exc: BaseException, what: str) -> None:
        """Record a per-extractor failure on the stage record and the trace (status=degraded, error text kept); never silent."""
        failures[name] = f"{type(exc).__name__}: {exc}"
        record.status = "degraded"
        record.notes.append(f"{name} failed ({failures[name]}); {what}: no {name} candidates")
        record.provenance.setdefault("extractor_failures", {})[name] = failures[name]
        if trace is not None:
            trace.warn(f"{name} failed: {failures[name]}")
            trace.degraded = True

    def _run_gliner(
        self,
        text: str,
        enabled: bool | str,
        record: StageRecord,
        trace: PipelineTrace | None,
        *,
        attempted: list[str] | None = None,
        failures: dict[str, str] | None = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        attempted = attempted if attempted is not None else []
        failures = failures if failures is not None else {}
        if self._gliner_failed:
            attempted.append("gliner")
            failures["gliner"] = self._gliner_failed
            record.status = "degraded"
            record.notes.append(f"gliner unavailable ({self._gliner_failed}); no GLiNER candidates")
            record.provenance.setdefault("extractor_failures", {})["gliner"] = self._gliner_failed
            return []
        if enabled == "auto" and self._gliner_injected is None and importlib.util.find_spec("gliner") is None:
            record.notes.append("gliner package not installed (gliner.enabled=auto): GLiNER skipped, no candidates")
            return []
        attempted.append("gliner")
        try:
            extractor = self._gliner_extractor()
            out = list(extractor.extract(text))
        except Exception as exc:  # e.g. HF hub offline: optional component must not break the run
            if strict:
                raise
            self._gliner_failed = f"{type(exc).__name__}: {exc}"
            self._note_extractor_failure(record, trace, failures, "gliner", exc, "gliner raised")
            return []
        loaded = extractor.is_model_loaded() if hasattr(extractor, "is_model_loaded") else True
        if not loaded:
            if out:  # hard-coded fallback candidates (allow_fallback=True): flag as stub
                _tag(out, "fallback_stub")
                for item in out:
                    item["metadata"]["stub"] = True
                record.status = "degraded"
                record.notes.append("gliner model not loaded: FALLBACK STUB candidates (not model output)")
                if trace is not None:
                    trace.flag_stub("gliner:fallback_stub")
            else:
                record.notes.append("gliner model not loaded (package missing?): no GLiNER candidates")
                if enabled is True:
                    record.status = "degraded"
            return out
        return _tag(out, "model")

    # ================================================================== stage 2
    def stage_merge(
        self,
        text: str,
        candidates: list[dict[str, Any]],
        *,
        anchor_mode: str | None = None,
        trace: PipelineTrace | None = None,
    ) -> CandidateArtifact:
        merger, policy = self.merger, "source_aware_add_only"
        if anchor_mode == "none" and not self._merger_injected:
            # The add-only policy gates non-anchor candidates so they can only ADD recall on top of a Raw Mistral anchor.
            # With no anchor at all, the default bars (e.g. dictionary 0.85 > the dictionary's own 0.82 confidence) would
            # reject everything, so run the merger with those gates open and leave filtering to verify/calibrate.
            # Unevaluated degrade-mode heuristic; flagged in the record.
            if self._no_anchor_merger is None:
                self._no_anchor_merger = CandidateMerger(
                    alias_paths=self._alias_paths,
                    config={"min_confidence_new_span": {}, "default_min_confidence_new_span": 0.0, "require_alias_evidence_for_new_span": False},
                )
            merger, policy = self._no_anchor_merger, "no_anchor_open_add_only"
        with self._Timer("merge", type(merger).__name__, trace, len(candidates)) as record:
            if policy != "source_aware_add_only":
                record.notes.append("no anchor: add-only gates opened (unevaluated degrade mode); verify/calibrate still filter")
            merged = merger.merge(candidates)
            self._check("merge", merged)
            anchor_in = sum(1 for c in candidates if is_anchor_candidate(c))
            anchor_out = sum(1 for c in merged if is_anchor_candidate(c))
            record.output_count = len(merged)
            record.provenance = {
                "policy": policy,
                "anchor_sources": sorted(ANCHOR_SOURCES),
                "anchor_candidates_in": anchor_in,
                "anchor_derived_out": anchor_out,
                "non_anchor_in": len(candidates) - anchor_in,
                "non_anchor_out": len(merged) - anchor_out,
                "config": {k: v for k, v in getattr(merger, "config", {}).items()},
            }
        return CandidateArtifact("merge", merged, record)

    # ================================================================== stage 3
    def stage_refine(self, text: str, candidates: list[dict[str, Any]], *, trace: PipelineTrace | None = None) -> CandidateArtifact:
        with self._Timer("refine", f"{type(self.refiner).__name__}+{type(self.selector).__name__}", trace, len(candidates)) as record:
            refined = self.refiner.refine(text, candidates, selector=self.selector)
            self._check("refine", refined)
            edits = Counter((c.get("metadata") or {}).get("boundary_edit", {}).get("edit_type", "none") for c in refined)
            record.output_count = len(refined)
            record.provenance = {"edit_types": dict(edits), "selector_allowed_edit_types": list(self.selector.config.get("allowed_edit_types", []))}
        return CandidateArtifact("refine", refined, record)

    # ================================================================== stage 4
    def stage_verify(self, candidates: list[dict[str, Any]], *, text: str = "", trace: PipelineTrace | None = None) -> CandidateArtifact:
        component = type(self.verifier).__name__ + ("+LearnedSpanVerifier" if self.learned_verifier is not None else "")
        with self._Timer("verify", component, trace, len(candidates)) as record:
            verified = self.verifier.filter(candidates)
            learned_info: dict[str, Any] = {"enabled": self.learned_verifier is not None}
            if self.learned_verifier is not None:
                if self.learned_verifier.available:
                    before = len(verified)
                    verified = self.learned_verifier.filter(text, verified)
                    learned_info.update({"available": True, "dropped": before - len(verified), "threshold": self.learned_verifier.threshold})
                else:
                    learned_info.update({"available": False, "reason": self.learned_verifier.unavailable_reason})
                    record.status = "degraded"
                    record.notes.append(f"learned span verifier unavailable ({self.learned_verifier.unavailable_reason}): rule verifier only")
            self._check("verify", verified)
            kept = {_drop_key(c) for c in verified}
            dropped = [
                {"surface_form": c["surface_form"], "start": c["start"], "end": c["end"], "label": c["label"], "reason": self.verifier.verify(c)["reason"]}
                for c in candidates
                if _drop_key(c) not in kept
            ]
            record.output_count = len(verified)
            record.provenance = {"protects_anchor_sources": sorted(self.verifier.protect_anchor_sources), "dropped": len(dropped), "learned_verifier": learned_info}
        return CandidateArtifact("verify", verified, record, dropped)

    # ================================================================== stage 5
    def stage_calibrate(self, candidates: list[dict[str, Any]], *, trace: PipelineTrace | None = None) -> CandidateArtifact:
        with self._Timer("calibrate", type(self.calibrator).__name__, trace, len(candidates)) as record:
            calibrated = self.calibrator.calibrate(candidates)
            self._check("calibrate", calibrated)
            kept = {_drop_key(c) for c in calibrated}
            dropped = [
                {"surface_form": c["surface_form"], "start": c["start"], "end": c["end"], "label": c["label"], "threshold": self.calibrator.threshold_for(c), "confidence": c["confidence"]}
                for c in candidates
                if _drop_key(c) not in kept
            ]
            record.output_count = len(calibrated)
            record.provenance = {
                "thresholds": {
                    "hard_skill": self.calibrator.hard_skill_threshold,
                    "soft_skill": self.calibrator.soft_skill_threshold,
                    "dictionary_only": self.calibrator.dictionary_only_threshold,
                    "multi_source": self.calibrator.multi_source_threshold,
                },
                "protects_anchor_sources": sorted(self.calibrator.protect_anchor_sources),
                "dropped": len(dropped),
            }
        return CandidateArtifact("calibrate", calibrated, record, dropped)

    # ================================================================== stage 6
    def stage_link(self, candidates: list[dict[str, Any]], *, trace: PipelineTrace | None = None) -> CandidateArtifact:
        """Attach-only taxonomy linking: a skill is never removed or re-spanned, whatever the link confidence."""
        min_conf = float(self.cfg["taxonomy"].get("attach_min_confidence", 0.0))
        with self._Timer("link", type(self.linker).__name__, trace, len(candidates)) as record:
            linked = self.linker.link_candidates(candidates)
            if len(linked) != len(candidates):
                raise PipelineStageError("link", f"linking must be attach-only but changed the candidate count {len(candidates)} -> {len(linked)}")
            applied = suppressed = 0
            output: list[dict[str, Any]] = []
            for before, after in zip(candidates, linked):
                item = dict(after)
                link = (after.get("metadata") or {}).get("taxonomy_link")
                if link is not None:
                    item["metadata"] = dict(after["metadata"])
                    item["metadata"]["taxonomy_link"] = dict(link)
                    if float(link.get("link_confidence", 0.0)) >= min_conf:
                        item["metadata"]["taxonomy_link"]["applied"] = True
                        applied += 1
                    else:  # low-confidence link: keep it as a suggestion, do not overwrite the candidate's canonical fields
                        item["canonical_skill_id"] = before.get("canonical_skill_id")
                        item["canonical_skill_name"] = before.get("canonical_skill_name")
                        item["metadata"]["taxonomy_link"]["applied"] = False
                        suppressed += 1
                output.append(item)
            self._check("link", output)
            methods = Counter((c.get("metadata") or {}).get("taxonomy_link", {}).get("link_method", "unlinked") for c in output)
            record.output_count = len(output)
            record.provenance = {
                "attach_min_confidence": min_conf,
                "links_applied": applied,
                "links_suppressed_low_confidence": suppressed,
                "link_methods": dict(methods),
                "removed_candidates": 0,
            }
        return CandidateArtifact("link", output, record)

    # ================================================================== stage 7
    def stage_gate(
        self,
        anchor: list[dict[str, Any]],
        wrapper_candidates: list[dict[str, Any]],
        *,
        anchor_mode: AnchorMode,
        candidate_metrics: Mapping[str, float] | None = None,
        anchor_metrics: Mapping[str, float] | None = None,
        trace: PipelineTrace | None = None,
    ) -> GateArtifact:
        """Safe-wrapper non-regression behaviour at inference time.

        The floors of ``eval.non_regression_gate`` are defined on gold-derived
        metrics, so a single request cannot recompute them. The decision is
        therefore taken, in priority order, from:

        1. explicit ``candidate_metrics``/``anchor_metrics`` -> ``evaluate_gate`` (same code as the offline gate);
        2. ``gate.mode`` forcing ``wrapper``/``anchor``;
        3. the persisted offline decision (``gate.status_path``) for a real-model anchor;
        4. ``gate.on_missing_status`` when a real-model anchor has no offline evidence.

        Then a structural guard (``gate.min_anchor_retention``) rejects wrapper
        output that dropped/shifted too many anchor spans. On fallback the
        selected extraction is the untouched anchor; the wrapper's enriched
        candidates are still returned for downstream stages.
        """
        from skillllm_v2.eval.non_regression_gate import evaluate_gate, load_guardrails

        gate_cfg = self.cfg["gate"]
        with self._Timer("gate", "non_regression_gate", trace, len(wrapper_candidates)) as record:
            decision, reason, basis, detail = "accept", "", "not_applicable", {}
            if anchor_mode == "none":
                reason = "no anchor available (anchor_mode=none): nothing to fall back to; gate not applicable"
            elif candidate_metrics is not None and anchor_metrics is not None:
                guardrails = load_guardrails(str(self._abs(gate_cfg["guardrails_path"])))
                result = evaluate_gate("inference_wrapper", dict(candidate_metrics), dict(anchor_metrics), guardrails)
                decision, reason, basis, detail = result.decision, result.reason, "metrics", result.to_dict()
            elif gate_cfg["mode"] == "wrapper":
                reason, basis = "gate.mode=wrapper: wrapper output forced", "forced_wrapper"
            elif gate_cfg["mode"] == "anchor":
                decision, reason, basis = "fallback_to_anchor", "gate.mode=anchor: untouched anchor forced", "forced_anchor"
            elif anchor_mode == "stub":
                reason = "anchor is a deterministic stub: the offline gate (measured on the real Mistral anchor) does not transfer; structural guard only"
                basis = "structural_only"
            else:
                status_path = self._abs(gate_cfg["status_path"])
                status = read_json(status_path, None)
                if not status:
                    decision = str(gate_cfg["on_missing_status"])
                    reason = f"no offline gate evidence at {gate_cfg['status_path']}; on_missing_status={decision}"
                    basis = "missing_status"
                else:
                    decision = str(status.get("final_decision", "fallback_to_anchor"))
                    final_variant = status.get("final_variant")
                    gate_result = next((g for g in status.get("gate_results", []) if g.get("variant") == final_variant), {})
                    reason = gate_result.get("reason") or f"offline gate final_decision={decision}"
                    basis = "offline_artifact"
                    detail = {"status_path": str(gate_cfg["status_path"]), "final_variant": final_variant, "gate_result": gate_result,
                              "extraction_output_policy": status.get("extraction_output_policy")}
            if decision not in {"accept", "fallback_to_anchor"}:
                raise PipelineStageError("gate", f"invalid gate decision {decision!r}")

            anchor_spans = {_span_key(c) for c in anchor if c.get("start") is not None and c.get("end") is not None}
            retention: float | None = None
            if anchor_spans:
                wrapper_spans = {_span_key(c) for c in wrapper_candidates}
                retention = len(anchor_spans & wrapper_spans) / len(anchor_spans)
                floor = float(gate_cfg["min_anchor_retention"])
                if decision == "accept" and anchor_mode != "none" and retention < floor:
                    decision = "fallback_to_anchor"
                    basis = f"{basis}+structural_guard"
                    reason = f"wrapper kept only {retention:.2f} of anchor spans (< min_anchor_retention {floor:.2f}); {reason}"

            selected = copy.deepcopy(anchor) if decision == "fallback_to_anchor" else wrapper_candidates
            record.output_count = len(selected)
            record.provenance = {"decision": decision, "basis": basis, "anchor_mode": anchor_mode, "anchor_retention": retention, "mode": gate_cfg["mode"]}
            if decision == "fallback_to_anchor":
                record.notes.append("extraction output = untouched anchor candidates; wrapper-enriched candidates still returned for downstream stages")
            if anchor_mode == "stub":
                record.notes.append("anchor is a deterministic stub: fallback target is stub output, not model output")
        return GateArtifact(decision, reason, basis, selected, copy.deepcopy(anchor), wrapper_candidates, retention, anchor_mode, record, detail)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ stages 1-7 driver
    def _run_extraction(
        self,
        text: str,
        enabled: set[str],
        skipped_components: set[str],
        *,
        apply_gate: bool,
        trace: PipelineTrace,
        candidate_metrics: Mapping[str, float] | None = None,
        anchor_metrics: Mapping[str, float] | None = None,
    ) -> ExtractionResult:
        extract = self.stage_extract(
            text,
            include_anchor="mistral" not in skipped_components,
            include_gliner="gliner" not in skipped_components,
            include_dictionary="dictionary" not in skipped_components,
            trace=trace,
        )
        current = extract.candidates
        arts: dict[str, CandidateArtifact] = {}
        steps: list[tuple[str, Callable[[list[dict[str, Any]]], CandidateArtifact]]] = [
            ("merge", lambda c: self.stage_merge(text, c, anchor_mode=trace.anchor_mode, trace=trace)),
            ("refine", lambda c: self.stage_refine(text, c, trace=trace)),
            ("verify", lambda c: self.stage_verify(c, text=text, trace=trace)),
            ("calibrate", lambda c: self.stage_calibrate(c, trace=trace)),
            ("link", lambda c: self.stage_link(c, trace=trace)),
        ]
        for name, step in steps:
            if name not in enabled:
                if name == "merge":  # pass-through must still be a valid candidate list
                    self._check("merge", current)
                record = self._skipped(trace, name, "stage skipped: candidates passed through unchanged")
                arts[name] = CandidateArtifact(name, current, record)
                continue
            try:
                arts[name] = step(current)
            except Exception as exc:
                if name == "merge" or not self.cfg["continue_on_error"]:
                    raise
                record = trace.get(name) or StageRecord(name=name, status="failed", error=f"{type(exc).__name__}: {exc}")
                trace.warn(f"stage {name} failed ({record.error}); candidates passed through unchanged")
                arts[name] = CandidateArtifact(name, current, record)
            current = arts[name].candidates

        gate: GateArtifact | None = None
        anchor_mode = trace.anchor_mode
        if apply_gate and "gate" in enabled:
            try:
                gate = self.stage_gate(extract.anchor, current, anchor_mode=anchor_mode, candidate_metrics=candidate_metrics, anchor_metrics=anchor_metrics, trace=trace)
            except Exception as exc:
                if not self.cfg["continue_on_error"]:
                    raise
                trace.warn(f"stage gate failed ({type(exc).__name__}: {exc}); wrapper output used without gate")
        elif apply_gate:
            self._skipped(trace, "gate", "stage skipped: wrapper output used without non-regression gate", "non_regression_gate")
        skills = gate.selected if gate is not None else current
        return ExtractionResult(text, skills, current, extract, arts["merge"], arts["refine"], arts["verify"], arts["calibrate"], arts["link"], gate, trace)

    # ================================================================== public: extraction
    def extract_candidates(
        self,
        text: str,
        *,
        include_anchor: bool = True,
        include_gliner: bool = True,
        include_dictionary: bool = True,
        stages: Iterable[str] | None = None,
        skip: Iterable[str] = (),
    ) -> ExtractionResult:
        """Stages 1-6 (no gate): exactly the legacy ``SkillLLMv2Pipeline.extract`` behaviour, with a trace."""
        trace = self._new_trace()
        t0 = time.perf_counter()
        base = {"extract", "merge", "refine", "verify", "calibrate", "link"}
        enabled, skipped = self._enabled_stages(stages if stages is not None else base, skip)
        skipped = set(skipped)
        for flag, name in ((include_anchor, "mistral"), (include_gliner, "gliner"), (include_dictionary, "dictionary")):
            if not flag:
                skipped.add(name)
        result = self._run_extraction(text, enabled, skipped, apply_gate=False, trace=trace)
        trace.total_seconds = time.perf_counter() - t0
        return result

    def run_extraction_only(
        self,
        text: str,
        *,
        stages: Iterable[str] | None = None,
        skip: Iterable[str] = (),
        candidate_metrics: Mapping[str, float] | None = None,
        anchor_metrics: Mapping[str, float] | None = None,
    ) -> ExtractionResult:
        """Stages 1-7 (extract .. gate). ``result.skills`` is the gated extraction output."""
        trace = self._new_trace()
        t0 = time.perf_counter()
        base = {"extract", "merge", "refine", "verify", "calibrate", "link", "gate"}
        if stages is None:
            stages = {name for name in STAGE_ORDER if self.cfg["stages"].get(name, True)} & base
        enabled, skipped = self._enabled_stages(stages, skip)
        result = self._run_extraction(text, enabled, skipped, apply_gate=True, trace=trace, candidate_metrics=candidate_metrics, anchor_metrics=anchor_metrics)
        trace.total_seconds = time.perf_counter() - t0
        return result

    # ================================================================== job input
    def _prepare_jobs(
        self,
        job: str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
        enabled: set[str],
        skipped_components: set[str],
        trace: PipelineTrace,
    ) -> list[dict[str, Any]]:
        if job is None:
            return []
        if isinstance(job, str):
            if not job.strip():
                return []
            result = self._extract_job_text(job, enabled, skipped_components, trace)
            return [result]
        rows = [dict(job)] if isinstance(job, Mapping) else [dict(item) for item in job]
        return [self._canonicalize_job(row, index) for index, row in enumerate(rows)]

    def _extract_job_text(self, text: str, enabled: set[str], skipped_components: set[str], trace: PipelineTrace) -> dict[str, Any]:
        """Raw job text -> job dict via stages 1-6 (HARD -> required, SOFT -> preferred), as the course pipeline does."""
        t0 = time.perf_counter()
        sub = self._new_trace()
        sub.anchor_mode = trace.anchor_mode
        sub_enabled = enabled & {"extract", "merge", "refine", "verify", "calibrate", "link"} | {"extract"}
        extraction = self._run_extraction(text, sub_enabled, skipped_components, apply_gate=False, trace=sub)
        skills = extraction.enriched_skills
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "Target Role")
        job = {
            "job_id": "target_role",
            "title": first_line[:120],
            "description": text,
            "required_skills": [c.get("canonical_skill_name") or c["surface_form"] for c in skills if c.get("label") == "HARD_SKILL"],
            "preferred_skills": [c.get("canonical_skill_name") or c["surface_form"] for c in skills if c.get("label") == "SOFT_SKILL"],
        }
        record = StageRecord(
            name=JOB_EXTRACT_STAGE,
            status="degraded" if sub.degraded else "ok",
            component="stages 1-6 on job text",
            duration_seconds=time.perf_counter() - t0,
            input_count=1,
            output_count=len(skills),
            notes=["job given as raw text: required=HARD_SKILL spans, preferred=SOFT_SKILL spans; semantic fit fields are 0.0 (not provided)"],
            provenance={"sub_trace_stages": sub.stage_names, "stub_components": list(sub.stub_components)},
        )
        sub_extract = sub.get("extract")
        if sub_extract is not None and sub_extract.provenance.get("extractor_failures"):
            record.status = "failed" if sub_extract.status == "failed" else "degraded"
            record.provenance["extractor_failures"] = dict(sub_extract.provenance["extractor_failures"])
            record.error = sub_extract.error
        for warning in sub.warnings:
            trace.warn(f"job_text: {warning}")
        trace.add(record)
        for component in sub.stub_components:
            trace.flag_stub(f"job_text:{component}")
        job = self._canonicalize_job(job, 0)
        job["input_form"] = "text_extracted"
        return job

    def _canonicalize_job(self, job: dict[str, Any], index: int) -> dict[str, Any]:
        out = dict(job)
        out["job_id"] = str(job.get("job_id") or f"job-{index}")
        out.setdefault("title", out["job_id"])
        out.setdefault("input_form", "structured")
        raw_required = job.get("required_skills", [])
        raw_preferred = job.get("preferred_skills", [])
        out["original_required_skills"] = list(raw_required)
        out["original_preferred_skills"] = list(raw_preferred)
        out["required_skills"] = list(dict.fromkeys(self.canonical(n)[0] for n in _skill_names(raw_required)))
        out["preferred_skills"] = list(dict.fromkeys(self.canonical(n)[0] for n in _skill_names(raw_preferred)))
        return out

    # ================================================================== user skill helpers
    def _user_skill_table(self, skills: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """key -> best skill dict (highest confidence) with the canonical name/id used by every downstream stage."""
        table: dict[str, dict[str, Any]] = {}
        for skill in skills:
            name, canonical_id, key = self.canonical(skill.get("canonical_skill_name") or skill["surface_form"])
            if not key:
                continue
            entry = table.get(key)
            confidence = float(skill.get("confidence", 0.0))
            if entry is None:
                table[key] = {
                    "surface_form": skill["surface_form"],
                    "label": skill.get("label", "HARD_SKILL"),
                    "canonical_skill_name": name,
                    "canonical_skill_id": skill.get("canonical_skill_id") or canonical_id,
                    "confidence": confidence,
                    "evidence": skill.get("evidence") or skill["surface_form"],
                    "source": skill.get("source", "unknown"),
                    "extractor_mode": (skill.get("metadata") or {}).get("extractor_mode"),
                }
            elif confidence > entry["confidence"]:
                entry["confidence"] = confidence
        return table

    def _matching_profile(self, user_id: str, table: dict[str, dict[str, Any]]) -> dict[str, Any]:
        threshold = float(self.cfg["matching"]["uncertain_confidence_below"])
        names = [entry["canonical_skill_name"] for entry in table.values()]
        return {
            "user_id": user_id,
            "skills": names,
            # canonical id + canonical name + surface form per skill, so jobs/catalogs written with either spelling meet
            "skill_entries": [
                {
                    "canonical_skill_id": entry["canonical_skill_id"],
                    "canonical_skill_name": entry["canonical_skill_name"],
                    "surface_form": entry["surface_form"],
                    "confidence": entry["confidence"],
                }
                for entry in table.values()
            ],
            "uncertain_skills": [entry["canonical_skill_name"] for entry in table.values() if entry["confidence"] < threshold],
        }

    def _choose_target(self, jobs: list[dict[str, Any]], profile: dict[str, Any], target_job_id: str | None) -> tuple[dict[str, Any] | None, str]:
        if not jobs:
            return None, "no_job"
        if target_job_id is not None:
            for job in jobs:
                if job["job_id"] == target_job_id:
                    return job, "explicit"
            raise ValueError(f"target_job_id {target_job_id!r} not among supplied jobs {[j['job_id'] for j in jobs]}")
        if len(jobs) == 1:
            return jobs[0], "single_job"
        scored = [(self.job_matcher.score(profile, job)["match_score"], -i, job) for i, job in enumerate(jobs)]
        return max(scored, key=lambda item: (item[0], item[1]))[2], "best_match_score"

    # ================================================================== stage 8
    def stage_profile(
        self,
        profile_text: str,
        skills: list[dict[str, Any]],
        *,
        user_id: str,
        jobs: list[dict[str, Any]] | None = None,
        target_job: dict[str, Any] | None = None,
        trace: PipelineTrace | None = None,
    ) -> ProfileArtifact:
        pcfg = self.cfg["profile"]
        with self._Timer("profile", "summarize_profile+skill_memory+compact_context+ContextRetriever+ContextualRetriever", trace, len(skills)) as record:
            table = self._user_skill_table(skills)
            # one pass: build_cached_context_packet -> compact_context -> summarize_profile + build_skill_memory
            packet = build_cached_context_packet(
                profile_text,
                skills,
                job_target=(target_job or {}).get("title"),
                taxonomy_aliases={entry["surface_form"]: entry["canonical_skill_name"] for entry in table.values()},
                user_id=user_id,
            )
            stable = packet["payload"]["stable_cag_context"]
            summary, memory = stable["profile_summary"], stable["skill_memory"]

            retrieved_skills: list[dict[str, Any]] = []
            if target_job is not None:
                query = " ".join([str(target_job.get("title", ""))] + _skill_names(target_job.get("required_skills")) + _skill_names(target_job.get("preferred_skills")))
                retrieved_skills = ContextRetriever(packet).retrieve_skills(query, top_k=int(pcfg["top_k_skills"]))
            retrieved_jobs: list[dict[str, Any]] = []
            if jobs:
                retriever = ContextualRetriever(jobs, context_fields=["job_id", "title", "description", "required_skills", "preferred_skills"])
                query = " ".join(entry["canonical_skill_name"] for entry in table.values())
                retrieved_jobs = retriever.search(query, top_k=int(pcfg["top_k_jobs"]))
            record.output_count = len(summary.get("skills", []))
            record.provenance = {
                "content_hash": packet["content_hash"],
                "skill_facts": len(summary.get("skills", [])),
                "memory_skill_count": memory.get("skill_count"),
                "conflict_notes": summary.get("conflict_notes", []),
                "retrieved_skills": len(retrieved_skills),
                "retrieved_jobs": len(retrieved_jobs),
                "skills_from": "extraction spans only (no inferred skills)",
            }
        return ProfileArtifact(summary, memory, packet, retrieved_skills, retrieved_jobs, record)

    # ================================================================== stage 9
    def stage_graph(
        self,
        user_id: str,
        skills: list[dict[str, Any]],
        jobs: list[dict[str, Any]],
        target_job: dict[str, Any] | None,
        *,
        trace: PipelineTrace | None = None,
    ) -> GraphArtifact:
        with self._Timer("graph", "SkillGraphBuilder+SkillGraphRetriever+explain_skill_path", trace, len(skills)) as record:
            builder = SkillGraphBuilder()
            user_node = f"user:{user_id}"
            builder.add_node(user_node, NodeType.USER, name=user_id)
            table = self._user_skill_table(skills)

            def skill_node(key: str, name: str) -> str:
                node = f"skill:{key}"
                builder.add_node(node, NodeType.SKILL, name=name)
                return node

            for key, entry in table.items():
                builder.add_user_skill(
                    user_node,
                    skill_node(graph_key(entry["canonical_skill_name"]) or key, entry["canonical_skill_name"]),
                    confidence=entry["confidence"],
                    evidence_text=str(entry.get("evidence") or ""),
                    source=str(entry.get("source") or "unknown"),
                )
            job_node: str | None = None
            for job in jobs:
                node = f"job:{job['job_id']}"
                builder.add_node(node, NodeType.JOB, name=job.get("title", job["job_id"]))
                for name in job["required_skills"]:
                    builder.add_job_requirement(node, skill_node(graph_key(name), name), required=True)
                for name in job["preferred_skills"]:
                    builder.add_job_requirement(node, skill_node(graph_key(name), name), required=False)
                if target_job is not None and job["job_id"] == target_job["job_id"]:
                    job_node = node

            retriever = SkillGraphRetriever(builder.graph)
            user_skills = retriever.skills_for_user(user_node) if builder.graph.has_node(user_node) else []
            required = retriever.required_skills_for_job(job_node) if job_node else []
            missing = [node for node in required if node not in set(user_skills)]
            for node in missing:
                builder.add_edge(user_node, node, EdgeType.MISSING_SKILL)

            explanations: list[dict[str, Any]] = []
            if job_node:
                max_hops = int(self.cfg["graph"]["max_hops"])
                for node in [n for n in required if n in set(user_skills)] + missing:
                    kind = "missing" if node in missing else "has"
                    sub = builder.graph.subgraph([user_node, node, job_node])  # directed: hops keep their stored direction / relation
                    path = explain_skill_path(sub, user_node, job_node, max_hops=max_hops)
                    names = [builder.graph.nodes[n].get("name", n) for n in path["path"]]
                    explanations.append({"skill": builder.graph.nodes[node].get("name", node), "kind": kind, **path, "path_names": names,
                                         "explanation": " -> ".join(names) if names else path["explanation"]})
            record.output_count = builder.graph.number_of_nodes()
            record.provenance = {
                "nodes": builder.graph.number_of_nodes(),
                "edges": builder.graph.number_of_edges(),
                "target_job": job_node,
                "retriever_calls": ["skills_for_user", "required_skills_for_job"],
                "missing_edges_added": len(missing),
            }
            catalog_courses, store_info = self._graph_store_lookup([builder.graph.nodes[n].get("name", n) for n in missing])
            record.provenance["graph_backend"] = store_info
            if store_info["fell_back"]:
                record.notes.append(f"graph store fell back to local: {store_info['error']}")
        return GraphArtifact(user_node, job_node, [builder.graph.nodes[n].get("name", n) for n in user_skills],
                             [builder.graph.nodes[n].get("name", n) for n in required], [builder.graph.nodes[n].get("name", n) for n in missing],
                             explanations, builder.graph.number_of_nodes(), builder.graph.number_of_edges(), record, builder.graph,
                             catalog_courses=catalog_courses)

    def _graph_store_lookup(self, missing_names: list[str]) -> tuple[dict[str, list[str]], dict[str, Any]]:
        """Optional Neo4j/local catalogue lookup for the missing skills (``graph.backend`` != ``local``).

        Never raises: a missing driver, missing credentials or an unreachable database is recorded in the returned
        info (``backend`` that answered, ``fell_back``, redacted ``error``) and the stage carries on. Nothing about
        the user or the job is written to the store.
        """
        requested = str(self.cfg["graph"].get("backend", "local")).lower()
        info: dict[str, Any] = {"requested": requested, "backend": "local", "fell_back": False, "error": None, "queries": 0}
        courses: dict[str, list[str]] = {}
        if requested == "local":
            return courses, info
        try:
            from skillllm_v2.graph.neo4j_adapters import CuratedSkillIndex

            store = self._graph_store(requested)
            health = store.health()
            info.update(backend=health.backend, fell_back=health.fell_back, error=health.error)
            index = CuratedSkillIndex.load()
            for name in missing_names:
                res = store.courses_teaching(index.resolve(name)[0])
                info["queries"] += 1
                info["backend"] = res.backend
                info["fell_back"] = info["fell_back"] or res.fell_back
                info["error"] = info["error"] or res.error
                if res.ok and res.value:
                    courses[name] = [row["course_id"].split(":", 1)[-1] for row in res.value]
        except Exception as exc:  # noqa: BLE001 - the graph store is optional; the stage must not fail because of it
            from skillllm_v2.graph.store_settings import redact_error

            info.update(backend="local", fell_back=True, error=redact_error(exc))
        return courses, info

    def _graph_store(self, backend: str) -> Any:
        cached = getattr(self, "_graph_store_cache", None)
        if cached is None or cached[0] != backend:
            from skillllm_v2.graph.neo4j_adapters import curated_local_loader
            from skillllm_v2.graph.store_factory import get_graph_store

            if cached is not None:
                cached[1].close()
            cached = (backend, get_graph_store(self.cfg["graph"], backend, local_loader=curated_local_loader(self.root)))
            self._graph_store_cache = cached
        return cached[1]

    def close_graph_store(self) -> None:
        """Close the Neo4j connection pool (if ``graph.backend`` opened one)."""
        cached = getattr(self, "_graph_store_cache", None)
        if cached is not None:
            cached[1].close()
            self._graph_store_cache = None

    # ================================================================== stage 10
    def stage_match(
        self,
        user_id: str,
        skills: list[dict[str, Any]],
        jobs: list[dict[str, Any]],
        *,
        target_job_id: str | None = None,
        trace: PipelineTrace | None = None,
    ) -> MatchArtifact:
        with self._Timer("match", f"{type(self.job_matcher).__name__}+{type(self.gap_detector).__name__}", trace, len(jobs)) as record:
            table = self._user_skill_table(skills)
            profile = self._matching_profile(user_id, table)
            matches = [self.job_matcher.score(profile, job) for job in jobs]
            target, how = self._choose_target(jobs, profile, target_job_id)
            job_match = next((m for m in matches if target and m["job_id"] == target["job_id"]), None)
            gaps = self.gap_detector.detect(profile, target) if target else []
            gap_vector = self.gap_detector.detect_vector(profile, target).to_dict() if target else None
            record.output_count = len(matches)
            fit_fields = [j.get("job_id") for j in jobs if any(k in j for k in ("semantic_role_fit", "experience_level_fit", "user_preference_fit"))]
            record.provenance = {
                "target_selection": how,
                "target_job_id": target["job_id"] if target else None,
                "gap_count": len(gaps),
                "uncertain_confidence_below": self.cfg["matching"]["uncertain_confidence_below"],
                "fit_fields_supplied_for_jobs": fit_fields,
                "skill_canonicalization": "exact alias evidence (AliasResolver) over the taxonomy linker's entries; match/gap compare shared skill keys (match.skill_keys: casefold, symbol-preserving) over canonical id + canonical name + surface form",
            }
            if jobs and not fit_fields:
                record.notes.append("no semantic/experience/preference fit fields on the job: those score terms are 0.0")
        return MatchArtifact(matches, target, job_match, gaps, gap_vector, profile, record)

    # ================================================================== stage 11
    def stage_courses(
        self,
        user_id: str,
        skills: list[dict[str, Any]],
        match: MatchArtifact,
        *,
        top_k: int | None = None,
        trace: PipelineTrace | None = None,
    ) -> CourseArtifact:
        """Course recommendation THROUGH ``CourseRecommenderPort`` ONLY (no direct import of course/recommend code)."""
        with self._Timer("courses", getattr(self.course_port, "name", type(self.course_port).__name__), trace, len(match.gaps)) as record:
            target = match.target_job
            if target is None:
                record.status = "skipped"
                record.notes.append("no target job: nothing to recommend courses for")
                return CourseArtifact(self.course_port.name, False, [], None, "no target job", record)
            table = self._user_skill_table(skills)
            user_profile = {
                "user_id": user_id,
                "skills": [
                    {k: v for k, v in entry.items() if k != "extractor_mode"} | {"name": entry["canonical_skill_name"]}
                    for entry in table.values()
                ],
                "target_roles": [target.get("title", target["job_id"])],
            }

            def port_skill(name: str, importance: float) -> dict[str, Any]:
                canon, canon_id, _ = self.canonical(name)
                return {"skill_name": canon, "canonical_skill_name": canon, "canonical_skill_id": canon_id, "importance": importance}

            port_job = {
                "job_id": target["job_id"],
                "title": target.get("title", target["job_id"]),
                "required_skills": [port_skill(n, 1.0) for n in target["required_skills"]],
                "preferred_skills": [port_skill(n, 0.7) for n in target["preferred_skills"]],
            }
            request = CourseRecommendationRequest(user_id, user_profile, port_job, match.gaps, int(top_k or self.cfg["courses"]["top_k"]))
            response = self.course_port.recommend(request)
            record.output_count = len(response.recommendations)
            record.provenance = {"port": self.course_port.name, "implementation": response.implementation, "available": response.available, "note": response.note}
            if not response.available:
                record.status = "degraded"
                record.notes.append(response.note)
                if trace is not None:
                    trace.warn(f"course recommender unavailable: {response.note}")
        return CourseArtifact(self.course_port.name, response.available, response.recommendations, response.learning_path, response.note, record)

    def _attach_courses_to_graph(self, graph: GraphArtifact, courses: CourseArtifact) -> None:
        """Add recommended courses as TEACHES edges and read them back with SkillGraphRetriever.courses_for_skill."""
        if graph.graph is None or not courses.recommendations:
            return
        g = graph.graph
        for rec in courses.recommendations:
            cid = f"course:{rec.get('course_id') or rec.get('course_name')}"
            g.add_node(cid, type=NodeType.COURSE.value, name=rec.get("course_name") or rec.get("title") or cid)
            for name in rec.get("covered_missing_skills") or rec.get("covers_skills") or []:
                node = f"skill:{graph_key(self.canonical(str(name))[0])}"
                if not g.has_node(node):
                    g.add_node(node, type=NodeType.SKILL.value, name=self.canonical(str(name))[0])
                g.add_edge(cid, node, type=EdgeType.TEACHES.value)
        retriever = SkillGraphRetriever(g)
        paths = []
        for node in graph.graph_missing_skills:
            skill_node = f"skill:{graph_key(node)}"
            if g.has_node(skill_node):
                teachers = [g.nodes[c].get("name", c) for c in retriever.courses_for_skill(skill_node)]
                if teachers:
                    paths.append({"missing_skill": node, "courses": teachers, "path": f"{graph.user_node} -> {node} -> {graph.job_node} -> course ({', '.join(teachers)})"})
        graph.course_paths = paths
        graph.node_count, graph.edge_count = g.number_of_nodes(), g.number_of_edges()
        graph.record.provenance["course_edges_attached"] = sum(len(p["courses"]) for p in paths)

    # ================================================================== stage 12
    def stage_explain(
        self,
        profile: ProfileArtifact | None,
        skills: list[dict[str, Any]],
        match: MatchArtifact,
        courses: CourseArtifact | None,
        graph: GraphArtifact | None,
        extraction: ExtractionResult,
        *,
        trace: PipelineTrace | None = None,
    ) -> ExplanationArtifact:
        with self._Timer("explain", "generate_recommendation_explanation", trace, len(match.gaps)) as record:
            if profile is not None:
                user_profile = profile.profile_summary
            else:
                user_profile = {"skills": [{"surface_form": s["surface_form"], "evidence": [{"text": s.get("evidence") or s["surface_form"]}]} for s in skills]}
            explanation = generate_recommendation_explanation(user_profile, match.job_match, match.gaps, courses.recommendations if courses else [])
            gate = extraction.gate
            limitations = list(explanation.get("limitations", []))
            if extraction.trace.anchor_mode == "stub":
                limitations.append("Extraction anchor was a deterministic STUB, not the Mistral model: skills are alias matches, not model output.")
            if extraction.trace.anchor_mode == "none":
                limitations.append("No anchor extractor was available: skills come from GLiNER/dictionary candidates only.")
            if gate is not None and gate.fell_back:
                limitations.append(f"Wrapper output was rejected by the non-regression gate ({gate.reason}); reported extraction is the untouched anchor.")
            if courses is not None and not courses.available:
                limitations.append(courses.note)
            explanation.update(
                {
                    "target_job_id": (match.target_job or {}).get("job_id"),
                    "extraction_policy": {
                        "gate_decision": gate.decision if gate else "not_run",
                        "gate_basis": gate.basis if gate else None,
                        "anchor_mode": extraction.trace.anchor_mode,
                        "stub_components": list(extraction.trace.stub_components),
                    },
                    "graph_paths": [e["explanation"] for e in (graph.explanations if graph else [])] + [p["path"] for p in (graph.course_paths if graph else [])],
                    "course_recommender": {"port": courses.port, "available": courses.available} if courses else None,
                    "limitations": limitations,
                }
            )
            record.output_count = len(explanation.get("evidence", []))
            record.provenance = {"limitations": len(limitations)}
        return ExplanationArtifact(explanation, record)

    # ================================================================== public: full run
    def run(
        self,
        profile_text: str | Mapping[str, Any],
        job_text: str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        *,
        stages: Iterable[str] | None = None,
        skip: Iterable[str] = (),
        user_id: str | None = None,
        target_job_id: str | None = None,
        top_k: int | None = None,
    ) -> PipelineResult:
        """Run stages 1-12.

        ``profile_text``: raw text, or a mapping with ``text`` (+ optional ``user_id``).
        ``job_text``: raw job text, one structured job dict, a list of job dicts, or ``None``.
        ``stages``: run only these stages (``extract`` is mandatory). ``skip``: stage names and/or
        component switches (``mistral``, ``gliner``, ``dictionary``) to switch off. Skipped
        extraction stages pass candidates through; skipped downstream stages yield ``None``.
        """
        text, uid = self._unpack_profile(profile_text, user_id)
        trace = self._new_trace()
        t0 = time.perf_counter()
        enabled, skipped_components = self._enabled_stages(stages, skip)

        extraction = self._run_extraction(text, enabled & ({"extract", "merge", "refine", "verify", "calibrate", "link", "gate"}), skipped_components, apply_gate=True, trace=trace)
        # _run_extraction records skipped=gate itself; downstream-stage names below are added in order
        downstream_from = self.cfg["gate"]["downstream_from"]
        skills = extraction.skills if downstream_from == "selected" else extraction.enriched_skills
        jobs = self._prepare_jobs(job_text, enabled, skipped_components, trace)
        profile_for_choice = self._matching_profile(uid, self._user_skill_table(skills))
        target, target_how = self._choose_target(jobs, profile_for_choice, target_job_id)

        def optional(name: str, fn: Callable[[], Any], reason_if_unavailable: str | None) -> Any:
            if name not in enabled:
                self._skipped(trace, name, "stage switched off")
                return None
            if reason_if_unavailable:
                self._skipped(trace, name, reason_if_unavailable)
                return None
            try:
                return fn()
            except Exception as exc:
                if not self.cfg["continue_on_error"]:
                    raise
                trace.warn(f"stage {name} failed ({type(exc).__name__}: {exc}); continuing without it")
                if trace.get(name) is None:
                    trace.add(StageRecord(name=name, status="failed", error=f"{type(exc).__name__}: {exc}"))
                return None

        profile = optional("profile", lambda: self.stage_profile(text, skills, user_id=uid, jobs=jobs, target_job=target, trace=trace), None)
        graph = optional("graph", lambda: self.stage_graph(uid, skills, jobs, target, trace=trace), None if jobs else "no job input")
        match = optional("match", lambda: self.stage_match(uid, skills, jobs, target_job_id=target["job_id"] if target else None, trace=trace), None if jobs else "no job input")
        courses = optional("courses", lambda: self.stage_courses(uid, skills, match, top_k=top_k, trace=trace), None if match is not None and match.target_job else "no match artifact / target job")
        if graph is not None and courses is not None:
            try:
                self._attach_courses_to_graph(graph, courses)
            except Exception as exc:
                trace.warn(f"attaching courses to graph failed ({type(exc).__name__}: {exc})")
        explanation = optional("explain", lambda: self.stage_explain(profile, skills, match, courses, graph, extraction, trace=trace), None if match is not None else "no match artifact")
        if match is not None and target_how:
            match.record.provenance["target_selection"] = target_how

        trace.total_seconds = time.perf_counter() - t0
        return PipelineResult(text, uid, extraction, profile, graph, match, courses, explanation, trace)

    def _unpack_profile(self, profile: str | Mapping[str, Any], user_id: str | None) -> tuple[str, str]:
        default_uid = str(self.cfg["profile"]["user_id"])
        if isinstance(profile, Mapping):
            text = profile.get("text") or profile.get("profile_text")
            if not text or not str(text).strip():
                raise ValueError("profile mapping needs a non-empty 'text' field")
            return str(text), str(user_id or profile.get("user_id") or default_uid)
        if not str(profile).strip():
            raise ValueError("profile text is empty")
        return str(profile), str(user_id or default_uid)


# --------------------------------------------------------------------------- CLI
def _load_profile_file(path: str) -> str | dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return read_json(p)
    return p.read_text(encoding="utf-8")


def _load_job_file(path: str) -> str | dict[str, Any] | list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(p)
    if suffix == ".json":
        return read_json(p)
    return p.read_text(encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SkillLLM_v2 end-to-end inference pipeline")
    parser.add_argument("--profile", required=True, help="profile file: .json ({'text', 'user_id'}) or plain text")
    parser.add_argument("--job", default=None, help="job file: .jsonl/.json (structured job(s)) or plain text")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--config", default=None, help=f"YAML config (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--no-mistral", action="store_true", help="never load Mistral weights: use the deterministic stub anchor (flagged in the trace)")
    parser.add_argument("--anchor", choices=["auto", "mistral", "stub", "none"], default=None, help="override anchor.mode (none = dictionary/GLiNER only)")
    parser.add_argument("--stages", default=None, help="comma-separated stages to run (default: all enabled)")
    parser.add_argument("--skip", default="", help=f"comma-separated stages/components to skip (stages: {','.join(STAGE_ORDER)}; components: {','.join(COMPONENT_SWITCHES)})")
    parser.add_argument("--courses", choices=["auto", "adapter", "null"], default=None, help="course port: adapter delegates to CourseRecommendationPipeline, null is a labelled no-op")
    parser.add_argument("--catalog", default=None, help="course catalog path for the adapter")
    parser.add_argument("--target-job", default=None, help="job_id to target when several jobs are supplied (default: best match)")
    parser.add_argument("--top-k", type=int, default=None, help="number of course recommendations")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    pipeline = SkillLLMV2Pipeline(args.config)
    if args.no_mistral:
        pipeline.cfg["anchor"]["mode"] = "stub"
    if args.anchor:
        pipeline.cfg["anchor"]["mode"] = args.anchor
    if args.courses or args.catalog:
        pipeline.course_port = build_course_recommender(args.courses or str(pipeline.cfg["courses"]["port"]), args.catalog or pipeline._opt_abs(pipeline.cfg["courses"]["catalog_path"]))
    stages = [s for s in args.stages.split(",") if s] if args.stages else None
    skip = [s for s in args.skip.split(",") if s]
    result = pipeline.run(
        _load_profile_file(args.profile),
        _load_job_file(args.job) if args.job else None,
        stages=stages,
        skip=skip,
        target_job_id=args.target_job,
        top_k=args.top_k,
    )
    payload = result.to_dict()
    payload["pipeline_version"] = PIPELINE_VERSION
    write_json(args.out, json.loads(json.dumps(payload, default=str)))
    trace = result.trace
    print(f"skills={len(result.skills)} anchor_mode={trace.anchor_mode} degraded={trace.degraded} stubs={trace.stub_components}")
    for record in trace.stages:
        print(f"  {record.name:12s} {record.status:9s} {record.duration_seconds * 1000:8.1f} ms  {record.component}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
