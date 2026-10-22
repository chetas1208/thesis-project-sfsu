"""Public, typed library API for SkillLLM_v2.

Quick start::

    from skillllm_v2 import SkillLLM

    # no model, no GPU: dictionary/alias matching only (clearly flagged in the result metadata)
    llm = SkillLLM.offline()
    result = llm.extract("Built ETL pipelines in Python and SQL on AWS.")
    for skill in result.skills:
        print(skill.surface_form, skill.label, skill.start, skill.end, skill.canonical_skill_name)

    # trained system: Mistral-Small-3.1-24B + LoRA adapters (weights are NOT shipped with the package)
    llm = SkillLLM.from_pretrained(
        "models/mistral-small-3.1-24b-instruct-2503",
        adapters=["artifacts/.../mistral_lora/model", ".../qlora_loop/best", ".../qlora_loop_v2/best"],
        vote="majority",        # keep a span if >= 2 of the 3 adapters predict it
    )

Importing this module never imports torch, transformers, peft or any other
optional dependency; those are imported only when a model is actually loaded,
and a missing one raises :class:`skillllm_v2.errors.MissingExtraError` naming
the extra to install.

Thread safety: a :class:`SkillLLM` that owns a GPU model switches adapters in
place, so use one instance per thread/process.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence, runtime_checkable

from . import _lazy
from .ensemble import VoteCalibration, default_vote_calibration, resolve_min_votes, vote_spans
from .errors import (
    ConfigError,
    DeviceError,
    FeatureUnavailableError,
    GPUMemoryError,
    MissingExtraError,
    ModelNotFoundError,
    SkillLLMError,
)

__all__ = [
    "SkillLLM",
    "Skill",
    "Provenance",
    "ExtractionResult",
    "ExtractionMetadata",
    "AnalysisResult",
    "SkillMatch",
    "SpanBackend",
    "OfflineBackend",
    "MistralPeftBackend",
    "RESULT_SCHEMA",
    "CONFIG_SCHEMA",
    "library_version",
    "VoteCalibration",
]

RESULT_SCHEMA = "skillllm.extraction/1"
ANALYSIS_SCHEMA = "skillllm.analysis/1"
CONFIG_SCHEMA = "skillllm.config/1"

#: Link methods attached by default. ``bm25`` is excluded because the BM25 fallback of the taxonomy linker
#: returns *some* entry for almost any string (see docs/INFERENCE_AND_EVALS.md section 0, item 7); it can be
#: opted into with ``link_methods=ALL_LINK_METHODS``.
DEFAULT_LINK_METHODS: tuple[str, ...] = ("alias", "normalized", "fuzzy")
ALL_LINK_METHODS: tuple[str, ...] = ("alias", "normalized", "fuzzy", "bm25", "dense")

OFFLINE_NOTE = (
    "OFFLINE MODE: dictionary/alias matching only. No trained model was used, so recall is limited to the curated "
    "alias list and the reported evaluation numbers do NOT apply. Confidence values are fixed priors, not probabilities."
)
CONFIDENCE_NOTE = "confidence is a fixed per-source prior (not a calibrated probability); see provenance.raw_confidence."
CALIBRATED_NOTE = (
    "confidence is the estimated precision of spans with this many agreeing adapters, looked up from a vote table "
    "(default table estimated on the SkillSpan test split; re-fit on your data with skillllm_v2.fit_vote_calibration). "
    "The extractor's own constant prior is kept in provenance.raw_confidence."
)

_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_ALIAS_TABLE = _RESOURCE_DIR / "offline_aliases.jsonl"

_BASE_MODEL_FILES = ("config.json",)
#: Rough free-GPU-memory need (GiB, summed over visible devices) for the 24B base, by quantization.
_GPU_NEED_GIB = {"4bit": 16.0, "8bit": 28.0, None: 50.0}


def library_version() -> str:
    from . import __version__

    return __version__


# --------------------------------------------------------------------------- result types
@dataclass
class Provenance:
    """Where a skill came from.

    ``systems`` lists the adapters (or the offline extractor) that predicted this exact span; for a voted result
    it is the set of voters, and ``n_votes``/``n_systems`` give the vote count.
    """

    extractor: str  # "mistral" | "dictionary" | custom backend name
    method: str  # "single" | "vote:majority" | "vote:<n>" | "dictionary"
    systems: list[str] = field(default_factory=list)
    n_votes: int = 1
    n_systems: int = 1
    verified: bool | None = None  # None = verification disabled
    verification_reason: str | None = None
    #: The extractor's own confidence before any calibration (constant 0.86 for the Mistral adapters; the mean over voters).
    raw_confidence: float | None = None
    #: "vote_calibration" when ``Skill.confidence`` is an estimated precision looked up from the vote table, else "raw_prior".
    confidence_source: str = "raw_prior"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        return cls(
            extractor=data["extractor"],
            method=data["method"],
            systems=list(data.get("systems", [])),
            n_votes=int(data.get("n_votes", 1)),
            n_systems=int(data.get("n_systems", 1)),
            verified=data.get("verified"),
            verification_reason=data.get("verification_reason"),
            raw_confidence=data.get("raw_confidence"),
            confidence_source=data.get("confidence_source", "raw_prior"),
        )


@dataclass
class Skill:
    """One extracted skill mention. ``start``/``end`` are character offsets into the input text (end exclusive)."""

    surface_form: str
    label: str  # "HARD_SKILL" | "SOFT_SKILL"
    start: int | None
    end: int | None
    confidence: float
    evidence: str
    canonical_skill_id: str | None = None
    canonical_skill_name: str | None = None
    taxonomy: str | None = None
    link_confidence: float | None = None
    link_method: str | None = None
    provenance: Provenance = field(default_factory=lambda: Provenance("unknown", "single"))
    #: How many of ``n_systems`` adapters predicted this exact (start, end, label) span.
    votes: int = 1
    n_systems: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Skill":
        return cls(
            surface_form=data["surface_form"],
            label=data["label"],
            start=data.get("start"),
            end=data.get("end"),
            confidence=float(data["confidence"]),
            evidence=data.get("evidence", data["surface_form"]),
            canonical_skill_id=data.get("canonical_skill_id"),
            canonical_skill_name=data.get("canonical_skill_name"),
            taxonomy=data.get("taxonomy"),
            link_confidence=data.get("link_confidence"),
            link_method=data.get("link_method"),
            votes=int(data.get("votes", 1)),
            n_systems=int(data.get("n_systems", 1)),
            provenance=Provenance.from_dict(data["provenance"]) if data.get("provenance") else Provenance("unknown", "single"),
        )

    @property
    def span(self) -> tuple[int | None, int | None, str]:
        return (self.start, self.end, self.label)


@dataclass
class ExtractionMetadata:
    """How a result was produced. ``offline=True`` means NO trained model was involved."""

    mode: str  # "model" | "offline" | "custom"
    offline: bool
    systems: list[str] = field(default_factory=list)
    vote: str | None = None
    min_votes: int | None = None
    base_model: str | None = None
    adapters: dict[str, str] = field(default_factory=dict)
    quantization: str | None = None
    linked: bool = False
    verified: bool = False
    link_methods: list[str] = field(default_factory=list)
    raw_span_counts: dict[str, int] = field(default_factory=dict)
    n_dropped_by_verifier: int = 0
    #: The vote -> precision table used for ``confidence`` (None: confidences are the raw per-extractor priors).
    calibration: dict[str, Any] | None = None
    latency_seconds: float = 0.0
    library_version: str = ""
    schema: str = RESULT_SCHEMA
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractionMetadata":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ExtractionResult:
    """Skills extracted from one text, plus how they were produced."""

    text: str
    skills: list[Skill]
    metadata: ExtractionMetadata

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.skills)

    def __len__(self) -> int:
        return len(self.skills)

    @property
    def is_offline(self) -> bool:
        return self.metadata.offline

    def names(self) -> list[str]:
        """Canonical name when linked, else the surface form; order of appearance, de-duplicated."""
        return list(dict.fromkeys(s.canonical_skill_name or s.surface_form for s in self.skills))

    def spans(self) -> list[tuple[int | None, int | None, str]]:
        return [s.span for s in self.skills]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": RESULT_SCHEMA, "text": self.text, "skills": [s.to_dict() for s in self.skills], "metadata": self.metadata.to_dict()}

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractionResult":
        return cls(text=data["text"], skills=[Skill.from_dict(s) for s in data["skills"]], metadata=ExtractionMetadata.from_dict(data["metadata"]))

    @classmethod
    def from_json(cls, payload: str) -> "ExtractionResult":
        return cls.from_dict(json.loads(payload))


@dataclass
class SkillMatch:
    """A job skill that the profile also has (matched by canonical id when linked, else normalised surface form)."""

    key: str
    skill: str
    profile_surface: str
    job_surface: str
    matched_by: str  # "canonical_skill_id" | "surface_form"


@dataclass
class AnalysisResult:
    """Profile-versus-job analysis. ``method`` says which engine produced the gap/match fields."""

    profile: ExtractionResult
    job: ExtractionResult
    matched: list[SkillMatch]
    missing: list[Skill]  # job skills absent from the profile
    extra: list[Skill]  # profile skills the job does not mention
    coverage: float  # matched job skills / job skills (1.0 when the job lists none)
    jaccard: float
    method: str  # "set_overlap" | "inference_pipeline"
    notes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)  # raw output of the inference pipeline when used
    #: Rule-weighted match score (0-1) and typed gaps from the inference pipeline; None / [] under ``set_overlap``.
    #: The score has no validation study behind it (see docs/INFERENCE_AND_EVALS.md); treat it as a heuristic.
    match_score: float | None = None
    gaps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_SCHEMA,
            "method": self.method,
            "coverage": self.coverage,
            "jaccard": self.jaccard,
            "matched": [asdict(m) for m in self.matched],
            "missing": [s.to_dict() for s in self.missing],
            "extra": [s.to_dict() for s in self.extra],
            "profile": self.profile.to_dict(),
            "job": self.job.to_dict(),
            "notes": list(self.notes),
            "match_score": self.match_score,
            "gaps": self.gaps,
            "details": self.details,
        }

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), default=str, **kwargs)


# --------------------------------------------------------------------------- backends
@runtime_checkable
class SpanBackend(Protocol):
    """Anything that can turn texts into raw span predictions for a named system.

    Raw spans are dicts with ``surface_form``, ``label``, ``start``, ``end``, ``confidence`` and ``evidence``
    (the shape stored in the evaluation ``predictions.jsonl`` files). Implement this to plug in a custom
    or fake model; ``tests/test_library_api.py`` uses a fake backend.
    """

    mode: str
    system_names: Sequence[str]

    def predict_batch(self, texts: Sequence[str], system: str) -> list[list[dict[str, Any]]]: ...

    def close(self) -> None: ...


def _resolve_overlaps(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the longest of overlapping spans (ties: earliest); return in text order."""
    ranked = sorted(spans, key=lambda s: (-(s["end"] - s["start"]), s["start"], s["label"]))
    taken: list[dict[str, Any]] = []
    for sp in ranked:
        if all(sp["end"] <= t["start"] or sp["start"] >= t["end"] for t in taken):
            taken.append(sp)
    return sorted(taken, key=lambda s: (s["start"], s["end"]))


class OfflineBackend:
    """Dictionary/alias extractor. No model, no GPU, no third-party packages. Exact whole-word alias matches only
    (fuzzy matching is off by default), longest match wins on overlap."""

    mode = "offline"
    system_names: Sequence[str] = ("dictionary",)

    def __init__(self, alias_paths: Sequence[str | os.PathLike[str]] | None = None, fuzzy: bool = False) -> None:
        from .extract.dictionary_extractor import DictionaryExtractor

        paths = [str(p) for p in (alias_paths or [DEFAULT_ALIAS_TABLE])]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise ModelNotFoundError(f"Alias table not found: {missing[0]}")
        self.alias_paths = paths
        self.fuzzy = fuzzy
        extractor_cls = DictionaryExtractor
        if not fuzzy:
            class _ExactOnly(DictionaryExtractor):  # skip the slow, false-positive-prone fuzzy pass
                def _fuzzy_candidates(self, text: str, seen: set[tuple[int, int, str]]) -> list[Any]:
                    return []

            extractor_cls = _ExactOnly
        self._extractor = extractor_cls(alias_paths=paths)

    def predict_batch(self, texts: Sequence[str], system: str = "dictionary") -> list[list[dict[str, Any]]]:
        out: list[list[dict[str, Any]]] = []
        for text in texts:
            spans = [s for s in self._extractor.extract(text) if s.get("start") is not None]
            out.append(_resolve_overlaps(spans))
        return out

    def close(self) -> None:
        return None


def _looks_like_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def _adapter_names(adapters: Sequence[str | os.PathLike[str]] | Mapping[str, str | os.PathLike[str]]) -> dict[str, str]:
    """Stable unique system names for adapter paths (a mapping keeps the caller's names)."""
    if isinstance(adapters, Mapping):
        return {str(k): str(v) for k, v in adapters.items()}
    generic = {"best", "model", "final", "adapter", "checkpoint"}
    names: dict[str, str] = {}
    for raw in adapters:
        p = Path(raw)
        base = p.parent.name if p.name.lower() in generic and p.parent.name else p.name
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base) or "adapter"
        name, i = base, 2
        while name in names:
            name, i = f"{base}-{i}", i + 1
        names[name] = str(raw)
    return names


class MistralPeftBackend:
    """Mistral-Small-3.1-24B (or any causal LM the extractor supports) with one or more LoRA adapters.

    The base model is loaded ONCE (optionally 4-bit NF4) and every adapter is attached to the same weights;
    adapters are switched with ``set_adapter`` between systems, so N adapters cost one base in memory and
    N generation passes. Prompting, greedy decoding and JSON parsing are those of
    :class:`skillllm_v2.extract.mistral_extractor.MistralExtractor`, which produced the evaluated predictions.
    Loading is lazy: nothing heavy happens until the first prediction (or ``load()``).
    """

    mode = "model"

    def __init__(
        self,
        base_model: str,
        adapters: Mapping[str, str] | None = None,
        quantization: Literal["4bit", "8bit"] | None = "4bit",
        device: str = "auto",
        max_new_tokens: int = 512,
        trust_remote_code: bool = False,
        allow_hub: bool = False,
        check_memory: Literal["warn", "error", "off"] = "warn",
    ) -> None:
        if quantization not in ("4bit", "8bit", None):
            raise ConfigError(f"quantization must be '4bit', '8bit' or None; got {quantization!r}")
        self.base_model = str(base_model)
        self.adapters: dict[str, str] = dict(adapters or {})
        self.quantization = quantization
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.trust_remote_code = trust_remote_code
        self.allow_hub = allow_hub
        self.check_memory = check_memory
        self.system_names: list[str] = list(self.adapters) or ["base"]
        self._model: Any = None
        self._tokenizer: Any = None
        self._extractor: Any = None
        self._active: str | None = None
        self._warnings: list[str] = []
        self.validate_paths()

    # -- pre-flight (cheap; no heavy imports) ---------------------------------------------------
    def validate_paths(self) -> None:
        base = Path(self.base_model).expanduser()
        if not base.exists():
            if not self.allow_hub:
                raise ModelNotFoundError(
                    f"Base model path does not exist: {self.base_model!r}. The base-model weights are not bundled with "
                    "skillllm: download Mistral-Small-3.1-24B-Instruct-2503 and pass its local directory, or pass "
                    "allow_hub=True to resolve a Hugging Face Hub repo id (needs network and an accepted licence)."
                )
        elif not base.is_dir() or not all((base / f).exists() for f in _BASE_MODEL_FILES):
            raise ModelNotFoundError(f"{self.base_model!r} exists but is not a model directory (no config.json found).")
        for name, raw in self.adapters.items():
            p = Path(raw).expanduser()
            if not p.exists():
                if self.allow_hub:
                    continue
                raise ModelNotFoundError(
                    f"Adapter {name!r} not found at {raw!r}. Pass the directory that contains adapter_config.json and "
                    "adapter_model.safetensors (a local training output, or a downloaded Hub snapshot)."
                )
            if not _looks_like_adapter_dir(p):
                raise ModelNotFoundError(
                    f"Adapter {name!r} at {raw!r} is not a PEFT adapter directory (adapter_config.json missing)."
                )
            if not (p / "adapter_model.safetensors").exists() and not (p / "adapter_model.bin").exists():
                raise ModelNotFoundError(f"Adapter {name!r} at {raw!r} has no adapter_model.safetensors weights file.")

    def required_modules(self) -> list[str]:
        mods = ["torch", "transformers", "accelerate"]
        if self.adapters:
            mods.append("peft")
        if self.quantization:
            mods.append("bitsandbytes")
        return mods

    # -- loading --------------------------------------------------------------------------------
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        purposes = {"torch": "model inference", "transformers": "loading the base model", "peft": "loading LoRA adapters",
                    "bitsandbytes": f"{self.quantization} quantization", "accelerate": "device_map loading"}
        for module in self.required_modules():
            _lazy.require(module, "model", purposes.get(module, ""))
        import torch  # noqa: PLC0415  (deliberately lazy)

        device_map = self._resolve_device(torch)
        self._check_memory(torch)
        import transformers  # noqa: PLC0415

        from .extract.mistral_extractor import MistralExtractor, _ensure_chat_template, _load_model, _load_tokenizer

        kwargs: dict[str, Any] = {"device_map": device_map, "trust_remote_code": self.trust_remote_code}
        if self.quantization == "4bit":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif self.quantization == "8bit":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        try:
            tokenizer = _load_tokenizer(transformers.AutoTokenizer, self.base_model, self.trust_remote_code)
            _ensure_chat_template(tokenizer, self.base_model)
            model = _load_model(transformers, self.base_model, kwargs)
            if self.adapters:
                import peft  # noqa: PLC0415

                names = list(self.adapters)
                model = peft.PeftModel.from_pretrained(model, self.adapters[names[0]], adapter_name=names[0])
                for name in names[1:]:
                    model.load_adapter(self.adapters[name], adapter_name=name)
                model.set_adapter(names[0])
                self._active = names[0]
            model.eval()
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, torch, "loading") from exc
        self._tokenizer, self._model = tokenizer, model
        extractor = MistralExtractor(model_path=self.base_model, use_4bit=self.quantization == "4bit",
                                     max_new_tokens=self.max_new_tokens, allow_fallback=False)
        extractor._model, extractor._tokenizer = model, tokenizer
        self._extractor = extractor

    def _resolve_device(self, torch: Any) -> Any:
        dev = (self.device or "auto").lower()
        has_cuda = bool(torch.cuda.is_available())
        if dev == "cpu":
            if self.quantization:
                raise DeviceError(
                    f"{self.quantization} quantization needs a CUDA GPU (bitsandbytes). On CPU pass quantization=None "
                    "(needs ~50 GB RAM and is very slow for a 24B model), or use SkillLLM.offline() for a model-free run."
                )
            warnings.warn("Running the 24B model on CPU in bfloat16 needs ~50 GB of RAM and is extremely slow.", stacklevel=3)
            return "cpu"
        if not has_cuda:
            raise DeviceError(
                "No CUDA GPU is visible (torch.cuda.is_available() is False). The 24B model needs a GPU "
                "(about 16 GB of free GPU memory with quantization='4bit'). Check `skillllm doctor` and CUDA_VISIBLE_DEVICES, "
                "run on a GPU machine, or use SkillLLM.offline() for the model-free dictionary extractor."
            )
        if dev == "auto" or dev == "cuda":
            return "auto"
        m = re.fullmatch(r"cuda:(\d+)", dev)
        if not m:
            raise ConfigError(f"device must be 'auto', 'cuda', 'cuda:<index>' or 'cpu'; got {self.device!r}")
        index = int(m.group(1))
        if index >= torch.cuda.device_count():
            raise DeviceError(f"device {self.device!r} requested but only {torch.cuda.device_count()} CUDA device(s) are visible.")
        return {"": index}

    def _check_memory(self, torch: Any) -> None:
        if self.check_memory == "off" or not torch.cuda.is_available():
            return
        need = _GPU_NEED_GIB[self.quantization]
        try:
            free = sum(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())) / 2**30
        except Exception:  # noqa: BLE001 - best effort only
            return
        if free < need:
            msg = (f"Only {free:.1f} GiB of GPU memory is free across visible devices; loading the 24B base with "
                   f"quantization={self.quantization!r} needs roughly {need:.0f} GiB. Free GPU memory (see `skillllm doctor`), "
                   "expose more devices via CUDA_VISIBLE_DEVICES, or pass check_memory='off' to try anyway.")
            if self.check_memory == "error":
                raise GPUMemoryError(msg)
            warnings.warn(msg, ResourceWarning, stacklevel=3)

    def _translate(self, exc: Exception, torch: Any, stage: str) -> Exception:
        if isinstance(exc, SkillLLMError):
            return exc
        oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if (oom is not None and isinstance(exc, oom)) or "out of memory" in str(exc).lower():
            return GPUMemoryError(
                f"CUDA ran out of memory while {stage} the model. Remedies: free other GPU jobs (`nvidia-smi`), keep "
                "quantization='4bit', spread across GPUs with device='auto', reduce max_new_tokens, split very long "
                "documents into chunks, or use SkillLLM.offline() without a GPU."
            )
        return exc

    # -- inference ------------------------------------------------------------------------------
    def predict_batch(self, texts: Sequence[str], system: str) -> list[list[dict[str, Any]]]:
        if system not in self.system_names:
            raise ConfigError(f"Unknown system {system!r}; available: {list(self.system_names)}")
        self.load()
        if self.adapters and self._active != system:
            self._model.set_adapter(system)
            self._active = system
        out: list[list[dict[str, Any]]] = []
        for text in texts:
            try:
                out.append(self._extractor.extract(text))
            except ValueError as exc:  # unparsable model output: no skills for this doc, but say so
                self._warnings.append(f"system {system!r}: model output was not valid JSON ({str(exc)[:120]}); treated as no skills")
                out.append([])
            except Exception as exc:  # noqa: BLE001
                raise self._translate(exc, sys.modules.get("torch"), "generating with") from exc
        return out

    def drain_warnings(self) -> list[str]:
        warns, self._warnings = self._warnings, []
        return warns

    def close(self) -> None:
        self._model = self._tokenizer = self._extractor = None
        self._active = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- main class
_Vote = Literal["majority", "unanimous", "union"]


class SkillLLM:
    """Skill extraction (and profile/job analysis) behind one small interface.

    Construct with :meth:`from_pretrained` (trained model + adapters), :meth:`offline` (no model), :meth:`from_config`
    or :meth:`from_backend` (custom/fake backend).
    """

    def __init__(
        self,
        backend: SpanBackend,
        *,
        vote: str | int | None = None,
        link: bool | Any = True,
        verify: bool = True,
        alias_table: str | os.PathLike[str] | None = None,
        link_methods: Sequence[str] = DEFAULT_LINK_METHODS,
        min_votes: int | None = None,
        calibration: "VoteCalibration | Mapping[str, Any] | str | bool | None" = "default",
        quantization: str | None = None,
        device: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.backend = backend
        self.systems: list[str] = list(backend.system_names)
        if min_votes is not None:
            vote = min_votes  # the explicit precision/recall dial wins over the named rule
        if len(self.systems) > 1:
            if vote is None:
                raise ConfigError(
                    f"{len(self.systems)} adapters were given but vote=None; pass vote='majority' (recommended), "
                    "'unanimous', 'union' or an int, or load a single adapter."
                )
            self.vote: str | int | None = vote
            self.min_votes: int | None = resolve_min_votes(vote, len(self.systems))
        else:
            self.vote, self.min_votes = None, None
        self.link = link
        self.verify = bool(verify)
        self.alias_table = str(alias_table) if alias_table else None
        self.link_methods = tuple(link_methods)
        self.quantization = quantization
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.calibration: VoteCalibration | None = self._resolve_calibration(calibration)
        self._calibration_arg = calibration
        self._linker: Any = link if (link not in (True, False, None)) else None
        self._verifier: Any = None

    def _resolve_calibration(self, calibration: Any) -> VoteCalibration | None:
        """``'default'`` -> the shipped table when it matches the number of adapters; ``None``/``False`` -> raw priors."""
        if len(self.systems) < 2 or calibration is None or calibration is False:
            return None
        if isinstance(calibration, VoteCalibration):
            cal = calibration
        elif isinstance(calibration, Mapping):
            cal = VoteCalibration.from_dict(calibration)
        elif calibration is True or calibration == "default":
            return default_vote_calibration(len(self.systems))
        else:
            raise ConfigError("calibration must be 'default', None/False, a VoteCalibration or its dict form.")
        if cal.n_systems != len(self.systems):
            raise ConfigError(f"calibration table is for {cal.n_systems} systems but {len(self.systems)} adapters are loaded.")
        return cal

    # -- constructors ---------------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        base_model: str | os.PathLike[str],
        adapter: str | os.PathLike[str] | None = None,
        adapters: Sequence[str | os.PathLike[str]] | Mapping[str, str | os.PathLike[str]] | None = None,
        vote: str | int | None = "majority",
        quantization: Literal["4bit", "8bit"] | None = "4bit",
        device: str = "auto",
        link: bool | Any = True,
        verify: bool = True,
        *,
        max_new_tokens: int = 512,
        trust_remote_code: bool = False,
        allow_hub: bool = False,
        check_memory: Literal["warn", "error", "off"] = "warn",
        alias_table: str | os.PathLike[str] | None = None,
        link_methods: Sequence[str] = DEFAULT_LINK_METHODS,
        min_votes: int | None = None,
        calibration: "VoteCalibration | Mapping[str, Any] | str | bool | None" = "default",
    ) -> "SkillLLM":
        """Build a model-backed extractor. Nothing heavy is imported or loaded until the first ``extract`` call
        (path/adapter problems are reported immediately, missing packages at first use with the extra to install).

        Args:
            base_model: local directory of the base model (weights are never bundled with this package).
            adapter: one LoRA adapter directory. Mutually exclusive with ``adapters``.
            adapters: several adapters (list of paths, or ``{name: path}``). With more than one, ``vote`` decides how
                their spans are combined; they run sequentially on ONE loaded base via adapter switching.
            vote: ``'majority'`` (keep a span predicted by > half of the adapters; 2 of 3), ``'unanimous'``, ``'union'``,
                an int minimum vote count, or ``None`` (only valid with a single adapter).
            quantization: ``'4bit'`` (NF4, evaluated setting; ~16 GB), ``'8bit'`` or ``None`` (bf16, ~50 GB).
            device: ``'auto'`` (all visible GPUs), ``'cuda'``, ``'cuda:N'`` or ``'cpu'``.
            link: attach canonical skill ids/names from the alias table (or pass a linker object with ``.link(str)``).
            min_votes: the precision/recall dial for 3 adapters (overrides ``vote``). 3 = high precision (held-out
                exact-span P 0.80, R 0.51); 2 = balanced default (P 0.72, R 0.66, F1 0.690 vs the single anchor 0.670);
                1 = union (P 0.59, R 0.78). Every setting runs all adapters, so generation costs N times a single adapter.
            calibration: ``'default'`` sets each span's ``confidence`` to the estimated precision for its vote count
                (3-adapter table estimated on the SkillSpan test split; re-fit on your data with
                :func:`skillllm_v2.fit_vote_calibration`), a :class:`VoteCalibration`/dict for your own table, or
                ``None`` to keep the extractor's constant prior (0.86).
            verify: apply the rule-based :class:`~skillllm_v2.verify.SpanVerifier` (drops spans without offsets, and
                generic/low-confidence non-model spans).
        """
        if adapter is not None and adapters is not None:
            raise ConfigError("Pass either adapter= or adapters=, not both.")
        named: dict[str, str] = _adapter_names([adapter]) if adapter is not None else (_adapter_names(adapters) if adapters else {})
        if named and len(named) == 1:
            vote = None
        backend = MistralPeftBackend(str(base_model), named, quantization, device, max_new_tokens, trust_remote_code, allow_hub, check_memory)
        if not named:
            warnings.warn("No adapter given: this is the zero-shot base model, NOT the fine-tuned system whose metrics are reported.", stacklevel=2)
        if named and len(named) == 1:
            min_votes = None
        return cls(backend, vote=vote, link=link, verify=verify, alias_table=alias_table, link_methods=link_methods,
                   min_votes=min_votes, calibration=calibration, quantization=quantization, device=device, max_new_tokens=max_new_tokens)

    @classmethod
    def offline(
        cls,
        link: bool = True,
        verify: bool = True,
        alias_paths: Sequence[str | os.PathLike[str]] | None = None,
        fuzzy: bool = False,
    ) -> "SkillLLM":
        """Model-free dictionary/alias extractor: instant, CPU-only, dependency-free, deterministic.

        Results are flagged ``metadata.offline = True`` and carry a note: this is a gazetteer baseline, not the
        trained system, and its recall is limited to the packaged alias list (or ``alias_paths``)."""
        backend = OfflineBackend(alias_paths, fuzzy=fuzzy)
        table = alias_paths[0] if alias_paths else None
        return cls(backend, link=link, verify=verify, alias_table=table)

    @classmethod
    def from_backend(cls, backend: SpanBackend, **kwargs: Any) -> "SkillLLM":
        """Wrap any :class:`SpanBackend` (custom model service, test double)."""
        return cls(backend, **kwargs)

    # -- config ---------------------------------------------------------------------------------
    def config(self) -> dict[str, Any]:
        backend = self.backend
        cfg: dict[str, Any] = {
            "schema": CONFIG_SCHEMA,
            "library_version": library_version(),
            "mode": backend.mode,
            "vote": self.vote,
            "link": bool(self.link) if isinstance(self.link, bool) else True,
            "verify": self.verify,
            "alias_table": self.alias_table,
            "link_methods": list(self.link_methods),
            "min_votes": self.min_votes,
        }
        arg = self._calibration_arg
        if isinstance(arg, (VoteCalibration, Mapping)):
            cfg["vote_calibration"] = (arg if isinstance(arg, VoteCalibration) else VoteCalibration.from_dict(arg)).to_dict()
        else:
            cfg["vote_calibration"] = "default" if arg in (True, "default") else None
        if isinstance(backend, MistralPeftBackend):
            cfg.update(base_model=backend.base_model, adapters=dict(backend.adapters), quantization=backend.quantization,
                       device=backend.device, max_new_tokens=backend.max_new_tokens, trust_remote_code=backend.trust_remote_code,
                       allow_hub=backend.allow_hub)
        elif isinstance(backend, OfflineBackend):
            cfg.update(fuzzy=backend.fuzzy, alias_paths=list(backend.alias_paths) if self.alias_table else None)
        else:
            cfg["mode"] = "custom"
        return cfg

    def save_config(self, path: str | os.PathLike[str]) -> Path:
        """Write the configuration (paths and options, never weights) as JSON. Custom backends cannot be saved."""
        if self.backend.mode == "custom" or not isinstance(self.backend, (MistralPeftBackend, OfflineBackend)):
            raise FeatureUnavailableError("save_config() supports model-backed and offline instances; a custom backend must be rebuilt by the caller.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.config(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_config(cls, path: str | os.PathLike[str] | Mapping[str, Any], **overrides: Any) -> "SkillLLM":
        """Rebuild an instance from :meth:`save_config` output (or an equivalent mapping). Relative paths that do not
        exist from the current directory are tried relative to the config file. ``overrides`` replace config keys."""
        base_dir: Path | None = None
        if isinstance(path, Mapping):
            cfg = dict(path)
        else:
            p = Path(path)
            if not p.exists():
                raise ModelNotFoundError(f"Config file not found: {p}")
            cfg = json.loads(p.read_text(encoding="utf-8"))
            base_dir = p.resolve().parent
        cfg.update(overrides)
        if cfg.get("schema") != CONFIG_SCHEMA:
            raise ConfigError(f"Unsupported config schema {cfg.get('schema')!r}; expected {CONFIG_SCHEMA!r}.")

        def fix(raw: str | None) -> str | None:
            if raw is None or base_dir is None:
                return raw
            p = Path(raw).expanduser()
            if not p.is_absolute() and not p.exists() and (base_dir / p).exists():
                return str(base_dir / p)
            return raw

        common = dict(link=cfg.get("link", True), verify=cfg.get("verify", True), link_methods=cfg.get("link_methods", DEFAULT_LINK_METHODS))
        cal_cfg = cfg.get("vote_calibration", "default")
        mode = cfg.get("mode")
        if mode == "offline":
            paths = [fix(p) for p in cfg["alias_paths"]] if cfg.get("alias_paths") else None
            return cls.offline(link=common["link"], verify=common["verify"], alias_paths=paths, fuzzy=cfg.get("fuzzy", False))  # type: ignore[arg-type]
        if mode == "model":
            adapters = {k: fix(v) for k, v in (cfg.get("adapters") or {}).items()}
            return cls.from_pretrained(
                fix(cfg["base_model"]), adapters=adapters or None, vote=cfg.get("vote", "majority"),  # type: ignore[arg-type]
                quantization=cfg.get("quantization", "4bit"), device=cfg.get("device", "auto"),
                max_new_tokens=cfg.get("max_new_tokens", 512), trust_remote_code=cfg.get("trust_remote_code", False),
                allow_hub=cfg.get("allow_hub", False), alias_table=fix(cfg.get("alias_table")),
                min_votes=cfg.get("min_votes"), calibration=cal_cfg, **common,
            )
        raise ConfigError(f"Unsupported config mode {mode!r} (expected 'model' or 'offline').")

    # -- lifecycle ------------------------------------------------------------------------------
    def close(self) -> None:
        """Release GPU memory held by a model-backed instance."""
        self.backend.close()

    def __enter__(self) -> "SkillLLM":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_offline(self) -> bool:
        return self.backend.mode == "offline"

    def __repr__(self) -> str:
        return f"SkillLLM(mode={self.backend.mode!r}, systems={self.systems}, vote={self.vote!r}, link={bool(self.link)}, verify={self.verify})"

    # -- extraction -----------------------------------------------------------------------------
    def extract(self, text: str) -> ExtractionResult:
        """Extract skills from one text."""
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: Iterable[str]) -> list[ExtractionResult]:
        """Extract skills from several texts. Systems run system-major (one adapter switch per system, not per
        text); generation itself is sequential per text, matching the evaluated protocol."""
        docs = list(texts)
        for i, t in enumerate(docs):
            if not isinstance(t, str):
                raise TypeError(f"texts[{i}] must be str, got {type(t).__name__}")
        t0 = time.perf_counter()
        per_system: dict[str, list[list[dict[str, Any]]]] = {}
        for name in self.systems:
            preds = self.backend.predict_batch(docs, name)
            if len(preds) != len(docs):
                raise SkillLLMError(f"Backend returned {len(preds)} prediction lists for {len(docs)} texts (system {name!r}).")
            per_system[name] = preds
        backend_warnings = list(getattr(self.backend, "drain_warnings", lambda: [])())
        elapsed = time.perf_counter() - t0
        return [self._assemble(doc, {n: per_system[n][i] for n in self.systems}, elapsed / max(len(docs), 1), backend_warnings)
                for i, doc in enumerate(docs)]

    # -- internals ------------------------------------------------------------------------------
    def _get_verifier(self) -> Any:
        if self._verifier is None:
            from .verify.span_verifier import SpanVerifier

            self._verifier = SpanVerifier()
        return self._verifier

    def _get_linker(self) -> Any:
        if self._linker is None:
            from .normalize.taxonomy_linker import TaxonomyLinker

            table = self.alias_table or str(DEFAULT_ALIAS_TABLE)
            if not Path(table).exists():
                raise ModelNotFoundError(f"Alias table not found: {table}")
            self._linker = TaxonomyLinker(alias_table_paths=[table])
        return self._linker

    def _combine(self, raw_by_system: dict[str, list[dict[str, Any]]]) -> list[tuple[dict[str, Any], Provenance]]:
        extractor = "dictionary" if self.is_offline else "mistral"
        if len(self.systems) == 1:
            name = self.systems[0]
            spans = sorted(raw_by_system[name], key=lambda s: (s.get("start") is None, s.get("start") or 0, s.get("end") or 0))
            prov = lambda: Provenance(extractor, "dictionary" if self.is_offline else "single", [name], 1, 1)  # noqa: E731
            result = []
            for s in spans:
                p = prov()
                p.raw_confidence = s.get("confidence")
                result.append((dict(s), p))
            return result
        assert self.min_votes is not None
        voted = vote_spans([raw_by_system[n] for n in self.systems], self.min_votes, system_names=self.systems,
                           annotate=True, order="position", skip_unlocated=True)
        label = self.vote if isinstance(self.vote, str) else f"{self.vote}"
        out = []
        for sp in voted:
            v = sp.pop("votes")
            sp["source"] = extractor
            out.append((sp, Provenance(extractor, f"vote:{label}", v["voted_by"], v["n_votes"], v["n_systems"],
                                       raw_confidence=sp.get("confidence"))))
        return out

    def _assemble(self, text: str, raw_by_system: dict[str, list[dict[str, Any]]], latency: float, backend_warnings: list[str]) -> ExtractionResult:
        combined = self._combine(raw_by_system)
        n_dropped = 0
        if self.verify and combined:
            verifier = self._get_verifier()
            kept: list[tuple[dict[str, Any], Provenance]] = []
            for sp, prov in combined:
                cand = _as_candidate(sp, text)
                decision = verifier.verify(cand)
                prov.verified, prov.verification_reason = bool(decision["is_valid_skill"]), decision["reason"]
                if decision["is_valid_skill"]:
                    kept.append((sp, prov))
                else:
                    n_dropped += 1
            combined = kept
        linker = self._get_linker() if self.link and combined else None
        skills: list[Skill] = []
        for sp, prov in combined:
            confidence = float(sp.get("confidence", 0.0))
            if self.calibration is not None:
                calibrated = self.calibration.confidence(prov.n_votes)
                if calibrated is not None:
                    confidence, prov.confidence_source = calibrated, "vote_calibration"
            skill = Skill(
                surface_form=str(sp["surface_form"]), label=str(sp.get("label") or "HARD_SKILL"),
                start=sp.get("start"), end=sp.get("end"), confidence=confidence,
                evidence=str(sp.get("evidence") or sp["surface_form"]), provenance=prov,
                votes=prov.n_votes, n_systems=prov.n_systems,
            )
            if linker is not None:
                match = linker.link(skill.surface_form)
                if match and match.get("link_method") in self.link_methods:
                    skill.canonical_skill_id = match["canonical_skill_id"]
                    skill.canonical_skill_name = match["canonical_skill_name"]
                    skill.taxonomy = match.get("taxonomy")
                    skill.link_confidence = float(match["link_confidence"])
                    skill.link_method = match["link_method"]
            skills.append(skill)
        backend = self.backend
        notes = [CALIBRATED_NOTE if self.calibration is not None else CONFIDENCE_NOTE]
        if self.is_offline:
            notes.insert(0, OFFLINE_NOTE)
        elif not getattr(backend, "adapters", True):
            notes.insert(0, "No adapter loaded: zero-shot base model, not the fine-tuned system.")
        meta = ExtractionMetadata(
            mode=backend.mode, offline=self.is_offline, systems=list(self.systems),
            vote=None if self.vote is None else str(self.vote), min_votes=self.min_votes,
            base_model=getattr(backend, "base_model", None), adapters=dict(getattr(backend, "adapters", {}) or {}),
            quantization=getattr(backend, "quantization", None), linked=bool(self.link),
            verified=self.verify, link_methods=list(self.link_methods) if self.link else [],
            raw_span_counts={n: len(raw_by_system[n]) for n in self.systems}, n_dropped_by_verifier=n_dropped,
            calibration=self.calibration.to_dict() if self.calibration is not None else None,
            latency_seconds=round(latency, 6), library_version=library_version(), notes=notes, warnings=list(backend_warnings),
        )
        return ExtractionResult(text=text, skills=skills, metadata=meta)

    # -- analysis / recommendation ----------------------------------------------------------------
    def analyze(self, profile_text: str, job_text: str) -> AnalysisResult:
        """Compare a profile (resume) with a job description: extracted skills, matched, missing and extra.

        Uses the end-to-end inference pipeline (``skillllm_v2.pipeline.inference``) when it is importable;
        otherwise a plain, clearly-labelled skill-set overlap (``method='set_overlap'``)."""
        profile, job = self.extract_batch([profile_text, job_text])
        result = _analyze_set_overlap(profile, job)
        pipeline_details = _try_inference_pipeline(self, profile, job)
        if pipeline_details is not None:
            result.method = "inference_pipeline"
            result.details = pipeline_details
            match = pipeline_details.get("match") or {}
            result.match_score = (match.get("job_match") or {}).get("match_score")
            result.gaps = list(match.get("gaps") or [])
            result.notes.append("details holds the raw output of skillllm_v2.pipeline.inference.")
        else:
            reason = _PIPELINE_ERRORS[-1] if _PIPELINE_ERRORS else "module not importable"
            result.notes.append(
                f"skillllm_v2.pipeline.inference is not available ({reason}), so only a plain skill-set overlap was computed "
                "(no graph-based gap engine, no match score)."
            )
        return result

    def recommend(
        self,
        profile_text: str,
        job_text: str,
        catalog_path: str | os.PathLike[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Course recommendations for the profile-to-job skill gap, through the existing course pipeline only.

        Requires the course recommender (``skillllm_v2.pipeline.course_recommendation_pipeline``, needs ``networkx``)
        and a course catalog JSONL. Raises :class:`FeatureUnavailableError` with the reason when either is missing."""
        try:
            from .pipeline.course_port import CourseRecommendationRequest, CourseRecommendationPipelineAdapter
        except Exception as exc:  # noqa: BLE001
            raise FeatureUnavailableError(
                f"Course recommendation is not available in this install ({type(exc).__name__}: {exc}). "
                "It needs the repository's course pipeline; install from source with the 'recommend' extra."
            ) from exc
        if catalog_path is None:
            raise FeatureUnavailableError(
                "recommend() needs a course catalog: pass catalog_path=<course_catalog.jsonl> "
                "(the repository ships a demo catalog at inputs/courses/course_catalog.jsonl) and install 'skillllm[recommend]'."
            )
        adapter = CourseRecommendationPipelineAdapter(catalog_path=catalog_path)
        if not adapter.is_available():
            raise FeatureUnavailableError(f"Course recommendation is unavailable: {adapter.unavailable_reason or 'catalog not configured'}. "
                                          "Pass catalog_path=<course_catalog.jsonl> and install the 'recommend' extra (networkx).")
        profile, job = self.extract_batch([profile_text, job_text])
        user_profile = {"user_id": "user", "skills": [_course_skill(s) for s in profile.skills], "target_roles": []}
        job_dict = {"job_id": "target_role", "title": "Target Role",
                    "required_skills": [s.canonical_skill_name or s.surface_form for s in job.skills if s.label == "HARD_SKILL"],
                    "preferred_skills": [s.canonical_skill_name or s.surface_form for s in job.skills if s.label == "SOFT_SKILL"]}
        try:
            response = adapter.recommend(CourseRecommendationRequest("user", user_profile, job_dict, top_k=top_k))
        except (ImportError, FileNotFoundError) as exc:
            raise FeatureUnavailableError(f"Course recommender unavailable: {exc}. Try: pip install 'skillllm[recommend]' and pass a valid catalog_path.") from exc
        return response.to_dict()


# --------------------------------------------------------------------------- helpers
def _as_candidate(sp: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Shape a raw span like a validated SkillCandidate dict for SpanVerifier."""
    label = sp.get("label") if sp.get("label") in ("HARD_SKILL", "SOFT_SKILL") else "HARD_SKILL"
    source = sp.get("source") if sp.get("source") in ("bert", "mistral", "gliner", "dictionary") else "mistral"
    return {
        "surface_form": sp["surface_form"], "label": label, "start": sp.get("start"), "end": sp.get("end"), "source": source,
        "confidence": float(sp.get("confidence", 0.5)), "evidence": sp.get("evidence") or sp["surface_form"],
        "canonical_skill_id": None, "canonical_skill_name": None, "metadata": {},
    }


def _key(skill: Skill) -> tuple[str, str]:
    from .schema import normalize_text

    if skill.canonical_skill_id:
        return skill.canonical_skill_id, "canonical_skill_id"
    return normalize_text(skill.surface_form), "surface_form"


def _analyze_set_overlap(profile: ExtractionResult, job: ExtractionResult) -> AnalysisResult:
    prof: dict[str, Skill] = {}
    for s in profile.skills:
        prof.setdefault(_key(s)[0], s)
    jobd: dict[str, Skill] = {}
    for s in job.skills:
        jobd.setdefault(_key(s)[0], s)
    matched = [SkillMatch(k, js.canonical_skill_name or js.surface_form, prof[k].surface_form, js.surface_form, _key(js)[1])
               for k, js in jobd.items() if k in prof]
    missing = [s for k, s in jobd.items() if k not in prof]
    extra = [s for k, s in prof.items() if k not in jobd]
    union = len(set(prof) | set(jobd))
    return AnalysisResult(
        profile=profile, job=job, matched=matched, missing=missing, extra=extra,
        coverage=(len(matched) / len(jobd)) if jobd else 1.0,
        jaccard=(len(set(prof) & set(jobd)) / union) if union else 0.0,
        method="set_overlap",
    )


def _course_skill(s: Skill) -> dict[str, Any]:
    return {"surface_form": s.surface_form, "label": s.label, "start": s.start, "end": s.end, "source": s.provenance.extractor,
            "confidence": s.confidence, "evidence": s.evidence, "canonical_skill_id": s.canonical_skill_id,
            "canonical_skill_name": s.canonical_skill_name, "metadata": {}}


class _ExtractionReplay:
    """Presents already-computed library results as the extractor the inference pipeline expects
    (``load`` / ``is_model_loaded`` / ``extract(text) -> candidate dicts``), so the pipeline does not run the model again."""

    def __init__(self, llm: "SkillLLM", results: Sequence[ExtractionResult], model_backed: bool) -> None:
        self.llm = llm
        self.model_backed = model_backed
        self.cache = {r.text: r for r in results}

    def load(self) -> None:
        return None

    def is_model_loaded(self) -> bool:
        return self.model_backed

    def extract(self, text: str) -> list[dict[str, Any]]:
        result = self.cache.get(text) or self.llm.extract(text)
        source = "dictionary" if self.llm.is_offline else "mistral"
        out = []
        for s in result.skills:
            out.append({
                "surface_form": s.surface_form, "label": s.label, "start": s.start, "end": s.end, "source": source,
                "confidence": s.confidence, "evidence": s.evidence, "canonical_skill_id": s.canonical_skill_id,
                "canonical_skill_name": s.canonical_skill_name,
                "metadata": {"votes": s.votes, "n_systems": s.n_systems, "voted_by": list(s.provenance.systems),
                             "raw_confidence": s.provenance.raw_confidence},
            })
        return out


def _try_inference_pipeline(llm: "SkillLLM", profile: ExtractionResult, job: ExtractionResult) -> dict[str, Any] | None:
    """Delegate gap/match to ``skillllm_v2.pipeline.inference.SkillLLMV2Pipeline`` (course stage skipped).

    The pipeline's anchor is this library's own, already-computed extraction (no second model run). Returns the
    pipeline's JSON-able result, or None when the pipeline module cannot be imported or fails (the caller then
    keeps the plain set-overlap analysis and says so)."""
    try:
        import importlib

        module = importlib.import_module("skillllm_v2.pipeline.inference")
        replay = _ExtractionReplay(llm, [profile, job], model_backed=not llm.is_offline)
        table = llm.alias_table or str(DEFAULT_ALIAS_TABLE)
        config: dict[str, Any] = {"inference": {
            "taxonomy": {"alias_table_paths": [str(Path(table).resolve())]},
            "courses": {"port": "null"},
            "gliner": {"enabled": False},
            "anchor": {"mode": "mistral" if not llm.is_offline else "none"},
        }}
        kwargs: dict[str, Any] = {"mistral": replay} if not llm.is_offline else {"dictionary": replay}
        pipeline = module.SkillLLMV2Pipeline(config, **kwargs)
        result = pipeline.run(profile.text, job.text, skip=("gliner", "courses"))
        data = result.to_dict()
        return {"pipeline_version": data.get("trace", {}).get("pipeline_version"), "match": data.get("match"),
                "graph": data.get("graph"), "explanation": data.get("explanation"), "trace": data.get("trace")}
    except Exception as exc:  # noqa: BLE001 - the pipeline is developed separately; never break the library over it
        _PIPELINE_ERRORS.append(f"{type(exc).__name__}: {exc}")
        return None


_PIPELINE_ERRORS: list[str] = []
