"""learned_verifier.py

Optional learned layer on top of the rule-based :class:`SpanVerifier`: loads the
scikit-learn TF-IDF/LogReg model written by ``train/train_span_verifier.py``
(``artifacts/skillllm_v2/span_verifier/model/verifier_model.joblib``) and scores
``"<surface_form> [SEP] <text>"`` (the training input format).

Graceful absence: when joblib/scikit-learn or the model file is missing,
``available`` is False and :meth:`filter` returns the candidates unchanged.

Policy (same as every other verifier stage): anchor-derived candidates are
annotated with the model probability but NEVER rejected; only non-anchor
candidates below ``threshold`` are dropped. The model has not been evaluated as
an inference component (see docs/INFERENCE_AND_EVALS.md), so it is off by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from skillllm_v2.policy import ANCHOR_SOURCES, is_anchor_candidate

DEFAULT_MODEL_PATH = "artifacts/skillllm_v2/span_verifier/model/verifier_model.joblib"


class LearnedSpanVerifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, threshold: float = 0.5, protect_anchor_sources: frozenset[str] | None = None) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self.protect_anchor_sources = frozenset(protect_anchor_sources) if protect_anchor_sources is not None else ANCHOR_SOURCES
        self._model: Any | None = None
        self.unavailable_reason: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.unavailable_reason = f"model file not found: {self.model_path}"
            return
        try:
            import joblib

            self._model = joblib.load(self.model_path)
        except Exception as exc:  # missing sklearn/joblib, corrupt file, version skew
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    def score(self, text: str, candidates: list[dict[str, Any]]) -> list[float]:
        if not self.available or not candidates:
            return []
        inputs = [f"{c['surface_form']} [SEP] {text}" for c in candidates]
        classes = list(getattr(self._model, "classes_", [0, 1]))
        positive = classes.index(1) if 1 in classes else -1
        return [float(row[positive]) for row in self._model.predict_proba(inputs)]

    def filter(self, text: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.available:
            return list(candidates)
        output: list[dict[str, Any]] = []
        for candidate, prob in zip(candidates, self.score(text, candidates)):
            item = dict(candidate)
            item["metadata"] = dict(candidate.get("metadata") or {})
            anchor = is_anchor_candidate(candidate, self.protect_anchor_sources)
            keep = anchor or prob >= self.threshold
            item["metadata"]["learned_verifier"] = {"probability": prob, "threshold": self.threshold, "protected_anchor": anchor, "kept": keep}
            if keep:
                output.append(item)
        return output
