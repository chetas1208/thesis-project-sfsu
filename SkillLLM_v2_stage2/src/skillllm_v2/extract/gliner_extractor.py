from __future__ import annotations

import os
from typing import Any

from skillllm_v2.schema import SkillCandidate


def resolve_gliner_device(requested: str | None = None) -> str:
    requested = requested or os.environ.get("SKILLLM_V2_GLINER_DEVICE", "auto")
    if requested != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class GLiNERExtractor:
    """Optional GLiNER recall booster.

    GLiNER is auxiliary in Stage-2. Missing dependencies never block the main
    Mistral path; they produce an empty candidate list unless fallback mode is on.
    """

    def __init__(self, model_name: str | None = None, allow_fallback: bool = True, threshold: float = 0.35, device: str | None = None) -> None:
        self.model_name = model_name or "urchade/gliner_medium-v2.1"
        self.allow_fallback = allow_fallback
        self.threshold = threshold
        self.device = resolve_gliner_device(device)
        self._model: Any | None = None

    def is_model_loaded(self) -> bool:
        """True only when a real GLiNER model is loaded (False means ``extract`` returns [] or the hard-coded fallback)."""
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from gliner import GLiNER
        except ImportError:
            return
        # huggingface_hub>=1.x dropped 'proxies'/'resume_download' from the
        # kwargs it passes to _from_pretrained; give them defaults so GLiNER
        # 0.2.x (which declares them as required keyword-only args) still works.
        # Keep this idempotent: ablations may instantiate GLiNERExtractor many
        # times in one Python process.
        if not getattr(GLiNER, "_skillllm_v2_hub_compat_patched", False):
            _orig = GLiNER._from_pretrained.__func__

            def _compat(cls, *, proxies=None, resume_download=False, **kw):
                return _orig(cls, proxies=proxies, resume_download=resume_download, **kw)

            GLiNER._from_pretrained = classmethod(_compat)
            GLiNER._skillllm_v2_hub_compat_patched = True
        self._model = GLiNER.from_pretrained(self.model_name)
        if hasattr(self._model, "to"):
            self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def extract(self, text: str) -> list[dict[str, Any]]:
        self.load()
        if self._model is None:
            return self._fallback(text) if self.allow_fallback else []
        entities = self._model.predict_entities(text, ["hard skill", "soft skill"], threshold=self.threshold)
        output: list[dict[str, Any]] = []
        for entity in entities:
            label = "SOFT_SKILL" if "soft" in str(entity.get("label", "")).lower() else "HARD_SKILL"
            start, end = int(entity["start"]), int(entity["end"])
            surface = text[start:end]
            output.append(
                SkillCandidate(
                    surface,
                    label,
                    start,
                    end,
                    "gliner",
                    float(entity.get("score", 0.55)),
                    surface,
                    metadata={"gliner_label": entity.get("label"), "device": self.device},
                ).to_dict()
            )
        return output

    def _fallback(self, text: str) -> list[dict[str, Any]]:
        candidates = []
        for phrase in ["teamwork", "collaboration", "problem solving", "communication"]:
            start = text.lower().find(phrase)
            if start >= 0:
                end = start + len(phrase)
                candidates.append(SkillCandidate(text[start:end], "SOFT_SKILL", start, end, "gliner", 0.48, text[start:end]).to_dict())
        return candidates
