from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from skillllm_v2.schema import SkillCandidate, find_offsets

SYSTEM_PROMPT = (
    "You are SkillsLLM, an information extraction model specialized in identifying skill mentions "
    "from resumes, job postings, and professional text. Your job is to extract only explicit skill "
    "spans from the input text. Do not infer skills that are not directly stated. Return valid JSON only."
)

USER_PROMPT_TEMPLATE = """Extract all skill spans from the text below.

Rules:
- Extract exact surface spans from the input text.
- Do not paraphrase.
- Do not normalize names unless a separate normalized field is requested.
- Do not invent missing skills.
- Include technical skills, tools, programming languages, frameworks, libraries, platforms, methodologies, domain skills, and soft skills only when explicitly mentioned.
- Ignore job titles, company names, degrees, dates, locations, and generic responsibilities unless they are explicitly skill mentions.
- If no skills are present, return an empty list.
- Return valid JSON only.

Text:
{text}

Output JSON schema:
{{
  "skills": [
    {{
      "text": "exact skill span from input",
      "label": "HARD_SKILL or SOFT_SKILL or SKILL"
    }}
  ]
}}

Important:
- The "text" value must exactly match a substring from the input.
- Use "HARD_SKILL" for technical/domain/tool skills.
- Use "SOFT_SKILL" for interpersonal or behavioral skills.
- Use "SKILL" only if the dataset does not distinguish hard and soft skills.
"""

# Compact prompt ("short"): same output schema and label set, no rules/few-shot prose (~41 fixed tokens instead of ~300).
# NOTE: the template is only meaningful for adapters TRAINED with it; the shipped anchor/v1/v2 adapters use "original".
SYSTEM_PROMPT_SHORT = "Extract skill spans."

USER_PROMPT_TEMPLATE_SHORT = """Output {{"skills":[{{"text":"<exact span>","label":"HARD_SKILL|SOFT_SKILL"}}]}}

Text:
{text}"""

DEFAULT_PROMPT_VERSION = "original"
PROMPT_VERSIONS: dict[str, tuple[str, str]] = {
    "original": (SYSTEM_PROMPT, USER_PROMPT_TEMPLATE),
    "short": (SYSTEM_PROMPT_SHORT, USER_PROMPT_TEMPLATE_SHORT),
}


def build_prompt_messages(text: str, prompt_version: str = DEFAULT_PROMPT_VERSION) -> list[dict[str, str]]:
    """Chat messages for ``text`` under ``prompt_version`` (default = the original prompt, byte-identical to before)."""
    try:
        system, template = PROMPT_VERSIONS[prompt_version]
    except KeyError:
        raise ValueError(f"unknown prompt_version {prompt_version!r}; choose one of {sorted(PROMPT_VERSIONS)}") from None
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": template.format(text=text)},
    ]


class MistralExtractor:
    """Mistral Small 3.1 24B LoRA extractor with smoke-mode fallback.

    The class contains no Gemma, Qwen, or Llama branches. If the configured model
    files are unavailable, smoke runs can still exercise parsing/alignment through
    a deterministic local candidate source.
    """

    def __init__(
        self,
        model_path: str | None = None,
        adapter_path: str | None = None,
        use_4bit: bool = True,
        max_new_tokens: int = 512,
        fallback_aliases: list[str] | None = None,
        allow_fallback: bool = True,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        record_logprobs: bool = False,
    ) -> None:
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"unknown prompt_version {prompt_version!r}; choose one of {sorted(PROMPT_VERSIONS)}")
        self.prompt_version = prompt_version
        self.record_logprobs = record_logprobs
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self.fallback_aliases = fallback_aliases or [
            "Python",
            "SQL",
            "machine learning",
            "communication",
            "leadership",
            "AWS",
            "Amazon Web Services",
            "Docker",
        ]
        self.allow_fallback = allow_fallback
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self.load_error: str | None = None

    def is_model_loaded(self) -> bool:
        """True only when real model weights are loaded (False means ``extract`` uses the deterministic fallback)."""
        return self._model is not None and self._tokenizer is not None

    def build_messages(self, text: str) -> list[dict[str, str]]:
        return build_prompt_messages(text, self.prompt_version)

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path or not Path(self.model_path).exists():
            if self.allow_fallback:
                return
            raise FileNotFoundError(f"Mistral model path does not exist: {self.model_path}")
        try:
            import torch
            import transformers
            from transformers import AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            if self.allow_fallback:
                return
            raise RuntimeError("transformers, torch, and bitsandbytes are required for Mistral inference") from exc

        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        try:
            self._tokenizer = _load_tokenizer(AutoTokenizer, self.model_path)
            _ensure_chat_template(self._tokenizer, self.model_path)
            self._model = _load_model(
                transformers,
                self.model_path,
                {
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                    "trust_remote_code": True,
                },
            )
            if self.adapter_path and Path(self.adapter_path).exists():
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        except Exception as exc:
            self.load_error = str(exc)
            self._model = None
            self._tokenizer = None
            if not self.allow_fallback:
                raise

    def extract(self, text: str) -> list[dict[str, Any]]:
        start_time = time.perf_counter()
        self.load()
        if self._model is None or self._tokenizer is None:
            skills = self._fallback_extract(text)
        else:
            if self.record_logprobs:
                raw_output, token_info = self._generate_with_logprobs(text)
            else:
                raw_output, token_info = self._generate(text), None
            parsed = parse_skill_json(raw_output)
            skills = self._skills_from_payload(text, parsed)
            if token_info is not None:
                attach_span_logprobs(skills, raw_output, token_info, self._tokenizer)
        latency = time.perf_counter() - start_time
        for skill in skills:
            skill.metadata.setdefault("latency_seconds", latency)
            if self.load_error:
                skill.metadata.setdefault("mistral_load_error", self.load_error)
        return [skill.to_dict() for skill in skills]

    def _generate(self, text: str) -> str:
        messages = self.build_messages(text)
        tokenizer = self._tokenizer
        model = self._model
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_with_logprobs(self, text: str) -> tuple[str, dict[str, Any]]:
        """Greedy generation that also returns the log-probability of every generated token (same decoding as ``_generate``)."""
        import torch

        messages = self.build_messages(text)
        tokenizer = self._tokenizer
        model = self._model
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True
        )
        gen_ids = out.sequences[0][inputs["input_ids"].shape[-1] :]
        logprobs: list[float] = []
        for step, ids in enumerate(gen_ids.tolist()):
            if step >= len(out.scores):
                break
            logprobs.append(float(torch.log_softmax(out.scores[step][0].float(), dim=-1)[ids]))
        ids_list = gen_ids.tolist()[: len(logprobs)]
        return tokenizer.decode(gen_ids, skip_special_tokens=True), {"ids": ids_list, "logprobs": logprobs}

    def _skills_from_payload(self, text: str, payload: dict[str, Any] | list[Any]) -> list[SkillCandidate]:
        rows = payload.get("skills", []) if isinstance(payload, dict) else payload
        used: set[tuple[int, int]] = set()
        candidates: list[SkillCandidate] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            surface = str(row.get("text") or row.get("surface_form") or "").strip()
            if not surface:
                continue
            label = str(row.get("label") or "HARD_SKILL")
            if label == "SKILL":
                label = "HARD_SKILL"
            if label not in {"HARD_SKILL", "SOFT_SKILL"}:
                label = "HARD_SKILL"
            start, end = find_offsets(text, surface, used)
            if start is not None and end is not None:
                used.add((start, end))
                evidence = text[start:end]
            else:
                evidence = surface
            candidates.append(
                SkillCandidate(surface, label, start, end, "mistral", 0.86, evidence, metadata={"raw_label": row.get("label")})
            )
        return candidates

    def _fallback_extract(self, text: str) -> list[SkillCandidate]:
        raw = {"skills": [{"text": alias, "label": _fallback_label(alias)} for alias in self.fallback_aliases if alias.lower() in text.lower()]}
        return self._skills_from_payload(text, raw)


_SPAN_VALUE_RE = re.compile(r'"(?:text|surface_form)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def span_value_char_ranges(raw_output: str) -> list[tuple[int, int, str]]:
    """(start, end, decoded value) of every ``"text"``/``"surface_form"`` JSON string value in the raw model output."""
    out: list[tuple[int, int, str]] = []
    for match in _SPAN_VALUE_RE.finditer(raw_output):
        try:
            value = json.loads('"' + match.group(1) + '"')
        except json.JSONDecodeError:
            value = match.group(1)
        out.append((match.start(1), match.end(1), value))
    return out


def token_char_ends(tokenizer: Any, ids: list[int]) -> list[int]:
    """Cumulative decoded character length after each generated token (monotone, clipped to the final decode)."""
    ends: list[int] = []
    prev = 0
    for k in range(1, len(ids) + 1):
        prev = max(prev, len(tokenizer.decode(ids[:k], skip_special_tokens=True)))
        ends.append(prev)
    return ends


def span_token_logprob_stats(
    token_logprobs: list[float], char_ends: list[int], char_start: int, char_end: int
) -> dict[str, Any] | None:
    """sum / mean / min log-probability over the generated tokens that overlap ``[char_start, char_end)``."""
    picked: list[float] = []
    prev_end = 0
    for lp, end in zip(token_logprobs, char_ends):
        if end > char_start and prev_end < char_end and end > prev_end:
            picked.append(lp)
        prev_end = end
    if not picked:
        return None
    return {"sum": sum(picked), "mean": sum(picked) / len(picked), "min": min(picked), "n_tokens": len(picked)}


def attach_span_logprobs(skills: list[SkillCandidate], raw_output: str, token_info: dict[str, Any], tokenizer: Any) -> None:
    """Store ``metadata['logprob']`` on each skill (sum/mean/min token log-prob of its generated ``text`` string).

    Skills are the parsed rows in order, so they line up 1:1 with the ``"text"`` values found in the raw output. If the
    counts or strings disagree (e.g. a malformed row) every skill gets ``metadata['logprob'] = None`` rather than a wrong
    number.
    """
    ranges = span_value_char_ranges(raw_output)
    aligned = len(ranges) == len(skills) and all(value.strip() == skill.surface_form for (_, _, value), skill in zip(ranges, skills))
    if aligned:
        ends = token_char_ends(tokenizer, token_info["ids"])
    for index, skill in enumerate(skills):
        if not aligned:
            skill.metadata["logprob"] = None
            continue
        start, end, _ = ranges[index]
        skill.metadata["logprob"] = span_token_logprob_stats(token_info["logprobs"], ends, start, end)


def parse_skill_json(raw_output: str) -> dict[str, Any] | list[Any]:
    cleaned = raw_output.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse Mistral JSON output: {raw_output[:200]}")


def _load_tokenizer(auto_tokenizer: Any, model_path: str, trust_remote_code: bool = True) -> Any:
    """Load the tokenizer. `trust_remote_code` defaults to True to keep the research scripts' behaviour on local
    weights; the public library (`skillllm_v2.api`) passes its own flag, which defaults to False."""
    attempts = [
        {"trust_remote_code": trust_remote_code, "fix_mistral_regex": True},
        {"trust_remote_code": trust_remote_code},
        {"trust_remote_code": trust_remote_code, "use_fast": False},
    ]
    errors = []
    for kwargs in attempts:
        try:
            return auto_tokenizer.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Failed to load tokenizer for {model_path}: {' | '.join(errors)}")


def _load_model(transformers: Any, model_path: str, model_kwargs: dict[str, Any]) -> Any:
    class_names = ["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"]
    errors = []
    for class_name in class_names:
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            return model_class.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Failed to load model for {model_path}: {' | '.join(errors)}")


def _ensure_chat_template(tokenizer: Any, model_path: str) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    template_file = Path(model_path) / "chat_template.json"
    if template_file.exists():
        try:
            payload = json.loads(template_file.read_text(encoding="utf-8"))
            template = payload.get("chat_template") if isinstance(payload, dict) else None
            if template:
                tokenizer.chat_template = template
                return
        except Exception:
            pass
    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}[SYSTEM_PROMPT]{{ message['content'] }}[/SYSTEM_PROMPT]{% endif %}"
        "{% if message['role'] == 'user' %}[INST]{{ message['content'] }}[/INST]{% endif %}"
        "{% if message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    )


def _fallback_label(surface: str) -> str:
    if surface.lower() in {"communication", "leadership", "collaboration"}:
        return "SOFT_SKILL"
    return "HARD_SKILL"
