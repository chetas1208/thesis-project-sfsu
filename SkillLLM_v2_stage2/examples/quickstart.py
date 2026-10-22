"""Quick start for the skillllm library.

Offline mode runs anywhere (no model, no GPU):

    python examples/quickstart.py

Model mode needs `pip install "skillllm[model]"`, a CUDA GPU with ~16 GB free, the base model directory and adapters:

    SKILLLM_BASE_MODEL=models/mistral-small-3.1-24b-instruct-2503 \
    SKILLLM_ADAPTERS="artifacts/skillllm_v2/mistral_lora/model:artifacts/skillllm_v2/qlora_loop/best:artifacts/skillllm_v2/qlora_loop_v2/best" \
    python examples/quickstart.py --model
"""
from __future__ import annotations

import os
import sys

from skillllm_v2 import SkillLLM, SkillLLMError

RESUME = "Data engineer with 5 years of Python, SQL and Apache Spark; built ETL pipelines on AWS. Strong communication skills."
JOB = "We need a data engineer: Python, SQL, Kubernetes and Docker experience, plus teamwork and communication."


def build(use_model: bool) -> SkillLLM:
    if not use_model:
        return SkillLLM.offline()
    base = os.environ["SKILLLM_BASE_MODEL"]
    adapters = [p for p in os.environ.get("SKILLLM_ADAPTERS", "").split(os.pathsep) if p]
    return SkillLLM.from_pretrained(base, adapters=adapters, min_votes=2, quantization="4bit")


def main() -> int:
    try:
        with build("--model" in sys.argv) as llm:
            result = llm.extract(RESUME)
            if result.metadata.offline:
                print("NOTE: offline dictionary mode, not the trained model\n")
            for s in result.skills:
                print(f"{s.surface_form!r:28} {s.label:11} [{s.start}:{s.end}] conf={s.confidence:.2f} votes={s.votes}/{s.n_systems} -> {s.canonical_skill_name}")
            gap = llm.analyze(RESUME, JOB)
            print("\nmatched:", [m.skill for m in gap.matched])
            print("missing:", [s.canonical_skill_name or s.surface_form for s in gap.missing])
            print("method:", gap.method)
    except SkillLLMError as exc:  # missing extra / weights / CUDA / OOM all arrive as SkillLLMError subclasses
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
