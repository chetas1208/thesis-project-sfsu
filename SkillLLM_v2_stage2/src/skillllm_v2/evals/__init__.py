"""Module-level evaluation suite for SkillLLM_v2 (distinct from the legacy `eval/` package).

Every eval module exposes `run(ctx: EvalContext) -> list[EvalResult]` and is registered in `registry.py`.
Results are plain data so one runner can aggregate them into a single reproducible report.
"""

from .contract import EvalContext, EvalResult, bootstrap_ci, paired_bootstrap

__all__ = ["EvalContext", "EvalResult", "bootstrap_ci", "paired_bootstrap"]
