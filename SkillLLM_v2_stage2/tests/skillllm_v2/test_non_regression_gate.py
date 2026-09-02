from skillllm_v2.eval.non_regression_gate import (
    DEFAULT_GUARDRAILS,
    DEFAULT_GUARDRAILS_PATH,
    evaluate_gate,
    load_guardrails,
    select_extraction,
)

ANCHOR_METRICS = {
    "exact_span_f1": 0.675034,
    "token_f1": 0.797798,
    "soft_skill_f1": 0.609792,
    "hard_skill_f1": 0.731138,
}


def test_load_guardrails_loads_real_config() -> None:
    guardrails = load_guardrails(DEFAULT_GUARDRAILS_PATH)
    assert guardrails["anchor_metrics"]["exact_span_f1"] == ANCHOR_METRICS["exact_span_f1"]
    assert guardrails["anchor_metrics"]["soft_skill_f1"] == ANCHOR_METRICS["soft_skill_f1"]
    assert guardrails["exact_span_f1"]["max_drop"] == 0.010034
    assert guardrails["soft_skill_f1"]["max_drop"] == 0.03


def test_load_guardrails_falls_back_to_default_when_missing() -> None:
    guardrails = load_guardrails("artifacts/skillllm_v2/does_not_exist.yaml")
    assert guardrails == DEFAULT_GUARDRAILS


def test_gate_accepts_and_marks_improved_when_candidate_matches_anchor() -> None:
    result = evaluate_gate("raw_mistral", ANCHOR_METRICS, ANCHOR_METRICS)
    assert result.decision == "accept"
    assert result.improved_over_anchor is True
    for check in result.checks:
        assert check.passed is True
        assert check.delta == 0.0


def test_gate_accepts_as_non_regressive_below_target() -> None:
    candidate = dict(ANCHOR_METRICS, exact_span_f1=0.670000)
    result = evaluate_gate("boundary_refiner", candidate, ANCHOR_METRICS)
    assert result.decision == "accept"
    assert result.improved_over_anchor is False
    assert "non-regressive" in result.reason


def test_gate_falls_back_when_exact_span_f1_drops_too_much() -> None:
    candidate = dict(ANCHOR_METRICS, exact_span_f1=0.570414)
    result = evaluate_gate("mistral_boundary", candidate, ANCHOR_METRICS)
    assert result.decision == "fallback_to_anchor"
    exact_check = next(c for c in result.checks if c.metric == "exact_span_f1")
    assert exact_check.passed is False
    assert "exact_span_f1" in result.reason


def test_gate_falls_back_when_soft_skill_f1_drops_too_much() -> None:
    candidate = dict(ANCHOR_METRICS, soft_skill_f1=0.407990)
    result = evaluate_gate("full_cag_graph", candidate, ANCHOR_METRICS)
    assert result.decision == "fallback_to_anchor"
    soft_check = next(c for c in result.checks if c.metric == "soft_skill_f1")
    assert soft_check.passed is False


def test_select_extraction_returns_candidate_predictions_when_accepted() -> None:
    anchor_preds = [["anchor"]]
    candidate_preds = [["candidate"]]
    selected, result = select_extraction("raw_mistral", anchor_preds, candidate_preds, ANCHOR_METRICS, ANCHOR_METRICS)
    assert result.decision == "accept"
    assert selected is candidate_preds


def test_select_extraction_returns_anchor_predictions_on_fallback() -> None:
    anchor_preds = [["anchor"]]
    candidate_preds = [["candidate"]]
    candidate_metrics = dict(ANCHOR_METRICS, exact_span_f1=0.0, soft_skill_f1=0.0)
    selected, result = select_extraction("full_cag_graph", anchor_preds, candidate_preds, candidate_metrics, ANCHOR_METRICS)
    assert result.decision == "fallback_to_anchor"
    assert selected is anchor_preds
