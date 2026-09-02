from skillllm_v2.baselines import GLiNERBaseline
from skillllm_v2.extract.gliner_extractor import resolve_gliner_device


def test_gliner_baseline_smoke(tmp_path) -> None:
    baseline = GLiNERBaseline(tmp_path)
    config = baseline.train_or_prepare()
    assert config["training_mode"] in {"zero_shot", "skipped_dependency_missing"}


def test_resolve_gliner_device_respects_explicit_cpu() -> None:
    assert resolve_gliner_device("cpu") == "cpu"
