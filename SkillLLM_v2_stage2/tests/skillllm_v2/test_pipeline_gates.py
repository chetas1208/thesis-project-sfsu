from skillllm_v2.pipeline.gates import require_no_legacy_active_paths


def test_no_legacy_active_paths() -> None:
    require_no_legacy_active_paths()
