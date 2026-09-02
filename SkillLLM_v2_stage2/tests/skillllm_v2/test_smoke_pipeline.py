from skillllm_v2.pipeline.pipeline_runner import PipelineRunner


def test_smoke_pipeline_schema_stage() -> None:
    runner = PipelineRunner("configs/skillllm_v2_smoke.yaml")
    runner.run_stage("setup_stage2_repo", mode="smoke", force=True)
    runner.run_stage("legacy_cleanup_check", mode="smoke", force=True)
    result = runner.run_stage("schema_smoke", mode="smoke", force=True)
    assert result["status"] == "success"
