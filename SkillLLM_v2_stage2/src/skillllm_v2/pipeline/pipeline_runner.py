from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from skillllm_v2.pipeline.artifact_tracker import ArtifactTracker, utc_now
from skillllm_v2.pipeline.gates import GateError, require_success
from skillllm_v2.pipeline.stage_registry import STAGE_ORDER, Stage, build_registry


class PipelineRunner:
    def __init__(self, config_path: str, artifacts_dir: str = "artifacts/skillllm_v2") -> None:
        self.config_path = config_path
        self.registry = build_registry()
        self.tracker = ArtifactTracker(artifacts_dir)

    def run(
        self,
        mode: str = "smoke",
        limit: int | None = None,
        force: bool = False,
        force_stage: str | None = None,
        start_at: str | None = None,
        stop_after: str | None = None,
        dry_run: bool = False,
        resume: bool = True,
        skip_optional: bool = False,
    ) -> list[dict[str, Any]]:
        selected = self._selected_stages(start_at, stop_after)
        results = []
        for stage_name in selected:
            stage = self.registry[stage_name]
            if skip_optional and stage.optional:
                continue
            should_force = force or force_stage == stage_name
            if dry_run:
                print(stage_name)
                continue
            ran_gate_smoke = False
            if mode == "full" and not self.tracker.has_success(stage_name):
                self._run_one(stage, "smoke", limit, force=True)
                ran_gate_smoke = True
            if resume and not ran_gate_smoke and not should_force and self.tracker.has_success(stage_name) and all(Path(path).exists() for path in stage.outputs):
                results.append({"stage": stage_name, "status": "skipped_existing"})
                continue
            results.append(self._run_one(stage, mode, limit, force=should_force))
        return results

    def run_stage(self, stage_name: str, mode: str = "smoke", limit: int | None = None, force: bool = False) -> dict[str, Any]:
        if stage_name not in self.registry:
            raise KeyError(f"Unknown stage: {stage_name}")
        stage = self.registry[stage_name]
        if mode == "full" and not self.tracker.has_success(stage_name):
            self._run_one(stage, "smoke", limit, force=True)
        return self._run_one(stage, mode, limit, force=force)

    def _run_one(self, stage: Stage, mode: str, limit: int | None, force: bool) -> dict[str, Any]:
        if stage.required_before:
            require_success(self.tracker, stage.required_before)
        started = utc_now()
        started_time = time.perf_counter()
        try:
            metrics = stage.runner(self.config_path, mode, limit)
            for output in stage.outputs:
                if not Path(output).exists():
                    raise GateError(f"Stage {stage.stage_name} did not create required output: {output}")
            self.tracker.write_status(stage.stage_name, stage.system_name, "success", started, started_time, self.config_path, stage.inputs, stage.outputs, metrics)
            return {"stage": stage.stage_name, "status": "success", "metrics": metrics}
        except Exception as exc:
            self.tracker.write_status(stage.stage_name, stage.system_name, "failed", started, started_time, self.config_path, stage.inputs, stage.outputs, error=str(exc))
            raise

    def _selected_stages(self, start_at: str | None, stop_after: str | None) -> list[str]:
        stages = list(STAGE_ORDER)
        if start_at:
            stages = stages[stages.index(start_at) :]
        if stop_after:
            stages = stages[: stages.index(stop_after) + 1]
        return stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/skillllm_v2_smoke.yaml")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--stage")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-stage")
    parser.add_argument("--start-at")
    parser.add_argument("--stop-after")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--skip-optional", action="store_true")
    # Stale-artifact remediation flags
    parser.add_argument(
        "--invalidate-downstream",
        metavar="STAGE",
        help="Delete success markers for STAGE and all stages after it, then run normally.",
    )
    parser.add_argument(
        "--ignore-stale-markers",
        action="store_true",
        help="Treat all success markers as absent (equivalent to --force for every stage).",
    )
    parser.add_argument("--fresh-run-id", metavar="ID", help="Tag written into status files for traceability.")
    args = parser.parse_args()
    runner = PipelineRunner(args.config)

    if args.invalidate_downstream:
        removed = runner.tracker.invalidate_downstream(args.invalidate_downstream)
        print(f"Invalidated {len(removed)} success marker(s): {', '.join(removed)}")

    force = args.force or args.ignore_stale_markers

    if args.stage:
        runner.run_stage(args.stage, args.mode, args.limit, force)
    else:
        runner.run(args.mode, args.limit, force, args.force_stage, args.start_at, args.stop_after, args.dry_run, args.resume, args.skip_optional)


if __name__ == "__main__":
    main()
