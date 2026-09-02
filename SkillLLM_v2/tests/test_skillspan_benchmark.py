from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.benchmark.alignment import align_predicted_skills
from src.benchmark.dataset import derive_smoke_output_dir, discover_dataset
from src.benchmark.parsing import parse_benchmark_output
from src.benchmark.pipeline import load_existing_model_result, normalize_predictions, resolve_model_config, should_skip_model
from src.runtime_utils import (
    build_bitsandbytes_config,
    patch_bitsandbytes_param_constructors,
    read_jsonl,
    serialize_skills,
    write_json,
    write_jsonl,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


class SkillSpanBenchmarkTests(unittest.TestCase):
    def test_discover_dataset_detects_jsonl_style_json_splits(self) -> None:
        discovery = discover_dataset(ROOT_DIR / "data" / "source" / "skillspan_hf")
        self.assertEqual(discovery.dataset_type, "jsonl")
        self.assertTrue(discovery.split_files["train"].name.endswith("train.json"))
        self.assertTrue(discovery.split_files["dev"].name.endswith("dev.json"))
        self.assertTrue(discovery.split_files["test"].name.endswith("test.json"))

    def test_label_only_skill_serialization(self) -> None:
        payload = serialize_skills(
            [
                {"text": "Python", "label": "HARD_SKILL", "start": 0, "end": 6},
                {"text": "communication", "label": "SOFT_SKILL", "start": 11, "end": 24},
            ],
            include_offsets=False,
        )
        self.assertNotIn('"start"', payload)
        self.assertNotIn('"end"', payload)
        self.assertIn('"label":"HARD_SKILL"', payload)

    def test_parse_clean_json(self) -> None:
        result = parse_benchmark_output('{"skills":[{"text":"Python","label":"HARD_SKILL"}]}')
        self.assertFalse(result.parse_failed)
        self.assertEqual(result.skills[0]["text"], "Python")

    def test_parse_markdown_fenced_json(self) -> None:
        result = parse_benchmark_output('```json\n{"skills":[{"text":"Leadership","label":"SOFT_SKILL"}]}\n```')
        self.assertFalse(result.parse_failed)
        self.assertEqual(result.skills[0]["label"], "SOFT_SKILL")

    def test_parse_recovers_first_json_object(self) -> None:
        result = parse_benchmark_output(
            'Here is the answer {"skills":[{"text":"SQL","label":"hard_skill"}]} trailing text'
        )
        self.assertFalse(result.parse_failed)
        self.assertEqual(result.skills[0]["label"], "HARD_SKILL")

    def test_parse_failure_is_reported(self) -> None:
        result = parse_benchmark_output("not valid output")
        self.assertTrue(result.parse_failed)
        self.assertTrue(result.parse_error)

    def test_alignment_uses_order_for_duplicate_spans(self) -> None:
        aligned, unaligned = align_predicted_skills(
            "Leadership and leadership matter.",
            [
                {"text": "leadership", "label": "SOFT_SKILL"},
                {"text": "leadership", "label": "SOFT_SKILL"},
            ],
        )
        self.assertEqual(unaligned, 0)
        self.assertEqual(aligned[0]["start"], 15)
        self.assertEqual(aligned[1]["start"], 0)

    def test_alignment_case_insensitive_fallback(self) -> None:
        aligned, unaligned = align_predicted_skills(
            "Experience with SQL is required.",
            [{"text": "sql", "label": "HARD_SKILL"}],
        )
        self.assertEqual(unaligned, 0)
        self.assertEqual(aligned[0]["text"], "SQL")
        self.assertEqual(aligned[0]["alignment_method"], "case_insensitive")

    def test_alignment_unresolved_offsets_are_null(self) -> None:
        aligned, unaligned = align_predicted_skills(
            "Experience with SQL is required.",
            [{"text": "Python", "label": "HARD_SKILL"}],
        )
        self.assertEqual(unaligned, 1)
        self.assertIsNone(aligned[0]["start"])
        self.assertIsNone(aligned[0]["end"])

    def test_smoke_output_dir_derivation(self) -> None:
        path = derive_smoke_output_dir("results/skillspan_baselines", 5)
        self.assertEqual(path.as_posix(), "results/skillspan_baselines_smoke_limit_5")

    def test_skip_logic_requires_predictions_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            artifacts = {
                "predictions": base / "predictions.jsonl",
                "metrics": base / "metrics.json",
            }
            artifacts["predictions"].write_text("", encoding="utf-8")
            self.assertFalse(should_skip_model(artifacts, overwrite=False))
            write_json(artifacts["metrics"], {"ok": True})
            self.assertTrue(should_skip_model(artifacts, overwrite=False))
            self.assertFalse(should_skip_model(artifacts, overwrite=True))

    def test_existing_model_result_preserves_failure_from_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model-a"
            model_dir.mkdir()
            write_jsonl(
                model_dir / "errors.jsonl",
                [{"type": "model_failure", "model_name": "model-a", "error": "load failed"}],
            )

            result = load_existing_model_result(temp_dir, "model-a")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_reason"], "load failed")

    def test_gemma_26b_benchmark_config_enables_cpu_offload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "benchmark": {"batch_size": 1},
            }
            model_config = {"name": "gemma-4-26b-a4b-it", "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024}
            resolved = resolve_model_config(
                model_name="gemma-4-26b-a4b-it",
                benchmark_config=config,
                benchmark_model_config=model_config,
                prepared_dir=Path(temp_dir),
                model_dir=Path(temp_dir) / "gemma-4-26b-a4b-it",
            )

            self.assertTrue(resolved["use_4bit"])
            self.assertFalse(resolved["use_8bit"])
            self.assertTrue(resolved["quantization_cpu_offload"])
            self.assertEqual(resolved["max_seq_length"], 256)
            self.assertEqual(resolved["max_memory"][0], "14GiB")
            self.assertEqual(resolved["max_memory"][1], "14GiB")
            self.assertEqual(resolved["preferred_model_classes"][0], "AutoModelForImageTextToText")
            self.assertFalse(resolved["skip_prepare_model_for_kbit_training"])
            self.assertIn("offload", resolved["offload_folder"])

    def test_gemma_e4b_benchmark_config_reduces_gpu0_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "benchmark": {"batch_size": 1},
            }
            model_config = {"name": "gemma-4-e4b-it", "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024}
            resolved = resolve_model_config(
                model_name="gemma-4-e4b-it",
                benchmark_config=config,
                benchmark_model_config=model_config,
                prepared_dir=Path(temp_dir),
                model_dir=Path(temp_dir) / "gemma-4-e4b-it",
            )

            self.assertTrue(resolved["use_4bit"])
            self.assertEqual(resolved["cuda_visible_devices"], "1")
            self.assertEqual(resolved["device_map"], {"": 0})
            self.assertEqual(resolved["max_memory"][0], "22GiB")
            self.assertEqual(resolved["max_seq_length"], 512)
            self.assertEqual(resolved["preferred_model_classes"][0], "AutoModelForImageTextToText")

    def test_bitsandbytes_config_accepts_cpu_offload_flag(self) -> None:
        import torch

        quantization_config = build_bitsandbytes_config(
            {"quantization_cpu_offload": True},
            dtype=torch.float16,
        )
        self.assertTrue(quantization_config.load_in_4bit)
        self.assertTrue(quantization_config.llm_int8_enable_fp32_cpu_offload)

    def test_bitsandbytes_config_supports_8bit_cpu_offload(self) -> None:
        import torch

        quantization_config = build_bitsandbytes_config(
            {"use_8bit": True, "quantization_cpu_offload": True},
            dtype=torch.float16,
        )
        self.assertTrue(quantization_config.load_in_8bit)
        self.assertTrue(quantization_config.llm_int8_enable_fp32_cpu_offload)

    def test_bitsandbytes_params_accept_transformers_init_flag(self) -> None:
        import torch
        from bitsandbytes.autograd._functions import MatmulLtState
        from bitsandbytes.nn.modules import Int8Params, Params4bit

        patch_bitsandbytes_param_constructors()
        int8_param = Int8Params(torch.zeros(1), requires_grad=False, _is_hf_initialized=True)
        params4bit = Params4bit(torch.zeros(1), requires_grad=False, _is_hf_initialized=True)

        self.assertEqual(tuple(int8_param.shape), (1,))
        self.assertEqual(tuple(params4bit.shape), (1,))
        self.assertFalse(MatmulLtState().memory_efficient_backward)

    def test_successful_retry_does_not_keep_stale_failure_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            model_dir = base / "llama-3.1-8b-instruct"
            model_dir.mkdir()
            write_json(model_dir / "metrics.json", {"exact_span_f1": 1.0})
            write_jsonl(
                model_dir / "errors.jsonl",
                [{"type": "model_failure", "model_name": "llama-3.1-8b-instruct", "error": "old failure"}],
            )

            result = load_existing_model_result(base, "llama-3.1-8b-instruct")
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("failure_reason", result)

    def test_normalize_predictions_records_parse_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            raw_file = base / "raw.jsonl"
            predictions_file = base / "predictions.jsonl"
            errors_file = base / "errors.jsonl"
            write_jsonl(
                raw_file,
                [
                    {
                        "id": "1",
                        "text": "Python and communication",
                        "raw_output": '{"skills":[{"text":"Python","label":"HARD_SKILL"}]}',
                        "generation_seconds": 0.1,
                    },
                    {
                        "id": "2",
                        "text": "No skills here",
                        "raw_output": "broken output",
                        "generation_seconds": 0.2,
                    },
                ],
            )

            summary = normalize_predictions(raw_file, predictions_file, errors_file)
            predictions = read_jsonl(predictions_file)
            errors = read_jsonl(errors_file)

            self.assertEqual(summary["parse_failures"], 1)
            self.assertEqual(len(predictions), 2)
            self.assertTrue(any(row["type"] == "parse_failure" for row in errors))


if __name__ == "__main__":
    unittest.main()
