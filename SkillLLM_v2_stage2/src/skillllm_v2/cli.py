"""``skillllm`` command line: ``extract``, ``doctor`` and ``version`` (all print JSON).

    skillllm extract --offline --text "Python and SQL"
    skillllm extract --base-model models/mistral-small-3.1-24b-instruct-2503 \\
        --adapters a/model b/best c/best --vote majority --file resume.txt
    skillllm doctor [--base-model PATH] [--adapters PATH ...]
    skillllm version

Environment defaults: ``SKILLLM_BASE_MODEL`` and ``SKILLLM_ADAPTERS`` (paths separated by ``os.pathsep``).
Exit codes: 0 ok, 1 runtime error (missing weights/extras/CUDA/OOM), 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from . import __version__, _lazy
from .errors import ConfigError, SkillLLMError

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2


def _dump(obj: Any, pretty: bool) -> str:
    return json.dumps(obj, indent=2 if pretty else None, ensure_ascii=False)


def _env_adapters() -> list[str]:
    raw = os.environ.get("SKILLLM_ADAPTERS", "")
    return [p for p in raw.split(os.pathsep) if p]


def _parse_adapters(values: Sequence[str] | None) -> Any:
    """``a b c`` -> list of paths; ``name=a name2=b`` -> mapping."""
    if not values:
        return None
    if all("=" in v for v in values):
        return dict(v.split("=", 1) for v in values)
    if any("=" in v for v in values):
        raise ConfigError("Give --adapters either all as PATH or all as NAME=PATH.")
    return list(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skillllm", description="SkillLLM_v2 skill extraction.")
    parser.add_argument("--version", action="version", version=f"skillllm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="extract skills from text and print JSON")
    src = ex.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", action="append", help="input text (repeatable)")
    src.add_argument("--file", action="append", help="text file to read, '-' for stdin (repeatable)")
    ex.add_argument("--offline", action="store_true", help="dictionary/alias extractor: no model, no GPU")
    ex.add_argument("--config", help="JSON config written by SkillLLM.save_config()")
    ex.add_argument("--base-model", default=None, help="base model directory (default: $SKILLLM_BASE_MODEL)")
    ex.add_argument("--adapters", nargs="+", help="adapter directories (PATH ... or NAME=PATH ...; default: $SKILLLM_ADAPTERS)")
    ex.add_argument("--vote", default="majority", help="majority | unanimous | union | N (used with several adapters)")
    ex.add_argument("--min-votes", type=int, default=None,
                    help="precision/recall dial with 3 adapters: 3 = high precision (P 0.80 R 0.51), 2 = balanced default "
                         "(P 0.72 R 0.66 F1 0.690), 1 = union (P 0.59 R 0.78). Overrides --vote. All adapters always run (3x cost).")
    ex.add_argument("--calibration", help="JSON vote->precision table (VoteCalibration.to_dict()); default: shipped 3-adapter table")
    ex.add_argument("--no-calibration", action="store_true", help="keep the extractor's constant confidence (0.86) instead of vote-calibrated precision")
    ex.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    ex.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    ex.add_argument("--max-new-tokens", type=int, default=512)
    ex.add_argument("--no-link", action="store_true", help="skip canonical skill linking")
    ex.add_argument("--no-verify", action="store_true", help="skip the rule-based span verifier")
    ex.add_argument("--alias-table", help="alias table JSONL for linking (default: packaged table)")
    ex.add_argument("--out", help="write JSON here instead of stdout")
    ex.add_argument("--pretty", action="store_true", help="indent the JSON output")

    doc = sub.add_parser("doctor", help="report environment, GPU and path readiness as JSON")
    doc.add_argument("--base-model", default=None)
    doc.add_argument("--adapters", nargs="+")
    doc.add_argument("--config", help="also check the paths in this config file")
    doc.add_argument("--pretty", action="store_true")

    ver = sub.add_parser("version", help="print the library version")
    ver.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


# --------------------------------------------------------------------------- extract
def _read_inputs(args: argparse.Namespace) -> list[str]:
    if args.text:
        return list(args.text)
    texts = []
    for name in args.file:
        if name == "-":
            texts.append(sys.stdin.read())
        else:
            path = Path(name)
            if not path.is_file():
                raise ConfigError(f"Input file not found: {name}")
            texts.append(path.read_text(encoding="utf-8"))
    return texts


def _build_model(args: argparse.Namespace) -> Any:
    from .api import SkillLLM

    if args.config:
        return SkillLLM.from_config(args.config)
    link = not args.no_link
    verify = not args.no_verify
    if args.offline:
        return SkillLLM.offline(link=link, verify=verify, alias_paths=[args.alias_table] if args.alias_table else None)
    base = args.base_model or os.environ.get("SKILLLM_BASE_MODEL")
    if not base:
        raise ConfigError(
            "No model selected. Pass --offline for the model-free dictionary extractor, or --base-model PATH "
            "(and --adapters ...), or set SKILLLM_BASE_MODEL / SKILLLM_ADAPTERS."
        )
    adapters = _parse_adapters(args.adapters) or _env_adapters() or None
    vote: str | int | None = int(args.vote) if str(args.vote).isdigit() else args.vote
    calibration: Any = "default"
    if args.no_calibration:
        calibration = None
    elif args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    return SkillLLM.from_pretrained(
        base, min_votes=args.min_votes, calibration=calibration, adapters=adapters, vote=vote, quantization=None if args.quantization == "none" else args.quantization,
        device=args.device, link=link, verify=verify, max_new_tokens=args.max_new_tokens, alias_table=args.alias_table,
    )


def _cmd_extract(args: argparse.Namespace) -> int:
    texts = _read_inputs(args)
    with _build_model(args) as model:
        results = model.extract_batch(texts)
    payload: Any = results[0].to_dict() if len(results) == 1 else [r.to_dict() for r in results]
    out = _dump(payload, args.pretty)
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
    else:
        print(out)
    return EXIT_OK


# --------------------------------------------------------------------------- doctor
def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _gpu_report() -> dict[str, Any]:
    report: dict[str, Any] = {"torch_installed": _lazy.is_available("torch"), "cuda_available": False, "devices": []}
    if report["torch_installed"]:
        try:
            import torch

            report["torch_version"] = torch.__version__
            report["torch_cuda_build"] = torch.version.cuda
            report["cuda_available"] = bool(torch.cuda.is_available())
            for i in range(torch.cuda.device_count() if report["cuda_available"] else 0):
                free, total = torch.cuda.mem_get_info(i)
                report["devices"].append({"index": i, "name": torch.cuda.get_device_name(i), "free_gib": round(free / 2**30, 2),
                                          "total_gib": round(total / 2**30, 2)})
        except Exception as exc:  # noqa: BLE001
            report["error"] = f"{type(exc).__name__}: {exc}"
    if not report["devices"] and shutil.which("nvidia-smi"):  # fallback when torch is absent or CPU-only
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10, check=False).stdout
            for line in out.strip().splitlines():
                idx, name, free, total = [p.strip() for p in line.split(",")]
                report["devices"].append({"index": int(idx), "name": name, "free_gib": round(float(free) / 1024, 2),
                                          "total_gib": round(float(total) / 1024, 2), "via": "nvidia-smi"})
        except Exception:  # noqa: BLE001
            pass
    return report


def _path_check(label: str, raw: str | None, kind: str) -> dict[str, Any]:
    if not raw:
        return {"label": label, "path": None, "exists": None, "note": "not configured"}
    p = Path(raw).expanduser()
    info: dict[str, Any] = {"label": label, "path": str(raw), "exists": p.exists()}
    if p.exists():
        key = "config.json" if kind == "base" else "adapter_config.json"
        info["valid"] = p.is_dir() and (p / key).exists()
        if not info["valid"]:
            info["note"] = f"directory lacks {key}"
    return info


def _cmd_doctor(args: argparse.Namespace) -> int:
    base = args.base_model or os.environ.get("SKILLLM_BASE_MODEL")
    adapters = _parse_adapters(args.adapters) or _env_adapters()
    if args.config and Path(args.config).exists():
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        base = base or cfg.get("base_model")
        adapters = adapters or cfg.get("adapters") or []
    adapter_items = list(adapters.items()) if isinstance(adapters, dict) else [(Path(a).name, a) for a in adapters]
    gpu = _gpu_report()
    report = {
        "skillllm": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {n: _dist_version(n) for n in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors", "gliner", "networkx")},
        "extras_installed": _lazy.extra_status(),
        "gpu": gpu,
        "paths": [_path_check("base_model", base, "base")] + [_path_check(f"adapter:{n}", a, "adapter") for n, a in adapter_items],
        "packaged_alias_table": str(Path(__file__).resolve().parent / "resources" / "offline_aliases.jsonl"),
        "ready": {},
    }
    free = sum(d["free_gib"] for d in gpu["devices"])
    report["packaged_alias_table_exists"] = Path(report["packaged_alias_table"]).exists()
    report["ready"]["offline"] = report["packaged_alias_table_exists"]
    model_deps = all(_lazy.is_available(m) for m in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"))
    paths_ok = bool(base) and all(p.get("valid") for p in report["paths"])
    report["ready"]["model_4bit"] = bool(model_deps and gpu["cuda_available"] and free >= 16 and paths_ok)
    problems = []
    if not model_deps:
        problems.append("model dependencies missing: pip install 'skillllm[model]'")
    if not gpu["cuda_available"]:
        problems.append("no CUDA GPU visible to torch")
    elif free < 16:
        problems.append(f"only {free:.1f} GiB GPU memory free; the 24B model at 4-bit needs about 16 GiB")
    if not paths_ok:
        problems.append("base model/adapter paths not configured or invalid (--base-model, --adapters, or SKILLLM_BASE_MODEL/SKILLLM_ADAPTERS)")
    report["problems_for_model_mode"] = problems
    print(_dump(report, args.pretty))
    return EXIT_OK


# --------------------------------------------------------------------------- version / main
def _cmd_version(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps({"skillllm": __version__, "python": platform.python_version()}))
    else:
        print(f"skillllm {__version__}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            return _cmd_extract(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        return _cmd_version(args)
    except ConfigError as exc:
        print(f"skillllm: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except SkillLLMError as exc:
        print(f"skillllm: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
