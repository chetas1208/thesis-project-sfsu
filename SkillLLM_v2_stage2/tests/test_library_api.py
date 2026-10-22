"""Public library API tests. No model weights, GPU or heavy packages are needed: a fake backend stands in for the model."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import pytest

import skillllm_v2
from skillllm_v2 import (
    ConfigError,
    DeviceError,
    ExtractionResult,
    FeatureUnavailableError,
    GPUMemoryError,
    MissingExtraError,
    MistralPeftBackend,
    ModelNotFoundError,
    SkillLLM,
    VoteCalibration,
)
from skillllm_v2 import _lazy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

TEXT = "We need Python, SQL and strong communication skills; Kubernetes is a plus."


def span(text: str, surface: str, label: str = "HARD_SKILL", conf: float = 0.86, nth: int = 0) -> dict[str, Any]:
    start = -1
    for _ in range(nth + 1):
        start = text.index(surface, start + 1)
    return {"surface_form": surface, "label": label, "start": start, "end": start + len(surface), "confidence": conf,
            "evidence": surface, "source": "mistral", "canonical_skill_id": None, "canonical_skill_name": None, "metadata": {}}


class FakeBackend:
    """Three 'adapters' that disagree in a controlled way; records call order."""

    mode = "model"

    def __init__(self, outputs: dict[str, list[str | tuple]] | None = None) -> None:
        self.system_names = ["anchor", "v1", "v2"]
        # Python: 3 votes; SQL: 2 votes; communication skills: 2 votes (SOFT); Kubernetes: 1 vote
        self.outputs = outputs or {
            "anchor": [("Python",), ("SQL",), ("communication skills", "SOFT_SKILL"), ("Kubernetes",)],
            "v1": [("Python",), ("SQL",)],
            "v2": [("Python",), ("communication skills", "SOFT_SKILL")],
        }
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    def predict_batch(self, texts: Sequence[str], system: str):
        self.calls.append((system, len(texts)))
        return [[span(t, *spec) for spec in self.outputs[system] if spec[0] in t] for t in texts]

    def close(self) -> None:
        self.closed = True


# ----------------------------------------------------------------------------- offline mode
def test_offline_extract_flags_and_schema():
    llm = SkillLLM.offline()
    res = llm.extract(TEXT)
    assert res.metadata.offline is True and res.metadata.mode == "offline"
    assert any("OFFLINE" in n for n in res.metadata.notes)
    names = {s.surface_form for s in res.skills}
    assert {"Python", "SQL", "communication skills", "Kubernetes"} <= names
    for s in res.skills:
        assert TEXT[s.start:s.end] == s.surface_form
        assert s.provenance.extractor == "dictionary" and s.provenance.systems == ["dictionary"]
        assert s.votes == 1 and s.n_systems == 1
        assert s.provenance.raw_confidence == s.confidence and s.provenance.confidence_source == "raw_prior"
    py = next(s for s in res.skills if s.surface_form == "Python")
    assert py.canonical_skill_id == "python" and py.link_method == "alias" and py.label == "HARD_SKILL"
    assert next(s for s in res.skills if s.surface_form == "communication skills").label == "SOFT_SKILL"
    assert [s.start for s in res.skills] == sorted(s.start for s in res.skills)


def test_offline_longest_match_and_empty_text():
    llm = SkillLLM.offline()
    res = llm.extract("Experience with machine learning and deep learning.")
    surfaces = [s.surface_form for s in res.skills]
    assert "machine learning" in surfaces and "deep learning" in surfaces
    assert llm.extract("").skills == []
    with pytest.raises(TypeError):
        llm.extract_batch([1])  # type: ignore[list-item]


def test_offline_no_link_no_verify():
    res = SkillLLM.offline(link=False, verify=False).extract(TEXT)
    assert all(s.canonical_skill_id is None for s in res.skills)
    assert res.metadata.linked is False and res.metadata.verified is False
    assert all(s.provenance.verified is None for s in res.skills)


def test_result_json_round_trip():
    res = SkillLLM.offline().extract(TEXT)
    again = ExtractionResult.from_json(res.to_json())
    assert again == res
    assert json.loads(res.to_json())["schema"] == "skillllm.extraction/1"
    assert again.names() == res.names()


def test_extract_batch_matches_single():
    llm = SkillLLM.offline()
    texts = [TEXT, "Docker and Python.", ""]
    batch = llm.extract_batch(texts)
    assert [r.text for r in batch] == texts
    for t, r in zip(texts, batch):
        single = llm.extract(t)
        assert [s.span for s in r.skills] == [s.span for s in single.skills]


# ----------------------------------------------------------------------------- voting with a fake backend
def test_majority_vote_provenance_and_calibrated_confidence():
    backend = FakeBackend()
    llm = SkillLLM.from_backend(backend, vote="majority")
    res = llm.extract(TEXT)
    by = {s.surface_form: s for s in res.skills}
    assert set(by) == {"Python", "SQL", "communication skills"}  # Kubernetes had 1 of 3 votes
    assert by["Python"].votes == 3 and by["Python"].n_systems == 3
    assert by["Python"].provenance.systems == ["anchor", "v1", "v2"]
    assert by["SQL"].votes == 2 and by["SQL"].provenance.systems == ["anchor", "v1"]
    assert by["communication skills"].provenance.systems == ["anchor", "v2"] and by["communication skills"].label == "SOFT_SKILL"
    # calibrated precision from the vote table; the extractor's constant prior is kept in provenance
    assert by["Python"].confidence == 0.8018 and by["SQL"].confidence == 0.5450
    assert all(s.provenance.raw_confidence == 0.86 for s in res.skills)
    assert all(s.provenance.confidence_source == "vote_calibration" for s in res.skills)
    assert res.metadata.calibration["note"].startswith("Estimated on the SkillSpan test split")
    assert res.metadata.vote == "majority" and res.metadata.min_votes == 2
    assert res.metadata.raw_span_counts == {"anchor": 4, "v1": 2, "v2": 2}
    assert res.metadata.offline is False
    assert [c[0] for c in backend.calls] == ["anchor", "v1", "v2"]  # sequential, one pass per system


def test_min_votes_dial():
    llm3 = SkillLLM.from_backend(FakeBackend(), min_votes=3)
    assert [s.surface_form for s in llm3.extract(TEXT).skills] == ["Python"]
    llm1 = SkillLLM.from_backend(FakeBackend(), min_votes=1)
    res = llm1.extract(TEXT)
    assert {s.surface_form: s.confidence for s in res.skills}["Kubernetes"] == 0.2901
    assert len(res.skills) == 4
    with pytest.raises(ValueError):
        SkillLLM.from_backend(FakeBackend(), min_votes=4)


def test_calibration_options():
    raw = SkillLLM.from_backend(FakeBackend(), vote="majority", calibration=None).extract(TEXT)
    assert all(s.confidence == 0.86 and s.provenance.confidence_source == "raw_prior" for s in raw.skills)
    assert raw.metadata.calibration is None
    custom = VoteCalibration(3, {1: 0.1, 2: 0.6, 3: 0.9}, source="mine")
    res = SkillLLM.from_backend(FakeBackend(), vote="majority", calibration=custom).extract(TEXT)
    assert {s.surface_form: s.confidence for s in res.skills}["Python"] == 0.9
    res2 = SkillLLM.from_backend(FakeBackend(), vote="majority", calibration=custom.to_dict()).extract(TEXT)
    assert res2.skills[0].confidence == res.skills[0].confidence
    with pytest.raises(ConfigError):
        SkillLLM.from_backend(FakeBackend(), vote="majority", calibration=VoteCalibration(2, {1: 0.3, 2: 0.7}))


def test_multiple_systems_require_vote_rule():
    with pytest.raises(ConfigError):
        SkillLLM.from_backend(FakeBackend(), vote=None)


def test_batch_runs_each_system_once_for_all_docs():
    backend = FakeBackend()
    SkillLLM.from_backend(backend, vote="majority").extract_batch([TEXT, TEXT, TEXT])
    assert backend.calls == [("anchor", 3), ("v1", 3), ("v2", 3)]  # system-major: no adapter thrash per document


def test_verifier_drops_unlocated_spans_and_counts_them():
    class OneSystem(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.system_names = ["only"]

        def predict_batch(self, texts, system):
            return [[span(t, "Python"), {"surface_form": "ghost skill", "label": "HARD_SKILL", "start": None, "end": None,
                                         "confidence": 0.86, "evidence": "ghost skill", "source": "mistral"}] for t in texts]

    res = SkillLLM.from_backend(OneSystem()).extract(TEXT)
    assert [s.surface_form for s in res.skills] == ["Python"]
    assert res.metadata.n_dropped_by_verifier == 1
    res2 = SkillLLM.from_backend(OneSystem(), verify=False).extract(TEXT)
    assert len(res2.skills) == 2


def test_link_methods_exclude_bm25_by_default():
    class Weird(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.system_names = ["only"]

        def predict_batch(self, texts, system):
            return [[span(t, "Python"), span(t, "communication")] for t in texts]

    res = SkillLLM.from_backend(Weird()).extract("Python communication zzzz")
    assert all(s.link_method in ("alias", "normalized", "fuzzy") for s in res.skills if s.canonical_skill_id)
    res2 = SkillLLM.from_backend(Weird(), link=False).extract("Python communication zzzz")
    assert all(s.canonical_skill_id is None for s in res2.skills)


def test_context_manager_closes_backend():
    backend = FakeBackend()
    with SkillLLM.from_backend(backend, vote="majority") as llm:
        llm.extract(TEXT)
    assert backend.closed


# ----------------------------------------------------------------------------- adapter switching on one loaded base
def _make_dirs(tmp_path: Path, names=("a", "b", "c")) -> tuple[Path, dict[str, str]]:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    adapters = {}
    for n in names:
        d = tmp_path / n
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"")
        adapters[n] = str(d)
    return base, adapters


class FakePeft:
    def __init__(self, first: str) -> None:
        self.active = first
        self.switches: list[str] = []

    def set_adapter(self, name: str) -> None:
        self.active = name
        self.switches.append(name)


class FakeExtractor:
    def __init__(self, model: FakePeft) -> None:
        self.model = model
        self.loads = 0

    def extract(self, text: str):
        return [span(text, "Python")] + ([span(text, "SQL")] if self.model.active != "c" else [])


def test_peft_backend_switches_adapters_on_one_loaded_base(tmp_path):
    base, adapters = _make_dirs(tmp_path)
    backend = MistralPeftBackend(str(base), adapters)
    peft_model = FakePeft("a")
    backend._model, backend._tokenizer, backend._extractor, backend._active = peft_model, object(), FakeExtractor(peft_model), "a"
    llm = SkillLLM.from_backend(backend, vote="majority", min_votes=2)
    res = llm.extract_batch([TEXT, TEXT])
    assert peft_model.switches == ["b", "c"]  # a was already active; one switch per system, not per doc
    assert {s.surface_form for s in res[0].skills} == {"Python", "SQL"}
    assert res[0].metadata.adapters == adapters and res[0].metadata.base_model == str(base)
    assert llm.save_config(tmp_path / "cfg.json").exists()


def test_from_pretrained_names_adapters_and_validates_paths(tmp_path):
    base, adapters = _make_dirs(tmp_path, ("mistral_lora", "qlora_loop", "qlora_loop_v2"))
    # conventional layout: .../<run>/model and .../<run>/best -> named after the run directory
    for run in ("anchor_run", "loop_run"):
        (tmp_path / run).mkdir()
    sub = []
    for run, leaf in (("anchor_run", "model"), ("loop_run", "best")):
        d = tmp_path / run / leaf
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"")
        sub.append(str(d))
    llm = SkillLLM.from_pretrained(str(base), adapters=sub)
    assert llm.systems == ["anchor_run", "loop_run"] and llm.min_votes == 2
    single = SkillLLM.from_pretrained(str(base), adapter=sub[0])
    assert single.vote is None and single.min_votes is None
    with pytest.raises(ConfigError):
        SkillLLM.from_pretrained(str(base), adapter=sub[0], adapters=sub)


# ----------------------------------------------------------------------------- config round trip
def test_offline_config_round_trip(tmp_path):
    llm = SkillLLM.offline(link=False)
    path = llm.save_config(tmp_path / "off.json")
    again = SkillLLM.from_config(path)
    assert again.is_offline and again.link is False
    assert [s.span for s in again.extract(TEXT).skills] == [s.span for s in llm.extract(TEXT).skills]


def test_model_config_round_trip_does_not_load_weights(tmp_path):
    base, adapters = _make_dirs(tmp_path)
    llm = SkillLLM.from_pretrained(str(base), adapters=adapters, min_votes=3, quantization="4bit")
    cfg_path = llm.save_config(tmp_path / "sub" / "cfg.json")
    cfg = json.loads(cfg_path.read_text())
    assert cfg["schema"] == "skillllm.config/1" and cfg["mode"] == "model" and cfg["min_votes"] == 3
    again = SkillLLM.from_config(cfg_path)
    assert again.config() == llm.config()
    assert not again.backend.is_loaded()  # type: ignore[attr-defined]
    with pytest.raises(ConfigError):
        SkillLLM.from_config({"schema": "nope"})
    with pytest.raises(ModelNotFoundError):
        SkillLLM.from_config(tmp_path / "missing.json")
    # custom calibration is stored and restored
    cal = VoteCalibration(3, {1: 0.1, 2: 0.5, 3: 0.9}, source="mine")
    llm2 = SkillLLM.from_pretrained(str(base), adapters=adapters, calibration=cal)
    p2 = llm2.save_config(tmp_path / "c2.json")
    assert SkillLLM.from_config(p2).calibration == cal
    with pytest.raises(FeatureUnavailableError):
        SkillLLM.from_backend(FakeBackend(), vote="majority").save_config(tmp_path / "x.json")


# ----------------------------------------------------------------------------- graceful errors
def test_missing_weights_errors(tmp_path):
    with pytest.raises(ModelNotFoundError, match="not bundled"):
        SkillLLM.from_pretrained(str(tmp_path / "nope"))
    base, adapters = _make_dirs(tmp_path)
    with pytest.raises(ModelNotFoundError, match="Adapter 'x'"):
        SkillLLM.from_pretrained(str(base), adapters={"x": str(tmp_path / "gone")})
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(ModelNotFoundError, match="adapter_config.json"):
        SkillLLM.from_pretrained(str(base), adapter=str(bad))
    notmodel = tmp_path / "empty"
    notmodel.mkdir()
    with pytest.raises(ModelNotFoundError, match="config.json"):
        SkillLLM.from_pretrained(str(notmodel))
    assert issubclass(ModelNotFoundError, FileNotFoundError)


def test_missing_extra_message_names_the_extra(tmp_path, monkeypatch):
    base, adapters = _make_dirs(tmp_path)
    for mod in ("torch", "transformers", "accelerate", "peft", "bitsandbytes"):
        monkeypatch.setitem(sys.modules, mod, None)  # makes `import mod` raise ImportError
    llm = SkillLLM.from_pretrained(str(base), adapters=adapters)
    with pytest.raises(MissingExtraError, match=r"skillllm\[model\]") as info:
        llm.extract(TEXT)
    assert isinstance(info.value, ImportError) and info.value.extra == "model"
    monkeypatch.setitem(sys.modules, "gliner", None)
    with pytest.raises(MissingExtraError, match=r"skillllm\[gliner\]"):
        _lazy.require("gliner")


def _fake_modules(monkeypatch, cuda: bool):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 1 if cuda else 0,
                                       mem_get_info=lambda i: (2 * 2**30, 24 * 2**30), OutOfMemoryError=type("OOM", (RuntimeError,), {}))
    torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    for mod in ("transformers", "accelerate", "peft", "bitsandbytes"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    return torch


def test_no_cuda_gives_actionable_device_error(tmp_path, monkeypatch):
    base, adapters = _make_dirs(tmp_path)
    _fake_modules(monkeypatch, cuda=False)
    llm = SkillLLM.from_pretrained(str(base), adapters=adapters)
    with pytest.raises(DeviceError, match="No CUDA GPU") as info:
        llm.extract(TEXT)
    assert "offline" in str(info.value)
    cpu = SkillLLM.from_pretrained(str(base), adapters=adapters, device="cpu", quantization="4bit")
    with pytest.raises(DeviceError, match="quantization"):
        cpu.extract(TEXT)


def test_low_free_gpu_memory_warns_or_errors(tmp_path, monkeypatch):
    base, adapters = _make_dirs(tmp_path)
    torch = _fake_modules(monkeypatch, cuda=True)
    backend = MistralPeftBackend(str(base), adapters, check_memory="error")
    with pytest.raises(GPUMemoryError, match="free"):
        backend._check_memory(torch)
    warn_backend = MistralPeftBackend(str(base), adapters, check_memory="warn")
    with pytest.warns(ResourceWarning):
        warn_backend._check_memory(torch)
    MistralPeftBackend(str(base), adapters, check_memory="off")._check_memory(torch)


def test_oom_is_translated_with_guidance(tmp_path):
    base, adapters = _make_dirs(tmp_path)
    backend = MistralPeftBackend(str(base), adapters)
    err = backend._translate(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), None, "generating with")
    assert isinstance(err, GPUMemoryError) and "4bit" in str(err) and "offline" in str(err)
    other = RuntimeError("boom")
    assert backend._translate(other, None, "loading") is other


def test_bad_quantization_and_vote_values(tmp_path):
    base, adapters = _make_dirs(tmp_path)
    with pytest.raises(ConfigError):
        SkillLLM.from_pretrained(str(base), adapters=adapters, quantization="3bit")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SkillLLM.from_pretrained(str(base), adapters=adapters, vote="plurality")


# ----------------------------------------------------------------------------- analyze / recommend
def test_analyze_gap_output():
    llm = SkillLLM.offline()
    res = llm.analyze("I know Python and SQL and leadership.", "Required: Python, Kubernetes, SQL.")
    assert res.method in {"set_overlap", "inference_pipeline"}
    assert {m.skill for m in res.matched} == {"Python", "SQL"}
    assert [s.canonical_skill_name for s in res.missing] == ["Kubernetes"]
    assert [s.canonical_skill_name for s in res.extra] == ["leadership"]
    assert res.coverage == pytest.approx(2 / 3)
    assert json.loads(res.to_json())["schema"] == "skillllm.analysis/1"
    if res.method == "set_overlap":
        assert any("not available" in n for n in res.notes)


def test_recommend_without_catalog_gives_clear_message(tmp_path):
    with pytest.raises(FeatureUnavailableError) as info:
        SkillLLM.offline().recommend("Python", "Kubernetes", catalog_path=tmp_path / "missing_catalog.jsonl")
    assert issubclass(FeatureUnavailableError, NotImplementedError)
    assert "catalog" in str(info.value).lower() or "course" in str(info.value).lower()


# ----------------------------------------------------------------------------- import hygiene and CLI
def _env() -> dict[str, str]:
    env = dict(os.environ)
    if SRC.exists():
        env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_import_is_lazy_and_does_not_import_torch():
    code = ("import sys, skillllm_v2; from skillllm_v2 import SkillLLM; SkillLLM.offline().extract('Python');"
            "bad=[m for m in ('torch','transformers','peft','bitsandbytes','numpy','yaml','sklearn','scipy','networkx','gliner') if m in sys.modules];"
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_env(), cwd=ROOT.parent)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_package_exports_and_version():
    assert skillllm_v2.__version__ == "0.1.0"
    for name in skillllm_v2.__all__:
        assert hasattr(skillllm_v2, name), name
    assert (SRC / "skillllm_v2" / "py.typed").exists()


def run_cli(*args: str, input: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "skillllm_v2", *args], capture_output=True, text=True, env=_env(), cwd=ROOT.parent, input=input)


def test_cli_version():
    proc = run_cli("version")
    assert proc.returncode == 0 and proc.stdout.strip() == "skillllm 0.1.0"
    assert json.loads(run_cli("version", "--json").stdout)["skillllm"] == "0.1.0"


def test_cli_extract_offline_json_and_file_and_stdin(tmp_path):
    proc = run_cli("extract", "--offline", "--text", TEXT)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["metadata"]["offline"] is True and {s["surface_form"] for s in payload["skills"]} >= {"Python", "SQL"}
    f = tmp_path / "t.txt"
    f.write_text("Docker and Python", encoding="utf-8")
    out = tmp_path / "o.json"
    proc = run_cli("extract", "--offline", "--file", str(f), "--out", str(out), "--no-link")
    assert proc.returncode == 0 and ExtractionResult.from_json(out.read_text()).skills[0].canonical_skill_id is None
    two = json.loads(run_cli("extract", "--offline", "--text", "Python", "--text", "SQL").stdout)
    assert isinstance(two, list) and len(two) == 2
    assert json.loads(run_cli("extract", "--offline", "--file", "-", input="Kubernetes").stdout)["skills"][0]["surface_form"] == "Kubernetes"


def test_cli_errors_are_clean(tmp_path):
    proc = run_cli("extract", "--text", "x")
    assert proc.returncode == 2 and "--offline" in proc.stderr and "Traceback" not in proc.stderr
    proc = run_cli("extract", "--base-model", str(tmp_path / "nope"), "--text", "x")
    assert proc.returncode == 1 and "not bundled" in proc.stderr and "Traceback" not in proc.stderr
    proc = run_cli("extract", "--offline", "--file", str(tmp_path / "missing.txt"))
    assert proc.returncode == 2


def test_cli_doctor_reports_environment(tmp_path):
    base, adapters = _make_dirs(tmp_path)
    proc = run_cli("doctor", "--base-model", str(base), "--adapters", *adapters.values())
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(proc.stdout)
    assert {"skillllm", "python", "packages", "extras_installed", "gpu", "paths", "ready", "problems_for_model_mode"} <= set(rep)
    assert set(rep["extras_installed"]) == {"model", "gliner", "eval", "test"}
    assert all(p["exists"] for p in rep["paths"]) and rep["ready"]["offline"] is True
    missing = json.loads(run_cli("doctor", "--base-model", str(tmp_path / "nope")).stdout)
    assert missing["paths"][0]["exists"] is False


def test_cli_min_votes_flag_parses():
    proc = run_cli("extract", "--help")
    assert "--min-votes" in proc.stdout and "--no-calibration" in proc.stdout


def test_analyze_uses_inference_pipeline_when_importable_with_fake_model():
    llm = SkillLLM.from_backend(FakeBackend(), vote="majority")
    res = llm.analyze("I know Python and SQL.", "We need Python, SQL and Kubernetes.")
    assert {m.skill for m in res.matched} >= {"Python", "SQL"}
    if res.method == "inference_pipeline":
        assert res.match_score is not None and 0.0 <= res.match_score <= 1.0
        assert res.details["trace"]["anchor_mode"] == "model"  # the library's own extraction is the anchor; no second model run
        assert llm.backend.calls == [("anchor", 2), ("v1", 2), ("v2", 2)]  # one batch pass per system; the pipeline did not re-run the model
    else:
        assert res.match_score is None


def test_recommend_requires_catalog_path():
    with pytest.raises(FeatureUnavailableError, match="catalog_path"):
        SkillLLM.offline().recommend("Python", "Kubernetes")
