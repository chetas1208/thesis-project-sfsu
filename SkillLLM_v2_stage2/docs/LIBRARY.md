# SkillLLM as a Python library

`skillllm` (import name `skillllm_v2`) is a typed Python package around the thesis system: skill-span extraction from resumes and job postings with Mistral-Small-3.1-24B plus LoRA adapters, an optional span-voting ensemble of three adapters, rule-based verification, canonical-skill linking, and profile-versus-job gap analysis. It also ships a model-free **offline** mode.

Nothing in this document has been published. Model weights (base or adapters) are never bundled in the wheel; you point the library at local directories.

## 1. Install

```bash
pip install skillllm                # core: result types, ensemble layer, offline extractor, CLI. Standard library only.
pip install "skillllm[model]"       # torch, transformers, peft, bitsandbytes, accelerate, safetensors (needs a CUDA GPU for 4-bit)
pip install "skillllm[gliner]"      # optional GLiNER candidate source used by the research pipeline
pip install "skillllm[eval]"        # numpy/scipy/scikit-learn for skillllm_v2.eval (significance tests, scripts/ensemble_eval.py)
pip install "skillllm[recommend]"   # networkx, for the existing course recommender behind SkillLLM.recommend()
pip install "skillllm[test]"        # pytest
```

From a checkout: `pip install -e ".[model,test]"`. Requires Python >= 3.10. A missing extra always produces an error that names the extra to install (`MissingExtraError`).

`import skillllm_v2` is fast and never imports torch, transformers, peft or numpy; heavy packages load only when a model is actually used.

Run `skillllm doctor` to see Python/torch/CUDA/bitsandbytes versions, free GPU memory, which extras are installed, and whether your model paths exist.

## 2. Quick start

```python
from skillllm_v2 import SkillLLM

llm = SkillLLM.offline()                       # no model, no GPU; see section 4
result = llm.extract("Built ETL pipelines in Python and SQL on AWS; strong communication skills.")
for s in result.skills:
    print(s.surface_form, s.label, (s.start, s.end), s.canonical_skill_name, s.confidence)
print(result.to_json(indent=2))                # JSON-serialisable; ExtractionResult.from_json() round-trips
```

Trained model with the three-adapter ensemble (see section 5 for what that costs):

```python
llm = SkillLLM.from_pretrained(
    "models/mistral-small-3.1-24b-instruct-2503",            # local base model directory (not bundled)
    adapters={
        "anchor": "artifacts/skillllm_v2/mistral_lora/model",   # 80-epoch r=16
        "v1":     "artifacts/skillllm_v2/qlora_loop/best",      # QLoRA loop, r=8
        "v2":     "artifacts/skillllm_v2/qlora_loop_v2/best",   # QLoRA loop v2, r=16
    },
    min_votes=2,                 # balanced default; 3 = high precision; 1 = union (section 6)
    quantization="4bit",         # the evaluated setting (~16 GB GPU memory)
)
with llm:
    result = llm.extract(text)
    gap = llm.analyze(resume_text, job_text)
```

A single adapter: `SkillLLM.from_pretrained(base, adapter="artifacts/skillllm_v2/mistral_lora/model")` (no voting, 1x generation cost).

`skillllm_v2.SkillLLM` methods: `from_pretrained`, `offline`, `from_backend`, `extract`, `extract_batch`, `analyze`, `recommend`, `save_config`, `from_config`, `close` (also a context manager). A runnable script is in `examples/quickstart.py`.

### Loading adapters from a local path

An adapter path is a directory containing `adapter_config.json` and `adapter_model.safetensors` (a training output such as `.../model` or `.../best`). Pass `adapter=` (one) or `adapters=` (a list, or `{name: path}` to choose the names that appear in provenance). With a list, names come from the directory (`.../mistral_lora/model` becomes `mistral_lora`). All adapters attach to one loaded base model and are switched with PEFT `set_adapter`, system by system, so memory stays at one base plus small adapters. Paths and options are checked immediately in `from_pretrained`; weights load lazily on the first `extract`.

### Saving and restoring a configuration

```python
llm.save_config("skillllm.json")               # paths and options only, never weights
llm = SkillLLM.from_config("skillllm.json")    # relative paths that do not exist are resolved against the config file's folder
```

## 3. Result schema (`skillllm.extraction/1`)

`ExtractionResult(text, skills: list[Skill], metadata: ExtractionMetadata)`; `.to_dict()` / `.to_json()` / `.from_dict()` / `.from_json()`.

`Skill` fields:

| Field | Meaning |
|---|---|
| `surface_form`, `evidence` | The exact text of the mention (`text[start:end]` when offsets exist). |
| `label` | `HARD_SKILL` or `SOFT_SKILL`. |
| `start`, `end` | Character offsets, end exclusive. `None` only if span verification is off and the model's string could not be located in the text. |
| `confidence` | With a 3-adapter vote and the default calibration: the estimated precision for this span's vote count (section 6). Otherwise the extractor's constant prior (0.86 for the Mistral adapters, 0.82 for the dictionary). **Not a probability in the second case.** |
| `votes`, `n_systems` | How many of the loaded adapters predicted this exact (start, end, label). `1/1` for single-adapter and offline results. |
| `canonical_skill_id`, `canonical_skill_name`, `taxonomy`, `link_confidence`, `link_method` | Attach-only linking against the alias table. Default accepted methods: `alias`, `normalized`, `fuzzy`. The BM25 fallback is off by default because it returns a link for nearly any string (`link_methods=` can opt in). |
| `provenance` | `extractor` (`mistral` or `dictionary`), `method` (`single`, `vote:...`, `dictionary`), `systems` (the adapters that voted for the span), `n_votes`, `n_systems`, `verified`, `verification_reason`, `raw_confidence` (the extractor's own value before calibration), `confidence_source` (`vote_calibration` or `raw_prior`). |

`ExtractionMetadata` records `mode` (`model` | `offline` | `custom`), `offline` (True means no trained model was involved), `systems`, `vote`, `min_votes`, `base_model`, `adapters`, `quantization`, `linked`, `verified`, `raw_span_counts` per system, `n_dropped_by_verifier`, `calibration` (the vote table used, or None), `latency_seconds`, `library_version`, `notes`, `warnings`. Unparsable model output is reported in `warnings` and counts as no skills for that document and system.

`analyze(profile_text, job_text)` returns `AnalysisResult` (`skillllm.analysis/1`): both extractions, `matched`, `missing` (job skills absent from the profile), `extra`, `coverage`, `jaccard`, `gaps`, `match_score`, and `method`. With `skillllm_v2.pipeline.inference` importable (it is in this repository) the pipeline's JobMatcher and GapDetector run on the library's own extraction, without a second model run (`method="inference_pipeline"`, raw output in `details`). Otherwise it falls back to a plain skill-set overlap (`method="set_overlap"`) and says so in `notes`. The match score is a rule-weighted heuristic with no validation study behind it.

`recommend(profile_text, job_text, catalog_path=..., top_k=5)` goes only through the existing course pipeline (`pipeline/course_port.py` adapter). It needs the `recommend` extra and a course catalog JSONL; otherwise it raises `FeatureUnavailableError` (a `NotImplementedError`) explaining what is missing. The recommender's own benchmark is on a synthetic catalogue; see `docs/INFERENCE_AND_EVALS.md`.

## 4. Offline mode

`SkillLLM.offline()` runs a dictionary/alias matcher over a packaged list of about 290 hand-curated aliases (162 skills). It needs no model, GPU or third-party package, is deterministic, and returns the same result types. Every result is flagged `metadata.offline = True` and carries a note saying so.

It is a gazetteer baseline, not the trained system. No evaluation numbers in this document apply to it (it was not evaluated on SkillSpan), it only finds skills in its list, longest match wins on overlap, and fuzzy matching is off by default (`fuzzy=True` to enable). Use `alias_paths=[...]` to supply your own table (JSONL rows with `alias`, `label`, `canonical_skill_id`, `canonical_skill_name`). The packaged list deliberately contains no SkillSpan-derived or model-prediction-derived rows; regenerate it with `scripts/build_offline_aliases.py`.

## 5. Ensemble layer and its cost

`skillllm_v2.ensemble` is pure Python. `vote_spans(per_system_predictions, min_votes)` keeps a span `(start, end, label)` if at least `min_votes` distinct systems predict it exactly (one vote per system per span); the kept span is the first voter's, with the mean of the voters' confidences and `source="ensemble:<k>of<n>"`. `vote_documents` does this over a batch and is output-identical (including order) to `vote()` in `scripts/ensemble_eval.py`, the reference used for the held-out evaluation; `tests/test_ensemble.py` proves it on the saved predictions. Differences, opt-in or in the API layer: provenance annotation, position ordering, and the API drops spans that have no offsets before voting (the reference collapses them per label; they can never match gold exactly).

**Cost.** The ensemble runs every adapter on every document: **3 generation passes instead of 1**. With 3 adapters on one 4-bit base, memory is about one adapter's worth (peak 16.3 GB allocated in the v2 evaluation), but wall time is about 3x. Measured generation was about 4.4 s per document per adapter on a shared RTX 3090 (v2 evaluation: 4,303 s for 974 documents), so expect roughly 13 s per document for the 3-adapter ensemble. Documents are processed one at a time (greedy decoding, no padding), matching how the reported numbers were produced; `extract_batch` runs system by system to avoid adapter thrashing but is not a batched-generation speed-up.

## 6. Precision/recall dial and vote-calibrated confidence

`min_votes` (library and `skillllm extract --min-votes N`) selects how many of the three adapters must agree. Held-out SkillSpan test, 974 documents, exact-span match (`reports/skillllm_v2/ensemble_eval.md`, `ensemble_vote_confidence.md`):

| `min_votes` | Precision | Recall | Exact-span F1 | Use |
|---|---|---|---|---|
| 3 (unanimous) | 0.802 | 0.506 | 0.620 | high precision, drops about a third of true skills |
| **2 (majority, default)** | **0.722** | **0.661** | **0.690** | balanced; single anchor is P 0.674, R 0.667, F1 0.670 |
| 1 (union) | 0.587 | 0.781 | 0.670 | high recall |

Default `confidence` for a 3-adapter vote is the precision of spans with that many votes: 1 of 3 -> 0.2901 (273/941), 2 of 3 -> 0.5450 (351/644), 3 of 3 -> 0.8018 (1,145/1,428). The table is **estimated on the SkillSpan test split itself** (the only split with predictions from all three adapters), so it shows the signal rather than being a fitted calibrator; it is labelled this way in the result metadata. Re-fit it on your own held-out data:

```python
from skillllm_v2 import fit_vote_calibration, SkillLLM
cal = fit_vote_calibration(gold_spans_per_doc, [anchor_preds, v1_preds, v2_preds], smoothing=1.0)  # [system][doc][span]
llm = SkillLLM.from_pretrained(base, adapters=..., calibration=cal)     # or calibration=None to keep the raw 0.86 prior
```

`fit_vote_calibration` returns a `VoteCalibration` with the per-vote precision (Laplace smoothed), the counts it came from, and the smoothing used; `.to_dict()` saves it and `calibration=` accepts it or its dict form (CLI: `--calibration FILE`, `--no-calibration`). The original extractor confidence stays in `provenance.raw_confidence`. The shipped table only exists for 3 adapters; with another number of adapters confidences stay raw unless you fit a table.

## 7. Performance (held-out SkillSpan test, 974 documents with at least one gold span)

| System | Exact P | Exact R | Exact-span F1 | Token F1 | Partial F1 | HARD F1 | SOFT F1 |
|---|---|---|---|---|---|---|---|
| anchor (80 ep, r=16) | 0.6735 | 0.6667 | 0.6701 | 0.7879 | 0.8138 | 0.7265 | 0.6040 |
| QLoRA loop v1 (r=8) | 0.6864 | 0.6291 | 0.6565 | 0.7846 | 0.8040 | 0.7046 | 0.5996 |
| QLoRA loop v2 (r=16) | 0.6720 | 0.6512 | 0.6614 | 0.7957 | 0.8170 | 0.7159 | 0.5989 |
| **majority ensemble (>=2 of 3)** | 0.7220 | 0.6605 | **0.6899** | 0.7962 | 0.8213 | 0.7473 | 0.6231 |

Uncertainty, from the paired document bootstrap (10,000 resamples):

- Ensemble minus anchor, exact-span F1: +0.0198, 95% CI [+0.0041, +0.0356]; raw p 0.016 (bootstrap) / 0.015 (permutation); **Holm-adjusted over the 5 metrics: p 0.081 / 0.073, not significant at 0.05**. The other four metrics have CIs that include zero (token F1 +0.0083 [-0.0036, +0.0203]). Document level: ensemble better on 148 documents, worse on 91, tied on 735 (sign test p = 0.0003).
- v1 minus anchor, exact-span F1: -0.0135, CI [-0.0342, +0.0071]; v2 minus anchor: -0.0086, CI [-0.0286, +0.0114]. The single adapters are statistically indistinguishable from the anchor.

Caveats, all of which apply:

- **One training seed per adapter and one test split.** The test set was also used to estimate the vote-precision table and to compare variants; the ensemble rule itself was fixed before looking at test results.
- The adapters share a base model and training data, so their errors are correlated and voting gains are bounded.
- The default precision table and the operating points above come from the test split, so they are somewhat optimistic for the table's own calibration; re-fit on your data.
- Scoring is pooled exact span match against SkillSpan annotations (a particular annotation convention); partial and token F1 are much higher than exact F1, meaning many errors are boundary disagreements.
- Wrapper stages (GLiNER, merger, boundary refiner) add at most +0.0024 exact-F1 and are not part of `SkillLLM.extract`.
- The offline mode, the taxonomy linking, `analyze` and `recommend` have no accuracy evaluation. "Linking accuracy" in the thesis tables is link coverage, not correctness.

## 8. Compute

| | Value | Source |
|---|---|---|
| 80-epoch anchor training | 90.06 h wall on 2 GPUs = 180.1 GPU-h | `training_summary.json` |
| QLoRA loop v1 / v2 training | 1.59 / 3.17 GPU-h (0.9% / 1.8% of the anchor) | `reports/skillllm_v2/qlora_loop_summary.md`, `qlora_loop_v2_summary.md` |
| Inference memory | about 16 GB GPU, 4-bit NF4 base + adapters; 8-bit about 28 GB; bf16 about 50 GB (not measured here) | v2 eval peak 16.3 GB; others estimated from model size |
| Inference speed | about 4.4 s per document per adapter (RTX 3090, shared) | v2 evaluation |
| Ensemble | 3x generation per document | section 5 |

Hardware: RTX 3090 24 GB GPUs. The offline mode runs on any CPU in milliseconds.

## 9. Errors you may meet

| Error | Cause and fix |
|---|---|
| `MissingExtraError` | A dependency is missing or fails to import; the message says `pip install 'skillllm[model]'` (or another extra). |
| `ModelNotFoundError` | Base or adapter path missing or not a model/adapter directory. Weights are not bundled. `allow_hub=True` lets a non-existent path be treated as a Hugging Face Hub id. |
| `DeviceError` | No CUDA GPU, or 4-bit/8-bit requested on CPU. Use a GPU, or `SkillLLM.offline()`. |
| `GPUMemoryError` | Out of memory while loading or generating (or `check_memory="error"` with too little free memory). Free the GPU (`skillllm doctor`), keep `quantization="4bit"`, limit `max_new_tokens`, chunk long documents. By default a free-memory check only warns. |
| `FeatureUnavailableError` | `recommend`/`save_config` for something not configured; the message says what. |
| `ConfigError` | Invalid option, for example several adapters with `vote=None`. |

All derive from `SkillLLMError`.

## 10. CLI

```bash
skillllm version [--json]
skillllm doctor [--base-model PATH] [--adapters PATH ...] [--pretty]
skillllm extract --offline --text "Python and SQL"                      # JSON on stdout
skillllm extract --file resume.txt --out result.json --pretty --offline
skillllm extract --base-model models/mistral-small-3.1-24b-instruct-2503 \
    --adapters anchor=.../mistral_lora/model v1=.../qlora_loop/best v2=.../qlora_loop_v2/best \
    --min-votes 2 --file resume.txt
```

`--text`/`--file` can be repeated (a list is printed); `--file -` reads stdin. Defaults come from `SKILLLM_BASE_MODEL` and `SKILLLM_ADAPTERS` (paths separated by `os.pathsep`). Other flags: `--vote`, `--min-votes`, `--quantization 4bit|8bit|none`, `--device`, `--max-new-tokens`, `--no-link`, `--no-verify`, `--alias-table`, `--calibration`, `--no-calibration`, `--config`. Exit codes: 0 ok, 1 runtime error, 2 usage error.

## 11. Publishing adapters to the Hugging Face Hub later (description only; nothing has been uploaded)

1. Check the licences first (section "Licences" in `MODEL_CARD.md`): the base model's licence, SkillSpan's CC BY 4.0 attribution, and whether you want to release adapters trained on that data.
2. Each adapter directory is already a PEFT adapter. Upload one repository per adapter containing `adapter_config.json`, `adapter_model.safetensors`, and a README from `MODEL_CARD.md`. Do not upload base weights or `checkpoint-*` folders and `training_args.bin`. Set `base_model` in the card metadata to the public base-model repo id (the local `adapter_config.json` records a local path that will not resolve for others, so edit `base_model_name_or_path`).
3. With `huggingface_hub` installed and `huggingface-cli login` done by the owner: `huggingface-cli upload <user>/<name> <adapter_dir> .`
4. Consumers then use `SkillLLM.from_pretrained("<local or Hub base>", adapters={"anchor": "<user>/skillllm-anchor", ...}, allow_hub=True)`; PEFT downloads the adapters. The base model must still be obtained by the user under its own licence.
5. Version the vote-calibration table together with the adapters (`VoteCalibration.to_dict()` as `vote_calibration.json`) and pass it via `calibration=`.

## 12. Testing

```bash
pip install -e ".[test]" && pytest tests/test_library_api.py tests/test_ensemble.py
```

These tests need no model, GPU or network: they use a fake backend, the offline extractor, subprocess CLI calls, and (when present) the saved prediction files. Real 24B loading is not covered by CI; see `MODEL_CARD.md` for what has been verified.
