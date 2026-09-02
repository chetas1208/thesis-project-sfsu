"""GLiNER trainer: fine-tune a GLiNER model on skill extraction data."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


def generate_gliner_training_config(
    model_config: Dict[str, Any],
    train_data_path: Path,
    val_data_path: Path,
    output_dir: Path,
) -> str:
    """Generate a YAML config string for GLiNER's train.py."""
    base_model = model_config["base_model"]
    train_cfg = model_config["training"]
    model_cfg = model_config["model"]

    lines = [
        "model:",
        f"  model_name: {base_model}",
        '  name: "skill_gliner"',
        f"  max_width: {model_cfg['max_width']}",
        "  hidden_size: 768",
        "  dropout: 0.3",
        "  fine_tune: true",
        f"  subtoken_pooling: {model_cfg['subtoken_pooling']}",
        f"  span_mode: {model_cfg['span_mode']}",
        f"  max_types: {model_cfg['max_types']}",
        f"  max_len: {model_cfg['max_len']}",
        f"  max_neg_type_ratio: {model_cfg['max_neg_type_ratio']}",
        '  decoder_mode: "span"',
        "",
        "data:",
        f"  root_dir: {output_dir}",
        f"  train_data: '{train_data_path}'",
        f"  val_data_dir: '{val_data_path}'",
        "",
        "training:",
        f"  prev_path: {base_model}",
        f"  num_steps: {train_cfg['num_steps']}",
        f"  train_batch_size: {train_cfg['train_batch_size']}",
        f"  eval_every: {train_cfg['eval_every']}",
        f"  warmup_ratio: {train_cfg['warmup_ratio']}",
        f"  scheduler_type: \"{train_cfg['scheduler_type']}\"",
        "  loss_alpha: 0.75",
        "  loss_gamma: 0",
        "  loss_prob_margin: 0",
        '  loss_reduction: "sum"',
        "  negatives: 1.0",
        "  masking: none",
        f"  lr_encoder: {train_cfg['lr_encoder']}",
        f"  lr_others: {train_cfg['lr_others']}",
        f"  weight_decay_encoder: {train_cfg['weight_decay_encoder']}",
        f"  weight_decay_other: {train_cfg['weight_decay_other']}",
        f"  max_grad_norm: {train_cfg['max_grad_norm']}",
        f"  save_total_limit: {train_cfg['save_total_limit']}",
    ]
    return "\n".join(lines) + "\n"


def filter_positive_records(records: List[Dict]) -> List[Dict]:
    """GLiNER training requires positive-only examples (records with NER annotations)."""
    return [r for r in records if r.get("ner")]


def train_gliner(
    model_config: Dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    gliner_repo_dir: Path | None = None,
) -> Path:
    """Fine-tune a GLiNER model.

    If gliner_repo_dir is provided, uses the repo's train.py.
    Otherwise, uses the installed gliner package's training API.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter data
    train_data = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    val_data = json.loads((data_dir / "dev.json").read_text(encoding="utf-8"))

    train_positive = filter_positive_records(train_data)
    val_positive = filter_positive_records(val_data)

    if not train_positive:
        raise ValueError("No positive GLiNER training examples after filtering")

    # Write filtered data
    train_path = runtime_dir / "train_positive.json"
    val_path = runtime_dir / "dev_positive.json"
    train_path.write_text(json.dumps(train_positive, ensure_ascii=False), encoding="utf-8")
    val_path.write_text(json.dumps(val_positive, ensure_ascii=False), encoding="utf-8")

    logger.info("GLiNER training: %d positive train, %d positive val", len(train_positive), len(val_positive))

    # Generate config
    config_str = generate_gliner_training_config(model_config, train_path, val_path, output_dir)
    config_path = runtime_dir / "gliner_config.yaml"
    config_path.write_text(config_str, encoding="utf-8")

    # Run training
    if gliner_repo_dir and (gliner_repo_dir / "train.py").exists():
        train_script = str(gliner_repo_dir / "train.py")
    else:
        project_root = Path(__file__).resolve().parents[2]
        train_script = str(project_root / "scripts" / "gliner_train.py")

    cmd = [sys.executable, train_script, "--config", str(config_path)]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(output_dir))

    checkpoint_dir = output_dir / "models"
    logger.info("GLiNER model saved to %s", checkpoint_dir)
    return checkpoint_dir


def predict_gliner(
    model_path: str | Path,
    test_records: Sequence[Dict],
    threshold: float = 0.5,
    entity_types: Sequence[str] = ("skill",),
) -> List[Dict]:
    """Run GLiNER inference on canonical test records."""
    from gliner import GLiNER

    model = GLiNER.from_pretrained(str(model_path))
    
    import torch
    if torch.cuda.is_available():
        model = model.to("cuda")
    results: List[Dict] = []

    for rec in test_records:
        text = rec.get("text", " ".join(rec.get("tokens", [])))
        entities = model.predict_entities(text, list(entity_types), threshold=threshold)

        predicted_spans = []
        predicted_texts = []
        seen: set = set()

        for ent in entities:
            ent_text = ent["text"].strip()
            if not ent_text:
                continue
            key = ent_text.lower()
            if key in seen:
                continue
            seen.add(key)
            predicted_spans.append({
                "text": ent_text,
                "start_token": -1,  # GLiNER returns char spans, not token spans
                "end_token": -1,
                "score": round(ent.get("score", 0.0), 4),
            })
            predicted_texts.append(ent_text)

        results.append({
            "id": rec.get("sentence_id", rec.get("id", "")),
            "doc_id": rec.get("doc_id", ""),
            "tokens": rec.get("tokens", []),
            "predicted_spans": predicted_spans,
            "predicted_skill_texts": predicted_texts,
        })

    return results
