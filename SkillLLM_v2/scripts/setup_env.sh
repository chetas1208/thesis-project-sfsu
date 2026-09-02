#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
SPACY_MODEL_URL="${SPACY_MODEL_URL:-https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl}"

choose_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return
  fi

  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
    return
  fi

  command -v python3
}

create_venv() {
  local selected_python="$1"

  rm -rf "$VENV_DIR"
  if "$selected_python" -m venv "$VENV_DIR" >/dev/null 2>&1; then
    return
  fi

  "$BOOTSTRAP_PYTHON" -m pip install --user virtualenv
  "$BOOTSTRAP_PYTHON" -m virtualenv -p "$selected_python" "$VENV_DIR"
}

PYTHON_BIN="$(choose_python)"

create_venv "$PYTHON_BIN"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"

EXTRA_PACKAGES=(
  "git+https://github.com/huggingface/transformers.git"
  "tokenizers==0.22.2"
  "accelerate==1.9.0"
  "bitsandbytes==0.49.2"
  "peft==0.11.1"
  "trl==0.8.6"
  "datasets==2.19.2"
  evaluate
  scikit-learn
  seqeval
  rapidfuzz
  sentencepiece
  "huggingface_hub>=1.5.0,<2.0"
  "wandb==0.16.2"
  orjson
  "pandas==2.2.1"
  "numpy==1.26.3"
  jinja2
  pyyaml
)

python -m pip install "${EXTRA_PACKAGES[@]}"

if [[ "$INSTALL_VLLM" == "1" ]]; then
  python -m pip install vllm
fi

python -m pip install "$SPACY_MODEL_URL"

printf 'Environment ready at %s\n' "$VENV_DIR"
printf 'Activate it with: source %s/bin/activate\n' "$VENV_DIR"
