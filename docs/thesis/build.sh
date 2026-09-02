#!/usr/bin/env bash
set -euo pipefail

THESIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THESIS_DIR"

ENTRYPOINT="SkillLLM_v2_Thesis.tex"
PDF="${ENTRYPOINT%.tex}.pdf"

if [[ -n "${TECTONIC_BIN:-}" ]]; then
  "$TECTONIC_BIN" --outdir . --keep-logs --keep-intermediates "$ENTRYPOINT"
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --outdir . --keep-logs --keep-intermediates "$ENTRYPOINT"
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$ENTRYPOINT"
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error "$ENTRYPOINT"
  bibtex "${ENTRYPOINT%.tex}"
  pdflatex -interaction=nonstopmode -halt-on-error "$ENTRYPOINT"
  pdflatex -interaction=nonstopmode -halt-on-error "$ENTRYPOINT"
else
  echo "A TeX engine is required. Install Tectonic, latexmk, or pdflatex plus bibtex." >&2
  exit 1
fi

echo "PDF: $THESIS_DIR/$PDF"
