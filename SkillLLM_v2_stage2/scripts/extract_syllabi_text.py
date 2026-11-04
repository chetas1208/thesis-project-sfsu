"""Batch-extract plain text from every syllabus and write an inventory.

Usage:
    python scripts/extract_syllabi_text.py [--input inputs/syllabi] [--output inputs/syllabi_text]

Writes one .txt per syllabus plus inventory.csv summarizing each file, so
scanned or badly extracted documents can be spotted before labeling.
"""

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pdfplumber

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Characters per page below which a PDF is probably a scan with no text layer.
LOW_TEXT_PER_PAGE = 200

COURSE_CODE = re.compile(r"\bCSC[\s-]?(\d{3})(?:\s*/\s*(\d{3}))?\b")
FILENAME_CODE = re.compile(r"CSC(\d{3})(?:-(\d{3}))?", re.IGNORECASE)


def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages), len(pages)


def extract_docx(path):
    """Read paragraph and table text straight from document.xml (no python-docx needed)."""
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    lines = []
    for paragraph in root.iter(f"{W_NS}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{W_NS}t"))
        if text.strip():
            lines.append(text)
    return "\n".join(lines), None


def classify(text, page_count):
    chars = len(text.strip())
    if chars == 0:
        return "empty"
    if page_count and chars / page_count < LOW_TEXT_PER_PAGE:
        return "low_text"
    return "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="inputs/syllabi")
    parser.add_argument("--output", default="inputs/syllabi_text")
    args = parser.parse_args()
    # Windows consoles default to cp1252, which can't print names like "Dujmović".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".pdf", ".docx"})
    rows = []
    for path in files:
        row = {"file": path.name, "format": path.suffix.lower().lstrip(".")}
        name_match = FILENAME_CODE.search(path.name)
        row["filename_course"] = "/".join(g for g in name_match.groups() if g) if name_match else ""
        try:
            if row["format"] == "pdf":
                text, page_count = extract_pdf(path)
            else:
                text, page_count = extract_docx(path)
        except Exception as error:  # keep going; record the failure in the inventory
            row.update(pages="", chars=0, status=f"error: {error}", csc_codes_in_text="",
                       has_prereq_label=False, has_outcomes_label=False)
            rows.append(row)
            print(f"[error] {path.name}: {error}", file=sys.stderr)
            continue

        (output_dir / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        codes = sorted({"/".join(g for g in m.groups() if g) for m in COURSE_CODE.finditer(text)})
        row.update(
            pages=page_count if page_count is not None else "",
            chars=len(text),
            status=classify(text, page_count),
            csc_codes_in_text=" ".join(codes),
            has_prereq_label=bool(re.search(r"(?i)pre-?requisite", text)),
            has_outcomes_label=bool(re.search(r"(?i)(learning )?outcomes|objectives", text)),
        )
        rows.append(row)
        print(f"[{row['status']}] {path.name}")

    fields = ["file", "format", "pages", "chars", "status", "filename_course",
              "csc_codes_in_text", "has_prereq_label", "has_outcomes_label"]
    with open(output_dir / "inventory.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    statuses = {}
    for row in rows:
        key = row["status"].split(":")[0]
        statuses[key] = statuses.get(key, 0) + 1
    print(f"\n{len(rows)} files -> {output_dir}  " + ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())))


if __name__ == "__main__":
    main()
