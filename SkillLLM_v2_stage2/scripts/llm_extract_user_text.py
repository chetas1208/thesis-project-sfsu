"""Extract a user's skills and competence levels from their own free-text description, with an LLM.

The user-side twin of llm_extract_syllabi.py:

    syllabus text -> llm_extract_syllabi.py   -> inputs/syllabi_parsed/<name>.json -> course recommender
    user text     -> llm_extract_user_text.py -> inputs/users_parsed/<name>.json   -> course recommender

    python scripts/llm_extract_user_text.py --text "I have 3 years of Java and know some Python." --name alex
    python scripts/llm_extract_user_text.py                     # every .txt in inputs/users_text/

Each saved file keeps the original text, the skills (with level, confidence and the quote they came
from), the skills the user said they do NOT have, and any skill dropped because its quote was not in
the text. to_profile() turns a saved file into the profile dict the course recommender accepts.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

STAGE2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2_DIR / "src"))

from skillllm_v2.extract.text_profile_extractor import DEFAULT_MODEL, TextProfileExtractor  # noqa: E402

TEXT_DIR = STAGE2_DIR / "inputs" / "users_text"
OUT_DIR = STAGE2_DIR / "inputs" / "users_parsed"


def read_inputs(text, name, folder):
    """(name, text) pairs: the --text given on the command line, or every .txt file in folder."""
    if text is not None:
        return [(name or "user", text)]
    return [(f.stem, f.read_text(encoding="utf-8")) for f in sorted(Path(folder).glob("*.txt"))]


def run_one(text, name, extractor, out_dir):
    """Extract one user's text and save it as <out_dir>/<name>.json. Returns the saved path."""
    result = asdict(extractor.extract(text))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{name}.json"
    out_file.write_text(json.dumps({
        "file": name,
        "model": extractor.model,
        "text": text,
        "extracted": {"skills": result["skills"]},
        "negated": result["negated"],
        "dropped_unverified": result["dropped_unverified"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def to_profile(record):
    """A saved user file -> the profile dict CourseRecommendationPipeline.run() accepts as user_input."""
    return {"user_id": record["file"], "skills": record["extracted"]["skills"], "target_roles": []}


def main():
    parser = argparse.ArgumentParser(description="Extract a user's skills and competence levels from free text with an LLM.")
    parser.add_argument("--text", help="the user's description; without it, every .txt in inputs/users_text/ is read")
    parser.add_argument("--name", help="name for the output file when using --text (default: user)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--overwrite", action="store_true", help="redo users that already have output")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    inputs = read_inputs(args.text, args.name, TEXT_DIR)
    if not inputs:
        sys.exit(f"Nothing to do: pass --text, or put .txt files in {TEXT_DIR}")

    extractor = TextProfileExtractor(model=args.model)
    for name, text in inputs:
        if (OUT_DIR / f"{name}.json").exists() and not args.overwrite and args.text is None:
            print(f"[skip] {name} (already extracted)")
            continue
        try:
            out_file = run_one(text, name, extractor, OUT_DIR)
        except Exception as error:  # keep going so one bad input doesn't stop the batch
            print(f"[error] {name}: {error}")
            continue
        record = json.loads(out_file.read_text(encoding="utf-8"))
        skills = ", ".join(f"{s['surface_form']} ({s['mastery_level']})" for s in record["extracted"]["skills"])
        print(f"[done] {name}: {skills or 'no skills'}")
        if record["negated"]:
            print(f"       not counted (user said no): {', '.join(n['skill'] for n in record['negated'])}")
        if record["dropped_unverified"]:
            print(f"       dropped (quote not in text): {', '.join(d['skill'] for d in record['dropped_unverified'])}")


if __name__ == "__main__":
    main()
