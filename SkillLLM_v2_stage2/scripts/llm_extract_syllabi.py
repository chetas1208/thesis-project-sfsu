

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

STAGE2_DIR = Path(__file__).resolve().parent.parent
TEXT_DIR = STAGE2_DIR / "inputs" / "syllabi_text"
OUT_DIR = STAGE2_DIR / "inputs" / "syllabi_parsed"

# Share of a quote's words that must appear in the syllabus to count as "approximate".
APPROX_WORD_SHARE = 0.85

SYSTEM_PROMPT = """You extract structured information from university course syllabi.

Rules:
- Use only information stated in the syllabus. Never guess course codes, titles, or prerequisites.
- If a field is not in the syllabus, use null (or [] for lists).
- course_id is this course's single primary code (e.g. "CSC 667"); put any other number it is
  Grad_equivalent as (e.g. the graduate "CSC 867" in "CSC 667/867") in Grad_equivalent.
- If a prerequisite is written as a bare number (e.g. "C or better in 413"), give it the course's
  department prefix (e.g. "CSC 413").
- Prerequisites are groups. Every group is required; any ONE course inside a group satisfies it.
  - Courses joined by "or" go in the same group: "CSC 210 or CSC 215" -> {"any_of": ["CSC 210", "CSC 215"]}.
  - Courses joined by "and" or commas go in separate groups: "CSC 256 and MATH 324" -> two groups.
  - A course that is only recommended (e.g. "CSC 656 recommended") gets its own group with "recommended": true.
  - Only course codes go in any_of. Anything else ("graduate standing", "consent of instructor",
    "grade of C or better") goes in other_requirements.
- Every "evidence" value must be ONE short passage copied word-for-word from the syllabus.
  Do not summarize, paraphrase, or join several passages with ";" — pick the single best quote.
- learning_outcomes and topics must be copied as written, not reworded.
- Skills:
  - "explicit": the skill is named in the syllabus (e.g. "Java", "SQL", "unit testing").
  - "inferred": not named, but clearly implied by what students do (e.g. a team software project implies version control).
    Give the quote that supports it and a one-sentence reason.
  - level: "familiarity" (introduced/exposed), "proficiency" (practiced/applied), or "mastery" (designed/built independently at depth).
  - Only technical skills: programming languages, tools, math, and CS/engineering concepts the course
    teaches. List every one of them, including the technical concepts in the course topics and
    schedule (e.g. TCP, recursion, hash tables, file I/O), not only those in the learning outcomes.
  - Leave out soft skills and course logistics: communication, teamwork, writing, presentation,
    ethics, academic integrity, vague abilities like "critical thinking" or "mathematical reasoning",
    and tools used only to attend class or submit work (Zoom, Canvas, iLearn, Gradescope, Piazza,
    Slack, PDF scanner apps). Keep a tool only if the course teaches it (e.g. Git, PyTorch, Wireshark).
  - Name skills concisely as they would appear on a resume.
- Return only JSON matching this shape:
{
  "course_id": "CSC 667" or null,
  "Grad_equivalent": ["CSC 867"],
  "course_title": string or null,
  "term": string or null,
  "instructor": string or null,
  "prerequisites": [{"any_of": ["CSC 210", "CSC 215"], "recommended": false, "evidence": "CSC 210 or CSC 215 with a grade of C or better"}],
  "other_requirements": [{"text": "Grade of C or better", "evidence": "with a grade of C or better"}],
  "learning_outcomes": [string],
  "topics": [string],
  "skills": [{"skill": string, "type": "explicit" | "inferred", "level": string, "evidence": string, "reason": string or null}],
  "evidence": {"course_id": "...", "course_title": "..."}
}"""


def load_api_key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env_file = STAGE2_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "OPENAI_API_KEY":
                return value.strip().strip('"').strip("'")
    sys.exit("No API key found. Set OPENAI_API_KEY or add OPENAI_API_KEY=... to SkillLLM_v2_stage2/.env")


def normalize(text):
    """Lowercase, unify quotes/dashes, and collapse whitespace so line breaks don't break matching."""
    text = text.lower()
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-", "•": " ", "": " "}))
    return re.sub(r"\s+", " ", text).strip()


def check_quote(quote, norm_text, text_words):
    """Return 'exact', 'approximate', or 'not_found' for a quote the model claims is in the syllabus."""
    if not quote:
        return "not_found"
    norm_quote = normalize(quote)
    # Whole-word match, so a short quote like "git" doesn't match inside "digital".
    if re.search(rf"(?<![a-z0-9]){re.escape(norm_quote)}(?![a-z0-9])", norm_text):
        return "exact"
    words = re.findall(r"[a-z0-9+#]+", norm_quote)
    if words and sum(w in text_words for w in words) / len(words) >= APPROX_WORD_SHARE:
        return "approximate"
    return "not_found"


CODE_PATTERN = re.compile(r"^\s*([A-Za-z]+)\.?[\s-]*0?(\d{3})(?!\d)")
DEPARTMENT_ALIASES = {"CS": "CSC"}


def canonical_code(code):
    """'CSc 600', 'CSC 0317', 'CSC 220.03', 'CSC 300GW', 'CS 220' -> 'CSC 600', 'CSC 317', 'CSC 220', 'CSC 300', 'CSC 220'."""
    match = CODE_PATTERN.match(code or "")
    if not match:
        return None
    department = match.group(1).upper()
    return f"{DEPARTMENT_ALIASES.get(department, department)} {match.group(2)}"


def group_courses(prereq):
    """The courses in a prerequisite group; old saved results use one {"course": ...} per entry."""
    if "any_of" in prereq:
        return list(prereq.get("any_of") or [])
    return [prereq.get("course")]


def normalize_result(raw):
    """Clean up the model's answer: canonical course codes, the lowest cross-listed number as the
    primary course, prerequisite groups without self-references or duplicates, and non-course
    requirements (e.g. "Graduate standing") moved from prerequisites to other_requirements."""
    result = json.loads(json.dumps(raw))  # deep copy; the raw answer is kept separately

    codes = []
    # The prompt calls the other number Grad_equivalent; older results used cross_listed
    for code in [result.get("course_id"), *(result.get("Grad_equivalent") or []), *(result.get("cross_listed") or [])]:
        code = canonical_code(code)
        if code and code not in codes:
            codes.append(code)
    codes.sort(key=lambda c: int(c.split()[1]))
    result["course_id"] = codes[0] if codes else None
    result["cross_listed"] = codes[1:]

    prerequisites, other = [], list(result.get("other_requirements") or [])
    for prereq in result.get("prerequisites") or []:
        group = []
        for course in group_courses(prereq):
            code = canonical_code(course)
            if code is None:
                if course:
                    other.append({"text": course, "evidence": prereq.get("evidence")})
            elif code not in codes and code not in group:
                group.append(code)
        # "CSC 210 or CSC 220" in CSC 220's own syllabus leaves an empty group: drop it,
        # and drop a group that repeats an earlier one.
        if group and sorted(group) not in (sorted(p["any_of"]) for p in prerequisites):
            prerequisites.append({"any_of": group, "recommended": bool(prereq.get("recommended")),
                                  "evidence": prereq.get("evidence")})
    result["prerequisites"] = prerequisites
    result["other_requirements"] = other
    return result


def check_course_code(code, text):
    """A code like 'CSC 413' is verified if its number appears in the text (allowing 'CSc', 'CSC 0413', bare '413')."""
    numbers = re.findall(r"(?<!\d)0?(\d{3})(?!\d)", code or "")
    if numbers and all(re.search(rf"(?<!\d)0?{n}(?!\d)", text) for n in numbers):
        return "exact"
    return "not_found"


def verify(result, text):
    norm_text = normalize(text)
    text_words = set(re.findall(r"[a-z0-9+#]+", norm_text))
    quote = lambda q: check_quote(q, norm_text, text_words)
    evidence = result.get("evidence") or {}

    report = {
        "course_id": check_course_code(result.get("course_id"), text),
        "course_id_evidence": quote(evidence.get("course_id")),
        "course_title": quote(result.get("course_title")),
        "cross_listed": [check_course_code(c, text) for c in result.get("cross_listed") or []],
        "prerequisites": [
            {"courses": [check_course_code(c, text) for c in p["any_of"]], "evidence": quote(p.get("evidence"))}
            for p in result.get("prerequisites") or []
        ],
        "other_requirements": [quote(r.get("evidence")) for r in result.get("other_requirements") or []],
        "learning_outcomes": [quote(o) for o in result.get("learning_outcomes") or []],
        "topics": [quote(t) for t in result.get("topics") or []],
        "skills": [
            {
                "skill_named_in_text": quote(s.get("skill")) != "not_found",
                "evidence": quote(s.get("evidence")),
            }
            for s in result.get("skills") or []
        ],
    }

    statuses = [report["course_id"], report["course_title"], *report["cross_listed"],
                *report["other_requirements"], *report["learning_outcomes"], *report["topics"]]
    statuses += [s for p in report["prerequisites"] for s in [*p["courses"], p["evidence"]]]
    statuses += [s["evidence"] for s in report["skills"]]
    report["summary"] = {
        "checked": len(statuses),
        "exact": statuses.count("exact"),
        "approximate": statuses.count("approximate"),
        "not_found": statuses.count("not_found"),
    }
    return report


def extract(client, model, text):
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # same syllabus -> (nearly) the same answer, so runs can be compared
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Syllabus:\n\n{text}"},
        ],
    )
    return json.loads(response.choices[0].message.content), response.usage


def save(out_file, name, model, raw, text):
    """Clean up and verify the model's answer, then write it with the raw answer kept for reference."""
    result = normalize_result(raw)
    report = verify(result, text)
    out_file.write_text(json.dumps(
        {"file": name, "model": model, "extracted": result, "verification": report, "llm_raw": raw},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return result, report


def main():
    parser = argparse.ArgumentParser(description="Extract course info and skills from syllabi with an LLM, then verify it.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--files", help="only process files whose name contains this text")
    parser.add_argument("--limit", type=int, help="only process the first N files")
    parser.add_argument("--overwrite", action="store_true", help="redo files that already have output")
    parser.add_argument("--reverify", action="store_true",
                        help="re-run cleanup and verification on saved results without calling the API")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.reverify:
        for out_file in sorted(OUT_DIR.glob("*.json")):
            record = json.loads(out_file.read_text(encoding="utf-8"))
            text = (TEXT_DIR / f"{record['file']}.txt").read_text(encoding="utf-8")
            raw = record.get("llm_raw", record["extracted"])
            save(out_file, record["file"], record["model"], raw, text)
            print(f"[reverified] {out_file.name}")
        return

    client = OpenAI(api_key=load_api_key())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(TEXT_DIR.glob("*.txt"))
    if args.files:
        files = [f for f in files if args.files.lower() in f.name.lower()]
    if args.limit:
        files = files[: args.limit]

    for txt_file in files:
        out_file = OUT_DIR / f"{txt_file.stem}.json"
        if out_file.exists() and not args.overwrite:
            print(f"[skip] {txt_file.name} (already extracted)")
            continue

        text = txt_file.read_text(encoding="utf-8")
        try:
            result, usage = extract(client, args.model, text)
        except Exception as error:  # keep going so one bad file doesn't stop the batch
            print(f"[error] {txt_file.name}: {error}")
            continue

        result, report = save(out_file, txt_file.stem, args.model, result, text)
        s = report["summary"]
        print(f"[done] {txt_file.name}: {result.get('course_id')} | "
              f"{len(result.get('skills') or [])} skills | "
              f"verified {s['exact']} exact, {s['approximate']} approx, {s['not_found']} not found | "
              f"{usage.total_tokens} tokens")


if __name__ == "__main__":
    main()
