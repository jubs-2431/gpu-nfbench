from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
DEFAULT_EXPANSIONS = [
    ROOT / "annotation" / "full_coverage_expansion_human_todo.csv",
    ROOT / "annotation" / "gold_expansion_1000_blind.csv",
]
OUT = ROOT / "data" / "processed" / "gold_benchmark_expanded.csv"
COUNTS = ROOT / "tables" / "expanded_gold_primary_counts.csv"
REPORT = ROOT / "reports" / "expanded_gold_merge_report.md"

ALLOWED_PRIMARY = {
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ISSUE_RE = re.compile(r"https://github.com/([^/]+/[^/]+)/issues/(\d+)")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def complete_human_label(row: dict[str, str]) -> bool:
    return (
        row.get("primary_failure_label", "").strip() in ALLOWED_PRIMARY
        and row.get("confidence", "").strip().lower() in ALLOWED_CONFIDENCE
        and bool(row.get("evidence_quote", "").strip())
        and row.get("is_true_numerical_failure", "").strip().lower() in {"yes", "no", "unclear"}
    )


def issue_parts(url: str) -> tuple[str, str]:
    match = ISSUE_RE.match(url or "")
    return match.groups() if match else ("", "")


def convert_expansion(row: dict[str, str]) -> dict[str, str]:
    repo, issue_number = issue_parts(row.get("url", ""))
    return {
        "blind_id": row.get("expansion_id", ""),
        "repository": row.get("repository", "") or repo,
        "issue_number": row.get("issue_number", "") or issue_number,
        "url": row.get("url", ""),
        "title": row.get("title", ""),
        "github_state": row.get("state", ""),
        "github_labels": row.get("github_labels", ""),
        "gold_primary_failure": row.get("primary_failure_label", "").strip(),
        "gold_secondary_cause_labels": row.get("secondary_cause_labels", "").strip() or "unknown",
        "gold_is_true_numerical_failure": row.get("is_true_numerical_failure", "").strip().lower(),
        "gold_evidence_quote": row.get("evidence_quote", "").strip(),
        "adjudicator_id": "human_expansion_review",
        "adjudication_notes": row.get("notes", "").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed human-reviewed expansion labels into the expanded gold benchmark.")
    parser.add_argument("--expansion", action="append", type=Path, help="Expansion CSV to merge. Can be supplied more than once.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    expansion_paths = args.expansion or DEFAULT_EXPANSIONS
    expansion_paths = [path if path.is_absolute() else ROOT / path for path in expansion_paths]
    out = args.out if args.out.is_absolute() else ROOT / args.out

    gold_rows = read_csv(GOLD)
    fieldnames = list(gold_rows[0].keys())
    seen_urls = {row["url"] for row in gold_rows}
    merged = list(gold_rows)
    completed = 0
    skipped_incomplete = 0
    skipped_duplicate = 0

    for path in expansion_paths:
        for row in read_csv(path):
            if not complete_human_label(row):
                skipped_incomplete += 1
                continue
            if row.get("url", "") in seen_urls:
                skipped_duplicate += 1
                continue
            converted = convert_expansion(row)
            merged.append({field: converted.get(field, "") for field in fieldnames})
            seen_urls.add(converted["url"])
            completed += 1

    write_csv(out, merged, fieldnames)
    counts = Counter(row["gold_primary_failure"] for row in merged)
    write_csv(COUNTS, [{"gold_primary_failure": label, "issues": count} for label, count in sorted(counts.items())], ["gold_primary_failure", "issues"])

    REPORT.write_text(
        "\n".join(
            [
                "# Expanded Gold Merge Report",
                "",
                f"Original gold rows: {len(gold_rows)}",
                f"Completed expansion rows merged: {completed}",
                f"Expanded gold rows written: {len(merged)}",
                f"Skipped incomplete rows: {skipped_incomplete}",
                f"Skipped duplicate URLs: {skipped_duplicate}",
                f"Output: `{out.relative_to(ROOT)}`",
                "",
                "A row is merged only if it has a valid primary label, high/medium/low confidence, yes/no/unclear true-failure value, and a nonempty evidence quote.",
                "",
                "## Expanded primary label counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(counts.items())],
            ]
        ),
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
