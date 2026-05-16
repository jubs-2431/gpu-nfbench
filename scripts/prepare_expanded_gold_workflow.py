from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "annotation"
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

CURRENT_GOLD = DATA / "gold_benchmark.csv"
EXPANSION_REVIEW = ANNOTATION / "full_coverage_expansion_review.csv"
EXPANSION_BLIND = ANNOTATION / "full_coverage_expansion_blind.csv"
EXPANSION_PREFILL = ANNOTATION / "full_coverage_expansion_ai_prefill.csv"
EXPANSION_TODO = ANNOTATION / "full_coverage_expansion_human_todo.csv"
EXPANDED_GOLD = DATA / "gold_benchmark_expanded.csv"
EXPANSION_REPORT = REPORTS / "expanded_gold_workflow.md"

PRIMARY_LABELS = {
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
    "needs_review",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def has_human_label(row: dict[str, str]) -> bool:
    label = row.get("primary_failure_label", "").strip()
    evidence = row.get("evidence_quote", "").strip()
    confidence = row.get("confidence", "").strip()
    return label in PRIMARY_LABELS and bool(evidence) and confidence in {"high", "medium", "low"}


def true_failure(label: str, value: str) -> str:
    value = (value or "").strip().lower()
    if value in {"yes", "no", "unclear"}:
        return value
    if label in {"nan_inf", "overflow_underflow", "precision_tolerance", "dtype_casting", "crash_compile"}:
        return "yes"
    if label in {"performance_only", "not_numerical_failure"}:
        return "no"
    return "unclear"


def make_prefill_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        candidate = row.get("candidate_primary_failure", "").strip()
        candidate_causes = row.get("candidate_cause_labels", "").strip()
        prefill_label = candidate if candidate in PRIMARY_LABELS else "needs_review"
        out.append(
            {
                **row,
                "primary_failure_label": prefill_label,
                "secondary_cause_labels": candidate_causes or row.get("candidate_failure_labels", ""),
                "is_true_numerical_failure": true_failure(prefill_label, row.get("is_true_numerical_failure", "")),
                "confidence": "low",
                "evidence_quote": "",
                "notes": "AI/candidate prefill only; human must verify label and add evidence before this row can become gold.",
            }
        )
    return out


def current_gold_rows() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(CURRENT_GOLD):
        rows.append(
            {
                **row,
                "source": "original_191_gold",
            }
        )
    return rows


def expansion_gold_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if not has_human_label(row):
            continue
        out.append(
            {
                "blind_id": row["expansion_id"],
                "repository": row["repository"],
                "issue_number": row["url"].rstrip("/").split("/")[-1],
                "url": row["url"],
                "title": row["title"],
                "github_state": row["state"],
                "github_labels": row.get("github_labels", ""),
                "gold_primary_failure": row["primary_failure_label"].strip(),
                "gold_secondary_cause_labels": row.get("secondary_cause_labels", "").strip() or "unknown",
                "gold_is_true_numerical_failure": true_failure(
                    row["primary_failure_label"].strip(),
                    row.get("is_true_numerical_failure", ""),
                ),
                "gold_evidence_quote": row.get("evidence_quote", "").strip(),
                "adjudicator_id": "expansion_human_review",
                "adjudication_notes": row.get("notes", "").strip(),
                "source": "full_coverage_expansion_review",
            }
        )
    return out


def main() -> None:
    review_rows = read_csv(EXPANSION_REVIEW)
    blind_rows = read_csv(EXPANSION_BLIND)

    labeled = [row for row in review_rows if has_human_label(row)]
    unlabeled = [row for row in review_rows if not has_human_label(row)]

    prefill = make_prefill_rows(review_rows)
    write_csv(EXPANSION_PREFILL, prefill, list(prefill[0].keys()))
    write_csv(EXPANSION_TODO, unlabeled, list(review_rows[0].keys()))

    expanded_rows = current_gold_rows() + expansion_gold_rows(review_rows)
    write_csv(EXPANDED_GOLD, expanded_rows, list(expanded_rows[0].keys()))

    counts = Counter(row["gold_primary_failure"] for row in expanded_rows)
    count_rows = [
        {"gold_primary_failure": label, "issues": count}
        for label, count in sorted(counts.items())
    ]
    write_csv(TABLES / "expanded_gold_primary_counts.csv", count_rows, ["gold_primary_failure", "issues"])

    lines = [
        "# Expanded Gold Workflow",
        "",
        "This report keeps human gold labels separate from AI/candidate prefills.",
        "",
        f"- Original gold rows: {len(read_csv(CURRENT_GOLD))}",
        f"- Expansion packet rows: {len(review_rows)}",
        f"- Expansion rows with complete human labels: {len(labeled)}",
        f"- Expansion rows still needing human review: {len(unlabeled)}",
        f"- Expanded gold rows written: {len(expanded_rows)}",
        "",
        "Generated files:",
        f"- `{EXPANSION_PREFILL.relative_to(ROOT)}`: candidate/AI prefill to speed annotation; not gold.",
        f"- `{EXPANSION_TODO.relative_to(ROOT)}`: rows still requiring human labels/evidence.",
        f"- `{EXPANDED_GOLD.relative_to(ROOT)}`: original gold plus completed expansion rows only.",
        f"- `{(TABLES / 'expanded_gold_primary_counts.csv').relative_to(ROOT)}`",
        "",
        "A row becomes gold only when `primary_failure_label`, `confidence`, and `evidence_quote` are filled by human review.",
    ]
    EXPANSION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(EXPANSION_REPORT)


if __name__ == "__main__":
    main()
