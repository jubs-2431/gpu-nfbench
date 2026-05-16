from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "annotation" / "gold_expansion_1000_filled_from_downloads.csv"
OUT = ROOT / "annotation" / "gold_expansion_1000_repaired.csv"
CHANGES = ROOT / "annotation" / "gold_expansion_1000_taxonomy_repair_changes.csv"
REPORT = ROOT / "reports" / "gold_expansion_1000_taxonomy_repair.md"

ALLOWED = {
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
}

PATTERNS = {
    "nan_inf": re.compile(r"\b(nan|inf|infinite|non[- ]?finite)\b", re.I),
    "overflow_underflow": re.compile(r"\b(over[- ]?flow|under[- ]?flow|saturat|wraparound)\b", re.I),
    "dtype_casting": re.compile(r"\b(dtype|cast|casting|promotion|float8|float16|bfloat16|bf16|fp8|fp16|fp32|fp64|int8|int16|int32|int64|complex)\b", re.I),
    "precision_tolerance": re.compile(r"\b(incorrect|wrong|mismatch|precision|tolerance|allclose|rtol|atol|rounding|transpos|false output|breakdown|check_grads|numerical error|order of operations)\b", re.I),
    "crash_compile": re.compile(r"\b(crash|segfault|exception|fatal|compile|compilation|build|can't build|cannot build|ptx|nvvm|llvm|attributeerror|error:|fails?)\b", re.I),
    "performance_only": re.compile(r"\b(performance|slow|slower|throughput|latency|benchmark|regression|overhead|tflops|spills?)\b", re.I),
    "not_numerical_failure": re.compile(r"\b(feature|request|proposal|documentation|docs|question|install|support|refactor|rfc|api|task|fea|qst)\b", re.I),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_secondary(existing: str, label: str) -> str:
    labels = [part.strip() for part in (existing or "").replace(",", "|").split("|") if part.strip()]
    if label not in labels:
        labels.append(label)
    return "|".join(labels) if labels else "unknown"


def heuristic_label(row: dict[str, str]) -> str:
    text = " ".join(
        [
            row.get("title", ""),
            row.get("github_labels", ""),
            row.get("body_excerpt", ""),
            row.get("evidence_quote", ""),
            row.get("notes", ""),
        ]
    )
    true_failure = row.get("is_true_numerical_failure", "").strip().lower()

    if true_failure == "no":
        if PATTERNS["performance_only"].search(text):
            return "performance_only"
        if PATTERNS["not_numerical_failure"].search(text):
            return "not_numerical_failure"
        if PATTERNS["crash_compile"].search(text):
            return "crash_compile"
        return "not_numerical_failure"

    for label in ["nan_inf", "overflow_underflow", "dtype_casting", "precision_tolerance", "crash_compile", "performance_only"]:
        if PATTERNS[label].search(text):
            return label
    return "precision_tolerance"


def repair_label(row: dict[str, str]) -> tuple[str, str, str]:
    original = row.get("primary_failure_label", "").strip()
    secondary = row.get("secondary_cause_labels", "").strip()

    if original in ALLOWED:
        return original, secondary or "unknown", "unchanged"

    if original == "api_feature_request":
        return "not_numerical_failure", add_secondary(secondary, "api_semantics"), "api_feature_request_to_not_numerical_failure"
    if original == "performance_regression":
        return "performance_only", secondary or "unknown", "performance_regression_to_performance_only"
    if original == "compiler_codegen":
        repaired = heuristic_label({**row, "is_true_numerical_failure": "yes" if row.get("is_true_numerical_failure") == "yes" else row.get("is_true_numerical_failure", "")})
        if repaired == "not_numerical_failure" and row.get("is_true_numerical_failure") == "yes":
            repaired = "precision_tolerance"
        return repaired, add_secondary(secondary, "compiler_codegen"), f"compiler_codegen_to_{repaired}"
    if original == "needs_review":
        repaired = heuristic_label(row)
        return repaired, secondary or "unknown", f"needs_review_to_{repaired}"

    repaired = heuristic_label(row)
    return repaired, secondary or "unknown", f"unknown_label_{original}_to_{repaired}"


def main() -> None:
    rows = read_csv(SOURCE)
    fieldnames = list(rows[0].keys())
    repaired_rows: list[dict[str, str]] = []
    change_rows: list[dict[str, str]] = []

    for row in rows:
        before = row.get("primary_failure_label", "").strip()
        primary, secondary, rule = repair_label(row)
        out = dict(row)
        out["primary_failure_label"] = primary
        out["secondary_cause_labels"] = secondary
        if rule != "unchanged":
            note = out.get("notes", "").strip()
            repair_note = f"Taxonomy repair: {rule}; original_primary={before}."
            out["notes"] = f"{note} {repair_note}".strip()
            change_rows.append(
                {
                    "expansion_id": row["expansion_id"],
                    "repository": row["repository"],
                    "title": row["title"],
                    "url": row["url"],
                    "original_primary_failure": before,
                    "repaired_primary_failure": primary,
                    "repaired_secondary_cause_labels": secondary,
                    "repair_rule": rule,
                    "is_true_numerical_failure": row.get("is_true_numerical_failure", ""),
                    "confidence": row.get("confidence", ""),
                }
            )
        repaired_rows.append(out)

    write_csv(OUT, repaired_rows, fieldnames)
    write_csv(
        CHANGES,
        change_rows,
        [
            "expansion_id",
            "repository",
            "title",
            "url",
            "original_primary_failure",
            "repaired_primary_failure",
            "repaired_secondary_cause_labels",
            "repair_rule",
            "is_true_numerical_failure",
            "confidence",
        ],
    )

    counts = Counter(row["primary_failure_label"] for row in repaired_rows)
    rule_counts = Counter(row["repair_rule"] for row in change_rows)
    invalid = [row for row in repaired_rows if row["primary_failure_label"] not in ALLOWED]
    REPORT.write_text(
        "\n".join(
            [
                "# Gold Expansion 1000 Taxonomy Repair",
                "",
                f"Source: `{SOURCE.relative_to(ROOT)}`",
                f"Repaired file: `{OUT.relative_to(ROOT)}`",
                f"Change log: `{CHANGES.relative_to(ROOT)}`",
                f"Rows repaired: {len(change_rows)}",
                f"Invalid labels remaining: {len(invalid)}",
                "",
                "Repairs only map out-of-taxonomy primary labels into the existing seven-label benchmark taxonomy. Original labels are preserved in the change log and appended to row notes.",
                "",
                "## Repair rules",
                "",
                *[f"- {rule}: {count}" for rule, count in sorted(rule_counts.items())],
                "",
                "## Repaired primary label counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(counts.items())],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
