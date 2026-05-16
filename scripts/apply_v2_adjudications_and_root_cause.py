from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path("/Users/aryanshah/Downloads")
TABLE_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "reports"

BASE_GOLD = ROOT / "data" / "processed" / "gold_benchmark_expanded.csv"
AUDIT_119 = ROOT / "annotation" / "expanded_gold_audit_119_adjudicated.csv"
ADJ_300_DOWNLOAD = DOWNLOADS / "gpu_nfbench_v2_adjudication_300_completed.csv"
ROOT_50_DOWNLOAD = DOWNLOADS / "gpu_nfbench_root_cause_50_adjudication_completed.csv"
ADJ_300 = ROOT / "annotation" / "gpu_nfbench_v2_adjudication_300_completed.csv"
ROOT_50 = ROOT / "annotation" / "gpu_nfbench_root_cause_50_adjudication_completed.csv"
V2_CANONICAL = ROOT / "data" / "processed" / "gold_benchmark_expanded_v2_canonical.csv"

PRIMARY_LABELS = {
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
}

ROOT_CAUSE_LABELS = {
    "compiler_backend_or_runtime",
    "dtype_or_type_semantics",
    "non_bug_feature_docs_support",
    "numerical_algorithm_or_tolerance",
    "performance_or_resource_behavior",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_completed_files() -> None:
    shutil.copy2(ADJ_300_DOWNLOAD, ADJ_300)
    shutil.copy2(ROOT_50_DOWNLOAD, ROOT_50)


def validate_primary(rows: list[dict[str, str]], label_field: str) -> None:
    invalid = sorted({row[label_field].strip() for row in rows} - PRIMARY_LABELS)
    if invalid:
        raise SystemExit(f"Invalid primary labels in {label_field}: {invalid}")


def apply_v2() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    base = read_csv(BASE_GOLD)
    audit = read_csv(AUDIT_119)
    adj300 = read_csv(ADJ_300)
    validate_primary(audit, "adjudicated_primary_failure")
    validate_primary(adj300, "adjudicated_primary_failure")

    updates: dict[str, dict[str, str]] = {}
    for row in audit:
        updates[row["blind_id"]] = {
            "new_primary": row["adjudicated_primary_failure"].strip(),
            "new_true_failure": row["adjudicated_is_true_numerical_failure"].strip(),
            "new_evidence": row["adjudication_reason"].strip(),
            "source": "audit_119",
        }
    for row in adj300:
        updates[row["blind_id"]] = {
            "new_primary": row["adjudicated_primary_failure"].strip(),
            "new_true_failure": row["adjudicated_is_true_numerical_failure"].strip(),
            "new_evidence": row["adjudication_reason"].strip(),
            "source": "adjudication_300",
        }

    out = []
    revision_rows = []
    for row in base:
        row = dict(row)
        update = updates.get(row["blind_id"])
        if update:
            old = row["gold_primary_failure"]
            row["gold_primary_failure"] = update["new_primary"]
            row["gold_is_true_numerical_failure"] = update["new_true_failure"]
            row["gold_evidence_quote"] = update["new_evidence"]
            row["adjudicator_id"] = "v2_human_adjudication"
            row["adjudication_notes"] = f"{update['source']}; previous_gold={old}"
            revision_rows.append(
                {
                    "blind_id": row["blind_id"],
                    "repository": row["repository"],
                    "issue_number": row["issue_number"],
                    "title": row["title"],
                    "previous_gold_primary_failure": old,
                    "v2_gold_primary_failure": row["gold_primary_failure"],
                    "source": update["source"],
                    "changed": str(old != row["gold_primary_failure"]).lower(),
                }
            )
        out.append(row)

    write_csv(V2_CANONICAL, out, list(base[0].keys()))
    write_csv(
        TABLE_DIR / "v2_gold_label_revisions.csv",
        revision_rows,
        ["blind_id", "repository", "issue_number", "title", "previous_gold_primary_failure", "v2_gold_primary_failure", "source", "changed"],
    )
    return out, revision_rows


def validate_root_cause() -> list[dict[str, str]]:
    rows = read_csv(ROOT_50)
    invalid = sorted({row["adjudicated_root_cause_label"].strip() for row in rows} - ROOT_CAUSE_LABELS)
    if invalid:
        raise SystemExit(f"Unexpected root-cause labels: {invalid}")
    missing = [
        row["root_cause_id"]
        for row in rows
        if not row["adjudicated_root_cause_label"].strip()
        or not row["fix_file_category"].strip()
        or not row["fix_evidence_quote"].strip()
        or not row["root_cause_confidence"].strip()
    ]
    if missing:
        raise SystemExit(f"Incomplete root-cause rows: {missing[:10]}")
    write_csv(
        TABLE_DIR / "root_cause_50_label_counts.csv",
        [{"adjudicated_root_cause_label": label, "issues": count} for label, count in sorted(Counter(row["adjudicated_root_cause_label"] for row in rows).items())],
        ["adjudicated_root_cause_label", "issues"],
    )
    write_csv(
        TABLE_DIR / "root_cause_50_fix_file_category_counts.csv",
        [{"fix_file_category": label, "issues": count} for label, count in sorted(Counter(row["fix_file_category"] for row in rows).items())],
        ["fix_file_category", "issues"],
    )
    return rows


def main() -> None:
    copy_completed_files()
    v2_rows, revisions = apply_v2()
    root_rows = validate_root_cause()
    changed = sum(row["changed"] == "true" for row in revisions)
    label_counts = Counter(row["gold_primary_failure"] for row in v2_rows)
    root_counts = Counter(row["adjudicated_root_cause_label"] for row in root_rows)
    REPORT_DIR.joinpath("v2_canonical_and_root_cause_integration.md").write_text(
        "\n".join(
            [
                "# V2 Canonical Dataset and Root-Cause Integration",
                "",
                f"V2 canonical rows: {len(v2_rows)}",
                f"Human adjudication updates applied: {len(revisions)}",
                f"Rows whose primary label changed from v1: {changed}",
                f"V2 canonical file: `{V2_CANONICAL.relative_to(ROOT)}`",
                "",
                "## V2 label counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(label_counts.items())],
                "",
                "## Root-cause extension",
                "",
                f"Root-cause rows: {len(root_rows)}",
                *[f"- {label}: {count}" for label, count in sorted(root_counts.items())],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(REPORT_DIR / "v2_canonical_and_root_cause_integration.md")


if __name__ == "__main__":
    main()
