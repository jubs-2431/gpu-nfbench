from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "processed" / "gold_benchmark_expanded_v2_canonical.csv"
LINKED = ROOT / "tables" / "linked_fix_evidence_subset.csv"
HUMAN_50 = ROOT / "annotation" / "gpu_nfbench_root_cause_50_adjudication_completed.csv"
OUT = ROOT / "tables" / "root_cause_200_evidence_coded.csv"
LABEL_COUNTS = ROOT / "tables" / "root_cause_200_label_counts.csv"
PROVENANCE_COUNTS = ROOT / "tables" / "root_cause_200_provenance_counts.csv"
REPORT = ROOT / "reports" / "root_cause_200_evidence_coded.md"

ROOT_LABELS = {
    "compiler_backend_or_runtime",
    "dtype_or_type_semantics",
    "non_bug_feature_docs_support",
    "numerical_algorithm_or_tolerance",
    "performance_or_resource_behavior",
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


def clean(value: str, limit: int = 300) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def root_label(row: dict[str, str], text: str) -> tuple[str, str]:
    lower = text.lower()
    issue_label = row.get("gold_primary_failure") or row.get("current_issue_label") or row.get("v2_gold_primary_failure", "")
    if issue_label in {"not_numerical_failure"} or re.search(r"\b(doc|docs|documentation|feature request|question|install|support|rfc)\b", lower):
        return "non_bug_feature_docs_support", "non-bug/support/documentation signal or issue label"
    if issue_label == "performance_only" or re.search(r"\b(performance|slow|memory|oom|throughput|latency|benchmark|resource|vram)\b", lower):
        return "performance_or_resource_behavior", "performance/resource signal or issue label"
    if issue_label == "dtype_casting" or re.search(r"\b(dtype|cast|casting|type promotion|type conversion|fp8|fp16|bf16|float16|int32|int64|uint|complex)\b", lower):
        return "dtype_or_type_semantics", "dtype/type semantic signal or issue label"
    if issue_label in {"nan_inf", "overflow_underflow", "precision_tolerance"} or re.search(r"\b(nan|inf|overflow|underflow|precision|tolerance|incorrect|wrong result|accuracy|numerical|gradient|rounding)\b", lower):
        return "numerical_algorithm_or_tolerance", "numerical/tolerance signal or issue label"
    if issue_label == "crash_compile" or re.search(r"\b(compile|compiler|lowering|runtime|cuda error|illegal memory|segfault|crash|ptx|llvm|triton|kernel|backend)\b", lower):
        return "compiler_backend_or_runtime", "compiler/backend/runtime signal or issue label"
    return "compiler_backend_or_runtime", "fallback to runtime/backend for ambiguous fix evidence"


def fix_category(label: str, text: str) -> str:
    lower = text.lower()
    if "test" in lower:
        return "tests"
    if re.search(r"\b(doc|readme|rst|md)\b", lower):
        return "docs"
    if re.search(r"\b(build|ci|cmake|setup|wheel|conda|bazel)\b", lower):
        return "build_ci_config"
    if label == "dtype_or_type_semantics":
        return "array_dtype_core"
    if label == "compiler_backend_or_runtime":
        return "compiler_runtime_backend"
    if label == "performance_or_resource_behavior":
        return "performance_resource"
    return "core_library"


def main() -> None:
    v2_by_id = {row["blind_id"]: row for row in read_csv(V2)}
    human_rows = read_csv(HUMAN_50)
    human_ids = {row["blind_id"] for row in human_rows}
    linked_by_id = {row["blind_id"]: row for row in read_csv(LINKED)}
    out_rows: list[dict[str, object]] = []

    for row in human_rows:
        out_rows.append(
            {
                "root_cause_id": row["root_cause_id"],
                "blind_id": row["blind_id"],
                "repository": row["repository"],
                "issue_number": row["issue_number"],
                "title": row["title"],
                "issue_label": row["current_issue_label"],
                "root_cause_label": row["adjudicated_root_cause_label"],
                "fix_file_category": row["fix_file_category"],
                "evidence_quote": clean(row["fix_evidence_quote"], 360),
                "confidence": row["root_cause_confidence"],
                "provenance": "human_adjudicated_50",
                "coding_rationale": row["root_cause_notes"],
            }
        )

    candidates: list[tuple[int, str, dict[str, str], str]] = []
    for blind_id, linked in linked_by_id.items():
        if blind_id in human_ids:
            continue
        v2 = v2_by_id.get(blind_id, {})
        evidence = " ".join(
            [
                linked.get("title", ""),
                linked.get("fix_or_root_cause_snippet", ""),
                linked.get("local_diff_or_patch_snippet", ""),
                v2.get("gold_evidence_quote", ""),
            ]
        )
        score = 0
        score += 5 if linked.get("evidence_tier") == "linked_pr_and_local_patch" else 0
        score += 4 if linked.get("evidence_tier") == "linked_pr" else 0
        score += 3 if linked.get("same_repo_fix_ref_urls") else 0
        score += 2 if linked.get("fix_or_root_cause_snippet") else 0
        candidates.append((score, blind_id, {**v2, **linked}, evidence))

    for score, blind_id, row, evidence in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if len(out_rows) >= 200:
            break
        label, rationale = root_label(row, evidence)
        out_rows.append(
            {
                "root_cause_id": f"RC-EV-{len(out_rows) + 1:03d}",
                "blind_id": blind_id,
                "repository": row["repository"],
                "issue_number": row["issue_number"],
                "title": row["title"],
                "issue_label": row.get("gold_primary_failure", ""),
                "root_cause_label": label,
                "fix_file_category": fix_category(label, evidence),
                "evidence_quote": clean(row.get("fix_or_root_cause_snippet") or row.get("gold_evidence_quote") or row.get("title", ""), 360),
                "confidence": "medium" if row.get("explicit_pull_urls") or row.get("same_repo_fix_ref_urls") else "low",
                "provenance": f"evidence_coded_{row.get('evidence_tier', 'issue_text')}",
                "coding_rationale": rationale,
            }
        )

    if len(out_rows) < 200:
        for blind_id, row in v2_by_id.items():
            if len(out_rows) >= 200:
                break
            if blind_id in {out["blind_id"] for out in out_rows}:
                continue
            evidence = " ".join([row.get("title", ""), row.get("gold_evidence_quote", ""), row.get("gold_secondary_cause_labels", "")])
            if not evidence.strip():
                continue
            label, rationale = root_label(row, evidence)
            out_rows.append(
                {
                    "root_cause_id": f"RC-EV-{len(out_rows) + 1:03d}",
                    "blind_id": blind_id,
                    "repository": row["repository"],
                    "issue_number": row["issue_number"],
                    "title": row["title"],
                    "issue_label": row["gold_primary_failure"],
                    "root_cause_label": label,
                    "fix_file_category": fix_category(label, evidence),
                    "evidence_quote": clean(row.get("gold_evidence_quote") or row.get("title", ""), 360),
                    "confidence": "low",
                    "provenance": "evidence_coded_issue_text_no_linked_fix",
                    "coding_rationale": rationale,
                }
            )

    label_counts = Counter(str(row["root_cause_label"]) for row in out_rows)
    provenance_counts = Counter(str(row["provenance"]) for row in out_rows)
    write_csv(
        OUT,
        out_rows,
        [
            "root_cause_id",
            "blind_id",
            "repository",
            "issue_number",
            "title",
            "issue_label",
            "root_cause_label",
            "fix_file_category",
            "evidence_quote",
            "confidence",
            "provenance",
            "coding_rationale",
        ],
    )
    write_csv(LABEL_COUNTS, [{"root_cause_label": k, "issues": v} for k, v in sorted(label_counts.items())], ["root_cause_label", "issues"])
    write_csv(PROVENANCE_COUNTS, [{"provenance": k, "issues": v} for k, v in sorted(provenance_counts.items())], ["provenance", "issues"])
    REPORT.write_text(
        "\n".join(
            [
                "# 200-Row Root-Cause Evidence-Coded Extension",
                "",
                "This extension increases root-cause coverage from the 50-row human-adjudicated subset to 200 evidence-coded rows. It does not relabel all 200 rows as human-adjudicated. The `provenance` column distinguishes the original human subset from linked-fix and issue-text evidence-coded rows.",
                "",
                f"Rows: {len(out_rows)}",
                f"Human-adjudicated rows retained: {sum(1 for row in out_rows if row['provenance'] == 'human_adjudicated_50')}",
                "",
                "## Root-cause label counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(label_counts.items())],
                "",
                "## Provenance counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(provenance_counts.items())],
                "",
                "Conference-use guidance: report the 50-row subset as human-adjudicated and the 200-row file as evidence-coded root-cause supervision/provenance for future work.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
