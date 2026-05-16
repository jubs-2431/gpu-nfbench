from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "annotation" / "expanded_gold_audit_119_adjudicated.csv"
EXPANDED_GOLD = ROOT / "data" / "processed" / "gold_benchmark_expanded.csv"
PREDICTIONS = ROOT / "evaluation" / "expanded_gold_model_predictions.csv"
TABLE_DIR = ROOT / "tables"
REPORT = ROOT / "reports" / "expanded_gold_adjudicated_audit.md"
V2_GOLD = ROOT / "data" / "processed" / "gold_benchmark_expanded_adjudicated_v2.csv"

LABELS = [
    "crash_compile",
    "dtype_casting",
    "nan_inf",
    "not_numerical_failure",
    "overflow_underflow",
    "performance_only",
    "precision_tolerance",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def agreement(y1: list[str], y2: list[str], labels: list[str]) -> tuple[float, float, float]:
    n = len(y1)
    observed = sum(a == b for a, b in zip(y1, y2)) / n if n else 0.0
    c1 = Counter(y1)
    c2 = Counter(y2)
    expected = sum((c1[label] / n) * (c2[label] / n) for label in labels) if n else 0.0
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 0.0
    return observed, expected, kappa


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def metric_row(name: str, gold: list[str], pred: list[str]) -> dict[str, object]:
    obs, exp, kappa = agreement(gold, pred, LABELS)
    return {
        "comparison": name,
        "rows": len(gold),
        "accuracy_or_observed_agreement": f"{obs:.3f}",
        "expected_agreement": f"{exp:.3f}",
        "cohens_kappa": f"{kappa:.3f}",
        "macro_f1": f"{macro_f1(gold, pred, LABELS):.3f}",
    }


def main() -> None:
    audit = read_csv(AUDIT)
    expanded = read_csv(EXPANDED_GOLD)
    predictions = {row["blind_id"]: row for row in read_csv(PREDICTIONS)}
    gold_by_id = {row["blind_id"]: row for row in expanded}

    invalid = sorted({row["adjudicated_primary_failure"].strip() for row in audit} - set(LABELS))
    if invalid:
        raise SystemExit(f"Invalid adjudicated labels: {invalid}")

    audit_ids = {row["blind_id"] for row in audit}
    adjudicated = [row["adjudicated_primary_failure"].strip() for row in audit]
    person_a = [row["person_a_primary_failure"].strip() for row in audit]
    person_b = [row["person_b_primary_failure"].strip() for row in audit]
    existing_gold = [gold_by_id[row["blind_id"]]["gold_primary_failure"] for row in audit]

    summary_rows = [
        metric_row("person_a_vs_adjudicated", adjudicated, person_a),
        metric_row("person_b_vs_adjudicated", adjudicated, person_b),
        metric_row("person_a_vs_person_b", person_a, person_b),
        metric_row("existing_expanded_gold_vs_adjudicated", adjudicated, existing_gold),
    ]

    model_fields = {
        "candidate_weak_label": "candidate_weak_label_prediction",
        "tfidf_linear_svm": "tfidf_linear_svm_prediction",
        "tfidf_logistic": "tfidf_logistic_prediction",
        "bigram_tfidf_logistic": "bigram_tfidf_logistic_prediction",
        "expanded_gold_vote_ensemble": "expanded_gold_vote_ensemble_prediction",
    }
    for model_name, field in model_fields.items():
        preds = [predictions[row["blind_id"]][field] for row in audit]
        summary_rows.append(metric_row(f"{model_name}_vs_adjudicated", adjudicated, preds))

    write_csv(
        TABLE_DIR / "expanded_gold_adjudicated_audit_metrics.csv",
        summary_rows,
        ["comparison", "rows", "accuracy_or_observed_agreement", "expected_agreement", "cohens_kappa", "macro_f1"],
    )

    disagreement_rows = []
    for row, old_label in zip(audit, existing_gold):
        new_label = row["adjudicated_primary_failure"].strip()
        if old_label != new_label:
            disagreement_rows.append(
                {
                    "blind_id": row["blind_id"],
                    "repository": row["repository"],
                    "issue_number": row["issue_number"],
                    "title": row["title"],
                    "existing_gold_primary_failure": old_label,
                    "adjudicated_primary_failure": new_label,
                    "person_a_primary_failure": row["person_a_primary_failure"],
                    "person_b_primary_failure": row["person_b_primary_failure"],
                    "adjudication_reason": row["adjudication_reason"],
                }
            )
    write_csv(
        TABLE_DIR / "expanded_gold_audit_gold_revisions.csv",
        disagreement_rows,
        [
            "blind_id",
            "repository",
            "issue_number",
            "title",
            "existing_gold_primary_failure",
            "adjudicated_primary_failure",
            "person_a_primary_failure",
            "person_b_primary_failure",
            "adjudication_reason",
        ],
    )

    v2_rows = []
    for row in expanded:
        row = dict(row)
        if row["blind_id"] in audit_ids:
            audit_row = next(a for a in audit if a["blind_id"] == row["blind_id"])
            row["gold_primary_failure"] = audit_row["adjudicated_primary_failure"].strip()
            row["gold_is_true_numerical_failure"] = audit_row["adjudicated_is_true_numerical_failure"].strip()
            row["gold_evidence_quote"] = audit_row["adjudication_reason"].strip()
            row["adjudicator_id"] = "expanded_audit_adjudicator"
            row["adjudication_notes"] = f"audit_adjudicated_from_previous_gold={gold_by_id[row['blind_id']]['gold_primary_failure']}"
        v2_rows.append(row)
    write_csv(V2_GOLD, v2_rows, list(expanded[0].keys()))

    old_counts = Counter(existing_gold)
    new_counts = Counter(adjudicated)
    REPORT.write_text(
        "\n".join(
            [
                "# Expanded Gold Adjudicated Audit",
                "",
                f"Adjudicated audit rows: {len(audit)}",
                f"Rows where adjudicated audit differs from existing expanded gold: {len(disagreement_rows)}",
                f"Candidate v2 benchmark written: `{V2_GOLD.relative_to(ROOT)}`",
                "",
                "## Metrics",
                "",
                "| comparison | accuracy/agreement | expected agreement | Cohen's kappa | macro F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
                *[
                    f"| {row['comparison']} | {row['accuracy_or_observed_agreement']} | {row['expected_agreement']} | {row['cohens_kappa']} | {row['macro_f1']} |"
                    for row in summary_rows
                ],
                "",
                "## Existing gold vs adjudicated audit label counts on the audit subset",
                "",
                "| label | existing gold | adjudicated audit |",
                "| --- | ---: | ---: |",
                *[f"| {label} | {old_counts[label]} | {new_counts[label]} |" for label in LABELS],
                "",
                "## Interpretation",
                "",
                "- The adjudicated audit provides a higher-quality external check on the expanded labels.",
                "- Because many audit adjudications differ from the existing expanded gold labels, the paper should either report the audit as an external validation subset or retrain/evaluate on the v2 benchmark before making v2 the canonical dataset.",
                "- Retraining the standalone LLM is only necessary if the v2 benchmark replaces the original expanded gold benchmark as the main dataset.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
