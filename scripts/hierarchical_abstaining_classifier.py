from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import gold_baseline_classifier as gbc  # noqa: E402


PREDICTIONS = ROOT / "evaluation" / "hierarchical_abstaining_predictions.csv"
METRICS = ROOT / "tables" / "hierarchical_abstaining_metrics.csv"
REPORT = ROOT / "reports" / "hierarchical_abstaining_classifier.md"

NON_NUMERICAL = {"not_numerical_failure", "performance_only"}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prf(labels: list[str], preds: list[str]) -> tuple[float, float]:
    if not labels:
        return 0.0, 0.0
    accuracy = sum(a == b for a, b in zip(labels, preds)) / len(labels)
    all_labels = sorted(set(labels) | set(preds))
    f1s = []
    for label in all_labels:
        tp = sum(a == label and b == label for a, b in zip(labels, preds))
        fp = sum(a != label and b == label for a, b in zip(labels, preds))
        fn = sum(a == label and b != label for a, b in zip(labels, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return accuracy, sum(f1s) / len(f1s) if f1s else 0.0


def binary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    converted = []
    for row in rows:
        label = "non_numerical_or_performance" if row["gold_primary_failure"] in NON_NUMERICAL else "numerical_failure"
        converted.append({**row, "gold_primary_failure": label})
    return converted


def metric_row(name: str, selected: list[dict[str, object]], total: int, pred_col: str = "final_prediction") -> dict[str, object]:
    labels = [str(row["gold_primary_failure"]) for row in selected]
    preds = [str(row[pred_col]) for row in selected]
    accuracy, macro_f1 = prf(labels, preds)
    return {
        "mode": name,
        "answered_rows": len(selected),
        "total_rows": total,
        "coverage": f"{len(selected) / total:.3f}" if total else "0.000",
        "accuracy": f"{accuracy:.3f}",
        "macro_f1": f"{macro_f1:.3f}",
    }


def main() -> None:
    rows = gbc.build_rows()
    labels = [row["gold_primary_failure"] for row in rows]

    binary_predictions = gbc.cross_val_predictions(binary_rows(rows), "tfidf_linear_svm")
    svm_predictions = gbc.cross_val_predictions(rows, "tfidf_linear_svm")
    logistic_predictions = gbc.cross_val_predictions(rows, "tfidf_logistic")
    bigram_predictions = gbc.cross_val_predictions(rows, "bigram_tfidf_logistic")
    nb_predictions = gbc.cross_val_predictions(rows, "naive_bayes")
    candidate_predictions = [row["candidate_primary_failure"] for row in rows]

    prediction_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        multiclass_votes = [
            svm_predictions[index],
            logistic_predictions[index],
            bigram_predictions[index],
            nb_predictions[index],
            candidate_predictions[index],
        ]
        counts = Counter(multiclass_votes)
        top_label, top_count = counts.most_common(1)[0]
        runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        binary_gate = binary_predictions[index]

        if binary_gate == "non_numerical_or_performance":
            non_numeric_votes = [vote for vote in multiclass_votes if vote in NON_NUMERICAL]
            final_prediction = Counter(non_numeric_votes).most_common(1)[0][0] if non_numeric_votes else "not_numerical_failure"
        else:
            numeric_votes = [vote for vote in multiclass_votes if vote not in NON_NUMERICAL]
            final_prediction = Counter(numeric_votes).most_common(1)[0][0] if numeric_votes else top_label

        prediction_rows.append(
            {
                "blind_id": row["blind_id"],
                "repository": row["repository"],
                "gold_primary_failure": row["gold_primary_failure"],
                "binary_gate_prediction": binary_gate,
                "tfidf_linear_svm": svm_predictions[index],
                "tfidf_logistic": logistic_predictions[index],
                "bigram_tfidf_logistic": bigram_predictions[index],
                "naive_bayes": nb_predictions[index],
                "candidate_weak_label": candidate_predictions[index],
                "final_prediction": final_prediction,
                "top_vote_label": top_label,
                "vote_count": top_count,
                "vote_margin": top_count - runner_up,
                "binary_gate_matches_final_family": str(
                    (binary_gate == "non_numerical_or_performance" and final_prediction in NON_NUMERICAL)
                    or (binary_gate == "numerical_failure" and final_prediction not in NON_NUMERICAL)
                ).lower(),
            }
        )

    metric_rows: list[dict[str, object]] = [metric_row("hierarchical_full_coverage", prediction_rows, len(rows))]
    for min_votes in range(2, 6):
        selected = [row for row in prediction_rows if int(row["vote_count"]) >= min_votes]
        metric_rows.append(metric_row(f"hierarchical_vote_at_least_{min_votes}", selected, len(rows)))
    selected = [
        row
        for row in prediction_rows
        if row["binary_gate_matches_final_family"] == "true" and int(row["vote_count"]) >= 3
    ]
    metric_rows.append(metric_row("hierarchical_gate_match_and_vote_at_least_3", selected, len(rows)))

    # Binary gate diagnostic.
    binary_gold = ["non_numerical_or_performance" if label in NON_NUMERICAL else "numerical_failure" for label in labels]
    binary_acc, binary_macro = prf(binary_gold, binary_predictions)
    metric_rows.append(
        {
            "mode": "binary_gate_only",
            "answered_rows": len(rows),
            "total_rows": len(rows),
            "coverage": "1.000",
            "accuracy": f"{binary_acc:.3f}",
            "macro_f1": f"{binary_macro:.3f}",
        }
    )

    write_csv(
        PREDICTIONS,
        prediction_rows,
        [
            "blind_id",
            "repository",
            "gold_primary_failure",
            "binary_gate_prediction",
            "tfidf_linear_svm",
            "tfidf_logistic",
            "bigram_tfidf_logistic",
            "naive_bayes",
            "candidate_weak_label",
            "final_prediction",
            "top_vote_label",
            "vote_count",
            "vote_margin",
            "binary_gate_matches_final_family",
        ],
    )
    write_csv(METRICS, metric_rows, ["mode", "answered_rows", "total_rows", "coverage", "accuracy", "macro_f1"])

    REPORT.write_text(
        "\n".join(
            [
                "# Hierarchical Abstaining Classifier",
                "",
                "This experiment separates numerical-failure detection from symptom classification. The first-stage gate predicts numerical_failure versus non_numerical_or_performance, then the second stage votes among primary failure labels. Abstaining modes answer only when enough model votes agree.",
                "",
                "| mode | answered | coverage | accuracy | macro F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
                *[
                    f"| {row['mode']} | {row['answered_rows']} | {row['coverage']} | {row['accuracy']} | {row['macro_f1']} |"
                    for row in metric_rows
                ],
                "",
                "Interpretation:",
                "",
                "- The binary gate tests whether the easier first question can be solved reliably before multiclass labeling.",
                "- The selective rows are the candidates suitable for automated triage; unanswered rows remain human-review cases.",
                "- These metrics are still evaluated only on existing gold labels; accuracy should be rerun after the 1000-row gold expansion is completed.",
            ]
        ),
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
