from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import gold_baseline_classifier as gbc  # noqa: E402


EXPANDED_GOLD = ROOT / "data" / "processed" / "gold_benchmark_expanded.csv"
ORIGINAL_PACKET = ROOT / "annotation" / "annotator_A_blind.csv"
ORIGINAL_SUGGESTIONS = ROOT / "annotation" / "candidate_label_suggestions_hidden_from_annotators.csv"
EXPANSION_PACKET = ROOT / "annotation" / "gold_expansion_1000_repaired.csv"
EXPANSION_QUEUE = ROOT / "annotation" / "gold_expansion_1000_queue.csv"
OUT_METRICS = ROOT / "tables" / "expanded_gold_classifier_metrics.csv"
OUT_PER_CLASS = ROOT / "tables" / "expanded_gold_classifier_per_class.csv"
OUT_CONFUSION = ROOT / "tables" / "expanded_gold_classifier_confusion.csv"
OUT_ABSTAIN = ROOT / "tables" / "expanded_gold_abstention_metrics.csv"
OUT_PREDICTIONS = ROOT / "evaluation" / "expanded_gold_model_predictions.csv"
REPORT = ROOT / "reports" / "expanded_gold_model_training.md"

MODEL_NAMES = [
    "bm25_knn",
    "naive_bayes",
    "tfidf_logistic",
    "tfidf_linear_svm",
    "bigram_tfidf_logistic",
]

NON_NUMERIC = {"not_numerical_failure", "performance_only"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def original_text(row: dict[str, str]) -> str:
    return " ".join(
        [
            row.get("repository", ""),
            row.get("title", ""),
            row.get("github_labels", ""),
            row.get("issue_body_excerpt", ""),
            row.get("comments_excerpt", ""),
        ]
    )


def expansion_text(row: dict[str, str]) -> str:
    return " ".join(
        [
            row.get("repository", ""),
            row.get("title", ""),
            row.get("github_labels", ""),
            row.get("body_excerpt", ""),
            row.get("evidence_quote", ""),
            row.get("notes", ""),
        ]
    )


def build_rows() -> list[dict[str, str]]:
    gold = read_csv(EXPANDED_GOLD)
    original_packet = {row["blind_id"]: row for row in read_csv(ORIGINAL_PACKET)}
    original_suggestions = {row["blind_id"]: row for row in read_csv(ORIGINAL_SUGGESTIONS)}
    expansion_packet = {row["expansion_id"]: row for row in read_csv(EXPANSION_PACKET)}
    expansion_queue = {row["expansion_id"]: row for row in read_csv(EXPANSION_QUEUE)}

    rows: list[dict[str, str]] = []
    for row in gold:
        blind_id = row["blind_id"]
        if blind_id in original_packet:
            packet = original_packet[blind_id]
            candidate = original_suggestions.get(blind_id, {}).get("candidate_primary_failure", "needs_review")
            text = original_text(packet)
        elif blind_id in expansion_packet:
            packet = expansion_packet[blind_id]
            candidate = expansion_queue.get(blind_id, {}).get("candidate_primary_failure", "needs_review")
            text = expansion_text(packet)
        else:
            packet = row
            candidate = "needs_review"
            text = " ".join([row.get("repository", ""), row.get("title", ""), row.get("github_labels", ""), row.get("gold_evidence_quote", "")])
        rows.append(
            {
                **row,
                "text": text,
                "candidate_primary_failure": candidate,
                "source_split": "original_191" if blind_id.startswith("GNF-") else "expanded_1000",
            }
        )
    return rows


def prf(labels: list[str], preds: list[str]) -> tuple[float, float]:
    _, accuracy, macro_f1 = gbc.prf(labels, preds)
    return accuracy, macro_f1


def confusion_rows(labels: list[str], preds: list[str]) -> list[dict[str, object]]:
    counts = Counter(zip(labels, preds))
    return [
        {"gold_primary_failure": gold, "predicted_primary_failure": pred, "issues": count}
        for (gold, pred), count in sorted(counts.items())
    ]


def binary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    converted = []
    for row in rows:
        label = "non_numeric_or_performance" if row["gold_primary_failure"] in NON_NUMERIC else "numerical_failure"
        converted.append({**row, "gold_primary_failure": label})
    return converted


def vote(predictions: dict[str, list[str]], tie_order: list[str]) -> tuple[list[str], list[int], list[int]]:
    n = len(next(iter(predictions.values())))
    output: list[str] = []
    vote_counts: list[int] = []
    margins: list[int] = []
    for index in range(n):
        counts = Counter(pred[index] for pred in predictions.values())
        top_count = counts.most_common(1)[0][1]
        runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        winners = {label for label, count in counts.items() if count == top_count}
        selected = next((predictions[name][index] for name in tie_order if predictions[name][index] in winners), counts.most_common(1)[0][0])
        output.append(selected)
        vote_counts.append(top_count)
        margins.append(top_count - runner_up)
    return output, vote_counts, margins


def main() -> None:
    rows = build_rows()
    labels = [row["gold_primary_failure"] for row in rows]

    prediction_sets: dict[str, list[str]] = {}
    metric_rows: list[dict[str, object]] = []

    majority = Counter(labels).most_common(1)[0][0]
    prediction_sets["majority_baseline"] = [majority for _ in rows]
    prediction_sets["candidate_weak_label"] = [row["candidate_primary_failure"] for row in rows]
    for model_name in MODEL_NAMES:
        prediction_sets[model_name] = gbc.cross_val_predictions(rows, model_name)

    binary_predictions = gbc.cross_val_predictions(binary_rows(rows), "tfidf_linear_svm")
    binary_gold = ["non_numeric_or_performance" if label in NON_NUMERIC else "numerical_failure" for label in labels]
    binary_accuracy, binary_macro_f1 = prf(binary_gold, binary_predictions)

    ensemble_inputs = {
        name: prediction_sets[name]
        for name in ["candidate_weak_label", "tfidf_linear_svm", "tfidf_logistic", "bigram_tfidf_logistic", "naive_bayes"]
    }
    ensemble, ensemble_vote_counts, ensemble_margins = vote(
        ensemble_inputs,
        ["tfidf_linear_svm", "tfidf_logistic", "candidate_weak_label", "bigram_tfidf_logistic", "naive_bayes"],
    )
    prediction_sets["expanded_gold_vote_ensemble"] = ensemble

    for model_name, preds in prediction_sets.items():
        accuracy, macro_f1 = prf(labels, preds)
        metric_rows.append(
            {
                "model_or_mode": model_name,
                "evaluation": "stratified_5fold",
                "answered_rows": len(rows),
                "coverage": "1.000",
                "accuracy": f"{accuracy:.3f}",
                "macro_f1": f"{macro_f1:.3f}",
            }
        )
    metric_rows.append(
        {
            "model_or_mode": "binary_gate_tfidf_linear_svm",
            "evaluation": "stratified_5fold_binary_gate",
            "answered_rows": len(rows),
            "coverage": "1.000",
            "accuracy": f"{binary_accuracy:.3f}",
            "macro_f1": f"{binary_macro_f1:.3f}",
        }
    )

    abstain_rows: list[dict[str, object]] = []
    for threshold in range(2, len(ensemble_inputs) + 1):
        selected = [index for index, count in enumerate(ensemble_vote_counts) if count >= threshold]
        selected_labels = [labels[index] for index in selected]
        selected_preds = [ensemble[index] for index in selected]
        accuracy, macro_f1 = prf(selected_labels, selected_preds) if selected else (0.0, 0.0)
        abstain_rows.append(
            {
                "mode": f"ensemble_vote_at_least_{threshold}",
                "answered_rows": len(selected),
                "total_rows": len(rows),
                "coverage": f"{len(selected) / len(rows):.3f}",
                "accuracy": f"{accuracy:.3f}",
                "macro_f1": f"{macro_f1:.3f}",
            }
        )

    best_model = max(
        [row for row in metric_rows if row["model_or_mode"] not in {"majority_baseline", "candidate_weak_label"} and row["evaluation"] == "stratified_5fold"],
        key=lambda row: (float(row["macro_f1"]), float(row["accuracy"])),
    )["model_or_mode"]
    best_preds = prediction_sets[str(best_model)]
    per_class, _, _ = gbc.prf(labels, best_preds)

    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "blind_id": row["blind_id"],
                "source_split": row["source_split"],
                "repository": row["repository"],
                "gold_primary_failure": row["gold_primary_failure"],
                **{f"{name}_prediction": preds[index] for name, preds in prediction_sets.items()},
                "ensemble_vote_count": ensemble_vote_counts[index],
                "ensemble_vote_margin": ensemble_margins[index],
            }
        )

    write_csv(OUT_METRICS, metric_rows, ["model_or_mode", "evaluation", "answered_rows", "coverage", "accuracy", "macro_f1"])
    write_csv(OUT_ABSTAIN, abstain_rows, ["mode", "answered_rows", "total_rows", "coverage", "accuracy", "macro_f1"])
    write_csv(OUT_PER_CLASS, per_class, ["label", "support", "precision", "recall", "f1"])
    write_csv(OUT_CONFUSION, confusion_rows(labels, best_preds), ["gold_primary_failure", "predicted_primary_failure", "issues"])
    write_csv(
        OUT_PREDICTIONS,
        prediction_rows,
        [
            "blind_id",
            "source_split",
            "repository",
            "gold_primary_failure",
            *[f"{name}_prediction" for name in prediction_sets],
            "ensemble_vote_count",
            "ensemble_vote_margin",
        ],
    )

    label_counts = Counter(labels)
    source_counts = Counter(row["source_split"] for row in rows)
    REPORT.write_text(
        "\n".join(
            [
                "# Expanded Gold Model Training",
                "",
                f"Expanded gold rows: {len(rows)}",
                f"Source split: {dict(source_counts)}",
                f"Best full-coverage model by macro F1: `{best_model}`",
                "",
                "## Label counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(label_counts.items())],
                "",
                "## Full-coverage metrics",
                "",
                "| model/mode | accuracy | macro F1 |",
                "| --- | ---: | ---: |",
                *[f"| {row['model_or_mode']} | {row['accuracy']} | {row['macro_f1']} |" for row in metric_rows],
                "",
                "## Abstention metrics",
                "",
                "| mode | answered | coverage | accuracy | macro F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
                *[
                    f"| {row['mode']} | {row['answered_rows']} | {row['coverage']} | {row['accuracy']} | {row['macro_f1']} |"
                    for row in abstain_rows
                ],
                "",
                "## Interpretation",
                "",
                "- The expanded 1,191-row gold set is now large enough to train and evaluate stronger deterministic triage baselines.",
                "- The binary gate is reported separately because it measures the easier first-stage decision: numerical failure versus non-numerical/performance-only issue.",
                "- The abstention rows show what accuracy is achievable when the model answers only high-agreement cases.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
