from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
OUT_METRICS = ROOT / "tables" / "llm_baseline_metrics.csv"
OUT_CONFUSION = ROOT / "tables" / "llm_baseline_confusion.csv"
OUT_CONFIDENCE = ROOT / "tables" / "llm_baseline_confidence_slices.csv"
REPORT = ROOT / "reports" / "llm_baseline_results.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prf(labels: list[str], predictions: list[str]) -> tuple[float, float]:
    all_labels = sorted(set(labels) | set(predictions))
    accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(labels) if labels else 0.0
    f1s = []
    for label in all_labels:
        tp = sum(y == label and p == label for y, p in zip(labels, predictions))
        fp = sum(y != label and p == label for y, p in zip(labels, predictions))
        fn = sum(y == label and p != label for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return accuracy, sum(f1s) / len(f1s) if f1s else 0.0


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 scripts/evaluate_llm_baseline_predictions.py path/to/predictions.csv")
    pred_path = Path(sys.argv[1])
    gold = {row["blind_id"]: row for row in read_csv(GOLD)}
    predictions = {row["blind_id"]: row for row in read_csv(pred_path)}
    common = sorted(set(gold) & set(predictions))
    if not common:
        raise SystemExit("No overlapping blind_id values between gold and prediction file.")

    labels = [gold[bid]["gold_primary_failure"] for bid in common]
    preds = [predictions[bid]["primary_failure_label"] for bid in common]
    accuracy, macro_f1 = prf(labels, preds)
    metrics = [
        {
            "prediction_file": str(pred_path),
            "evaluated_rows": len(common),
            "accuracy": f"{accuracy:.3f}",
            "macro_f1": f"{macro_f1:.3f}",
        }
    ]
    confidence_rows = []
    if "confidence" in next(iter(predictions.values())).keys():
        for confidence in ["high", "medium", "low"]:
            subset = [bid for bid in common if predictions[bid].get("confidence") == confidence]
            if subset:
                subset_labels = [gold[bid]["gold_primary_failure"] for bid in subset]
                subset_preds = [predictions[bid]["primary_failure_label"] for bid in subset]
                subset_accuracy, subset_macro_f1 = prf(subset_labels, subset_preds)
            else:
                subset_accuracy, subset_macro_f1 = 0.0, 0.0
            confidence_rows.append(
                {
                    "confidence": confidence,
                    "evaluated_rows": len(subset),
                    "accuracy": f"{subset_accuracy:.3f}",
                    "macro_f1": f"{subset_macro_f1:.3f}",
                }
            )
    confusion = [
        {"gold_primary_failure": gold_label, "predicted_primary_failure": pred_label, "issues": count}
        for (gold_label, pred_label), count in sorted(Counter(zip(labels, preds)).items())
    ]
    write_csv(OUT_METRICS, metrics, ["prediction_file", "evaluated_rows", "accuracy", "macro_f1"])
    write_csv(OUT_CONFUSION, confusion, ["gold_primary_failure", "predicted_primary_failure", "issues"])
    if confidence_rows:
        write_csv(OUT_CONFIDENCE, confidence_rows, ["confidence", "evaluated_rows", "accuracy", "macro_f1"])
    error_count = sum(bool(predictions[bid].get("error")) for bid in common)
    model_names = sorted(set(predictions[bid].get("model", "unknown") for bid in common))
    total_elapsed = 0.0
    for bid in common:
        try:
            total_elapsed += float(predictions[bid].get("elapsed_seconds", "0") or 0)
        except ValueError:
            pass
    confidence_lines = []
    if confidence_rows:
        confidence_lines = [
            "",
            "Confidence slices:",
            "",
            "| confidence | rows | accuracy | macro F1 |",
            "| --- | ---: | ---: | ---: |",
            *[
                f"| {row['confidence']} | {row['evaluated_rows']} | {row['accuracy']} | {row['macro_f1']} |"
                for row in confidence_rows
            ],
        ]
    REPORT.write_text(
        "\n".join(
            [
                "# LLM Baseline Results",
                "",
                f"Prediction file: `{pred_path}`",
                f"Model(s): {', '.join(model_names)}",
                f"Evaluated rows: {len(common)}",
                f"Accuracy: {accuracy:.3f}",
                f"Macro F1: {macro_f1:.3f}",
                f"Prediction rows with runner errors: {error_count}",
                f"Total recorded generation time: {total_elapsed:.2f} seconds",
                *confidence_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
