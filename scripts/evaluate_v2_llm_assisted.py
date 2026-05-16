from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC = ROOT / "evaluation" / "v2_gold_model_predictions.csv"
LLM_PREDICTIONS = ROOT / "evaluation" / "v2_standalone_seq2seq_llm_predictions.csv"
OUT_PREDICTIONS = ROOT / "evaluation" / "v2_llm_assisted_predictions.csv"
OUT_METRICS = ROOT / "tables" / "v2_llm_assisted_metrics.csv"
OUT_REPORT = ROOT / "reports" / "v2_llm_assisted_training.md"

VOTE_COLUMNS = [
    "tfidf_logistic_prediction",
    "tfidf_linear_svm_prediction",
    "bigram_tfidf_logistic_prediction",
    "expanded_gold_vote_ensemble_prediction",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def majority_vote(labels: list[str]) -> tuple[str, int, int]:
    counts = Counter(labels)
    if not counts:
        return "needs_review", 0, 0
    ranked = counts.most_common()
    top_label, top_count = ranked[0]
    margin = top_count - (ranked[1][1] if len(ranked) > 1 else 0)
    return top_label, top_count, margin


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


def metric_row(mode: str, rows: list[dict[str, object]], pred_key: str, answered_only: bool = False) -> dict[str, object]:
    scored = [row for row in rows if not answered_only or row[pred_key] != "abstain"]
    labels = [str(row["gold_primary_failure"]) for row in scored]
    preds = [str(row[pred_key]) for row in scored]
    accuracy, macro_f1 = prf(labels, preds)
    return {
        "model_or_mode": mode,
        "answered_rows": len(scored),
        "coverage": f"{len(scored) / len(rows):.3f}" if rows else "0.000",
        "accuracy": f"{accuracy:.3f}",
        "macro_f1": f"{macro_f1:.3f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v2 LLM-assisted GPU-NFBench classifiers.")
    parser.add_argument("--deterministic", type=Path, default=DETERMINISTIC)
    parser.add_argument("--llm-predictions", type=Path, default=LLM_PREDICTIONS)
    parser.add_argument("--predictions-out", type=Path, default=OUT_PREDICTIONS)
    parser.add_argument("--metrics-out", type=Path, default=OUT_METRICS)
    parser.add_argument("--report-out", type=Path, default=OUT_REPORT)
    args = parser.parse_args()

    deterministic = {row["blind_id"]: row for row in read_csv(args.deterministic)}
    llm_rows = read_csv(args.llm_predictions)
    rows: list[dict[str, object]] = []
    for llm in llm_rows:
        row_id = llm["id"]
        det = deterministic.get(row_id)
        if det is None:
            continue
        vote_labels = [det[column] for column in VOTE_COLUMNS] + [llm["predicted_primary_failure"]]
        assisted, vote_count, vote_margin = majority_vote(vote_labels)
        det_ensemble = det["expanded_gold_vote_ensemble_prediction"]
        llm_prediction = llm["predicted_primary_failure"]
        rows.append(
            {
                "id": row_id,
                "repository": llm["repository"],
                "gold_primary_failure": llm["gold_primary_failure"],
                "deterministic_ensemble_prediction": det_ensemble,
                "tfidf_linear_svm_prediction": det["tfidf_linear_svm_prediction"],
                "standalone_llm_prediction": llm_prediction,
                "llm_assisted_vote_prediction": assisted,
                "llm_assisted_vote_count": vote_count,
                "llm_assisted_vote_margin": vote_margin,
                "agreement_abstention_prediction": det_ensemble if det_ensemble == llm_prediction else "abstain",
            }
        )

    metric_rows = [
        metric_row("deterministic_ensemble_on_llm_test", rows, "deterministic_ensemble_prediction"),
        metric_row("tfidf_linear_svm_on_llm_test", rows, "tfidf_linear_svm_prediction"),
        metric_row("standalone_llm_on_llm_test", rows, "standalone_llm_prediction"),
        metric_row("llm_assisted_vote_on_llm_test", rows, "llm_assisted_vote_prediction"),
        metric_row("llm_deterministic_agreement_abstention", rows, "agreement_abstention_prediction", answered_only=True),
    ]
    write_csv(
        args.predictions_out,
        rows,
        [
            "id",
            "repository",
            "gold_primary_failure",
            "deterministic_ensemble_prediction",
            "tfidf_linear_svm_prediction",
            "standalone_llm_prediction",
            "llm_assisted_vote_prediction",
            "llm_assisted_vote_count",
            "llm_assisted_vote_margin",
            "agreement_abstention_prediction",
        ],
    )
    write_csv(args.metrics_out, metric_rows, ["model_or_mode", "answered_rows", "coverage", "accuracy", "macro_f1"])
    args.report_out.write_text(
        "\n".join(
            [
                "# V2 LLM-Assisted Classifier Evaluation",
                "",
                f"Standalone LLM predictions: `{args.llm_predictions.relative_to(ROOT)}`",
                f"Deterministic predictions: `{args.deterministic.relative_to(ROOT)}`",
                f"Evaluation rows: {len(rows)}",
                "",
                "| mode | answered rows | coverage | accuracy | macro F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
                *[
                    f"| {row['model_or_mode']} | {row['answered_rows']} | {row['coverage']} | {row['accuracy']} | {row['macro_f1']} |"
                    for row in metric_rows
                ],
                "",
                "The assisted vote combines the standalone LLM with the strongest deterministic classifier outputs. "
                "The agreement-abstention mode answers only when the deterministic ensemble and standalone LLM match.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.report_out)


if __name__ == "__main__":
    main()
