from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
FULL_COVERAGE = ROOT / "evaluation" / "full_coverage_ensemble_predictions.csv"
OUT_TABLE = ROOT / "tables" / "llm_enhanced_ensemble_metrics.csv"
OUT_PREDICTIONS = ROOT / "evaluation" / "llm_enhanced_ensemble_predictions.csv"
REPORT = ROOT / "reports" / "llm_enhanced_ensemble_results.md"


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


def weighted_vote(votes: list[tuple[str, int]], fallback: str) -> str:
    scores: Counter[str] = Counter()
    for label, weight in votes:
        if label:
            scores[label] += weight
    if not scores:
        return fallback
    best_score = scores.most_common(1)[0][1]
    winners = sorted(label for label, score in scores.items() if score == best_score)
    return fallback if fallback in winners else winners[0]


def external_weight(row: dict[str, str]) -> int:
    confidence = row.get("confidence", "").lower()
    if confidence == "high":
        return 2
    if confidence == "medium":
        return 1
    return 0


def evaluate_mode(name: str, gold_by_id: dict[str, str], predictions: dict[str, str], total: int, notes: str) -> dict[str, object]:
    ids = sorted(set(gold_by_id) & set(predictions))
    labels = [gold_by_id[blind_id] for blind_id in ids]
    preds = [predictions[blind_id] for blind_id in ids]
    accuracy, macro_f1 = prf(labels, preds) if ids else (0.0, 0.0)
    return {
        "model_or_mode": name,
        "answered_rows": len(ids),
        "total_rows": total,
        "coverage": f"{len(ids) / total:.3f}" if total else "0.000",
        "accuracy": f"{accuracy:.3f}",
        "macro_f1": f"{macro_f1:.3f}",
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic+LLM ensemble variants for GPU-NFBench.")
    parser.add_argument("prediction_files", nargs="+", type=Path)
    parser.add_argument("--full-coverage", type=Path, default=FULL_COVERAGE)
    parser.add_argument("--out-table", type=Path, default=OUT_TABLE)
    parser.add_argument("--out-predictions", type=Path, default=OUT_PREDICTIONS)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    gold_rows = read_csv(GOLD)
    gold_by_id = {row["blind_id"]: row["gold_primary_failure"] for row in gold_rows}
    total = len(gold_by_id)
    deterministic_rows = {row["blind_id"]: row for row in read_csv(args.full_coverage)}
    deterministic = {
        blind_id: row["deterministic_ensemble_prediction"]
        for blind_id, row in deterministic_rows.items()
    }

    metric_rows: list[dict[str, object]] = [
        evaluate_mode(
            "deterministic_ensemble",
            gold_by_id,
            deterministic,
            total,
            "Existing no-abstention deterministic ensemble.",
        )
    ]
    output_rows: list[dict[str, object]] = []

    for pred_path in args.prediction_files:
        llm_rows = {row["blind_id"]: row for row in read_csv(pred_path)}
        llm_name = pred_path.stem
        direct = {
            blind_id: row["primary_failure_label"]
            for blind_id, row in llm_rows.items()
            if row.get("primary_failure_label") and not row.get("error")
        }
        metric_rows.append(
            evaluate_mode(
                f"{llm_name}_direct",
                gold_by_id,
                direct,
                total,
                f"Direct predictions from `{pred_path}`.",
            )
        )

        weighted_predictions: dict[str, str] = {}
        gated_predictions: dict[str, str] = {}
        selective_predictions: dict[str, str] = {}
        for blind_id, det_row in deterministic_rows.items():
            fallback = deterministic[blind_id]
            llm = llm_rows.get(blind_id, {})
            llm_pred = llm.get("primary_failure_label", "")
            llm_ok = bool(llm_pred) and not llm.get("error") and llm_pred != "needs_review"
            deterministic_votes = [
                det_row.get("weak_candidate_no_needs_review_fallback", ""),
                det_row.get("augmented_tfidf_svm", ""),
                det_row.get("augmented_tfidf_logistic", ""),
                det_row.get("two_stage_hierarchical", ""),
                det_row.get("deterministic_ensemble_prediction", ""),
            ]
            llm_vote_weight = external_weight(llm) if llm_ok else 0
            weighted_predictions[blind_id] = weighted_vote(
                [(vote, 1) for vote in deterministic_votes] + [(llm_pred, llm_vote_weight)],
                fallback,
            )

            det_agreement = sum(vote == llm_pred for vote in deterministic_votes)
            if llm_ok and llm_vote_weight > 0 and det_agreement >= 2:
                gated_predictions[blind_id] = llm_pred
            else:
                gated_predictions[blind_id] = fallback

            if llm_ok and (llm_pred == fallback or det_agreement >= 2):
                selective_predictions[blind_id] = llm_pred

            output_rows.append(
                {
                    "blind_id": blind_id,
                    "gold_primary_failure": gold_by_id.get(blind_id, ""),
                    "deterministic_ensemble": fallback,
                    "llm_prediction_file": str(pred_path),
                    "llm_prediction": llm_pred,
                    "llm_confidence": llm.get("confidence", ""),
                    "deterministic_models_agreeing_with_llm": det_agreement,
                    "weighted_vote_prediction": weighted_predictions[blind_id],
                    "gated_override_prediction": gated_predictions[blind_id],
                    "selective_prediction": selective_predictions.get(blind_id, ""),
                }
            )

        metric_rows.extend(
            [
                evaluate_mode(
                    f"{llm_name}_weighted_vote",
                    gold_by_id,
                    weighted_predictions,
                    total,
                    "Vote over deterministic model family plus confidence-weighted LLM vote; deterministic ensemble breaks ties.",
                ),
                evaluate_mode(
                    f"{llm_name}_gated_override",
                    gold_by_id,
                    gated_predictions,
                    total,
                    "LLM can override only when non-low confidence and at least two deterministic models agree with it.",
                ),
                evaluate_mode(
                    f"{llm_name}_agreement_selective",
                    gold_by_id,
                    selective_predictions,
                    total,
                    "Selective mode answers only when LLM agrees with deterministic ensemble or at least two deterministic models.",
                ),
            ]
        )

    write_csv(
        args.out_table,
        metric_rows,
        ["model_or_mode", "answered_rows", "total_rows", "coverage", "accuracy", "macro_f1", "notes"],
    )
    write_csv(
        args.out_predictions,
        output_rows,
        [
            "blind_id",
            "gold_primary_failure",
            "deterministic_ensemble",
            "llm_prediction_file",
            "llm_prediction",
            "llm_confidence",
            "deterministic_models_agreeing_with_llm",
            "weighted_vote_prediction",
            "gated_override_prediction",
            "selective_prediction",
        ],
    )

    lines = [
        "# LLM-Enhanced Ensemble Results",
        "",
        "This report evaluates whether external LLM predictions improve GPU-NFBench classification when combined with the deterministic full-coverage ensemble.",
        "",
        "| model or mode | answered rows | coverage | accuracy | macro F1 | notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        *[
            f"| {row['model_or_mode']} | {row['answered_rows']} | {row['coverage']} | {row['accuracy']} | {row['macro_f1']} | {row['notes']} |"
            for row in metric_rows
        ],
        "",
        "Interpretation: the deterministic ensemble remains the full-coverage fallback unless an external model produces a measurable improvement under the fixed gated and weighted rules above.",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
