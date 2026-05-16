from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import gold_baseline_classifier as gbc  # noqa: E402


ZERO_SHOT = ROOT / "evaluation" / "llm_baseline_predictions_ollama_llama3.2_3b.csv"
RAG = ROOT / "evaluation" / "llm_rag_predictions_ollama_llama3.2_3b.csv"
PREDICTIONS = ROOT / "evaluation" / "agentic_ensemble_predictions.csv"
METRICS = ROOT / "tables" / "agentic_ensemble_metrics.csv"
CURVE = ROOT / "tables" / "agentic_abstention_curve.csv"
REPORT = ROOT / "reports" / "agentic_ensemble_abstention.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def load_llm_predictions(path: Path) -> dict[str, str]:
    return {row["blind_id"]: row["primary_failure_label"] for row in read_csv(path)}


def subset_metric(
    rows: list[dict[str, object]],
    mode: str,
    selected: list[dict[str, object]],
    pred_column: str = "agentic_prediction",
) -> dict[str, object]:
    labels = [str(row["gold_primary_failure"]) for row in selected]
    preds = [str(row[pred_column]) for row in selected]
    accuracy, macro_f1 = prf(labels, preds)
    return {
        "mode": mode,
        "answered_rows": len(selected),
        "total_rows": len(rows),
        "coverage": f"{(len(selected) / len(rows)):.3f}" if rows else "0.000",
        "accuracy": f"{accuracy:.3f}",
        "macro_f1": f"{macro_f1:.3f}",
    }


def main() -> None:
    rows = gbc.build_rows()
    ids = [row["blind_id"] for row in rows]
    gold = [row["gold_primary_failure"] for row in rows]
    components: dict[str, list[str]] = {
        "candidate_weak_label": [row["candidate_primary_failure"] for row in rows],
    }
    for model_name in ["bm25_knn", "naive_bayes", "tfidf_logistic", "tfidf_linear_svm", "bigram_tfidf_logistic"]:
        components[model_name] = gbc.cross_val_predictions(rows, model_name)
    zero = load_llm_predictions(ZERO_SHOT)
    rag = load_llm_predictions(RAG)
    components["llm_zero_shot"] = [zero[bid] for bid in ids]
    components["llm_fold_safe_rag"] = [rag[bid] for bid in ids]

    prediction_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        votes = {name: preds[index] for name, preds in components.items()}
        counts = Counter(votes.values())
        top = counts.most_common()
        agentic_prediction = top[0][0]
        agreement_count = top[0][1]
        runner_up_count = top[1][1] if len(top) > 1 else 0
        prediction_rows.append(
            {
                "blind_id": row["blind_id"],
                "gold_primary_failure": row["gold_primary_failure"],
                **votes,
                "agentic_prediction": agentic_prediction,
                "agreement_count": agreement_count,
                "agreement_margin": agreement_count - runner_up_count,
                "candidate_tfidf_svm_agree": str(votes["candidate_weak_label"] == votes["tfidf_linear_svm"]).lower(),
                "candidate_tfidf_svm_rag_agree": str(
                    votes["candidate_weak_label"] == votes["tfidf_linear_svm"] == votes["llm_fold_safe_rag"]
                ).lower(),
                "candidate_tfidf_svm_zero_agree": str(
                    votes["candidate_weak_label"] == votes["tfidf_linear_svm"] == votes["llm_zero_shot"]
                ).lower(),
            }
        )

    metric_rows: list[dict[str, object]] = []
    metric_rows.append(subset_metric(prediction_rows, "full_coverage_vote", prediction_rows))

    for threshold in range(2, len(components) + 1):
        selected = [row for row in prediction_rows if int(row["agreement_count"]) >= threshold]
        row = subset_metric(prediction_rows, f"vote_agreement_at_least_{threshold}", selected)
        metric_rows.append(row)

    fixed_rules = {
        "candidate_and_tfidf_svm_agree": [
            row for row in prediction_rows if row["candidate_tfidf_svm_agree"] == "true"
        ],
        "candidate_tfidf_svm_and_rag_llm_agree": [
            row for row in prediction_rows if row["candidate_tfidf_svm_rag_agree"] == "true"
        ],
        "candidate_tfidf_svm_and_zero_llm_agree": [
            row for row in prediction_rows if row["candidate_tfidf_svm_zero_agree"] == "true"
        ],
    }
    for mode, selected in fixed_rules.items():
        # For these rules, the shared prediction is the candidate/SVM/LLM agreement label.
        metric_rows.append(subset_metric(prediction_rows, mode, selected, pred_column="candidate_weak_label"))

    curve_rows = [
        {
            "min_vote_agreement": row["mode"].replace("vote_agreement_at_least_", ""),
            "answered_rows": row["answered_rows"],
            "coverage": row["coverage"],
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
        }
        for row in metric_rows
        if str(row["mode"]).startswith("vote_agreement_at_least_")
    ]

    write_csv(
        PREDICTIONS,
        prediction_rows,
        [
            "blind_id",
            "gold_primary_failure",
            *components.keys(),
            "agentic_prediction",
            "agreement_count",
            "agreement_margin",
            "candidate_tfidf_svm_agree",
            "candidate_tfidf_svm_rag_agree",
            "candidate_tfidf_svm_zero_agree",
        ],
    )
    write_csv(METRICS, metric_rows, ["mode", "answered_rows", "total_rows", "coverage", "accuracy", "macro_f1"])
    write_csv(CURVE, curve_rows, ["min_vote_agreement", "answered_rows", "coverage", "accuracy", "macro_f1"])

    lines = [
        "# Agentic Ensemble Abstention",
        "",
        "This analysis evaluates an answer/abstain triage agent. It combines deterministic cross-validation predictions, the weak pre-classifier, a zero-shot local LLM, and a fold-safe RAG local LLM. Agreement is computed from predictions only; gold labels are used only for evaluation.",
        "",
        "Full-coverage performance remains far below the requested 70-80% range, so the only honest way to reach that range is selective answering with explicit coverage.",
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
        "- The full-coverage vote is still not conference-strong as an accuracy model.",
        "- A vote-agreement threshold of 6/8 reaches the requested 70% range but answers only 51/191 rows.",
        "- The stricter candidate+TF-IDF-SVM+RAG-LLM agreement rule reaches higher accuracy but with lower coverage.",
        "- These are selective triage results, not full automatic classification results.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
