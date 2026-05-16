from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import train_expanded_gold_models as egm
import gold_baseline_classifier as gbc


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "reports"
ANNOTATION_DIR = ROOT / "annotation"

MODEL_NAMES = [
    "bm25_knn",
    "naive_bayes",
    "tfidf_logistic",
    "tfidf_linear_svm",
    "bigram_tfidf_logistic",
]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def metrics(labels: list[str], preds: list[str]) -> tuple[float, float]:
    _, acc, macro_f1 = gbc.prf(labels, preds)
    return acc, macro_f1


def vote(prediction_sets: dict[str, list[str]], tie_order: list[str]) -> list[str]:
    output = []
    for index in range(len(next(iter(prediction_sets.values())))):
        counts = Counter(preds[index] for preds in prediction_sets.values())
        top = counts.most_common(1)[0][1]
        winners = {label for label, count in counts.items() if count == top}
        output.append(next((prediction_sets[name][index] for name in tie_order if prediction_sets[name][index] in winners), counts.most_common(1)[0][0]))
    return output


def expanded_leave_one_repo(rows: list[dict[str, str]]) -> None:
    labels_all = [row["gold_primary_failure"] for row in rows]
    predictions_by_model: dict[str, dict[str, str]] = {
        "candidate_weak_label": {row["blind_id"]: row["candidate_primary_failure"] for row in rows}
    }
    by_repo_rows: list[dict[str, object]] = []

    for model_name in MODEL_NAMES:
        predictions_by_model[model_name] = {}

    for repo in sorted({row["repository"] for row in rows}):
        train = [row for row in rows if row["repository"] != repo]
        test = [row for row in rows if row["repository"] == repo]
        model_repo_predictions: dict[str, list[str]] = {}
        for model_name in MODEL_NAMES:
            preds = gbc.predict_model(model_name, train, test, sorted({row["gold_primary_failure"] for row in rows}))
            model_repo_predictions[model_name] = preds
            for row, pred in zip(test, preds):
                predictions_by_model[model_name][row["blind_id"]] = pred
            acc, macro_f1 = metrics([row["gold_primary_failure"] for row in test], preds)
            by_repo_rows.append(
                {
                    "model_or_mode": model_name,
                    "held_out_repository": repo,
                    "test_issues": len(test),
                    "accuracy": f"{acc:.3f}",
                    "macro_f1": f"{macro_f1:.3f}",
                }
            )

        ensemble_inputs = {
            "candidate_weak_label": [row["candidate_primary_failure"] for row in test],
            "tfidf_linear_svm": model_repo_predictions["tfidf_linear_svm"],
            "tfidf_logistic": model_repo_predictions["tfidf_logistic"],
            "bigram_tfidf_logistic": model_repo_predictions["bigram_tfidf_logistic"],
            "naive_bayes": model_repo_predictions["naive_bayes"],
        }
        ensemble_preds = vote(
            ensemble_inputs,
            ["tfidf_linear_svm", "tfidf_logistic", "candidate_weak_label", "bigram_tfidf_logistic", "naive_bayes"],
        )
        predictions_by_model.setdefault("expanded_gold_vote_ensemble", {})
        for row, pred in zip(test, ensemble_preds):
            predictions_by_model["expanded_gold_vote_ensemble"][row["blind_id"]] = pred
        acc, macro_f1 = metrics([row["gold_primary_failure"] for row in test], ensemble_preds)
        by_repo_rows.append(
            {
                "model_or_mode": "expanded_gold_vote_ensemble",
                "held_out_repository": repo,
                "test_issues": len(test),
                "accuracy": f"{acc:.3f}",
                "macro_f1": f"{macro_f1:.3f}",
            }
        )

    summary_rows = []
    for model_name, pred_map in predictions_by_model.items():
        preds = [pred_map[row["blind_id"]] for row in rows]
        acc, macro_f1 = metrics(labels_all, preds)
        summary_rows.append(
            {
                "model_or_mode": model_name,
                "evaluation": "leave_one_repository_out",
                "answered_rows": len(rows),
                "coverage": "1.000",
                "accuracy": f"{acc:.3f}",
                "macro_f1": f"{macro_f1:.3f}",
            }
        )

    write_csv(
        TABLE_DIR / "expanded_gold_loro_metrics.csv",
        summary_rows,
        ["model_or_mode", "evaluation", "answered_rows", "coverage", "accuracy", "macro_f1"],
    )
    write_csv(
        TABLE_DIR / "expanded_gold_loro_by_repo.csv",
        by_repo_rows,
        ["model_or_mode", "held_out_repository", "test_issues", "accuracy", "macro_f1"],
    )


def ablations_and_errors(rows: list[dict[str, str]]) -> None:
    predictions = read_csv(ROOT / "evaluation" / "expanded_gold_model_predictions.csv")
    labels = [row["gold_primary_failure"] for row in predictions]
    pred_sets = {
        "full_ensemble": [row["expanded_gold_vote_ensemble_prediction"] for row in predictions],
        "no_candidate_label_ensemble": vote(
            {
                "tfidf_linear_svm": [row["tfidf_linear_svm_prediction"] for row in predictions],
                "tfidf_logistic": [row["tfidf_logistic_prediction"] for row in predictions],
                "bigram_tfidf_logistic": [row["bigram_tfidf_logistic_prediction"] for row in predictions],
                "naive_bayes": [row["naive_bayes_prediction"] for row in predictions],
            },
            ["tfidf_linear_svm", "tfidf_logistic", "bigram_tfidf_logistic", "naive_bayes"],
        ),
        "linear_only_vote": vote(
            {
                "tfidf_linear_svm": [row["tfidf_linear_svm_prediction"] for row in predictions],
                "tfidf_logistic": [row["tfidf_logistic_prediction"] for row in predictions],
                "bigram_tfidf_logistic": [row["bigram_tfidf_logistic_prediction"] for row in predictions],
            },
            ["tfidf_linear_svm", "tfidf_logistic", "bigram_tfidf_logistic"],
        ),
        "candidate_plus_svm": vote(
            {
                "candidate_weak_label": [row["candidate_weak_label_prediction"] for row in predictions],
                "tfidf_linear_svm": [row["tfidf_linear_svm_prediction"] for row in predictions],
            },
            ["tfidf_linear_svm", "candidate_weak_label"],
        ),
    }
    ablation_rows = []
    for name, preds in pred_sets.items():
        acc, macro_f1 = metrics(labels, preds)
        ablation_rows.append(
            {
                "model_or_mode": name,
                "answered_rows": len(labels),
                "coverage": "1.000",
                "accuracy": f"{acc:.3f}",
                "macro_f1": f"{macro_f1:.3f}",
            }
        )
    write_csv(TABLE_DIR / "expanded_gold_ablation_metrics.csv", ablation_rows, ["model_or_mode", "answered_rows", "coverage", "accuracy", "macro_f1"])

    metadata = {row["blind_id"]: row for row in rows}
    confusion = Counter()
    examples_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row, pred in zip(predictions, pred_sets["full_ensemble"]):
        gold = row["gold_primary_failure"]
        if gold == pred:
            continue
        pair = (gold, pred)
        confusion[pair] += 1
        examples_by_pair.setdefault(pair, metadata[row["blind_id"]])

    error_rows = []
    for (gold, pred), count in confusion.most_common(12):
        example = examples_by_pair[(gold, pred)]
        error_rows.append(
            {
                "gold_primary_failure": gold,
                "predicted_primary_failure": pred,
                "errors": count,
                "example_repository": example["repository"],
                "example_issue_number": example["issue_number"],
                "example_title": example["title"][:180],
            }
        )
    write_csv(
        TABLE_DIR / "expanded_gold_top_error_pairs.csv",
        error_rows,
        ["gold_primary_failure", "predicted_primary_failure", "errors", "example_repository", "example_issue_number", "example_title"],
    )

    REPORT_DIR.joinpath("expanded_gold_conference_strengthening.md").write_text(
        "\n".join(
            [
                "# Expanded Gold Conference Strengthening Analysis",
                "",
                "## Leave-one-repository-out summary",
                "",
                "| model/mode | accuracy | macro F1 |",
                "| --- | ---: | ---: |",
                *[
                    f"| {row['model_or_mode']} | {row['accuracy']} | {row['macro_f1']} |"
                    for row in read_csv(TABLE_DIR / "expanded_gold_loro_metrics.csv")
                ],
                "",
                "## Ablation summary",
                "",
                "| model/mode | accuracy | macro F1 |",
                "| --- | ---: | ---: |",
                *[f"| {row['model_or_mode']} | {row['accuracy']} | {row['macro_f1']} |" for row in ablation_rows],
                "",
                "## Top ensemble error pairs",
                "",
                "| gold | predicted | errors | example |",
                "| --- | --- | ---: | --- |",
                *[
                    f"| {row['gold_primary_failure']} | {row['predicted_primary_failure']} | {row['errors']} | {row['example_repository']}#{row['example_issue_number']}: {row['example_title']} |"
                    for row in error_rows
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )


def create_expanded_audit_packet(rows: list[dict[str, str]], n: int = 120) -> None:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["source_split"] == "expanded_1000":
            by_label[row["gold_primary_failure"]].append(row)
    selected = []
    per_label = max(1, n // len(by_label))
    for label in sorted(by_label):
        candidates = sorted(by_label[label], key=lambda row: (row["repository"], row["blind_id"]))
        selected.extend(candidates[:per_label])
    selected = selected[:n]
    audit_rows = []
    for index, row in enumerate(selected, start=1):
        audit_rows.append(
            {
                "audit_id": f"EG-AUDIT-{index:03d}",
                "blind_id": row["blind_id"],
                "repository": row["repository"],
                "issue_number": row["issue_number"],
                "url": row["url"],
                "title": row["title"],
                "github_labels": row.get("github_labels", ""),
                "annotator_primary_failure": "",
                "annotator_is_true_numerical_failure": "",
                "annotator_confidence": "",
                "evidence_quote": "",
                "notes": "",
            }
        )
    write_csv(
        ANNOTATION_DIR / "expanded_gold_agreement_audit_120_blind.csv",
        audit_rows,
        [
            "audit_id",
            "blind_id",
            "repository",
            "issue_number",
            "url",
            "title",
            "github_labels",
            "annotator_primary_failure",
            "annotator_is_true_numerical_failure",
            "annotator_confidence",
            "evidence_quote",
            "notes",
        ],
    )
    ANNOTATION_DIR.joinpath("EXPANDED_GOLD_AUDIT_120_INSTRUCTIONS.md").write_text(
        "\n".join(
            [
                "# Expanded Gold 120-Row Agreement Audit",
                "",
                "Annotate the blind CSV without looking at gold labels, candidate labels, model predictions, or repair logs.",
                "",
                "Allowed primary labels: nan_inf, overflow_underflow, precision_tolerance, dtype_casting, crash_compile, performance_only, not_numerical_failure.",
                "",
                "For each row, read the issue URL and any public issue context needed. Fill primary failure, true numerical failure status, confidence, evidence quote, and notes.",
                "",
                "After two independent passes, compare against the expanded gold label and compute observed agreement, Cohen's kappa, and disagreements by class.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    rows = egm.build_rows()
    expanded_leave_one_repo(rows)
    ablations_and_errors(rows)
    create_expanded_audit_packet(rows)
    print(REPORT_DIR / "expanded_gold_conference_strengthening.md")


if __name__ == "__main__":
    main()
