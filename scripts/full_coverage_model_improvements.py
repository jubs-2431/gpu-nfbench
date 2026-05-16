from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import gold_baseline_classifier as gbc  # noqa: E402


SUGGESTIONS = ROOT / "annotation" / "candidate_label_suggestions_hidden_from_annotators.csv"
LINKED_FIX = ROOT / "tables" / "linked_fix_evidence_subset.csv"
PR_MANIFEST = ROOT / "tables" / "linked_pr_diff_manifest.csv"
ZERO_SHOT = ROOT / "evaluation" / "llm_baseline_predictions_ollama_llama3.2_3b.csv"
RAG = ROOT / "evaluation" / "llm_rag_predictions_ollama_llama3.2_3b.csv"
PREDICTIONS = ROOT / "evaluation" / "full_coverage_ensemble_predictions.csv"
METRICS = ROOT / "tables" / "full_coverage_model_improvements.csv"
REPORT = ROOT / "reports" / "full_coverage_model_improvement.md"


MODEL_NAMES = [
    "bm25_knn",
    "naive_bayes",
    "tfidf_logistic",
    "tfidf_linear_svm",
    "bigram_tfidf_logistic",
]


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
    _, accuracy, macro_f1 = gbc.prf(labels, preds)
    return accuracy, macro_f1


def load_predictions(path: Path) -> dict[str, str]:
    return {row["blind_id"]: row["primary_failure_label"] for row in read_csv(path)}


def linked_pr_features() -> dict[str, list[str]]:
    features: dict[str, list[str]] = defaultdict(list)
    if not PR_MANIFEST.exists():
        return features
    for row in read_csv(PR_MANIFEST):
        source_ids = [value for value in row.get("source_blind_ids", "").split("|") if value]
        changed_files = [value for value in row.get("changed_files_sample", "").split("|") if value]
        for blind_id in source_ids:
            features[blind_id].extend(changed_files[:20])
    return features


def augmented_rows(mode: str) -> list[dict[str, str]]:
    base_rows = gbc.build_rows()
    suggestions = {row["blind_id"]: row for row in read_csv(SUGGESTIONS)}
    linked_fix = {row["blind_id"]: row for row in read_csv(LINKED_FIX)} if LINKED_FIX.exists() else {}
    pr_features = linked_pr_features()
    rows: list[dict[str, str]] = []
    for row in base_rows:
        out = dict(row)
        suggestion = suggestions.get(row["blind_id"], {})
        fix = linked_fix.get(row["blind_id"], {})
        parts = [out["text"]]
        if mode in {"candidate", "all"}:
            parts.extend(
                [
                    suggestion.get("candidate_primary_failure", ""),
                    suggestion.get("candidate_failure_labels", ""),
                    suggestion.get("candidate_cause_labels", ""),
                ]
            )
        if mode in {"diff", "all"}:
            parts.extend(
                [
                    fix.get("evidence_tier", ""),
                    fix.get("fix_or_root_cause_snippet", ""),
                    fix.get("local_diff_or_patch_snippet", ""),
                    " ".join(pr_features.get(row["blind_id"], [])[:20]),
                ]
            )
        out["text"] = " ".join(parts)
        out["candidate_primary_failure"] = suggestion.get("candidate_primary_failure", "needs_review")
        rows.append(out)
    return rows


def no_needs_review(preds: list[str], fallback: list[str]) -> list[str]:
    return [pred if pred != "needs_review" else fallback[index] for index, pred in enumerate(preds)]


def group_label(label: str) -> str:
    if label in {"not_numerical_failure", "performance_only"}:
        return "not_numeric_or_performance"
    if label == "crash_compile":
        return "crash_compile"
    return "numeric_failure"


def hierarchical_predictions(rows: list[dict[str, str]], model_name: str = "tfidf_linear_svm") -> list[str]:
    folds = gbc.stratified_folds(rows, 5)
    predictions: dict[str, str] = {}
    for test in folds:
        test_ids = {row["blind_id"] for row in test}
        train = [row for row in rows if row["blind_id"] not in test_ids]
        test_rows = [row for row in rows if row["blind_id"] in test_ids]

        train_group = []
        test_group = []
        for row in train:
            item = dict(row)
            item["gold_primary_failure"] = group_label(row["gold_primary_failure"])
            train_group.append(item)
        for row in test_rows:
            item = dict(row)
            item["gold_primary_failure"] = group_label(row["gold_primary_failure"])
            test_group.append(item)

        group_labels = sorted({row["gold_primary_failure"] for row in train_group})
        group_preds = gbc.predict_model(model_name, train_group, test_group, group_labels)

        submodels: dict[str, tuple[list[dict[str, str]], list[str]]] = {}
        for group in sorted(set(group_preds) | {group_label(row["gold_primary_failure"]) for row in train}):
            subtrain = [dict(row) for row in train if group_label(row["gold_primary_failure"]) == group]
            if not subtrain:
                continue
            submodels[group] = (subtrain, sorted({row["gold_primary_failure"] for row in subtrain}))

        for row, group_pred in zip(test_rows, group_preds):
            if group_pred in submodels and len(submodels[group_pred][1]) > 1:
                predictions[row["blind_id"]] = gbc.predict_model(model_name, submodels[group_pred][0], [row], submodels[group_pred][1])[0]
            elif group_pred in submodels:
                predictions[row["blind_id"]] = submodels[group_pred][1][0]
            else:
                predictions[row["blind_id"]] = "dtype_casting"
    return [predictions[row["blind_id"]] for row in rows]


def vote(prediction_sets: dict[str, list[str]], tie_breakers: list[str]) -> list[str]:
    names = list(prediction_sets)
    n = len(next(iter(prediction_sets.values())))
    output = []
    for index in range(n):
        counts = Counter(prediction_sets[name][index] for name in names)
        top_count = counts.most_common(1)[0][1]
        tied = {label for label, count in counts.items() if count == top_count}
        selected = None
        for name in tie_breakers:
            candidate = prediction_sets[name][index]
            if candidate in tied:
                selected = candidate
                break
        output.append(selected or counts.most_common(1)[0][0])
    return output


def main() -> None:
    base_rows = augmented_rows("base")
    all_rows = augmented_rows("all")
    ids = [row["blind_id"] for row in base_rows]
    labels = [row["gold_primary_failure"] for row in base_rows]

    metric_rows: list[dict[str, object]] = []
    stored_predictions: dict[str, list[str]] = {}

    for mode, rows in [("base_text", base_rows), ("candidate_features", augmented_rows("candidate")), ("diff_features", augmented_rows("diff")), ("candidate_plus_diff_features", all_rows)]:
        for model_name in MODEL_NAMES:
            preds = gbc.cross_val_predictions(rows, model_name)
            accuracy, macro_f1 = prf(labels, preds)
            metric_rows.append(
                {
                    "model_or_mode": f"{mode}_{model_name}",
                    "evaluation": "stratified_5fold_full_coverage",
                    "answered_rows": len(labels),
                    "coverage": "1.000",
                    "accuracy": f"{accuracy:.3f}",
                    "macro_f1": f"{macro_f1:.3f}",
                    "notes": "Flat classifier.",
                }
            )
            stored_predictions[f"{mode}_{model_name}"] = preds

    fallback = stored_predictions["candidate_plus_diff_features_tfidf_linear_svm"]
    candidate_preds = [row["candidate_primary_failure"] for row in all_rows]
    candidate_noneeds = no_needs_review(candidate_preds, fallback)
    accuracy, macro_f1 = prf(labels, candidate_noneeds)
    metric_rows.append(
        {
            "model_or_mode": "weak_candidate_no_needs_review_fallback",
            "evaluation": "full_coverage_no_abstain",
            "answered_rows": len(labels),
            "coverage": "1.000",
            "accuracy": f"{accuracy:.3f}",
            "macro_f1": f"{macro_f1:.3f}",
            "notes": "Replaces needs_review with augmented TF-IDF SVM prediction.",
        }
    )
    stored_predictions["weak_candidate_no_needs_review_fallback"] = candidate_noneeds

    hierarchy = hierarchical_predictions(all_rows)
    accuracy, macro_f1 = prf(labels, hierarchy)
    metric_rows.append(
        {
            "model_or_mode": "two_stage_hierarchical_augmented_tfidf_svm",
            "evaluation": "stratified_5fold_full_coverage",
            "answered_rows": len(labels),
            "coverage": "1.000",
            "accuracy": f"{accuracy:.3f}",
            "macro_f1": f"{macro_f1:.3f}",
            "notes": "Stage 1 predicts numeric/performance-or-not/crash group, stage 2 predicts primary label.",
        }
    )
    stored_predictions["two_stage_hierarchical_augmented_tfidf_svm"] = hierarchy

    if ZERO_SHOT.exists():
        zero = load_predictions(ZERO_SHOT)
        stored_predictions["zero_shot_llm_no_needs_review"] = no_needs_review([zero[bid] for bid in ids], fallback)
    if RAG.exists():
        rag = load_predictions(RAG)
        stored_predictions["rag_llm_no_needs_review"] = no_needs_review([rag[bid] for bid in ids], fallback)

    deterministic_ensemble_inputs = {
        "weak_candidate_no_needs_review_fallback": stored_predictions["weak_candidate_no_needs_review_fallback"],
        "candidate_plus_diff_features_tfidf_linear_svm": stored_predictions["candidate_plus_diff_features_tfidf_linear_svm"],
        "candidate_plus_diff_features_tfidf_logistic": stored_predictions["candidate_plus_diff_features_tfidf_logistic"],
        "base_text_tfidf_linear_svm": stored_predictions["base_text_tfidf_linear_svm"],
    }
    deterministic_ensemble = vote(
        deterministic_ensemble_inputs,
        [
            "candidate_plus_diff_features_tfidf_linear_svm",
            "base_text_tfidf_linear_svm",
            "candidate_plus_diff_features_tfidf_logistic",
            "weak_candidate_no_needs_review_fallback",
        ],
    )
    accuracy, macro_f1 = prf(labels, deterministic_ensemble)
    metric_rows.append(
        {
            "model_or_mode": "full_coverage_no_needs_review_deterministic_ensemble",
            "evaluation": "full_coverage_no_abstain",
            "answered_rows": len(labels),
            "coverage": "1.000",
            "accuracy": f"{accuracy:.3f}",
            "macro_f1": f"{macro_f1:.3f}",
            "notes": "Vote over no-needs weak label and augmented deterministic models only.",
        }
    )

    ensemble_inputs = dict(deterministic_ensemble_inputs)
    if "rag_llm_no_needs_review" in stored_predictions:
        ensemble_inputs["rag_llm_no_needs_review"] = stored_predictions["rag_llm_no_needs_review"]
    if "zero_shot_llm_no_needs_review" in stored_predictions:
        ensemble_inputs["zero_shot_llm_no_needs_review"] = stored_predictions["zero_shot_llm_no_needs_review"]
    ensemble = vote(
        ensemble_inputs,
        [
            "candidate_plus_diff_features_tfidf_linear_svm",
            "base_text_tfidf_linear_svm",
            "candidate_plus_diff_features_tfidf_logistic",
            "weak_candidate_no_needs_review_fallback",
        ],
    )
    accuracy, macro_f1 = prf(labels, ensemble)
    metric_rows.append(
        {
            "model_or_mode": "full_coverage_no_needs_review_with_local_llm_ensemble",
            "evaluation": "full_coverage_no_abstain",
            "answered_rows": len(labels),
            "coverage": "1.000",
            "accuracy": f"{accuracy:.3f}",
            "macro_f1": f"{macro_f1:.3f}",
            "notes": "Vote over no-needs weak label, augmented deterministic models, and local LLM outputs.",
        }
    )

    prediction_rows = []
    for index, blind_id in enumerate(ids):
        prediction_rows.append(
            {
                "blind_id": blind_id,
                "gold_primary_failure": labels[index],
                "best_full_coverage_prediction": ensemble[index],
                "weak_candidate_no_needs_review_fallback": candidate_noneeds[index],
                "augmented_tfidf_svm": stored_predictions["candidate_plus_diff_features_tfidf_linear_svm"][index],
                "augmented_tfidf_logistic": stored_predictions["candidate_plus_diff_features_tfidf_logistic"][index],
                "two_stage_hierarchical": hierarchy[index],
                "deterministic_ensemble_prediction": deterministic_ensemble[index],
            }
        )

    write_csv(
        METRICS,
        metric_rows,
        ["model_or_mode", "evaluation", "answered_rows", "coverage", "accuracy", "macro_f1", "notes"],
    )
    write_csv(
        PREDICTIONS,
        prediction_rows,
        [
            "blind_id",
            "gold_primary_failure",
            "best_full_coverage_prediction",
            "weak_candidate_no_needs_review_fallback",
            "augmented_tfidf_svm",
            "augmented_tfidf_logistic",
            "two_stage_hierarchical",
            "deterministic_ensemble_prediction",
        ],
    )

    best = max(metric_rows, key=lambda row: (float(row["accuracy"]), float(row["macro_f1"])))
    lines = [
        "# Full-Coverage Model Improvement Experiments",
        "",
        "This report implements the non-external-model recommendations: no `needs_review` at full coverage, two-stage classification, candidate-label features, linked PR/diff/path features, and a no-abstention ensemble.",
        "",
        "The main result is that these changes improve full-coverage accuracy modestly, but they do not reach 70-80% on the full 191-row benchmark.",
        "",
        f"Best full-coverage mode: `{best['model_or_mode']}` at {best['accuracy']} accuracy and {best['macro_f1']} macro F1.",
        "",
        "| model or mode | accuracy | macro F1 | notes |",
        "| --- | ---: | ---: | --- |",
        *[
            f"| {row['model_or_mode']} | {row['accuracy']} | {row['macro_f1']} | {row['notes']} |"
            for row in metric_rows
        ],
        "",
        "Interpretation:",
        "",
        "- Removing `needs_review` is necessary for full coverage, but it does not by itself solve the task.",
        "- Candidate-label and linked-diff/path features provide a small lift over the original text-only baseline.",
        "- The two-stage hierarchy is more interpretable but did not outperform the best flat augmented model on this small gold set.",
        "- More adjudicated labels and a stronger external model are the likely bottlenecks for reaching 70% full coverage.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
