from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import gold_baseline_classifier as gbc
import train_expanded_gold_models as egm


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "reports"
RELEASE_DIR = ROOT / "release" / "gpu-nfbench-artifact"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_dt(value: str) -> datetime:
    if not value:
        return datetime.min
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def enriched_rows() -> list[dict[str, str]]:
    rows = egm.build_rows()
    seed_by_url = {row["url"]: row for row in read_csv(ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv")}
    expansion_by_id = {row["expansion_id"]: row for row in read_csv(ROOT / "annotation" / "gold_expansion_1000_repaired.csv")}
    enriched = []
    for row in rows:
        created_at = ""
        updated_at = ""
        if row["blind_id"].startswith("EGNF"):
            source = expansion_by_id.get(row["blind_id"], {})
            created_at = source.get("created_at", "")
            updated_at = source.get("updated_at", "")
        else:
            source = seed_by_url.get(row["url"], {})
            created_at = source.get("created_at", "")
            updated_at = source.get("updated_at", "")
        enriched.append({**row, "created_at": created_at, "updated_at": updated_at})
    return [row for row in enriched if row["created_at"]]


def vote(prediction_sets: dict[str, list[str]], tie_order: list[str]) -> list[str]:
    output = []
    for idx in range(len(next(iter(prediction_sets.values())))):
        counts = Counter(preds[idx] for preds in prediction_sets.values())
        top = counts.most_common(1)[0][1]
        winners = {label for label, count in counts.items() if count == top}
        output.append(next((prediction_sets[name][idx] for name in tie_order if prediction_sets[name][idx] in winners), counts.most_common(1)[0][0]))
    return output


def metric(labels: list[str], preds: list[str]) -> tuple[float, float]:
    _, acc, macro_f1 = gbc.prf(labels, preds)
    return acc, macro_f1


def run_time_split(rows: list[dict[str, str]]) -> None:
    ordered = sorted(rows, key=lambda row: (parse_dt(row["created_at"]), row["blind_id"]))
    split_idx = int(len(ordered) * 0.8)
    train = ordered[:split_idx]
    test = ordered[split_idx:]
    labels = sorted({row["gold_primary_failure"] for row in ordered})

    model_names = [
        "candidate_weak_label",
        "bm25_knn",
        "naive_bayes",
        "tfidf_logistic",
        "tfidf_linear_svm",
        "bigram_tfidf_logistic",
    ]
    predictions: dict[str, list[str]] = {
        "candidate_weak_label": [row["candidate_primary_failure"] for row in test]
    }
    for model_name in model_names[1:]:
        predictions[model_name] = gbc.predict_model(model_name, train, test, labels)

    predictions["expanded_gold_vote_ensemble"] = vote(
        {
            "candidate_weak_label": predictions["candidate_weak_label"],
            "tfidf_linear_svm": predictions["tfidf_linear_svm"],
            "tfidf_logistic": predictions["tfidf_logistic"],
            "bigram_tfidf_logistic": predictions["bigram_tfidf_logistic"],
            "naive_bayes": predictions["naive_bayes"],
        },
        ["tfidf_linear_svm", "tfidf_logistic", "candidate_weak_label", "bigram_tfidf_logistic", "naive_bayes"],
    )

    y = [row["gold_primary_failure"] for row in test]
    rows_out = []
    for model_name, preds in predictions.items():
        acc, macro_f1 = metric(y, preds)
        rows_out.append(
            {
                "model_or_mode": model_name,
                "evaluation": "chronological_80_20",
                "train_rows": len(train),
                "test_rows": len(test),
                "train_end_created_at": train[-1]["created_at"],
                "test_start_created_at": test[0]["created_at"],
                "accuracy": f"{acc:.3f}",
                "macro_f1": f"{macro_f1:.3f}",
            }
        )

    write_csv(
        TABLE_DIR / "expanded_gold_time_split_metrics.csv",
        rows_out,
        [
            "model_or_mode",
            "evaluation",
            "train_rows",
            "test_rows",
            "train_end_created_at",
            "test_start_created_at",
            "accuracy",
            "macro_f1",
        ],
    )

    test_distribution = Counter(row["gold_primary_failure"] for row in test)
    REPORT_DIR.joinpath("expanded_gold_time_split.md").write_text(
        "\n".join(
            [
                "# Expanded Gold Chronological Split",
                "",
                f"Rows with creation timestamps: {len(ordered)}",
                f"Train rows: {len(train)}",
                f"Test rows: {len(test)}",
                f"Train end: {train[-1]['created_at']}",
                f"Test start: {test[0]['created_at']}",
                "",
                "## Test label distribution",
                "",
                *[f"- {label}: {count}" for label, count in sorted(test_distribution.items())],
                "",
                "## Metrics",
                "",
                "| model/mode | accuracy | macro F1 |",
                "| --- | ---: | ---: |",
                *[f"| {row['model_or_mode']} | {row['accuracy']} | {row['macro_f1']} |" for row in rows_out],
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_qualitative_examples() -> None:
    examples = read_csv(TABLE_DIR / "gold_representative_examples.csv")
    error_pairs = read_csv(TABLE_DIR / "expanded_gold_top_error_pairs.csv")
    lines = [
        "# Qualitative Examples for Conference Version",
        "",
        "## Clear taxonomy examples",
        "",
    ]
    for row in examples:
        lines.append(f"- `{row['gold_label']}`: {row['repository']}#{row['issue_number']} ({row['title']}). Evidence: {row['evidence_quote'][:220]}")
    lines.extend(["", "## Boundary/error examples", ""])
    for row in error_pairs[:6]:
        lines.append(
            f"- Gold `{row['gold_primary_failure']}` predicted as `{row['predicted_primary_failure']}`: "
            f"{row['example_repository']}#{row['example_issue_number']} ({row['example_title']})."
        )
    REPORT_DIR.joinpath("qualitative_examples_for_paper.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_cause_summary() -> None:
    changed_files = read_csv(TABLE_DIR / "linked_pr_changed_files.csv")
    linked_subset = read_csv(TABLE_DIR / "linked_fix_evidence_subset.csv")
    by_ext = Counter()
    by_path_signal = Counter()
    for row in changed_files:
        path = row.get("changed_file", "") or row.get("filename", "")
        suffix = Path(path).suffix.lower() or "(none)"
        by_ext[suffix] += 1
        lower = path.lower()
        if "test" in lower:
            by_path_signal["test_files"] += 1
        if "cuda" in lower or "cudf" in lower or "gpu" in lower or "triton" in lower:
            by_path_signal["gpu_stack_paths"] += 1
        if "dtype" in lower or "type" in lower or "cast" in lower:
            by_path_signal["type_cast_paths"] += 1
        if "doc" in lower or lower.endswith(".md") or lower.endswith(".rst"):
            by_path_signal["docs"] += 1
    top_ext = by_ext.most_common(10)
    linked_by_label = Counter(row["gold_primary_failure"] for row in linked_subset)
    REPORT_DIR.joinpath("linked_fix_root_cause_summary.md").write_text(
        "\n".join(
            [
                "# Linked Fix Root-Cause Evidence Summary",
                "",
                "This summary uses already fetched public PR diffs as evidence for a future root-cause extension. It does not convert report labels into confirmed root-cause labels.",
                "",
                f"Rows with at least one linked-fix evidence signal: {len(linked_subset)}",
                f"Changed-file rows: {len(changed_files)}",
                "",
                "## Linked-fix evidence by benchmark label",
                "",
                *[f"- {label}: {count}" for label, count in sorted(linked_by_label.items())],
                "",
                "## Path-signal counts in fetched PR diffs",
                "",
                *[f"- {label}: {count}" for label, count in sorted(by_path_signal.items())],
                "",
                "## Top file extensions in linked PR diffs",
                "",
                *[f"- {ext}: {count}" for ext, count in top_ext],
                "",
            ]
        ),
        encoding="utf-8",
    )


def release_bundle() -> None:
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True)
    for rel in [
        "ARTIFACT_DATA_CARD.md",
        "REPRODUCIBILITY.md",
        "CITATION.cff",
        ".zenodo.json",
        "RELEASE_NOTES_v1.0-conference.md",
        "GITHUB_ZENODO_RELEASE_COMMANDS.md",
        "data/processed/gold_benchmark_expanded.csv",
        "data/processed/gold_benchmark_expanded_adjudicated_v2.csv",
        "data/processed/gold_benchmark_expanded_v2_canonical.csv",
        "data/processed/gold_benchmark.csv",
        "data/processed/gpu_numerical_issue_seed.csv",
        "annotation/ANNOTATION_GUIDE.md",
        "annotation/EXPANDED_GOLD_AUDIT_120_INSTRUCTIONS.md",
        "annotation/expanded_gold_agreement_audit_120_blind.csv",
        "annotation/expanded_gold_agreement_audit_120_personA_personB_filled.csv",
        "annotation/expanded_gold_audit_119_adjudicated.csv",
        "annotation/gpu_nfbench_v2_adjudication_300_completed.csv",
        "annotation/gpu_nfbench_root_cause_50_adjudication_completed.csv",
        "annotation/gold_expansion_1000_taxonomy_repair_changes.csv",
        "tables/expanded_gold_classifier_metrics.csv",
        "tables/expanded_gold_audit_agreement.csv",
        "tables/expanded_gold_adjudicated_audit_metrics.csv",
        "tables/expanded_gold_audit_gold_revisions.csv",
        "tables/expanded_gold_audit_person_a_b_disagreements.csv",
        "tables/expanded_gold_audit_vs_gold_disagreements.csv",
        "tables/expanded_gold_loro_metrics.csv",
        "tables/expanded_gold_time_split_metrics.csv",
        "tables/expanded_gold_ablation_metrics.csv",
        "tables/expanded_gold_abstention_metrics.csv",
        "tables/expanded_gold_top_error_pairs.csv",
        "tables/v2_gold_label_revisions.csv",
        "tables/v2_gold_classifier_metrics.csv",
        "tables/v2_gold_classifier_per_class.csv",
        "tables/v2_gold_abstention_metrics.csv",
        "tables/v2_gold_loro_metrics.csv",
        "tables/v2_gold_time_split_metrics.csv",
        "tables/v2_standalone_seq2seq_llm_metrics.csv",
        "tables/v2_llm_assisted_metrics.csv",
        "tables/v2_local_llama32_3b_metrics.csv",
        "tables/v2_modern_api_baseline_metrics.csv",
        "tables/v2_cross_repo_weakness_summary.csv",
        "tables/v2_error_case_appendix.csv",
        "tables/root_cause_250_evidence_coded.csv",
        "tables/root_cause_250_label_counts.csv",
        "tables/root_cause_250_provenance_counts.csv",
        "tables/root_cause_50_label_counts.csv",
        "tables/root_cause_50_fix_file_category_counts.csv",
        "data/processed/external_repo_candidate_issue_pool.csv",
        "data/raw_online/external_repo_candidate_issues.jsonl",
        "evaluation/v2_standalone_seq2seq_llm_predictions.csv",
        "evaluation/v2_llm_assisted_predictions.csv",
        "evaluation/v2_heldout_llm_baseline_prompts.jsonl",
        "evaluation/v2_local_llama32_3b_predictions.csv",
        "evaluation/v2_local_llama32_3b_comparison.csv",
        "evaluation/v2_modern_api_baseline_on_llm_test.csv",
        "llm/finetune/gpu_nfbench_v2_standalone_train.jsonl",
        "llm/finetune/gpu_nfbench_v2_standalone_val.jsonl",
        "llm/finetune/gpu_nfbench_v2_standalone_test.jsonl",
        "llm/finetune/openai_chat_finetune_v2_train.jsonl",
        "llm/finetune/openai_chat_finetune_v2_val.jsonl",
        "llm/finetune/label_map_v2.json",
        "reports/expanded_gold_model_training.md",
        "reports/expanded_gold_audit_agreement.md",
        "reports/expanded_gold_adjudicated_audit.md",
        "reports/expanded_gold_conference_strengthening.md",
        "reports/expanded_gold_time_split.md",
        "reports/v2_canonical_and_root_cause_integration.md",
        "reports/v2_gold_model_training.md",
        "reports/v2_gold_generalization.md",
        "reports/v2_llm_finetune_files.md",
        "reports/v2_standalone_seq2seq_llm_training.md",
        "reports/v2_llm_assisted_training.md",
        "reports/v2_local_llama32_3b_baseline.md",
        "reports/v2_modern_api_baseline_comparison.md",
        "reports/v2_cross_repo_weakness_analysis.md",
        "reports/v2_error_case_appendix.md",
        "reports/root_cause_250_evidence_coded.md",
        "reports/external_repo_candidate_collection.md",
        "reports/final_submission_readiness.md",
        "reports/qualitative_examples_for_paper.md",
        "reports/linked_fix_root_cause_summary.md",
        "reports/standalone_llm_training_summary.md",
        "paper/GPU-NFBench_IEEE_Manuscript.pdf",
        "paper/gpu_numerical_failure_taxonomy_ieee.tex",
    ]:
        src = ROOT / rel
        dst = RELEASE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    manifest = {
        "artifact": "GPU-NFBench",
        "created_from": str(ROOT),
        "rows_expanded_gold": 1191,
        "primary_task": "GPU numerical-failure issue triage",
        "included": sorted(str(path.relative_to(RELEASE_DIR)) for path in RELEASE_DIR.rglob("*") if path.is_file()),
    }
    (RELEASE_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (RELEASE_DIR / "README.md").write_text(
        "\n".join(
            [
                "# GPU-NFBench Artifact Bundle",
                "",
                "This bundle contains the conference paper, benchmark data, evaluation tables, annotation audit packet, and data card for GPU-NFBench.",
                "It also includes a reproducibility guide, a 250-row evidence-coded root-cause extension, and an external-repository candidate pool for future expansion.",
                "",
                "Recommended archival release steps:",
                "",
                "1. Create a public GitHub repository named `gpu-nfbench`.",
                "2. Upload this bundle without private keys, local caches, or virtual environments.",
                "3. Add a release tag such as `v1.0-conference`.",
                "4. Archive the tagged release on Zenodo and include the DOI in the paper.",
                "5. Review audit-vs-gold disagreements before using audit labels to revise the benchmark.",
                "",
                "Primary paper file: `paper/GPU-NFBench_IEEE_Manuscript.pdf`",
                "Primary benchmark file: `data/processed/gold_benchmark_expanded_v2_canonical.csv`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    zip_path = ROOT / "release" / "gpu-nfbench-artifact.zip"
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zipf:
        for path in RELEASE_DIR.rglob("*"):
            if path.is_file():
                zipf.write(path, path.relative_to(RELEASE_DIR.parent))


def main() -> None:
    rows = enriched_rows()
    run_time_split(rows)
    write_qualitative_examples()
    write_root_cause_summary()
    release_bundle()
    print(REPORT_DIR / "expanded_gold_time_split.md")
    print(ROOT / "release" / "gpu-nfbench-artifact.zip")


if __name__ == "__main__":
    main()
