from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path("/Users/aryanshah/Downloads")

EXPANDED = ROOT / "data" / "processed" / "gold_benchmark_expanded.csv"
V2 = ROOT / "data" / "processed" / "gold_benchmark_expanded_adjudicated_v2.csv"
PREDICTIONS = ROOT / "evaluation" / "expanded_gold_model_predictions.csv"
ADJUDICATED_AUDIT = ROOT / "annotation" / "expanded_gold_audit_119_adjudicated.csv"
LINKED_FIX = ROOT / "tables" / "linked_fix_evidence_subset.csv"


LABELS = [
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
]

ROOT_CAUSE_LABELS = [
    "dtype_or_casting_semantics",
    "precision_or_tolerance_logic",
    "overflow_or_range_handling",
    "nan_or_invalid_value_handling",
    "compiler_codegen_or_lowering",
    "memory_layout_or_bounds",
    "synchronization_or_ordering",
    "dependency_build_or_environment",
    "performance_or_scheduling",
    "docs_api_or_feature_request",
    "unclear_from_public_evidence",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_v2_adjudication_packet() -> None:
    expanded = {row["blind_id"]: row for row in read_csv(EXPANDED)}
    v2 = {row["blind_id"]: row for row in read_csv(V2)}
    predictions = read_csv(PREDICTIONS)
    already_adjudicated = {row["blind_id"] for row in read_csv(ADJUDICATED_AUDIT)}

    candidates = []
    for pred in predictions:
        blind_id = pred["blind_id"]
        if blind_id in already_adjudicated:
            continue
        row = expanded[blind_id]
        ensemble_wrong = pred["expanded_gold_vote_ensemble_prediction"] != pred["gold_primary_failure"]
        candidate_wrong = pred["candidate_weak_label_prediction"] != pred["gold_primary_failure"]
        vote_count = int(pred["ensemble_vote_count"])
        vote_margin = int(pred["ensemble_vote_margin"])
        score = 0
        reasons = []
        if ensemble_wrong:
            score += 5
            reasons.append("ensemble_disagrees_with_current_gold")
        if candidate_wrong:
            score += 2
            reasons.append("candidate_label_disagrees_with_current_gold")
        if vote_count <= 3:
            score += 2
            reasons.append("low_ensemble_vote_count")
        if vote_margin <= 1:
            score += 2
            reasons.append("low_ensemble_margin")
        if row["gold_primary_failure"] in {"overflow_underflow", "performance_only", "not_numerical_failure", "dtype_casting"}:
            score += 1
            reasons.append("review_priority_class")
        if not reasons:
            continue
        candidates.append((score, blind_id, pred, row, reasons))

    candidates.sort(key=lambda item: (-item[0], item[3]["repository"], item[1]))
    selected = []
    label_counts: Counter[str] = Counter()
    repo_counts: Counter[str] = Counter()
    for score, blind_id, pred, row, reasons in candidates:
        label = row["gold_primary_failure"]
        repo = row["repository"]
        if label_counts[label] >= 55:
            continue
        if repo_counts[repo] >= 80:
            continue
        selected.append((score, blind_id, pred, row, reasons))
        label_counts[label] += 1
        repo_counts[repo] += 1
        if len(selected) >= 300:
            break

    out_rows = []
    for index, (score, blind_id, pred, row, reasons) in enumerate(selected, start=1):
        out_rows.append(
            {
                "review_id": f"V2-ADJ-{index:03d}",
                "blind_id": blind_id,
                "repository": row["repository"],
                "issue_number": row["issue_number"],
                "url": row["url"],
                "title": row["title"],
                "github_labels": row["github_labels"],
                "current_gold_primary_failure": row["gold_primary_failure"],
                "candidate_weak_label_prediction": pred["candidate_weak_label_prediction"],
                "tfidf_linear_svm_prediction": pred["tfidf_linear_svm_prediction"],
                "expanded_gold_vote_ensemble_prediction": pred["expanded_gold_vote_ensemble_prediction"],
                "ensemble_vote_count": pred["ensemble_vote_count"],
                "ensemble_vote_margin": pred["ensemble_vote_margin"],
                "selection_reasons": "|".join(reasons),
                "adjudicated_primary_failure": "",
                "adjudicated_is_true_numerical_failure": "",
                "adjudicated_confidence": "",
                "adjudication_reason": "",
            }
        )
    write_csv(
        DOWNLOADS / "gpu_nfbench_v2_adjudication_300_todo.csv",
        out_rows,
        [
            "review_id",
            "blind_id",
            "repository",
            "issue_number",
            "url",
            "title",
            "github_labels",
            "current_gold_primary_failure",
            "candidate_weak_label_prediction",
            "tfidf_linear_svm_prediction",
            "expanded_gold_vote_ensemble_prediction",
            "ensemble_vote_count",
            "ensemble_vote_margin",
            "selection_reasons",
            "adjudicated_primary_failure",
            "adjudicated_is_true_numerical_failure",
            "adjudicated_confidence",
            "adjudication_reason",
        ],
    )


def make_root_cause_packet() -> None:
    rows = read_csv(LINKED_FIX)
    priority = {"linked_pr_and_local_patch": 0, "linked_pr": 1, "local_patch": 2, "text_signal": 3}
    rows.sort(key=lambda row: (priority.get(row["evidence_tier"], 9), -int(row.get("comment_count") or 0), row["repository"], row["issue_number"]))
    selected = rows[:50]
    out_rows = []
    for index, row in enumerate(selected, start=1):
        out_rows.append(
            {
                "root_cause_id": f"RC-ADJ-{index:03d}",
                "blind_id": row["blind_id"],
                "repository": row["repository"],
                "issue_number": row["issue_number"],
                "title": row["title"],
                "current_issue_label": row["gold_primary_failure"],
                "issue_url": f"https://github.com/{row['repository']}/issues/{row['issue_number']}",
                "explicit_pull_urls": row["explicit_pull_urls"],
                "same_repo_fix_ref_urls": row["same_repo_fix_ref_urls"],
                "evidence_tier": row["evidence_tier"],
                "fix_or_root_cause_snippet": row["fix_or_root_cause_snippet"],
                "local_diff_or_patch_snippet": row["local_diff_or_patch_snippet"],
                "adjudicated_root_cause_label": "",
                "fix_file_category": "",
                "fix_evidence_quote": "",
                "root_cause_confidence": "",
                "root_cause_notes": "",
            }
        )
    write_csv(
        DOWNLOADS / "gpu_nfbench_root_cause_50_adjudication_todo.csv",
        out_rows,
        [
            "root_cause_id",
            "blind_id",
            "repository",
            "issue_number",
            "title",
            "current_issue_label",
            "issue_url",
            "explicit_pull_urls",
            "same_repo_fix_ref_urls",
            "evidence_tier",
            "fix_or_root_cause_snippet",
            "local_diff_or_patch_snippet",
            "adjudicated_root_cause_label",
            "fix_file_category",
            "fix_evidence_quote",
            "root_cause_confidence",
            "root_cause_notes",
        ],
    )


def write_instructions() -> None:
    (DOWNLOADS / "GPU_NFBench_Next_Adjudication_Instructions.md").write_text(
        "\n".join(
            [
                "# GPU-NFBench Next Adjudication Instructions",
                "",
                "## File 1: gpu_nfbench_v2_adjudication_300_todo.csv",
                "",
                "Goal: review 300 high-impact benchmark rows selected because models disagreed, confidence was low, or the class is known to be ambiguous.",
                "",
                "For each row, open the `url`, inspect the issue, and fill:",
                "",
                "- `adjudicated_primary_failure`: one of nan_inf, overflow_underflow, precision_tolerance, dtype_casting, crash_compile, performance_only, not_numerical_failure.",
                "- `adjudicated_is_true_numerical_failure`: yes, no, or unclear.",
                "- `adjudicated_confidence`: high, medium, or low.",
                "- `adjudication_reason`: one sentence explaining the choice.",
                "",
                "Use the current gold label and model predictions as context, but do not copy them blindly. Prefer the dominant user-visible failure.",
                "",
                "## File 2: gpu_nfbench_root_cause_50_adjudication_todo.csv",
                "",
                "Goal: create a small root-cause/fix-evidence extension from 50 linked-fix cases.",
                "",
                "Allowed `adjudicated_root_cause_label` values:",
                "",
                *[f"- {label}" for label in ROOT_CAUSE_LABELS],
                "",
                "For `fix_file_category`, use a short phrase such as test, compiler, dtype, docs, build, cuda_kernel, python_api, scheduler, memory_layout, dependency, or unknown.",
                "",
                "For `fix_evidence_quote`, paste a short public phrase from the issue, PR, or diff that supports the root-cause label.",
                "",
                "Use `unclear_from_public_evidence` when the public PR/issue does not justify a confident root-cause claim.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_release_and_venue_docs() -> None:
    (DOWNLOADS / "GPU_NFBench_Public_Artifact_DOI_Checklist.md").write_text(
        "\n".join(
            [
                "# GPU-NFBench Public Artifact + DOI Checklist",
                "",
                "1. Create a public GitHub repo named `gpu-nfbench`.",
                "2. Upload `gpu-nfbench-artifact.zip` contents, not local virtualenvs or caches.",
                "3. Add README, artifact data card, paper PDF, benchmark CSVs, scripts, reports, and tables.",
                "4. Create a release tag: `v1.0-conference`.",
                "5. Connect the GitHub repo to Zenodo.",
                "6. Archive the tagged release on Zenodo and copy the DOI.",
                "7. Add the DOI/URL to the paper artifact section.",
                "8. If the 300-row adjudication changes labels, create `v2.0` after retraining.",
                "",
                "Current artifact zip to upload:",
                "`/Users/aryanshah/Downloads/gpu_numerical_failure_taxonomy/release/gpu-nfbench-artifact.zip`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (DOWNLOADS / "GPU_NFBench_Venue_Rewrite_Plan.md").write_text(
        "\n".join(
            [
                "# GPU-NFBench Venue-Specific Rewrite Plan",
                "",
                "## MSR",
                "Title angle: benchmark construction and mining public GPU issue reports.",
                "Emphasize repository mining, label provenance, adjudication, cross-repo/time splits, and artifact reproducibility.",
                "",
                "## ISSRE",
                "Title angle: reliability triage for GPU numerical failures.",
                "Emphasize risk: silent wrong outputs, NaN/Inf propagation, dtype/range failures, maintainers needing triage support.",
                "",
                "## ASE Tools/Data",
                "Title angle: reusable benchmark + triage artifact.",
                "Emphasize the released dataset, scripts, model baselines, audit packet, root-cause extension, and reproducible pipeline.",
                "",
                "## Recommendation",
                "After the 300-row adjudication and 50-row root-cause packet are complete, target ASE Tools/Data or MSR first. ISSRE becomes stronger if the paper adds more reliability-focused root-cause and severity analysis.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    make_v2_adjudication_packet()
    make_root_cause_packet()
    write_instructions()
    write_release_and_venue_docs()
    print(DOWNLOADS / "gpu_nfbench_v2_adjudication_300_todo.csv")
    print(DOWNLOADS / "gpu_nfbench_root_cause_50_adjudication_todo.csv")


if __name__ == "__main__":
    main()
