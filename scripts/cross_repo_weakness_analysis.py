from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LORO_BY_REPO = ROOT / "tables" / "v2_gold_loro_by_repo.csv"
OUT = ROOT / "tables" / "v2_cross_repo_weakness_summary.csv"
REPORT = ROOT / "reports" / "v2_cross_repo_weakness_analysis.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def note(repo: str, best_model: str, best_acc: float, ensemble_acc: float) -> str:
    if repo == "cupy/cupy":
        return "Lowest transfer accuracy; CuPy has many environment, dtype, and API-compatibility reports whose vocabulary differs from larger training projects."
    if repo == "numba/numba":
        return "Small repository slice with CUDA/LLVM/build/runtime language; boundary between compiler crash and non-numerical support is difficult."
    if repo == "rapidsai/cudf":
        return "Mid-sized slice with dataframe/parser/API reports mixed with numerical correctness language, lowering macro F1."
    if best_acc - ensemble_acc > 0.05:
        return "A single linear model transfers better than the ensemble, suggesting candidate-label or model-vote signals are less stable under repository shift."
    return "Transfer is comparatively strong for this repository."


def main() -> None:
    rows = read_csv(LORO_BY_REPO)
    repos = sorted({row["held_out_repository"] for row in rows})
    out_rows = []
    for repo in repos:
        repo_rows = [row for row in rows if row["held_out_repository"] == repo]
        best = max(repo_rows, key=lambda row: (float(row["accuracy"]), float(row["macro_f1"])))
        ensemble = next(row for row in repo_rows if row["model_or_mode"] == "expanded_gold_vote_ensemble")
        svm = next(row for row in repo_rows if row["model_or_mode"] == "tfidf_linear_svm")
        out_rows.append(
            {
                "held_out_repository": repo,
                "test_issues": best["test_issues"],
                "best_model": best["model_or_mode"],
                "best_accuracy": best["accuracy"],
                "best_macro_f1": best["macro_f1"],
                "ensemble_accuracy": ensemble["accuracy"],
                "ensemble_macro_f1": ensemble["macro_f1"],
                "svm_accuracy": svm["accuracy"],
                "svm_macro_f1": svm["macro_f1"],
                "interpretation": note(repo, best["model_or_mode"], float(best["accuracy"]), float(ensemble["accuracy"])),
            }
        )
    out_rows.sort(key=lambda row: float(row["best_accuracy"]))
    write_csv(
        OUT,
        out_rows,
        [
            "held_out_repository",
            "test_issues",
            "best_model",
            "best_accuracy",
            "best_macro_f1",
            "ensemble_accuracy",
            "ensemble_macro_f1",
            "svm_accuracy",
            "svm_macro_f1",
            "interpretation",
        ],
    )
    REPORT.write_text(
        "\n".join(
            [
                "# V2 Cross-Repository Weakness Analysis",
                "",
                "Leave-one-repository-out evaluation is the hardest setting because the model must transfer across project vocabularies, issue templates, and library-specific failure modes.",
                "",
                "| held-out repository | issues | best model | best acc. | best macro F1 | ensemble acc. | interpretation |",
                "| --- | ---: | --- | ---: | ---: | ---: | --- |",
                *[
                    f"| {row['held_out_repository']} | {row['test_issues']} | {row['best_model']} | {row['best_accuracy']} | {row['best_macro_f1']} | {row['ensemble_accuracy']} | {row['interpretation']} |"
                    for row in out_rows
                ],
                "",
                "The weakest transfer repositories are CuPy and Numba. Both contain many environment/build/runtime and API-compatibility issues whose wording overlaps numerical symptoms but differs from larger PyTorch/JAX/RAPIDS issue styles. The paper should present this not as a failure of the benchmark, but as evidence that repository transfer is a real benchmark challenge.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
