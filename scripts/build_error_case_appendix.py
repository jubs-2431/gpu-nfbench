from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "processed" / "gold_benchmark_expanded_v2_canonical.csv"
PREDS = ROOT / "evaluation" / "v2_gold_model_predictions.csv"
OUT = ROOT / "tables" / "v2_error_case_appendix.csv"
REPORT = ROOT / "reports" / "v2_error_case_appendix.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean(value: str, limit: int = 280) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def explanation(gold: str, pred: str) -> str:
    pair = (gold, pred)
    if pair == ("dtype_casting", "precision_tolerance"):
        return "Dtype/type changes are often reported through numerical mismatch tests, making symptom and cause easy to conflate."
    if pair == ("nan_inf", "precision_tolerance"):
        return "NaN/Inf symptoms appear inside broader correctness or tolerance failures."
    if pair == ("precision_tolerance", "dtype_casting"):
        return "Precision failures mention fp16/bf16/fp32 or type promotion, which pulls the classifier toward dtype semantics."
    if pair == ("crash_compile", "precision_tolerance"):
        return "Compiler/runtime errors can be triggered by correctness tests whose title emphasizes numerical mismatch."
    if pair[0] == "not_numerical_failure":
        return "Support/API/environment reports include technical stack traces or dtype words despite not being numerical defects."
    if pair[1] == "crash_compile":
        return "The issue contains runtime/build/stack-trace language that competes with the numerical symptom."
    return "The issue mixes multiple GPU failure cues, so the dominant primary label is ambiguous from report text alone."


def main() -> None:
    v2_by_id = {row["blind_id"]: row for row in read_csv(V2)}
    pred_rows = read_csv(PREDS)
    errors = [
        row for row in pred_rows if row["gold_primary_failure"] != row["expanded_gold_vote_ensemble_prediction"]
    ]
    pair_counts = Counter((row["gold_primary_failure"], row["expanded_gold_vote_ensemble_prediction"]) for row in errors)
    selected = []
    seen_pairs = Counter()
    for row in sorted(errors, key=lambda r: (-pair_counts[(r["gold_primary_failure"], r["expanded_gold_vote_ensemble_prediction"])], r["blind_id"])):
        pair = (row["gold_primary_failure"], row["expanded_gold_vote_ensemble_prediction"])
        if seen_pairs[pair] >= 2:
            continue
        src = v2_by_id.get(row["blind_id"], {})
        selected.append(
            {
                "blind_id": row["blind_id"],
                "repository": row["repository"],
                "issue_number": src.get("issue_number", ""),
                "title": clean(src.get("title", ""), 160),
                "gold_primary_failure": row["gold_primary_failure"],
                "ensemble_prediction": row["expanded_gold_vote_ensemble_prediction"],
                "snippet": clean(src.get("gold_evidence_quote", "") or src.get("title", ""), 300),
                "why_boundary_is_hard": explanation(*pair),
            }
        )
        seen_pairs[pair] += 1
        if len(selected) >= 12:
            break

    write_csv(
        OUT,
        selected,
        [
            "blind_id",
            "repository",
            "issue_number",
            "title",
            "gold_primary_failure",
            "ensemble_prediction",
            "snippet",
            "why_boundary_is_hard",
        ],
    )
    REPORT.write_text(
        "\n".join(
            [
                "# V2 Error Case Appendix",
                "",
                "These examples are real v2 deterministic-ensemble errors selected from the most common confusion families. They are intended for an appendix or artifact report to make the benchmark boundaries concrete.",
                "",
                "| id | repo | gold | predicted | issue/snippet | why hard |",
                "| --- | --- | --- | --- | --- | --- |",
                *[
                    f"| {row['blind_id']} | {row['repository']} | {row['gold_primary_failure']} | {row['ensemble_prediction']} | {row['title']}: {row['snippet']} | {row['why_boundary_is_hard']} |"
                    for row in selected
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
