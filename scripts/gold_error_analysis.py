from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
A_FILE = ROOT / "annotation" / "annotator_A_blind.csv"
B_FILE = ROOT / "annotation" / "annotator_B_blind.csv"
SUGGESTIONS = ROOT / "annotation" / "candidate_label_suggestions_hidden_from_annotators.csv"
TABLE_DIR = ROOT / "tables"
REPORT = ROOT / "reports" / "gold_error_analysis.md"


BOUNDARY_RULES = {
    ("dtype_casting", "precision_tolerance"): "Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch.",
    ("dtype_casting", "overflow_underflow"): "Prefer overflow_underflow when the observed failure is range blow-up, saturation, integer wraparound, or underflow; retain dtype_casting as a cause when narrowing/promotion explains it.",
    ("nan_inf", "precision_tolerance"): "Prefer nan_inf when non-finite values are the observed symptom; prefer precision_tolerance when NaN/Inf appears only in tests, masks, or tolerance text.",
    ("crash_compile", "dtype_casting"): "Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure.",
    ("performance_only", "not_numerical_failure"): "Use performance_only when the issue is primarily speed/throughput but still discusses numerical kernels; use not_numerical_failure for search false positives with no correctness/performance numerical task.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean(value: str, limit: int = 180) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def label_note(gold: str, weak: str, a_label: str, b_label: str) -> str:
    labels = {gold, weak, a_label, b_label}
    for pair, note in BOUNDARY_RULES.items():
        if pair[0] in labels and pair[1] in labels:
            return note
    if weak != gold:
        return "Weak search/title labels drifted after full issue/comment context and adjudication."
    if a_label != b_label:
        return "Blind annotators disagreed despite the weak label matching gold; adjudication resolved the boundary case."
    return "Representative clear example for this gold class."


def main() -> None:
    gold_rows = read_csv(GOLD)
    a_rows = {row["blind_id"]: row for row in read_csv(A_FILE)}
    b_rows = {row["blind_id"]: row for row in read_csv(B_FILE)}
    suggestions = {row["blind_id"]: row for row in read_csv(SUGGESTIONS)}

    mismatch_counts: Counter[tuple[str, str]] = Counter()
    disagreement_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    label_seen: set[str] = set()

    for row in gold_rows:
        bid = row["blind_id"]
        weak = suggestions.get(bid, {}).get("candidate_primary_failure", "")
        a_label = a_rows.get(bid, {}).get("primary_failure_label", "")
        b_label = b_rows.get(bid, {}).get("primary_failure_label", "")
        gold = row["gold_primary_failure"]
        if weak and weak != gold:
            mismatch_counts[(weak, gold)] += 1
        if a_label != b_label or weak != gold:
            disagreement_rows.append(
                {
                    "blind_id": bid,
                    "repository": row["repository"],
                    "issue_number": row["issue_number"],
                    "title": clean(row["title"], 140),
                    "weak_label": weak,
                    "annotator_a": a_label,
                    "annotator_b": b_label,
                    "gold_label": gold,
                    "gold_evidence_quote": clean(row["gold_evidence_quote"], 180),
                    "why_difficult": label_note(gold, weak, a_label, b_label),
                }
            )
        if gold not in label_seen:
            representative_rows.append(
                {
                    "gold_label": gold,
                    "blind_id": bid,
                    "repository": row["repository"],
                    "issue_number": row["issue_number"],
                    "title": clean(row["title"], 140),
                    "weak_label": weak,
                    "annotator_a": a_label,
                    "annotator_b": b_label,
                    "evidence_quote": clean(row["gold_evidence_quote"], 180),
                    "why_in_benchmark": label_note(gold, weak, a_label, b_label),
                }
            )
            label_seen.add(gold)

    mismatch_rows = [
        {"weak_candidate_label": weak, "gold_label": gold, "issues": count}
        for (weak, gold), count in mismatch_counts.most_common()
    ]
    top_disagreements = disagreement_rows[:25]

    write_csv(TABLE_DIR / "gold_weak_mismatch_pairs.csv", mismatch_rows, ["weak_candidate_label", "gold_label", "issues"])
    write_csv(
        TABLE_DIR / "gold_annotation_disagreements.csv",
        disagreement_rows,
        [
            "blind_id",
            "repository",
            "issue_number",
            "title",
            "weak_label",
            "annotator_a",
            "annotator_b",
            "gold_label",
            "gold_evidence_quote",
            "why_difficult",
        ],
    )
    write_csv(
        TABLE_DIR / "gold_representative_examples.csv",
        representative_rows,
        [
            "gold_label",
            "blind_id",
            "repository",
            "issue_number",
            "title",
            "weak_label",
            "annotator_a",
            "annotator_b",
            "evidence_quote",
            "why_in_benchmark",
        ],
    )

    lines = [
        "# Gold Error Analysis",
        "",
        "This report summarizes where weak labels and blind human annotations diverge from the adjudicated gold labels.",
        "",
        "## Most common weak-label to gold-label corrections",
        "",
        "| weak_candidate_label | gold_label | issues |",
        "| --- | --- | ---: |",
        *[f"| {r['weak_candidate_label']} | {r['gold_label']} | {r['issues']} |" for r in mismatch_rows[:12]],
        "",
        "## Representative gold examples",
        "",
        "| gold_label | blind_id | repository | title | evidence_quote | why_in_benchmark |",
        "| --- | --- | --- | --- | --- | --- |",
        *[
            f"| {r['gold_label']} | {r['blind_id']} | {r['repository']} | {r['title']} | {r['evidence_quote']} | {r['why_in_benchmark']} |"
            for r in representative_rows
        ],
        "",
        "## Refined adjudication rules",
        "",
        *[f"- **{a} vs. {b}:** {note}" for (a, b), note in BOUNDARY_RULES.items()],
        "",
        "## High-value disagreement examples",
        "",
        "| blind_id | weak_label | annotator_a | annotator_b | gold_label | why_difficult |",
        "| --- | --- | --- | --- | --- | --- |",
        *[
            f"| {r['blind_id']} | {r['weak_label']} | {r['annotator_a']} | {r['annotator_b']} | {r['gold_label']} | {r['why_difficult']} |"
            for r in top_disagreements
        ],
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
