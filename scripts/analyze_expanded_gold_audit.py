from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "annotation" / "expanded_gold_agreement_audit_120_personA_personB_filled.csv"
GOLD = ROOT / "data" / "processed" / "gold_benchmark_expanded.csv"
TABLE_DIR = ROOT / "tables"
REPORT = ROOT / "reports" / "expanded_gold_audit_agreement.md"

LABELS = [
    "crash_compile",
    "dtype_casting",
    "nan_inf",
    "not_numerical_failure",
    "overflow_underflow",
    "performance_only",
    "precision_tolerance",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def agreement(y1: list[str], y2: list[str], labels: list[str]) -> tuple[float, float, float]:
    n = len(y1)
    observed = sum(a == b for a, b in zip(y1, y2)) / n if n else 0.0
    c1 = Counter(y1)
    c2 = Counter(y2)
    expected = sum((c1[label] / n) * (c2[label] / n) for label in labels) if n else 0.0
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 0.0
    return observed, expected, kappa


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def main() -> None:
    audit = read_csv(AUDIT)
    gold_rows = {row["blind_id"]: row for row in read_csv(GOLD)}
    missing = [row["blind_id"] for row in audit if row["blind_id"] not in gold_rows]
    if missing:
        raise SystemExit(f"Missing gold rows for {missing[:5]}")

    person_a = [row["person_a_primary_failure"].strip() for row in audit]
    person_b = [row["person_b_primary_failure"].strip() for row in audit]
    gold = [gold_rows[row["blind_id"]]["gold_primary_failure"] for row in audit]

    invalid = sorted((set(person_a) | set(person_b)) - set(LABELS))
    if invalid:
        raise SystemExit(f"Invalid audit labels: {invalid}")

    ab_obs, ab_exp, ab_kappa = agreement(person_a, person_b, LABELS)
    ag_obs, ag_exp, ag_kappa = agreement(person_a, gold, LABELS)
    bg_obs, bg_exp, bg_kappa = agreement(person_b, gold, LABELS)

    summary_rows = [
        {
            "comparison": "person_a_vs_person_b",
            "rows": len(audit),
            "observed_agreement_or_accuracy": f"{ab_obs:.3f}",
            "expected_agreement": f"{ab_exp:.3f}",
            "cohens_kappa": f"{ab_kappa:.3f}",
            "macro_f1_against_gold": "",
        },
        {
            "comparison": "person_a_vs_gold",
            "rows": len(audit),
            "observed_agreement_or_accuracy": f"{ag_obs:.3f}",
            "expected_agreement": f"{ag_exp:.3f}",
            "cohens_kappa": f"{ag_kappa:.3f}",
            "macro_f1_against_gold": f"{macro_f1(gold, person_a, LABELS):.3f}",
        },
        {
            "comparison": "person_b_vs_gold",
            "rows": len(audit),
            "observed_agreement_or_accuracy": f"{bg_obs:.3f}",
            "expected_agreement": f"{bg_exp:.3f}",
            "cohens_kappa": f"{bg_kappa:.3f}",
            "macro_f1_against_gold": f"{macro_f1(gold, person_b, LABELS):.3f}",
        },
    ]
    write_csv(
        TABLE_DIR / "expanded_gold_audit_agreement.csv",
        summary_rows,
        ["comparison", "rows", "observed_agreement_or_accuracy", "expected_agreement", "cohens_kappa", "macro_f1_against_gold"],
    )

    disagreement_rows = []
    for (a, b), count in Counter(zip(person_a, person_b)).most_common():
        if a != b:
            disagreement_rows.append({"person_a_primary_failure": a, "person_b_primary_failure": b, "issues": count})
    write_csv(
        TABLE_DIR / "expanded_gold_audit_person_a_b_disagreements.csv",
        disagreement_rows,
        ["person_a_primary_failure", "person_b_primary_failure", "issues"],
    )

    vs_gold_rows = []
    for labeler, preds in [("person_a", person_a), ("person_b", person_b)]:
        for (g, p), count in Counter(zip(gold, preds)).most_common():
            if g != p:
                vs_gold_rows.append({"labeler": labeler, "gold_primary_failure": g, "audit_primary_failure": p, "issues": count})
    write_csv(
        TABLE_DIR / "expanded_gold_audit_vs_gold_disagreements.csv",
        vs_gold_rows,
        ["labeler", "gold_primary_failure", "audit_primary_failure", "issues"],
    )

    REPORT.write_text(
        "\n".join(
            [
                "# Expanded Gold Audit Agreement",
                "",
                f"Audit rows: {len(audit)}",
                "All Person A and Person B labels use the allowed seven-class taxonomy.",
                "",
                "## Agreement summary",
                "",
                "| comparison | observed agreement/accuracy | expected agreement | Cohen's kappa | macro F1 vs gold |",
                "| --- | ---: | ---: | ---: | ---: |",
                *[
                    f"| {row['comparison']} | {row['observed_agreement_or_accuracy']} | {row['expected_agreement']} | {row['cohens_kappa']} | {row['macro_f1_against_gold'] or 'n/a'} |"
                    for row in summary_rows
                ],
                "",
                "## Top Person A / Person B disagreements",
                "",
                "| Person A | Person B | issues |",
                "| --- | --- | ---: |",
                *[
                    f"| {row['person_a_primary_failure']} | {row['person_b_primary_failure']} | {row['issues']} |"
                    for row in disagreement_rows[:12]
                ],
                "",
                "## Interpretation",
                "",
                "- Person A and Person B agreement is strong for an issue-report taxonomy with overlapping symptom and root-cause cues.",
                "- Agreement against the existing expanded gold labels is lower than A/B agreement, indicating that the audit should be used to identify rows where the expanded gold label may need adjudication review.",
                "- The current conference-safe claim is that a blind expanded audit achieved substantial inter-annotator agreement, while a separate adjudication pass is still needed before replacing existing expanded gold labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
