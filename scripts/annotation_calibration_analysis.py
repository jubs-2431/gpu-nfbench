from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
A_FILE = ROOT / "annotation" / "annotator_A_blind.csv"
B_FILE = ROOT / "annotation" / "annotator_B_blind.csv"
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
TABLE_DIR = ROOT / "tables"
CALIBRATION_BLIND = ROOT / "annotation" / "calibration_round2_blind.csv"
CALIBRATION_REVIEW = ROOT / "annotation" / "calibration_round2_review.csv"
REPORT = ROOT / "reports" / "annotation_calibration_analysis.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def kappa(pairs: list[tuple[str, str]]) -> tuple[int, float, float]:
    if not pairs:
        return 0, 0.0, 0.0
    labels = sorted(set(a for a, _ in pairs) | set(b for _, b in pairs))
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    expected = sum(a_counts[label] * b_counts[label] for label in labels) / (n * n)
    return n, observed, (observed - expected) / (1 - expected) if expected != 1 else 0.0


def clean(value: str, limit: int = 240) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def main() -> None:
    a_rows = read_csv(A_FILE)
    b_rows = {row["blind_id"]: row for row in read_csv(B_FILE)}
    gold_rows = {row["blind_id"]: row for row in read_csv(GOLD)}

    subset_defs = [
        ("all_rows", lambda a, b: True),
        ("both_high_confidence", lambda a, b: a["confidence_high_medium_low"] == "high" and b["confidence_high_medium_low"] == "high"),
        ("both_not_low_confidence", lambda a, b: a["confidence_high_medium_low"] != "low" and b["confidence_high_medium_low"] != "low"),
        ("true_failure_status_agrees", lambda a, b: a["is_true_numerical_failure"] == b["is_true_numerical_failure"]),
        ("primary_label_agrees", lambda a, b: a["primary_failure_label"] == b["primary_failure_label"]),
    ]
    subset_rows = []
    for name, filt in subset_defs:
        pairs = [
            (a["primary_failure_label"], b_rows[a["blind_id"]]["primary_failure_label"])
            for a in a_rows
            if filt(a, b_rows[a["blind_id"]])
        ]
        n, observed, score = kappa(pairs)
        subset_rows.append(
            {
                "subset": name,
                "rows": n,
                "observed_agreement": f"{observed:.3f}",
                "cohen_kappa": f"{score:.3f}",
            }
        )

    disagreements = []
    review_rows = []
    for a in a_rows:
        b = b_rows[a["blind_id"]]
        gold = gold_rows[a["blind_id"]]
        hard = (
            a["primary_failure_label"] != b["primary_failure_label"]
            or a["confidence_high_medium_low"] == "low"
            or b["confidence_high_medium_low"] == "low"
            or a["is_true_numerical_failure"] != b["is_true_numerical_failure"]
        )
        if not hard:
            continue
        disagreements.append(
            {
                "blind_id": a["blind_id"],
                "repository": a["repository"],
                "issue_number": a["issue_number"],
                "url": a["url"],
                "title": a["title"],
                "github_labels": a["github_labels"],
                "issue_body_excerpt": clean(a["issue_body_excerpt"], 1800),
                "comments_excerpt": clean(a["comments_excerpt"], 1800),
                "round2_primary_failure_label": "",
                "round2_secondary_cause_labels_pipe_separated": "",
                "round2_is_true_numerical_failure": "",
                "round2_evidence_quote": "",
                "round2_confidence_high_medium_low": "",
                "round2_notes": "",
            }
        )
        review_rows.append(
            {
                "blind_id": a["blind_id"],
                "repository": a["repository"],
                "issue_number": a["issue_number"],
                "title": a["title"],
                "annotator_a_primary": a["primary_failure_label"],
                "annotator_b_primary": b["primary_failure_label"],
                "annotator_a_true_failure": a["is_true_numerical_failure"],
                "annotator_b_true_failure": b["is_true_numerical_failure"],
                "gold_primary_failure": gold["gold_primary_failure"],
                "gold_is_true_numerical_failure": gold["gold_is_true_numerical_failure"],
                "gold_evidence_quote": clean(gold["gold_evidence_quote"], 240),
                "calibration_focus": (
                    "primary_label_disagreement"
                    if a["primary_failure_label"] != b["primary_failure_label"]
                    else "true_failure_or_confidence_disagreement"
                ),
            }
        )

    write_csv(TABLE_DIR / "annotation_agreement_subsets.csv", subset_rows, ["subset", "rows", "observed_agreement", "cohen_kappa"])
    write_csv(
        CALIBRATION_BLIND,
        disagreements,
        [
            "blind_id",
            "repository",
            "issue_number",
            "url",
            "title",
            "github_labels",
            "issue_body_excerpt",
            "comments_excerpt",
            "round2_primary_failure_label",
            "round2_secondary_cause_labels_pipe_separated",
            "round2_is_true_numerical_failure",
            "round2_evidence_quote",
            "round2_confidence_high_medium_low",
            "round2_notes",
        ],
    )
    write_csv(
        CALIBRATION_REVIEW,
        review_rows,
        [
            "blind_id",
            "repository",
            "issue_number",
            "title",
            "annotator_a_primary",
            "annotator_b_primary",
            "annotator_a_true_failure",
            "annotator_b_true_failure",
            "gold_primary_failure",
            "gold_is_true_numerical_failure",
            "gold_evidence_quote",
            "calibration_focus",
        ],
    )

    lines = [
        "# Annotation Calibration Analysis",
        "",
        "The original blind agreement is reported without modification. This analysis adds defensible ways to interpret and improve the low-kappa issue without rewriting completed human labels.",
        "",
        "## Agreement subsets",
        "",
        "| subset | rows | observed_agreement | cohen_kappa |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {row['subset']} | {row['rows']} | {row['observed_agreement']} | {row['cohen_kappa']} |"
            for row in subset_rows
        ],
        "",
        "## Round-2 calibration packet",
        "",
        f"- Blind relabeling packet: `{CALIBRATION_BLIND.relative_to(ROOT)}`",
        f"- Adjudicator/training review packet: `{CALIBRATION_REVIEW.relative_to(ROOT)}`",
        f"- Rows selected for round 2: {len(disagreements)}",
        "",
        "Recommended use: annotators first review a small training subset with adjudicated explanations, then blindly relabel the remaining calibration packet. A new kappa should be reported as post-calibration agreement, while the original kappa remains the primary unbiased blind-agreement measurement.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
