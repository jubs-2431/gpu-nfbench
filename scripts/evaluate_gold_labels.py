from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "annotation"
REPORT = ROOT / "reports" / "gold_label_agreement.md"
AGREEMENT_TABLE = ROOT / "tables" / "gold_label_agreement.csv"
GOLD_OUT = ROOT / "data" / "processed" / "gold_benchmark.csv"


REQUIRED_ANNOTATOR_FIELDS = [
    "primary_failure_label",
    "secondary_cause_labels_pipe_separated",
    "is_true_numerical_failure",
    "evidence_quote",
    "confidence_high_medium_low",
]
REQUIRED_GOLD_FIELDS = [
    "gold_primary_failure",
    "gold_secondary_cause_labels_pipe_separated",
    "gold_is_true_numerical_failure",
    "gold_evidence_quote",
    "adjudicator_id",
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


def complete(row: dict[str, str], fields: list[str]) -> bool:
    return all(row.get(field, "").strip() for field in fields)


def cohen_kappa(a_labels: list[str], b_labels: list[str]) -> tuple[float, float]:
    n = len(a_labels)
    if n == 0:
        return 0.0, 0.0
    observed = sum(a == b for a, b in zip(a_labels, b_labels)) / n
    a_counts = Counter(a_labels)
    b_counts = Counter(b_labels)
    expected = sum((a_counts[label] / n) * (b_counts[label] / n) for label in set(a_counts) | set(b_counts))
    if expected == 1.0:
        return observed, 1.0
    return observed, (observed - expected) / (1 - expected)


def split_labels(value: str) -> set[str]:
    return {item.strip() for item in value.split("|") if item.strip()}


def mean_jaccard(a_values: list[str], b_values: list[str]) -> float:
    scores = []
    for a, b in zip(a_values, b_values):
        aset = split_labels(a)
        bset = split_labels(b)
        if not aset and not bset:
            scores.append(1.0)
        elif not aset or not bset:
            scores.append(0.0)
        else:
            scores.append(len(aset & bset) / len(aset | bset))
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    a_path = ANNOTATION / "annotator_A_blind.csv"
    b_path = ANNOTATION / "annotator_B_blind.csv"
    adjudication_path = ANNOTATION / "adjudication_template.csv"
    if not a_path.exists() or not b_path.exists() or not adjudication_path.exists():
        raise SystemExit("Annotation packet files are missing. Run scripts/create_annotation_packets.py first.")

    a_rows = {row["blind_id"]: row for row in read_csv(a_path)}
    b_rows = {row["blind_id"]: row for row in read_csv(b_path)}
    adjudication_rows = read_csv(adjudication_path)
    ids = sorted(set(a_rows) & set(b_rows))

    complete_ids = [
        bid
        for bid in ids
        if complete(a_rows[bid], REQUIRED_ANNOTATOR_FIELDS)
        and complete(b_rows[bid], REQUIRED_ANNOTATOR_FIELDS)
    ]
    a_primary = [a_rows[bid]["primary_failure_label"].strip() for bid in complete_ids]
    b_primary = [b_rows[bid]["primary_failure_label"].strip() for bid in complete_ids]
    primary_agreement, primary_kappa = cohen_kappa(a_primary, b_primary)
    cause_jaccard = mean_jaccard(
        [a_rows[bid]["secondary_cause_labels_pipe_separated"] for bid in complete_ids],
        [b_rows[bid]["secondary_cause_labels_pipe_separated"] for bid in complete_ids],
    )

    adjudicated_rows = [
        row
        for row in adjudication_rows
        if complete(row, REQUIRED_GOLD_FIELDS)
    ]
    if adjudicated_rows:
        gold_rows: list[dict[str, object]] = []
        for row in adjudicated_rows:
            bid = row["blind_id"]
            source = a_rows.get(bid) or b_rows.get(bid)
            if not source:
                continue
            gold_rows.append(
                {
                    "blind_id": bid,
                    "repository": source["repository"],
                    "issue_number": source["issue_number"],
                    "url": source["url"],
                    "title": source["title"],
                    "github_state": source["github_state"],
                    "github_labels": source["github_labels"],
                    "gold_primary_failure": row["gold_primary_failure"],
                    "gold_secondary_cause_labels": row["gold_secondary_cause_labels_pipe_separated"],
                    "gold_is_true_numerical_failure": row["gold_is_true_numerical_failure"],
                    "gold_evidence_quote": row["gold_evidence_quote"],
                    "adjudicator_id": row["adjudicator_id"],
                    "adjudication_notes": row["adjudication_notes"],
                }
            )
        write_csv(GOLD_OUT, gold_rows, list(gold_rows[0].keys()))

    agreement_rows = [
        {"metric": "candidate_gold_rows", "value": len(ids)},
        {"metric": "complete_double_annotated_rows", "value": len(complete_ids)},
        {"metric": "primary_label_observed_agreement", "value": f"{primary_agreement:.3f}"},
        {"metric": "primary_label_cohen_kappa", "value": f"{primary_kappa:.3f}"},
        {"metric": "secondary_cause_mean_jaccard", "value": f"{cause_jaccard:.3f}"},
        {"metric": "adjudicated_gold_rows", "value": len(adjudicated_rows)},
    ]
    write_csv(AGREEMENT_TABLE, agreement_rows, ["metric", "value"])

    status = "complete" if len(adjudicated_rows) == len(ids) and ids else "pending"
    lines = [
        "# Gold Label Agreement",
        "",
        f"Gold benchmark status: {status}.",
        "",
        "| metric | value |",
        "| --- | --- |",
        *[f"| {row['metric']} | {row['value']} |" for row in agreement_rows],
        "",
    ]
    if status != "complete":
        lines.extend(
            [
                "The gold benchmark is not complete yet. Fill both blind annotator files, adjudicate disagreements, then rerun this script.",
                "",
            ]
        )
    else:
        lines.extend([f"Gold benchmark written to `{GOLD_OUT.relative_to(ROOT)}`.", ""])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)
    if GOLD_OUT.exists():
        print(GOLD_OUT)


if __name__ == "__main__":
    main()
