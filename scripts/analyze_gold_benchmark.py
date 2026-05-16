from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
SUGGESTIONS = ROOT / "annotation" / "candidate_label_suggestions_hidden_from_annotators.csv"
A_FILE = ROOT / "annotation" / "annotator_A_blind.csv"
B_FILE = ROOT / "annotation" / "annotator_B_blind.csv"
TABLE_DIR = ROOT / "tables"
REPORT = ROOT / "reports" / "gold_benchmark_analysis.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "0.0%"


def md_table(rows: list[dict[str, object]], headers: list[str]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return out


def main() -> None:
    gold = read_csv(GOLD)
    suggestions = {row["blind_id"]: row for row in read_csv(SUGGESTIONS)}
    a_rows = {row["blind_id"]: row for row in read_csv(A_FILE)}
    b_rows = {row["blind_id"]: row for row in read_csv(B_FILE)}
    total = len(gold)

    label_counts = Counter(row["gold_primary_failure"] for row in gold)
    true_counts = Counter(row["gold_is_true_numerical_failure"] for row in gold)
    repo_counts = Counter(row["repository"] for row in gold)
    cause_counts = Counter()
    repo_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    silver_gold = Counter()
    a_gold = Counter()
    b_gold = Counter()
    a_b = Counter()

    for row in gold:
        bid = row["blind_id"]
        gold_label = row["gold_primary_failure"]
        repo_by_label[row["repository"]][gold_label] += 1
        for cause in row["gold_secondary_cause_labels"].split("|"):
            if cause:
                cause_counts[cause] += 1
        silver = suggestions.get(bid, {}).get("candidate_primary_failure", "")
        if silver:
            silver_gold[(silver, gold_label)] += 1
        if bid in a_rows and bid in b_rows:
            a_label = a_rows[bid]["primary_failure_label"]
            b_label = b_rows[bid]["primary_failure_label"]
            a_gold[(a_label, gold_label)] += 1
            b_gold[(b_label, gold_label)] += 1
            a_b[(a_label, b_label)] += 1

    label_rows = [
        {"gold_primary_failure": label, "issues": count, "share": pct(count, total)}
        for label, count in label_counts.most_common()
    ]
    true_rows = [
        {"gold_is_true_numerical_failure": label, "issues": count, "share": pct(count, total)}
        for label, count in true_counts.most_common()
    ]
    repo_rows = [
        {"repository": repo, "issues": count, "share": pct(count, total)}
        for repo, count in repo_counts.most_common()
    ]
    cause_rows = [
        {"gold_secondary_cause": cause, "issues": count, "share": pct(count, total)}
        for cause, count in cause_counts.most_common()
    ]
    labels = [row["gold_primary_failure"] for row in label_rows]
    matrix_rows: list[dict[str, object]] = []
    for repo in sorted(repo_by_label):
        item: dict[str, object] = {"repository": repo, "total": sum(repo_by_label[repo].values())}
        for label in labels:
            item[label] = repo_by_label[repo][label]
        matrix_rows.append(item)
    silver_gold_rows = [
        {"candidate_primary_failure": silver, "gold_primary_failure": gold_label, "issues": count}
        for (silver, gold_label), count in sorted(silver_gold.items())
    ]
    a_b_rows = [
        {"annotator_a_primary": a_label, "annotator_b_primary": b_label, "issues": count}
        for (a_label, b_label), count in sorted(a_b.items())
    ]

    candidate_match = sum(
        1
        for row in gold
        if suggestions.get(row["blind_id"], {}).get("candidate_primary_failure") == row["gold_primary_failure"]
    )
    a_gold_match = sum(
        1
        for row in gold
        if a_rows.get(row["blind_id"], {}).get("primary_failure_label") == row["gold_primary_failure"]
    )
    b_gold_match = sum(
        1
        for row in gold
        if b_rows.get(row["blind_id"], {}).get("primary_failure_label") == row["gold_primary_failure"]
    )

    write_csv(TABLE_DIR / "gold_primary_counts.csv", label_rows, ["gold_primary_failure", "issues", "share"])
    write_csv(TABLE_DIR / "gold_true_failure_counts.csv", true_rows, ["gold_is_true_numerical_failure", "issues", "share"])
    write_csv(TABLE_DIR / "gold_repo_counts.csv", repo_rows, ["repository", "issues", "share"])
    write_csv(TABLE_DIR / "gold_cause_counts.csv", cause_rows, ["gold_secondary_cause", "issues", "share"])
    write_csv(TABLE_DIR / "gold_repo_by_primary_matrix.csv", matrix_rows, ["repository", "total", *labels])
    write_csv(TABLE_DIR / "silver_vs_gold_confusion.csv", silver_gold_rows, ["candidate_primary_failure", "gold_primary_failure", "issues"])
    write_csv(TABLE_DIR / "annotator_a_vs_b_confusion.csv", a_b_rows, ["annotator_a_primary", "annotator_b_primary", "issues"])

    lines = [
        "# Gold Benchmark Analysis",
        "",
        f"Gold benchmark size: {total} adjudicated issues.",
        "",
        "## Gold primary-label distribution",
        *md_table(label_rows, ["gold_primary_failure", "issues", "share"]),
        "",
        "## True numerical-failure status",
        *md_table(true_rows, ["gold_is_true_numerical_failure", "issues", "share"]),
        "",
        "## Repository distribution",
        *md_table(repo_rows, ["repository", "issues", "share"]),
        "",
        "## Gold secondary-cause distribution",
        *md_table(cause_rows, ["gold_secondary_cause", "issues", "share"]),
        "",
        "## Drift from weak labels",
        "",
        f"- Candidate weak label matched adjudicated gold label for {candidate_match}/{total} issues ({pct(candidate_match, total)}).",
        f"- Annotator A matched adjudicated gold label for {a_gold_match}/{total} issues ({pct(a_gold_match, total)}).",
        f"- Annotator B matched adjudicated gold label for {b_gold_match}/{total} issues ({pct(b_gold_match, total)}).",
        "",
        "## Main benchmark caveats",
        "",
        "- The gold set is intentionally challenging and class-imbalanced after adjudication.",
        "- Crash/compile and performance-only are small classes in the final gold distribution.",
        "- Low annotator agreement should be reported as a finding: GPU numerical issue labeling is genuinely ambiguous without adjudication.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
