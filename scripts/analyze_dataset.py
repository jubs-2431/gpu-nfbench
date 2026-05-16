from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT = ROOT / "reports" / "analysis_summary.md"


def read_rows() -> list[dict[str, str]]:
    with DATA.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def year_of(value: str) -> str:
    try:
        return str(datetime.fromisoformat(value.replace("Z", "+00:00")).year)
    except Exception:
        return "unknown"


def md_table(rows: list[dict[str, object]], headers: list[str]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return out


def main() -> None:
    rows = read_rows()
    repo_counts = Counter(r["repository"] for r in rows)
    failure_counts = Counter(r["candidate_primary_failure"] for r in rows)
    cause_counts = Counter()
    by_repo_failure: dict[str, Counter[str]] = defaultdict(Counter)
    by_year = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        repo = row["repository"]
        failure = row["candidate_primary_failure"]
        by_repo_failure[repo][failure] += 1
        by_year[year_of(row["created_at"])] += 1
        for cause in row["candidate_cause_labels"].split("|"):
            if cause and cause != "needs_review":
                cause_counts[cause] += 1
        if failure != "needs_review" and len(examples[failure]) < 3:
            examples[failure].append(row)

    repo_rows = [
        {"repository": repo, "issues": count}
        for repo, count in repo_counts.most_common()
    ]
    failure_rows = [
        {"failure_label": label, "issues": count, "share_pct": round(100 * count / len(rows), 1)}
        for label, count in failure_counts.most_common()
    ]
    cause_rows = [
        {"cause_label": label, "issues": count, "share_pct": round(100 * count / len(rows), 1)}
        for label, count in cause_counts.most_common()
    ]
    year_rows = [
        {"year": year, "issues": count}
        for year, count in sorted(by_year.items())
    ]
    matrix_rows: list[dict[str, object]] = []
    labels = [row["failure_label"] for row in failure_rows]
    for repo in sorted(by_repo_failure):
        total = sum(by_repo_failure[repo].values())
        item: dict[str, object] = {"repository": repo, "total": total}
        for label in labels:
            item[label] = by_repo_failure[repo].get(label, 0)
        matrix_rows.append(item)

    write_csv(TABLE_DIR / "repo_counts.csv", repo_rows, ["repository", "issues"])
    write_csv(TABLE_DIR / "failure_counts.csv", failure_rows, ["failure_label", "issues", "share_pct"])
    write_csv(TABLE_DIR / "cause_counts.csv", cause_rows, ["cause_label", "issues", "share_pct"])
    write_csv(TABLE_DIR / "year_counts.csv", year_rows, ["year", "issues"])
    write_csv(TABLE_DIR / "repo_by_failure_matrix.csv", matrix_rows, ["repository", "total", *labels])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "failure_counts.dat").write_text(
        "\n".join(f"{row['failure_label']} {row['issues']}" for row in failure_rows) + "\n",
        encoding="utf-8",
    )
    (FIG_DIR / "cause_counts.dat").write_text(
        "\n".join(f"{row['cause_label']} {row['issues']}" for row in cause_rows) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Analysis Summary",
        "",
        f"Dataset size: {len(rows)} unique public GitHub issues.",
        "",
        "## Repositories",
        *md_table(repo_rows, ["repository", "issues"]),
        "",
        "## Primary failure labels",
        *md_table(failure_rows, ["failure_label", "issues", "share_pct"]),
        "",
        "## Secondary suspected-cause labels",
        *md_table(cause_rows, ["cause_label", "issues", "share_pct"]),
        "",
        "## Representative issue examples",
        "",
    ]
    for label, items in examples.items():
        lines.append(f"### {label}")
        for item in items:
            lines.append(f"- {item['repository']}: [{item['title']}]({item['url']})")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

