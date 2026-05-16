from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
CONTEXT = ROOT / "data" / "gold_context"
TABLE_DIR = ROOT / "tables"
REPORT = ROOT / "reports" / "root_cause_context_analysis.md"


FIX_RE = re.compile(r"\b(fix(?:es|ed)?|resolve(?:s|d)?|workaround|regression|root cause|closed by|caused by)\b", re.I)
PR_URL_RE = re.compile(r"https://github\.com/[^/\s]+/[^/\s]+/pull/\d+")
ISSUE_OR_PR_REF_RE = re.compile(r"(?<![A-Za-z0-9_])#(\d+)")
COMMIT_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.I)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def context_paths(repository: str, issue_number: str) -> tuple[Path, Path]:
    stem = repository.replace("/", "__") + "__" + issue_number
    return CONTEXT / f"{stem}.issue.json", CONTEXT / f"{stem}.comments.json"


def load_text(repository: str, issue_number: str) -> str:
    issue_path, comments_path = context_paths(repository, issue_number)
    parts: list[str] = []
    if issue_path.exists():
        issue = json.loads(issue_path.read_text(encoding="utf-8"))
        parts.append(str(issue.get("title", "")))
        parts.append(str(issue.get("body", "")))
    if comments_path.exists():
        comments = json.loads(comments_path.read_text(encoding="utf-8"))
        if isinstance(comments, list):
            parts.extend(str(comment.get("body", "")) for comment in comments if isinstance(comment, dict))
    return "\n".join(parts)


def snippet(text: str) -> str:
    match = FIX_RE.search(text)
    if not match:
        return ""
    start = max(0, match.start() - 90)
    end = min(len(text), match.end() + 180)
    return " ".join(text[start:end].split())


def main() -> None:
    rows = read_csv(GOLD)
    out_rows: list[dict[str, object]] = []
    with_fix = 0
    with_pr = 0
    with_commit = 0

    for row in rows:
        text = load_text(row["repository"], row["issue_number"])
        pr_urls = sorted(set(PR_URL_RE.findall(text)))
        refs = sorted(set(ISSUE_OR_PR_REF_RE.findall(text)))[:10]
        commits = sorted(set(COMMIT_RE.findall(text)))[:10]
        has_fix_signal = bool(FIX_RE.search(text))
        has_pr_signal = bool(pr_urls or refs)
        has_commit_signal = bool(commits)
        with_fix += int(has_fix_signal)
        with_pr += int(has_pr_signal)
        with_commit += int(has_commit_signal)
        out_rows.append(
            {
                "blind_id": row["blind_id"],
                "repository": row["repository"],
                "issue_number": row["issue_number"],
                "gold_primary_failure": row["gold_primary_failure"],
                "github_state": row["github_state"],
                "has_fix_workaround_regression_text": str(has_fix_signal).lower(),
                "has_pr_or_issue_reference": str(has_pr_signal).lower(),
                "has_commit_hash_reference": str(has_commit_signal).lower(),
                "linked_pull_urls": "|".join(pr_urls[:5]),
                "issue_or_pr_refs": "|".join(refs),
                "commit_refs": "|".join(commits),
                "fix_context_snippet": snippet(text),
            }
        )

    total = len(rows)
    write_csv(
        TABLE_DIR / "root_cause_context_signals.csv",
        out_rows,
        [
            "blind_id",
            "repository",
            "issue_number",
            "gold_primary_failure",
            "github_state",
            "has_fix_workaround_regression_text",
            "has_pr_or_issue_reference",
            "has_commit_hash_reference",
            "linked_pull_urls",
            "issue_or_pr_refs",
            "commit_refs",
            "fix_context_snippet",
        ],
    )

    lines = [
        "# Root-Cause Context Signal Analysis",
        "",
        "This report does not claim definitive root-cause recovery. It measures how much public issue/comment context contains fix, workaround, regression, pull-request, issue-reference, or commit-reference signals that could support a later root-cause benchmark extension.",
        "",
        "| signal | issues | share |",
        "| --- | ---: | ---: |",
        f"| fix/workaround/regression/root-cause text | {with_fix} | {100 * with_fix / total:.1f}% |",
        f"| pull-request or issue reference | {with_pr} | {100 * with_pr / total:.1f}% |",
        f"| commit-hash-like reference | {with_commit} | {100 * with_commit / total:.1f}% |",
        "",
        "The generated table `tables/root_cause_context_signals.csv` identifies candidate rows for future linked-PR/code-diff analysis.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
