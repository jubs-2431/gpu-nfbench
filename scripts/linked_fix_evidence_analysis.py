from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
CONTEXT = ROOT / "data" / "gold_context"
TABLE_DIR = ROOT / "tables"
REPORT = ROOT / "reports" / "linked_fix_evidence_analysis.md"
PR_DIFF_MANIFEST = TABLE_DIR / "linked_pr_diff_manifest.csv"
PR_CHANGED_FILES = TABLE_DIR / "linked_pr_changed_files.csv"


FIX_RE = re.compile(r"\b(fix(?:es|ed)?|resolve(?:s|d)?|workaround|regression|root cause|closed by|caused by|patch)\b", re.I)
PR_URL_RE = re.compile(r"https://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)")
DIFF_RE = re.compile(r"```diff(?P<body>.*?)```", re.S | re.I)
PATCH_RE = re.compile(r"(diff --git .*?)(?:\n```|\Z)", re.S)
REF_RE = re.compile(r"(?i)\b(?:PR|pull request|fix(?:es|ed)?|resolved? by|closed by|patch)\s*#(\d+)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def linked_pr_diff_summary() -> list[str]:
    if not PR_DIFF_MANIFEST.exists() or not PR_CHANGED_FILES.exists():
        return []
    manifest_rows = read_csv(PR_DIFF_MANIFEST)
    changed_file_rows = read_csv(PR_CHANGED_FILES)
    fetched = [row for row in manifest_rows if row.get("status") in {"cached", "fetched"}]
    non_empty = [row for row in fetched if int(row.get("bytes", "0") or 0) > 0]
    unique_changed_files = {row.get("changed_file", "") for row in changed_file_rows if row.get("changed_file")}
    return [
        f"A follow-on public diff fetch reached {len(fetched)}/{len(manifest_rows)} referenced PR diff URLs, cached {len(non_empty)} non-empty diffs, and extracted {len(changed_file_rows)} changed-file rows from {len(unique_changed_files)} unique changed files.",
        "",
        "Additional generated artifacts:",
        "",
        "- `reports/linked_pr_diff_fetch_report.md`",
        "- `tables/linked_pr_diff_manifest.csv`",
        "- `tables/linked_pr_changed_files.csv`",
        "- `data/linked_pr_diffs/*.diff`",
        "",
    ]


def context_paths(repository: str, issue_number: str) -> tuple[Path, Path]:
    stem = repository.replace("/", "__") + "__" + issue_number
    return CONTEXT / f"{stem}.issue.json", CONTEXT / f"{stem}.comments.json"


def load_context(repository: str, issue_number: str) -> tuple[str, int]:
    issue_path, comments_path = context_paths(repository, issue_number)
    parts: list[str] = []
    comment_count = 0
    if issue_path.exists():
        issue = json.loads(issue_path.read_text(encoding="utf-8"))
        parts.append(str(issue.get("title", "")))
        parts.append(str(issue.get("body", "")))
    if comments_path.exists():
        comments = json.loads(comments_path.read_text(encoding="utf-8"))
        if isinstance(comments, list):
            comment_count = len(comments)
            parts.extend(str(comment.get("body", "")) for comment in comments if isinstance(comment, dict))
    return "\n".join(parts), comment_count


def clean(value: str, limit: int = 260) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def evidence_snippet(text: str) -> str:
    match = FIX_RE.search(text)
    if not match:
        return ""
    start = max(0, match.start() - 120)
    end = min(len(text), match.end() + 260)
    return clean(text[start:end], 380)


def diff_snippet(text: str) -> str:
    match = DIFF_RE.search(text) or PATCH_RE.search(text)
    if not match:
        return ""
    body = match.groupdict().get("body") if match.groupdict() else match.group(1)
    return clean(body or match.group(0), 380)


def same_repo_ref_urls(repository: str, text: str) -> list[str]:
    refs = sorted(set(REF_RE.findall(text)))
    return [f"https://github.com/{repository}/pull/{ref}" for ref in refs[:8]]


def main() -> None:
    rows = read_csv(GOLD)
    out_rows: list[dict[str, object]] = []
    explicit_pr_rows = 0
    same_repo_ref_rows = 0
    diff_rows = 0
    fix_snippet_rows = 0

    for row in rows:
        text, comment_count = load_context(row["repository"], row["issue_number"])
        pr_urls = [f"https://github.com/{owner}/{repo}/pull/{num}" for owner, repo, num in sorted(set(PR_URL_RE.findall(text)))]
        inferred_refs = same_repo_ref_urls(row["repository"], text)
        local_diff = diff_snippet(text)
        snippet = evidence_snippet(text)
        if pr_urls:
            explicit_pr_rows += 1
        if inferred_refs:
            same_repo_ref_rows += 1
        if local_diff:
            diff_rows += 1
        if snippet:
            fix_snippet_rows += 1
        if pr_urls or inferred_refs or local_diff or snippet:
            out_rows.append(
                {
                    "blind_id": row["blind_id"],
                    "repository": row["repository"],
                    "issue_number": row["issue_number"],
                    "gold_primary_failure": row["gold_primary_failure"],
                    "title": row["title"],
                    "comment_count": comment_count,
                    "explicit_pull_urls": "|".join(pr_urls[:8]),
                    "same_repo_fix_ref_urls": "|".join(inferred_refs[:8]),
                    "has_local_diff_or_patch_snippet": str(bool(local_diff)).lower(),
                    "local_diff_or_patch_snippet": local_diff,
                    "fix_or_root_cause_snippet": snippet,
                    "evidence_tier": (
                        "linked_pr_and_local_patch"
                        if pr_urls and local_diff
                        else "linked_pr"
                        if pr_urls
                        else "local_patch"
                        if local_diff
                        else "fix_text_or_same_repo_ref"
                    ),
                }
            )

    priority = {
        "linked_pr_and_local_patch": 0,
        "linked_pr": 1,
        "local_patch": 2,
        "fix_text_or_same_repo_ref": 3,
    }
    out_rows.sort(key=lambda r: (priority[str(r["evidence_tier"])], r["repository"], r["blind_id"]))
    write_csv(
        TABLE_DIR / "linked_fix_evidence_subset.csv",
        out_rows,
        [
            "blind_id",
            "repository",
            "issue_number",
            "gold_primary_failure",
            "title",
            "comment_count",
            "explicit_pull_urls",
            "same_repo_fix_ref_urls",
            "has_local_diff_or_patch_snippet",
            "local_diff_or_patch_snippet",
            "fix_or_root_cause_snippet",
            "evidence_tier",
        ],
    )

    top = out_rows[:40]
    write_csv(
        TABLE_DIR / "linked_fix_evidence_top40.csv",
        top,
        [
            "blind_id",
            "repository",
            "issue_number",
            "gold_primary_failure",
            "title",
            "comment_count",
            "explicit_pull_urls",
            "same_repo_fix_ref_urls",
            "has_local_diff_or_patch_snippet",
            "local_diff_or_patch_snippet",
            "fix_or_root_cause_snippet",
            "evidence_tier",
        ],
    )

    lines = [
        "# Linked Fix Evidence Analysis",
        "",
        "This analysis mines the already-fetched public issue/comment context for linked pull requests, same-repository fix references, and inline patch/diff snippets. It is source-backed but does not claim full code-diff root-cause adjudication.",
        "",
        "| signal | issues |",
        "| --- | ---: |",
        f"| explicit pull-request URL | {explicit_pr_rows} |",
        f"| same-repository fix/PR reference pattern | {same_repo_ref_rows} |",
        f"| inline diff or patch snippet in issue context | {diff_rows} |",
        f"| fix/root-cause/workaround text snippet | {fix_snippet_rows} |",
        f"| rows with at least one linked-fix evidence signal | {len(out_rows)} |",
        "",
        "Generated tables:",
        "",
        "- `tables/linked_fix_evidence_subset.csv`",
        "- `tables/linked_fix_evidence_top40.csv`",
        "",
        *linked_pr_diff_summary(),
        "The top-40 table can be used as a conference appendix or as the seed for future linked-PR/code-diff adjudication. The linked PR diffs are provenance for future root-cause labels, not adjudicated root-cause labels in the current benchmark.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
