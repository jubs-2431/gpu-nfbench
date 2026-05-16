from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import collect_online_issue_candidates as collector


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_online" / "external_repo_candidate_issues.jsonl"
CSV_OUT = ROOT / "data" / "processed" / "external_repo_candidate_issue_pool.csv"
REPORT = ROOT / "reports" / "external_repo_candidate_collection.md"


def main() -> None:
    seen: dict[str, dict[str, str]] = {}
    malformed = 0
    if not RAW.exists():
        raise SystemExit(f"Missing raw candidate file: {RAW}")

    with RAW.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            url = str(item.get("url") or "")
            if not url:
                continue
            labels = [label.get("name", "") for label in item.get("labels", []) if isinstance(label, dict)]
            title = str(item.get("title") or "")
            body = str(item.get("body") or "")
            text = f"{title}\n{' '.join(labels)}\n{body}"
            failure_labels = collector.labels_from_patterns(text, collector.FAILURE_PATTERNS)
            cause_labels = collector.labels_from_patterns(text, collector.CAUSE_PATTERNS)
            row = {
                "source": "github_search_external_partial",
                "source_query": str(item.get("_source_query") or ""),
                "repository": str(item.get("_source_repo") or ""),
                "issue_number": str(item.get("number") or ""),
                "title": collector.clean(title, 300),
                "url": url,
                "state": str(item.get("state") or ""),
                "created_at": str(item.get("createdAt") or ""),
                "updated_at": str(item.get("updatedAt") or ""),
                "github_labels": "|".join(label for label in labels if label),
                "candidate_failure_labels": "|".join(failure_labels) if failure_labels else "needs_review",
                "candidate_primary_failure": collector.primary_failure_label(text),
                "candidate_cause_labels": "|".join(cause_labels) if cause_labels else "unknown",
                "body_excerpt": collector.clean(body, 1200),
            }
            row["priority_score"] = str(collector.priority_score(row))
            old = seen.get(url)
            if old is None or int(row["priority_score"]) > int(old["priority_score"]):
                seen[url] = row

    rows = sorted(seen.values(), key=lambda row: (-int(row["priority_score"]), row["repository"], row["url"]))
    fieldnames = [
        "source",
        "source_query",
        "repository",
        "issue_number",
        "title",
        "url",
        "state",
        "created_at",
        "updated_at",
        "github_labels",
        "candidate_failure_labels",
        "candidate_primary_failure",
        "candidate_cause_labels",
        "priority_score",
        "body_excerpt",
    ]
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    repo_counts = Counter(row["repository"] for row in rows)
    label_counts = Counter(row["candidate_primary_failure"] for row in rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# External Repository Candidate Pool",
                "",
                "This file materializes the partial raw GitHub search results collected before the authenticated GitHub Search API hit its rate limit. These rows are expansion candidates only; they are not included in the gold benchmark and must not be described as human labels.",
                "",
                f"Raw JSONL rows: {sum(1 for _ in RAW.open(encoding='utf-8'))}",
                f"Unique candidates: {len(rows)}",
                f"Malformed raw rows skipped: {malformed}",
                f"Raw JSONL: `{RAW.relative_to(ROOT)}`",
                f"Candidate CSV: `{CSV_OUT.relative_to(ROOT)}`",
                "",
                "## Candidate label counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(label_counts.items())],
                "",
                "## Repository counts",
                "",
                *[f"- {repo}: {count}" for repo, count in repo_counts.most_common()],
                "",
                "## Submission guidance",
                "",
                "Use this as future-work evidence that additional GPU-kernel repositories can be mined. Do not fold these rows into the headline benchmark until they are blind-reviewed and adjudicated.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(CSV_OUT)
    print(REPORT)


if __name__ == "__main__":
    main()
