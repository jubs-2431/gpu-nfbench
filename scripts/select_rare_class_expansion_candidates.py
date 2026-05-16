from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv"
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
OUT = ROOT / "annotation" / "rare_class_expansion_candidates.csv"
REPORT = ROOT / "reports" / "rare_class_expansion_plan.md"


TARGET_LABELS = {"crash_compile", "performance_only"}
MAX_PER_LABEL = 25


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seed = read_csv(SEED)
    gold_urls = {row["url"] for row in read_csv(GOLD)}
    selected: list[dict[str, object]] = []
    by_label: Counter[str] = Counter()

    for row in sorted(seed, key=lambda r: (r.get("candidate_primary_failure", ""), r.get("repository", ""), r.get("updated_at", "")), reverse=True):
        label = row.get("candidate_primary_failure", "")
        if label not in TARGET_LABELS or row.get("url", "") in gold_urls:
            continue
        if by_label[label] >= MAX_PER_LABEL:
            continue
        by_label[label] += 1
        selected.append(
            {
                "target_label": label,
                "repository": row.get("repository", ""),
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "state": row.get("state", ""),
                "created_at": row.get("created_at", ""),
                "updated_at": row.get("updated_at", ""),
                "github_labels": row.get("github_labels", ""),
                "body_excerpt": row.get("body_excerpt", ""),
                "annotation_status": "candidate_not_gold_needs_context_and_human_review",
            }
        )

    write_csv(
        OUT,
        selected,
        [
            "target_label",
            "repository",
            "title",
            "url",
            "state",
            "created_at",
            "updated_at",
            "github_labels",
            "body_excerpt",
            "annotation_status",
        ],
    )

    lines = [
        "# Rare-Class Expansion Plan",
        "",
        "The adjudicated benchmark has low support for `crash_compile` and `performance_only`. This file selects additional seed-dataset candidates for a future human-labeled expansion. These rows are not gold labels.",
        "",
        "| target_label | candidates |",
        "| --- | ---: |",
        *[f"| {label} | {by_label[label]} |" for label in sorted(TARGET_LABELS)],
        "",
        f"Candidate packet: `{OUT.relative_to(ROOT)}`",
        "",
        "Required next step before using these rows as benchmark evidence: fetch full context, blind-label with two human annotators, adjudicate disagreements, and rerun agreement metrics.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
