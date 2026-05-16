from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv"
OUT = ROOT / "data" / "processed" / "gold_candidate_subset.csv"
COVERAGE_OUT = ROOT / "tables" / "gold_candidate_coverage.csv"

TARGET_LABELS = [
    "dtype_casting",
    "nan_inf",
    "precision_tolerance",
    "overflow_underflow",
    "crash_compile",
    "performance_only",
    "needs_review",
]
MAX_PER_REPO_LABEL = 5


def pick_temporal_spread(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    rows = sorted(rows, key=lambda r: (r["created_at"], r["url"]))
    if len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[0]]
    indices = [round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)]
    seen = set()
    picked = []
    for idx in indices:
        if idx not in seen:
            picked.append(rows[idx])
            seen.add(idx)
    return picked


def main() -> None:
    rows = list(csv.DictReader(DATA.open(newline="", encoding="utf-8")))
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row["repository"], row["candidate_primary_failure"])].append(row)

    selected: list[dict[str, str]] = []
    for repo in sorted({row["repository"] for row in rows}):
        for label in TARGET_LABELS:
            selected.extend(pick_temporal_spread(buckets.get((repo, label), []), MAX_PER_REPO_LABEL))

    selected = sorted(
        selected,
        key=lambda r: (r["repository"], r["candidate_primary_failure"], r["created_at"], r["url"]),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    repo_counts = Counter(row["repository"] for row in selected)
    label_counts = Counter(row["candidate_primary_failure"] for row in selected)
    coverage_rows = (
        [{"dimension": "repository", "value": key, "issues": value} for key, value in sorted(repo_counts.items())]
        + [{"dimension": "candidate_primary_failure", "value": key, "issues": value} for key, value in sorted(label_counts.items())]
    )
    COVERAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with COVERAGE_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dimension", "value", "issues"])
        writer.writeheader()
        writer.writerows(coverage_rows)
    print(f"wrote {len(selected)} gold-candidate issues to {OUT}")


if __name__ == "__main__":
    main()
