from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv"
OUT = ROOT / "data" / "processed" / "validation_subset.csv"


TARGET_LABELS = [
    "dtype_casting",
    "nan_inf",
    "precision_tolerance",
    "overflow_underflow",
    "crash_compile",
    "performance_only",
    "needs_review",
]


def main() -> None:
    with DATA.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row["repository"], row["candidate_primary_failure"])].append(row)

    selected: list[dict[str, str]] = []
    for repo in sorted({row["repository"] for row in rows}):
        for label in TARGET_LABELS:
            candidates = buckets.get((repo, label), [])
            candidates = sorted(candidates, key=lambda r: (r["created_at"], r["url"]))
            selected.extend(candidates[:2])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    print(f"wrote {len(selected)} validation issues to {OUT}")


if __name__ == "__main__":
    main()

