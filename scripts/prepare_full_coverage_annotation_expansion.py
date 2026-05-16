from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv"
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
BLIND_OUT = ROOT / "annotation" / "full_coverage_expansion_blind.csv"
REVIEW_OUT = ROOT / "annotation" / "full_coverage_expansion_review.csv"

BLIND_FIELDS = [
    "expansion_id",
    "repository",
    "title",
    "url",
    "state",
    "created_at",
    "updated_at",
    "github_labels",
    "body_excerpt",
    "primary_failure_label",
    "secondary_cause_labels",
    "is_true_numerical_failure",
    "confidence",
    "evidence_quote",
    "notes",
]
REVIEW_FIELDS = BLIND_FIELDS + [
    "candidate_primary_failure",
    "candidate_failure_labels",
    "candidate_cause_labels",
    "source_file",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expansion_id(url: str, index: int) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"EGNF-{index:04d}-{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare blind human-labeling packets for expanding full-coverage GPU-NFBench.")
    parser.add_argument("--per-label", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--blind-out", type=Path, default=BLIND_OUT)
    parser.add_argument("--review-out", type=Path, default=REVIEW_OUT)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    gold_urls = {row["url"] for row in read_csv(GOLD)}
    rows = [row for row in read_csv(SEED) if row["url"] not in gold_urls]

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row.get("candidate_primary_failure") or "needs_review"].append(row)

    selected = []
    for label in sorted(buckets):
        bucket = buckets[label]
        rng.shuffle(bucket)
        selected.extend(bucket[: args.per_label])
    selected.sort(key=lambda row: (row.get("candidate_primary_failure", ""), row["repository"], row["url"]))

    blind_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        base = {
            "expansion_id": expansion_id(row["url"], index),
            "repository": row["repository"],
            "title": row["title"],
            "url": row["url"],
            "state": row["state"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "github_labels": row["github_labels"],
            "body_excerpt": row["body_excerpt"],
            "primary_failure_label": "",
            "secondary_cause_labels": "",
            "is_true_numerical_failure": "",
            "confidence": "",
            "evidence_quote": "",
            "notes": "",
        }
        blind_rows.append(base)
        review = dict(base)
        review.update(
            {
                "candidate_primary_failure": row["candidate_primary_failure"],
                "candidate_failure_labels": row["candidate_failure_labels"],
                "candidate_cause_labels": row["candidate_cause_labels"],
                "source_file": row["source_file"],
            }
        )
        review_rows.append(review)

    write_csv(args.blind_out, blind_rows, BLIND_FIELDS)
    write_csv(args.review_out, review_rows, REVIEW_FIELDS)
    print(f"{args.blind_out} ({len(blind_rows)} rows)")
    print(f"{args.review_out} ({len(review_rows)} rows)")


if __name__ == "__main__":
    main()
