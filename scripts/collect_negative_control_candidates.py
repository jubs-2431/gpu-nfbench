from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "collect_online_issue_candidates.py"

NEGATIVE_TERMS = [
    "documentation",
    "install",
    "build error",
    "feature request",
    "question",
    "api proposal",
    "refactor",
    "ci failure",
    "windows install",
    "dependency",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect likely negative-control GitHub issues for not_numerical_failure review.")
    parser.add_argument("--per-query", type=int, default=12)
    parser.add_argument("--max-total", type=int, default=300)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(COLLECTOR),
        "--per-query",
        str(args.per_query),
        "--max-total",
        str(args.max_total),
        "--sleep",
        "0.05",
        "--raw-out",
        "data/raw_online/github_negative_control_candidates.jsonl",
        "--csv-out",
        "data/processed/negative_control_candidate_issue_pool.csv",
        "--report",
        "reports/negative_control_candidate_collection.md",
        "--terms",
        *NEGATIVE_TERMS,
    ]
    raise SystemExit(subprocess.run(cmd, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
