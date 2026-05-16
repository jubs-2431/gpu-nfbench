from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "llm" / "finetune" / "gpu_nfbench_v2_standalone_test.jsonl"
OUT = ROOT / "evaluation" / "v2_heldout_llm_baseline_prompts.jsonl"

PRIMARY_LABELS = [
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
    "needs_review",
]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def trim(value: str, limit: int = 2500) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in read_jsonl(TEST):
            item = {
                "blind_id": row["id"],
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You classify public GPU/kernel issue reports for a research benchmark. "
                            "Return JSON only with primary_failure_label, secondary_cause_labels, "
                            "is_true_numerical_failure, evidence_quote, and confidence."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                f"Allowed primary labels: {', '.join(PRIMARY_LABELS)}",
                                "",
                                f"blind_id: {row['id']}",
                                f"repository: {row['repository']}",
                                "",
                                "Issue:",
                                trim(str(row["input"])),
                            ]
                        ),
                    },
                ],
                "metadata": {"repository": row["repository"]},
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
