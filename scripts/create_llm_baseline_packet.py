from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
PACKET = ROOT / "annotation" / "annotator_A_blind.csv"
EVAL_DIR = ROOT / "evaluation"
PROMPTS = EVAL_DIR / "llm_baseline_prompts.jsonl"
SCHEMA = EVAL_DIR / "llm_baseline_prediction_schema.json"
REPORT = ROOT / "reports" / "llm_baseline_protocol.md"


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

SECONDARY_LABELS = [
    "memory_mask_bounds",
    "compiler_codegen",
    "async_race_ordering",
    "hardware_backend",
    "reduction_accumulation",
    "api_semantics",
    "environment_configuration",
    "unknown",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def trim(value: str, limit: int = 1800) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def system_prompt() -> str:
    return (
        "You classify public GPU/kernel issue reports for a research benchmark. "
        "Use only the supplied title, labels, body excerpt, comments excerpt, and URL. "
        "Return JSON only. Choose one primary_failure_label from the allowed list, "
        "one or more secondary_cause_labels from the allowed list, is_true_numerical_failure as yes/no/unclear, "
        "a short evidence_quote copied from the supplied text, and confidence as high/medium/low."
    )


def user_prompt(row: dict[str, str]) -> str:
    return "\n".join(
        [
            f"Allowed primary labels: {', '.join(PRIMARY_LABELS)}",
            f"Allowed secondary cause labels: {', '.join(SECONDARY_LABELS)}",
            "",
            f"blind_id: {row['blind_id']}",
            f"repository: {row['repository']}",
            f"url: {row['url']}",
            f"title: {row['title']}",
            f"github_labels: {row.get('github_labels', '')}",
            "",
            "issue_body_excerpt:",
            trim(row.get("issue_body_excerpt", "")),
            "",
            "comments_excerpt:",
            trim(row.get("comments_excerpt", "")),
        ]
    )


def main() -> None:
    gold = {row["blind_id"]: row for row in read_csv(GOLD)}
    packet_rows = read_csv(PACKET)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with PROMPTS.open("w", encoding="utf-8") as fh:
        for row in packet_rows:
            item = {
                "blind_id": row["blind_id"],
                "messages": [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": user_prompt(row)},
                ],
                "metadata": {
                    "repository": row["repository"],
                    "issue_number": row["issue_number"],
                    "url": row["url"],
                },
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    SCHEMA.write_text(
        json.dumps(
            {
                "required_csv_columns": [
                    "blind_id",
                    "primary_failure_label",
                    "secondary_cause_labels_pipe_separated",
                    "is_true_numerical_failure",
                    "evidence_quote",
                    "confidence_high_medium_low",
                ],
                "allowed_primary_failure_labels": PRIMARY_LABELS,
                "allowed_secondary_cause_labels": SECONDARY_LABELS,
                "allowed_true_failure_values": ["yes", "no", "unclear"],
                "allowed_confidence_values": ["high", "medium", "low"],
                "evaluation_script": "scripts/evaluate_llm_baseline_predictions.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    by_label: dict[str, list[str]] = defaultdict(list)
    for bid, row in gold.items():
        by_label[row["gold_primary_failure"]].append(bid)
    few_shot_ids = []
    for label in PRIMARY_LABELS:
        if by_label.get(label):
            few_shot_ids.append(sorted(by_label[label])[0])

    lines = [
        "# LLM Baseline Protocol",
        "",
        "This packet prepares an external zero-shot or few-shot LLM baseline without exposing gold labels in the prompt file.",
        "",
        "## Files",
        "",
        f"- Prompt JSONL: `{PROMPTS.relative_to(ROOT)}`",
        f"- Prediction schema: `{SCHEMA.relative_to(ROOT)}`",
        "- Evaluator: `scripts/evaluate_llm_baseline_predictions.py`",
        "",
        "## Recommended evaluation",
        "",
        "1. Run the same prompt file through the chosen LLM with temperature 0.",
        "2. Save one prediction per row using the CSV schema.",
        "3. Evaluate against `data/processed/gold_benchmark.csv` with the evaluator script.",
        "4. Report exact model name, date, temperature, prompt file checksum, and whether examples were zero-shot or few-shot.",
        "",
        "## Few-shot pool",
        "",
        "If a few-shot condition is used, draw examples only from a training fold and never from the held-out fold. A convenient balanced pool for fold construction is:",
        "",
        ", ".join(few_shot_ids),
        "",
        "No LLM predictions are included in this artifact because the local environment does not contain valid external-model credentials.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(PROMPTS)


if __name__ == "__main__":
    main()
