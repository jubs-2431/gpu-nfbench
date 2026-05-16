from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark_expanded.csv"
PREDICTIONS = ROOT / "evaluation" / "expanded_gold_model_predictions.csv"
REPAIRED = ROOT / "annotation" / "gold_expansion_1000_repaired.csv"
ORIGINAL_PACKET = ROOT / "annotation" / "annotator_A_blind.csv"
OUT_JSONL = ROOT / "evaluation" / "expanded_gold_llm_training_packet.jsonl"
OUT_FEWSHOT = ROOT / "evaluation" / "expanded_gold_balanced_fewshot_examples.json"
REPORT = ROOT / "reports" / "expanded_llm_training_packet.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def trim(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def original_text(row: dict[str, str]) -> str:
    return "\n".join(
        [
            f"repository: {row.get('repository', '')}",
            f"title: {row.get('title', '')}",
            f"github_labels: {row.get('github_labels', '')}",
            f"issue_body_excerpt: {trim(row.get('issue_body_excerpt', ''), 1400)}",
            f"comments_excerpt: {trim(row.get('comments_excerpt', ''), 900)}",
        ]
    )


def expansion_text(row: dict[str, str]) -> str:
    return "\n".join(
        [
            f"repository: {row.get('repository', '')}",
            f"title: {row.get('title', '')}",
            f"github_labels: {row.get('github_labels', '')}",
            f"body_excerpt: {trim(row.get('body_excerpt', ''), 1800)}",
        ]
    )


def main() -> None:
    gold = {row["blind_id"]: row for row in read_csv(GOLD)}
    predictions = {row["blind_id"]: row for row in read_csv(PREDICTIONS)}
    repaired = {row["expansion_id"]: row for row in read_csv(REPAIRED)}
    original = {row["blind_id"]: row for row in read_csv(ORIGINAL_PACKET)}

    packet_rows = []
    for blind_id, row in gold.items():
        if blind_id in repaired:
            source = repaired[blind_id]
            issue_text = expansion_text(source)
        elif blind_id in original:
            source = original[blind_id]
            issue_text = original_text(source)
        else:
            source = row
            issue_text = f"repository: {row.get('repository', '')}\ntitle: {row.get('title', '')}\nevidence: {row.get('gold_evidence_quote', '')}"
        pred = predictions.get(blind_id, {})
        packet_rows.append(
            {
                "id": blind_id,
                "repository": row.get("repository", ""),
                "input": issue_text,
                "output": {
                    "primary_failure_label": row["gold_primary_failure"],
                    "secondary_cause_labels": row["gold_secondary_cause_labels"],
                    "is_true_numerical_failure": row["gold_is_true_numerical_failure"],
                    "evidence_quote": row["gold_evidence_quote"],
                },
                "metadata": {
                    "source_split": pred.get("source_split", "unknown"),
                    "current_ensemble_prediction": pred.get("expanded_gold_vote_ensemble_prediction", ""),
                    "current_ensemble_vote_count": pred.get("ensemble_vote_count", ""),
                    "current_ensemble_vote_margin": pred.get("ensemble_vote_margin", ""),
                },
            }
        )

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for row in packet_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    by_label: dict[str, list[dict[str, object]]] = {}
    for row in packet_rows:
        by_label.setdefault(str(row["output"]["primary_failure_label"]), []).append(row)
    fewshot = {}
    for label, rows in sorted(by_label.items()):
        rows = sorted(rows, key=lambda row: (row["metadata"].get("current_ensemble_vote_count", ""), row["id"]), reverse=True)
        fewshot[label] = rows[:8]
    OUT_FEWSHOT.write_text(json.dumps(fewshot, indent=2, sort_keys=True), encoding="utf-8")

    counts = Counter(str(row["output"]["primary_failure_label"]) for row in packet_rows)
    REPORT.write_text(
        "\n".join(
            [
                "# Expanded LLM Training Packet",
                "",
                f"Rows written: {len(packet_rows)}",
                f"JSONL packet: `{OUT_JSONL.relative_to(ROOT)}`",
                f"Balanced few-shot examples: `{OUT_FEWSHOT.relative_to(ROOT)}`",
                "",
                "This is an LLM-ready supervised/RAG packet. It is suitable for retrieval-augmented prompting, few-shot prompting, or later fine-tuning with a provider that supports JSONL supervised examples.",
                "",
                "## Label counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(counts.items())],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
