from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark_expanded_v2_canonical.csv"
PREDICTIONS = ROOT / "evaluation" / "v2_gold_model_predictions.csv"
REPAIRED = ROOT / "annotation" / "gold_expansion_1000_repaired.csv"
ORIGINAL_PACKET = ROOT / "annotation" / "annotator_A_blind.csv"
OUT_DIR = ROOT / "llm" / "finetune"
PACKET = ROOT / "evaluation" / "v2_gold_llm_training_packet.jsonl"
FEWSHOT = ROOT / "evaluation" / "v2_gold_balanced_fewshot_examples.json"
TRAIN = OUT_DIR / "gpu_nfbench_v2_standalone_train.jsonl"
VAL = OUT_DIR / "gpu_nfbench_v2_standalone_val.jsonl"
TEST = OUT_DIR / "gpu_nfbench_v2_standalone_test.jsonl"
OPENAI_TRAIN = OUT_DIR / "openai_chat_finetune_v2_train.jsonl"
OPENAI_VAL = OUT_DIR / "openai_chat_finetune_v2_val.jsonl"
LABEL_MAP = OUT_DIR / "label_map_v2.json"
REPORT = ROOT / "reports" / "v2_llm_finetune_files.md"

SYSTEM = """You are GPU-NFBench-Triage, a strict standalone classifier for GPU/kernel issue reports.
Return JSON only. Choose exactly one primary_failure_label from:
nan_inf, overflow_underflow, precision_tolerance, dtype_casting, crash_compile, performance_only, not_numerical_failure.
Also return secondary_cause_labels, is_true_numerical_failure, evidence_quote, and confidence.
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
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


def output_json(row: dict[str, object]) -> str:
    out = row["output"]
    return json.dumps(
        {
            "primary_failure_label": out["primary_failure_label"],
            "secondary_cause_labels": str(out["secondary_cause_labels"]).split("|"),
            "is_true_numerical_failure": out["is_true_numerical_failure"],
            "evidence_quote": out["evidence_quote"],
            "confidence": "high",
        },
        sort_keys=True,
    )


def chat_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": str(row["input"])},
            {"role": "assistant", "content": output_json(row)},
        ]
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def split_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["output"]["primary_failure_label"])].append(row)
    train: list[dict[str, object]] = []
    val: list[dict[str, object]] = []
    test: list[dict[str, object]] = []
    for _, items in sorted(by_label.items()):
        items = sorted(items, key=lambda row: str(row["id"]))
        for index, row in enumerate(items):
            bucket = index % 10
            if bucket == 0:
                test.append(row)
            elif bucket == 1:
                val.append(row)
            else:
                train.append(row)
    return sorted(train, key=lambda row: str(row["id"])), sorted(val, key=lambda row: str(row["id"])), sorted(test, key=lambda row: str(row["id"]))


def main() -> None:
    gold = {row["blind_id"]: row for row in read_csv(GOLD)}
    predictions = {row["blind_id"]: row for row in read_csv(PREDICTIONS)}
    repaired = {row["expansion_id"]: row for row in read_csv(REPAIRED)}
    original = {row["blind_id"]: row for row in read_csv(ORIGINAL_PACKET)}
    packet_rows: list[dict[str, object]] = []
    for blind_id, row in gold.items():
        if blind_id in repaired:
            issue_text = expansion_text(repaired[blind_id])
        elif blind_id in original:
            issue_text = original_text(original[blind_id])
        else:
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

    write_jsonl(PACKET, packet_rows)
    train, val, test = split_rows(packet_rows)
    write_jsonl(TRAIN, train)
    write_jsonl(VAL, val)
    write_jsonl(TEST, test)
    write_jsonl(OPENAI_TRAIN, [chat_row(row) for row in train])
    write_jsonl(OPENAI_VAL, [chat_row(row) for row in val])

    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in packet_rows:
        by_label[str(row["output"]["primary_failure_label"])].append(row)
    FEWSHOT.write_text(
        json.dumps({label: rows[:8] for label, rows in sorted(by_label.items())}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    counts = Counter(str(row["output"]["primary_failure_label"]) for row in packet_rows)
    LABEL_MAP.write_text(json.dumps({"primary_labels": sorted(counts)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "\n".join(
            [
                "# V2 LLM Fine-Tune Files",
                "",
                f"Total rows: {len(packet_rows)}",
                f"Train rows: {len(train)}",
                f"Validation rows: {len(val)}",
                f"Test rows: {len(test)}",
                "",
                f"Packet: `{PACKET.relative_to(ROOT)}`",
                f"Train: `{TRAIN.relative_to(ROOT)}`",
                f"Validation: `{VAL.relative_to(ROOT)}`",
                f"Test: `{TEST.relative_to(ROOT)}`",
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
