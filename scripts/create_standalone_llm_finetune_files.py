from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "evaluation" / "expanded_gold_llm_training_packet.jsonl"
OUT_DIR = ROOT / "llm" / "finetune"
TRAIN_JSONL = OUT_DIR / "gpu_nfbench_standalone_train.jsonl"
VAL_JSONL = OUT_DIR / "gpu_nfbench_standalone_val.jsonl"
TEST_JSONL = OUT_DIR / "gpu_nfbench_standalone_test.jsonl"
OPENAI_TRAIN = OUT_DIR / "openai_chat_finetune_train.jsonl"
OPENAI_VAL = OUT_DIR / "openai_chat_finetune_val.jsonl"
LABEL_MAP = OUT_DIR / "label_map.json"
REPORT = ROOT / "reports" / "standalone_llm_finetune_files.md"


SYSTEM = """You are GPU-NFBench-Triage, a strict standalone classifier for GPU/kernel issue reports.
Return JSON only. Choose exactly one primary_failure_label from:
nan_inf, overflow_underflow, precision_tolerance, dtype_casting, crash_compile, performance_only, not_numerical_failure.
Also return secondary_cause_labels, is_true_numerical_failure, evidence_quote, and confidence.
"""


def read_packet() -> list[dict[str, object]]:
    rows = []
    with PACKET.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["output"]["primary_failure_label"])].append(row)
    train = []
    val = []
    test = []
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


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


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


def counts(rows: list[dict[str, object]]) -> Counter[str]:
    return Counter(str(row["output"]["primary_failure_label"]) for row in rows)


def main() -> None:
    rows = read_packet()
    train, val, test = split_rows(rows)
    write_jsonl(TRAIN_JSONL, train)
    write_jsonl(VAL_JSONL, val)
    write_jsonl(TEST_JSONL, test)
    write_jsonl(OPENAI_TRAIN, [chat_row(row) for row in train])
    write_jsonl(OPENAI_VAL, [chat_row(row) for row in val])

    label_counts = counts(rows)
    LABEL_MAP.write_text(
        json.dumps(
            {
                "primary_labels": sorted(label_counts),
                "secondary_labels": [
                    "memory_mask_bounds",
                    "compiler_codegen",
                    "async_race_ordering",
                    "hardware_backend",
                    "reduction_accumulation",
                    "api_semantics",
                    "environment_configuration",
                    "unknown",
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Standalone LLM Fine-Tune Files",
        "",
        f"Total rows: {len(rows)}",
        f"Train rows: {len(train)}",
        f"Validation rows: {len(val)}",
        f"Test rows: {len(test)}",
        "",
        "Files:",
        "",
        f"- `{TRAIN_JSONL.relative_to(ROOT)}`",
        f"- `{VAL_JSONL.relative_to(ROOT)}`",
        f"- `{TEST_JSONL.relative_to(ROOT)}`",
        f"- `{OPENAI_TRAIN.relative_to(ROOT)}`",
        f"- `{OPENAI_VAL.relative_to(ROOT)}`",
        f"- `{LABEL_MAP.relative_to(ROOT)}`",
        "",
        "These files are ready for a true standalone LLM fine-tune. The test split should remain untouched until final evaluation.",
        "",
        "## Label counts",
        "",
        *[f"- {label}: {count}" for label, count in sorted(label_counts.items())],
        "",
        "## Current local status",
        "",
        "This machine does not currently have a local fine-tuning stack such as MLX-LM, transformers, PEFT, or TRL installed. The local Ollama Modelfile is a prompted model wrapper, not a weight-updated fine-tune.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
