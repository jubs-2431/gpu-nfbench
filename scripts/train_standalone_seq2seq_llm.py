from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "llm" / "finetune" / "gpu_nfbench_standalone_train.jsonl"
VAL = ROOT / "llm" / "finetune" / "gpu_nfbench_standalone_val.jsonl"
TEST = ROOT / "llm" / "finetune" / "gpu_nfbench_standalone_test.jsonl"
OUT_DIR = ROOT / "llm" / "models" / "gpu_nfbench_flan_t5_small"
PREDICTIONS = ROOT / "evaluation" / "standalone_seq2seq_llm_predictions.csv"
METRICS = ROOT / "tables" / "standalone_seq2seq_llm_metrics.csv"
REPORT = ROOT / "reports" / "standalone_seq2seq_llm_training.md"

PRIMARY_LABELS = {
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def trim(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def target_json(row: dict[str, Any]) -> str:
    out = row["output"]
    return json.dumps(
        {
            "primary_failure_label": out["primary_failure_label"],
            "secondary_cause_labels": str(out["secondary_cause_labels"]).split("|"),
            "is_true_numerical_failure": out["is_true_numerical_failure"],
            "evidence_quote": trim(str(out["evidence_quote"]), 180),
            "confidence": "high",
        },
        sort_keys=True,
    )


def target_text(row: dict[str, Any], target_format: str) -> str:
    if target_format == "label":
        return str(row["output"]["primary_failure_label"])
    return target_json(row)


def prompt(row: dict[str, Any], target_format: str = "label") -> str:
    if target_format == "label":
        return (
            "Classify this GPU/kernel GitHub issue into exactly one label. "
            "Allowed labels: nan_inf, overflow_underflow, precision_tolerance, dtype_casting, "
            "crash_compile, performance_only, not_numerical_failure. "
            "Answer with only the label string.\n\n"
            f"Issue:\n{trim(str(row['input']), 2200)}"
        )
    return (
        "Classify this GPU/kernel GitHub issue into exactly one primary_failure_label. "
        "Allowed labels: nan_inf, overflow_underflow, precision_tolerance, dtype_casting, "
        "crash_compile, performance_only, not_numerical_failure. "
        "Return JSON with primary_failure_label, secondary_cause_labels, "
        "is_true_numerical_failure, evidence_quote, confidence.\n\n"
        f"Issue:\n{trim(str(row['input']), 2200)}"
    )


class JsonDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_input: int, max_output: int, target_format: str) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output
        self.target_format = target_format

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            prompt(row, self.target_format),
            max_length=self.max_input,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            target_text(row, self.target_format),
            max_length=self.max_output,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def normalize_primary(text: str) -> str:
    stripped = text.strip()
    if stripped in PRIMARY_LABELS:
        return stripped
    parsed = extract_json(text)
    label = str(parsed.get("primary_failure_label", "")).strip()
    if label in PRIMARY_LABELS:
        return label
    for candidate in PRIMARY_LABELS:
        if candidate in text:
            return candidate
    return "needs_review"


def prf(labels: list[str], preds: list[str]) -> tuple[float, float]:
    if not labels:
        return 0.0, 0.0
    accuracy = sum(a == b for a, b in zip(labels, preds)) / len(labels)
    all_labels = sorted(set(labels) | set(preds))
    f1s = []
    for label in all_labels:
        tp = sum(a == label and b == label for a, b in zip(labels, preds))
        fp = sum(a != label and b == label for a, b in zip(labels, preds))
        fn = sum(a == label and b != label for a, b in zip(labels, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return accuracy, sum(f1s) / len(f1s) if f1s else 0.0


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def balance_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["output"]["primary_failure_label"]), []).append(row)
    max_count = max(len(items) for items in by_label.values())
    balanced: list[dict[str, Any]] = []
    for _, items in sorted(by_label.items()):
        if not items:
            continue
        balanced.extend(items)
        needed = max_count - len(items)
        balanced.extend(rng.choice(items) for _ in range(needed))
    rng.shuffle(balanced)
    return balanced


def display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@torch.no_grad()
def evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: torch.device, max_input: int, max_new_tokens: int, target_format: str) -> tuple[list[dict[str, object]], float, float]:
    model.eval()
    out_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        encoded = tokenizer(prompt(row, target_format), max_length=max_input, truncation=True, return_tensors="pt").to(device)
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, num_beams=1)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        pred = normalize_primary(text)
        gold = row["output"]["primary_failure_label"]
        out_rows.append(
            {
                "id": row["id"],
                "repository": row["repository"],
                "gold_primary_failure": gold,
                "predicted_primary_failure": pred,
                "raw_generation": text,
                "correct": str(gold == pred).lower(),
            }
        )
        print(f"eval {index}/{len(rows)} {row['id']} gold={gold} pred={pred}", flush=True)
    labels = [str(row["gold_primary_failure"]) for row in out_rows]
    preds = [str(row["predicted_primary_failure"]) for row in out_rows]
    accuracy, macro_f1 = prf(labels, preds)
    return out_rows, accuracy, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a standalone seq2seq LLM for GPU-NFBench.")
    parser.add_argument("--base-model", default="google/flan-t5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-input", type=int, default=512)
    parser.add_argument("--max-output", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--target-format", choices=["label", "json"], default="label")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--balance-train", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN)
    parser.add_argument("--val-jsonl", type=Path, default=VAL)
    parser.add_argument("--test-jsonl", type=Path, default=TEST)
    parser.add_argument("--predictions-out", type=Path, default=PREDICTIONS)
    parser.add_argument("--metrics-out", type=Path, default=METRICS)
    parser.add_argument("--report-out", type=Path, default=REPORT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    test_rows = read_jsonl(args.test_jsonl)
    random.shuffle(train_rows)
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    if args.balance_train:
        train_rows = balance_rows(train_rows, args.seed)
    if args.max_val:
        val_rows = val_rows[: args.max_val]
    if args.max_test:
        test_rows = test_rows[: args.max_test]

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps is not available.")
        device = torch.device("mps")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(
        f"device={device} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
        f"base_model={args.base_model} local_files_only={args.local_files_only}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, local_files_only=args.local_files_only).to(device)

    dataset = JsonDataset(train_rows, tokenizer, args.max_input, args.max_output, args.target_format)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 20), num_training_steps=total_steps)

    model.train()
    global_step = 0
    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            losses.append(float(loss.detach().cpu()))
            if global_step % 25 == 0:
                recent = sum(losses[-25:]) / len(losses[-25:])
                print(f"epoch={epoch} step={global_step}/{total_steps} loss={recent:.4f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    val_predictions, val_accuracy, val_macro_f1 = evaluate(model, tokenizer, val_rows, device, args.max_input, args.max_new_tokens, args.target_format)
    test_predictions, test_accuracy, test_macro_f1 = evaluate(model, tokenizer, test_rows, device, args.max_input, args.max_new_tokens, args.target_format)
    write_csv(args.predictions_out, test_predictions, ["id", "repository", "gold_primary_failure", "predicted_primary_failure", "raw_generation", "correct"])
    metric_rows = [
        {
            "split": "validation",
            "rows": len(val_rows),
            "accuracy": f"{val_accuracy:.3f}",
            "macro_f1": f"{val_macro_f1:.3f}",
        },
        {
            "split": "test",
            "rows": len(test_rows),
            "accuracy": f"{test_accuracy:.3f}",
            "macro_f1": f"{test_macro_f1:.3f}",
        },
    ]
    write_csv(args.metrics_out, metric_rows, ["split", "rows", "accuracy", "macro_f1"])
    pred_counts = Counter(row["predicted_primary_failure"] for row in test_predictions)
    gold_counts = Counter(row["gold_primary_failure"] for row in test_predictions)
    args.report_out.write_text(
        "\n".join(
            [
                "# Standalone Seq2Seq LLM Training",
                "",
                f"Base model: `{args.base_model}`",
                f"Saved model: `{display_path(args.out_dir)}`",
                f"Train rows: {len(train_rows)}",
                f"Validation rows: {len(val_rows)}",
                f"Test rows: {len(test_rows)}",
                f"Epochs: {args.epochs}",
                f"Batch size: {args.batch_size}",
                f"Device: `{device}`",
                f"Target format: `{args.target_format}`",
                "",
                "| split | rows | accuracy | macro F1 |",
                "| --- | ---: | ---: | ---: |",
                *[f"| {row['split']} | {row['rows']} | {row['accuracy']} | {row['macro_f1']} |" for row in metric_rows],
                "",
                "## Test Gold Counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(gold_counts.items())],
                "",
                "## Test Prediction Counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(pred_counts.items())],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.report_out, flush=True)


if __name__ == "__main__":
    main()
