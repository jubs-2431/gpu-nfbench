from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evaluation" / "v2_standalone_seq2seq_llm_predictions.csv"
LOCAL = ROOT / "evaluation" / "v2_local_llama32_3b_predictions.csv"
OUT_PRED = ROOT / "evaluation" / "v2_local_llama32_3b_comparison.csv"
OUT_METRICS = ROOT / "tables" / "v2_local_llama32_3b_metrics.csv"
REPORT = ROOT / "reports" / "v2_local_llama32_3b_baseline.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prf(labels: list[str], preds: list[str]) -> tuple[float, float]:
    acc = sum(a == b for a, b in zip(labels, preds)) / len(labels) if labels else 0.0
    f1s = []
    for label in sorted(set(labels) | set(preds)):
        tp = sum(a == label and b == label for a, b in zip(labels, preds))
        fp = sum(a != label and b == label for a, b in zip(labels, preds))
        fn = sum(a == label and b != label for a, b in zip(labels, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return acc, sum(f1s) / len(f1s) if f1s else 0.0


def main() -> None:
    gold_rows = {row["id"]: row for row in read_csv(GOLD)}
    local_rows = read_csv(LOCAL)
    rows = []
    for local in local_rows:
        gold = gold_rows.get(local["blind_id"])
        if not gold:
            continue
        rows.append(
            {
                "id": local["blind_id"],
                "repository": gold["repository"],
                "gold_primary_failure": gold["gold_primary_failure"],
                "finetuned_flan_t5_prediction": gold["predicted_primary_failure"],
                "local_llama32_3b_prediction": local["primary_failure_label"],
                "local_llama32_3b_confidence": local["confidence"],
                "finetuned_correct": str(gold["gold_primary_failure"] == gold["predicted_primary_failure"]).lower(),
                "local_llama32_3b_correct": str(gold["gold_primary_failure"] == local["primary_failure_label"]).lower(),
            }
        )

    labels = [row["gold_primary_failure"] for row in rows]
    ft_preds = [row["finetuned_flan_t5_prediction"] for row in rows]
    local_preds = [row["local_llama32_3b_prediction"] for row in rows]
    ft_acc, ft_f1 = prf(labels, ft_preds)
    local_acc, local_f1 = prf(labels, local_preds)
    metrics = [
        {"model_or_mode": "fine_tuned_flan_t5_base", "rows": len(rows), "accuracy": f"{ft_acc:.3f}", "macro_f1": f"{ft_f1:.3f}"},
        {"model_or_mode": "local_llama3.2_3b_zero_shot", "rows": len(rows), "accuracy": f"{local_acc:.3f}", "macro_f1": f"{local_f1:.3f}"},
    ]
    write_csv(
        OUT_PRED,
        rows,
        [
            "id",
            "repository",
            "gold_primary_failure",
            "finetuned_flan_t5_prediction",
            "local_llama32_3b_prediction",
            "local_llama32_3b_confidence",
            "finetuned_correct",
            "local_llama32_3b_correct",
        ],
    )
    write_csv(OUT_METRICS, metrics, ["model_or_mode", "rows", "accuracy", "macro_f1"])
    REPORT.write_text(
        "\n".join(
            [
                "# V2 Local LLM Baseline",
                "",
                "This baseline evaluates a local-only zero-shot Ollama `llama3.2:3b` model on the same 123-row v2 held-out split used for the fine-tuned FLAN-T5 standalone model.",
                "",
                "| model/mode | rows | accuracy | macro F1 |",
                "| --- | ---: | ---: | ---: |",
                *[f"| {row['model_or_mode']} | {row['rows']} | {row['accuracy']} | {row['macro_f1']} |" for row in metrics],
                "",
                "A cloud-backed `qwen3.5:cloud` smoke test was not run because exporting held-out benchmark prompts to the cloud was blocked. The paper should present this as a local zero-shot LLM baseline, not as an external frontier API comparison.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
