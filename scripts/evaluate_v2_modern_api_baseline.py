from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_LLM_TEST = ROOT / "evaluation" / "v2_standalone_seq2seq_llm_predictions.csv"
API_PREDICTIONS = ROOT / "evaluation" / "llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv"
OUT_PREDICTIONS = ROOT / "evaluation" / "v2_modern_api_baseline_on_llm_test.csv"
OUT_METRICS = ROOT / "tables" / "v2_modern_api_baseline_metrics.csv"
REPORT = ROOT / "reports" / "v2_modern_api_baseline_comparison.md"


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
    accuracy = sum(a == b for a, b in zip(labels, preds)) / len(labels) if labels else 0.0
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing modern API LLM predictions on the v2 held-out LLM test split.")
    parser.add_argument("--api-predictions", type=Path, default=API_PREDICTIONS)
    parser.add_argument("--v2-llm-test", type=Path, default=V2_LLM_TEST)
    args = parser.parse_args()

    test_rows = read_csv(args.v2_llm_test)
    api_by_id = {}
    for row in read_csv(args.api_predictions):
        blind_id = row.get("blind_id", "")
        if blind_id and blind_id not in api_by_id and not row.get("error"):
            api_by_id[blind_id] = row

    out_rows = []
    missing = []
    for row in test_rows:
        blind_id = row["id"]
        api = api_by_id.get(blind_id)
        if not api:
            missing.append(blind_id)
            continue
        out_rows.append(
            {
                "id": blind_id,
                "repository": row["repository"],
                "gold_primary_failure": row["gold_primary_failure"],
                "finetuned_flan_t5_prediction": row["predicted_primary_failure"],
                "modern_api_prediction": api["primary_failure_label"],
                "modern_api_confidence": api.get("confidence", ""),
                "modern_api_model": api.get("model", ""),
                "modern_api_provider": api.get("provider", ""),
                "finetuned_correct": str(row["gold_primary_failure"] == row["predicted_primary_failure"]).lower(),
                "modern_api_correct": str(row["gold_primary_failure"] == api["primary_failure_label"]).lower(),
            }
        )

    labels = [row["gold_primary_failure"] for row in out_rows]
    api_preds = [row["modern_api_prediction"] for row in out_rows]
    ft_preds = [row["finetuned_flan_t5_prediction"] for row in out_rows]
    api_acc, api_f1 = prf(labels, api_preds)
    ft_acc, ft_f1 = prf(labels, ft_preds)
    model = out_rows[0]["modern_api_model"] if out_rows else "unknown"
    provider = out_rows[0]["modern_api_provider"] if out_rows else "unknown"
    metrics = [
        {
            "model_or_mode": "fine_tuned_flan_t5_base",
            "evaluation_rows": len(out_rows),
            "accuracy": f"{ft_acc:.3f}",
            "macro_f1": f"{ft_f1:.3f}",
        },
        {
            "model_or_mode": f"{provider}_{model}",
            "evaluation_rows": len(out_rows),
            "accuracy": f"{api_acc:.3f}",
            "macro_f1": f"{api_f1:.3f}",
        },
    ]
    write_csv(
        OUT_PREDICTIONS,
        out_rows,
        [
            "id",
            "repository",
            "gold_primary_failure",
            "finetuned_flan_t5_prediction",
            "modern_api_prediction",
            "modern_api_confidence",
            "modern_api_model",
            "modern_api_provider",
            "finetuned_correct",
            "modern_api_correct",
        ],
    )
    write_csv(OUT_METRICS, metrics, ["model_or_mode", "evaluation_rows", "accuracy", "macro_f1"])
    REPORT.write_text(
        "\n".join(
            [
                "# V2 Modern API Baseline Comparison",
                "",
                f"Modern API prediction file: `{args.api_predictions.relative_to(ROOT)}`",
                f"Shared v2 held-out rows evaluated: {len(out_rows)}",
                f"Missing API predictions for held-out rows: {len(missing)}",
                "",
                "| model/mode | rows | accuracy | macro F1 |",
                "| --- | ---: | ---: | ---: |",
                *[f"| {row['model_or_mode']} | {row['evaluation_rows']} | {row['accuracy']} | {row['macro_f1']} |" for row in metrics],
                "",
                "These results compare the fine-tuned standalone FLAN-T5 classifier against an already-generated modern Gemini API baseline on the same v2 held-out split. No fresh API call was made in this run because no API key was present in the shell environment.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
