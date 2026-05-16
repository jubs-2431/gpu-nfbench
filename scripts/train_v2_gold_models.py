from __future__ import annotations

from pathlib import Path

import train_expanded_gold_models as egm


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    egm.EXPANDED_GOLD = ROOT / "data" / "processed" / "gold_benchmark_expanded_v2_canonical.csv"
    egm.OUT_METRICS = ROOT / "tables" / "v2_gold_classifier_metrics.csv"
    egm.OUT_PER_CLASS = ROOT / "tables" / "v2_gold_classifier_per_class.csv"
    egm.OUT_CONFUSION = ROOT / "tables" / "v2_gold_classifier_confusion.csv"
    egm.OUT_ABSTAIN = ROOT / "tables" / "v2_gold_abstention_metrics.csv"
    egm.OUT_PREDICTIONS = ROOT / "evaluation" / "v2_gold_model_predictions.csv"
    egm.REPORT = ROOT / "reports" / "v2_gold_model_training.md"
    egm.main()


if __name__ == "__main__":
    main()
