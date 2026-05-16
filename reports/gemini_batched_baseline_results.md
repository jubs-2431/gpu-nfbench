# Gemini Batched Baseline Results

Prediction file: `evaluation/llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv`

Model: `gemini-3.1-flash-lite`

Protocol: gold-hidden prompt packet, batched Gemini runner, temperature 0, 191/191 rows completed after repairing temporary 503 batches with smaller requests.

Results:

- Evaluated rows: 191
- Runner errors: 0
- Accuracy: 0.220
- Macro F1: 0.203
- High-confidence slice: 180 rows, 0.228 accuracy, 0.202 macro F1
- Medium-confidence slice: 11 rows, 0.091 accuracy, 0.083 macro F1

Interpretation:

The external Gemini baseline is valid as a reproducible comparison point, but it does not solve the benchmark as a direct classifier. It over-predicts `crash_compile` and `not_numerical_failure`, despite reporting high confidence on most rows.

As an ensemble signal, it gives a small but real improvement: fixed confidence-weighted voting with the deterministic model family reaches 0.539 full-coverage accuracy and 0.272 macro F1, compared with 0.529 and 0.267 for the deterministic-only ensemble. A selective deterministic+Gemini agreement mode reaches 0.880 accuracy on 25/191 answered rows. This supports the paper's claim that low-cost LLMs are most useful as agreement/calibration signals here, not as standalone full-coverage classifiers.
