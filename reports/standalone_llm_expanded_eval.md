# Standalone LLM Expanded Evaluation

Model: `gpu-nfbench-triage`
Training/RAG rows: 951
Held-out test rows: 25
Predictions: `evaluation/standalone_llm_expanded_predictions_smoke25.csv`

| mode | answered | coverage | accuracy | macro F1 |
| --- | ---: | ---: | ---: | ---: |
| standalone_llm_holdout_full_coverage | 25 | 1.000 | 0.520 | 0.395 |
| standalone_llm_holdout_excluding_needs_review | 25 | 1.000 | 0.520 | 0.395 |

## Gold label counts in evaluated rows

- crash_compile: 5
- dtype_casting: 1
- nan_inf: 1
- not_numerical_failure: 2
- overflow_underflow: 1
- performance_only: 2
- precision_tolerance: 13

## Predicted label counts

- crash_compile: 2
- dtype_casting: 4
- nan_inf: 3
- not_numerical_failure: 1
- overflow_underflow: 1
- performance_only: 1
- precision_tolerance: 13
