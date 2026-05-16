# Standalone Seq2Seq LLM Training

Base model: `llm/models/gpu_nfbench_flan_t5_small_label_balanced`
Saved model: `llm/models/gpu_nfbench_v2_flan_t5_small_smoke`
Train rows: 784
Validation rows: 60
Test rows: 60
Epochs: 1
Batch size: 1
Device: `cpu`
Target format: `label`

| split | rows | accuracy | macro F1 |
| --- | ---: | ---: | ---: |
| validation | 60 | 0.750 | 0.703 |
| test | 60 | 0.700 | 0.646 |

## Test Gold Counts

- crash_compile: 11
- dtype_casting: 5
- nan_inf: 10
- not_numerical_failure: 5
- overflow_underflow: 4
- performance_only: 4
- precision_tolerance: 21

## Test Prediction Counts

- crash_compile: 13
- dtype_casting: 5
- nan_inf: 12
- not_numerical_failure: 3
- overflow_underflow: 5
- performance_only: 7
- precision_tolerance: 15
