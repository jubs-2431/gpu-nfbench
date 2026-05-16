# Standalone Seq2Seq LLM Training

Base model: `llm/models/gpu_nfbench_flan_t5_base_label_balanced_e3`
Saved model: `llm/models/gpu_nfbench_v2_flan_t5_base_from_v1_e3`
Train rows: 2058
Validation rows: 123
Test rows: 123
Epochs: 1
Batch size: 2
Device: `cpu`
Target format: `label`

| split | rows | accuracy | macro F1 |
| --- | ---: | ---: | ---: |
| validation | 123 | 0.748 | 0.735 |
| test | 123 | 0.805 | 0.778 |

## Test Gold Counts

- crash_compile: 20
- dtype_casting: 20
- nan_inf: 15
- not_numerical_failure: 15
- overflow_underflow: 10
- performance_only: 6
- precision_tolerance: 37

## Test Prediction Counts

- crash_compile: 24
- dtype_casting: 19
- nan_inf: 18
- not_numerical_failure: 11
- overflow_underflow: 9
- performance_only: 7
- precision_tolerance: 35
