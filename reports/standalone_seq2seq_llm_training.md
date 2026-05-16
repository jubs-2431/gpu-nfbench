# Standalone Seq2Seq LLM Training

Base model: `llm/models/gpu_nfbench_flan_t5_base_label_balanced`
Saved model: `llm/models/gpu_nfbench_flan_t5_base_label_balanced_e3`
Train rows: 2009
Validation rows: 121
Test rows: 121
Epochs: 1
Batch size: 2
Device: `mps`
Target format: `label`

| split | rows | accuracy | macro F1 |
| --- | ---: | ---: | ---: |
| validation | 121 | 0.744 | 0.721 |
| test | 121 | 0.686 | 0.640 |

## Test Gold Counts

- crash_compile: 14
- dtype_casting: 21
- nan_inf: 18
- not_numerical_failure: 13
- overflow_underflow: 13
- performance_only: 6
- precision_tolerance: 36

## Test Prediction Counts

- crash_compile: 14
- dtype_casting: 18
- nan_inf: 17
- not_numerical_failure: 18
- overflow_underflow: 12
- performance_only: 8
- precision_tolerance: 34
