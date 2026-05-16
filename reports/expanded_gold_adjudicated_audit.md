# Expanded Gold Adjudicated Audit

Adjudicated audit rows: 119
Rows where adjudicated audit differs from existing expanded gold: 52
Candidate v2 benchmark written: `data/processed/gold_benchmark_expanded_adjudicated_v2.csv`

## Metrics

| comparison | accuracy/agreement | expected agreement | Cohen's kappa | macro F1 |
| --- | ---: | ---: | ---: | ---: |
| person_a_vs_adjudicated | 0.370 | 0.112 | 0.290 | 0.422 |
| person_b_vs_adjudicated | 0.336 | 0.114 | 0.251 | 0.365 |
| person_a_vs_person_b | 0.807 | 0.306 | 0.721 | 0.653 |
| existing_expanded_gold_vs_adjudicated | 0.563 | 0.143 | 0.490 | 0.561 |
| candidate_weak_label_vs_adjudicated | 0.471 | 0.131 | 0.390 | 0.438 |
| tfidf_linear_svm_vs_adjudicated | 0.529 | 0.155 | 0.443 | 0.510 |
| tfidf_logistic_vs_adjudicated | 0.563 | 0.158 | 0.481 | 0.541 |
| bigram_tfidf_logistic_vs_adjudicated | 0.546 | 0.162 | 0.458 | 0.519 |
| expanded_gold_vote_ensemble_vs_adjudicated | 0.555 | 0.158 | 0.471 | 0.532 |

## Existing gold vs adjudicated audit label counts on the audit subset

| label | existing gold | adjudicated audit |
| --- | ---: | ---: |
| crash_compile | 17 | 31 |
| dtype_casting | 17 | 9 |
| nan_inf | 17 | 8 |
| not_numerical_failure | 17 | 20 |
| overflow_underflow | 17 | 8 |
| performance_only | 17 | 12 |
| precision_tolerance | 17 | 31 |

## Interpretation

- The adjudicated audit provides a higher-quality external check on the expanded labels.
- Because many audit adjudications differ from the existing expanded gold labels, the paper should either report the audit as an external validation subset or retrain/evaluate on the v2 benchmark before making v2 the canonical dataset.
- Retraining the standalone LLM is only necessary if the v2 benchmark replaces the original expanded gold benchmark as the main dataset.
