# Expanded Gold Model Training

Expanded gold rows: 1191
Source split: {'original_191': 191, 'expanded_1000': 1000}
Best full-coverage model by macro F1: `expanded_gold_vote_ensemble`

## Label counts

- crash_compile: 138
- dtype_casting: 205
- nan_inf: 174
- not_numerical_failure: 128
- overflow_underflow: 129
- performance_only: 58
- precision_tolerance: 359

## Full-coverage metrics

| model/mode | accuracy | macro F1 |
| --- | ---: | ---: |
| majority_baseline | 0.301 | 0.066 |
| candidate_weak_label | 0.537 | 0.447 |
| bm25_knn | 0.589 | 0.580 |
| naive_bayes | 0.646 | 0.584 |
| tfidf_logistic | 0.728 | 0.707 |
| tfidf_linear_svm | 0.765 | 0.751 |
| bigram_tfidf_logistic | 0.762 | 0.751 |
| expanded_gold_vote_ensemble | 0.768 | 0.753 |
| binary_gate_tfidf_linear_svm | 0.940 | 0.874 |

## Abstention metrics

| mode | answered | coverage | accuracy | macro F1 |
| --- | ---: | ---: | ---: | ---: |
| ensemble_vote_at_least_2 | 1191 | 1.000 | 0.768 | 0.753 |
| ensemble_vote_at_least_3 | 1110 | 0.932 | 0.792 | 0.778 |
| ensemble_vote_at_least_4 | 885 | 0.743 | 0.859 | 0.834 |
| ensemble_vote_at_least_5 | 428 | 0.359 | 0.953 | 0.811 |

## Interpretation

- The expanded 1,191-row gold set is now large enough to train and evaluate stronger deterministic triage baselines.
- The binary gate is reported separately because it measures the easier first-stage decision: numerical failure versus non-numerical/performance-only issue.
- The abstention rows show what accuracy is achievable when the model answers only high-agreement cases.
