# Expanded Gold Model Training

Expanded gold rows: 1191
Source split: {'original_191': 191, 'expanded_1000': 1000}
Best full-coverage model by macro F1: `expanded_gold_vote_ensemble`

## Label counts

- crash_compile: 192
- dtype_casting: 193
- nan_inf: 143
- not_numerical_failure: 147
- overflow_underflow: 94
- performance_only: 54
- precision_tolerance: 368

## Full-coverage metrics

| model/mode | accuracy | macro F1 |
| --- | ---: | ---: |
| majority_baseline | 0.309 | 0.067 |
| candidate_weak_label | 0.505 | 0.426 |
| bm25_knn | 0.571 | 0.534 |
| naive_bayes | 0.610 | 0.501 |
| tfidf_logistic | 0.715 | 0.688 |
| tfidf_linear_svm | 0.732 | 0.703 |
| bigram_tfidf_logistic | 0.724 | 0.694 |
| expanded_gold_vote_ensemble | 0.738 | 0.712 |
| binary_gate_tfidf_linear_svm | 0.892 | 0.796 |

## Abstention metrics

| mode | answered | coverage | accuracy | macro F1 |
| --- | ---: | ---: | ---: | ---: |
| ensemble_vote_at_least_2 | 1190 | 0.999 | 0.739 | 0.713 |
| ensemble_vote_at_least_3 | 1103 | 0.926 | 0.769 | 0.744 |
| ensemble_vote_at_least_4 | 875 | 0.735 | 0.842 | 0.803 |
| ensemble_vote_at_least_5 | 358 | 0.301 | 0.939 | 0.733 |

## Interpretation

- The expanded 1,191-row gold set is now large enough to train and evaluate stronger deterministic triage baselines.
- The binary gate is reported separately because it measures the easier first-stage decision: numerical failure versus non-numerical/performance-only issue.
- The abstention rows show what accuracy is achievable when the model answers only high-agreement cases.
