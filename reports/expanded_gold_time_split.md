# Expanded Gold Chronological Split

Rows with creation timestamps: 1191
Train rows: 952
Test rows: 239
Train end: 2025-12-17T07:22:19Z
Test start: 2025-12-17T20:26:11Z

## Test label distribution

- crash_compile: 19
- dtype_casting: 57
- nan_inf: 38
- not_numerical_failure: 13
- overflow_underflow: 17
- performance_only: 11
- precision_tolerance: 84

## Metrics

| model/mode | accuracy | macro F1 |
| --- | ---: | ---: |
| candidate_weak_label | 0.515 | 0.348 |
| bm25_knn | 0.490 | 0.452 |
| naive_bayes | 0.636 | 0.496 |
| tfidf_logistic | 0.745 | 0.710 |
| tfidf_linear_svm | 0.795 | 0.763 |
| bigram_tfidf_logistic | 0.774 | 0.745 |
| expanded_gold_vote_ensemble | 0.812 | 0.788 |
