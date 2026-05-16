# Gold Baseline Classifier

This experiment evaluates deterministic text baselines against adjudicated gold labels.
No external ML packages are required; BM25 k-nearest-neighbor retrieval, TF-IDF, multinomial Naive Bayes, softmax regression, and one-vs-rest linear SVM are implemented in `scripts/gold_baseline_classifier.py`.

## Stratified 5-fold evaluation

| model | accuracy | macro_f1 |
| --- | ---: | ---: |
| majority_baseline | 0.382 | 0.079 |
| candidate_weak_label | 0.414 | 0.306 |
| bm25_knn | 0.356 | 0.185 |
| naive_bayes | 0.403 | 0.231 |
| tfidf_logistic | 0.487 | 0.243 |
| tfidf_linear_svm | 0.503 | 0.270 |
| bigram_tfidf_logistic | 0.471 | 0.228 |

## Leave-one-repository-out evaluation

| model | accuracy | macro_f1 |
| --- | ---: | ---: |
| candidate_weak_label | 0.414 | 0.306 |
| bm25_knn | 0.277 | 0.145 |
| naive_bayes | 0.393 | 0.228 |
| tfidf_logistic | 0.503 | 0.247 |
| tfidf_linear_svm | 0.461 | 0.241 |
| bigram_tfidf_logistic | 0.487 | 0.227 |

## Best 5-fold model per-class performance: `tfidf_linear_svm`

| label | support | precision | recall | f1 |
| --- | ---: | ---: | ---: | ---: |
| crash_compile | 4 | 0.000 | 0.000 | 0.000 |
| dtype_casting | 73 | 0.523 | 0.781 | 0.626 |
| nan_inf | 37 | 0.571 | 0.541 | 0.556 |
| not_numerical_failure | 22 | 0.182 | 0.091 | 0.121 |
| overflow_underflow | 37 | 0.500 | 0.432 | 0.464 |
| performance_only | 5 | 0.000 | 0.000 | 0.000 |
| precision_tolerance | 13 | 0.333 | 0.077 | 0.125 |

## Interpretation

- BM25 and TF-IDF models improve over the majority baseline but still struggle with rare classes.
- Leave-one-repository-out evaluation is harder than stratified folds because project vocabulary and issue templates shift across repositories.
- These results support the paper's claim that GPU numerical-failure triage needs full context and adjudicated labels, not only keyword matching.
