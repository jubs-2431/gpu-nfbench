# Full-Coverage Model Improvement Experiments

This report implements the non-external-model recommendations: no `needs_review` at full coverage, two-stage classification, candidate-label features, linked PR/diff/path features, and a no-abstention ensemble.

The main result is that these changes improve full-coverage accuracy modestly, but they do not reach 70-80% on the full 191-row benchmark.

Best full-coverage mode: `full_coverage_no_needs_review_deterministic_ensemble` at 0.529 accuracy and 0.267 macro F1.

| model or mode | accuracy | macro F1 | notes |
| --- | ---: | ---: | --- |
| base_text_bm25_knn | 0.356 | 0.185 | Flat classifier. |
| base_text_naive_bayes | 0.403 | 0.231 | Flat classifier. |
| base_text_tfidf_logistic | 0.487 | 0.243 | Flat classifier. |
| base_text_tfidf_linear_svm | 0.503 | 0.270 | Flat classifier. |
| base_text_bigram_tfidf_logistic | 0.471 | 0.228 | Flat classifier. |
| candidate_features_bm25_knn | 0.366 | 0.184 | Flat classifier. |
| candidate_features_naive_bayes | 0.429 | 0.247 | Flat classifier. |
| candidate_features_tfidf_logistic | 0.518 | 0.262 | Flat classifier. |
| candidate_features_tfidf_linear_svm | 0.492 | 0.253 | Flat classifier. |
| candidate_features_bigram_tfidf_logistic | 0.492 | 0.242 | Flat classifier. |
| diff_features_bm25_knn | 0.351 | 0.174 | Flat classifier. |
| diff_features_naive_bayes | 0.424 | 0.229 | Flat classifier. |
| diff_features_tfidf_logistic | 0.482 | 0.233 | Flat classifier. |
| diff_features_tfidf_linear_svm | 0.487 | 0.242 | Flat classifier. |
| diff_features_bigram_tfidf_logistic | 0.461 | 0.224 | Flat classifier. |
| candidate_plus_diff_features_bm25_knn | 0.366 | 0.185 | Flat classifier. |
| candidate_plus_diff_features_naive_bayes | 0.429 | 0.234 | Flat classifier. |
| candidate_plus_diff_features_tfidf_logistic | 0.508 | 0.249 | Flat classifier. |
| candidate_plus_diff_features_tfidf_linear_svm | 0.513 | 0.257 | Flat classifier. |
| candidate_plus_diff_features_bigram_tfidf_logistic | 0.492 | 0.244 | Flat classifier. |
| weak_candidate_no_needs_review_fallback | 0.466 | 0.355 | Replaces needs_review with augmented TF-IDF SVM prediction. |
| two_stage_hierarchical_augmented_tfidf_svm | 0.503 | 0.241 | Stage 1 predicts numeric/performance-or-not/crash group, stage 2 predicts primary label. |
| full_coverage_no_needs_review_deterministic_ensemble | 0.529 | 0.267 | Vote over no-needs weak label and augmented deterministic models only. |
| full_coverage_no_needs_review_with_local_llm_ensemble | 0.518 | 0.266 | Vote over no-needs weak label, augmented deterministic models, and local LLM outputs. |

Interpretation:

- Removing `needs_review` is necessary for full coverage, but it does not by itself solve the task.
- Candidate-label and linked-diff/path features provide a small lift over the original text-only baseline.
- The two-stage hierarchy is more interpretable but did not outperform the best flat augmented model on this small gold set.
- More adjudicated labels and a stronger external model are the likely bottlenecks for reaching 70% full coverage.
