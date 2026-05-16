# Agentic Ensemble Abstention

This analysis evaluates an answer/abstain triage agent. It combines deterministic cross-validation predictions, the weak pre-classifier, a zero-shot local LLM, and a fold-safe RAG local LLM. Agreement is computed from predictions only; gold labels are used only for evaluation.

Full-coverage performance remains far below the requested 70-80% range, so the only honest way to reach that range is selective answering with explicit coverage.

| mode | answered | coverage | accuracy | macro F1 |
| --- | ---: | ---: | ---: | ---: |
| full_coverage_vote | 191 | 1.000 | 0.518 | 0.329 |
| vote_agreement_at_least_2 | 191 | 1.000 | 0.518 | 0.329 |
| vote_agreement_at_least_3 | 188 | 0.984 | 0.511 | 0.266 |
| vote_agreement_at_least_4 | 160 | 0.838 | 0.544 | 0.266 |
| vote_agreement_at_least_5 | 108 | 0.565 | 0.565 | 0.266 |
| vote_agreement_at_least_6 | 51 | 0.267 | 0.706 | 0.408 |
| vote_agreement_at_least_7 | 23 | 0.120 | 0.870 | 0.558 |
| vote_agreement_at_least_8 | 3 | 0.016 | 1.000 | 1.000 |
| candidate_and_tfidf_svm_agree | 55 | 0.288 | 0.891 | 0.560 |
| candidate_tfidf_svm_and_rag_llm_agree | 28 | 0.147 | 0.964 | 0.737 |
| candidate_tfidf_svm_and_zero_llm_agree | 16 | 0.084 | 1.000 | 1.000 |

Interpretation:

- The full-coverage vote is still not conference-strong as an accuracy model.
- A vote-agreement threshold of 6/8 reaches the requested 70% range but answers only 51/191 rows.
- The stricter candidate+TF-IDF-SVM+RAG-LLM agreement rule reaches higher accuracy but with lower coverage.
- These are selective triage results, not full automatic classification results.
