# V2 LLM-Assisted Classifier Evaluation

Standalone LLM predictions: `evaluation/v2_standalone_seq2seq_llm_predictions.csv`
Deterministic predictions: `evaluation/v2_gold_model_predictions.csv`
Evaluation rows: 123

| mode | answered rows | coverage | accuracy | macro F1 |
| --- | ---: | ---: | ---: | ---: |
| deterministic_ensemble_on_llm_test | 123 | 1.000 | 0.683 | 0.654 |
| tfidf_linear_svm_on_llm_test | 123 | 1.000 | 0.691 | 0.669 |
| standalone_llm_on_llm_test | 123 | 1.000 | 0.805 | 0.778 |
| llm_assisted_vote_on_llm_test | 123 | 1.000 | 0.691 | 0.659 |
| llm_deterministic_agreement_abstention | 84 | 0.683 | 0.893 | 0.844 |

The assisted vote combines the standalone LLM with the strongest deterministic classifier outputs. The agreement-abstention mode answers only when the deterministic ensemble and standalone LLM match.
