# Hierarchical Abstaining Classifier

This experiment separates numerical-failure detection from symptom classification. The first-stage gate predicts numerical_failure versus non_numerical_or_performance, then the second stage votes among primary failure labels. Abstaining modes answer only when enough model votes agree.

| mode | answered | coverage | accuracy | macro F1 |
| --- | ---: | ---: | ---: | ---: |
| hierarchical_full_coverage | 191 | 1.000 | 0.497 | 0.248 |
| hierarchical_vote_at_least_2 | 191 | 1.000 | 0.497 | 0.248 |
| hierarchical_vote_at_least_3 | 178 | 0.932 | 0.517 | 0.240 |
| hierarchical_vote_at_least_4 | 120 | 0.628 | 0.567 | 0.273 |
| hierarchical_vote_at_least_5 | 38 | 0.199 | 0.842 | 0.545 |
| hierarchical_gate_match_and_vote_at_least_3 | 177 | 0.927 | 0.520 | 0.241 |
| binary_gate_only | 191 | 1.000 | 0.822 | 0.451 |

Interpretation:

- The binary gate tests whether the easier first question can be solved reliably before multiclass labeling.
- The selective rows are the candidates suitable for automated triage; unanswered rows remain human-review cases.
- These metrics are still evaluated only on existing gold labels; accuracy should be rerun after the 1000-row gold expansion is completed.