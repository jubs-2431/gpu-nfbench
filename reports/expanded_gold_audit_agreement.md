# Expanded Gold Audit Agreement

Audit rows: 119
All Person A and Person B labels use the allowed seven-class taxonomy.

## Agreement summary

| comparison | observed agreement/accuracy | expected agreement | Cohen's kappa | macro F1 vs gold |
| --- | ---: | ---: | ---: | ---: |
| person_a_vs_person_b | 0.807 | 0.306 | 0.721 | n/a |
| person_a_vs_gold | 0.538 | 0.143 | 0.461 | 0.552 |
| person_b_vs_gold | 0.370 | 0.143 | 0.265 | 0.352 |

## Top Person A / Person B disagreements

| Person A | Person B | issues |
| --- | --- | ---: |
| performance_only | dtype_casting | 9 |
| crash_compile | precision_tolerance | 4 |
| dtype_casting | crash_compile | 3 |
| not_numerical_failure | precision_tolerance | 3 |
| not_numerical_failure | dtype_casting | 2 |
| crash_compile | dtype_casting | 1 |
| not_numerical_failure | performance_only | 1 |

## Interpretation

- Person A and Person B agreement is strong for an issue-report taxonomy with overlapping symptom and root-cause cues.
- Agreement against the existing expanded gold labels is lower than A/B agreement, indicating that the audit should be used to identify rows where the expanded gold label may need adjudication review.
- The current conference-safe claim is that a blind expanded audit achieved substantial inter-annotator agreement, while a separate adjudication pass is still needed before replacing existing expanded gold labels.
