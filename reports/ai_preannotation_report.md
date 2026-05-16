# AI Preannotation Report

These files are AI-generated prelabels. They are not human annotations and must not be reported as an independent human-labeled gold benchmark.

Rows pre-labeled: 191
Pass A/B disagreements: 10

## Pass A context-only labels
| label | issues |
| --- | --- |
| dtype_casting | 65 |
| performance_only | 25 |
| precision_tolerance | 25 |
| crash_compile | 22 |
| needs_review | 17 |
| nan_inf | 16 |
| overflow_underflow | 13 |
| not_numerical_failure | 8 |

## Pass B candidate-aware labels
| label | issues |
| --- | --- |
| dtype_casting | 65 |
| crash_compile | 27 |
| performance_only | 26 |
| precision_tolerance | 25 |
| nan_inf | 17 |
| overflow_underflow | 16 |
| not_numerical_failure | 8 |
| needs_review | 7 |

## Outputs

- `annotation/ai_prelabel_pass_A_context_only.csv`
- `annotation/ai_prelabel_pass_B_candidate_aware.csv`
- `annotation/ai_prelabel_disagreements.csv`
