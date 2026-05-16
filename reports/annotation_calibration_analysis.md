# Annotation Calibration Analysis

The original blind agreement is reported without modification. This analysis adds defensible ways to interpret and improve the low-kappa issue without rewriting completed human labels.

## Agreement subsets

| subset | rows | observed_agreement | cohen_kappa |
| --- | ---: | ---: | ---: |
| all_rows | 191 | 0.372 | 0.280 |
| both_high_confidence | 29 | 0.759 | 0.624 |
| both_not_low_confidence | 175 | 0.389 | 0.296 |
| true_failure_status_agrees | 113 | 0.593 | 0.531 |
| primary_label_agrees | 71 | 1.000 | 1.000 |

## Round-2 calibration packet

- Blind relabeling packet: `annotation/calibration_round2_blind.csv`
- Adjudicator/training review packet: `annotation/calibration_round2_review.csv`
- Rows selected for round 2: 125

Recommended use: annotators first review a small training subset with adjudicated explanations, then blindly relabel the remaining calibration packet. A new kappa should be reported as post-calibration agreement, while the original kappa remains the primary unbiased blind-agreement measurement.
