# Gold Benchmark Analysis

Gold benchmark size: 191 adjudicated issues.

## Gold primary-label distribution
| gold_primary_failure | issues | share |
| --- | --- | --- |
| dtype_casting | 73 | 38.2% |
| overflow_underflow | 37 | 19.4% |
| nan_inf | 37 | 19.4% |
| not_numerical_failure | 22 | 11.5% |
| precision_tolerance | 13 | 6.8% |
| performance_only | 5 | 2.6% |
| crash_compile | 4 | 2.1% |

## True numerical-failure status
| gold_is_true_numerical_failure | issues | share |
| --- | --- | --- |
| yes | 153 | 80.1% |
| no | 27 | 14.1% |
| unclear | 11 | 5.8% |

## Repository distribution
| repository | issues | share |
| --- | --- | --- |
| triton-lang/triton | 35 | 18.3% |
| rapidsai/cudf | 35 | 18.3% |
| cupy/cupy | 33 | 17.3% |
| pytorch/pytorch | 31 | 16.2% |
| numba/numba | 29 | 15.2% |
| jax-ml/jax | 28 | 14.7% |

## Gold secondary-cause distribution
| gold_secondary_cause | issues | share |
| --- | --- | --- |
| hardware_backend | 175 | 91.6% |
| memory_mask_bounds | 145 | 75.9% |
| compiler_codegen | 141 | 73.8% |
| async_race_ordering | 101 | 52.9% |
| api_semantics | 76 | 39.8% |
| reduction_accumulation | 61 | 31.9% |
| environment_configuration | 27 | 14.1% |

## Drift from weak labels

- Candidate weak label matched adjudicated gold label for 79/191 issues (41.4%).
- Annotator A matched adjudicated gold label for 93/191 issues (48.7%).
- Annotator B matched adjudicated gold label for 71/191 issues (37.2%).

## Main benchmark caveats

- The gold set is intentionally challenging and class-imbalanced after adjudication.
- Crash/compile and performance-only are small classes in the final gold distribution.
- Low annotator agreement should be reported as a finding: GPU numerical issue labeling is genuinely ambiguous without adjudication.
