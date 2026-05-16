# Validation Adjudication Report

Validation subset size: 82 public GitHub issues.

This pass uses full issue bodies and public comments fetched through the GitHub API. The labels are context-adjudicated research labels, not official project labels and not a substitute for the 191-issue blind human-adjudicated gold benchmark.

## Quality and agreement
| metric | value | share |
| --- | --- | --- |
| validation_issues | 82 | 100.0% |
| candidate_context_agreement | 44 | 53.7% |
| has_reproducer_or_code | 74 | 90.2% |
| has_stack_trace_or_error_log | 31 | 37.8% |
| has_fix_or_workaround_signal | 36 | 43.9% |
| closed_issues | 34 | 41.5% |

## Context primary-label distribution
| context_primary_failure | issues | share |
| --- | --- | --- |
| dtype_casting | 22 | 26.8% |
| performance_only | 15 | 18.3% |
| precision_tolerance | 12 | 14.6% |
| crash_compile | 10 | 12.2% |
| overflow_underflow | 9 | 11.0% |
| needs_review | 9 | 11.0% |
| nan_inf | 5 | 6.1% |

## Candidate primary-label distribution in validation subset
| candidate_primary_failure | issues | share |
| --- | --- | --- |
| dtype_casting | 12 | 14.6% |
| nan_inf | 12 | 14.6% |
| precision_tolerance | 12 | 14.6% |
| overflow_underflow | 12 | 14.6% |
| crash_compile | 12 | 14.6% |
| needs_review | 12 | 14.6% |
| performance_only | 10 | 12.2% |

## Confidence distribution
| context_confidence | issues | share |
| --- | --- | --- |
| high | 46 | 56.1% |
| medium | 27 | 32.9% |
| low | 9 | 11.0% |

## Representative candidate/context disagreements

- cupy/cupy#1166: candidate=dtype_casting, context=precision_tolerance, confidence=high; https://github.com/cupy/cupy/issues/1166
  - evidence: cupy.arange behavior is inconsistent with numpy when non-integer step is specified
- cupy/cupy#2351: candidate=nan_inf, context=overflow_underflow, confidence=high; https://github.com/cupy/cupy/issues/2351
  - evidence: sgesvd_bufferSize int32 overflow with CUDA 10.1
- cupy/cupy#6693: candidate=precision_tolerance, context=dtype_casting, confidence=high; https://github.com/cupy/cupy/issues/6693
  - evidence: [Tracker] Fix signature mismatch with NumPy/SciPy
- cupy/cupy#2248: candidate=overflow_underflow, context=precision_tolerance, confidence=medium; https://github.com/cupy/cupy/issues/2248
  - evidence: nsorflow-gpu have put effort into ensuring results match the cpu implementions (e.g. convolving with tensorflow or tensorflow-gpu or numpy produces the same values, even including precision errors)
- cupy/cupy#8260: candidate=crash_compile, context=needs_review, confidence=low; https://github.com/cupy/cupy/issues/8260
- cupy/cupy#2066: candidate=performance_only, context=needs_review, confidence=low; https://github.com/cupy/cupy/issues/2066
- cupy/cupy#1427: candidate=needs_review, context=dtype_casting, confidence=high; https://github.com/cupy/cupy/issues/1427
  - evidence: rray([3])) Traceback (most recent call last): File "<stdin>", line 1, in <module> File "cupy/random/sample.py", line 112, in randint return rs.randint(low, high, size, dtype) File "cupy/random/generator.py", line 473, in randint x = self.interval(diff, size).astype(dtype, copy=False) File "cupy/random/generator.py", line 210, in interval sample = cupy.empty((n,), dtype=dtype
- cupy/cupy#1989: candidate=needs_review, context=dtype_casting, confidence=high; https://github.com/cupy/cupy/issues/1989
  - evidence: pyx in cupy.core.core.ndarray.__richcmp__() cupy/core/_kernel.pyx in cupy.core._kernel.ufunc.__call__() cupy/core/_kernel.pyx in cupy.core._kernel._preprocess_args() TypeError: Unsupported type <class 'NoneType'> ```
- jax-ml/jax#10197: candidate=dtype_casting, context=performance_only, confidence=high; https://github.com/jax-ml/jax/issues/10197
  - evidence: Big performance discrepancy between JAX and TensorFlow with in-place updates
- jax-ml/jax#10255: candidate=nan_inf, context=precision_tolerance, confidence=high; https://github.com/jax-ml/jax/issues/10255
  - evidence: jax.scipy.sparse.linalg.cg inconsistent results between runs
- jax-ml/jax#30874: candidate=overflow_underflow, context=needs_review, confidence=low; https://github.com/jax-ml/jax/issues/30874
- jax-ml/jax#17912: candidate=crash_compile, context=needs_review, confidence=low; https://github.com/jax-ml/jax/issues/17912
