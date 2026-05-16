# Gold Error Analysis

This report summarizes where weak labels and blind human annotations diverge from the adjudicated gold labels.

## Most common weak-label to gold-label corrections

| weak_candidate_label | gold_label | issues |
| --- | --- | ---: |
| precision_tolerance | dtype_casting | 25 |
| crash_compile | dtype_casting | 10 |
| needs_review | dtype_casting | 8 |
| needs_review | precision_tolerance | 8 |
| crash_compile | not_numerical_failure | 7 |
| needs_review | not_numerical_failure | 6 |
| performance_only | dtype_casting | 5 |
| needs_review | overflow_underflow | 5 |
| dtype_casting | crash_compile | 4 |
| crash_compile | precision_tolerance | 4 |
| crash_compile | overflow_underflow | 4 |
| performance_only | overflow_underflow | 4 |

## Representative gold examples

| gold_label | blind_id | repository | title | evidence_quote | why_in_benchmark |
| --- | --- | --- | --- | --- | --- |
| dtype_casting | GNF-0001-f56f80f3 | triton-lang/triton | add libdevice.remquo | dtype("fp32"), core | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| overflow_underflow | GNF-0005-80aeca70 | cupy/cupy | Wrong overflow with matmul of uint16 arrays | Wrong overflow with matmul of uint16 arrays ### Description The following arrays give an appar | Representative clear example for this gold class. |
| nan_inf | GNF-0010-ff390b39 | triton-lang/triton | Value 'sm_89' is not defined for option 'gpu-name' | il but training with the compiled model does fail (showing `nan` as a loss after the 1st iteration) | Weak search/title labels drifted after full issue/comment context and adjudication. |
| performance_only | GNF-0012-d3d20064 | numba/numba | Support For CPU Atomics | c/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent) requires atomic floating-point addition | Use performance_only when the issue is primarily speed/throughput but still discusses numerical kernels; use not_numerical_failure for search false positives with no correctness/performance numerical task. |
| crash_compile | GNF-0017-f27cd16b | numba/numba | DeviceNDArray.bind() does not seem to bind the stream to self | DeviceNDArray.bind() does not seem to bind the stream to self | Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure. |
| not_numerical_failure | GNF-0029-5f61e128 | rapidsai/cudf | [QST] TypeError: Argument 'real' has incorrect type (expected numpy.ndarray, got ndarray) | [QST] TypeError: Argument 'real' has incorrect type (expected numpy | Weak search/title labels drifted after full issue/comment context and adjudication. |
| precision_tolerance | GNF-0035-1d6522c2 | pytorch/pytorch | Return a view from tensor(requires_grad=False) in autograd function may cause incorrect requires_grad attribute. | view from tensor(requires_grad=False) in autograd function may cause incorrect requires_grad attribute | Weak search/title labels drifted after full issue/comment context and adjudication. |

## Refined adjudication rules

- **dtype_casting vs. precision_tolerance:** Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch.
- **dtype_casting vs. overflow_underflow:** Prefer overflow_underflow when the observed failure is range blow-up, saturation, integer wraparound, or underflow; retain dtype_casting as a cause when narrowing/promotion explains it.
- **nan_inf vs. precision_tolerance:** Prefer nan_inf when non-finite values are the observed symptom; prefer precision_tolerance when NaN/Inf appears only in tests, masks, or tolerance text.
- **crash_compile vs. dtype_casting:** Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure.
- **performance_only vs. not_numerical_failure:** Use performance_only when the issue is primarily speed/throughput but still discusses numerical kernels; use not_numerical_failure for search false positives with no correctness/performance numerical task.

## High-value disagreement examples

| blind_id | weak_label | annotator_a | annotator_b | gold_label | why_difficult |
| --- | --- | --- | --- | --- | --- |
| GNF-0001-f56f80f3 | precision_tolerance | dtype_casting | not_numerical_failure | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0003-eaa63fac | performance_only | precision_tolerance | needs_review | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0004-e5162a33 | dtype_casting | dtype_casting | crash_compile | dtype_casting | Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure. |
| GNF-0006-82eaa858 | precision_tolerance | dtype_casting | dtype_casting | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0007-5a435abf | precision_tolerance | precision_tolerance | dtype_casting | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0008-4625af98 | precision_tolerance | precision_tolerance | not_numerical_failure | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0009-f786b33b | precision_tolerance | dtype_casting | dtype_casting | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0010-ff390b39 | needs_review | nan_inf | nan_inf | nan_inf | Weak search/title labels drifted after full issue/comment context and adjudication. |
| GNF-0012-d3d20064 | performance_only | not_numerical_failure | performance_only | performance_only | Use performance_only when the issue is primarily speed/throughput but still discusses numerical kernels; use not_numerical_failure for search false positives with no correctness/performance numerical task. |
| GNF-0013-bd88dbb3 | nan_inf | nan_inf | needs_review | nan_inf | Blind annotators disagreed despite the weak label matching gold; adjudication resolved the boundary case. |
| GNF-0014-9fb698b0 | dtype_casting | performance_only | dtype_casting | dtype_casting | Blind annotators disagreed despite the weak label matching gold; adjudication resolved the boundary case. |
| GNF-0015-04273d95 | precision_tolerance | nan_inf | nan_inf | nan_inf | Prefer nan_inf when non-finite values are the observed symptom; prefer precision_tolerance when NaN/Inf appears only in tests, masks, or tolerance text. |
| GNF-0017-f27cd16b | dtype_casting | needs_review | needs_review | crash_compile | Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure. |
| GNF-0018-86e41701 | crash_compile | performance_only | performance_only | dtype_casting | Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure. |
| GNF-0020-322072f7 | crash_compile | crash_compile | crash_compile | dtype_casting | Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure. |
| GNF-0023-93595bc9 | dtype_casting | dtype_casting | crash_compile | crash_compile | Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure. |
| GNF-0027-933bf558 | precision_tolerance | not_numerical_failure | not_numerical_failure | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0029-5f61e128 | crash_compile | precision_tolerance | crash_compile | not_numerical_failure | Weak search/title labels drifted after full issue/comment context and adjudication. |
| GNF-0030-a4314978 | nan_inf | precision_tolerance | nan_inf | nan_inf | Prefer nan_inf when non-finite values are the observed symptom; prefer precision_tolerance when NaN/Inf appears only in tests, masks, or tolerance text. |
| GNF-0031-dff9514d | precision_tolerance | not_numerical_failure | performance_only | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0032-76c19db8 | precision_tolerance | dtype_casting | dtype_casting | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0033-1f6d8975 | precision_tolerance | precision_tolerance | dtype_casting | dtype_casting | Prefer dtype_casting when the evidence names dtype, casting, promotion, or low-precision format semantics; prefer precision_tolerance when the dtype is incidental and the issue is primarily a tolerance/reference mismatch. |
| GNF-0034-77cd048a | needs_review | dtype_casting | crash_compile | dtype_casting | Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure. |
| GNF-0035-1d6522c2 | crash_compile | precision_tolerance | not_numerical_failure | precision_tolerance | Weak search/title labels drifted after full issue/comment context and adjudication. |
| GNF-0036-b3e8f03a | dtype_casting | dtype_casting | crash_compile | dtype_casting | Prefer crash_compile when the user-visible failure is a compiler/runtime exception; use dtype_casting only when type semantics are the central failure. |
