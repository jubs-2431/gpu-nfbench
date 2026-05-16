# Gold Expansion 1000 Plan

Candidate pool after removing existing gold URLs: 1730
Rows selected for human labeling: 1000
Review queue: `annotation/gold_expansion_1000_queue.csv`
Blind annotation queue: `annotation/gold_expansion_1000_blind.csv`

The review queue includes model and weak-label suggestions for project planning. The blind queue hides these fields and is the file to send to annotators if preserving independent human labels.

## Candidate label distribution

- nan_inf: 160
- overflow_underflow: 160
- precision_tolerance: 241
- dtype_casting: 160
- crash_compile: 86
- performance_only: 33
- not_numerical_failure: 0
- needs_review: 160

## Selection reasons

- label_balance_quota: 1000
- cross_repository_generalization: 765
- rare_or_undercovered_candidate_label: 360
- model_candidate_disagreement: 153
- low_model_confidence: 14

## Top repositories

- jax-ml/jax: 271
- pytorch/pytorch: 263
- rapidsai/cuml: 134
- triton-lang/triton: 114
- rapidsai/cudf: 97
- cupy/cupy: 49
- numba/numba: 36
- apache/tvm: 36