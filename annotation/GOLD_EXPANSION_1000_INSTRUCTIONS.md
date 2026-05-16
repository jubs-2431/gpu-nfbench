# GPU-NFBench 1000-Row Gold Expansion Instructions

Fill this file:

`annotation/gold_expansion_1000_blind.csv`

Do not fill labels directly from model suggestions. The suggestions file is only for later audit and triage:

`annotation/gold_expansion_1000_model_suggestions.csv`

## Required Fields

For each row, fill:

- `primary_failure_label`
- `secondary_cause_labels`
- `is_true_numerical_failure`
- `confidence`
- `evidence_quote`
- `notes`

## Primary Labels

Use exactly one:

- `nan_inf`
- `overflow_underflow`
- `precision_tolerance`
- `dtype_casting`
- `crash_compile`
- `performance_only`
- `not_numerical_failure`

Do not use `needs_review` in the final gold file. If unclear, choose the best label, set `confidence` to `low`, and explain the ambiguity in `notes`.

## Secondary Cause Labels

Use pipe-separated labels when multiple apply:

- `memory_mask_bounds`
- `compiler_codegen`
- `async_race_ordering`
- `hardware_backend`
- `reduction_accumulation`
- `api_semantics`
- `environment_configuration`
- `unknown`

Example:

`compiler_codegen|hardware_backend`

## True Numerical Failure

Use exactly one:

- `yes`
- `no`
- `unclear`

`performance_only` and `not_numerical_failure` are usually `no`, unless the issue includes a real correctness failure in addition to the non-correctness discussion.

## Evidence Quote

Paste a short quote from the issue body that justifies the label. The merge script requires this field to be nonempty before a row can enter the expanded gold benchmark.

## Merge After Annotation

After the file is completed, run:

```bash
python3 scripts/merge_completed_gold_expansion.py
```

Then retrain/evaluate:

```bash
python3 scripts/gold_baseline_classifier.py
python3 scripts/full_coverage_model_improvements.py
python3 scripts/hierarchical_abstaining_classifier.py
python3 scripts/agentic_ensemble_abstention.py
```
