# GPU-NFBench Annotation Guide

This folder contains the blind annotation materials used to create the
human-adjudicated gold benchmark. Two human annotators completed
`annotator_A_blind.csv` and `annotator_B_blind.csv` independently, candidate
weak labels were hidden during annotation, disagreements were adjudicated in
`adjudication_template.csv`, and agreement metrics were computed with
`scripts/evaluate_gold_labels.py`.

## Allowed Primary Labels

- `nan_inf`
- `overflow_underflow`
- `precision_tolerance`
- `dtype_casting`
- `crash_compile`
- `performance_only`
- `not_numerical_failure`
- `needs_review`

## Allowed Secondary Cause Labels

Use pipe-separated labels when multiple causes apply.

- `memory_mask_bounds`
- `compiler_codegen`
- `async_race_ordering`
- `hardware_backend`
- `reduction_accumulation`
- `api_semantics`
- `environment_configuration`
- `unknown`

## Annotation Rules

1. Read the title, issue body excerpt, comments excerpt, and source URL if needed.
2. Assign exactly one `primary_failure_label`.
3. Set `is_true_numerical_failure` to `yes`, `no`, or `unclear`.
4. Add one or more secondary cause labels if the issue text supports them.
5. Provide a short evidence quote from the public issue or comment text.
6. Use `needs_review` when the issue is too ambiguous to classify from public evidence.
7. Use `not_numerical_failure` when the query matched numerical words but the issue is not actually a numerical correctness failure.
8. Do not look at `candidate_label_suggestions_hidden_from_annotators.csv` while annotating.

## Boundary Rules Added After Adjudication

- Prefer `dtype_casting` over `precision_tolerance` when the evidence names
  dtype, casting, promotion, or low-precision format semantics; prefer
  `precision_tolerance` when dtype is incidental and the issue is primarily a
  tolerance/reference mismatch.
- Prefer `overflow_underflow` when the observed failure is range blow-up,
  saturation, integer wraparound, or underflow; retain dtype details as
  secondary cause/context when narrowing or promotion explains it.
- Prefer `nan_inf` when non-finite values are the observed symptom; prefer
  `precision_tolerance` when NaN/Inf appears only in tests, masks, or tolerance
  text.
- Prefer `crash_compile` when the user-visible failure is a compiler/runtime
  exception; use `dtype_casting` only when type semantics are the central
  failure.
- Use `performance_only` when the issue is primarily speed/throughput but still
  concerns numerical kernels; use `not_numerical_failure` for search false
  positives with no numerical correctness or performance task.

## Gold Release Rule

A row becomes gold only after the adjudication file has:

- `gold_primary_failure`
- `gold_secondary_cause_labels_pipe_separated`
- `gold_is_true_numerical_failure`
- `gold_evidence_quote`
- `adjudicator_id`

Run `python3 scripts/evaluate_gold_labels.py` from the project root after both
blind files and the adjudication file are complete.
