# Gold Expansion 1000 Filled Validation

Source: `/Users/aryanshah/Downloads/gold_expansion_1000_filled.csv`
Copied to: `annotation/gold_expansion_1000_filled_from_downloads.csv`
Rows: 1000
Merge-ready rows under current taxonomy: 820
Rows needing taxonomy fix: 180
Rows with blank secondary cause labels: 190

## Primary label counts

- precision_tolerance: 346
- nan_inf: 137
- dtype_casting: 132
- crash_compile: 113
- overflow_underflow: 92
- needs_review: 55
- api_feature_request: 44
- compiler_codegen: 42
- performance_regression: 39

## True numerical failure counts

- yes: 613
- no: 387

## Confidence counts

- high: 606
- medium: 339
- low: 55

## Required fixes

- Replace `needs_review` with one of the seven allowed primary labels.
- Replace `api_feature_request` with `not_numerical_failure` unless the issue reports an actual numerical failure.
- Replace `performance_regression` with `performance_only` unless the issue reports numerical correctness failure.
- Move `compiler_codegen` to `secondary_cause_labels`; choose a primary symptom such as `crash_compile`, `precision_tolerance`, `dtype_casting`, or `not_numerical_failure`.

Rows to fix: `annotation/gold_expansion_1000_rows_needing_taxonomy_fix.csv`
