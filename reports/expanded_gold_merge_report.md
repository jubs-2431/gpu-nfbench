# Expanded Gold Merge Report

Original gold rows: 191
Completed expansion rows merged: 1000
Expanded gold rows written: 1191
Skipped incomplete rows: 0
Skipped duplicate URLs: 0
Output: `data/processed/gold_benchmark_expanded.csv`

A row is merged only if it has a valid primary label, high/medium/low confidence, yes/no/unclear true-failure value, and a nonempty evidence quote.

## Expanded primary label counts

- crash_compile: 138
- dtype_casting: 205
- nan_inf: 174
- not_numerical_failure: 128
- overflow_underflow: 129
- performance_only: 58
- precision_tolerance: 359