# Gold Expansion 1000 Taxonomy Repair

Source: `annotation/gold_expansion_1000_filled_from_downloads.csv`
Repaired file: `annotation/gold_expansion_1000_repaired.csv`
Change log: `annotation/gold_expansion_1000_taxonomy_repair_changes.csv`
Rows repaired: 180
Invalid labels remaining: 0

Repairs only map out-of-taxonomy primary labels into the existing seven-label benchmark taxonomy. Original labels are preserved in the change log and appended to row notes.

## Repair rules

- api_feature_request_to_not_numerical_failure: 44
- compiler_codegen_to_crash_compile: 10
- compiler_codegen_to_not_numerical_failure: 25
- compiler_codegen_to_performance_only: 7
- needs_review_to_crash_compile: 11
- needs_review_to_not_numerical_failure: 37
- needs_review_to_performance_only: 7
- performance_regression_to_performance_only: 39

## Repaired primary label counts

- crash_compile: 134
- dtype_casting: 132
- nan_inf: 137
- not_numerical_failure: 106
- overflow_underflow: 92
- performance_only: 53
- precision_tolerance: 346
