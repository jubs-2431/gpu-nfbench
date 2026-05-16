# Rare-Class Expansion Plan

The adjudicated benchmark has low support for `crash_compile` and `performance_only`. This file selects additional seed-dataset candidates for a future human-labeled expansion. These rows are not gold labels.

| target_label | candidates |
| --- | ---: |
| crash_compile | 9 |
| performance_only | 6 |

Candidate packet: `annotation/rare_class_expansion_candidates.csv`

Required next step before using these rows as benchmark evidence: fetch full context, blind-label with two human annotators, adjudicate disagreements, and rerun agreement metrics.
