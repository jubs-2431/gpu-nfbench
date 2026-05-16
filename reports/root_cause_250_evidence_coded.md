# 250-Row Root-Cause Evidence-Coded Extension

This extension increases root-cause coverage from the 50-row human-adjudicated subset to 250 evidence-coded rows. It does not relabel all 250 rows as human-adjudicated. The `provenance` column distinguishes the original human subset from linked-fix and issue-text evidence-coded rows.

Rows: 250
Human-adjudicated rows retained: 50

## Root-cause label counts

- compiler_backend_or_runtime: 30
- dtype_or_type_semantics: 50
- non_bug_feature_docs_support: 69
- numerical_algorithm_or_tolerance: 74
- performance_or_resource_behavior: 27

## Provenance counts

- evidence_coded_fix_text_or_same_repo_ref: 55
- evidence_coded_issue_text_no_linked_fix: 145
- human_adjudicated_50: 50

Conference-use guidance: report the 50-row subset as human-adjudicated and the 250-row file as evidence-coded root-cause supervision/provenance for future work.
