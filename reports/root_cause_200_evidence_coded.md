# 200-Row Root-Cause Evidence-Coded Extension

This extension increases root-cause coverage from the 50-row human-adjudicated subset to 200 evidence-coded rows. It does not relabel all 200 rows as human-adjudicated. The `provenance` column distinguishes the original human subset from linked-fix and issue-text evidence-coded rows.

Rows: 200
Human-adjudicated rows retained: 50

## Root-cause label counts

- compiler_backend_or_runtime: 21
- dtype_or_type_semantics: 47
- non_bug_feature_docs_support: 62
- numerical_algorithm_or_tolerance: 52
- performance_or_resource_behavior: 18

## Provenance counts

- evidence_coded_fix_text_or_same_repo_ref: 55
- evidence_coded_issue_text_no_linked_fix: 95
- human_adjudicated_50: 50

Conference-use guidance: report the 50-row subset as human-adjudicated and the 200-row file as evidence-coded root-cause supervision/provenance for future work.
