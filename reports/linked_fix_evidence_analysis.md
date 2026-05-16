# Linked Fix Evidence Analysis

This analysis mines the already-fetched public issue/comment context for linked pull requests, same-repository fix references, and inline patch/diff snippets. It is source-backed but does not claim full code-diff root-cause adjudication.

| signal | issues |
| --- | ---: |
| explicit pull-request URL | 39 |
| same-repository fix/PR reference pattern | 9 |
| inline diff or patch snippet in issue context | 9 |
| fix/root-cause/workaround text snippet | 96 |
| rows with at least one linked-fix evidence signal | 105 |

Generated tables:

- `tables/linked_fix_evidence_subset.csv`
- `tables/linked_fix_evidence_top40.csv`

A follow-on public diff fetch reached 75/75 referenced PR diff URLs, cached 74 non-empty diffs, and extracted 488 changed-file rows from 453 unique changed files.

Additional generated artifacts:

- `reports/linked_pr_diff_fetch_report.md`
- `tables/linked_pr_diff_manifest.csv`
- `tables/linked_pr_changed_files.csv`
- `data/linked_pr_diffs/*.diff`

The top-40 table can be used as a conference appendix or as the seed for future linked-PR/code-diff adjudication. The linked PR diffs are provenance for future root-cause labels, not adjudicated root-cause labels in the current benchmark.
