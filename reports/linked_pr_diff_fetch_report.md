# Linked PR Diff Fetch Report

This report fetches public GitHub `.diff` files for pull requests referenced by the linked-fix evidence table. It does not require `gh` authentication; unavailable, private, deleted, or non-pull-request references are retained in the manifest as failures instead of being silently dropped.

| metric | value |
| --- | ---: |
| unique PR URLs attempted | 75 |
| fetched or cached diffs | 75 |
| non-empty diffs | 74 |
| zero-byte HTTP-200 diffs | 1 |
| failed fetches | 0 |
| skipped oversized diffs | 0 |
| total fetched bytes | 2360837 |
| changed-file rows extracted | 488 |
| unique changed files extracted | 453 |

Generated artifacts:

- `tables/linked_pr_diff_manifest.csv`
- `tables/linked_pr_changed_files.csv`
- `data/linked_pr_diffs/*.diff`

The manifest keeps source `blind_id` links so a reviewer can trace each code diff back to the adjudicated benchmark row that referenced it.
