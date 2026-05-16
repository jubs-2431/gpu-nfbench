# External Repository Candidate Pool

This file materializes the partial raw GitHub search results collected before the authenticated GitHub Search API hit its rate limit. These rows are expansion candidates only; they are not included in the gold benchmark and must not be described as human labels.

Raw JSONL rows: 303
Unique candidates: 262
Malformed raw rows skipped: 0
Raw JSONL: `data/raw_online/external_repo_candidate_issues.jsonl`
Candidate CSV: `data/processed/external_repo_candidate_issue_pool.csv`

## Candidate label counts

- crash_compile: 12
- dtype_casting: 91
- nan_inf: 30
- needs_review: 31
- overflow_underflow: 30
- performance_only: 5
- precision_tolerance: 63

## Repository counts

- vllm-project/vllm: 101
- NVIDIA/cutlass: 84
- NVIDIA/cccl: 58
- HazyResearch/ThunderKittens: 19

## Submission guidance

Use this as future-work evidence that additional GPU-kernel repositories can be mined. Do not fold these rows into the headline benchmark until they are blind-reviewed and adjudicated.
