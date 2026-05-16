# Online Candidate Collection

Queries attempted: 24
Unique candidates written: 160
Raw JSONL: `data/raw_online/github_negative_control_candidates.jsonl`
Candidate CSV: `data/processed/negative_control_candidate_issue_pool.csv`

These rows are candidate issues only. They are not gold labels until humans review `primary_failure_label`, `confidence`, and `evidence_quote` fields in a blind packet.

## Candidate label counts

- crash_compile: 15
- dtype_casting: 12
- nan_inf: 2
- needs_review: 103
- overflow_underflow: 6
- performance_only: 12
- precision_tolerance: 10

## Repository counts

- pytorch/pytorch: 78
- triton-lang/triton: 54
- cupy/cupy: 28

## Fetch errors

- none