# Expanded Gold Workflow

This report keeps human gold labels separate from AI/candidate prefills.

- Original gold rows: 191
- Expansion packet rows: 215
- Expansion rows with complete human labels: 0
- Expansion rows still needing human review: 215
- Expanded gold rows written: 191

Generated files:
- `annotation/full_coverage_expansion_ai_prefill.csv`: candidate/AI prefill to speed annotation; not gold.
- `annotation/full_coverage_expansion_human_todo.csv`: rows still requiring human labels/evidence.
- `data/processed/gold_benchmark_expanded.csv`: original gold plus completed expansion rows only.
- `tables/expanded_gold_primary_counts.csv`

A row becomes gold only when `primary_failure_label`, `confidence`, and `evidence_quote` are filled by human review.
