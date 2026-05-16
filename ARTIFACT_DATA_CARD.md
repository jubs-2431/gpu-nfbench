# GPU-NFBench Artifact and Data Card

## Intended use

GPU-NFBench is intended for research on GPU numerical-failure triage from
public issue reports. Supported tasks include primary failure classification,
numerical-failure gating, selective prediction with abstention, and
cross-repository robustness evaluation.

## Not intended use

The labels should not be treated as official labels from the maintainers of the
source repositories. The benchmark classifies issue reports, not confirmed root
causes, unless linked fix evidence is separately available.

## Data sources

The benchmark uses public GitHub issues from GPU-related open-source projects:
Triton, CuPy, PyTorch, JAX, Numba, RAPIDS cuDF, RAPIDS cuML, and Apache TVM.
Each row stores public issue metadata and benchmark labels.

## Label taxonomy

Primary labels are:

- `nan_inf`
- `overflow_underflow`
- `precision_tolerance`
- `dtype_casting`
- `crash_compile`
- `performance_only`
- `not_numerical_failure`

## Dataset files

- `data/processed/gpu_numerical_issue_seed.csv`: 930-row silver seed.
- `data/processed/gold_benchmark.csv`: 191-row adjudicated pilot benchmark.
- `data/processed/gold_benchmark_expanded.csv`: 1,191-row expanded gold benchmark.
- `annotation/gold_expansion_1000_taxonomy_repair_changes.csv`: taxonomy-normalization log.
- `annotation/expanded_gold_agreement_audit_120_blind.csv`: original blind audit packet.
- `annotation/expanded_gold_agreement_audit_120_personA_personB_filled.csv`: completed two-person audit.
- `annotation/expanded_gold_audit_119_adjudicated.csv`: completed adjudicated audit.
- `tables/expanded_gold_audit_agreement.csv`: audit agreement/kappa summary.
- `tables/expanded_gold_adjudicated_audit_metrics.csv`: adjudicated-audit agreement and model comparison.
- `tables/expanded_gold_audit_gold_revisions.csv`: rows where audit adjudication differs from existing expanded gold.
- `data/processed/gold_benchmark_expanded_adjudicated_v2.csv`: candidate v2 benchmark with audit adjudications applied.

## Evaluation files

- `tables/expanded_gold_classifier_metrics.csv`: stratified full-coverage metrics.
- `tables/expanded_gold_loro_metrics.csv`: leave-one-repository-out metrics.
- `tables/expanded_gold_ablation_metrics.csv`: ensemble ablation metrics.
- `tables/expanded_gold_top_error_pairs.csv`: highest-frequency error pairs.
- `tables/expanded_gold_abstention_metrics.csv`: selective prediction metrics.
- `reports/standalone_llm_training_summary.md`: standalone LLM training summary.

## Provenance and limitations

The original 191-row benchmark has blind annotation and adjudication artifacts.
The 1,000-row expansion is human-filled and then normalized into the final
taxonomy with a deterministic, logged repair pass. A completed 119-row blind
audit measured Person A/B agreement at 80.7% with Cohen's kappa of 0.721. A
separate adjudication pass completed final labels for the 119 audit rows and
identified 52 rows where the adjudicated audit label differs from the existing
expanded gold label. The v2 benchmark file applies these audit adjudications as
a candidate revision, but models should be retrained before treating v2 as the
canonical benchmark. Public issue text may contain ambiguous or incomplete
evidence, so benchmark labels represent report-level triage rather than
verified maintainer-confirmed root cause.
