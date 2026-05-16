# GPU-NFBench Reproducibility Guide

This guide records the exact artifact checks used for the conference-ready release. It is intended for artifact reviewers and future maintainers.

## Environment

- macOS or Linux shell with Python 3.10+.
- Recommended virtual environment: `.llm_venv`.
- Core packages used by the deterministic scripts: `scikit-learn`, `pandas`, `numpy`.
- Optional manuscript tools: `tectonic`, `pdftotext`, `pdftoppm`.
- Optional local LLM baseline: `ollama` with `llama3.2:3b`.

The artifact does not require private data. Public GitHub issue text is already captured in the processed files included in the release.

## Primary Files

- Canonical benchmark: `data/processed/gold_benchmark_expanded_v2_canonical.csv`
- Human adjudication updates: `tables/v2_gold_label_revisions.csv`
- Root-cause subset: `annotation/gpu_nfbench_root_cause_50_adjudication_completed.csv`
- Root-cause extension: `tables/root_cause_250_evidence_coded.csv`
- Held-out LLM predictions: `evaluation/v2_standalone_seq2seq_llm_predictions.csv`
- Manuscript source: `paper/gpu_numerical_failure_taxonomy_ieee.tex`
- Manuscript PDF: `paper/GPU-NFBench_IEEE_Manuscript.pdf`

## Recreate Main Tables

Run:

```bash
.llm_venv/bin/python scripts/train_v2_gold_models.py
.llm_venv/bin/python scripts/v2_generalization_analysis.py
.llm_venv/bin/python scripts/cross_repo_weakness_analysis.py
.llm_venv/bin/python scripts/evaluate_v2_local_llm_baseline.py
```

Expected headline values:

- Full-coverage deterministic ensemble: 0.738 accuracy, 0.712 macro F1.
- Leave-one-repository-out best deterministic result: 0.668 accuracy.
- Chronological best deterministic result: 0.766 accuracy, 0.738 macro F1.
- Standalone FLAN-T5-base held-out test: 0.805 accuracy, 0.778 macro F1.
- Local zero-shot Llama 3.2 3B held-out test: 0.317 accuracy, 0.322 macro F1.

## Recreate Root-Cause and Appendix Files

Run:

```bash
.llm_venv/bin/python scripts/expand_root_cause_evidence_coded.py
.llm_venv/bin/python scripts/build_error_case_appendix.py
.llm_venv/bin/python scripts/materialize_external_candidate_pool.py
```

Important provenance rule: only the 50-row root-cause subset is human-adjudicated. The 250-row file is evidence-coded and includes provenance columns separating human, linked-fix, and issue-text evidence.

## Rebuild the Paper

Run:

```bash
tectonic -X compile paper/gpu_numerical_failure_taxonomy_ieee.tex --keep-logs
cp paper/gpu_numerical_failure_taxonomy_ieee.pdf paper/GPU-NFBench_IEEE_Manuscript.pdf
```

Optional text check:

```bash
pdftotext paper/GPU-NFBench_IEEE_Manuscript.pdf - | rg "80.5|89.3|250-row|Zenodo|66.8|31.7"
```

## Rebuild Release Bundle

Run:

```bash
.llm_venv/bin/python scripts/time_split_and_release_artifacts.py
unzip -l release/gpu-nfbench-artifact.zip | rg "REPRODUCIBILITY|root_cause_250|external_repo|v2_error_case|llama32|zenodo"
```

Expected release asset:

- `release/gpu-nfbench-artifact.zip`

## GitHub and Zenodo

GitHub release:

- `https://github.com/jubs-2431/gpu-nfbench/releases/tag/v1.0-conference`

Zenodo DOI status:

- The artifact contains `.zenodo.json` and `CITATION.cff`.
- A DOI must be minted from the tagged GitHub release through the repository owner's Zenodo account.
- After Zenodo creates the DOI, update the paper's Artifact Availability paragraph and `CITATION.cff`.

## Known Non-Reproducible or Account-Gated Steps

- External API LLM baselines require a provider API key and may incur cost or rate limits.
- GitHub issue expansion queries can hit GitHub Search API rate limits.
- Human adjudication cannot be regenerated from code; the completed annotation CSVs are included as artifacts.
