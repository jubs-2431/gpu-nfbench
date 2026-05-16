# GPU-NFBench Benchmark Card

## Intended Use

GPU-NFBench is intended for research on numerical-failure triage in GPU-related
software issue reports. The benchmark supports studies of failure-mode
classification, bug-taxonomy design, retrieval over debugging evidence, and
agent-assisted issue triage.

## Current Status

The repository contains:

- A 930-issue silver seed dataset mined from public GitHub issues.
- A 191-issue human-adjudicated benchmark with full public issue/comment context.
- Completed blind human annotation sheets, an adjudication file, and an agreement/evaluation script.
- `data/processed/gold_benchmark.csv`, produced after adjudication.
- AI preannotation files for reviewer triage. These are not human labels and
  must not be reported as gold annotations.

The 191-issue packet has completed blind human annotation and adjudication
files. Candidate weak labels were hidden from annotators during blind review.
The released tables use anonymized annotator/adjudicator identifiers.

AI prelabels are stored separately in `annotation/ai_prelabel_*` files to help
reviewers move faster. They should not be copied into the blind human files
unless a human reviewer has independently checked and accepted each row.

## Source Projects

- `triton-lang/triton`
- `cupy/cupy`
- `pytorch/pytorch`
- `jax-ml/jax`
- `numba/numba`
- `rapidsai/cudf`

## Label Space

Primary labels:

- `nan_inf`
- `overflow_underflow`
- `precision_tolerance`
- `dtype_casting`
- `crash_compile`
- `performance_only`
- `not_numerical_failure`
- `needs_review`

Secondary cause labels:

- `memory_mask_bounds`
- `compiler_codegen`
- `async_race_ordering`
- `hardware_backend`
- `reduction_accumulation`
- `api_semantics`
- `environment_configuration`
- `unknown`

## Annotation Protocol

1. Annotator A and Annotator B independently complete the blind CSV files in
   `annotation/`.
2. Annotators must not inspect
   `candidate_label_suggestions_hidden_from_annotators.csv` during labeling.
3. Every labeled row must include a primary label, secondary labels,
   yes/no/unclear numerical-failure status, confidence, and evidence quote.
4. Disagreements are resolved in `annotation/adjudication_template.csv`.
5. `scripts/evaluate_gold_labels.py` computes agreement metrics and writes
   `data/processed/gold_benchmark.csv` only when adjudicated labels are filled.

## Current Gold Benchmark Metrics

- Candidate rows: 191.
- Complete double human annotations: 191.
- Adjudicated gold rows: 191.
- Primary-label observed agreement: 0.372.
- Primary-label Cohen's kappa: 0.280.
- Secondary-cause mean Jaccard: 0.585.
- Weak candidate label vs. adjudicated gold agreement: 79/191 (41.4%).
- Both-high-confidence subset Cohen's kappa: 0.624.
- BM25 kNN baseline: 35.6% accuracy and 0.185 macro F1.
- Best deterministic gold baseline: TF-IDF linear SVM, 50.3% accuracy and
  0.270 macro F1 under stratified 5-fold evaluation.
- Local Ollama `llama3.2:3b` zero-shot baseline: 19.4% accuracy and 0.140
  macro F1 on all 191 gold rows. The model produced no runner errors, but it
  underperformed deterministic text baselines; its high-confidence subset was
  54 rows at 37.0% accuracy and 0.171 macro F1.
- Fold-safe RAG `llama3.2:3b` baseline: 28.8% accuracy and 0.138 macro F1 on
  all 191 rows.
- External batched Gemini `gemini-3.1-flash-lite` baseline: 22.0% accuracy
  and 0.203 macro F1 on all 191 rows with zero runner errors.
- Full-coverage no-abstention improvement sweep: best deterministic ensemble
  at 52.9% accuracy and 0.267 macro F1 after adding candidate-label,
  linked-diff/path, and no-`needs_review` voting features.
- Agentic full-coverage ensemble vote: 51.8% accuracy and 0.329 macro F1.
- Agentic answer/abstain mode: 70.6% accuracy at 51/191 answered rows when
  at least 6/8 signals agree; 96.4% accuracy at 28/191 answered rows when
  weak label, TF-IDF SVM, and RAG LLM all agree.

## Error and Root-Cause Support

- `reports/gold_error_analysis.md` documents the main weak-label correction
  patterns and refined adjudication rules.
- `tables/gold_representative_examples.csv` gives representative examples for
  each gold class.
- `reports/root_cause_context_analysis.md` measures fix/workaround, reference,
  and commit-like signals in public issue/comment context.
- `reports/linked_fix_evidence_analysis.md` identifies 105 rows with stricter
  linked-fix evidence signals, including 39 explicit PR URLs and 9 inline
  diff/patch snippets.
- `scripts/fetch_linked_pr_diffs.py`, `tables/linked_pr_diff_manifest.csv`,
  `tables/linked_pr_changed_files.csv`, and `data/linked_pr_diffs/` provide
  public code-diff provenance for 75/75 referenced PR diff URLs, with 74
  non-empty diffs cached and 488 changed-file rows from 453 unique changed
  files.
- `reports/annotation_calibration_analysis.md` reports high-confidence subset
  agreement and creates a round-2 blind calibration packet.
- `evaluation/llm_baseline_prompts.jsonl` and
  `scripts/evaluate_llm_baseline_predictions.py` support an external LLM
  baseline without exposing gold labels in prompts.
- `scripts/run_ollama_llm_baseline.py`,
  `evaluation/llm_baseline_predictions_ollama_llama3.2_3b.csv`,
  `reports/llm_baseline_results.md`, and
  `tables/llm_baseline_confidence_slices.csv` provide a reproducible local
  LLM baseline.
- `scripts/run_ollama_rag_llm_baseline.py` and
  `evaluation/llm_rag_predictions_ollama_llama3.2_3b.csv` provide the
  fold-safe retrieval/few-shot LLM ablation.
- `scripts/agentic_ensemble_abstention.py`,
  `evaluation/agentic_ensemble_predictions.csv`,
  `tables/agentic_ensemble_metrics.csv`, and
  `reports/agentic_ensemble_abstention.md` provide the selective
  answer/abstain triage result.
- `scripts/full_coverage_model_improvements.py`,
  `tables/full_coverage_model_improvements.csv`,
  `evaluation/full_coverage_ensemble_predictions.csv`, and
  `reports/full_coverage_model_improvement.md` document full-coverage
  improvement experiments that still do not reach 70% without abstention.
- `scripts/run_external_llm_baseline.py` can run the gold-hidden prompt packet
  against OpenAI-compatible or Gemini-compatible external providers when a
  valid free/quota API key is available.
- `scripts/run_gemini_batched_llm_baseline.py`,
  `evaluation/llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv`,
  and `reports/gemini_batched_baseline_results.md` provide a completed
  external Gemini baseline under a batched free-tier protocol.
- `annotation/rare_class_expansion_candidates.csv` contains non-gold candidate
  rows for future expansion of rare classes.
- `annotation/full_coverage_expansion_blind.csv` and
  `annotation/full_coverage_expansion_review.csv` contain a 215-row
  full-coverage expansion packet; candidate labels are hidden from the blind
  file and shown only in the review file.

## Known Limitations

- Public GitHub issues are noisy and reflect project-specific reporting
  practices.
- GitHub search terms bias the candidate pool toward issues containing terms
  such as `nan`, `dtype`, `overflow`, `precision`, and `incorrect`.
- Some issue reports include incomplete reproductions or unresolved root
  causes.
- Linked PR diffs are provided as provenance for future root-cause work, but
  they are not treated as adjudicated root-cause labels in the current
  benchmark.
- The final gold labels are class-imbalanced after adjudication, with small
  `crash_compile` and `performance_only` classes.
- LLM or rule-based labels are kept separate as silver labels or hidden
  suggestions and should not be reported as human labels.
- The local 3B LLM result should be treated as a small-model baseline, not as
  evidence about frontier LLM performance.
- The 70%+ agentic result is selective: it applies only to answered rows under
  explicit abstention rules, not to full-coverage automatic labeling.
- The strongest current no-abstention full-coverage model reaches 52.9%
  accuracy, so any 70%+ full-coverage claim requires new evidence from
  stronger external models and/or more adjudicated human labels.
