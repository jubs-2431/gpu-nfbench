# Artifact Index

Project: GPU-NFBench: A Reproducible Benchmark Framework for Numerical Failure Reports in GPU Kernels

## Manuscript

- `paper/gpu_numerical_failure_taxonomy_ieee.tex`
- `paper/gpu_numerical_failure_taxonomy_ieee.pdf`
- `paper/GPU-NFBench_IEEE_Manuscript.pdf`
- `BENCHMARK_CARD.md`

## Data

- `data/raw/`: initial public GitHub issue-search JSON files.
- `data/raw_more/`: expanded public GitHub issue-search JSON files.
- `data/processed/gpu_numerical_issue_seed.csv`: 930 unique public issues after URL deduplication.
- `data/processed/validation_subset.csv`: 82-issue stratified validation subset.
- `data/validation_context/`: fetched public issue bodies and comments for the validation subset.
- `data/processed/validation_adjudicated.csv`: context labels, confidence, evidence snippets, and validation quality flags.
- `data/processed/gold_candidate_subset.csv`: 191-issue gold-candidate benchmark subset.
- `data/gold_context/`: fetched public issue bodies and comments for the 191 gold-candidate issues.
- `data/processed/gold_benchmark.csv`: 191-row adjudicated benchmark.
- `data/linked_pr_diffs/`: fetched public GitHub `.diff` files for PRs referenced by the linked-fix evidence table.

## Gold Annotation Packet

- `annotation/ANNOTATION_GUIDE.md`
- `annotation/annotator_A_blind.csv`
- `annotation/annotator_B_blind.csv`
- `annotation/adjudication_template.csv`
- `annotation/calibration_round2_blind.csv`
- `annotation/calibration_round2_review.csv`
- `annotation/rare_class_expansion_candidates.csv`
- `annotation/full_coverage_expansion_blind.csv`
- `annotation/full_coverage_expansion_review.csv`
- `annotation/candidate_label_suggestions_hidden_from_annotators.csv`
- `annotation/ai_prelabel_pass_A_context_only.csv`
- `annotation/ai_prelabel_pass_B_candidate_aware.csv`
- `annotation/ai_prelabel_disagreements.csv`

- `evaluation/llm_baseline_prompts.jsonl`
- `evaluation/llm_baseline_prediction_schema.json`
- `evaluation/llm_baseline_predictions_ollama_llama3.2_3b.csv`
- `evaluation/llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv`
- `evaluation/llm_enhanced_ensemble_predictions.csv`
- `evaluation/llm_rag_predictions_ollama_llama3.2_3b.csv`
- `evaluation/agentic_ensemble_predictions.csv`
- `evaluation/full_coverage_ensemble_predictions.csv`

## Evaluation Support

- `scripts/evaluate_llm_baseline_predictions.py`
- `scripts/fetch_linked_pr_diffs.py`
- `scripts/run_ollama_llm_baseline.py`
- `scripts/run_ollama_rag_llm_baseline.py`
- `scripts/agentic_ensemble_abstention.py`
- `scripts/full_coverage_model_improvements.py`
- `scripts/run_external_llm_baseline.py`
- `scripts/run_gemini_batched_llm_baseline.py`
- `scripts/run_gemini_rag_batched_llm_baseline.py`
- `scripts/evaluate_llm_enhanced_ensembles.py`
- `scripts/prepare_full_coverage_annotation_expansion.py`

## Reports

- `reports/analysis_summary.md`: repository counts, failure-label counts, cause-label counts, and representative examples.
- `reports/silver_label_classifier.md`: Naive Bayes classifier metrics against silver labels.
- `reports/validation_adjudication.md`: validation-subset agreement, quality statistics, and representative disagreements.
- `reports/gold_label_agreement.md`: current gold-label completion/agreement status.
- `reports/gold_benchmark_analysis.md`: adjudicated gold-label distributions, cause labels, repository coverage, and weak-label drift.
- `reports/gold_baseline_classifier.md`: text baselines evaluated against adjudicated gold labels.
- `reports/llm_baseline_protocol.md`: local and external LLM-baseline protocol.
- `reports/llm_baseline_results.md`: most recent evaluated LLM prediction-file report from the shared evaluator.
- `reports/gemini_batched_baseline_results.md`: external Gemini batched baseline; all 191 rows completed with 22.0% accuracy and 0.203 macro F1.
- `reports/llm_enhanced_ensemble_results.md`: fixed deterministic+Gemini ensemble rules; best full-coverage result is 53.9% accuracy and 0.272 macro F1, with 88.0% selective accuracy on 25/191 rows.
- `reports/agentic_ensemble_abstention.md`: answer/abstain ensemble metrics and coverage/accuracy tradeoff.
- `reports/full_coverage_model_improvement.md`: no-abstention improvement experiments; deterministic-only full-coverage accuracy is 52.9%, and external-Gemini weighted voting improves it to 53.9%, still below the 70% selective triage target.
- `reports/external_model_options.md`: free/free-quota external model options and runner commands.
- `reports/annotation_calibration_analysis.md`: agreement subset analysis and round-2 calibration packet.
- `reports/gold_error_analysis.md`: weak-label corrections, representative examples, disagreement cases, and refined adjudication rules.
- `reports/root_cause_context_analysis.md`: fix/workaround/root-cause and linked-reference signals in public context.
- `reports/linked_fix_evidence_analysis.md`: explicit PR URL, same-repository reference, and inline patch/diff evidence.
- `reports/linked_pr_diff_fetch_report.md`: public PR diff-fetch summary and changed-file extraction counts.
- `reports/rare_class_expansion_plan.md`: future human-label expansion plan for rare classes.
- `reports/conference_readiness_review.md`: submission-readiness review, remaining risks, and mitigations.
- `reports/ai_preannotation_report.md`: AI prelabel counts and disagreement summary, not human-gold results.

## Tables and Figures

- `tables/repo_counts.csv`
- `tables/failure_counts.csv`
- `tables/cause_counts.csv`
- `tables/year_counts.csv`
- `tables/repo_by_failure_matrix.csv`
- `tables/silver_classifier_metrics.csv`
- `tables/silver_classifier_confusion.csv`
- `tables/validation_agreement.csv`
- `tables/validation_quality.csv`
- `tables/gold_label_agreement.csv`
- `tables/gold_candidate_coverage.csv`
- `tables/gold_primary_counts.csv`
- `tables/gold_true_failure_counts.csv`
- `tables/gold_repo_counts.csv`
- `tables/gold_cause_counts.csv`
- `tables/gold_repo_by_primary_matrix.csv`
- `tables/silver_vs_gold_confusion.csv`
- `tables/annotator_a_vs_b_confusion.csv`
- `tables/gold_classifier_metrics.csv`
- `tables/gold_classifier_loro_metrics.csv`
- `tables/gold_classifier_loro_by_repo.csv`
- `tables/gold_classifier_per_class.csv`
- `tables/gold_classifier_per_class_all_models.csv`
- `tables/gold_classifier_confusion.csv`
- `tables/llm_baseline_metrics.csv`
- `tables/llm_baseline_confusion.csv`
- `tables/llm_baseline_confidence_slices.csv`
- `tables/llm_enhanced_ensemble_metrics.csv`
- `tables/llm_agentic_baseline_comparison.csv`
- `tables/agentic_ensemble_metrics.csv`
- `tables/agentic_abstention_curve.csv`
- `tables/full_coverage_model_improvements.csv`
- `tables/gold_weak_mismatch_pairs.csv`
- `tables/gold_annotation_disagreements.csv`
- `tables/gold_representative_examples.csv`
- `tables/root_cause_context_signals.csv`
- `tables/annotation_agreement_subsets.csv`
- `tables/linked_fix_evidence_subset.csv`
- `tables/linked_fix_evidence_top40.csv`
- `tables/linked_pr_diff_manifest.csv`
- `tables/linked_pr_changed_files.csv`
- `figures/failure_counts.dat`
- `figures/cause_counts.dat`

## Reproducibility

Run from the project root:

```bash
python3 scripts/build_issue_dataset.py
python3 scripts/analyze_dataset.py
python3 scripts/select_validation_subset.py
python3 scripts/fetch_issue_context.py
python3 scripts/adjudicate_validation_subset.py
python3 scripts/silver_label_classifier.py
python3 scripts/select_gold_candidate_subset.py
python3 scripts/fetch_issue_context.py --subset data/processed/gold_candidate_subset.csv --out data/gold_context
python3 scripts/create_annotation_packets.py
python3 scripts/evaluate_gold_labels.py
python3 scripts/analyze_gold_benchmark.py
python3 scripts/gold_baseline_classifier.py
python3 scripts/create_llm_baseline_packet.py
python3 scripts/run_ollama_llm_baseline.py --model llama3.2:3b
python3 scripts/evaluate_llm_baseline_predictions.py evaluation/llm_baseline_predictions_ollama_llama3.2_3b.csv
python3 scripts/run_ollama_rag_llm_baseline.py --model llama3.2:3b
python3 scripts/evaluate_llm_baseline_predictions.py evaluation/llm_rag_predictions_ollama_llama3.2_3b.csv
python3 scripts/run_gemini_batched_llm_baseline.py --model gemini-3.1-flash-lite
python3 scripts/evaluate_llm_baseline_predictions.py evaluation/llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv
python3 scripts/evaluate_llm_enhanced_ensembles.py evaluation/llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv
# Optional external run if GEMINI_API_KEY is exported and free-tier quota is sufficient:
python3 scripts/run_gemini_rag_batched_llm_baseline.py --model gemini-3.1-flash-lite --batch-size 8
python3 scripts/agentic_ensemble_abstention.py
python3 scripts/full_coverage_model_improvements.py
python3 scripts/annotation_calibration_analysis.py
python3 scripts/gold_error_analysis.py
python3 scripts/root_cause_context_analysis.py
python3 scripts/linked_fix_evidence_analysis.py
python3 scripts/fetch_linked_pr_diffs.py
python3 scripts/select_rare_class_expansion_candidates.py
python3 scripts/prepare_full_coverage_annotation_expansion.py
tectonic --keep-logs paper/gpu_numerical_failure_taxonomy_ieee.tex
```

`scripts/fetch_issue_context.py` requires GitHub API access through `gh`.
`scripts/fetch_linked_pr_diffs.py` uses public GitHub `.diff` URLs and does
not require `gh` authentication.
