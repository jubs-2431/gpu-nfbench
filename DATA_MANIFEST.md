# Data Manifest

## Data source

All raw records are public GitHub issue search results collected with the GitHub CLI. Each JSON file stores public issue title, body, URL, repository, state, timestamps, and public GitHub labels returned by `gh search issues`.

## Repositories

- `triton-lang/triton`
- `cupy/cupy`
- `pytorch/pytorch`
- `jax-ml/jax`
- `numba/numba`
- `rapidsai/cudf`

## Query terms

The collection used combinations of:

- `nan`
- `overflow`
- `dtype`
- `precision`
- `incorrect`
- repository-specific context terms such as `triton` or `cuda`

## Processed dataset

Processed dataset:

`data/processed/gpu_numerical_issue_seed.csv`

Current processed size:

930 unique public issues.

Validation files:

- `data/processed/validation_subset.csv`: stratified validation subset selected by repository and candidate primary label.
- `data/validation_context/*.issue.json`: full public GitHub issue bodies for the validation subset.
- `data/validation_context/*.comments.json`: public GitHub comments for the validation subset.
- `data/validation_context/manifest.json`: mapping from validation rows to fetched context files.
- `data/processed/validation_adjudicated.csv`: context-adjudicated labels, confidence fields, evidence snippets, and data-quality flags.

Gold-candidate benchmark files:

- `data/processed/gold_candidate_subset.csv`: 191-issue candidate set selected with up to five issues per repository/candidate-label bucket and temporal spread inside each bucket.
- `data/gold_context/*.issue.json`: full public GitHub issue bodies for the gold-candidate set.
- `data/gold_context/*.comments.json`: public GitHub comments for the gold-candidate set.
- `data/gold_context/manifest.json`: mapping from gold-candidate rows to fetched context files.
- `data/linked_pr_diffs/`: public GitHub `.diff` files fetched for pull requests referenced by the linked-fix evidence table.
- `annotation/annotator_A_blind.csv`: first completed blind human annotation packet.
- `annotation/annotator_B_blind.csv`: second completed blind human annotation packet.
- `annotation/adjudication_template.csv`: completed adjudication file for resolving disagreements.
- `annotation/calibration_round2_blind.csv`: blind post-calibration relabeling packet for disagreement/low-confidence rows.
- `annotation/calibration_round2_review.csv`: adjudicator/training review packet for calibration.
- `annotation/rare_class_expansion_candidates.csv`: non-gold candidate packet for future rare-class human annotation.
- `annotation/full_coverage_expansion_blind.csv`: 215-row non-gold blind packet for expanding full-coverage human labels.
- `annotation/full_coverage_expansion_review.csv`: same 215-row expansion packet with hidden candidate labels exposed for review after blind labeling.
- `annotation/candidate_label_suggestions_hidden_from_annotators.csv`: silver labels kept separate from the blind annotation packets.
- `annotation/ai_prelabel_pass_A_context_only.csv`: AI-generated context-only prelabels for reviewer triage; not human labels.
- `annotation/ai_prelabel_pass_B_candidate_aware.csv`: AI-generated candidate-aware prelabels for reviewer triage; not human labels.
- `annotation/ai_prelabel_disagreements.csv`: rows where the two AI prelabel passes disagree.
- `data/processed/gold_benchmark.csv`: adjudicated 191-row benchmark produced by `scripts/evaluate_gold_labels.py`.
- `reports/gold_label_agreement.md`: agreement/status report produced by `scripts/evaluate_gold_labels.py`.
- `reports/gold_benchmark_analysis.md`: gold-label distribution, repository coverage, secondary-cause distribution, and weak-vs-gold drift.
- `evaluation/llm_baseline_prompts.jsonl`: gold-hidden prompt packet for a future external LLM baseline.
- `evaluation/llm_baseline_prediction_schema.json`: required prediction schema for LLM outputs.
- `evaluation/llm_baseline_predictions_ollama_llama3.2_3b.csv`: zero-shot local Ollama `llama3.2:3b` predictions over the 191-row gold benchmark.
- `evaluation/llm_rag_predictions_ollama_llama3.2_3b.csv`: fold-safe RAG/few-shot local Ollama predictions over the 191-row gold benchmark.
- `evaluation/llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv`: external batched Gemini `gemini-3.1-flash-lite` predictions over the 191-row gold benchmark.
- `evaluation/agentic_ensemble_predictions.csv`: component predictions and agreement counts for the answer/abstain ensemble.
- `evaluation/full_coverage_ensemble_predictions.csv`: full-coverage no-abstention improvement predictions over the 191-row gold benchmark.
- `reports/gold_baseline_classifier.md`: deterministic BM25, Naive Bayes, TF-IDF logistic, TF-IDF linear SVM, bigram TF-IDF, and leave-one-repository-out baselines evaluated against adjudicated gold labels.
- `reports/llm_baseline_protocol.md`: local and external LLM-baseline protocol and evaluator instructions.
- `reports/llm_baseline_results.md`: most recent evaluated LLM prediction-file report from `scripts/evaluate_llm_baseline_predictions.py`.
- `reports/gemini_batched_baseline_results.md`: completed external Gemini batched baseline results.
- `reports/agentic_ensemble_abstention.md`: full-coverage and selective answer/abstain ensemble metrics.
- `reports/full_coverage_model_improvement.md`: no-abstention improvement experiments with candidate-label, linked-diff/path, hierarchy, and deterministic ensemble variants.
- `reports/external_model_options.md`: current free/free-quota external LLM options and runner commands for future full-coverage experiments.
- `tables/llm_agentic_baseline_comparison.csv`: side-by-side zero-shot, RAG, full-vote, and selective agentic metrics.
- `reports/annotation_calibration_analysis.md`: high-confidence/subset kappa analysis and round-2 calibration packet description.
- `reports/gold_error_analysis.md`: weak-vs-gold corrections, representative examples, disagreement cases, and refined adjudication rules.
- `reports/root_cause_context_analysis.md`: fix/workaround/root-cause and linked-reference signals found in public issue/comment context.
- `reports/linked_fix_evidence_analysis.md`: stricter linked-fix, PR, and inline patch evidence mined from public context.
- `reports/linked_pr_diff_fetch_report.md`: public PR diff-fetch summary, including fetched/cached counts and changed-file extraction counts.
- `reports/rare_class_expansion_plan.md`: non-gold expansion plan for rare gold classes.
- `reports/ai_preannotation_report.md`: AI prelabel summary and warning that these labels are not human-gold annotations.

## Label provenance

The label columns are not official project labels.

- `candidate_failure_labels`: transparent keyword/LLM-assisted failure-mode labels.
- `candidate_primary_failure`: primary failure label chosen from the candidate set.
- `candidate_cause_labels`: suspected cause/context tags from issue text.
- `agent_review_*`: reserved for deeper manual/LLM adjudication; currently blank unless explicitly filled in a later pass.
- `context_primary_failure`: validation-subset label assigned after reading the full issue body and public comments.
- `context_confidence`: high/medium/low confidence attached to the validation-subset label.
- `evidence_snippet`: short source-backed excerpt supporting the context label.
- `has_reproducer`, `has_stack_trace`, `has_linked_fix_signal`: validation-subset quality flags derived from issue/comment context.
- `gold_primary_failure`: adjudicated primary label for the 191-row benchmark.
- `gold_secondary_cause_labels_pipe_separated`: adjudicated multi-label cause/context tags.
- `gold_is_true_numerical_failure`: adjudicated yes/no/unclear flag for whether the issue is a true numerical failure.

Current validation summary:

- 82 validation issues.
- 44/82 candidate labels agreed with context labels (53.7%).
- 74/82 validation issues contained a reproducer or code snippet (90.2%).
- 31/82 contained a stack trace or error log (37.8%).
- 36/82 contained a fix, workaround, or closure-related signal (43.9%).

Current gold benchmark status:

- 191 candidate rows.
- 191/191 candidate rows have fetched public issue/comment context.
- 2 blind annotation packets completed.
- 191/191 rows have completed double annotation.
- 191/191 rows have adjudicated gold labels.
- Primary-label observed agreement: 0.372.
- Primary-label Cohen's kappa: 0.280.
- Both-high-confidence subset Cohen's kappa: 0.624.
- Secondary-cause mean Jaccard overlap: 0.585.
- Weak candidate labels match adjudicated gold labels for 79/191 rows (41.4%).
- BM25 kNN gold baseline: 35.6% accuracy and 0.185 macro F1.
- Local Ollama `llama3.2:3b` zero-shot baseline: 19.4% accuracy and 0.140 macro F1.
- Fold-safe RAG `llama3.2:3b` baseline: 28.8% accuracy and 0.138 macro F1.
- External batched Gemini `gemini-3.1-flash-lite` baseline: 22.0% accuracy and 0.203 macro F1.
- Best no-abstention full-coverage improvement: deterministic ensemble at 52.9% accuracy and 0.267 macro F1.
- Agentic full-coverage ensemble vote: 51.8% accuracy and 0.329 macro F1.
- Agentic selective mode: 70.6% accuracy at 26.7% coverage for vote agreement >=6/8; 96.4% accuracy at 14.7% coverage when weak label, TF-IDF SVM, and RAG LLM agree.
- Best stratified 5-fold gold baseline: TF-IDF linear SVM at 50.3% accuracy and 0.270 macro F1.
- Best leave-one-repository-out accuracy: TF-IDF logistic at 50.3% accuracy and 0.247 macro F1.
- Linked-fix evidence subset: 105 rows with at least one linked-fix signal, 39 explicit PR URLs, 9 inline diff/patch snippets.
- Linked PR diffs: 75/75 referenced public PR diff URLs reached, 74 non-empty diffs cached, producing 488 changed-file rows from 453 unique changed files.

Provenance note:

- AI prelabels and weak labels are not human labels. Blind annotation files
  were completed by human annotators, and adjudication uses anonymized human
  adjudicator metadata.

## Reproducibility commands

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
python3 scripts/agentic_ensemble_abstention.py
python3 scripts/full_coverage_model_improvements.py
python3 scripts/annotation_calibration_analysis.py
python3 scripts/gold_error_analysis.py
python3 scripts/root_cause_context_analysis.py
python3 scripts/linked_fix_evidence_analysis.py
python3 scripts/fetch_linked_pr_diffs.py
python3 scripts/select_rare_class_expansion_candidates.py
python3 scripts/prepare_full_coverage_annotation_expansion.py
```
