# GPU-NFBench: Numerical Failure Reports in GPU Kernels

## Research question

Can real GPU/kernel bug reports be organized into a reliable taxonomy of numerical failure modes, and can an LLM-assisted classifier predict those failure categories from issue text and code context?

## Real data only

Acceptable data:
- Public GitHub issues and pull requests from Triton, CuPy, CUDA-adjacent Python libraries, and GPU numerical libraries.
- Issue text, labels, linked PRs, reproduction snippets, and maintainer comments.
- Manually verified bug categories with source URLs.

Not acceptable:
- Invented issues.
- Synthetic bug reports used as evidence.
- Claims about detection accuracy without a labeled evaluation set.

## Current artifact status

- 930 unique public GitHub issues across Triton, CuPy, PyTorch, JAX, Numba, and RAPIDS cuDF.
- 82-issue stratified validation subset with full issue/comment context fetched through the GitHub API.
- 191-issue human-adjudicated benchmark packet with full issue/comment context fetched through the GitHub API.
- Completed blind human annotation sheets, adjudication file, agreement report, and `data/processed/gold_benchmark.csv`.
- Context-adjudicated validation labels, confidence fields, evidence snippets, and quality flags.
- Naive Bayes silver-label classifier with train/test split and majority baseline.
- Gold-label analysis, error analysis, stronger deterministic TF-IDF baselines, and leave-one-repository-out evaluation.
- BM25 retrieval baseline, local Ollama LLM baseline, fold-safe RAG LLM baseline, Gemini fold-safe RAG runner, agentic answer/abstain ensemble, and external LLM-baseline prompt/evaluator packet.
- Batched Gemini external LLM baseline runner and completed `gemini-3.1-flash-lite` prediction file.
- Full-coverage model-improvement sweep covering no-`needs_review` prediction, candidate-label features, linked PR/diff/path features, a two-stage hierarchy, a deterministic no-abstention ensemble, and fixed deterministic+Gemini ensemble rules.
- Root-cause context signal analysis, linked-fix evidence table, public PR diff cache, calibration packet, and rare-class expansion candidate packet.
- Blind 215-row full-coverage expansion packet for additional human labels.
- IEEE-format manuscript at `paper/gpu_numerical_failure_taxonomy_ieee.tex`.

## Current research claim

- The completed benchmark has 191 adjudicated rows.
- Primary-label observed agreement is 37.2%; Cohen's kappa is 0.280.
- Secondary-cause mean Jaccard overlap is 0.585.
- Weak candidate labels match adjudicated gold labels for 79/191 rows (41.4%).
- Gold-label TF-IDF linear SVM baseline: 50.3% accuracy, 0.270 macro F1.
- BM25 kNN baseline: 35.6% accuracy, 0.185 macro F1.
- Local Ollama `llama3.2:3b` zero-shot baseline: 19.4% accuracy, 0.140 macro F1; high-confidence slice: 37.0% accuracy, 0.171 macro F1.
- Fold-safe RAG `llama3.2:3b` baseline: 28.8% accuracy, 0.138 macro F1.
- External batched Gemini `gemini-3.1-flash-lite` baseline: 22.0% accuracy, 0.203 macro F1, 0 runner errors.
- Best no-abstention full-coverage improvement: deterministic+Gemini weighted vote at 53.9% accuracy, 0.272 macro F1; deterministic-only ensemble is 52.9% accuracy, 0.267 macro F1.
- Agentic full-coverage ensemble vote: 51.8% accuracy, 0.329 macro F1.
- External Gemini agreement-selective mode: 88.0% accuracy at 25/191 answered rows.
- Agentic abstaining mode: 70.6% accuracy at 51/191 answered rows when at least 6/8 signals agree; 96.4% accuracy at 28/191 answered rows when weak label, TF-IDF SVM, and RAG LLM all agree.
- Both-high-confidence annotation subset: Cohen's kappa 0.624.
- Root-cause context signals: 94/191 rows have fix/workaround/regression/root-cause text; 106/191 have pull-request or issue references.
- Linked-fix evidence: 105 rows have at least one stricter source-backed fix signal; 39 have explicit PR URLs; 9 contain inline diff/patch snippets.
- Linked PR diff cache: 75/75 referenced public PR diff URLs reached; 74 non-empty diffs cached; 488 changed-file rows extracted from 453 unique changed files.

## Remaining before strongest submission

- Preserve anonymized human annotator provenance with the submission materials.
- Run the fold-safe Gemini RAG baseline against a stronger/free-tier-sufficient model if valid external-model credentials become available.
- Treat 70%+ model claims as selective answer/abstain claims unless a future full-coverage model actually reaches that range.
- Use `annotation/full_coverage_expansion_blind.csv` to add human labels across the hardest/least-covered classes.
- Use `annotation/calibration_round2_blind.csv` for a post-calibration human pass if reviewers focus on all-row kappa.
- Human-label the rare-class expansion candidates if the target venue expects stronger per-class modeling claims.
