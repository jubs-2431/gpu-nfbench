# Conference Readiness Review

## Strongest current contributions

- Real public data: 930 unique GitHub issues from Triton, CuPy, PyTorch, JAX, Numba, and RAPIDS cuDF.
- Reproducible benchmark construction: raw records, processed CSVs, context fetches, annotation packets, adjudication output, scripts, reports, tables, and manuscript are all included.
- Completed human-adjudicated benchmark: 191 issues with full issue/comment context, two completed blind human annotation passes, and adjudicated gold labels.
- Evidence of task difficulty: weak labels match adjudicated gold labels for only 79/191 issues (41.4%); blind primary-label agreement is 37.2% with Cohen's kappa 0.280.
- Kappa concern is handled defensibly: the both-high-confidence subset has kappa 0.624, and a 125-row round-2 calibration packet is included instead of rewriting original labels.
- Baseline difficulty is measured: TF-IDF linear SVM on adjudicated gold labels reaches 50.3% accuracy and 0.270 macro F1, showing that simple text classification is still not enough.
- Retrieval is measured: BM25 kNN reaches 35.6% accuracy and 0.185 macro F1, below the stronger TF-IDF model.
- AI baseline is now measured rather than hypothetical: local Ollama `llama3.2:3b` zero-shot prompting reaches 19.4% accuracy and 0.140 macro F1, with zero runner errors. This small-model result is weak, but it strengthens the benchmark-difficulty claim.
- External low-cost LLM baseline is now measured: batched `gemini-3.1-flash-lite` reaches 22.0% accuracy and 0.203 macro F1 on all 191 rows with zero runner errors, still below deterministic baselines.
- Full-coverage improvement experiments are now measured: removing `needs_review`, adding candidate-label and linked-diff/path features, testing a two-stage hierarchy, and voting deterministic models raises no-abstention accuracy to 52.9% with 0.267 macro F1. Adding the completed Gemini predictions as a fixed confidence-weighted vote gives a small lift to 53.9% accuracy and 0.272 macro F1. This prevents a weak or inflated 70% full-coverage claim.
- External Gemini is more useful as a selective agreement signal than as a direct classifier: a deterministic+Gemini agreement mode reaches 88.0% accuracy on 25/191 answered rows.
- Agentic selective triage reaches the requested range only with abstention: the full-coverage ensemble vote reaches 51.8% accuracy and 0.329 macro F1, while vote agreement >=6/8 reaches 70.6% accuracy on 51/191 answered rows. A stricter weak-label + TF-IDF SVM + RAG LLM agreement rule reaches 96.4% accuracy on 28/191 rows.
- A 215-row blind full-coverage expansion packet is included for future human labeling across hard/under-covered classes.
- Root-cause extension potential is measured: 105 rows have at least one linked-fix evidence signal, including 39 explicit PR URLs and 9 inline patch/diff snippets.
- The linked-PR limitation is now materially reduced: 75/75 referenced public PR diff URLs were reached without `gh` authentication; 74 non-empty diffs were cached, producing 488 changed-file rows from 453 unique changed files.
- External LLM baseline infrastructure is still included as a gold-hidden prompt packet plus evaluator; the artifact also includes the local Ollama baseline results.

## Weak areas addressed in the manuscript

- Weak-label noise is not hidden. The paper reports both validation-set agreement (53.7%) and weak-vs-gold agreement (41.4%).
- Ambiguous boundaries are treated as a benchmark property. The discussion explains why dtype, precision, overflow, crash/compile, and false-positive categories overlap in real reports.
- Gold-label class imbalance is explicit. Rare `crash_compile` and `performance_only` classes are not overclaimed.
- Human annotation provenance is handled with anonymized annotator/adjudicator identifiers.
- Search bias is documented. The paper states that counts reflect search terms and reporting vocabulary, not project-level bug rates.

## Remaining work for a stronger venue submission

- Add a small appendix table with representative gold-labeled examples if page limits allow.
- Human-label the new 215-row full-coverage expansion packet and/or the rare-class expansion candidates before making stronger per-class or full-coverage model claims.
- Compare against a stronger named external LLM at temperature 0 if external-model credentials and quota are available; keep the local 3B result as a lower-bound baseline and run the new fold-safe Gemini RAG runner before making stronger full-coverage claims.
- If reporting the 70%+ result, state the coverage and abstention rule in the same sentence.
