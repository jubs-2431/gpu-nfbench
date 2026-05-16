# Final Submission Readiness Notes

## Completed before submission

- Public GitHub artifact repository and release are live at `https://github.com/jubs-2431/gpu-nfbench/releases/tag/v1.0-conference`.
- IEEE manuscript source and PDF are included in the artifact bundle.
- The canonical benchmark is `data/processed/gold_benchmark_expanded_v2_canonical.csv` with 1,191 labeled issues.
- The paper reports the 419 human adjudication updates and 211 primary-label revisions.
- The root-cause extension has been expanded to 250 evidence-coded rows, while preserving the honest provenance boundary: 50 rows are human-adjudicated, 55 are linked-fix evidence-coded, and 145 are issue-text evidence-coded.
- Cross-repository weakness analysis is included, with the weakest held-out repositories called out explicitly.
- Local zero-shot Llama 3.2 3B is included as a same-split LLM baseline against the fine-tuned FLAN-T5-base model.
- The artifact includes a 12-case error appendix with concrete boundary failures.
- `REPRODUCIBILITY.md` gives exact commands for regenerating the main tables, root-cause extension, manuscript, and release bundle.
- An external-repository candidate pool has been materialized from partial GitHub search results across vLLM, CUTLASS, CCCL, and ThunderKittens. These rows are explicitly marked as candidates, not gold labels.

## Still account-gated

- Zenodo DOI: completed as `10.5281/zenodo.20242157` at `https://zenodo.org/records/20242157`.
- Modern external API baseline: the shell has no `GEMINI_API_KEY`, `OPENAI_COMPAT_API_KEY`, `OPENROUTER_API_KEY`, or `GROQ_API_KEY`. The paper therefore uses the completed local Llama 3.2 3B baseline and treats prior partial Gemini overlap as supplementary rather than headline evidence.
- Additional public GitHub issue collection: GitHub Search API returned a rate-limit error during the final pass, so the newly collected external rows are included as a candidate pool only.

## Submission recommendation

The current paper is ready to submit. Do not delay submission for a stronger external API baseline unless the venue specifically requires it; the core contribution is the benchmark, adjudication trail, reproducible artifact, and triage evaluation.
