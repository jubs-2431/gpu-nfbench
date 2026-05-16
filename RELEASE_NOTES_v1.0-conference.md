# GPU-NFBench v1.0-conference

This release packages the conference-ready GPU-NFBench artifact.

## Included

- 1,191-row canonical v2 GPU numerical-failure benchmark.
- 419 human adjudication updates applied to the expanded benchmark.
- 50-row human-adjudicated root-cause subset.
- 200-row evidence-coded root-cause extension with provenance labels.
- Deterministic baselines, leave-one-repository-out, chronological, and abstention evaluations.
- Fine-tuned FLAN-T5-base standalone LLM outputs.
- Local zero-shot Llama 3.2 3B baseline on the same 123-row held-out split.
- Cross-repository weakness analysis.
- 12-case error appendix.
- IEEE-format manuscript PDF and LaTeX source.
- Data card, artifact index, and citation metadata.

## Archival Steps

1. Create or re-authenticate GitHub access for `jubs-2431`.
2. Create a public repository named `gpu-nfbench`.
3. Push this folder, excluding local virtual environments and caches.
4. Create tag `v1.0-conference`.
5. Upload `release/gpu-nfbench-artifact.zip` as a GitHub release asset.
6. Archive the GitHub release on Zenodo and copy the DOI into the paper artifact section.
