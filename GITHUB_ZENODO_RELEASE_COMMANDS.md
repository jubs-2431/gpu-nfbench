# GitHub and Zenodo Release Commands

`gh` is currently not authenticated in this shell. After running `gh auth login -h github.com`, use:

```bash
cd /Users/aryanshah/Downloads/gpu_numerical_failure_taxonomy
git init
git add .
git status --short
git commit -m "Release GPU-NFBench v1.0 conference artifact"
gh repo create jubs-2431/gpu-nfbench --public --source=. --remote=origin --push
git tag v1.0-conference
git push origin v1.0-conference
gh release create v1.0-conference release/gpu-nfbench-artifact.zip --title "GPU-NFBench v1.0 Conference Artifact" --notes-file RELEASE_NOTES_v1.0-conference.md
```

Zenodo:

1. Log into Zenodo with GitHub.
2. Enable the `jubs-2431/gpu-nfbench` repository in Zenodo GitHub integration.
3. Create a new GitHub release if Zenodo did not auto-detect `v1.0-conference`.
4. Copy the generated DOI into `paper/gpu_numerical_failure_taxonomy_ieee.tex`.
5. Recompile the manuscript and rebuild `release/gpu-nfbench-artifact.zip`.

Do not archive `.llm_venv`, `tmp/`, local caches, or private API keys.
