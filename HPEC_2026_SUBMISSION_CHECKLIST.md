# HPEC 2026 Submission Checklist

Submission page checked: https://ieee-hpec.org/submit/

## Files to Submit

- Paper PDF: `paper/GPU-NFBench_HPEC2026_Manuscript.pdf`
- Optional source: `paper/gpu_nfbench_hpec_2026.tex`
- CMT text abstract: `HPEC_2026_SUBMISSION_ABSTRACT.txt`

## HPEC Rule Check

- Uses IEEE conference format via `IEEEtran`.
- Not anonymous; HPEC explicitly says papers should not be anonymous.
- Full-paper length is 5 pages total, under the 6-page upload limit.
- Includes author name, affiliation, city/state/country, and email.
- Quantitative results are foregrounded: LLM-based triage, full-coverage deterministic baselines, cross-repository transfer, chronological transfer, high-confidence abstention, audit quality, and artifact results.
- HPEC-fit framing is explicit: high-performance ML software, GPU kernels, graph compilers, low/mixed precision, reliability, and LLM-based triage.
- Main claim is model/system-first: the paper presents an LLM-based triage system that detects and categorizes GPU numerical-failure reports; GPU-NFBench is the training/evaluation benchmark supporting that claim.
- Artifact is public and citable:
  - GitHub release: https://github.com/jubs-2431/gpu-nfbench/releases/tag/v1.0-conference
  - Zenodo DOI: https://doi.org/10.5281/zenodo.20242157

## Submission Details

- Submission system: Microsoft CMT, https://cmt3.research.microsoft.com/HPEC2026/
- Deadline: July 7, 2026, midnight anywhere on earth.
- Notification: August 19, 2026.
- Camera-ready deadline: September 4, 2026.
- HPEC accepts full papers up to 6 pages; the site notes references and acknowledgments are not included, but the upload page also says paper file up to 6 pages, so this version is 5 pages total to avoid ambiguity.

## Generative AI Disclosure

HPEC links to a "Use of AI tools for authoring research papers" policy page, but the public submission pages do not require a disclosure paragraph inside the manuscript. If CMT asks for a generative-AI disclosure, use:

> Generative AI tools were used only for editorial assistance and LaTeX compiling/formatting support. The author reviewed and is responsible for the final manuscript and submitted content.
