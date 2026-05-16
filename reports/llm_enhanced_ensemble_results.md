# LLM-Enhanced Ensemble Results

This report evaluates whether external LLM predictions improve GPU-NFBench classification when combined with the deterministic full-coverage ensemble.

| model or mode | answered rows | coverage | accuracy | macro F1 | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| deterministic_ensemble | 191 | 1.000 | 0.529 | 0.267 | Existing no-abstention deterministic ensemble. |
| llm_baseline_predictions_gemini31_flash_lite_batched_merged_direct | 191 | 1.000 | 0.220 | 0.203 | Direct predictions from `evaluation/llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv`. |
| llm_baseline_predictions_gemini31_flash_lite_batched_merged_weighted_vote | 191 | 1.000 | 0.539 | 0.272 | Vote over deterministic model family plus confidence-weighted LLM vote; deterministic ensemble breaks ties. |
| llm_baseline_predictions_gemini31_flash_lite_batched_merged_gated_override | 191 | 1.000 | 0.534 | 0.268 | LLM can override only when non-low confidence and at least two deterministic models agree with it. |
| llm_baseline_predictions_gemini31_flash_lite_batched_merged_agreement_selective | 25 | 0.131 | 0.880 | 0.664 | Selective mode answers only when LLM agrees with deterministic ensemble or at least two deterministic models. |

Interpretation: the deterministic ensemble remains the full-coverage fallback unless an external model produces a measurable improvement under the fixed gated and weighted rules above.
