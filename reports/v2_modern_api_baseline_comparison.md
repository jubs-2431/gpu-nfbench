# V2 Modern API Baseline Comparison

Modern API prediction file: `evaluation/llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv`
Shared v2 held-out rows evaluated: 19
Missing API predictions for held-out rows: 104

| model/mode | rows | accuracy | macro F1 |
| --- | ---: | ---: | ---: |
| fine_tuned_flan_t5_base | 19 | 0.789 | 0.748 |
| gemini_batched_gemini-3.1-flash-lite | 19 | 0.368 | 0.385 |

These results compare the fine-tuned standalone FLAN-T5 classifier against an already-generated modern Gemini API baseline on the same v2 held-out split. No fresh API call was made in this run because no API key was present in the shell environment.
