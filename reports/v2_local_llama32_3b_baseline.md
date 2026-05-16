# V2 Local LLM Baseline

This baseline evaluates a local-only zero-shot Ollama `llama3.2:3b` model on the same 123-row v2 held-out split used for the fine-tuned FLAN-T5 standalone model.

| model/mode | rows | accuracy | macro F1 |
| --- | ---: | ---: | ---: |
| fine_tuned_flan_t5_base | 123 | 0.805 | 0.778 |
| local_llama3.2_3b_zero_shot | 123 | 0.317 | 0.322 |

A cloud-backed `qwen3.5:cloud` smoke test was not run because exporting held-out benchmark prompts to the cloud was blocked. The paper should present this as a local zero-shot LLM baseline, not as an external frontier API comparison.
