# Standalone LLM Fine-Tune Files

Total rows: 1191
Train rows: 949
Validation rows: 121
Test rows: 121

Files:

- `llm/finetune/gpu_nfbench_standalone_train.jsonl`
- `llm/finetune/gpu_nfbench_standalone_val.jsonl`
- `llm/finetune/gpu_nfbench_standalone_test.jsonl`
- `llm/finetune/openai_chat_finetune_train.jsonl`
- `llm/finetune/openai_chat_finetune_val.jsonl`
- `llm/finetune/label_map.json`

These files are ready for a true standalone LLM fine-tune. The test split should remain untouched until final evaluation.

## Label counts

- crash_compile: 138
- dtype_casting: 205
- nan_inf: 174
- not_numerical_failure: 128
- overflow_underflow: 129
- performance_only: 58
- precision_tolerance: 359

## Current local status

This machine does not currently have a local fine-tuning stack such as MLX-LM, transformers, PEFT, or TRL installed. The local Ollama Modelfile is a prompted model wrapper, not a weight-updated fine-tune.
