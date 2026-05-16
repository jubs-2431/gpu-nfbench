# Expanded LLM Training Packet

Rows written: 1191
JSONL packet: `evaluation/expanded_gold_llm_training_packet.jsonl`
Balanced few-shot examples: `evaluation/expanded_gold_balanced_fewshot_examples.json`

This is an LLM-ready supervised/RAG packet. It is suitable for retrieval-augmented prompting, few-shot prompting, or later fine-tuning with a provider that supports JSONL supervised examples.

## Label counts

- crash_compile: 138
- dtype_casting: 205
- nan_inf: 174
- not_numerical_failure: 128
- overflow_underflow: 129
- performance_only: 58
- precision_tolerance: 359
