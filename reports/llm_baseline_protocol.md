# LLM Baseline Protocol

This packet prepares zero-shot or few-shot LLM baselines without exposing gold labels in the prompt file.

## Files

- Prompt JSONL: `evaluation/llm_baseline_prompts.jsonl`
- Prediction schema: `evaluation/llm_baseline_prediction_schema.json`
- Local Ollama runner: `scripts/run_ollama_llm_baseline.py`
- Fold-safe RAG runner: `scripts/run_ollama_rag_llm_baseline.py`
- Fold-safe Gemini RAG runner: `scripts/run_gemini_rag_batched_llm_baseline.py`
- Agentic abstention evaluator: `scripts/agentic_ensemble_abstention.py`
- Evaluator: `scripts/evaluate_llm_baseline_predictions.py`
- Deterministic+LLM ensemble evaluator: `scripts/evaluate_llm_enhanced_ensembles.py`
- Local prediction file: `evaluation/llm_baseline_predictions_ollama_llama3.2_3b.csv`
- RAG prediction file: `evaluation/llm_rag_predictions_ollama_llama3.2_3b.csv`
- Shared evaluator result report: `reports/llm_baseline_results.md`
- External batched Gemini report: `reports/gemini_batched_baseline_results.md`

## Recommended evaluation

1. Run the same prompt file through the chosen LLM with temperature 0.
2. Save one prediction per row using the CSV schema.
3. Evaluate against `data/processed/gold_benchmark.csv` with the evaluator script.
4. Report exact model name, date, temperature, prompt file checksum, and whether examples were zero-shot or few-shot.

## Local baseline already run

The artifact includes a local zero-shot Ollama baseline:

```bash
python3 scripts/run_ollama_llm_baseline.py --model llama3.2:3b
python3 scripts/evaluate_llm_baseline_predictions.py evaluation/llm_baseline_predictions_ollama_llama3.2_3b.csv
python3 scripts/run_ollama_rag_llm_baseline.py --model llama3.2:3b
python3 scripts/evaluate_llm_baseline_predictions.py evaluation/llm_rag_predictions_ollama_llama3.2_3b.csv
python3 scripts/agentic_ensemble_abstention.py
python3 scripts/evaluate_llm_enhanced_ensembles.py evaluation/llm_baseline_predictions_gemini31_flash_lite_batched_merged.csv
```

`llama3.2:3b` evaluated 191/191 rows with zero runner errors, 19.4% accuracy,
and 0.140 macro F1. Its high-confidence slice contained 54 rows at 37.0%
accuracy and 0.171 macro F1. This should be interpreted as a small local-model
baseline, not a frontier-LLM result.

The fold-safe RAG variant reached 28.8% accuracy and 0.138 macro F1. The
agentic full-coverage vote reached 51.8% accuracy and 0.329 macro F1. The
70%+ results are selective: vote agreement >=6/8 answered 51/191 rows at
70.6% accuracy, and weak-label + TF-IDF SVM + RAG LLM agreement answered
28/191 rows at 96.4% accuracy.

## External Gemini baseline already run

The completed `gemini-3.1-flash-lite` batched prediction file covers all 191
rows with zero runner errors but reaches only 22.0% direct accuracy and 0.203
macro F1. When used as a confidence-weighted vote with the deterministic model
family, it improves full-coverage accuracy from 52.9% to 53.9% and macro F1
from 0.267 to 0.272. A selective deterministic+Gemini agreement mode answers
25/191 rows at 88.0% accuracy.

The fold-safe Gemini RAG runner is implemented but should be launched only when
`GEMINI_API_KEY` is exported and free-tier quota is sufficient:

```bash
python3 scripts/run_gemini_rag_batched_llm_baseline.py --model gemini-3.1-flash-lite --batch-size 8
```

## Few-shot pool

If a few-shot condition is used, draw examples only from a training fold and never from the held-out fold. A convenient balanced pool for fold construction is:

GNF-0010-ff390b39, GNF-0005-80aeca70, GNF-0035-1d6522c2, GNF-0001-f56f80f3, GNF-0017-f27cd16b, GNF-0012-d3d20064, GNF-0029-5f61e128

External LLM reruns require credentials in the local environment. Do not store
API keys in repository files or generated artifacts.
