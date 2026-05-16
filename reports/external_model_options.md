# External Model Options for Full-Coverage Accuracy

The local experiments show that the current 191-row gold set is not enough to honestly reach 70-80% full-coverage accuracy with small local models and TF-IDF classifiers. The next experiment should run fold-safe retrieval-augmented prompts through a stronger external model, then evaluate both direct predictions and deterministic+LLM ensemble rules.

Recommended free or free-quota options, in the order I would try them:

1. **OpenRouter free router or free model variants**
   - Official docs: https://openrouter.ai/openrouter/free/api
   - API key env var: `OPENROUTER_API_KEY`
   - Runner mode: `openai_compat`
   - Base URL: `https://openrouter.ai/api/v1`
   - Model to try first: `openrouter/free`

2. **GroqCloud free tier**
   - Official docs: https://console.groq.com/docs/rate-limits and https://console.groq.com/docs/openai
   - API key env var: `GROQ_API_KEY`
   - Runner mode: `openai_compat`
   - Base URL: `https://api.groq.com/openai/v1`
   - Model to try first: a current high-context Llama or Qwen model available in the Groq console.

3. **Google AI Studio / Gemini API free tier**
   - Official docs: https://ai.google.dev/gemini-api/docs/rate-limits
   - API key env var: `GEMINI_API_KEY`
   - Runner mode: `gemini`
   - Model to try first: a current Flash model from Google AI Studio.
   - For free-tier request caps, prefer the batched runners: `scripts/run_gemini_batched_llm_baseline.py` for zero-shot and `scripts/run_gemini_rag_batched_llm_baseline.py` for fold-safe RAG.

4. **GitHub Models included free usage**
   - Official docs: https://docs.github.com/en/billing/concepts/product-billing/github-models
   - Useful if your GitHub account has access to a stronger model in the GitHub Models catalog.
   - This may require a separate endpoint configuration, so it is second-priority unless you already have it working.

Example commands:

```bash
export OPENROUTER_API_KEY="..."
python3 scripts/run_external_llm_baseline.py \
  --provider openai_compat \
  --api-key-env OPENROUTER_API_KEY \
  --base-url https://openrouter.ai/api/v1 \
  --model openrouter/free \
  --sleep 0.5 \
  --retries 3
python3 scripts/evaluate_llm_baseline_predictions.py evaluation/llm_baseline_predictions_openai_compat_openrouter_free.csv
```

```bash
export GROQ_API_KEY="..."
python3 scripts/run_external_llm_baseline.py \
  --provider openai_compat \
  --api-key-env GROQ_API_KEY \
  --base-url https://api.groq.com/openai/v1 \
  --model MODEL_FROM_GROQ_CONSOLE \
  --sleep 0.5 \
  --retries 3
```

```bash
export GEMINI_API_KEY="..."
python3 scripts/run_external_llm_baseline.py \
  --provider gemini \
  --api-key-env GEMINI_API_KEY \
  --model MODEL_FROM_AI_STUDIO \
  --sleep 0.5 \
  --retries 3
```

Fold-safe Gemini RAG command:

```bash
export GEMINI_API_KEY="..."
python3 scripts/run_gemini_rag_batched_llm_baseline.py \
  --model gemini-3.1-flash-lite \
  --batch-size 8 \
  --sleep 2 \
  --retries 3
python3 scripts/evaluate_llm_baseline_predictions.py evaluation/llm_rag_predictions_gemini31_flash_lite.csv
python3 scripts/evaluate_llm_enhanced_ensembles.py evaluation/llm_rag_predictions_gemini31_flash_lite.csv
```

Do not report external-model accuracy until the resulting prediction CSV is evaluated against the hidden gold file. If the external model improves only selective/high-confidence accuracy, keep that distinction explicit.

Run note from the first OpenRouter check: the API key reached OpenRouter, but both the random `openrouter/free` router and specific free variants returned upstream provider 429 rate-limit errors during smoke tests. If this persists, use a Gemini/Groq free key, add your own upstream provider key in OpenRouter integrations, or allow a low-cost paid OpenRouter model for the 191-row run.
