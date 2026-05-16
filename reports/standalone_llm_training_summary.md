# Standalone LLM Training Summary

Local environment:

- Virtual environment: `.llm_venv`
- Installed stacks: `torch`, `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `scikit-learn`, `mlx`, `mlx-lm`
- MLX installed but crashes on Metal initialization in this agent session.
- PyTorch training ran successfully on `mps`.

Completed standalone models:

| model | training setup | validation accuracy | validation macro F1 | test accuracy | test macro F1 | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `gpu-nfbench-triage` Ollama prompt wrapper | no weight update, 25-row smoke | n/a | n/a | 0.520 | 0.395 | weak baseline |
| `flan-t5-small` | balanced label-only seq2seq, 3 epochs | 0.719 | 0.677 | 0.678 | 0.635 | trained |
| `flan-t5-base` | balanced label-only seq2seq, 2 epochs | 0.744 | 0.697 | 0.744 | 0.684 | best standalone checkpoint |
| `flan-t5-base` continued | +1 extra epoch from best checkpoint | 0.744 | 0.721 | 0.686 | 0.640 | overfit/regressed |

Best standalone checkpoint:

`llm/models/gpu_nfbench_flan_t5_base_label_balanced`

Best standalone result:

- Test accuracy: `74.4%`
- Test macro F1: `0.684`

Comparison:

- Best standalone LLM: `74.4%` test accuracy
- Best full-coverage ensemble: `76.8%` accuracy, `0.753` macro F1
- Best abstaining ensemble: `85.9%` accuracy at `74.3%` coverage

Paper-safe framing:

The standalone fine-tuned LLM approaches the hybrid ensemble but does not surpass it. The strongest paper claim is that GPU-NFBench supports both standalone LLM training and higher-performing hybrid/abstaining triage. The best full-coverage system remains the ensemble, while the best standalone model provides a cleaner single-model baseline.
