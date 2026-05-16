# V2 Cross-Repository Weakness Analysis

Leave-one-repository-out evaluation is the hardest setting because the model must transfer across project vocabularies, issue templates, and library-specific failure modes.

| held-out repository | issues | best model | best acc. | best macro F1 | ensemble acc. | interpretation |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| numba/numba | 65 | tfidf_linear_svm | 0.492 | 0.540 | 0.462 | Small repository slice with CUDA/LLVM/build/runtime language; boundary between compiler crash and non-numerical support is difficult. |
| cupy/cupy | 82 | tfidf_linear_svm | 0.500 | 0.415 | 0.439 | Lowest transfer accuracy; CuPy has many environment, dtype, and API-compatibility reports whose vocabulary differs from larger training projects. |
| rapidsai/cudf | 132 | tfidf_linear_svm | 0.583 | 0.600 | 0.545 | Mid-sized slice with dataframe/parser/API reports mixed with numerical correctness language, lowering macro F1. |
| triton-lang/triton | 149 | expanded_gold_vote_ensemble | 0.651 | 0.619 | 0.651 | Transfer is comparatively strong for this repository. |
| pytorch/pytorch | 294 | tfidf_linear_svm | 0.724 | 0.620 | 0.711 | Transfer is comparatively strong for this repository. |
| jax-ml/jax | 299 | bigram_tfidf_logistic | 0.746 | 0.683 | 0.736 | Transfer is comparatively strong for this repository. |
| rapidsai/cuml | 134 | bigram_tfidf_logistic | 0.791 | 0.765 | 0.761 | Transfer is comparatively strong for this repository. |
| apache/tvm | 36 | naive_bayes | 0.889 | 0.562 | 0.806 | A single linear model transfers better than the ensemble, suggesting candidate-label or model-vote signals are less stable under repository shift. |

The weakest transfer repositories are CuPy and Numba. Both contain many environment/build/runtime and API-compatibility issues whose wording overlaps numerical symptoms but differs from larger PyTorch/JAX/RAPIDS issue styles. The paper should present this not as a failure of the benchmark, but as evidence that repository transfer is a real benchmark challenge.
