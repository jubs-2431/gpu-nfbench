# Expanded Gold Conference Strengthening Analysis

## Leave-one-repository-out summary

| model/mode | accuracy | macro F1 |
| --- | ---: | ---: |
| candidate_weak_label | 0.537 | 0.447 |
| bm25_knn | 0.544 | 0.574 |
| naive_bayes | 0.545 | 0.507 |
| tfidf_logistic | 0.679 | 0.665 |
| tfidf_linear_svm | 0.726 | 0.711 |
| bigram_tfidf_logistic | 0.717 | 0.711 |
| expanded_gold_vote_ensemble | 0.715 | 0.705 |

## Ablation summary

| model/mode | accuracy | macro F1 |
| --- | ---: | ---: |
| full_ensemble | 0.768 | 0.753 |
| no_candidate_label_ensemble | 0.761 | 0.748 |
| linear_only_vote | 0.757 | 0.745 |
| candidate_plus_svm | 0.765 | 0.751 |

## Top ensemble error pairs

| gold | predicted | errors | example |
| --- | --- | ---: | --- |
| dtype_casting | precision_tolerance | 37 | pytorch/pytorch#175156: [inductor] Multiple randint calls cause inconsistent RNG results between eager and compiled mode after fixing the rng seed |
| nan_inf | precision_tolerance | 25 | pytorch/pytorch#181146: torch.compile: autograd.Function.apply with aliased inputs drops per-slot gradient contributions |
| overflow_underflow | precision_tolerance | 24 | pytorch/pytorch#180026: DISABLED test_combo_kernel_yz_overflow (__main__.ComboKernelTestsPerSubkernelBlocks) |
| not_numerical_failure | dtype_casting | 15 | rapidsai/cudf#16029: [QST] TypeError: Argument 'real' has incorrect type (expected numpy.ndarray, got ndarray) |
| overflow_underflow | dtype_casting | 14 | cupy/cupy#6715: Wrong overflow with matmul of uint16 arrays |
| not_numerical_failure | crash_compile | 14 | numba/numba#4713: Numba is not detecting the icc_rt libraries when installed with pip |
| precision_tolerance | dtype_casting | 14 | rapidsai/cudf#28: Support for windows? |
| performance_only | not_numerical_failure | 10 | numba/numba#2988: Support For CPU Atomics |
| crash_compile | precision_tolerance | 10 | numba/numba#7063: Numba installation scripts exhibit a race condition during parallel builds |
| crash_compile | dtype_casting | 9 | numba/numba#5158: DeviceNDArray.bind() does not seem to bind the stream to self |
| precision_tolerance | nan_inf | 9 | pytorch/pytorch#166131: Return a view from tensor(requires_grad=False) in autograd function may cause incorrect requires_grad attribute. |
| nan_inf | dtype_casting | 9 | numba/numba#10325: test failures related to tolerance (corruption / refcounting) |
