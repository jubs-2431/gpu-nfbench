# Analysis Summary

Dataset size: 930 unique public GitHub issues.

## Repositories
| repository | issues |
| --- | --- |
| triton-lang/triton | 175 |
| jax-ml/jax | 174 |
| pytorch/pytorch | 168 |
| rapidsai/cudf | 148 |
| cupy/cupy | 143 |
| numba/numba | 122 |

## Primary failure labels
| failure_label | issues | share_pct |
| --- | --- | --- |
| dtype_casting | 271 | 29.1 |
| nan_inf | 217 | 23.3 |
| precision_tolerance | 142 | 15.3 |
| overflow_underflow | 120 | 12.9 |
| needs_review | 119 | 12.8 |
| crash_compile | 36 | 3.9 |
| performance_only | 25 | 2.7 |

## Secondary suspected-cause labels
| cause_label | issues | share_pct |
| --- | --- | --- |
| hardware_backend | 764 | 82.2 |
| compiler_codegen | 358 | 38.5 |
| reduction_accumulation | 313 | 33.7 |
| async_race_ordering | 287 | 30.9 |
| memory_mask_bounds | 276 | 29.7 |

## Representative issue examples

### dtype_casting
- cupy/cupy: [`cupyx.scipy.special.erfcx` not working](https://github.com/cupy/cupy/issues/9907)
- cupy/cupy: [Support string dtype?](https://github.com/cupy/cupy/issues/9658)
- cupy/cupy: [signal.square fails with float32 inputs](https://github.com/cupy/cupy/issues/9541)

### crash_compile
- cupy/cupy: [Make CCCL compilation warning-free](https://github.com/cupy/cupy/issues/9558)
- cupy/cupy: [Runtime compilation failed - CUDA synchronization primitives are only supported for sm_70 and up](https://github.com/cupy/cupy/issues/8260)
- cupy/cupy: [Unexpected error while assigning array element](https://github.com/cupy/cupy/issues/5328)

### precision_tolerance
- cupy/cupy: [Casting an array to a byteswapped dtype does a view not a cast](https://github.com/cupy/cupy/issues/9015)
- cupy/cupy: [Create a numpy_cupy_allclose with `NULP` (or similar) mechanism](https://github.com/cupy/cupy/issues/9785)
- cupy/cupy: [lsmr with complex linear operator](https://github.com/cupy/cupy/issues/9272)

### overflow_underflow
- cupy/cupy: [Compatibility of integer overflow results in `cupy.linspace`](https://github.com/cupy/cupy/issues/8893)
- cupy/cupy: [Overflow warning in CI](https://github.com/cupy/cupy/issues/2334)
- cupy/cupy: [Overflow on computing pointers](https://github.com/cupy/cupy/issues/5737)

### performance_only
- cupy/cupy: [`cupyx.scipy.sparse.block_diag` working with cupy sparse matrix on GPU](https://github.com/cupy/cupy/issues/7058)
- cupy/cupy: [Performance Degradation in CuPy cuFFT: Runtime Increases from 6 Minutes to 10 Hours After Doubling Input Array Size](https://github.com/cupy/cupy/issues/7978)
- cupy/cupy: [Discussion on multi-GPU support for cuFFT and cuSOLVER](https://github.com/cupy/cupy/issues/2742)

### nan_inf
- cupy/cupy: [cupy.sign(NaN) == 0, unlike numpy](https://github.com/cupy/cupy/issues/8327)
- cupy/cupy: [cupyx.scipy.special.betainc edge cases](https://github.com/cupy/cupy/issues/8934)
- cupy/cupy: [BUG: `cupyx.scipy.special.gammainc`: returns finite results with NaN input](https://github.com/cupy/cupy/issues/8451)
