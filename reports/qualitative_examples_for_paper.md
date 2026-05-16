# Qualitative Examples for Conference Version

## Clear taxonomy examples

- `dtype_casting`: triton-lang/triton#2176 (add libdevice.remquo). Evidence: dtype("fp32"), core
- `overflow_underflow`: cupy/cupy#6715 (Wrong overflow with matmul of uint16 arrays). Evidence: Wrong overflow with matmul of uint16 arrays ### Description The following arrays give an appar
- `nan_inf`: triton-lang/triton#1121 (Value 'sm_89' is not defined for option 'gpu-name'). Evidence: il but training with the compiled model does fail (showing `nan` as a loss after the 1st iteration)
- `performance_only`: numba/numba#2988 (Support For CPU Atomics). Evidence: c/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent) requires atomic floating-point addition
- `crash_compile`: numba/numba#5158 (DeviceNDArray.bind() does not seem to bind the stream to self). Evidence: DeviceNDArray.bind() does not seem to bind the stream to self
- `not_numerical_failure`: rapidsai/cudf#16029 ([QST] TypeError: Argument 'real' has incorrect type (expected numpy.ndarray, got ndarray)). Evidence: [QST] TypeError: Argument 'real' has incorrect type (expected numpy
- `precision_tolerance`: pytorch/pytorch#166131 (Return a view from tensor(requires_grad=False) in autograd function may cause incorrect requires_grad attribute.). Evidence: view from tensor(requires_grad=False) in autograd function may cause incorrect requires_grad attribute

## Boundary/error examples

- Gold `dtype_casting` predicted as `precision_tolerance`: pytorch/pytorch#175156 ([inductor] Multiple randint calls cause inconsistent RNG results between eager and compiled mode after fixing the rng seed).
- Gold `nan_inf` predicted as `precision_tolerance`: pytorch/pytorch#181146 (torch.compile: autograd.Function.apply with aliased inputs drops per-slot gradient contributions).
- Gold `overflow_underflow` predicted as `precision_tolerance`: pytorch/pytorch#180026 (DISABLED test_combo_kernel_yz_overflow (__main__.ComboKernelTestsPerSubkernelBlocks)).
- Gold `not_numerical_failure` predicted as `dtype_casting`: rapidsai/cudf#16029 ([QST] TypeError: Argument 'real' has incorrect type (expected numpy.ndarray, got ndarray)).
- Gold `overflow_underflow` predicted as `dtype_casting`: cupy/cupy#6715 (Wrong overflow with matmul of uint16 arrays).
- Gold `not_numerical_failure` predicted as `crash_compile`: numba/numba#4713 (Numba is not detecting the icc_rt libraries when installed with pip).
