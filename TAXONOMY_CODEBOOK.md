# GPU Numerical Failure Taxonomy Codebook

This codebook defines labels used for the seed study. Labels are produced by a transparent LLM-assisted pass over public issue titles and bodies. They should be treated as research annotations, not as official project labels.

The project now has three label layers:

- Candidate labels over the 930-issue seed dataset, generated from issue title/body fields.
- Context-adjudicated validation labels over an 82-issue stratified subset, generated after fetching full public issue bodies and comments.
- Human-adjudicated gold labels over a 191-issue packet, generated from two completed blind human annotation passes and adjudication.

## Primary failure labels

`nan_inf`
: Issue reports NaN, Inf, infinite values, non-finite values, or numerical contamination by invalid values.

`overflow_underflow`
: Issue reports overflow, underflow, range saturation, exponent blow-up, or values too large/small for the target representation.

`precision_tolerance`
: Issue reports wrong numerical outputs, tolerance failures, rounding differences, inaccurate results, or mismatch against a reference implementation.

`dtype_casting`
: Issue centers on dtype semantics, casting, promotion, unsupported types, fp16/bf16/fp32/fp64 differences, integer width, or complex dtype handling.

`crash_compile`
: Issue primarily reports compilation failure, assertion, crash, or runtime exception in a numerical/kernel path.

`performance_only`
: Issue includes numerical keywords but is primarily a performance/throughput regression rather than a correctness failure.

`not_numerical_failure`
: Gold-annotation label for false positives where the issue matched query terms but does not describe a numerical correctness failure.

`needs_review`
: Issue matched collection queries but lacks enough clear evidence in title/body or validation context for a primary label.

## Secondary suspected-cause labels

`memory_mask_bounds`
: Mentions masks, boundaries, strides, descriptors, pointers, offsets, broadcasting, or out-of-bounds behavior.

`compiler_codegen`
: Mentions compiler, lowering, code generation, PTX, LLVM, XLA, Inductor, fusion, or nightly regressions.

`async_race_ordering`
: Mentions asynchronous execution, synchronization, streams, barriers, races, nondeterminism, or ordering.

`hardware_backend`
: Mentions CUDA/GPU/backend-specific devices, architectures, accelerators, ROCm, MPS, XLA, or similar runtime context.

`reduction_accumulation`
: Mentions reductions, summation, accumulation, softmax, attention, matrix multiplication, exponentials, logarithms, or related accumulation-heavy kernels.

`api_semantics`
: Mentions API compatibility, NumPy/PyTorch/JAX semantic differences, function signatures, unsupported operations, or user-visible behavior that is not purely arithmetic.

`environment_configuration`
: Mentions installation, CUDA/driver/library loading, paths, package versions, build settings, or runtime configuration as the likely cause.

`unknown`
: Use when the issue is classifiable by symptom but the cause is not supported by public evidence.

## Labeling limitations

- Labels are assigned from public issue title/body text, not private reproductions.
- Some issues contain multiple failure modes; the primary label is the first salient failure category.
- GitHub search results include false positives, especially performance-only issues containing NaN in benchmark tables.
- The validation subset uses full issue/comment context and stores evidence snippets, but it is still a research adjudication pass rather than the primary gold benchmark.
- The 191-issue gold benchmark uses two independent blind human annotation passes, adjudication, evidence quotes, and agreement reporting.

## Adjudication boundary refinements

- Prefer `dtype_casting` over `precision_tolerance` when dtype, casting,
  promotion, fp16/bf16/fp32/fp64, integer width, or complex dtype behavior is
  the decisive evidence.
- Prefer `overflow_underflow` over `dtype_casting` when range blow-up,
  underflow, saturation, or integer wraparound is the observed failure; keep
  dtype as secondary context when applicable.
- Prefer `nan_inf` over `precision_tolerance` when non-finite values are the
  observed symptom, not merely a test fixture or placeholder.
- Prefer `crash_compile` when the primary user-visible failure is a compiler,
  assertion, or runtime exception.
- Use `not_numerical_failure` for query false positives, even when terms such
  as `nan`, `dtype`, or `precision` appear incidentally.
