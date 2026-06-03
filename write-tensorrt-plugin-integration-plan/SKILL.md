# Skill: Write a TensorRT Plugin Integration Plan

## Purpose
Guide the agent through planning how a custom CUDA kernel will be wrapped as a TensorRT plugin so it can be invoked from inside a TensorRT engine — covering API choice (IPluginV3 vs IPluginV2DynamicExt), the plugin lifecycle, dynamic shape handling, serialization, mixed precision (FP16/INT8/FP8), workspace management, CUDA graph compatibility, and the C++/Python binding strategy. The output is an integration plan with explicit decisions, not the plugin source code itself.

## Use this when
- A custom CUDA kernel must be deployed inside a TensorRT engine (production inference) and no built-in TRT layer covers it.
- An ONNX model contains an unsupported op that must be lowered to a custom plugin during ONNX → TRT conversion.
- A TRT-LLM-style codebase needs a new operator wrapped as a plugin to participate in engine builds.
- Migrating an existing IPluginV2DynamicExt plugin to IPluginV3 for a TensorRT 9.x or 10.x upgrade.

## Do not use this when
- The operation is already covered by a built-in TRT layer (most GEMMs, conv shapes, softmax, layernorm fused into MHA, common activations) — the built-in path is almost always faster than a plugin because it participates in TRT's graph optimizer and tactic selection.
- The op can be expressed as a composition of existing TRT layers (e.g. `IElementWiseLayer`, `IMatrixMultiplyLayer`, `IShuffleLayer`) — TRT's optimizer fuses these well; an opaque plugin blocks fusion.
- The deployment target is not TensorRT. For vLLM use a vLLM custom op; for ONNX Runtime use an ORT custom op; for raw PyTorch use `torch.library`.
- The kernel relies on JIT autotuning at runtime (e.g. Triton with Python-side autotune). TRT engines are built ahead of time and cannot host Python or runtime JIT inside `enqueue`.
- The kernel allocates host memory or synchronizes inside the launch path — incompatible with CUDA graph capture, which is how production TRT inference servers reach low latency.

## Inputs the agent should gather first
- **TensorRT version**: 8.x, 9.x, or 10.x. The plugin API changes are major across these — IPluginV3 was introduced in 9.x and is the only forward-compatible API in 10.x. Targeting TRT 8.6 means IPluginV2DynamicExt; targeting 9.3+ means IPluginV3 unless matching an existing codebase's convention.
- **Target precision(s)**: FP32, FP16, BF16, INT8, FP8 (E4M3/E5M2). Each precision the plugin must support adds a row to the format combination table and a code path in the kernel.
- **Dynamic shape requirements**: which input dimensions are dynamic, what are the min/opt/max profiles, and is multi-profile support required? Static-shape-only plugins are simpler but rarely sufficient for transformer inference.
- **Engine save/load required**: production engines are built once and loaded many times. If yes, full serialization (`getSerializationSize`, `serialize`, `deserialize`) must be designed correctly.
- **ONNX import path**: if the engine is built from an ONNX model, the plugin must register an ONNX op spec via `trtexec --plugins` or the ONNX-TRT parser's plugin namespace.
- **CUDA graph capture required**: most modern TRT inference (Triton Inference Server, TRT-LLM) captures into CUDA graphs. This rules out host allocations and stream synchronization inside `enqueue`.
- **C++ vs Python build path**: the engine builder is typically C++ for production, Python for prototyping. Plugin must be loadable from both.
- **Existing kernel's launch surface**: what does the kernel need at launch time — input/output pointers, shapes, scales, optional masks, workspace? This determines the plugin's attribute set and `enqueue` signature.

## Required reasoning process

1. **Choose the plugin API.** This is the first decision and constrains everything downstream:

   | API | TRT version | Recommended for | Notes |
   |---|---|---|---|
   | **IPluginV3** | 9.x+, required in 10.x | New plugins, long-term codebases | Splits responsibilities across `IPluginV3OneCore`, `IPluginV3OneBuild`, `IPluginV3OneRuntime`. Cleaner lifecycle, better forward compatibility. |
   | **IPluginV2DynamicExt** | 8.x, 9.x (deprecated path) | Matching an existing codebase that uses it (TRT-LLM 0.x, NVIDIA samples) | Single-class API. Still functional in 9.x but flagged for removal. |
   | **IPluginV2Ext / IPluginV2** | Legacy | Static-shape-only inference, almost never appropriate | No dynamic shape support; do not use for transformer or vision-with-variable-resolution workloads. |

   Default for a greenfield plugin on TRT ≥ 9.3: **IPluginV3**. Default when extending a codebase already on IPluginV2DynamicExt: match the convention; do not mix APIs in one project.

2. **Map the plugin lifecycle to your kernel.** For IPluginV3OneBuild + IPluginV3OneRuntime, every method has a specific role and serializing the wrong state into the wrong method is the most common bug class:

   - `getNbOutputs`, `getOutputDataTypes`, `getOutputShapes`: build-time, deterministic. Output shapes must be expressed symbolically in terms of input shapes (TRT supports a limited shape arithmetic via `IExprBuilder`).
   - `supportsFormatCombination`: build-time. Returns whether the plugin accepts a given combination of dtype + tensor format for each input/output. Get this wrong and TRT will either reject the plugin or insert reformat ops around it that destroy the performance gain.
   - `configurePlugin`: called once per engine build with concrete shapes from the optimization profile. Use it to compute and cache *attributes* derived from shapes, not runtime tensor pointers.
   - `getWorkspaceSize`: build-time. Return an upper bound on scratch memory needed for any shape in the profile. TRT owns and pools this memory across plugin instances.
   - `enqueue`: runtime. Receives `cudaStream_t`, `void* const*` inputs, `void* const*` outputs, `void* workspace`, and `PluginTensorDesc`. Launch the kernel here. **Do not** call `cudaMalloc`, `cudaMemcpy` from host, or any synchronizing API in this path.
   - `clone`: must produce a fresh plugin instance with the same attributes. Required because TRT may instantiate one plugin per execution context. Forgetting `clone` causes data corruption when running multiple contexts concurrently.

3. **Design the dynamic-shape contract.** TRT engines are built against optimization profiles, each with min/opt/max shapes. The plugin must:
   - Compute output shapes symbolically. If the output is `[B, H, S_q, S_kv]` from inputs `[B, H, S_q, D]` and `[B, H, S_kv, D]`, the symbolic expression goes in `getOutputShapes` using `IExprBuilder::operation(...)`.
   - Validate alignment requirements via `supportsFormatCombination`. If the kernel requires `S_kv % 16 == 0`, the plugin cannot silently accept arbitrary `S_kv` — declare the constraint or assert at runtime.
   - Handle the entire profile range. A plugin that works at the `opt` shape but segfaults at `min` (e.g. when `S_q == 1`) is the single most common production failure mode.

4. **Design the serialization contract.** Engines are built once and loaded N times in production:
   - Serialize *attributes* (kernel hyperparameters, precision flags, head dimension, scale factors set at construction time). These are static for the engine's lifetime.
   - Never serialize runtime state — workspace pointers, configured shapes from `configurePlugin`, CUDA stream handles. These are reconstructed on engine load.
   - The plugin creator (`IPluginCreatorV3One` for IPluginV3) is registered globally via `REGISTER_TENSORRT_PLUGIN` and looked up at deserialize time by `getPluginName()` + `getPluginVersion()` + `getPluginNamespace()`. A version mismatch silently fails with a confusing "plugin not found" error.
   - Bump the plugin version when changing the serialization format. Old engines must not deserialize against an incompatible new plugin.

5. **Plan the format combination table.** This is a table of `(input_dtype, input_format, output_dtype, output_format)` tuples the plugin advertises as supported. Decisions:
   - For an FP16 kernel: advertise `(kHALF, kLINEAR, kHALF, kLINEAR)` only. Do not advertise FP32 if the kernel does not support it — TRT will pick FP32 at the boundary and silently insert casts.
   - For INT8: advertise `(kINT8, kLINEAR, kINT8, kLINEAR)` and provide per-tensor scales via the calibration path. Mixed-precision (INT8 input, FP16 output for an epilogue) is a common pattern but doubles the format combinations.
   - For FP8 (E4M3 input, E5M2 output is unusual; usually both are E4M3): only on TRT 9.x+ targeting Hopper (sm_90a) or Blackwell. Advertise `kFP8` with `kLINEAR` and verify the minimum TRT version supports it.
   - Reformat-free input: declaring fewer format combinations forces TRT to insert reformat ops; declaring too many means the plugin must implement them all. Declare exactly the formats the kernel actually handles.

6. **Plan workspace memory.** TRT owns workspace; the plugin requests it:
   - `getWorkspaceSize` returns the *maximum* bytes needed across the profile range. Underestimating corrupts other ops' workspace (silent data corruption, hard to debug). Overestimating wastes device memory but is safe.
   - Inside `enqueue`, the workspace pointer is provided. Carve it up with explicit offsets; do not call `cudaMalloc`. The pointer is valid only for the duration of the `enqueue` call.
   - If the kernel needs persistent state across calls (rare — autoregressive KV cache is the main case), it must be passed in as an explicit input/output tensor, not held inside the plugin instance, because TRT may schedule plugin calls across multiple streams.

7. **Plan for CUDA graph capture.** Production TRT inference uses `enqueueV3` with CUDA graphs to amortize launch overhead:
   - No host allocation in `enqueue`. No `std::vector` resize, no `new`, no `cudaMalloc`, no `cudaMallocAsync`. All allocation must happen at `configurePlugin` or via `getWorkspaceSize`.
   - No synchronization in `enqueue`. No `cudaStreamSynchronize`, no `cudaDeviceSynchronize`, no event wait that targets a non-captured stream.
   - All kernel launches inside `enqueue` must use the provided `cudaStream_t`.
   - Triton autotuned kernels (Python-side autotune that JIT-compiles on first call) violate every one of these rules and are not TRT-plugin-compatible. The kernel must be ahead-of-time compiled CUDA C++ or pre-tuned PTX.

8. **Plan the build and binding artifacts.**
   - Build the plugin as a shared library (`libmyplugin.so`) linked against `libnvinfer.so` and `libcudart.so`. Export the creator via `REGISTER_TENSORRT_PLUGIN`.
   - Loading at runtime: `nvinfer1::initLibNvInferPlugins(&logger, "")` registers all built-in plugins; for custom plugins set `LD_LIBRARY_PATH` and call `dlopen` or use `IPluginRegistry::loadLibrary`.
   - For production, prefer static linking the plugin into the engine builder binary so version skew between the builder and the runtime cannot occur silently.
   - Python access: after the `.so` is on `LD_LIBRARY_PATH`, `tensorrt.init_libnvinfer_plugins(logger, namespace)` exposes the creator to the Python API. Use this for prototyping; production stays in C++.

9. **Validate the plan against existing alternatives.** Before committing:
   - Is there a built-in TRT layer that covers this op? Check the TRT operator reference for the target version. Built-in layers participate in tactic selection and graph fusion; plugins do not.
   - Can the op be expressed as a composition of existing layers? TRT's optimizer fuses elementwise chains and adjacent matmul + bias + activation patterns aggressively.
   - Does TRT-LLM already ship a similar plugin (RMSNorm, GPT attention, RoPE, paged KV cache)? Reuse it instead of writing a parallel implementation.
   - Has the kernel been benchmarked against the TRT built-in equivalent? If TRT's built-in is within 10% of the custom kernel, the maintenance cost of a plugin is rarely justified.

## Kernel design rules
- Treat the plugin class as a thin wrapper: no algorithmic logic in the plugin C++ file. The CUDA kernel lives in its own translation unit; the plugin's `enqueue` only marshals pointers and launches.
- Make every attribute that affects shape inference, output dtype, or workspace size a constructor parameter that is serialized. Runtime-derived state (configured shapes, cached descriptors) lives in members but is never serialized.
- Make `clone()` a deep copy of attributes. Never share mutable state between clones.
- Match plugin name, version, and namespace exactly between the creator and the plugin instance. Mismatches surface as deserialization failures with confusing messages.
- Implement `terminate()` and `destroy()` to release any device resources allocated in `configurePlugin` (e.g. cuBLAS handles, persistent device buffers). TRT calls these when the engine is destroyed.
- Document every assumption the kernel makes about input layout (contiguous, NCHW vs NHWC, alignment of `S_kv`, etc.) in a comment at the top of the plugin source. Future maintainers will not infer this from the kernel.

## Correctness requirements
- The plugin must produce bitwise-identical output to the standalone kernel for the same inputs and stream, given identical RNG state. Any divergence indicates either a marshaling bug in `enqueue` or a precision mismatch in the format combination table.
- The plugin must handle every shape in the optimization profile range, not just the `opt` shape. Test explicitly at `min`, `opt`, and `max` per profile.
- Engine save/load must round-trip. After serializing an engine to disk and loading it in a fresh process, the plugin output must match the freshly-built engine's output for the same inputs.
- Multiple concurrent execution contexts must produce independent, correct outputs. This validates `clone()` and the absence of shared mutable state.
- INT8 and FP8 paths must validate scales: a plugin that quietly drops the input scale produces garbage output that may still be in-range and pass shape checks.

## Performance requirements
- The plugin must not block TRT's graph fusion across its boundary unnecessarily. If declaring an extra format combination would let TRT fuse a preceding cast into the plugin, declare it.
- Workspace size must be tight. Returning gigabytes for a small kernel pessimizes co-located plugins and can cause OOM on multi-engine deployments.
- `enqueue` must not introduce host-device synchronization. Any sync (even an implicit one from a host-blocking API) destroys CUDA graph compatibility and adds tens of microseconds per call.
- Document the expected per-call overhead of the plugin (kernel time + TRT dispatch) and compare against the equivalent built-in or composition-of-layers path. A plugin that is slower than the built-in alternative should not be deployed.
- Do not claim performance parity with a hand-tuned TRT built-in without measurement on the target hardware and TRT version.

## Output format
The final response must include:
1. **API decision**: IPluginV3 vs IPluginV2DynamicExt with explicit justification (TRT version, codebase convention, forward compatibility).
2. **Lifecycle table**: each lifecycle method mapped to what the plugin does in it, including which attributes are read/written.
3. **Format combination table**: the exact `(input_dtype, format, output_dtype, format)` tuples the plugin advertises, with a precision support row for FP32, FP16, BF16, INT8, FP8 as applicable.
4. **Dynamic shape contract**: which input dims are dynamic, the symbolic output shape expression, and any alignment constraints declared in `supportsFormatCombination`.
5. **Serialization spec**: the full list of fields serialized, with sizes; explicit confirmation that no runtime state is serialized.
6. **Workspace plan**: upper-bound formula for `getWorkspaceSize`, with the layout of how the workspace is partitioned inside `enqueue`.
7. **CUDA graph compatibility checklist**: confirmation that no host alloc, no sync, all launches on the provided stream.
8. **Build/binding plan**: shared library name, link line, registration mechanism, and Python loading path if applicable.
9. **Risk register**: list of known failure modes for this specific plugin (precision edge cases, profile boundary shapes, version compatibility).

## Common failure modes
- **Serializing runtime state instead of attributes**: the plugin saves shape descriptors from `configurePlugin` into the serialized blob. Engine fails to load on a second run because the descriptors point to freed memory.
- **Wrong workspace size**: `getWorkspaceSize` returns the size for the `opt` shape only. At `max` shape the plugin overruns into the next op's workspace, causing data corruption that manifests two layers downstream and is nearly impossible to debug.
- **Plugin name/version mismatch between creator and instance**: deserialization fails with "no creator registered for plugin X version Y" and the actual mismatch is a typo or a forgotten version bump.
- **Dynamic shape edge case at min profile**: kernel works at `S_q == 64` (opt) but segfaults at `S_q == 1` (min). The min profile is rarely tested in development; production hits it on the first short request.
- **Forgetting to override `clone()`**: TRT silently shares the plugin instance across execution contexts; concurrent inference produces interleaved outputs that look like a numerical bug.
- **Host allocation inside `enqueue`**: the plugin uses `std::vector<int> tmp(N)` as scratch. Single-stream inference works; CUDA graph capture fails with `cudaErrorStreamCaptureUnsupported` at deployment time.
- **Triton-autotuned kernel wrapped as a plugin**: the kernel JIT-compiles on first call. The first call inside a captured graph fails because Python is not callable from a captured stream. The agent must be told upfront that Triton autotune and TRT plugins are incompatible.
- **Reformat ops inserted around the plugin**: the plugin advertises only `kLINEAR` `kHALF` but its neighbors run in `kHWC8` `kHALF`. TRT inserts reformat ops on both sides; the plugin's measured speedup vanishes.
- **INT8 plugin without proper scale handling**: the plugin reads the input pointer but ignores `PluginTensorDesc::scale`. Output is numerically wrong but in-range, passes basic shape tests, and is caught only by an end-to-end accuracy regression.
- **Missing ONNX op registration**: the plugin works when building the engine from C++ but the ONNX → TRT path cannot find the op. Requires registering the op name in the ONNX parser's plugin namespace.

## Review checklist
- [ ] Has the TRT version been pinned and the API choice (IPluginV3 vs IPluginV2DynamicExt) justified against it?
- [ ] Is every lifecycle method mapped to a specific responsibility, with a clear separation between build-time and runtime work?
- [ ] Are output shapes expressed symbolically via `IExprBuilder` rather than computed only at runtime?
- [ ] Does the format combination table list exactly the precision/format pairs the kernel actually supports — no more, no less?
- [ ] Is `getWorkspaceSize` an upper bound across the entire optimization profile range?
- [ ] Has it been confirmed that nothing serialized is runtime-derived state?
- [ ] Has plugin name, version, and namespace been chosen and held consistent between creator and instance?
- [ ] Has `clone()` been designed and tested for multi-context concurrency?
- [ ] Is `enqueue` free of host allocations and synchronization, with all launches on the provided stream?
- [ ] If INT8 or FP8: is the scale path explicit, with the calibration or quantization spec documented?
- [ ] Has every shape in min/opt/max for every profile been listed as a test case?
- [ ] Has the alternative of using a built-in TRT layer or a composition of existing layers been ruled out with measurement, not assumption?
- [ ] If the kernel uses runtime autotuning (Triton or otherwise), has the plan replaced it with an ahead-of-time compiled equivalent before plugin wrapping?
