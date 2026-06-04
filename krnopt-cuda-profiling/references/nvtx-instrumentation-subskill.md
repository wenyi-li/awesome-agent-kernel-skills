# NVTX Instrumentation Subskill

Use this integrated subskill when the user needs phase and launch correlation
for profiling, especially in `nsys`.

## What NVTX Is For

NVTX is for correlating profiler evidence with:

- high-level phases
- iterations
- wrappers
- benchmark cases
- launch paths

It is not exact kernel source-line attribution.

## Recommended Annotation Targets

Add NVTX around:

- end-to-end pipeline phases
- one benchmark iteration or one representative trial
- wrapper functions that launch several kernels
- variant or workload labels when multiple paths are compared

Prefer small, consistent names over verbose prose.

## Useful Patterns

### Phase ranges

```cpp
nvtxRangePush("prefill");
run_prefill();
nvtxRangePop();
```

### Wrapper correlation

```cpp
nvtxRangePush("router-wrapper");
launch_router_kernels(...);
nvtxRangePop();
```

### Iteration labeling

```cpp
char name[64];
snprintf(name, sizeof(name), "iter-%d", iter);
nvtxRangePush(name);
run_iteration(iter);
nvtxRangePop();
```

## Important Boundary

Teach this explicitly:

- NVTX + `nsys` -> phase and launch-path correlation
- `-lineinfo` + focused `ncu` -> exact kernel source attribution

Do not imply that NVTX alone can identify the hot source line inside a kernel.
