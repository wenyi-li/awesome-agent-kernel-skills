# Hot-Path Partitioning

Use this reference when the optimization target is better understood as a
sequence of parts than as one blob called "the kernel."

The KB repeatedly shows that the hottest line is not always the right redesign
site. The real mismatch may sit at the boundary between parts.

## Partition The Hot Path

Use this decomposition:

```text
[launch/orchestration]
    -> [ingress/layout/metadata]
    -> [main compute loop]
    -> [reduction/sync]
    -> [epilogue/postprocess]
    -> [materialize/handoff]
```

## Questions By Part

### Launch or orchestration

Ask:

- are there many tiny launches
- are host gaps or explicit syncs visible
- is launch fragmentation itself the bottleneck

Typical redesigns:

- fusion
- batching
- CUDA Graphs
- persistent execution

### Ingress, layout, or metadata

Ask:

- are sorting, prefix maps, descriptors, or padding expensive
- are full payload copies being used just to encode schedule information
- is descriptor churn dominating irregular grouped work

Typical redesigns:

- compact index arrays
- prefix maps
- descriptor pools
- index-driven execution
- metadata compaction instead of payload movement

### Main compute loop

Ask:

- is the ownership model wrong
- is the primitive wrong
- is the schedule mismatched to irregularity
- is the kernel reimplementing a tuned primitive badly, and if so has the
  standard-library-style ladder been checked before handcrafted replacement

Typical redesigns:

- precompiled vendor primitive substitution
- header/template or generated-kernel substitution
- grouped GEMM reformulation
- block-sparse reformulation
- retiled ownership
- handcrafted kernel only when higher-tier primitives fail feature,
  integration, performance, or fusion requirements

### Reduction or synchronization

Ask:

- does one CTA own too much serialized work
- are block-wide sweep patterns dominating
- is the barrier domain larger than it needs to be

Typical redesigns:

- warp-level reductions
- shorter barrier domains
- different reduction tree
- pipelined stage order

### Epilogue or postprocess

Ask:

- are activation, scaling, dequant, combine, or writeback separate kernels
- does the producer already hold the data hot when the next pass begins

Typical redesigns:

- epilogue fusion
- prologue fusion
- fused scaling or dequant

### Materialize or handoff

Ask:

- is the next consumer forcing a pointless HBM round-trip
- are intermediates materialized only to be immediately consumed

Typical redesigns:

- boundary elimination
- producer-consumer fusion
- decomposition into a better primitive sequence

## Reading Rule

Do not just ask "where is the hottest line?" Ask:

1. which part spends the bytes
2. which part spends the launches
3. which part owns the wait
4. which boundary makes the next part expensive
