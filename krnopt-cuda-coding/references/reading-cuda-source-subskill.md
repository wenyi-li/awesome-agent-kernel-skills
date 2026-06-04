# Reading CUDA Source Subskill

Use this integrated subskill when the main question is "how should I read this
existing CUDA code and tell whether it is structurally good?"

## Reading Order

Read the code in this order:

1. reconstruct thread, warp, and block ownership
2. check bounds, synchronization, and numeric semantics
3. trace lane-to-address behavior for loads and stores
4. identify where reuse is created or missing
5. inspect shared-memory layout and register-heavy private state
6. inspect divergence, synchronization shape, and instruction or datatype path

## What Reading Should Produce

Reading should produce:

- a plain-language model of what the kernel is doing
- a short list of correctness risks, if any
- a short list of likely structural performance issues, if any
- a judgment about whether profiling is needed next

## Reading Discipline

- do not confuse suspicious structure with proven runtime dominance
- do not start from counters when the source already reveals a semantic bug
- do not summarize at the level of syntax; summarize at the level of mappings,
  reuse, waits, and hardware path
- prefer one or two concrete hypotheses over a broad wall of maybe-problems
