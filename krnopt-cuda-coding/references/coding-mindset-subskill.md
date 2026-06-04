# Coding Mindset Subskill

Use this integrated subskill when the first question is not "what does the
profiler say?" but "how should a CUDA programmer read and shape this code?"

## Core Rules

- Correctness is a prerequisite for optimization.
- Think in thread-to-data mappings, not just source statements.
- Write for warp behavior, memory transactions, shared-memory banks, and
  register budgets.
- Make performance intent visible in the code structure.
- Prefer disciplined simplicity over opaque cleverness.

## Ordered Questions

1. Is the code correct and easy to validate again?
2. What does one thread, one warp, and one block compute?
3. What data moves from global memory, and where is reuse created?
4. What waits, diverges, spills, or falls onto the wrong instruction path?
5. Which one or two source-level hypotheses deserve profiling or a patch next?

## What This Mindset Should Prevent

It should stop the user from:

- treating a kernel as a pile of statements instead of a machine mapping
- optimizing before re-establishing correctness checks
- adding shared memory without a reuse or access-pattern reason
- chasing occupancy blindly instead of balancing it against locality and spills
- assuming a suspicious source pattern automatically dominates runtime

## Relationship To Other Guidance

- Use `correctness-first-subskill.md` to check semantics and validation posture.
- Use `mapping-and-memory-subskill.md` for coalescing, reuse, and bandwidth
  reasoning.
- Use `shared-register-residency-subskill.md` for shared-memory, register, and
  occupancy tradeoffs.
- Use `control-flow-and-dtype-subskill.md` for control, sync, and instruction
  path checks.
- Use `krnopt-cuda-profiling` once runtime dominance must be measured.

## Doctrine From CUDA Best Practices

The NVIDIA CUDA Best Practices Guide and Programming Guide converge on a small
set of attitudes that should sit under every concrete decision in this skill.

### Think in mappings, not in statements

A line of CUDA source is really a statement about which threads touch which
addresses, which values stay live, and which warps or blocks must wait. The
fastest way to evaluate a kernel is to redescribe it at three levels before
reading its body:

- what does one thread compute?
- what does one warp compute?
- what does one block compute?

If those three questions cannot be answered from the source in under a minute,
the kernel is already hard to optimize.

### Write for the machine's structure, not for the compiler's convenience

Warps (32-thread execution groups), memory transactions (combined lane requests
into wide accesses), shared-memory banks, and register budgets are part of the
design surface. Good CUDA code makes these legible:

- thread-block sizes are multiples of 32
- adjacent lanes usually touch adjacent addresses
- reuse hierarchy (global -> shared -> register) shows up in the structure
- synchronization scope appears where the data-sharing semantics require it

### Prefer disciplined simplicity over decorative cleverness

Optimization passes should be scoped, measurable, and testable. Make one
optimization family at a time and keep the correctness path cheap to re-run.
Opaque rewrites that combine several changes at once are hard to bisect when
they regress. The best practices guide's APOD loop (Assess, Parallelize,
Optimize, Deploy) is essentially this discipline made explicit.

### Treat bandwidth as a design budget

Every extra load, over-fetch, or redundant movement should be suspected. The
guide's memory chapter opens with the blunt claim that memory optimization is
often the most important performance work, and its priority order is:

1. minimize host-device traffic (PCIe is slow enough to dominate)
2. coalesce global accesses
3. remove redundant loads via shared memory or cache locality
4. fix bank conflicts
5. check for register spills into local memory
6. only then pursue narrower tricks

### Do not maximize occupancy blindly

Occupancy is a latency-hiding metric, not an end state. Low occupancy is
dangerous because memory and dependency latency become harder to hide. Higher
occupancy is not automatically better because it can force register budgets
that spill, smaller tiles that weaken reuse, or block shapes that fight the
data layout. The right question is never "is occupancy high?" but "which
resource limits residency, and would raising that limit actually hurt
reuse or tile size?"

### Keep control flow warp-aware

Short, simple, warp-aligned decisions are friendlier than ornate branch
structure. When a condition depends on thread ID, write it so whole warps take
the same path. Branching on `threadIdx.x / warpSize` is the canonical shape:
the same warp stays on one side of the branch. Tiny conditionals can often be
predicated away by the compiler; help that path by keeping branch bodies short.

### Avoid expensive generality in the hot path

Generic math, accidental FP64 promotion from literals, and broad abstractions
can silently pick slow instruction paths. Division and modulo, trig on large
arguments, and `pow` for small integer exponents all have well-known cheaper
alternatives. Performance intent should be visible, not accidentally lost in
helper layers.

## The Programmer's Working Loop

When reading or shaping a kernel, walk the same path each time:

1. make the kernel correct and easy to validate
2. read it as a thread-to-data mapping
3. inspect memory traffic and reuse
4. inspect shared-memory layout, synchronization shape, and register footprint
5. inspect instruction and datatype path
6. form one or two concrete correctness or performance hypotheses
7. confirm them with profiling before overcommitting

Pure source inspection is excellent for generating hypotheses and for rejecting
obviously bad structures, but it is not enough by itself to prove what
dominates runtime. Source inspection owns the programmer's mindset; profiling
owns the diagnosis mindset.
