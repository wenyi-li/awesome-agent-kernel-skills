# NVTX Correlation Subskill

Use this integrated subskill when NVTX appears in the evidence.

## What NVTX Is Good For

Use NVTX to correlate:

- model or pipeline phases
- iteration boundaries
- wrapper functions or launch paths
- groups of kernels that belong to one logical stage

This is useful for deciding which kernel group deserves optimization attention.

## What NVTX Does Not Prove

NVTX does not by itself prove:

- which source line inside the kernel is slow
- whether a load is uncoalesced
- whether a shared-memory layout has bank conflicts
- whether occupancy or tensor-pipe utilization is the dominant limiter

Use NVTX for phase correlation, then combine it with source structure and
`ncu` evidence when you move from "where" to "why".

## Optimization-Direction Use

When NVTX is the main context signal:

- identify the hot phase or wrapper
- identify the kernels launched inside it
- connect those kernels to the relevant source files or kernel templates
- state which parts are phase-backed facts versus source-level guesses
