# Case Snapshots

Use this reference when the user would benefit from concrete pattern memories
instead of abstract structural rules.

## ADO-Style Decomposition Case

Pattern:

- underfill first
- then uncoalesced access
- then shared-memory reduction pressure
- then real global-load limit near roofline

Structural takeaway:

- the right final move may be to split the work and hand the dense core to a
  tuned GEMM instead of continuing local kernel-body tuning

## MoE Dispatch And Compute Case

Pattern:

- dispatch, padding, combine, or materialization traffic dominates
- grouped compute itself may not be the primary problem yet

Structural takeaway:

- compare standard grouped GEMM against fused scatter-GEMM, static batching,
  and epilogue fusion rather than only retuning the GEMM body

## Attention Overlap Case

Pattern:

- the problem is not only math throughput but serialized stage order inside the
  CTA

Structural takeaway:

- redesign the internal schedule with producer-consumer roles, overlap, and
  staged buffers rather than treating the kernel as one flat loop

## Quantized Path Case

Pattern:

- low-precision arithmetic exists, but Q/DQ or scale work still materializes
  extra tensors

Structural takeaway:

- move quant, dequant, or scaling into the prologue or epilogue if the exposed
  tile structure allows it
