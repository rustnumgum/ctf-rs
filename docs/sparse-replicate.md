# Sparse-output replicated/virtual contraction

`sparse_contraction_comm` ports the sparse-output portion of pinned
`spctr_replicate::run` and connects it to the existing virtual-block traversal:

- `replicated_csr`: CSR A / CSR B -> CSR C.
- `replicated_ccsr_dense`: CCSR A / dense B -> CCSR C.

Callers supply explicit input replication fibers, ordered orthogonal output
fibers, virtual dimensions/operand indices, matrix blocks and alpha/beta.
Sparse blocks broadcast their variable serialized sizes and stored entries;
dense B broadcasts only the fixed-size value buffer implied by its local block
shapes. No callback backend or C++ packed-buffer ABI is introduced.

Only the intersection of output-fiber roots retains old C and beta. Other
ranks start with empty C/zero beta. `sparse_virtual::execute` applies beta once
per visited output block, then one on later contributions. Native CSR/CCSR
leaves compute sparse results and invoke the partitioned matrix reductions on
each output fiber. Following the source, nonroots stop before later orthogonal
fibers; arbitrary overlapping communicator lists are not a supported topology.

The return value is `Some(blocks)` only at the output-root intersection; with
no output fibers, every caller receives its local result. Temporary replicated
sparse input vectors are cleared on nonroots; dense B replicas are zeroed.
Caller-owned fibers are borrowed, not freed by this operation. Temporary
subcommunicators inside sparse matrix reduction are explicitly closed.

This implements the explicit replicated/virtual sparse-output layer. The full
automatic sparse plan builder remains unfinished. Sparse node-aware reordering
is disabled by the pinned source (source-node-aware-boundary.md).
Explicit CSR/CCSR nested moving-output levels are now available
separately in sparse-2d.md; full automatic composition is not claimed.
