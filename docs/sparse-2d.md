# Explicit sparse 2D contraction levels

`sparse_2d::execute_csr` and `execute_ccsr_dense` port the corresponding
CSR/CSR/CSR and CCSR/dense/CCSR branches of pinned spctr_2d_general.cxx.
`execute_csr_dense` and `execute_csr_sparse_dense` cover its folded mixed
CSR/dense/dense and CSR/CSR/dense branches. Sparse inputs remain CSR; dense
output uses native MPI Reduce rather than sparse structural reduction.
`execute_pairs_dense` covers the nonfolded raw sparse-A/dense-B/dense-C
branch. Each A block is a sorted local-key/value list; byte sizes precede
the concatenated pair payload. Keys, ordering and explicitly stored zeros
are preserved without matrix conversion. The child supplies local shapes
and labels to sparse_sequential::sequential (or its custom-function variant).
They accept an explicit edge, Layers, A/B/C Panels and a child contraction.
The child returns resized sparse blocks and can invoke another 2D level or
a native sparse leaf; this matches the source recursive child, not a backend
plugin interface.

Panel.outer counts strips; Panel.inner counts consecutive matrix blocks in
each strip. Zero inner denotes a stationary whole operand. Communicated panels
have cyclic step owners and require edge divisibility by communicator size.
Sparse input broadcasts carry per-block lengths and stored entries; dense B
uses its known fixed shape. Moving output reduces each block to its cyclic
owner and restores the local strip order. No full-input gather is used.

## Literal beta and layer behavior

- Stationary whole C receives beta on the first executed step, then one.
- Stationary contiguous C applies beta to each independently selected panel.
- Stationary strided sparse C starts an empty panel and replaces old output.
- Stationary strided dense C starts a zero panel and scatters new + beta*old.
- Moving dense C reduces a zero-initialized child panel to its cyclic owner,
  which scatters new + beta*old. This differs from the sparse-C convention.
- Moving sparse C starts each child from empty output with zero beta, reduces
  contributions, then adds old C **unscaled**. The source top-level sparse
  contraction instead uses an empty temporary and applies user beta through
  its later summation; do not substitute dense ctr_2d beta rules here.
- If the edge divides the layer count, layer quotient/remainder passes the
  remaining layers to the child. If the layer count divides the edge, each
  layer executes its step subset. Otherwise all steps retain the child layers.

These are explicit recursive folded and raw sparse/mixed levels. Automatic
tensor-to-plan assembly and sparse node-aware execution remain unfinished;
the raw executor does not infer a SparseTensor's mappings or local key space.
