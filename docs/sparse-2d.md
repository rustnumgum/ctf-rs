# Explicit sparse 2D contraction levels

`sparse_2d::execute_csr` and `execute_ccsr_dense` port the corresponding
CSR/CSR/CSR and CCSR/dense/CCSR branches of pinned spctr_2d_general.cxx.
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
- Moving sparse C starts each child from empty output with zero beta, reduces
  contributions, then adds old C **unscaled**. The source top-level sparse
  contraction instead uses an empty temporary and applies user beta through
  its later summation; do not substitute dense ctr_2d beta rules here.
- If the edge divides the layer count, layer quotient/remainder passes the
  remaining layers to the child. If the layer count divides the edge, each
  layer executes its step subset. Otherwise all steps retain the child layers.

These are explicit recursive sparse-output levels. Automatic sparse plan
assembly, dense-output mixed 2D variants, raw nonfolded sparse-pair integration
and sparse node-aware execution remain unfinished.
