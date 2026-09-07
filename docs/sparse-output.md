# Source-backed sparse output

Pinned source: `f69cbb46e23bc2f39cda5722ce096f56301dab4f`.

`SparseTensor::gemm_sparse_dense` implements sparse A times dense B into sparse
C using CCSR (source krnl_type 5). Sparse A panels use count/value broadcasts;
dense B panels use the existing dense broadcasts. C is accumulated in compressed
rows with beta only on the first k panel, then emitted directly as coordinates
and redistributed to its original layout. No full dense output or full CSR
row-pointer array is constructed for this path.

`SparseTensor::contract_from_sparse_dense` supplies the existing fold pipeline:
input-only summation, repeated-label extraction/reinsertion, indexed permutation,
reshape, batch slicing, CCSR product and output-layout restoration. Sparse
structure, including represented zero rows and old-only entries when beta is
nonzero, follows the source kernel. The existing CCSR kernel's omission of a
final all-zero B column remains a source padding convention, not general zero
pruning.

## Do not invent a nonfolded sparse-output leaf

`contraction.cxx:4313-4335` selects CSR*CSR->CSR (krnl4) and sparse*dense->CCSR
(krnl5) only after folding/inner construction. Its non-inner branch explicitly
asserts that B and C are dense and selects sparse-A/dense-B/dense-C (krnl0).
Dispatch is in `spctr_tsr.cxx:505-524`. Therefore an unrestricted nonfolded
sparse-output leaf is not a missing working source algorithm to manufacture.

Source ordinary dense-A/sparse-B can swap operands; custom sparse-B is rejected
(`contraction.cxx:5382-5388`). Dense+dense->sparse computes through a dense output
then sparsifies (`:5373-5379`); it is not a native sparse-output kernel.

These ordinary storage branches are now exposed as `SparseTensor::gemm_dense`,
`gemm_dense_sparse`, `contract_from_dense` and `contract_from_dense_sparse`.
Dense+dense uses a distributed dense copy of old C and transfers the final
owned sparse blocks back without changing C's context or distribution. The
source's post-contraction predicate is literally `v != caddid` on pointers,
not values: every valid output entry, including zero, is retained. No numerical
zero pruning is substituted for that source quirk.

For dense-A/sparse-B, the source ordinary branch swaps the operands without a
commutativity check. Matrix execution transposes operands/output and reverses
the process grid before using the sparse-first CCSR path; indexed execution
swaps operand labels. Noncommutative elements consequently multiply as B*A,
not A*B. This behavior is preserved and tested, not silently corrected.

These source boundaries do not establish full Rust automatic dispatch:
automatic sparse plan assembly and nested moving-output communication remain
unfinished. Sparse virtual/replicated communication in the source wraps these
inner CSR/CCSR leaves rather than supplying a new general scalar sparse output.
