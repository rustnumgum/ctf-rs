# Sparse contraction virtualization

`sparse_virtual::execute` ports the `spctr_virt::run` block traversal in
`contraction/spctr_tsr.cxx:699-898`. Operand strides multiply virtual dimensions
in operand-axis order; repeated labels add their strides. A dimension-zero-first
odometer updates the three operand block offsets. Each output block receives
beta on its first call and the multiplicative identity on subsequent calls.
Off-diagonal blocks excluded by repeated output labels are not modified. An
empty union invokes the scalar child exactly once.

The layer does not reinterpret values or allocate dense storage. A child closure
can update dense slices or replace variable-length sparse output buckets; Rust
ownership replaces the source's byte pointers and conditional deallocations.

The production `contract_from_sparse_dense_on` and custom-function counterpart
now accept `virtual_factors: &[(u8, usize)]` after `physical_labels`. Factors are
positive multiplicities of the corresponding label's physical phase. An empty
list explicitly requests no virtualization. Stored sparse entries are moved to
their virtual block and local key; dense B/C are sliced by block offsets. The
existing fiber broadcasts and output reductions still execute distributively.
There is no dense materialization of sparse A or whole-tensor gather.

This integrates virtualization for sparse-A/dense-B/dense-C, including empty
sparse blocks and the pinned local custom-function branch. The mapped interface
retains its unique-label and A-only-label restrictions. Variable sparse output
bucket traversal is also used by the explicit CSR/CCSR replicated sparse-output
layer (sparse-replicate.md), including variable output block reductions.
Explicit CSR/CCSR moving-output levels can now recurse (sparse-2d.md);
automatic plan assembly and the remaining mixed/raw variants remain unfinished.
