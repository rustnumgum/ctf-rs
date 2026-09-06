# Source provenance

Reference repository: https://gitlab.cc4s.org/cc4s/ctf
Reference commit: f69cbb46e23bc2f39cda5722ce096f56301dab4f

`LICENSE` preserves the upstream notice and terms for adapted material. It is
not a declaration that Edgar Solomonik authored the new Rust implementation.
Do not stamp independently written files with the upstream author's copyright.

* `src/mapping.rs`: map-chain phase/rank and topology reorder/inverse routines
  adapted from `src/mapping/{mapping,topology,distribution}.cxx`.
* `src/map_tensor.rs`: physical-axis assignment and symmetry-phase coordination
  adapted from `map_tensor`/`map_symtsr` in `src/mapping/mapping.cxx`.
* `src/topology_candidates.rs`: ordered factorization, adjacent folding and
  permutation/folding enumeration adapted from `src/mapping/topology.cxx`.
* `src/node_aware.rs`: inter-node grid factor assignment/tree enumeration adapted
  from `src/mapping/node_aware_dist.cxx`, whose author notice names Andreas Irmler;
  its corresponding upstream header carries the 2022 Edgar Solomonik notice.
* `src/tensor.rs`: Rust distributed storage implementation using the upstream
  cyclic distribution and key-bucket exchange approach; not a literal complete
  port of all optimized redistribution kernels.
  Dense slice extraction follows `redistribution/slice.cxx`'s local extraction
  and physical cyclic rank-shift communication; Rust reindexes virtual blocks
  explicitly rather than relying on raw byte copies.
* `src/algebra.rs`, `src/context.rs`, `src/ffi/mpi.rs`: new Rust traits, lifetimes,
  serialization and native-call encapsulation. Upstream responsibility references
  document compatibility targets, not authorship of these files.
* `src/summation.rs`: local NS sequential summation control flow and alpha/beta
  ordering adapted from `src/summation/sym_seq_sum.cxx`; Rust index offset tables
  replace byte offsets and explicit C++ buffer management.
  Its virtual-block and replicated-block layers follow `tsum_virt::run` and
  `tsum_replicate::run` in `src/summation/sum_tsr.cxx`.
* `src/contraction.rs`: dense NS reference contraction and custom function/alpha
  ordering adapted from `src/contraction/sym_seq_ctr.cxx`.
  Virtual traversal follows `ctr_virt::run` in `ctr_tsr.cxx`; replicated execution
  follows `ctr_replicate::run` in `ctr_comm.cxx`, including root reduction and
  input-replica clearing.
  Folded CPU batch layout follows `interface/semiring.cxx::gemm_batch`; output
  operand swapping and prescaling follow `sym_seq_ctr_inr`.
* `src/ctr_2d.rs`: panel packing/broadcast, output reduction/scatter and layer
  propagation adapted from `ctr_2d_general::run` and `find_bsizes` in
  `src/contraction/ctr_2d_general.cxx`.
* `src/diagonal.rs`: repeated-coordinate key projection/insertion follows the
  index deletion/insertion rules in `tensor::extract_diag`. Current dense data
  transfer uses canonical-key redistribution, not the optimized upstream dense
  mapped-summation path; do not claim exact communication parity for this path.
* `tests/upstream_dense.rs`: numerical identities from `test/diag_ctr.cxx` and
  `test/reduce_bcast.cxx`, retaining original residual metrics and tolerances.
* `src/linalg.rs`, `src/ffi/linalg.rs`: new local-kernel interface and bindings to
  standard BLAS/LAPACK operations. No C++ CTF wrapper is linked or copied here.

Keep attribution with actual adaptations and add per-source provenance as the
port grows. The source/test inventory is a scope ledger, not a claim that all
listed source files have been ported.

## Sparse matrix formats (2026-09-07)

`src/sparse_formats.rs` adapts `interface/set.h` COO conversion ordering and
prefix sums, `sparse_formats/{csr,ccsr}.cxx` cyclic partitions/assembly and
symbolic union/scatter addition, and `interface/semiring.h` gen_csrmm,
gen_csrmultcsr and gen_ccsrmm. One-based logical indices and structural zeros
are retained. C++ aligned byte-buffer headers are replaced by owned Rust vectors.
CCSR addition shares the CSR row-union kernel for overlapping compressed rows;
it never expands the full logical row dimension.

Source memory mistakes are not ABI requirements: empty CCSR uses just IA=[1]
(the source seq_coo_to_ccsr writes IA[1] even with zero represented rows);
gen_csrmultcsr allocates exactly the symbolic nnz instead of initializing beyond
its allocation; CCSR beta scaling targets values, not the packed metadata header.
Zero-column CCSR outputs retain the Rust operation's declared shape instead of
the source empty-path hard-coded column count 1. Source's one-final-zero-column
padding omission is retained for nonempty column dimensions. COO duplicate ties
have no specified upstream sort order; conversion retains duplicates, while
sparse addition requires unique coordinates as its input contract.

The generic source gen_csrmultd discards fadd's returned value. That kernel has
not been ported or silently repaired; sparse-sparse-to-dense and the specialized
native kernels remain pending. The new exact local tests are analytic layout
and algebra checks, not a claim to have migrated the upstream sparse CPU suite.

## Explicit-grid plan cache (2026-09-07)

`src/planning.rs` separates the previously implemented unique-label grid mapping
from execution and supplies the signature/map-plan/cache responsibilities of
`contraction_signature.{h,cxx}`, `contraction_plan.h`, and `World::ctr_sig_map_`.
The actual mapping algorithm and redistribution/aligned contraction/restore
sequence are unchanged. This does not port the automatic candidate search or
claim the explicit-grid mapping heuristic is its replacement.

The cache borrows an explicit Context rather than living in a process-global
World static. It retains full distribution/map-chain fields and compares full
keys rather than upstream's hash-only equality. Equivalent index renamings
normalize by first appearance; scalars/values are excluded. Requested topology
is additionally keyed because this API exposes an explicit grid choice. This
is not an upstream hash/serialization ABI. Dense NS/unique-label plans only:
symmetry/sparse planning, candidate metadata, model costs, diagnostic steps and
plan packing remain unimplemented, not populated with placeholder estimates.

## Performance model training (2026-09-07)

`src/model.rs` adapts shared/model.cxx LinModel prediction, circular observation
history, error totals, threshold 16*np*nparam, regularization, local QR/Q^T b,
allgather of reduced R/y, and DGELSD fit. Cubic feature ordering follows
cube_params. No full observation gather or normal-equations replacement is used.
New local kernel methods qr_reduce/least_squares use DGEQRF/DORMQR/DGELSD through
internal FFI, preserving the compile-time boundary for a future faer backend.

The source's threshold<threshold deactivation predicate is always false;
should_observe remains true rather than silently correcting this behavior.
The source names an overprediction under_time and an underprediction over_time;
diagnostics retain these names and explain them. Caller-owned models replace
process-global registration. Coefficient file I/O, initial coefficient tables,
planner call-site integration and automated model instrumentation are still
pending. Construction takes explicit coefficients and history size; no new
machine calibration or guessed performance constants were introduced.
