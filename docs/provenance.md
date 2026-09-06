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

## Static CPU cost formulas (2026-09-07)

`src/initial_models.rs` copies all 32 non-offload coefficient arrays from
shared/init_models.cxx. `src/cost.rs` adapts CommData broadcast/reduce/allreduce/
alltoall/alltoallv features, seq_tsr_ctr CPU custom/inner model dispatch, and
nosym_transp contiguous-prefix model selection. Coefficients remain upstream
seeds, not estimates calibrated on this host. The model bank exposes explicit
load/write and mutable model access for training, without global registration.
Coefficient records keep source names and four-fractional-digit scientific
precision; malformed/missing CPU records return errors instead of source's
silent zero/substitution behavior. Unknown names, including offload records, are
ignored. Bulk writer emits the CPU bank; it does not edit unrelated file records.
These formulas are not yet wired to full contraction-tree candidate selection.

## Collective candidate selection (2026-09-07)

`src/selector.rs` adapts contraction_selector.h's per-signature candidate storage,
AND filters, replication measure (physical replicas times virtual copies), and
selectCandidate availability-allgather followed by lowest-owner size/payload
broadcast. Selector borrows an explicit Context instead of global universe.
GridPlan packing preserves full mapping chains and uses explicit u64/f64 bits;
no C++ integer-buffer ABI compatibility is intended. Received plans execute via
Tensor::contract_with_plan. Time/memory metadata must be supplied explicitly;
automatic full-tree costs and candidate generation remain pending. The upstream
unimplemented allgather() method is not exported as a Rust placeholder.
Clearing or changing candidate signature also clears stale selected state.

## Recursive execution-tree cost formulas (2026-09-07)

`src/plan_cost.rs` ports local, virtual, replicated and 2D-panel time/mem_rec
formulas from ctr_tsr.cxx, ctr_comm.cxx and ctr_2d_general.cxx. Panel work memory
includes all three panels plus the largest moving auxiliary panel; virtual
bookkeeping retains source sizeof(int)=4 and VIRT_NTD=1. Layer division resets
the child's layer argument to 1 at panel nodes, as upstream. Node counts use
the source comm_nodes volume formula, not inferred hardware measurements.
ctr_virt does not override the base internode-volume estimator, so its reported
volume is zero in this revision, even though its time estimate recurses.

These are source-model work bytes, not actual Rust peak RSS or total residency.
Automatic execution-tree construction, redist/fold resident-memory accounting
and feeding complete costs into candidate discovery remain pending.

## Normal-search objective selection (2026-09-07)

Selector::select_best ports evaluate_mappings' strict memory exclusion, dense
INT_MAX element limit, time-only/normalized time-plus-memory scoring, and
first-in-order tie handling. Local winners send time and i64 memory with two
MPI_Gathers to rank 0; winner rank and executable payload are broadcast.
The |weight|>1e-8 cutoff and positive weighted baseline requirements are retained.
Unlike the upstream surrounding mutable tensor search, this consumes explicitly
supplied complete candidate estimates. It excludes exhaustive candidates: their
different incumbent/refinement logic and automatic mapping generation remain
pending. Selection cannot compensate for incomplete supplied memory estimates.

## Local packed symmetry layout (2026-09-07)

`src/symmetry.rs` adapts shared/util.cxx packed_size/sy_packed_size recurrences
and combinatorial offset summation from shared/iter_tsr.h. Rust-owned Packed
adds local canonical coordinate reads/additive writes, adjacent-swap AS parity
and AS/SH structural zeros. This is not a port of sym_indices alignment or
symmetrization contraction factors. u128 intermediate products replace signed
intermediate products. True AS/SH sizes and SY-compatible intermediate sizes are
separate. Distributed packed tensor ownership and contraction remain pending.

`src/sym_indices.rs` directly adapts Devin Matthews' relativeSign, two-operand
align_symmetric_indices and both overcounting_factor overloads from
symmetry/sym_indices.{h,cxx}. Source author attribution is retained. Three-operand
alignment and integration into symmetry contraction are still pending; factors
require the source's aligned-index assumptions.

## Combined symmetry operations batch (2026-09-07)

The three-operand align_symmetric_indices overload is now implemented in
sym_indices.rs, retaining operand-incidence grouping and per-operand sort/sign
order. `src/sym_permutations.rs` ports both order_perm/add_sym_perm/get_sym_perms
paths using a fixed two/three-operand array. It retains the expanding discovery
list and source index-map equality (duplicate signs are not accumulated).
Inverse index lookup takes the last occurrence, as inv_idx does. Circular
generator order/parity follows cmp_sym_perms.

`Layout::coordinates` is a Rust packed-domain iterator with O(order) state;
Packed indexed scaling keeps sym_seq_scl's right multiplication order and indexed
endomorphisms touch only represented coordinates. These use the true compressed
layout, not a port of the source SY-padded intermediate iterator ABI. The tensor
symmetrize/desymmetrize routines, permutation execution, and distributed packed
mapping still need implementation. No dense tensor expansion is used here.

## Integrated high-order BLAS folding (2026-09-07)

`src/folding.rs` adapts get_len_ordering/calc_fold_lnmk's fully foldable NS path:
AB/AC/BC/ABC label classes become k/m/n/l; canonical packed buffers A[k,m,l],
B[k,n,l], C[m,n,l] use the existing folded_f64 kernel with A transposed. It
restores C's original axis order after the batch GEMMs. This implements one
source layout choice, not the six-way transpose-cost search. Single-operand
labels and repeated labels explicitly reject at this entry point; no silent
reference fallback is introduced. Partial/symmetric folding remains pending.

Tensor::contract_blas_on_grid combines existing GridPlan mapping with local
folding through ctr_replicate-style input broadcasts, root-only beta, virtual
block traversal and output root Reduce. Explicit key redistribution restores the
output tensor distribution. Input clones preserve caller ownership. It does not
gather a global tensor or replace the requested distributed execution with a
single-rank GEMM. Automatic generic API dispatch and node-aware/2D plan building
are separate, still unfinished tasks. Compile-time LocalKernels remains the
native BLAS/faer substitution boundary.

`tests/upstream_gemm4d.rs` ports the NS associativity branch of test/gemm_4D.cxx,
using n=7, rank-seeded drand48 and the original strict elementwise 1e-6 bound.
Rust uses separate intermediates instead of aliasing output/input expressions.

## Distributed ScaLAPACK Cholesky and triangular solves (2026-09-07)

`src/matrix.rs` adapts interface/matrix.cxx's NS cholesky/solve_tri sequence:
read into descriptor distribution, PDPOTRF or PDTRSM, reconstruct distributed
tensor and (for Cholesky) retain the requested triangle. This first entry point
accepts an explicit grid and uses block size 1, retaining cyclic local storage;
it does not implement automatic descriptor selection or the separate solve_spd
padding/PPOSV algorithm. Source tensors remain unchanged via owned local clones.

`src/ffi/scalapack.rs` independently binds BLACS/DESCINIT/PDPOTRF/PDTRSM. Native
handles remain internal; the grid comes from the supplied MPI subcommunicator,
not an implicit MPI_COMM_WORLD. Explicit gridexit/system-handle cleanup occurs
before returning, including LAPACK info errors; Drop never communicates.
native-scalapack is default-enabled and depends on native-linalg. WSL links
scalapack-openmpi; native Windows library selection remains pending. Local
BLAS/LAPACK's compile-time replacement boundary is unchanged.

Validation ports Cholesky/triangular reconstruction and triangle criteria from
test/python/test_la.py into Rust (no Python interface/runtime). Fixtures are
deterministic SPD/triangular matrices rather than a claim of identical NumPy
random inputs. QR/SVD/eigh distributed implementations are not implied.

## Distributed thin QR and SVD (2026-09-07)

matrix.rs now follows interface/matrix.cxx QR's PDGEQRF -> retain R -> PDORGQR
sequence, and SVD's descriptor-specific PDGESVD('V','V') sequence. Shapes of thin
outputs are Q(m,k), R(k,n), U(m,k), S(k), VT(k,n), k=min(m,n); vectors/matrices
are returned as Rust tensors. Packing/extraction is rank-local with cyclic
block size 1; neither algorithm gathers matrix factors. ScaLAPACK's replicated
singular values are copied directly to vector owners. Native symbol declarations
and workspace management are confined to ffi/scalapack.rs.

Explicit grids replace automatic descriptor choice for these entry points.
Full thin SVD is implemented here; source rank/threshold truncation and randomized
paths remain pending. Test metrics follow scalapack_tests/qr.cxx and svd.cxx;
fixtures are deterministic real matrices, not every upstream dtype/test branch.
