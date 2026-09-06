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

## Distributed truncated and randomized SVD (2026-09-07)

matrix.rs now implements source svd rank/threshold slicing and svd_rand's QR,
power iteration A*A^T*U, projected U^T*A decomposition and final rotation. All
matrix products and decompositions remain distributed. A supplied guess bypasses
initial QR as upstream; generated guesses use explicit rank-seeded drand48 values
in [-1,1), rather than the source's implicit global random state. Source's zero
retained-rank behavior (no slice, full factors returned) is preserved and stated
in the API. This is not an adaptive iteration or oversampling redesign.

## Symmetric eigensolver and square subworlds (2026-09-07)

Tensor::eigh follows matrix.cxx's largest-square-rank subset strategy and
lapack_symbs.cxx peigs<double>'s PDSYEVX(V,A,U), zero ABSTOL/ORFAC and queried
WORK/IWORK. Canonical matrix keys route to the square grid via parent all-to-all;
eigenvectors and eigenvalues are routed back to parent output distributions.
The square grid comes from the explicit child MPI context, also when called
inside a user subcommunicator. Native handles and scratch storage stay in FFI.
For two parent ranks the source strategy uses one computing rank; for four it
uses all four. This is not a claim of two-rank parallel eigensolver execution.
The matrix transfer is key-based subworld routing rather than the remaining
optimized add_to_subworld implementation; no matrix gather replaces the four-rank
ScaLAPACK computation.

## SPD solve and dense TTTP (2026-09-07)

Tensor::solve_spd follows matrix.cxx's paired-prime physical grid and virtual
column factor, redistributes the coefficient/RHS, pads missing cyclic rows with
identity equations, then interprets local buffers using square-block descriptors
for PDPOSV(L). The output returns to the RHS distribution. Inputs are unchanged;
no matrix gather or substitution with POTRF/TRSM is used. The direct API solves
A X = B, while the upstream Python wrapper presents its transposed convention.
The Rust path explicitly chooses the source-derived grid, including when an
input already has equal row/column phases. Zero global dimensions are rejected.

multilinear.rs adapts TTTP's vector products, matrix auxiliary sum, balanced
k/div + (d < k%div) slicing and delayed multiplication for multiple blocks.
Mode factors align to physical tensor mappings, without virtual factor storage.
Current factor replication uses existing key-based tensor redistribution, not
the upstream specialized redistribution plus fiber broadcast. Explicit divisions
expose the blocked execution but do not implement automatic available-memory
selection. Dense f64 only; sparse and generic-semiring TTTP remain pending.
For nonconsecutive matrix modes, indexing uses the selected mode's mapping:
the source matrix loop's phys_phase[j] instead of phys_phase[modes[j]] is not
reproduced. This avoids indexing a different mode's physical partition.

## Dense MTTKRP (2026-09-07)

multilinear.rs and multilinear_kernel.rs adapt interface/multilinear.cxx's
mode-aligned factors and complementary-fiber reduction, and semiring.h's
fiber-grouped MTTKRP arithmetic. Factors route to each physical shard's root
through collective indexed reads, then broadcast to its complementary fiber.
The output is reduced to that fiber root and written into the explicitly
requested output distribution. Tensor entries are never gathered globally.
Virtual local blocks are key-sorted before source fiber grouping; replicated
tensor layers contribute only their canonical owners.

The direct API receives factors excluding the output mode and returns a new
factor, rather than replacing a member of a pointer array. Vectors and
auxiliary-first matrices are supported: the pinned semiring matrix kernel
asserts on auxiliary-last storage. f64 arithmetic only at present; generic
semirings, sparse tensor integration and specialized source redistribution
kernels remain pending. Native-library selection is unaffected.

## Tensor SVD and reshape (2026-09-07)

tensor_svd.rs follows interface/multilinear.cxx's indexed tensor SVD: order
left non-auxiliary modes before right non-auxiliary modes, flatten to a matrix,
invoke the existing distributed deterministic/truncated or randomized matrix
SVD, reshape the factors, then move the auxiliary index to its requested
position. The singular-value vector determines the actual retained rank.
No C++ expressions, pointers to output tensors or full matrix gather are used.

reshape.rs implements reshape_tensor's general canonical-key read/write path
from tensor/untyped_tensor.cxx, preserving column-major flattened indices.
It accepts an explicit target distribution and works with the crate's monoids.
Upstream's optimized merge_modes/split_modes and unit-length aliases remain
unported; using the general key path does not close those optimization items.
Tensor SVD is not tensor-train SVD or batched SVD; those remain distinct scope.

The installed ScaLAPACK 2.2 one-row SVD has a PBLAS row/column ambiguity:
PDGEBD2 -> PDLARFG -> PDNRM2 on a one-element tail with matrix row count and
increment both one returns the norm only to its owner. PDLARF then conditionally
enters DGSUM2D with inconsistent TAUP. Reference ScaLAPACK SRC/pdlarfg.f:176-229,
SRC/pdgebd2.f:389-415 and SRC/pdlarf.f:557-601, together with PBLAS PDNRM2's
documented special case, explain the observed rank stacks. Upstream CTF's
padding changes local storage, not logical dimensions, and does not fix this.
Rust selects a one-column native grid for nontrivial one-row SVD, retains all
ranks in PDGESVD, and redistributes factors back to the requested layout.
There is no solver retry, matrix gather, padding-induced spectrum change or
alternate numerical solver in this path.

## Pending Solve_Factor source audit

interface/multilinear.cxx:689-1003 forms one rank-by-rank Gram system per
target-mode row: LHS_i += T_entry * product_other_factors * product_other_factors^T,
using lower SYR, then solves against RHS using POSV. Its complementary-mode
communicator performs Reduce_scatter of Gram blocks, Scatter of RHS rows,
local solves, then Gather of solutions. This is a different algorithm from
MTTKRP, not a wrapper around it. The Python production path is f64 and
auxiliary-first matrices only. test_einsum.py::test_Solve_Factor_mat uses
numpy.allclose defaults (rtol=1e-5, atol=1e-8), not that file's L1 helper.

The source hardcodes double/MPI_DOUBLE, leaves POSV info unchecked, and has
broken vector/auxiliary-last branches. Its equal-size RHS Scatter can exceed
the natural row buffer when rows do not divide communicator size; its Gather
leaves replicated non-root RHS buffers uncleared. A port must preserve the
distributed Gram/solve steps without reproducing these memory/error defects.
No Solve_Factor implementation or acceptance is claimed by this audit.

## Distributed dense Solve_Factor (2026-09-07)

solve_factor.rs now ports the audited f64 auxiliary-first production algorithm
for dense tensor weights. Factor shards are read on canonical physical roots
and broadcast over complementary fibers. DSYR forms lower Gram systems;
MPI_Reduce_scatter divides them among the target-mode fiber, MPI_Scatter
distributes padded RHS rows, and each rank calls DPOSV on its assigned systems.
Only solutions, not coefficient matrices for root factorization, are gathered
within the fiber and then written into the requested RHS distribution.

Padding is explicit before equal-count Scatter; mathematically absent rows
are not solved. Only fiber roots write solved values, excluding stale replicas.
Nonzero DPOSV info is propagated collectively before returning an error.
These are intentional corrections of the audited memory/error defects, not
replacement algorithms. Sparse source fixtures and non-f64 algebra remain
unimplemented; the broken C++ vector and auxiliary-last branches are not exposed.

## Distributed sparse storage and I/O (2026-09-07)

sparse.rs follows sparse_rw.cxx's sorted key/value pairs within virtual blocks,
duplicate reduction with the algebra's addition, and absent-key additive
identity. Only stored entries are allocated or redistributed; dense local
storage size is not used for allocation. Physical ownership/virtual offsets
reuse the shared mapping implementation. Canonical source owners send entries
to destination replicas, avoiding replica multiplication during redistribution
and reduction. No tensor gather or implicit collective destructor is used.

As in sp_write:1379-1485, old-only keys stay unchanged, requested existing keys
are weighted once by beta, incoming values are right-multiplied by alpha and
duplicate contributions accumulate. Explicit zero results remain stored.
For an overlapping key, the source adds the first incoming value before the
old value; later duplicate requests are appended. This noncommutative order
is retained independently of the right-hand scalar multiplication order.
Sparsify is a separate predicate-based operation (untyped_tensor.cxx:1670-1711).
Stored transforms deliberately do not evaluate absent entries. sp_read intends
missing keys to produce the algebra identity; its trailing-request control-flow
defect is not reproduced. Sparse views currently use canonical-key routing,
not all optimized upstream sparse reshuffle kernels.

The source's key-only std::sort leaves equal-key ordering unspecified. Rust
uses stable sorting and source-rank/request order for duplicate additions;
this makes the tie order deterministic rather than claiming it reproduces
an unspecified C++ ordering for noncommutative monoids.
