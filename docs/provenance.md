# Source provenance

`src/symmetric_random.rs` ports interface/tensor.cxx:1578-1608
fill_random_base for four real/complex types to packed storage. It consumes
the existing source-compatible Generator once per allocated slot before
clearing padding, rather than visiting only valid canonical values.

`tests/upstream_ccsdt_t3_to_t2.rs` ports the active pinned CCSDT T3-to-T2
equations, including the misleadingly named NS_B/NS_C tensors' partial AS
compression. Dimensions and input fixture are bounded/deterministic; the
source norm and residual inequalities are unchanged.

`tests/upstream_ccsdt_map.rs` reimplements pinned `test/ccsdt_map_test.cxx` at
its default n=4. It preserves the zero-initialized six-order contraction rather
than inventing a nonzero accuracy criterion absent from the original driver.

`tests/upstream_scalar.rs` and `tests/upstream_speye.rs` reimplement the active
fixtures in pinned `test/scalar.cxx` and `test/speye.cxx`. They preserve the source
acceptance inequalities with bounded dimensions (scalar n=3; speye n=4/order=3),
excluding source-disabled blocks. No C++ interface or library is linked by Cargo.

Reference repository: https://gitlab.cc4s.org/cc4s/ctf
Reference commit: f69cbb46e23bc2f39cda5722ce096f56301dab4f

`LICENSE` preserves the upstream notice and terms for adapted material. It is
not a declaration that Edgar Solomonik authored the new Rust implementation.
Do not stamp independently written files with the upstream author's copyright.

* `src/symmetric_permuted_io.rs`, `src/sparse_permuted_io.rs`: mixed-storage
  coordinate read/write phases follow `untyped_tensor.cxx:928-992`. Compressed
  full-orbit gather and canonical-only scatter follow `read_local:2349-2395`
  versus `read_local_nnz:1848-1894`; signed canonical duplicates follow
  `sparse_rw.cxx:962-1063`. No implicit normalization or sparse-target gather.

* `src/permuted_io.rs`: `tensor/untyped_tensor.cxx:928-992` coordinate permutation
  read/write phases, with Option maps replacing -1 coordinates and explicit
  optional children replacing dummy scalars. Dense/packed indexed write order
  follows `redistribution/sparse_rw.cxx:795-839`, including signed duplicate writes.

* `tests/upstream_{readall,readwrite,sptensor_sum,subworld_gemm}.rs`:
  migrate the matching pinned `test/{readall_test,readwrite_test,sptensor_sum,
  subworld_gemm}.cxx` driver equations and acceptance rules. Source RNG uses its
  POSIX recurrence directly in Rust; fixture sizes and literal quirks are recorded
  in the inventory and validation evidence.

* `src/sparse_virtual.rs` and mapped sparse contraction integration:
  operand/union strides, block offset updates and beta-once state follow
  `contraction/spctr_tsr.cxx:699-898`; owned sparse block vectors replace byte
  offset/deallocation bookkeeping.

* `src/subworld.rs`, `src/symmetric_subworld.rs`: parent/child accumulation follows
  `untyped_tensor.cxx` subworld phases and `algstrct::acc` coefficient order.
  Ordered value streams reuse physical-residue/symmetry-bounded traversal;
  explicit rank orientation replaces source mirror-buffer forwarding.

* `src/symmetric_reshuffle.rs`: symmetry-bounded global-order traversal from
  `pad_cyclic_pup_virt_buff` in `redistribution/cyclic_reshuffle.cxx`; canonical
  offsets use the existing Rust packed distribution, with no dense expansion.

* `src/cyclic_reshuffle.rs` and dense `Tensor::redistribute`: NS physical-residue
  traversal and ordered value-stream packing/unpacking follow
  `redistribution/cyclic_reshuffle.cxx`. Rust local offsets encode virtual blocks;
  explicit replica destinations retain the established Rust storage invariant.

* `src/schedule.rs`, `src/schedule_graph.rs`, `src/schedule_partition.rs`:
  dependency release, cost-window partitioning and subworld execution phases
  adapted from `interface/schedule.{h,cxx}`. Rust closures replace C++ recorded
  expressions. `schedule.md` lists explicit fixes to broken source paths.

* `src/binary_io.rs`, `src/ffi/binary_io.rs`: consecutive-rank chunks and native
  independent-offset file I/O follow `tensor/untyped_tensor.cxx:3819-3909`.
  Rust Wire encoding, explicit path/offset APIs and ownership replace raw pair
  buffers and the C++ MPI_File interface. See `binary-io.md` for source corrections.

* `src/pair_read.rs`, `src/ffi/allgather.rs`: explicit all-rank pair/data
  extraction follows `tensor/untyped_tensor.cxx:2443-2557`; stored sparse zeros
  and packed nonzero-only semantics follow `read_local_nnz:1880-1894` and
  `interface/tensor.cxx:383-420`. MPI bindings and Wire serialization are Rust
  implementations, not upstream authorship or C++ ABI compatibility layers.

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

## Sparse and mixed indexed summation (2026-09-07)

sparse_sum.rs implements NS sparse-to-sparse, sparse-to-dense and dense-to-sparse
indexed sums, using canonical stored keys, diagonal projection, reindexing,
reduced-label duplicate accumulation and output-only index expansion. Repeated
indices are projected before scaling, as high-level summation.cxx:1499-1512
does; off-diagonal destination values are not overwritten. Sparse beta=zero
retains explicit structural zero entries, matching spspsum. Dense input is
sparsified before sparse merging (summation.cxx:1514-1517); the raw
dnA_spB_seq_sum kernel itself is an upstream assertion, not a usable algorithm.

Sparse merging uses value*alpha/value*beta; sparse-to-dense preserves the raw
kernel's scalar-input alpha*value exception and dense beta scaling convention.
This path routes keys using existing distributed writes rather than claiming
completion of upstream's optimized folding/replication sparse sum planner.
Custom unary/accumulator functions and compressed-symmetry sums remain pending.

## Explicit-grid distributed sparse matrix contractions (2026-09-07)

sparse_gemm.rs aligns A(m,k), B(k,n) and C(m,n) to a physical 2D grid and
an LCM contraction phase, using virtual k blocks when process row/column
counts differ. Each step broadcasts variable sparse entry counts followed by
serialized panel entries along the A row/B column fibers. Sparse panels are
converted to owned CSR, then invoke the existing CSR sparse or dense kernels.
The sparse-by-dense path broadcasts fixed-size dense B panels. Beta applies
on the first step only; empty panels still participate. Inputs remain unchanged
and output redistributes back to its original physical/virtual layout.

Sparse output never uses a dense m*n intermediate. Sparse-by-sparse dense
output uses the intended gen_csrmultd loop; unlike the pinned generic source,
the returned addition value is assigned instead of discarded. No C++ sparse
buffer ABI is retained. This closes explicit-grid NS matrix paths only, not
automatic sparse plan selection, arbitrary-order sparse folds, output-moving
2D levels, compressed symmetry or node-aware sparse scheduling.

## High-order sparse folding and reshape (2026-09-07)

sparse_fold.rs groups unique NS labels by their occurrence in A/B/C: AB is k,
AC is m, BC is n, and ABC is an independent batch. Canonical sparse matrices
use A[m,k,batch], B[k,n,batch], C[m,n,batch], dispatching each batch through
the distributed sparse matrix panel kernels before inverse axis permutation
and restoration of the requested output distribution. Stored-key reshape
follows the source sparse reshape path, without dense temporary storage.

This is a fully foldable unique-index path, not a reference Cartesian-product
fallback. Repeated labels, one-operand-only labels and compressed symmetry are
still explicitly unsupported by this folding entry point. Automatic candidate
selection and source-specific sparse packing optimizations remain separate
unfinished responsibilities.

tests/upstream_sparse_mp3.rs ports examples/sparse_mp3.cxx's MP3 equations with
dense T, testing dense integrals against sparse integrals through the new fold
path. Orbital-denominator sums and all five MP3 contribution terms are retained.
The nv=3/no=2 fixture uses deterministic global-key values in the source's
intervals and threshold 0.8 rather than rank-seeded drand48. The test sparsifies
all five integral tensors (the source calls Vabcd.sparsify twice and leaves
Vijab dense); this changes storage, not the equations or values. The original
relative-energy criterion is retained. The custom-function sparse-T branch
is not claimed implemented or tested here.

## Sparse functions, dense-by-sparse products and sparse-T MP3 (2026-09-07)

sparse_functions.rs provides typed local maps of stored pairs and indexed
accumulator transforms restricted to existing output keys. Dense input's
additive-identity values are skipped, matching high-level summation's implicit
sparsify before its sparse kernel. For already-sparse input, explicit stored
zeros invoke the accumulator, while missing entries do not. Repeated output
indices select only the diagonal. The current implementation routes requested
input keys, not the remaining optimized broadcast/replication function planner.
Input-only reduction labels and full custom function contraction are pending.

Dense-by-sparse matrix and folded contractions now broadcast dense A panels
and sparse CSR B panels without swapping the operands. The local kernel
preserves alpha*(A*B) for noncommutative semirings and restores C's distribution.
tests/upstream_sparse_mp3_t.rs implements the source's sparse DPair numerator/
denominator chain, four orbital-energy accumulations, sparse amplitude map-back,
dense Fock times sparse T, sparse integral times sparse T and final energy.
Its deterministic fixture and Vijab-storage adaptation match the prior dense-T
test; the original relative-energy acceptance rule is unchanged.

## Sparse diagonal projection and reinsertion (2026-09-07)

src/sparse_diagonal.rs extends the tensor extract_diag/reinsertion approach
to stored sparse keys. Canonical owners route selected entries into projected
cyclic storage; reinsertion replaces only selected diagonal structure and
retains the original distribution and all off-diagonal keys. It does not
allocate a dense tensor or gather the global sparse tensor.
src/sparse_fold.rs applies this preprocessing to all four sparse/mixed
contraction APIs, then uses the existing unique-label folding and panel
communication. Shape/index and foldability validation precedes redistribution.
One-operand-only labels and optimized upstream sparse diagonal kernels remain
pending. The new exact fixtures are independently written Rust tests.

The pinned distributed symmetry source audit is recorded in
distributed-symmetry-layout.md. It establishes local SY-sized storage with
padding/noncanonical holes, including for AS/SH; it is not an implementation
or acceptance claim for distributed compressed symmetry.

## Distributed compressed storage (2026-09-07)

src/symmetric_distribution.rs implements the pinned layout rules documented
in distributed-symmetry-layout.md: equal group total phases, per-virtual-block
SY-sized packing for SY/AS/SH, and removal of global noncanonical/padded holes
from visible local pairs. Global normalization preserves AS permutation sign.
The rectangular distribution remains the physical ownership model, not the
compressed local offset model.

src/symmetric_tensor.rs owns those compressed slots and borrows its explicit
communication context. Indexed writes canonicalize and sign-adjust before
routing to owners/replicas; reads preserve caller ordering and return AS/SH
structural zeros. Same-symmetry redistribution routes only canonical owner
entries and never gathers or materializes the full NS tensor. Local transforms
touch valid canonical entries only, leaving allocated holes at additive zero.
Communication currently uses key-based exchange rather than all optimized
upstream cyclic reshuffle/readwrite kernels. Symmetry-changing repack,
distributed symmetric sums/contractions and automatic symmetry mapping remain
pending; this storage implementation does not imply their completion.

## Compressed indexed operations and group-preserving repack (2026-09-07)

src/symmetric_operations.rs applies the existing sym_seq_scl right-scalar
multiplication order to distributed canonical entries, including repeated-label
diagonal selection and custom transforms. Holes are not operands. Its
repack_groups follows tensor/untyped_tensor.cxx 223-270's no-more/no-less-symmetry
branch: preserve the physical layout and raw canonical values, then clear slots
invalid under the new SY/AS/SH kind. It deliberately does not introduce factorial
normalization, sign averaging, or NS expansion. Changing NS group boundaries
belongs to the source summation-based branch and is still pending.

SymmetricTensor::write_scaled canonicalizes keys and AS signs, applies incoming
value times alpha plus old value times beta, and applies beta once per touched
canonical key. Untouched entries and AS/SH structural zeros are unchanged.
Equivalent requested permutations contribute to the same canonical entry.

## Symmetry-boundary repack (2026-09-07)

repack_to implements tensor/untyped_tensor.cxx 261-269's identity summation
with home_sum_tsr(true,false). The false argument bypasses sym_sum_tsr
(summation.cxx 948-958,1080-1087). sym_seq_sum.cxx 39-75,126-166 intersects
source and destination canonical bounds; no orbit expansion, signs, averages,
or multiplicity factors are applied. test/repack.cxx 33-63 explicitly expects
SY-to-NS to populate only the original canonical chamber, not its mirror.
Thus this operation differs intentionally from mathematical symmetrization
and from semantic reads, which do mirror/sign-adjust.

The Rust operation filters unique-owner canonical source entries by target
canonical validity and routes unchanged pairs to explicit target owners and
replicas. It supports adding or removing NS boundaries but, like the source,
rejects simultaneously adding and removing them. Source automatic mapping and
optimized aligned summation communication remain pending; current key routing
does not gather or allocate a full global tensor. Genuine symmetry-aware sums
and contractions remain separate unfinished work.

## Packed sequential and replicated summation (2026-09-07)

src/symmetric_sum.rs ports the scalar sym_seq_sum_ref canonical-bound traversal
and packed offsets from summation/sym_seq_sum.cxx. All non-NS physical links
use inclusive local bounds and SY-sized offsets, including AS/SH: a diagonal
local quotient is not necessarily a diagonal global coordinate. This raw kernel
does not perform signed semantic reads or clear globally invalid padding.
Input multiplies alpha on the right, then is added before existing output;
beta multiplies output on the left. Repeated output labels restrict the beta
operation to the indexed diagonal, following SCAL_B (232-261).

src/symmetric_sum_comm.rs ports explicit tsum_virt visits and tsum_replicate's
input broadcasts, root-only old output, and ordered output-fiber reductions.
Beta applies once per visited virtual output block. Layouts, aligned phases,
communicator fibers, and additive commutativity are explicit caller inputs.
It does not claim automatic symmetric mapping, broken-symmetry permutation
expansion, global padding cleanup, or a complete high-level symmetric sum API.

## Raw packed contraction execution (2026-09-07)

src/symmetric_contraction.rs ports sym_seq_ctr_ref's inclusive local canonical
bounds and SY-sized physical offsets for all non-NS kinds. The product order
is (A*B)*alpha, then scaled product plus C. The all-scalar source branch uses
C*beta; the nonscalar-index branch prescales the whole C allocation by beta
on the left. In particular, source lines 444-450 explicitly note that prescaling
the full buffer is wrong for subset iterators: this low-level port preserves
that behavior, and a public diagonal operation must extract/reinsert its output
in the upper layer. No extra beta==one gate was introduced.

src/symmetric_contraction_comm.rs implements explicit virtual traversal and
ctr_replicate ordering: broadcast A and B, beta-scale old output only on roots,
execute child blocks with root/nonroot beta one/zero, Reduce output on supplied
fibers, then clear nonroot broadcast input replicas. This is not Allreduce.
Shared local shapes and virtual phases are caller-aligned. It does not supply
automatic mapping, 2D symmetric communication, operation permutation/sign
expansion, or symmetry multiplicity normalization. These remain upper-layer
work; no globally gathered substitute or implicit signed full tensor was added.

## Tensor-level canonical indexed sum (2026-09-07)

src/symmetric_sum_tensor.rs integrates the symmetry-disabled sum_tensors
canonical-domain operation with distributed compressed storage. Projection
selects repeated indices, input-only indices reduce by destination key, and
output-only indices expand into target coordinates. Only already-canonical
valid output keys are retained. Unique source owners route to all destination
replicas, with alpha right multiplication and selected old-output beta left
multiplication once. There is no full-tensor gather or signed read expansion.

The public name sum_canonical_from distinguishes it from future symmetry-aware
sum_from. The remaining source orchestration, especially broken AS/SH link
unfolding before permutation enumeration, is documented in
symmetric-sum-orchestration.md. Existing permutation helpers cannot simply be
called on arbitrary original operands to claim full operation semantics.

## Hollow symmetry-aware indexed sums (2026-09-07)

src/symmetric_hollow_sum.rs implements sum_hollow_from for unique-index
NS/AS/SH operations. It follows summation.cxx's broken-link test and recursion:
relax the first broken link, materialize a desymmetrized input when more links
remain, or compute into a zero less-symmetric output and symmetrize it back.
Only the terminal broken-link case enumerates permutations on current operands.
This is not a direct enumeration shortcut on the original mixed AS/SH layouts.

Alignment sign and factorial/cancellation factors use algebra addition and
negation, not scalar casts. Input/output materialization uses the same explicit
physical distribution with relaxed links and existing canonical key exchanges;
the path allocates distributed intermediate tensors as the source does, never
a global gather. Unique labels and absence of SY are explicit method contracts.
Repeated-label preprocessing, SY coincidence-surface corrections and the full
automatic communication planner remain required for unrestricted sums.

## Compressed diagonal projection (2026-09-07)

src/symmetric_diagonal.rs deletes later duplicate axes in tensor::extract_diag
first-pair order. The bridge at deleted axis j is preserved only when old links
j-1 and j agree, otherwise it becomes NS. Shape/mapping axes are deleted while
the topology is retained, so removing a physical mapping introduces replicas.
Projected canonical owners request expanded source coordinates; reinsertion
clears only the corresponding target canonical keys before writing replacements.
No full NS expansion or global gather is used.

Supported patterns are repeated singleton NS axes, an isolated SY pair ii,
and structurally zero equalities within AS/SH groups. Unaffected symmetry groups
retain their compression. Cross-group repeats involving nontrivial symmetry
are deliberately rejected: source extraction invokes symmetry-aware summation,
which is not generally interchangeable with sampling signed semantic reads.
sum_hollow_from now extracts/reinserts supported repeated-label patterns before
its unique-index recursion. Unrestricted diagonal symmetry handling is still
pending, not hidden behind a silent NS fallback.

## f64 SY-aware summation (2026-09-07)

src/symmetric_sy_sum.rs extends the source recursive orchestration to SY for
Arithmetic<f64>. It preserves input-SY materialization and beta-dependent
output-SY unfolding, pair-diagonal exclusion in transpose contributions, and
the active higher-order coincidence-surface scaling in desymmetrize. The
disabled `if(0)` symmetrize algorithm is not ported: the active branch uses a
fresh target-symmetry tensor followed by accumulation. No global gather is used.
Other scalar types and unsupported cross-group repeated-index extraction remain
pending; the source's higher-order desymmetrize FIXME is not silently redesigned.

A three-axis fixture exposed a missing source step in sum_canonical_from:
sum_tensors aligns preserved symmetric index relations even with symmetry
processing disabled (summation.cxx around 1518). That alignment now happens
before canonical key routing and retains the source positive-sign precondition.
Consequently transposed SY-to-SY raw tasks cover the full aligned canonical
domain, rather than only the unaligned diagonal intersection. Earlier notes
describing that transposed case as diagonal-only are superseded by this source
correction; identity-map repack semantics remain unchanged.

## Generic SY scalar algebra (2026-09-07)

The SY orchestration now uses Group/Semiring algebra operations and owned
coefficients instead of hardcoded f64 arithmetic. The independently written
CastFromF64 capability in scalar_conversion.rs makes the source cast_double
coefficient conversion explicit for custom algebras. Built-in arithmetic
supports f32/f64, i32/i64 and complex f32/f64. Integer fractional conversions
truncate, matching the relevant source coefficient casts; they are not silently
promoted to floating point. Conversion callers require finite representable
coefficients. No numerical backend framework or implicit custom-algebra
fallback is introduced.

The source recursion, transpose tasks and coincidence-surface formula are
unchanged. Algebra and element cloning follows ownership needs, with no Copy
requirement on custom elements. Full cross-group diagonal semantics and source
higher-order coincidence limitations remain separate from this scalar extension.

## Explicit mapped packed tensor contraction (2026-09-07)

src/symmetric_contract_tensor.rs connects compressed tensors to the existing
packed sequential/virtual/replication kernels. contract_canonical_on takes an
explicit topology and one union index label per physical axis, aligns preserved
symmetric indices, then applies source coordinate_symmetry to the union mapping
table. Each operand is redistributed into its mapped compressed layout. Missing
operand dimensions use explicit topology-fiber broadcasts; missing output
dimensions use root Reduce. Canonical output owners restore the original
destination distribution through indexed writes. Temporary fibers are closed
explicitly and tensor destruction does not communicate.

This route does not substitute sparse contraction or a globally gathered dense
tensor. Replicated local operands/output buffers follow the original replication
layer. It remains a canonical operation: unique operand labels and positive
aligned sign are required; upper-layer orbit expansion, overcounting factors,
diagonal preprocessing and automatic physical-label selection are not supplied
by this method. Current key-based redistribution is not a claim that all
optimized source redistribution kernels have been ported.

## Symmetry-aware contraction on explicit mappings (2026-09-07)

src/symmetric_contract.rs ports contraction::sym_contract and its
three-operand broken-link test, not the two-operand summation recursion. It
aligns signs and computes reduced-group overcounting factors, then unfolds
broken operands or uses operation permutations according to mapping feasibility.
Recursive unfolding receives signed alpha without the parent overcount factor;
terminal raw tasks receive the adjusted coefficient. Output temporaries follow
source beta scaling and symmetrization. Supported repeated labels are extracted
and reinserted before recursion.

Terminal SY reductions apply source prescale_operands/scale_diagonals to the
selected input, correcting repeated-coordinate multiplicity before factorial
overcounting. Input selection uses the mapped local packed allocation sizes,
not the global rectangular shape product. Temporary tensor ownership replaces
source alias/pull-alias memory management while source kernels and collective
steps remain explicit. This connects to contract_canonical_on rather than
gathering tensors or switching to an unrelated contraction implementation.

The topology and physical-label map remain caller-supplied; automatic cost-based
mapping selection and full optimized 2D symmetric planning are not completed.
General cross-group repeated-index extraction and the full upstream CPU test
suite remain open. Availability of this operation is not whole-project acceptance.

## General source diagonal stack (2026-09-07)

The diagonal-pattern restriction has been removed. Each extract_diag step now
deletes exactly one later duplicate axis and transfers data through
sym_sum_tsr(run_diag=true), with source-generated maps independent of outer
labels. Replacement reconstructs intermediate output diagonals and reinserts
them in reverse order. These operations require Group, Semiring and explicit
CastFromF64 capabilities because SY unfolding may use fractional coefficients.
Desymmetrization/symmetrization transfers use identity maps, as in the source;
propagating an outer repeated map into them would cause incorrect recursion.

Preserved source caveat: reverse cross-AS diagonal summation uses beta=0 only
for the first permutation, then beta=1 (summation.cxx 1375,1402-1404). SCAL_B
clears only the first task's literal repeated-index selection
(sym_seq_sum.cxx 233-260,329-347). The other canonical chamber retains its old
value before adding the replacement term. This is source rw=0 semantics, not
ordinary assignment to every symmetry-equivalent coordinate. The new regression
records this behavior rather than silently changing the pinned algorithm.

## Upstream symmetry and diagonal test batch (2026-09-07)

The dedicated upstream_diag_ctr and upstream_reduce_bcast tests exercise their
original identities through the newer compressed tensor APIs, complementing the
earlier dense-port coverage. upstream_multi_tsr_sym preserves identical-input
A[ik]*A[jk] contractions into NS and SY output and compares their difference.
This explicitly exercises the source same-input symmetry-preservation branch.

In pinned sy_times_ns, A.write and B.write are commented out, while C.write is
active. upstream_sy_times_ns therefore distinguishes literal zero-A/B behavior
from a separately labeled nonzero deterministic fixture using the same indexed
equations. The nonzero case is not presented as literal upstream initialization.
All original residual inequalities remain intact; rank-dependent random values
are replaced by global-key fixtures for reproducible MPI-rank comparisons.

## Sparse multilinear operations

src/sparse_multilinear.rs ports the stored-entry TTTP/MTTKRP paths from pinned
interface/multilinear.cxx and interface/semiring.h. Factors are aligned with the
tensor mode's physical mapping and broadcast along complementary process fibers.
MTTKRP reduces the local stored-entry contributions on output fibers and routes
the resulting factor to the requested distribution. Canonical sparse owners
contribute once even when tensor data are replicated. No dense tensor conversion
or global tensor gather is used. TTTP retains stored keys and explicit zeros;
matrix TTTP supports both auxiliary orientations and balanced auxiliary blocks.
The direct Rust MTTKRP matrix API uses auxiliary-first factors, matching the
existing dense API. The source one-physical-axis-per-mode restriction remains.

Sparse weighted Solve_Factor shares the dense algorithm in src/solve_factor.rs.
Pinned multilinear.cxx:714-720 differs only in obtaining stored pairs versus
materializing dense local pairs; factor alignment, Gram construction and solves
then share the same source path. Lines 910-930 form q as the elementwise product
of non-output factor rows and accumulate weight*q*q^T using lower-triangle SYR;
weights are not squared. Reduce-scatter assigns Gram systems to fiber workers,
each worker calls POSV, and only solved factors are gathered within the fiber.
Sparse absence contributes zero without allocating the logical tensor domain.
The existing Rust padding, canonical-owner and collective INFO rules also apply
to this path. A module-local macro emits the algorithm for both concrete receiver
types; no runtime storage or native-library backend abstraction is introduced.

## Shared TTTP factor staging and sparse fiber-grouped MTTKRP

Dense TTTP now uses the same source mode-physical shard and complementary-fiber
broadcast as sparse TTTP, implemented in multilinear_factor.rs. Each root reads
only its valid factor shard/block, broadcasts its padded local buffer, and closes
the communicator explicitly. This replaces clone/slice plus replicated factor
redistribution; the underlying root read remains the current key-routed I/O path.
Both orientations and balanced auxiliary blocks retain the same scalar operation
order. Automatic memory-based division selection remains a separate pending item.

Sparse MTTKRP now calls the existing multilinear_kernel.rs source fiber-grouped
kernel instead of independently multiplying every sparse entry's factor rows.
Canonical local stored pairs are key-sorted across virtual blocks; mode-zero
fibers reuse the product of remaining factor rows, matching interface/semiring.h.

## Sparse contraction input-only pre-reduction

contraction.cxx:5024-5049 calls self_reduce on A, recurses if it changed, and only
then tries B. untyped_tensor.cxx:3939-4017 removes the first unmatched unique axis
and performs a summation with multiplicative identities before returning. The
four sparse/mixed Rust contraction entrypoints now follow that order, one axis
per recursive pass, before diagonal extraction and folding. Positional distinct
summation labels prevent unrelated repeated contraction indices from selecting
diagonals prematurely. Surviving mapping edges/topology remain; a removed mapped
axis becomes a replica fiber. Current indexed summation supplies communication.

This does not implement C-only indices by contracting then broadcasting. In the
source those indices use map_extra_indices (virtual phase one) and the general
sp_seq_ctr loop. That path remains pending and still returns OneOperandLabel(C)
from the folded entrypoints before communication; it is not counted as complete.

## General sparse key recursion and mapped replication

src/sparse_sequential.rs ports the nonsymmetric sparse-A/dense-B/dense-C branch
of sp_seq_ctr.cxx. Labels are normalized in A/B/C first-occurrence order and
traversed from highest to lowest. Sorted sparse keys restrict A-axis recursion;
B-only/BC/C-only dimensions are traversed without materializing sparse zeros.
It retains ((A*B)*alpha), contribution-before-old-C addition, left beta scaling
of nonscalar C, and the source distinct right beta scaling in the all-scalar path.
Repeated indices and A-only reduction belong above this local kernel.

src/sparse_contract_general.rs exposes the corresponding explicit physical-label
mapping. Singleton labels cannot be physically mapped (map_extra_indices).
Canonical A keys move once to mapped owners; count/payload broadcasts distribute
sparse local blocks along missing-label fibers. B/C root blocks use indexed reads,
B is broadcast, and C is reduced with beta applied only on the root contribution.
Only canonical reduced output owners restore the original C distribution. There
is no full tensor gather, dense expansion of A, or contraction-then-broadcast C
substitute. Source 2D sparse and virtual plan assembly, automatic mapping, and
other general sparse storage combinations remain separate pending paths.

## Pinned general sparse custom-function branch

sequential_function and contract_from_sparse_dense_function_on retain the actual
sp_seq_ctr.cxx custom branch, rather than supplying an algebraically broader
replacement. At the lowest normalized label, custom evaluation is implemented
only when A has no axis there (scalar A under source A-first label normalization)
and alpha is the multiplicative identity. The function result is accumulated as
old_C + f(A,B), following Bivar_Function::acc_f. Stored scalar zero is evaluated;
an absent sparse scalar performs no calls. The all-scalar branch instead performs
ordinary multiplication, ignores the function and retains right-beta ordering.
Other reached custom branches assert in the fixed source. Folded custom kernels
and their high-level dispatch are separate working CPU paths still to be ported.
No distributivity is inferred for a user function and no pre-reduction is added.

## Folded CSR custom functions and distributed panels

sparse_function_kernel.rs ports Bivar_Function::csrmm (functions.h:196-224) and
csrmultd (227-254), separately from the limited general sparse custom branch.
The first loops rowA/columnB/stored-A-entry; the second loops rowA/stored-A-entry/
matching-stored-B-entry. Both accumulate old_C+f(A,B), never evaluate missing
sparse entries, and preserve explicitly stored zeros. CSR custom dispatch
requires unit alpha (csr.cxx:135-183). The caller's arbitrary beta is applied
once as zero-fill or left scaling, following spctr_tsr.cxx:346-353 and 489-500.

gemm_sparse_dense_function and gemm_sparse_function reuse the existing source
2D panel broadcasts with those kernels. Only the first panel receives caller
beta; subsequent panels use one. Sparse and dense layouts remain distinct; a
user function is not disguised as a semiring multiplication implementation.
Output is restored to its original distribution after panel execution.

## Custom CSR sparse output

csr_sparse_output ports functions.h:255-333 symbolic reachable-column discovery,
direct first numeric write, subsequent ordered additions, and final old-CSR-first
csr_add. It exposes the raw source kernel without alpha/beta parameters. The
distributed SparseTensor::gemm_sparse_function requires unit alpha, accumulates
panels into an empty sparse product, then uses indexed sparse summation to update
the existing output, following home_contract at contraction.cxx:5275-5288.
This high-level merge right-multiplies old values by beta and adds product first
on collisions (spr_seq_sum.cxx:235-285). Old-only coordinates remain present with
zero values when beta=zero. This ordering is deliberately distinguished from
both the raw csr_add and existing ordinary sparse GEMM's beta convention.
No dense m*n intermediate is allocated; symbolic row workspace is O(n).

## Fullyfoldable high-order sparse custom contraction

sparse_fold_function.rs combines the existing source k/m/n/l fold layout with
the custom CSR kernels, for sparse/sparse-to-sparse, sparse/sparse-to-dense and
sparse/dense-to-dense operations. It uses the same axis permutations, batch
slices, mapped panel execution and output distribution restoration as ordinary
sparse folding. Repeated operand labels are extracted and output diagonals are
reinserted without modifying off-diagonal values. Projected Plan validation
occurs before communication. Unit alpha follows custom CSR dispatch. Input-only
pre-reduction is deliberately not performed for arbitrary functions, whose
distributivity cannot be assumed; nonfoldable indices remain an explicit error.

## Distributed dense bivariate contraction

dense_function.rs exposes explicit physical-label mapping and source broadcast /
local recursive custom kernel / output reduction. Its recursive leaf uses the
existing sequential_function, matching sym_seq_ctr.cxx:241-265: f(A,B), right
alpha scaling, then contribution-first addition. Canonical mapped roots read
only valid local keys. Rank-specific extents exclude cyclic padding before any
function evaluation, which is necessary for functions not annihilating zero.
Input/output repeated labels use diagonal extraction and output reinsertion.
No distributive pre-sum is performed; singleton labels remain physically unmapped.
Final canonical output contributions restore the original distribution by I/O.

upstream_bivar_function preserves test/bivar_function.cxx's n=5 shape [6,5,7,8],
f(a,b)=a*b+b*a, beta=.5 update, and strict absolute error <1e-6. A clone expresses
the source aliased old-A operand in Rust. Global-key values replace rank-seeded
random data; the acceptance reference is the original per-element identity.

## Indexed dense accumulator transforms and unary source tests

dense_transform.rs provides mutable-output bivariate transforms with independently
typed dense inputs. Each input's indices must occur in the output; repeated output
indices select its diagonal. Required input keys are read collectively, then the
function updates each selected local output exactly once, retaining its layout.
This pointwise/subset path does not convert an accumulator into additive function
contributions or assume a reduction law. It uses current indexed I/O; optimized
source redistribution/broadcast transform plans remain to be integrated.

The source bivar_transform identity c=a*c*a+b*c*b, endomorphism cube, and
univar_function .25*a+a^4 use original n=5 shapes and strict abs(error)<1e-6.
Global-key fixtures replace rank-seeded random initialization. The source unary
function's printed cube label is a typo; the test and implementation use its actual
fourth-power formula. Additional typed scalar-broadcast/diagonal checks in the
bivariate transform test are explicitly separate from the original source case.

## Custom name endomorphism tests

The dense custom endomorphism test retains empty-name identity and longest actual
string addition, with left operand winning ties. The transform caches actual
null-terminated name length and checks exact equality. Its sparse counterpart
checks stored entries only. Rust uses a fixed wire encoding, not a C++ ABI shim.
The sparse source uses Set with unique-value writes; Rust's required transport
monoid is supplied with unique canonical-owner writes so no duplicate addition
changes those values. Deterministic key-derived lengths replace random values.

## Executable aligned plan cost extraction

grid_plan_cost.rs derives source Replicated/Virtual/Local cost nodes from GridPlan's
mapped layouts and normalized labels. Message sizes, unique-label FLOP product,
virtual multiplicity and bookkeeping follow ctr_comm/ctr_tsr. The supplied node
counts are explicit topology facts. This is the inner aligned execution estimate,
not total plan time/memory; redistribution/folding and candidate generation remain
pending. See automatic-planning.md for pinned source integration boundaries.

## Unfolded dense candidate redistribution costs

redist_cost.rs implements dense est_redist_time/get_redist_mem and phase-based
can_block_reshuffle (untyped_tensor.cxx:3196-3244; redist.cxx:619-628). There is no
invented physical-permutation transpose surcharge. GridPlan::estimate_unfolded
adds these costs with the exact A/B residency, C roundtrip and maximum workspace
rule of contraction.cxx:2750-2810. These are source model estimates; actual Rust
key-routing and measured process memory remain separately documented.

## Exhaustive NS mapping variants and preflight

mapping_variants.rs ports get_num_map_variants, mapping construction and phase
equalization from contraction.cxx:2008-2041,2143-2388, plus get_choice/calc_idx_arr
from shared/util.cxx. It preserves raw discovery order and source quirks instead
of restricting candidates to aligned maps. mapping_preflight.rs ports the unique-
label NS part of check_mapping (1339-1572) and physical-chain self checks from
mapping.cxx:338-390. Three-way labels require identical maps; pair-only labels
require equal total phases, and mismatched physical dimensions must have the
source corresponding matched coverage. Singleton physical assignments fail the
same bookkeeping. Symmetric/repeated self-mapping, topology canonicalization and
automatic candidate execution remain separate unfinished responsibilities.

## Topology canonicalization and rank-local exhaustive enumeration

Variant::canonicalize follows switch_topo_perm (contraction.cxx:2044-2142): place
folded physical pairs consecutively in A/B/C order, reject conflicts, append unused
axes, select the first matching catalog topology, then rename physical axes.
visit_local_exhaustive follows 3031-3105: cumulative raw topology counts define
global IDs; only global_id % world_size == rank is decoded. Canonicalization and
preflight rejection leave holes, not renumbered survivors. Enumeration is local
and streaming; no candidate-array allgather or aligned-only replacement is used.
Memory/size filtering and selected-plan reconstruction remain separate stages.

## Normal mapping and rank-local normal enumeration

normal_mapping.rs directly adapts contraction.cxx:1586-2007 and 2392-2553:
common-index mapping, paired contracted dimensions, singleton restrictions,
A-fill/C-nonfill copying, and the repeated mapping/symmetry pass. NS tables are
explicit; compressed symmetry mapping is not claimed here. normal_search.rs
adapts 2834-2913 through mapping preflight: six source permutations, seven retained
old-layout subsets, fresh catalog choices, rank partitioning and original IDs.
The source permits invalid intermediate maps; preflight, rather than a validated
Distribution constructor, rejects these before candidate emission.

normal_search::reconstruct and mapping_variants::reconstruct_exhaustive adapt
contraction.cxx:3364-3427. Normal IDs retain 6*template+permutation encoding;
exhaustive IDs retain cumulative raw counts and rejected holes. Both rebuild the
selected map locally, with final preflight rather than a fallback candidate.

mapped_cost.rs adapts dense unfolded build_ctr (contraction.cxx:3557-3911),
ctr_2d_gen_build (ctr_2d_general.cxx:13-167), panel buffer/cost formulas (225-317),
and local scalar work (ctr_tsr.cxx:407-441). Source pair order is BC, CA or AB
according to the missing operand, with panels nested in normalized-label order.
Physical head communicator selection, total-phase LCM step counts, zero stationary
panel buffers and residual virtual-phase assignment retain source semantics.
The total estimate adds contraction.cxx:2739-2810 dense unfolded redistribution
terms. Folded/sparse estimates and actual candidate execution are not claimed.

dense_search.rs connects contraction.cxx:2834-3190 selection with 3280-3342
two-pass weighting and exhaustive refinement. Early mapped storage is summed as
double, as in the source's pre-integer-overflow rejection. Normal scores start at
DBL_MAX; exhaustive scores start at the normal time or normalized zero. Global
selection gathers one time/memory pair per rank, broadcasts the winner rank and
ID/cost payload, and rebuilds that ID locally. Full structural topology/node
facts replace pointer identity. No timing probes or alternate mapping fallback
are introduced during candidate selection.

dense_execution.rs retains the dense build_ctr/ctr_2d_gen_build metadata needed
at runtime: physical fiber axis, outer strip count, contiguous inner strip size,
panel edge length, and residual virtual phases. It executes existing ctr_2d and
contraction kernels, not a replacement aligned contraction. Input/output owner
redistribution uses the existing key-routed Rust path; optimized source reshuffle
execution remains a separate unfinished responsibility. SearchCache extends
signature-based reuse to the raw selected layouts with immutable model/catalog
configuration and explicit context lifetime. It caches no tensor contents or MPI
subcommunicators; execution creates/closes fibers explicitly.

ctr_2d::execute and raw dense execution now use the same Semiring/Wire algorithm
for integer, complex and user elements, not parallel f64 and custom implementations.
Panel scatter retains work + beta*old and local products retain right alpha.
Replication-layer presence follows build_ctr's missing-axis test even if all axes
are unused and its communicator lists are empty: that layer left-scales C by beta.
An all-scalar leaf with topology order zero has no such layer and retains the
source sequential right-beta special case. Ordered MPI user reductions replace
the previous f64-only panel reduction in this generic entry point.

folding::Plan::select adapts get_len_ordering/select_ctr_perm
(contraction.cxx:601-699, 765-921). Original normalized label IDs persist when
roles are permuted; lnmk remains defined by original A/B/C. Costs multiply each
original operand's virtual count and double the permuted third operand, which is
not necessarily original C. Summation follows permuted role order, costs map back
to original operands, and dense <= comparison chooses the last tie. The six
source transpose flags drive existing folded_f64 and its transposed-output swap.
Sparse permutation restrictions and partially folded symmetric tensors remain
separate unfinished paths.

fold_indices.rs adapts get_fold_indices/can_fold (contraction.cxx:405-594).
Original normalized IDs, literal relative-position comparisons, A/C/B traversal,
and scans starting at every dimension through its remaining symmetry group are
preserved. The sparse predicate is the literal odd-order or half-order<nfold
condition, not a corrected requirement that every label fold; triple/weigh labels
are rejected separately. fold_layout.rs adapts tensor::fold
(untyped_tensor.cxx:2922-3006): calc_dim local lengths are supplied explicitly,
sy_packed_size is evaluated per adjacent symmetry group, and selection uses the
terminal label. Folded index IDs are positions in the global fold list. No alias
buffer lifetime, local transpose or implicit unfolding is invented by this stage.

FoldLayout::transpose implements the ownership-based local storage conversion
needed by nosym_transpose/tensor folding: virtual blocks remain independent,
symmetry groups stay packed, and backward conversion restores original group
order. partial_fold.rs composes get_fold_ctr/select_ctr_perm/map_fold metadata
(contraction.cxx:701-1030). Full group lengths, selected-prefix permutations and
residual groups participate in estimates; group-terminal fold-index holes remain
explicit and are never converted to fictitious dimensions. Role membership is
independent of role order; transpose cost addition retains source role order.

partial_fold_kernel.rs adapts sym_seq_ctr_inr (sym_seq_ctr.cxx:746-930), folded
dimension collapse (ctr_tsr.cxx:253-320), and shared/iter_tsr.h GET_MIN_MAX,
CHECK_SYM and RESET_IDX. It preserves label-zero-fast traversal, initial call
before symmetry checks, non-SEQ nondecreasing linked-coordinate checks, source
packed-offset recurrence and beta-once scaling before batched GEMM.
The pinned CMakeLists.txt:86-120 does not define SEQ; strict AS/SH CHECK_SYM is
the alternate branch at iter_tsr.h:30-50, while this default port follows 51-64.

Source reinspection corrected a prior fold-metadata error: tensor::fold and
RESET_IDX use sy_packed_size (shared/util.cxx:10-35), which increments its group
extent for all link kinds. They do not use packed_size (38 onward), which shrinks
AS/SH extents. FoldLayout now retains local AS/SH diagonal holes with SY capacity;
the kernel shares that exact recurrence. Logical compact symmetry Layout remains
unchanged. This is a source-layout correction, not a tolerance adjustment.

folded_cost.rs adapts dense detail_estimate_mem_and_time
(contraction.cxx:2640-2810) and folded est_membw/est_fp (ctr_tsr.cxx:407-441).
Communication construction precedes folded sequential dimension collapse. Dense
fold residency includes all A/B/C mapped local bytes and has no extra sparse
formatting temporary. The source leaf model multiplies residual sizes by mn/mk/nk
and residual work by mnk but omits l; this estimator quirk is retained, while
actual folded BLAS execution continues to process every batch. Floating-point
work and total-time accumulation follow source order, not an integer work product.

dense_search now follows detail_estimate_mem_and_time's FOLD_TSR/can_fold branch
while evaluating each normal/exhaustive candidate. The explicit enable_folding
option models that source policy; dense custom functions remain ineligible.
Folded memory affects the same strict filters and weighted objectives as unfolded
memory. The selected fold descriptor is reconstructed from the winning raw map
after the existing winner-ID communication, not serialized with tensor contents.

Distributed folded execution preserves map_fold's local virtual-block transpose
before ctr_replicate/ctr_2d_general communication. Panel outer/inner metadata
selects whole virtual blocks, so packed leaves must not transpose again.
The shared recursive executor now accepts a statically typed leaf callback;
the ordinary semiring leaf remains contraction::virtualized. Folded leaves
apply beta once per output virtual block and call the packed sym_seq_ctr_inr
port, then inverse-transpose C before restoring its original distribution.

Node reordering ports topology.cxx:22-53's forward/inverse rank recurrences
and contraction.cxx:4690-4739,4841-4861's strict volume selection and local block
exchange. Reordered communicator peer counts are inter_node_lens-1, as in
topology.cxx:199-203, not the number of nodes including self. Packed f64 block
exchange retains source tags 1322/1323/1324 and C backmapping tag 1327.

Pinned replication-layer audit: ctr_comm.cxx:22-29 initializes layers to 1/0;
the only assignment from wholly unused topology axes at :61-70 is commented.
Consequently the raw executor must not activate speculative 2.5D splitting on
unused axes. The empty replication wrapper and its coefficient semantics remain.

original_peer_counts ports topology.cxx:138-154's stride/group/node-boundary
recurrence, including its accumulation order and integer np/ppn loop bound.
For topology [3,4] and ppn=4 the peer counts are [0.5,2], not [0,2] or [1,2].
Collective volume multiplies bytes converted to f64 by that average. Raw and
aligned costs, folded estimates and search facts now share this representation;
SearchCache uses to_bits for its node-fact key rather than truncating to integer.

Dense folded low-memory ownership follows contraction.cxx:5670-5812:
the source aliases the three home buffers into temporary tensor metadata,
runs sym_contract without retaining those home buffers, unfolds/restores C,
then A and B, and rejects zero-edge, sparse, custom-bivariate and aliased inputs.
Rust's distinct mutable borrows replace its pointer-identity assertions and
manual alias flags. This stage connects that policy to the existing NS folded
raw execution; automatic compressed-symmetry low-memory orchestration is pending.

The same lowmem_contract ownership transfer now applies to the generic raw
semiring executor. Its scalar coefficient ordering follows sym_seq_ctr rather
than assuming multiplication commutes; MPI reductions retain the explicit
noncommutative custom-operation flag. No f64 conversion of generic values is used.

Generic node-aware exchange follows the same contraction.cxx rank permutation
and source tags as the f64 folded path. A Wire serialization buffer replaces
the source runtime MPI element datatype, allowing non-Copy/custom Rust values
without exposing their object representation. Mutable low-memory inputs are
exchanged back; disposable immutable-input working copies are not.

Typed BLAS calls preserve interface/semiring.cxx's native float/double and
complex float/double GEMM dispatch. Complex values use the Rust-owned repr(C)
real/imaginary pair at this internal native boundary; MPI still uses Wire.
The folded contraction's transpose flag is ordinary T, not conjugate transpose.
Typed local folding preserves the existing contiguous batch offsets, beta-once
scaling and residual-index traversal rather than promoting values to f64.

Typed distributed folding reuses the same map_fold/ctr_replicate/ctr_2d/ctr_virt
sequence for f32, f64 and both complex types. Virtual blocks remain independently
packed once before communication and restored once afterward. Node permutation
uses Wire for every scalar; output reduction uses the established ordered
generic monoid MPI implementation. The low-memory ownership path still restores
the original typed A/B data rather than retaining home copies or f64 surrogates.

The transpose estimator intentionally remains element-count based across
scalar types: nosym_transp.cxx:759-793 accepts sr but does not use its element
size, feeding {1, total_elements} to its three models. Actual storage and
communication memory estimates still use the selected scalar's byte width.

Distributed POTRF/POSV/TRSM now dispatch to ScaLAPACK's S/D/C/Z families
without scalar promotion. The tensor-level algorithms retain matrix.cxx's
cyclic read_mat descriptors and triangle extraction, and the existing paired
prime-grid/virtual-column/identity-padding POSV path. Complex positive-definite
factorization is Hermitian; solve_tri's transpose boolean remains the source
plain N/T operation, not implicit conjugation. QR/SVD/eigensolver code is unchanged.

Distributed thin QR now uses S/D GEQRF+ORGQR and C/Z GEQRF+UNGQR. The
matrix.cxx sequence is unchanged: cyclic read_mat, native factorization,
retained packed R, explicit Q generation, and triangle extraction into thin
distributed factors. Reflector tau uses the source column-distributed numroc
layout. Both native workspace queries are honored; complex query sizes are
read from the real component without converting matrix data to a real type.

Thin distributed SVD now dispatches all S/D/C/Z GESVD families. The complex
bindings follow lapack_symbs.cxx:618-690: native singular values and auxiliary
RWORK are real, RWORK has 4*max(m,n)+1 entries, and returned tensor singular
values have zero imaginary part. Rust uses separate owned buffers instead of
the source's overlapping real/complex pointer aliases. The right factor is V^H
for complex scalars. The established one-row native grid adjustment and
requested-grid restoration remain, without gathering either matrix factor.

Four-type SVD rank/threshold postprocessing retains matrix.cxx:1103-1138's
inclusive magnitude boundary and computed-zero-rank/full-factor behavior.
The public threshold is converted to the scalar's real precision before
comparison, matching the source dtype threshold conversion. Complex singular
values have zero imaginary part, so real-component absolute value avoids
unnecessary squaring/overflow. Only the singular-value vector is read; factors
are sliced collectively in their existing distributed layouts.

Randomized-SVD source audit (matrix.cxx:1151-1190, matrix.h:324-334) confirms
that supplied U_guess is in/out and retains the final oversampled iterate.
The Rust argument is now Option<&mut Self>; after positive iteration counts it
receives the updated subspace before rank cropping. At zero iterations it stays
unchanged, including its distribution, because the source skips initial QR for
a supplied guess. This corrects the previous read-only cloned-guess behavior.

Remaining randomized-SVD fidelity work: source complex iteration/projection
uses plain A*A^T and U^T*A even though native complex SVD returns V^H; do not
silently substitute a Hermitian power iteration. Source complex random guesses
are real-only (tensor.cxx:1578-1608).

The former 48-bit LCG auto-guess has now been replaced by the exact MT19937-64
parameters in interface/common.cxx:29-43. Generator owns its state explicitly;
unit_interval divides the converted u64 result by converted u64::MAX exactly
as get_rand48 does, without endpoint clamping. Tensor fill_random consumes a
draw for every local storage slot before zeroing padding; f32 casts the sample
before typed span arithmetic. Randomized SVD retains its explicit seed API,
initializing the rank-local stream with seed+rank (wrapping u64 addition).
Persistent streams can be supplied directly to fill_random rather than relying
on the source's process-global RNG. Complex guesses with real bounds are real.
