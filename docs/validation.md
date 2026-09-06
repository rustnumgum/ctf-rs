# Validation evidence

## 2026-09-06: foundations and local numerical boundary

Environment: WSL Ubuntu-26.04, Rust 1.93.1, Open MPI 5.0.10.
Workspace: `/mnt/d/projects/ctf-rs`.
Build cache: `/home/xylxp/.cache/ctf-rs-target` (Linux filesystem).
Command: `bash scripts/acceptance-wsl.sh`.

### Exact checks: PASS at 1, 2, 4 ranks

* Physical/virtual cyclic offsets and padding; explicit virtual-block offsets.
* Upstream 6x4 topology / 3x2 intra-node permutation and inverse.
* Distributed additive writes, duplicate keys, arbitrary remote reads.
* Row-to-column distribution switch with virtualization, replication and return.
* Integer reductions without counting replicas multiple times.
* Zero-length dimensions and empty local shards.
* Scalars, custom Boolean semiring reduction, communicator exclusion/splitting,
  shared-node communicator and topology fiber.

These tests are new foundation checks, not a completed port of `test_suite.cxx`.

### Local numerical checks: 4 tests PASS

BLAS integer-valued GEMM: exact expected result.
LAPACK checks use the upstream decomposition Frobenius bounds, not eigenvector
entry comparisons. OPENBLAS_NUM_THREADS=1.

| Quantity | Error | Bound |
|---|---:|---:|
| QR orthogonality | 9.09e-16 | 1.5e-5 |
| QR reconstruction | 1.46e-15 | 4.5e-5 |
| SVD U orthogonality | 1.09e-15 | 1.5e-5 |
| SVD V orthogonality | 9.87e-16 | 1.5e-5 |
| SVD reconstruction | 1.33e-15 | 4.5e-5 |
| eigh orthogonality | 4.51e-16 | 1.6e-5 |
| eigh reconstruction | 9.27e-15 | 1.6e-5 |

Class R. One successful execution of each required configuration, no repeated
precision checks. Zero extra numerical diagnostic computations. Before this pass,
an MPI argument failure was fixed by using distinct allocated backing buffers
for zero-count Alltoallv payloads; empty Vec pointers had aliased. No tolerance
was changed. An earlier WSL kernel soft lockup/RCU stall was resolved by a
user-authorized WSL shutdown/restart; it provided no test pass evidence.

### Still outstanding

General distributed dense/sparse/symmetric contractions, optimized mapping and
redistribution kernels, node-aware contraction execution, plan/cost/cache paths,
distributed decompositions and solvers, multilinear operations, scheduling,
remaining upstream tests and benchmarks, and Windows native acceptance.
Local Cholesky is implemented but not yet covered by an upstream-derived test.
The faer replacement interface exists; a faer implementation is not supplied yet.
Full project completion is **not** established by the above results.

## 2026-09-06: dense views

`cargo test --test dense_views` with the Cargo runner set to
`mpirun --oversubscribe -n N`, for N=1,2,4: PASS in all three configurations.
Each configuration executed once; zero diagnostic computations.

Exact integer checks cover offset slices, local virtual-block packing, physical
owner shifts with MPI_Sendrecv, tensor-axis transposition with mapping
transposition, nested slices, one-element/empty slices and repeated-label
endomorphisms. The new slice retains mappings and does not gather the global
tensor. Rank-shift semantics follow upstream `redistribution/slice.cxx`.

This closes only these new tests. Slice insertion/accumulation, arbitrary index
permutations, full summation and contraction are still outstanding.

## 2026-09-06: topology candidate ordering

`cargo test --test topology_candidates`: 2 tests PASS, one execution, exact
integer comparisons. Tests preserve the ordered prime-power divisor enumeration
for sizes 1, 4, 7, 12, adjacent-dimension folding order, and permutation/folding
deduplication order. Implementation ports `get_all_shapes`, `peel_torus`, and
`peel_perm_torus` from the pinned `src/mapping/topology.cxx`.
This supplies candidate topology shapes, not a completed contraction planner.

## 2026-09-06: inter-node grids

`cargo test --test node_aware`: exact branch-order and grid-invariant tests PASS.
The source's retained-tree traversal was subsequently preserved explicitly to
handle the zero-dimensional, one-process topology; the new `scalar_topology`
check PASS with the other two already-passing checks filtered out.
No floating-point computation or precision study was involved.
Node-aware contraction communication remains outstanding.

## 2026-09-06: physical-axis assignment

`cargo test --test map_tensor`: 3 exact tests PASS in one execution.
Covered longest-local-edge selection, adjacent-axis folding, fill/restriction
behavior, symmetric LCM virtualization and rejection of impossible/over-limit
candidates. No floating-point checks or additional diagnostic computations.
The assignment primitive is exposed for planner integration; the tensor
constructor still takes an explicit distribution. Automatic plan search and
all upstream mapping edge cases are not yet claimed complete.

## 2026-09-06: local dense summation kernel

`cargo test --test sequential_sum`: 4 exact integer tests PASS in one execution.
Covered transpose with alpha/beta, reduction plus broadcast, repeated input and
output indices, zero-length reduction, and custom-function application after
input scaling. The kernel processes local column-major blocks; distributed
tsum replication/reduction and compressed-symmetry integration remain pending.

## 2026-09-06: virtual and replicated summation layers

`cargo test --test virtual_sum`: 3 exact tests PASS once. Covered beta-on-first-
output-block-visit, virtual broadcast/permutation and virtual diagonals.
`cargo test --test replicated_sum` under MPI runners with 1,2,4 ranks: each PASS
once. Covered input block broadcast, native f64 block Allreduce, retaining beta
only on reduction roots, nested virtual reduction and zero-count output blocks.
Values in these MPI checks are exactly representable small integers; equality
is exact, with no floating-point tolerance study or diagnostic reruns.

The communication sequence is ported from `tsum_replicate::run`, while virtual
traversal follows `tsum_virt::run`. No global tensor gather is used. This is the
native-double layer; custom reduction operators, automatic mapping, complete
Tensor-level summation and packed symmetry are still outstanding.

## 2026-09-06: aligned Tensor summation

`cargo test --test tensor_sum` with MPI runners at 1,2,4 ranks: all PASS once.
Exact small-integer-valued f64 checks cover local-index reduction, physical-axis
reduction with virtual blocks and padding, output broadcast and transposition.
The `sum_from_aligned` Tensor entry point uses explicitly aligned unique-label
distributions; it creates and explicitly closes the required topology fibers.
Automatic distribution alignment, repeated-label Tensor summation and generic
algebra communication remain separate unfinished requirements.

## 2026-09-06: local reference contraction

`cargo test --test sequential_contraction`: 4 tests PASS once using exact integer
and Boolean semiring expectations. Covered matrix product with alpha/beta,
repeated input labels, scalar products, empty contraction dimension, and custom
bivariate function before alpha scaling. As upstream's local reference kernel,
the supplied output block is beta-scaled in full; Tensor-level diagonal extraction
is not implemented by this primitive. BLAS folding, distributed contraction
communication and Tensor-level contraction remain outstanding.

## 2026-09-06: virtual and replicated contraction

`cargo test --test replicated_contraction` under MPI runners at 1,2,4 ranks:
each PASS once, using exact small-integer-valued results. Covered virtual
reduction with beta applied once, root-only beta before native MPI_Reduce,
two-input broadcast and cleanup of nonroot input replicas. The implementation
preserves ctr_replicate's Reduce semantics rather than substituting Allreduce.
These are local-block execution layers; 2D communication, BLAS folding and
Tensor-level contraction integration remain unfinished.

## 2026-09-06: folded local BLAS batches

`OPENBLAS_NUM_THREADS=1 cargo test --test folded_contraction`: 3 tests PASS once.
Exact small-integer-valued f64 cases cover contiguous batches with alpha/beta,
planner-supplied transposed-output flags/operand swapping, and zero reduction
length. The entry point calls the replaceable LocalKernels trait, initially BLAS.
It executes an already-folded plan; automatic folding and the 2D communication
planner remain unfinished.

## 2026-09-06: 2D communication execution level

`cargo test --test ctr_2d` with MPI runners at 1,2,4 ranks: all PASS once with
exact small-integer-valued results. Covered cyclic input panel broadcasts,
moving-output MPI_Reduce to cyclic owners, and noncontiguous output scatter with
beta. Code also carries upstream replication-layer partitioning; that branch
has not yet been separately accepted. This is one execution level with explicit
panel metadata and a child kernel, not the complete 2D plan builder or a complete
Tensor-level distributed contraction. No global tensor gather is performed.

## 2026-09-06: Tensor 2D GEMM integration

`cargo test --test tensor_gemm` with OPENBLAS_NUM_THREADS=1 and MPI runners at
1,2,4 ranks: PASS. Exact integer-valued cases cover a 5x7 times 7x3 matrix product,
nonuniform cyclic shards, preserving the output's original distribution and zero
reduction length. Grids were 1x1, 2x1, 2x2 respectively. Each configuration passed
once after fixing a zero-count Bcast buffer-address error discovered on the first
1-rank attempt; no numerical tolerance was changed or diagnostic study run.

`Tensor::gemm_2d` aligns cyclic reduction phases, redistributes local data to the
explicit grid, executes panel broadcasts plus BLAS, and restores the output
distribution. No global tensor gather is used. This is explicit-grid f64 matrix
multiplication, not general indexed contraction or automatic cost-based planning.

## 2026-09-06: native custom monoid reduction

`cargo test --test custom_reduce` under MPI at 1,2,4 ranks: PASS once each.
Exact tests use noncommutative affine-function composition to check rank order,
a closure-captured modular-addition algebra, multiple elements and zero counts.
The implementation creates an MPI contiguous element datatype and user operation,
executes native Allreduce, and explicitly frees both handles before returning.
After these passes a fixed-width serialization assertion was added before the
FFI call to enforce the Wire buffer-size contract; no numerical rerun was needed.
The existing tensor reduction API is unchanged; generic contraction/summation
communication integration remains unfinished.

## 2026-09-06: generic aligned Tensor summation

`cargo test --test algebra_sum --test tensor_sum` with MPI runners at 1,2,4:
all PASS once after a compile-time temporary-borrow fix. New exact checks cover
integer and Boolean semiring Tensor reductions. The existing f64 Tensor sum
checks were rerun because this refactor changes their executed communication
path to the native MPI user operation. Other passing suites were not rerun.
`sum_from_aligned` now accepts any Semiring with Wire elements; it remains an
explicitly aligned, unique-label interface, not complete general summation.

## 2026-09-06: complex scalar algebra

`cargo test --test complex_scalar` under MPI at 1,2,4: PASS once each after a
test-only temporary-borrow compile fix. Exact small-integer complex values cover
f32/f64 arithmetic, conjugation, norm squared, local indexed contraction, native
MPI user reduction, distributed key I/O and scaling. Complex native BLAS/LAPACK/
ScaLAPACK bindings are not supplied by this scalar-algebra addition.

## 2026-09-06: summation distribution alignment

`cargo test --test sum_remap` under MPI at 1,2,4: PASS once each, exact integers.
The new `sum_from_on_grid` assigns union indices with map_tensor on a requested
topology, redistributes local data, executes generic aligned sums, and restores
the output layout. Tested combined reduction/broadcast and transpose on uneven
dimensions, including a 2x2 grid. No global tensor gather is used. Repeated labels
and cost-based choice among topology candidates remain pending.

## 2026-09-06: generic contraction root reductions

`cargo test --test algebra_contraction` under MPI at 1,2,4: PASS once each with
exact integer/Boolean results. Generic replicated contraction now uses MPI user
Reduce with root-only beta and native datatype/operator cleanup. Checks cover
integer and Boolean contractions and a nonzero reduction root. Generic Tensor-
level contraction and native typed 2D kernels beyond f64 are still pending.

## 2026-09-06: generic aligned Tensor contraction

`cargo test --test tensor_contract` under MPI at 1,2,4: PASS once each using exact
integer matrix expectations. The new `contract_from_aligned` identifies topology
fibers from aligned unique-label maps, invokes generic replicated/virtual/local
layers, and restores valid output replicas from canonical owners. Tests cover
combined physical and virtual k reduction and preserving input tensors.
General repeated-index contraction, automatic remapping/folding and planning are
not established by this explicitly aligned entry point.

## 2026-09-06: dense indexed API integration

Previously completed `contract_remap` PASS at 1/2/4 ranks is retained without
rerunning it: high-order two-index reduction with differing input distributions.
New `dense_semantics` PASS once at 1/2/4 ranks: diagonal extraction/replacement,
repeated-label sums and contractions, output off-diagonal preservation, nonlinear
unary sum ordering, empty diagonal reductions, offset slice insertion, and
duplicate-key affine writes. All those checks use exact integers.

`upstream_dense` migrates the numerical identities of upstream `diag_ctr.cxx`
and both `reduce_bcast.cxx` variants. PASS at 1/2/4 ranks, original bounds 1e-10
(absolute trace difference) and 1e-6 (Frobenius residual). Observed residuals
were zero. No stricter check or precision explanation was pursued.
`subcomm_dense` PASS at 1/2/4 world ranks: split-context repeated-label sum,
generic contraction and MPI+BLAS, including two simultaneously active groups.

Each newly required configuration ran once, no numerical diagnostic computations.
Representative timing/RSS measurements are separately recorded in `performance.md`.

Diagonal projection currently uses canonical-key redistribution and rank-local
storage; the source's optimized dense diagonal extraction via mapped summation
is not yet an exact communication-level port. This fidelity work, general
cost-based planning, sparse/packed symmetry and distributed decompositions remain
explicitly unfinished. Passing numerical identities do not close those gaps.

## 2026-09-07: owned sparse matrix formats and local kernels

WSL Ubuntu-26.04, source `/home/xylxp/ctf-rs-work`, target cache
`/home/xylxp/.cache/ctf-rs-target`: `cargo test --test sparse_formats` PASS,
5 tests. Initial compilation exposed missing explicit PartialEq method bounds;
those were fixed before the single numerical acceptance run. No numerical
diagnostic runs or repeats of passing dense suites.

Exact i64/bool acceptance covers one-based IA/JA/row encodings, conversion with
unsorted input and explicit zeros, duplicate COO preservation, cyclic partition
and assemble with 1/2/4/9 parts, empty matrices/parts, CCSR logical row count 2^40,
CSR and CCSR sparse union, CSR*dense, CSR*CSR, CCSR*dense, alpha/beta and prior
output, cancellation retaining structural zeros, last-column padding, and a
Boolean semiring product. Partition counts are local strips, NOT MPI rank runs.

DIGIT / PASS: class R, reference pinned source layout/kernel rules and exact
analytic integer/Boolean results, tolerance 0, 5/5 checks. This closes only these
local paths; distributed sparse execution, upstream sparse test migration and
Windows native acceptance remain open.

## 2026-09-07: executable grid plan cache

WSL Linux source copy and Linux target cache: `plan_cache` and affected
`contract_remap` PASS at 1/2/4 MPI ranks, once per configuration, exact i64
criteria. No numerical diagnostics or tighter checks were needed.

`plan_cache` exercises a miss then a hit with equivalent renamed labels and
changed input values/alpha/beta; different requested topology and input mapping
produce misses. It checks actual contraction values, restored output layout,
standalone prepared-plan execution, explicit clear, scalar operands, zero-length
reduction, and repeats the operations in split subcommunicators. Empty local
shards arise with the 1x1 output at multiple ranks. `contract_remap` rechecks the
existing higher-order two-contracted-label path affected by extracting mapping
preparation into GridPlan. Other unchanged passing tests were not rerun.

DIGIT / PASS: class R, exact layout/cache expectations and integer contraction
identities, tolerance 0. This is explicit-grid mapping-plan reuse, not acceptance
of automatic candidate selection, cost prediction, low-memory execution or
full cc4s plan-cache parity.

## 2026-09-07: distributed performance-model QR/SVD update

`model_training` PASS once at WSL 1/2/4 ranks, including split subcontexts.
Exact checks cover circular history, clipped linear prediction, diagnostic totals,
inactive tuning below threshold, and cubic feature order/prediction. Training
checks cover full-rank fits, rank-deficient fits and all observations on rank 0
(other ranks contribute zero local reduced systems). Training uses local QR and
MPI allgather of only R/y before a reduced DGELSD solve on each rank.

The acceptance quantity is the synthetic observation reconstruction norm, using
the existing upstream QR reconstruction rule m*n*n*1e-6 (n=2). No upstream model
trainer test was claimed as migrated. Largest observed residual was about
1.14e-12, within all configuration bounds (2.56e-4 to 1.024e-3). No comparison of
individual rank-deficient coefficients, precision chasing or diagnostic runs.

DIGIT / PASS: history/features exact, reconstruction within the fixed QR bound.
Numerical verification closed for these paths. Subsequent-update regularization
with nonzero prior average, model persistence, initial coefficient tables and
planner integration remain outside this batch's verified coverage. This is not
a validated wall-time predictor or completed automatic planner.

## 2026-09-07: static model bank and CPU cost formulas

WSL `cost_models`: 3/3 tests PASS once. Strict equality checks cover all 32 CPU
seed arrays through write/load roundtrip, broadcast/reduction/all-to-all feature
construction and builtin/custom dispatch, zero message/one-rank semantics,
all four CPU local contraction model choices, and transpose prefix boundaries
4/64/65 plus no-op permutations. Integer-valued replacement coefficients make
feature/dispatch checks exact; no observed wall-time accuracy is claimed.

DIGIT / PASS, class R, exact coefficient and source-formula checks; no numerical
diagnostics or repeats. Existing distributed training tests were not rerun since
their update algorithm was unchanged. Automatic candidate generation/selection,
communication-tree memory costs, model instrumentation and Windows acceptance
remain open.

## 2026-09-07: cross-rank candidate selection

WSL `selector` PASS once at 1/2/4 MPI ranks, also within split contexts. Exact
checks cover a plan available only on the last rank, size/payload broadcast,
received plan signatures/maps/scalars, actual integer contraction execution,
lowest-rank selection when IDs coincide, absent IDs, exhaustive flag matching,
time/memory filters, changed-signature invalidation, reset and virtual replication
factor. Metadata uses exactly representable values; no timing accuracy is claimed.
DIGIT / PASS, class R, tolerance 0. Automatic candidate discovery and full-tree
time/peak-memory estimation are not validated by this explicit selection test.

## 2026-09-07: recursive tree cost arithmetic

WSL `plan_cost` 2 tests PASS with exact synthetic coefficients: virtual repeats,
panel layers including layers>steps, nested panel child-layer reset, replicated
broadcast/reduction, auxiliary maxima and additive work memory. Source inspection
then identified that ctr_virt inherits a zero internode-volume estimator rather
than multiplying child volume; this fidelity defect was corrected and only the
affected recursion_and_layers test rerun, PASS. Unchanged replica test was not
repeated. DIGIT / PASS, class R, exact formulas; no wall-time or RSS accuracy claim.

## 2026-09-07: time/memory objective selection

WSL selection_objective PASS once at 1/2/4 ranks and split subcontexts. Exact
synthetic costs check time-optimal versus memory-weighted winners, equality at
the strict memory limit, no feasible candidate, the 1e-8 cutoff, local/rank tie
order and exclusion of exhaustive candidates. The selected plan executes an
exact integer contraction. DIGIT / PASS, class R, tolerance 0. This does not
validate automatic candidate enumeration, exhaustive refinement, total-memory
estimation or a low-memory executor.

## 2026-09-07: local compressed symmetry

symmetry_layout: both local tests now PASS. Exact checks enumerate canonical
3-axis coordinates at length 4 for SY/AS/SH, verifying sizes and column-major
offsets; mixed groups, scalars, empty groups, AS permutation parity, signed
additive writes and SH diagonal zeros are covered. Initial mixed-group test
expected 64 incorrectly; the analytical offset is 4+6+4*12=58. Corrected that
reference and reran only the failed test; signed-storage pass was not repeated.
DIGIT / PASS, class R, tolerance 0. No floating-point runs. Distributed packed
symmetry, alignment multiplicities and upstream symmetry CPU tests remain open.

## 2026-09-07: symmetry index factors

WSL sym_indices: two exact tests PASS for permutation signs, pair alignment,
AS/SY sign differences, partially shared groups, contraction factorial factors,
AS summation cancellation and SH versus SY reduction factors. First run found
usize intermediate underflow in i-run+1; changing it to i+1-run preserves the
source signed arithmetic result. Only the failed multiplicity test was rerun
after that correction; the passing sign test was not repeated. DIGIT / PASS,
class R, tolerance 0. Three-operand alignment and distributed symmetry execution
remain unfinished.

## 2026-09-07: combined alignment/permutation/packed-operation batch

WSL sym_triple 2/2 PASS once: sixteen three-party permutation/sign combinations,
AB-only/BC-only/AC-only groups, partner NS boundaries and scalars. Then new
sym_operations 2/2 PASS once: broken versus preserved symmetry permutations,
factorial discovery, contraction permutation path, circular generator parity,
packed iteration offset bijection for mixed SY/AS/SH groups, empty/scalar domains,
repeated-label scaling/endomorphisms and no writes to AS structural diagonals.

DIGIT / PASS, class R, exact integer/index/sign expectations. No failed numerical
runs or extra diagnostics; previous passing suites were not rerun. These remain
local checks, not full upstream distributed symmetry acceptance. This batch
combines the related APIs and tests in one stage commit rather than stopping
after the initial triple-alignment helper.

## 2026-09-07: high-order BLAS layout and distributed execution

One combined source/API review of the delegated local fold implementation and
its parent MPI integration was completed before acceptance. Local folding 3/3
PASS once: reordered high-order/multiple contraction labels/shared batches,
outer products, alpha/beta and explicit unsupported-fold rejection. New
tensor_blas_fold PASS once at 1/2/4 ranks, including split subcontexts, uneven
high-order distributed shapes, output layout restoration, empty contracted
dimension, scalar operands and outer products. Those f64 fixtures represent
small integers and use exact equality.

upstream_gemm4d NS PASS once at 1/2/4 ranks. Maximum observed elementwise
associativity difference across those runs was approximately 4.0e-15, below the
source strict 1e-6 bound. DIGIT / PASS; no precision explanation, stricter bound,
extra numerical diagnostics or repetitions of earlier passing suites.

This closes the new explicit-grid full-NS BLAS path only. SY/AS upstream test
branches, partial folding, automatic full candidate discovery, low-memory
execution, distributed decompositions and Windows native acceptance remain open.

## 2026-09-07: native distributed Cholesky / triangular solve

Combined review of delegated FFI and parent descriptor/local-storage integration
completed once before execution. distributed_matrix PASS once at WSL 1/2/4 MPI
ranks, with OPENBLAS_NUM_THREADS=1 and the Linux source/cache directories.
The test covers upper/lower Cholesky for dimensions 1,4,5, empty mathematical
local rows at n=1, uneven cyclic blocks, all lower/upper left/right transpose
triangular solves for a 4x7 RHS, and split subcommunicators. Factors/solutions
retain the caller's original distribution; reconstruction uses distributed GEMM.

Acceptance: source test_la.py L1 reconstruction error <=1e-3 OR relative L1
error <=1e-3; opposite-triangle Frobenius norm <=1e-6. All passed. DIGIT / PASS,
class R; no tighter checks, vector-component comparisons or numerical diagnostics.
No global tensor gather and no C++ CTF linkage. This validates only the stated
f64 operations, not distributed QR/SVD/eigh, solve_spd, all scalar types, or
native Windows. Whole-goal acceptance remains incomplete.

## 2026-09-07: distributed QR and SVD

One combined review of new delegated FFI and parent output layout integration
preceded tests. distributed_qr_svd PASS once at WSL 1/2/4 ranks and split
subcommunicators. Shapes 13x7, 5x8 and 1x1 cover tall/wide matrices, nonuniform
local dimensions and empty mathematical local rows/columns. Reconstruction and
orthogonality use distributed GEMM, no matrix gather. Only singular-value vectors
are collectively read for test reconstruction.

Original bounds retained: QR and both SVD orthogonality Frobenius norms <=m*n*1e-6;
reconstruction <=m*n*n*1e-6. Largest printed QR reconstruction residual ~2.64e-15,
SVD ~5.54e-15; all required norms pass. DIGIT / PASS, class R. No coefficient/
singular-vector component comparison, tighter precision run or repeated passing
suite. Complex/f32, eigensolvers, rank truncation, randomized paths and Windows
native remain unfinished; the whole goal is not complete.

## 2026-09-07: truncated/randomized SVD paths

Combined test/implementation review performed once. Initial compilation required
explicit context/runtime lifetimes in fixture helpers; fixed before numerical
execution. distributed_svd_paths PASS once at WSL 1/2/4 ranks and split contexts:
rank and threshold truncation, equality at threshold, source zero-retained-rank
behavior, supplied guess, fixed-seed random guess, oversampling and one power
iteration on an exact rank-two fixture. Shapes are checked exactly; reconstruction
and orthogonality retain m*n*n*1e-6 and m*n*1e-6 Frobenius bounds. DIGIT / PASS,
class R. No numerical diagnostics or repetitions of passing suites. This is not
a stochastic accuracy study or evidence for arbitrary fixed-rank approximation.

Pre-existing broad local edits were preserved and excluded from this stage
commit. Acceptance ran against the current working copy, not a clean-tree full
regression. Full CPU coverage and Windows native acceptance remain incomplete.

## 2026-09-07: square-subworld symmetric eigensolver

distributed_eigh PASS once at WSL 1/2/4 parent ranks and split subcontexts,
covering n=5 dense symmetric and degenerate spectra, plus n=1 with empty local
fragments. The actual eigensolver grid uses 1/1/4 computing ranks respectively,
as the pinned nonsquare-process strategy requires. Full parent distributions
are reconstructed after the native solve. No vector-component comparisons.

Original scalapack_tests/eigh.cxx orthogonality/reconstruction bounds n*n*1e-6
retained. Maximum printed reconstruction norm ~3.58e-14; all criteria pass.
DIGIT / PASS, class R; no numerical diagnostics or repeated passing checks.
Combined FFI/subworld integration review was performed once. An agent's accidental
cargo check was interrupted during compilation; it ran no numerical tests.
Prior broad dirty edits remain preserved outside this commit. Remaining scalar
types, SPD solve and full CPU/native Windows acceptance are not closed.

## SPD and dense TTTP batch (2026-09-07)

WSL Ubuntu-26.04, source /home/xylxp/ctf-rs-work, Cargo cache on Linux FS.
distributed_spd passed once each with 1/2/4 MPI ranks, world and parity split
contexts. Shapes (1,1), (5,3), and n=11 with RHS counts 1/4/12/15/31 cover
empty mathematical local shards, nonuniform blocks, virtual columns on two
ranks, and padded square-block PDPOSV. Original test_la.py test_solve L1
residual <=1e-3 OR relative L1 <=1e-3 retained; result layout exact.

distributed_tttp passed once each with 1/2/4 MPI ranks. It covers vector products (exact integer-valued results) and
matrix factors (original test_einsum.py global L1 <=1e-5), both auxiliary
orientations, divisions 1/3 for k=5, selected/all modes, first/third-mode
physical distribution, empty local shards, world/parity contexts. Matrix
fixtures use fractional entries; no extra precision or backend comparison.
Factor communication currently uses general redistribution rather than the
source's specialized broadcast. Explicit blocked TTTP is not completion of
automatic low-memory planning. Sparse TTTP and other multilinear routines,
full CPU coverage and native Windows acceptance remain open.

## Dense MTTKRP (2026-09-07)

distributed_mttkrp passed once each at 1/2/4 MPI ranks in the Linux work copy,
including world and parity subcommunicators. Shape [3,2,5] and [1,2,1], every
output mode, first-mode cyclic, third-mode physical and third-mode virtual-2
layouts cover mode-aligned factor broadcasts, complementary-fiber reductions,
nonuniform partitions, empty shards and explicit output redistribution.
Vector fixtures use exact integer arithmetic; auxiliary-first matrix factors
use fractional values, k=3, and the pinned test_einsum.py global L1 <=1e-5.
The reference evaluates the defining contraction on the small deterministic
fixture, not eigenvectors, a different backend, or a gathered tensor.
Combined delegated-kernel/test integration was reviewed once before execution;
the test oracle's auxiliary coordinate and scratch-buffer reuse were corrected
before that first run. All passed: DIGIT / PASS, no diagnostics or repeats.
This closes the tested dense f64 MTTKRP path, not sparse/generic multilinear
scope, Solve_Factor, tensor SVD, or the full Windows-native acceptance.

## Indexed tensor SVD and reshape (2026-09-07)

distributed_tensor_svd passed on WSL Ubuntu-26.04 with 1/2 ranks initially;
4 ranks passed after the native one-row SVD layout correction below,
including parity subcommunicators. Native matrices stay distributed throughout.
Source test_la.py::test_tsvd shape [4,5,6,3] and output layouts ija/akl,
ika/ajl, iakj/la, alk/jai exercise input regrouping and arbitrary auxiliary
placement. Smaller [3,2,2] fixtures also cover explicit rank-one truncation
and fixed-seed randomized tensor SVD, without adding stochastic studies.
Reconstruction uses the source Frobenius norm / total elements <1e-6;
factor orthogonality uses source L1 <=1e-3 OR relative L1 <=1e-3.
No eigenvector or singular-vector component comparisons are used.

Reshape preserves exact flattened values and the requested distribution for
[3,2,2] -> [4,3] with ownership changes and [1] -> [1,1] with empty shards.
The combined module/test review corrected the test Gram-output index labels
before the first run. DIGIT / PASS for 1/2/4 ranks; no repeated passing runs.
Other scalar types, optimized merge/split reshapes, tensor-train/batched SVD
and remaining CPU/native Windows coverage are not claimed complete.

The initial four-rank randomized projection had shape 1x2 on a 2x2 grid.
Live GDB stacks found one rank in PDLARF/DGSUM2D and others in PDGESVD's final
DGAMN2D. A trial 1x4 grid failed identically and was removed. Source diagnosis
then identified PDNRM2's documented N=MX=INCX=1 ambiguity: only the tail owner
receives its norm, producing inconsistent TAUP and conditional collective calls.
The correction selects a full-rank Nx1 grid before the one-row native call,
then restores the requested U/VT distributions. The affected four-rank test
passed; the prior 1/2-rank configurations and tolerances were unchanged.
Both hung runs were explicitly terminated after stack diagnosis, not restarted
on an observation timeout. No precision diagnostics were performed.

## Distributed dense Solve_Factor (2026-09-07)

distributed_solve_factor passed once each at 1/2/4 MPI ranks in the Linux work
copy, including parity subcommunicators. Shapes [3,4,5] and [1,3,2], every
output mode, first-mode cyclic and third-mode physical/virtual-2 layouts cover
factor broadcasts, Gram Reduce_scatter, padded RHS Scatter, local DPOSV and
solution Gather/redistribution. Rank-two deterministic positive Gram systems
use an analytic solution to construct RHS, with test_Solve_Factor_mat's
numpy.allclose rule abs(error) <=1e-8 +1e-5*abs(reference) per component.
The result distribution matches RHS exactly. Zero weights exercise singular
systems: every rank returns Err(1), with no stranded collective participants.

Combined FFI/test integration was reviewed before the first run; all passed,
DIGIT / PASS. No reruns or precision diagnostics. Source's random sparse
rank-ten fixtures have not been migrated by this dense-path test, and sparse
Solve_Factor and broader CPU/native Windows acceptance remain open.
