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

## Distributed sparse storage/I/O (2026-09-07)

distributed_sparse_io passed once each with 1/2/4 MPI ranks on the Linux work
copy, including parity subcommunicators. Exact i64 cases cover duplicate
additive/scaled writes, beta applied once per touched key, unchanged old-only
keys, repeated/out-of-order/missing reads, explicit-zero retention and separate
pruning, scaling/stored transforms, slice/permutation, physical/virtual layout
switches and full replication without reduction overcount.

A 1,000,000 x 1,000,000 logical tensor stores at most two local entries and
successfully reads/reduces them without dense allocation. Custom max monoids
use i64::MIN as the absent identity; a noncommutative first-nonzero monoid
checks source new-before-old overlap order. The combined source/test review
resolved the source's exact operand order before the first run. All passed,
DIGIT / PASS with exact comparisons; no reruns or numerical diagnostics.
This is sparse storage and I/O coverage, not yet distributed sparse/mixed
contraction, compressed symmetry or the complete upstream sparse CPU suite.

## Distributed sparse and mixed sums (2026-09-07)

distributed_sparse_sum passed with 1/2/4 MPI ranks, including parity contexts.
The original sptensor_sum key/value fixture produces union keys {1,2,3,4,8},
key 2 = 66 and sum 76. Exact i64 fixtures cover sparse/sparse and sparse/dense
permutation, reduced labels, broadcasts, trace, diagonal-only destination
updates, affine alpha/beta, dense-to-sparse conversion, virtual-2 distribution
changes and explicit zeros after alpha=beta=0.

Combined source/test integration was reviewed before execution. The first
1-rank run exposed an arithmetic typo in the test trace oracle: 3*10 +
2*(2+5+7) is 58, not 62. The oracle was corrected without changing code or
tolerances; the affected run and first 2/4-rank runs passed. DIGIT / PASS,
exact comparisons only, no repeated passing tests or precision diagnostics.
Full sparse contraction and optimized sparse sum execution remain open.

## Distributed sparse matrix contractions (2026-09-07)

distributed_sparse_gemm passed once each with 1/2/4 MPI ranks in WSL's Linux
work copy, including parity contexts. Shapes (m,k,n)=(5,7,3),(1,1,1),(3,1,5)
cover sparse*sparse -> sparse/dense and sparse*dense -> dense, alpha=2/beta=3,
nonuniform/empty panels, rectangular and square process grids, and restoration
of virtual-2 output layouts. Entirely empty A still applies beta correctly.
All arithmetic fixtures compare exact i64 values. A custom min-plus semiring
checks nonnumeric-zero identity and distributed panel/kernel composition.

Source protocol and delegated kernel/tests were reviewed before execution.
The initial compile found byte payloads passed to the typed Wire broadcast;
this was corrected to the existing raw-byte MPI FFI entry point before any
numerical test executed. All runs passed, DIGIT / PASS; no reruns or precision
diagnostics. This does not close arbitrary-order sparse contraction, automatic
planning, moving sparse output panels, node-aware sparse execution or the
remaining native Windows/full CPU acceptance.

## High-order sparse folding and MP3 (2026-09-07)

distributed_sparse_fold and upstream_sparse_mp3 passed once each with 1/2/4
MPI ranks on the Linux work copy, including parity contexts. Exact i64 folding
tests merge two contraction labels, retain batch labels, permute all operands,
restore virtual-2 output layouts, and cover outer products/scalar dot products.
Sparse and mixed inputs dispatch through distributed matrix panels. A two-entry
1,000,000 x 1,000,000 tensor reshapes to a trillion-element vector without
dense allocation. Input pairs/distributions stay unchanged.

The upstream MP3 dense-T equation chain compares dense and sparse-integral
execution with the original abs((dense-sparse)/dense)<1e-6 criterion. Energy
is approximately -2.74094e-3; maximum observed relative difference across the
requested configurations was 7.9112e-16. DIGIT / PASS; no repeated passing
tests, additional precision, or unrequested benchmark iterations.
Combined delegated implementation/test review removed constructor-only wrappers
before the first run. Sparse-T custom functions, general non-foldable sparse
indices and full native Windows/CPU coverage remain open.

A read-only native readiness probe found stable-x86_64-pc-windows-gnu installed
under C:/Users/xylxp/.cargo/bin but not on this shell's PATH. No mpiexec was
found on PATH or the usual Microsoft MPI directories. No native numerical
build/test was attempted in this batch; this is not Windows acceptance.

## Sparse functions and sparse-amplitude MP3 (2026-09-07)

distributed_sparse_transform, distributed_dense_sparse and upstream_sparse_mp3_t
passed once each with 1/2/4 MPI ranks in the Linux work copy, including parity
contexts. Typed Pair/Mat wire formats, sparse structure preservation, explicit
zeros, missing input keys, diagonal restrictions, virtual redistribution and
dense-input identity filtering have exact integer checks. Matrix-valued scalar
multiplication verifies E12*E21=E11 rather than E22 through dense-by-sparse
matrix and high-order APIs, preserving noncommutative operand order.

The sparse-T MP3 path retains the source DPair transformation chain and energy
criterion abs((dense-sparse)/dense)<1e-6. Maximum observed relative difference
was 7.9112e-16. Combined source/implementation/test review corrected the initial
dense-zero callback expectation to the upstream sparsify behavior before the
first run; all tests passed without retries. DIGIT / PASS, no extra precision
or repeated passing checks. General custom function contractions, non-foldable
sparse indices, compressed symmetry, automatic planning and native Windows
acceptance remain unfinished.

## Sparse repeated indices (2026-09-07)

distributed_sparse_diagonal passed once at each of 1/2/4 MPI ranks in the
Linux work copy, with world and parity contexts. Exact i64 oracles cover all
four sparse/mixed high-order contraction APIs, repeated A/B/output labels,
alpha=2 and beta=3, virtual-2 output distribution, triple-index extraction,
replacement with absent/explicit-zero keys and empty local shards.
Off-diagonal input entries are ignored; off-diagonal output entries remain
unchanged, including under nonunit beta. The delegated fixture initially
scaled off-diagonal output by beta; this expectation was corrected during
integration review before the first run. No runtime failures or retries.
DIGIT / PASS; integer tolerance zero; no additional numerical checks.

## Distributed compressed symmetric I/O (2026-09-07)

distributed_symmetric_io passed once at each of 1/2/4 MPI ranks with world and
parity contexts. Exact i64 checks cover SY/AS/SH canonical normalization,
reversed-coordinate writes, AS permutation signs including three axes,
structural zeros, repeated requests, duplicate/equivalent writes, local
transforms, virtual-block redistribution and replicated storage round trips.
The 5x5 matrix on a 2x2 grid asserts six allocated slots per rank and the
specific valid offsets/keys on rank coordinates (1,0). All allocated holes
remain zero. Extent-one cases exercise empty canonical local shards.
No compilation/runtime failures or retries; DIGIT / PASS, tolerance zero.
These checks establish compressed storage/I/O, not symmetric contraction or
symmetry-changing repack. Native Windows acceptance remains pending.

## Compressed symmetric operations (2026-09-07)

distributed_symmetric_operations passed once at each of 1/2/4 MPI ranks,
including parity contexts. Exact i64 acceptance covers repeated-index scaling
and transforms, SY-to-AS-to-SH-to-SY group-preserving repack, zero diagonals
after repack, virtual blocks and empty valid local slices, equivalent-key
scaled writes with alpha=2/beta=3, beta applied once per touched canonical key,
untouched values, and zero allocation holes. Integration review corrected the
ordinary scale implementation to use source right multiplication before the
first run. No runtime failures or retries. DIGIT / PASS; tolerance zero.
Noncommutative scaling order is source-inspected here, not separately validated
by this integer fixture. NS-boundary changes and symmetric contractions remain
unfinished.

## Boundary repack and Windows build (2026-09-07)

distributed_symmetric_repack passed once at 1/2/4 WSL MPI ranks, world and
parity contexts. Exact i64 checks cover NS-to-SY/AS/SH canonical sampling,
canonical-only reverse copies, three-axis and partial groups, virtual blocks,
and replicated destination/source ownership. No numerical failures or retries.
DIGIT / PASS for this repack scope only.

Native Windows GNU compiled and linked all then-current Cargo test targets;
the newly added repack target also compiled and linked separately afterward.
Only native library names changed for Windows (OpenBLAS and ScaLAPACK).
The acceptance PowerShell script passed syntax parsing. Native execution is
not accepted: Microsoft MPI runtime installation was canceled (0x800704c7),
msmpi.dll was absent, and local_linalg exited before any test output. See
native-windows.md for exact setup and remaining runtime acceptance.

## Physical packed summation execution (2026-09-07)

distributed_packed_sum passed once at each of 1/2/4 WSL MPI ranks, world and
parity contexts. Exact i64 checks exercise canonical packed-to-NS overlap,
repeated output diagonals with untouched off-diagonal beta, trace reduction,
inclusive AS physical diagonal slots, explicit input broadcast/output reduction,
virtual-block reduction with beta once, and empty local extents. No failures or
retries. DIGIT / PASS for the raw packed execution layers, tolerance zero.
The same new target compiled and linked once on Windows GNU; native runtime
acceptance still awaits Microsoft MPI runtime installation. No installer retry
or repetition of passed WSL checks was performed.

## Raw packed contraction layers (2026-09-07)

distributed_packed_contraction passed once at each of 1/2/4 WSL MPI ranks,
including parity contexts. Exact i64 checks cover canonical i<=k<=j products,
repeated-index products, the source whole-buffer beta behavior, scalar dot
reductions, physical AS diagonal slots, virtual reduction with beta once,
root-only MPI Reduce results, clearing broadcast replicas and empty extents.
No failures or retries; DIGIT / PASS, tolerance zero. New target also compiled
and linked once on Windows GNU; runtime MPI acceptance is still pending.
This validates the explicit raw execution layers, not a complete high-level
distributed symmetric tensor contraction operation.

## Tensor-level canonical indexed sums (2026-09-07)

distributed_canonical_sum passed once at 1/2/4 WSL MPI ranks, world and parity
contexts. Exact i64 checks cover packed-to-NS canonical overlap, row and trace
reductions, repeated output diagonal updates preserving off-diagonal values,
output-label broadcasting into AS storage, transposed canonical-domain
intersection, replicated and virtual layouts, and empty input with beta-only
output. No failures or retries; DIGIT / PASS, tolerance zero. The new target
also compiled and linked once on Windows GNU; native execution remains pending.
No general symmetry-aware sum or mixed-symmetry cancellation is claimed.

## Hollow symmetry-aware summation (2026-09-07)

distributed_hollow_sum passed once at each of 1/2/4 WSL MPI ranks, including
parity contexts. Exact i64 checks cover AS/SH signed or unsigned expansion into
NS, NS-to-AS/SH projection, preserved-symmetry transposes, AS reduction
cancellation, SH factorial reduction, mixed AS/SH cancellation through recursive
unfolding, and three-axis input/output recursion. Replica and virtual layouts
are included. No failures or retries; DIGIT / PASS, tolerance zero.
Integration review corrected recursive coefficient ownership before the first
run: source recursion reruns alignment/factors on the incoming coefficient,
not the already-adjusted parent coefficient. The new target also compiled and
linked once on Windows GNU. SY, repeated-label hollow sums and native runtime
acceptance remain pending; the explicit method contract does not hide these gaps.

## Compressed diagonal extraction and hollow preprocessing (2026-09-07)

distributed_symmetric_diagonal passed once at 1/2/4 WSL MPI ranks, including
parity contexts. Exact i64 checks cover recursive NS iii extraction, deletion
of a physically mapped axis, virtual/replicated projected storage, preservation
of an unaffected AS group, SY ii extraction/reinsertion, AS/SH structural-zero
diagonals, and repeated iik-to-ii hollow summation with unchanged off-diagonal
output. No failures or retries; DIGIT / PASS, tolerance zero. The new target
also compiled and linked once on Windows GNU; native execution remains pending.
Cross-group symmetry-breaking diagonal patterns were not tested or claimed.

## f64 SY sums and raw-task alignment correction (2026-09-07)

distributed_sy_sum fixes atol=1e-6 per entry against analytic references,
requiring finite values. The first 1-rank run passed matrix assertions but
failed three-axis expansion with an actual 0 where 7 was required. Source
inspection identified the omitted mandatory sum_tensors index-alignment step;
no precision/backend sweep or tolerance adjustment was used.

After that code fix, distributed_sy_sum and the affected
distributed_canonical_sum passed at 1/2/4 WSL MPI ranks, world and parity
contexts. Only these affected suites were run. Coverage includes SY expansion
with diagonal counted once, NS-to-SY orbit projection including diagonal
multiplicity, beta handling, full reduction, transposition, mixed AS
cancellation, supported repeated diagonals, and a three-axis distinct-index
fixture. Both targets compiled and linked on Windows GNU. DIGIT / PASS;
verification closed. This does not establish arbitrary high-order coincidence
surfaces, other SY scalar types, or native MPI runtime acceptance.

## Generic SY scalar paths (2026-09-07)

distributed_sy_scalars passed once at each of 1/2/4 WSL MPI ranks, world and
parity contexts. Exactly representable fixtures verify f32, i32/i64, complex
f32/f64, and a heap-owned non-Copy custom ring through SY expansion and
reduction. Complex alpha includes a nonzero imaginary component. The custom
ring supplies its explicit CastFromF64 and Wire implementations.
No failures/retries; DIGIT / PASS. Existing passing f64 numerical fixtures were
not rerun. The new target compiled and linked once on Windows GNU; native MPI
runtime acceptance is still pending. These tests do not add a study of
higher-order fractional coincidence corrections.

## Tensor-connected packed contraction (2026-09-07)

distributed_canonical_contraction passed once at each of 1/2/4 WSL MPI ranks,
world and parity contexts. Exact i64 fixtures cover physical mapping on a
reduced index (root Reduce), mapping output indices (input broadcasts, including
a 2x2 grid), equalized symmetric virtual phases, nonuniform padded extents and
empty local slices, packed SY Hadamard/dot products, AS structural holes, and
restoration of the original output mapping. No failures/retries; DIGIT / PASS,
tolerance zero. The new target compiled and linked once on Windows GNU.
No symmetry overcount factors or full semantic symmetric contraction are
claimed by these canonical tests; native runtime acceptance is still pending.

## Explicit-map symmetry-aware contractions (2026-09-07)

distributed_symmetric_contraction passed once at 1/2/4 WSL MPI ranks, with
world/parity contexts. Fixed atol=1e-6 and finite-value checks compare full-domain
SY/AS/SH matrix products and dot products to analytic references, not merely the
canonical chamber. Fixtures cover SY diagonal prescaling versus factorial
overcounting, mixed SY/AS cancellation, preserved SY Hadamard symmetry, beta
handling and supported repeated diagonal output with off-diagonal preservation.
No numerical failures/retries; DIGIT / PASS, verification closed.
Integration review corrected prescale input selection to mapped packed local
size before the first run. The new target also compiled and linked once on
Windows GNU; native MPI runtime acceptance remains pending.

## General run_diag and upstream diagonal identity (2026-09-07)

distributed_cross_diagonal, upstream_diag_sym and the affected
distributed_symmetric_diagonal passed at 1/2/4 WSL MPI ranks with world/parity
contexts. The upstream test retains norm<1e-10; its rank-4 paired-SY diagonal
identity uses deterministic dyadic fixtures. Cross-AS extraction/reinsertion
uses atol=1e-6; the existing simpler diagonal suite retains exact i64 checks.

The initial cross-AS replacement expectation assumed ordinary assignment and
failed (-30 versus -20). Source inspection established the first-permutation
beta-only clearing rule; the oracle was corrected to pinned rw=0 behavior.
No production change or tolerance adjustment was made for that discrepancy.
Only the affected/new suites were then executed; all passed. DIGIT / PASS,
verification closed. All three targets also compiled and linked on Windows GNU.
Native runtime and complete upstream CPU coverage remain unaccepted.

## Four upstream identities through compressed tensor APIs (2026-09-07)

upstream_diag_ctr, upstream_reduce_bcast, upstream_multi_tsr_sym and
upstream_sy_times_ns passed once each at 1/2/4 WSL MPI ranks, world/parity
contexts. diag_ctr retains its nonzero initial trace and 1e-10 residual checks;
reduce_bcast retains norm<=1e-6; multi_tsr_sym retains norm<1e-6;
sy_times_ns retains norm<1e-10 for both literal-source and adapted nonzero cases.
The adapted SY-times-NS maximum observed norm was 2.2591401799415137e-16;
the NS/SY Gram differences were zero. No failures or additional precision runs.
DIGIT / PASS, verification closed. All four new targets compiled and linked
once on Windows GNU; execution still awaits native MPI runtime acceptance.

## Sparse TTTP and MTTKRP (2026-09-07)

distributed_sparse_multilinear passed once at 1/2/4 WSL MPI ranks, world and
parity subcommunicators. Fixed analytic stored-entry products/sums use finite
values and absolute error <=1e-6; the vector TTTP stored-key sequence is exact.
Coverage includes vector factors, both matrix auxiliary orientations, uneven
auxiliary blocks, virtual mapping, empty local sparse shards, output distribution
changes, and a two-billion-element logical tensor with only two stored entries.
DIGIT / PASS; no diagnostic computations or tighter precision runs. The new
target compiled and linked once on native Windows GNU. Windows execution remains
unaccepted because the Microsoft MPI runtime installation was canceled earlier.

## Sparse weighted Solve_Factor (2026-09-07)

distributed_sparse_solve_factor and the affected distributed_solve_factor target
passed once at 1/2/4 WSL MPI ranks, world/parity subcommunicators. The new fixture
forms RHS from an analytic known solution and its sparse weighted Gram matrix;
the existing dense acceptance bound is retained: abs(error)<=1e-8+1e-5*abs(ref),
with finite results and exact output distribution. All output modes, all physical
mode placements, virtual blocks, replicated weights, empty local sparse shards,
explicit stored zero and empty/singular normal systems are covered. Singular
systems return POSV INFO=1 on all ranks. No failures or extra precision runs.
DIGIT / PASS, verification closed. Both changed targets compiled and linked once
on Windows GNU; native runtime acceptance remains outstanding.

## TTTP factor broadcasts and source sparse MTTKRP kernel (2026-09-07)

Affected distributed_tttp and distributed_sparse_multilinear targets passed once
at 1/2/4 WSL MPI ranks, world/parity subcommunicators. Dense vector TTTP remains
exact, dense matrix TTTP retains global L1<=1e-5, and sparse operations retain
finite values with absolute error<=1e-6. Sparse MTTKRP now exercises every output
mode with both vector and matrix factors through the source fiber-grouped kernel.
Existing empty-shard, virtual-block and auxiliary-division checks remain active.
DIGIT / PASS, no diagnostic or tighter-precision runs. Both targets compiled and
linked once on Windows GNU; this does not constitute native runtime acceptance.

## Sparse input-only contraction reduction (2026-09-07)

distributed_sparse_input_reduction plus affected distributed_sparse_fold and
distributed_sparse_diagonal passed once at 1/2/4 WSL ranks, world/parity contexts.
The new exact-i64 test covers A[ixpk]*B[kqj]->C[ij] and repeated-index variants
A[ixpkk]*B[kqqj]->C[ijj], all four sparse/mixed combinations, nontrivial alpha/beta,
empty input-only reduction extent, replicas, virtual blocks, and 2x2 rank grid.
Off-diagonal output entries remain exact. DIGIT / PASS; no diagnostic runs.
All three targets compiled and linked once on Windows GNU. Native runtime and
the source general sparse C-only execution path remain unaccepted/incomplete.

## General sparse sequential and mapped contraction (2026-09-07)

Three sparse_sequential local tests passed: exact i64 B/C-only label traversal,
empty sparse/zero extent beta behavior, and noncommutative matrix-semiring scalar
operand order. An initial test compilation needed Arithmetic::<i64> specified;
no numerical criterion changed. distributed_sparse_general passed at 1/2/4 WSL
ranks on world/parity contexts, exact i64. It covers physical i/j/k mappings and
2x2 ij, variable sparse broadcasts, output reduction, C-only x, empty sparse
shards, and original distribution restoration. The distributed target was rerun
after staging changed to canonical-root-only transfer; local passing tests were
not rerun. Final distributed implementation also passed all required rank counts.
DIGIT / PASS. Both targets compiled/linked on Windows GNU, and the changed
distributed staging target was rebuilt; native runtime remains unaccepted.

## General sparse custom-function branch (2026-09-07)

sparse_function and affected sparse_sequential passed once locally (six tests).
The custom tests cover stored scalar zero, dense zeros, absent structure, exact
callback count and pinned all-scalar ordinary-multiplication behavior. The
distributed_sparse_function and affected distributed_sparse_general targets
passed once at 1/2/4 WSL ranks, world/parity, with exact i64 values. The supported
custom path uses scalar A, alpha=one, mapped B/C shared label and C-only output;
unsupported source custom branches are not presented as accepted capability.
DIGIT / PASS; no failures or extra numerical runs. All four targets compiled and
linked once on Windows GNU. Native runtime acceptance remains outstanding.

## Folded sparse custom-function GEMM (2026-09-07)

sparse_function_kernel passed its exact-i64 CSR sparse/dense and sparse/sparse
oracle. distributed_sparse_gemm_function and affected distributed_sparse_gemm
passed once at 1/2/4 WSL ranks, world/parity. Custom checks use f(a,b)=a+b+1,
unit alpha and beta=3, explicitly distinguishing sparse stored zeros from missing
entries and preserving dense zero evaluations. Empty sparse A, padded/empty local
panels, 2x2 grid at four ranks, and original output distribution are covered.
DIGIT / PASS; no failures or diagnostic runs. All three targets compiled and
linked once on Windows GNU; native runtime acceptance remains outstanding.

## Custom sparse-output CSR contraction (2026-09-07)

The new sparse_function_kernel sparse-output case passed once, covering multiple
numeric paths, structural zero retention and old CSR row merge. Previously passed
unchanged dense-output cases were filtered out. distributed_sparse_function_output
passed once at 1/2/4 WSL ranks, world/parity, with exact local keys and i64 values.
Beta=zero retains old-only zero coordinates per pinned sparse summation; beta=3
scales and merges old entries. Output distribution is unchanged. DIGIT / PASS,
no failures or extra numerical runs. Both targets compiled/linked on Windows GNU;
native runtime remains unaccepted.

## Upstream unary, endomorphism and bivariate-transform batch (2026-09-07)

upstream_univar_function, upstream_endomorphism and upstream_bivar_transform each
passed once at 1/2/4 WSL ranks, world/parity subcommunicators. All three source
identities retain strict abs(error)<1e-6; typed scalar broadcast and repeated-output
diagonal transform additionally match exact representable values, including empty
local shards. DIGIT / PASS; no failures or extra numerical runs. All three targets
compiled and linked once on Windows GNU. Native runtime acceptance remains open.

## Custom endomorphisms and structured measurements (2026-09-07)

upstream_endomorphism_cust and upstream_endomorphism_cust_sp passed once at 1/2/4
WSL ranks, world/parity, exact cached string lengths and unchanged sparse nnz.
Both tests and mpi_structured_bench compiled/linked on Windows GNU; native runtime
remains unaccepted. DIGIT / PASS. A benchmark compilation initially assumed Clone
for SymmetricTensor; constructing the second independent operand fixed it before
any benchmark execution. No numerical failures or diagnostic runs occurred.

The release example mpi_structured_bench ran each case once per rank count with
OPENBLAS_NUM_THREADS=1 and mpirun --oversubscribe. Both use 96x96 matrices. Sparse
inputs contain unit entries on modular masks (moduli 11 and 13); symmetric inputs
are compressed SY matrices of ones. Timings include the operation and final
barrier, exclude setup/checks. Three exact output probes check each run. GNU time
wraps every MPI worker; reported RSS is the maximum worker lifetime peak, including
MPI/library overhead and setup, not summed rank memory or operation-only allocation.

| Case | Ranks | Elapsed seconds | Maximum worker peak RSS KiB |
|---|---:|---:|---:|
| Sparse GEMM | 1 | 0.001417 | 26540 |
| Sparse GEMM | 2 | 0.000909 | 26552 |
| Sparse GEMM | 4 | 0.000729 | 26624 |
| SY x SY -> NS | 1 | 0.051151 | 27116 |
| SY x SY -> NS | 2 | 0.029630 | 27224 |
| SY x SY -> NS | 4 | 0.017829 | 27172 |

These single samples are representative records, not a statistical comparison
or a speedup claim. They do not close automatic planning/low-memory acceptance.

## GridPlan-derived execution cost tree (2026-09-07)

grid_plan_cost passed once in WSL using 1/2/4-process topology shapes, without MPI
execution. Analytic model coefficients isolate FLOPs and communicated bytes;
expected estimates 120/168/156, volumes 0/96/192 and zero source workspace match
exactly. These are synthetic coefficient checks, not measured seconds. The target
compiled and linked on Windows GNU. DIGIT / PASS; no repeated precision checks.
Automatic selection and total candidate costs remain incomplete.

## High-order sparse custom contraction (2026-09-07)

distributed_sparse_fold_function passed once at 1/2/4 WSL ranks, world/parity,
with exact i64 values. It covers three storage combinations through A[ikl] and
B[kjl] into permuted C[jil], plus repeated A[ikkl] and C[jiil]. Stored A/B zeros,
missing sparse values versus dense zeros, separate l batches, off-diagonal input
exclusion, output off-diagonal preservation and original distributions are
checked. DIGIT / PASS; no failures or further precision runs. The target compiled
and linked once on Windows GNU; native runtime acceptance remains outstanding.

## Distributed dense custom functions and upstream bivar_function (2026-09-07)

upstream_bivar_function and distributed_dense_function passed once at 1/2/4 WSL
ranks, world/parity. The upstream four-dimensional identity retains its strict
abs(error)<1e-6 rule with finite results. The exact i64 custom test checks a
non-distributive f(a,b)=a+b+1, physical i/j/k mappings, output-only axes, repeated
output diagonals and off-diagonal preservation. In particular k=3 at four ranks
must contribute no padded function evaluations. DIGIT / PASS; no failures or
additional numerical runs. Both targets compiled/linked once on Windows GNU;
native runtime remains unaccepted.
