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
