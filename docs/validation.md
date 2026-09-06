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
