# Pinned CPU scope and correspondence

Reference: cc4s/ctf, f69cbb46e23bc2f39cda5722ce096f56301dab4f.
Statuses below describe implementation, not acceptance passes.

| Phase | Upstream responsibility | Rust destination | Required validation | Status |
|---|---|---|---|---|
| 1 | interface/{set,monoid,group,semiring,ring}; tensor/algstrct | algebra | scalar, endomorphism_cust, discrete semirings | in progress |
| 1 | interface/world; interface/common MPI CommData | context; ffi/mpi | subworld_gemm, permute_multiworld, empty participants | in progress |
| 1 | mapping/{topology,mapping,distribution,node_aware_dist} | mapping | exact physical/virtual layout, node permutations | in progress |
| 1 | tensor/untyped_tensor; redistribution/{redist,cyclic_reshuffle,glb_cyclic_reshuffle,dgtog_*,pad,sparse_rw} | tensor; redistribution | readwrite_test, readall_test, repack; uneven/empty local slices | basic dense key I/O and distribution switching implemented; optimized upstream kernels pending |
| 2 | scaling/*; summation/*; redistribution/{slice,nosym_transp} | scaling; summation; redistribution | diag_sym, diag_ctr, scalar, permutations and slices | dense offset slice, axis transpose and indexed endomorphisms implemented; remaining paths pending |
| 2 | contraction/{contraction,ctr_tsr,ctr_comm,ctr_2d_general,sym_seq_ctr}; shared/blas_symbs | contraction; ffi/linalg | gemm_4D, weigh_4D, ccsdt_t3_to_t2, sy_times_ns; node-aware execution | pending |
| 3 | contraction/{contraction_signature,contraction_plan,contraction_selector}; shared/{model,init_models,memcontrol,int_timer} | planning; contraction; shared | cache reuse, candidate selection, distribution switching, low-memory path, timings/peak memory | explicit-grid reusable plans and context-owned cache integrated; automatic selector, costs, low-memory paths and diagnostic plan steps pending |
| 4 | sparse_formats/*; contraction/sp*; summation/spr*; tensor sparse paths | sparse; sparse_sum; sparse_gemm; sparse_fold; sparse_sequential; sparse_contract_general; sparse_formats | speye, sptensor_sum, sparse_mp3, custom sparse endomorphisms | folded sparse/mixed products with input-only pre-reduction and repeated indices; explicit-mapped sparse-A/dense-B/dense-C general kernel supports output-only labels; remaining sparse general combinations, automatic planning and full upstream sparse suite pending |
| 4 | symmetry/*; symmetric sequential kernels | symmetry; symmetric_distribution; symmetric_tensor | NS/SY/AS/SH compressed layouts, mixed symmetry and repeated indices | distributed compressed I/O, redistribution, repack, general diagonal stack, symmetry-aware summation and explicit-mapping contraction implemented; automatic planning and compressed sparse symmetry pending |
| 4 | interface/{functions,fun_term}; transforms in tensor | algebra; tensor; sparse_functions; sparse_function_kernel; sparse_fold_function | univar_function, bivar_function, bivar_transform, endomorphism* | local transforms, typed sparse maps, subset-index accumulators, distributed CSR custom products and fullyfoldable high-order sparse custom contractions implemented; general dense function dispatch, other kernels and full upstream function tests pending |
| 5 | interface/matrix; shared/lapack_symbs | matrix; ffi/linalg; ffi/scalapack | qr, svd, eigh, Cholesky, SPD/triangular solves, randomized SVD | f64 distributed Cholesky/triangular solve/thin QR/SVD, truncated/randomized SVD, square-subworld eigh and padded virtual-column SPD implemented; other scalars and remaining routines pending |
| 5 | interface/multilinear | multilinear; multilinear_factor; sparse_multilinear; solve_factor; tensor_svd; reshape | TTTP, MTTKRP, Solve_Factor, tensor SVD | dense/sparse f64 TTTP with mode-fiber factor broadcast, source fiber-grouped MTTKRP, distributed weighted Solve_Factor, indexed tensor SVD and key-based reshape implemented; optimized reshape communication, additional scalar kernels and automatic memory selection pending |
| 5 | interface/schedule; tensor persistence/graph I/O; shared diagnostics | schedule; tensor; shared | schedules, checkpoint, graph input, diagnostics | pending |

All CPU test, example, benchmark and study files are inventoried separately;
the test suite includes examples and studies, not only `test/*.cxx`.
Python tests can specify numerical semantics even though Python APIs are excluded.
Interface expression/operator classes are replaced by direct Rust operations.
CUDA/offload code and Python/C++ API compatibility are excluded, not CPU paths.

## Upstream incomplete declarations

`ContractionSelector::allgather()` asserts unconditionally in the pinned source.
`interface/decomposition.h` contains an unfinished HoSVD class; distinguish these
from implemented tensor SVD and the working `examples/hosvd.cxx` algorithm.
Do not port upstream stubs as Rust placeholders, or count declarations as working
upstream capability. Working CPU algorithms still belong to the scope.

## Decomposition acceptance quantities

From `scalapack_tests/qr.cxx`: orthogonality norm <= m*n*1e-6;
reconstruction norm <= m*n*n*1e-6.
From `scalapack_tests/svd.cxx`: both orthogonality norms <= m*n*1e-6;
reconstruction norm <= m*n*n*1e-6.
From `scalapack_tests/eigh.cxx`: orthogonality and reconstruction norms <= n*n*1e-6.
These are Frobenius norms (`norm2`), not entrywise vector comparisons.

## Current evidence

See `validation.md`: basic exact foundation checks pass at 1/2/4 MPI ranks;
local f64 BLAS GEMM and LAPACK QR/SVD/eigh checks pass. This does not close any
whole phase. `upstream-inventory.tsv` remains a per-file backlog, not a pass list.
