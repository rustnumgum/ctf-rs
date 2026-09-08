# Pinned CPU scope and correspondence

Reference: cc4s/ctf, f69cbb46e23bc2f39cda5722ce096f56301dab4f.
Statuses below describe implementation, not acceptance passes.

Dense and packed `fill_random` implement all six source specializations:
f32/f64/complex32/complex64/i32/i64. They retain allocation-order draws and
post-fill padding cleanup; integers truncate after double-precision scaling.

| Phase | Upstream responsibility | Rust destination | Required validation | Status |
|---|---|---|---|---|
| 1 | interface/{set,monoid,group,semiring,ring}; tensor/algstrct | algebra | scalar, endomorphism_cust, discrete semirings | native set/monoid/semiring/ring traits and typed arithmetic implemented, including wrapping u32/u64 rings; direct Rust operations replace expression/operator compatibility classes |
| 1 | interface/world; interface/common MPI CommData | context; rsmpi; ffi/mpi | subworld_gemm, permute_multiworld, empty participants | host-owned rsmpi Universe via Context::world and Context::from_communicator; borrowed host communicators, checked Funneled/main-thread contract, world/parity splits and explicit split close; no Runtime or C++ World wrapper; R1 accepted after Tensor replica restoration, full WSL and native gates passed; evidence in validation.md |
| 1 | mapping/{topology,mapping,distribution,node_aware_dist} | mapping; node_aware; node_reordering | exact physical/virtual layout, node permutations | dense topology catalogs, map chains, physical/virtual assignment and node-aware rank-reordered execution implemented; optional machine-specific BG/Q discovery is excluded and sparse candidate extensions remain with S1 |
| 1 | tensor/untyped_tensor; redistribution/{redist,cyclic_reshuffle,glb_cyclic_reshuffle,dgtog_*,pad,sparse_rw} | tensor; redistribution | readwrite_test, readall_test, repack; uneven/empty local slices | optimized dense DGTOG ROR, block/global reshuffle and packed padding implemented; sparse read/write optimization remains pending |
| 2 | scaling/*; summation/*; redistribution/{slice,nosym_transp} | scaling; summation; redistribution | diag_sym, diag_ctr, scalar, permutations and slices | dense scale/strip, packed indexed scaling, shifted slice, optimized axis transpose, indexed endomorphisms and automatic dense/symmetric summation implemented; remaining optimized custom/sparse summation belongs to S1 |
| 2 | contraction/{contraction,ctr_tsr,ctr_comm,ctr_2d_general,sym_seq_ctr}; shared/blas_symbs | contraction; ffi/linalg | gemm_4D, weigh_4D, ccsdt_t3_to_t2, sy_times_ns; node-aware execution | dense replication, four-type folded panels, automatic compressed planning and four-type plain-transpose SYR implemented; AXPY/SCAL/COPY/DDOT use typed Rust operations and optional batch GEMM uses the existing typed loop |
| 3 | contraction/{contraction_signature,contraction_plan,contraction_selector}; shared/{model,init_models,memcontrol,int_timer} | planning; contraction; model; memcontrol; int_timer | cache reuse, candidate selection, distribution switching, low-memory path, timings/peak memory | dense candidate search/selection/execution, context-owned cache and models, OS/cgroup-aware budgets, low-memory execution and named timers implemented; the upstream selector allgather is an assertion stub and the process-global C++ model registry is replaced by explicit ownership; sparse automatic candidates remain with S1 |
| 4 | sparse_formats/*; contraction/sp*; summation/spr*; tensor sparse paths | sparse; sparse_sum; sparse_gemm; sparse_fold; sparse_sequential; sparse_contract_general; sparse_formats | speye, sptensor_sum, sparse_mp3, custom sparse endomorphisms | folded sparse/mixed products including native sparse-A/dense-B/CCSR-output with input-only pre-reduction and repeated indices; explicit-mapped sparse-A/dense-B/dense-C general kernel supports output-only labels; automatic sparse planning, nested moving-output communication and full upstream sparse suite pending; non-inner sparse-output source assertion documented in sparse-output.md |
| 4 | symmetry/*; symmetric sequential kernels | symmetry; symmetric_distribution; symmetric_tensor | NS/SY/AS/SH compressed layouts, mixed symmetry and repeated indices | distributed compressed I/O, redistribution, repack, general diagonal stack, symmetry-aware summation and automatic dense contraction/sum mapping implemented; compressed sparse symmetry remains pending |
| 4 | interface/{functions,fun_term}; transforms in tensor | algebra; tensor; dense_function; symmetric_contract; sparse_functions; sparse_function_kernel; sparse_fold_function | univar_function, bivar_function, bivar_transform, endomorphism* | local transforms, typed sparse maps, subset-index accumulators, distributed CSR custom products, fullyfoldable high-order sparse custom contractions, explicitly mapped general dense functions and packed symmetry custom-function CPU/MPI dispatch implemented; custom folded kernels, other kernels and full upstream function tests pending |
| 5 | interface/matrix; shared/lapack_symbs | matrix; ffi/linalg; ffi/scalapack | qr, svd, eigh, Cholesky, SPD/triangular solves, randomized SVD | f32/f64/complex32/complex64 distributed Cholesky/triangular solve/thin QR/SVD, truncated/randomized SVD, padded virtual-column SPD and square-subworld symmetric/Hermitian eigh implemented; the pinned dense QR/SVD/eigh drivers are closed, while optional undeployed routine families are not claimed |
| 5 | interface/multilinear | multilinear; multilinear_factor; sparse_multilinear; solve_factor; tensor_svd; reshape | TTTP, MTTKRP, Solve_Factor, tensor SVD | generic semiring dense/sparse TTTP, source fiber-grouped four-type MTTKRP, f64 distributed weighted Solve_Factor, four-type indexed tensor SVD and key-based reshape implemented; remaining work is sparse-path optimization rather than a dense acceptance gap |
| 5 | interface/{partition,vector,scalar,common}; schedule; tensor persistence/graph I/O; shared diagnostics | partition; vector; scalar; common; schedule; tensor; shared; sparse_text; ffi/mpi_io | FFT partitioning, scalar/value access, schedules, checkpoint, graph input, diagnostics | native indexed partitions, tensor-backed vector/scalar operations, common helpers, dense MPI-IO checkpointing, scheduling, timers and flop snapshots implemented; graph/sparse persistence remains outside the dense close |

All CPU test, example, benchmark and study files are inventoried separately;
the test suite includes examples and studies, not only `test/*.cxx`.
Python tests can specify numerical semantics even though Python APIs are excluded.
Interface expression/operator classes are replaced by direct Rust operations.
CUDA/offload code and Python/C++ API compatibility are excluded, not CPU paths.

### Remaining tensor-interface boundaries from the pinned implementation

- Dense-to-sparse conversion: tensor.cxx:895-910 and untyped_tensor.cxx:1647-1820;
  implemented as consuming Tensor::into_sparse(predicate), including primary
  layers and padding filter order. SparseTensor::into_dense is collective;
  SparseTensor::sparsify separately filters existing sparse storage.
- Sparse text I/O: tensor.cxx:945-1015 and graph_io_aux.cxx:45-248; four-type
  dense/sparse codecs and communicator-scoped MPI-IO are implemented, including
  source overlap, rank-prefix writes, reversed indices and no-value records.
  Compressed-symmetry export writes canonical nonzero pairs, as in the source.
- Generic dense cross-world accumulation: tensor.cxx:1019-1062; reusable
  add_to/from_subworld APIs now support explicit child distributions and arbitrary
  child-rank orientation, and are used by the square-subworld eigensolver.
  Optimized cyclic-reshuffle buffers and sparse/compressed variants remain open.
- Explicit all-pair/all-data extraction: tensor.cxx:275-426,823-835;
  indexed reads and local_pairs do not provide those collective operations.
- Sparse random fill: tensor.cxx:1623-1707; fill_random_sparse implements both
  storage branches and all seven pinned scalar families using explicit RNG state.
- Dense/sparse/compressed real norms and complex norm2 are implemented in norms.rs,
  including original-precision compressed squaring and the source NS manual path.
  Bool norm1/norm_infty and compressed Boolean algebra contracts remain open.

Solve_Factor is f64-only in the pinned working implementation: multilinear.cxx
uses double buffers and MPI_DOUBLE (935-965). Missing f32/complex versions are
not counted as unported CPU capabilities merely because its declaration is templated.

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
