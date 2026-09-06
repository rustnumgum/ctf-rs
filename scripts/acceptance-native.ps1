param(
    [string]$MingwRoot = 'C:\msys64\mingw64',
    [string]$TargetDir = 'D:\ctf-rs-native-target',
    [switch]$BuildOnly
)
$ErrorActionPreference = 'Stop'
$env:PATH = "$HOME\.cargo\bin;$MingwRoot\bin;C:\Program Files\Microsoft MPI\Bin;" + $env:PATH
$env:MSMPI_INC = "$MingwRoot\include"
$env:MSMPI_LIB64 = "$MingwRoot\lib"
$env:LIBCLANG_PATH = "$MingwRoot\bin"
$env:RUSTFLAGS = "-L native=$($MingwRoot.Replace('\','/'))/lib"
$env:CARGO_TARGET_DIR = $TargetDir
$env:OPENBLAS_NUM_THREADS = '1'
if ($BuildOnly) {
    cargo test --tests --no-run
    exit $LASTEXITCODE
}

$mpiTests = @(
    'foundation','dense_views','replicated_sum','tensor_sum','custom_reduce',
    'algebra_sum','complex_scalar','sum_remap','replicated_contraction','ctr_2d',
    'tensor_gemm','algebra_contraction','tensor_contract','contract_remap',
    'dense_semantics','upstream_dense','subcomm_dense','plan_cache','model_training',
    'selector','selection_objective','tensor_blas_fold','upstream_gemm4d',
    'distributed_matrix','distributed_qr_svd','distributed_svd_paths','distributed_eigh',
    'distributed_spd','distributed_tttp','distributed_mttkrp','distributed_tensor_svd',
    'distributed_solve_factor','distributed_sparse_io','distributed_sparse_sum',
    'distributed_sparse_gemm','distributed_sparse_fold','upstream_sparse_mp3',
    'distributed_sparse_transform','distributed_dense_sparse','upstream_sparse_mp3_t',
    'distributed_sparse_diagonal','distributed_symmetric_io','distributed_symmetric_operations',
    'distributed_symmetric_repack','distributed_packed_sum','distributed_packed_contraction',
    'distributed_canonical_sum','distributed_hollow_sum','distributed_symmetric_diagonal'
)
$arguments = @()
foreach ($test in $mpiTests) { $arguments += @('--test', $test) }
foreach ($ranks in 1,2,4) {
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER = "mpiexec -n $ranks"
    cargo test @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Remove-Item Env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER
$localTests = @('local_linalg','topology_candidates','node_aware','map_tensor',
    'sequential_sum','virtual_sum','sequential_contraction','folded_contraction',
    'sparse_formats','cost_models','plan_cost','symmetry_layout','sym_indices',
    'sym_triple','sym_operations','folding')
$arguments = @()
foreach ($test in $localTests) { $arguments += @('--test', $test) }
cargo test @arguments -- --nocapture
exit $LASTEXITCODE
