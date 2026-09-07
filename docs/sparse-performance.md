# Representative sparse contraction measurement

One measurement per rank configuration, 2026-09-08, WSL Ubuntu-26.04.
`examples/sparse_gemm_bench.rs` uses the optimized Cargo release build and
OPENBLAS_NUM_THREADS=1. No warmup, repeats, size sweep or speedup estimate.

The fixed f64 operation is sparse A[127,83] * B[83,97] -> sparse C[127,97],
alpha=1, beta=0. A stores global keys divisible by 5, B by 7, with deterministic
positive values. All runs produced 12319 stored output entries. These outputs
are structurally dense; this is not a sparse-output-density scaling claim.

| MPI ranks | Grid | Timed contraction (s) | Maximum rank peak RSS (KiB) |
|---:|---:|---:|---:|
| 1 | 1x1 | 0.003011 | 26716 |
| 2 | 2x1 | 0.002549 | 26824 |
| 4 | 2x2 | 0.001364 | 27080 |

Time is rank-zero wall time bracketed by barriers and includes the complete
gemm_sparse call: redistribution, layout conversion, contraction and output
distribution restoration. Input construction is excluded. `/usr/bin/time`
wraps **each MPI rank**, not just mpirun. RSS is a process-lifetime high-water
mark including MPI/library/input overhead, not incremental tensor allocation
or a sum of concurrent rank peaks. Per-rank RSS was 26824/26808 at two ranks
and 26564/27080/26548/26356 at four ranks (rank order).

Reproduction after the Linux work copy is synchronized:

```sh
export CARGO_TARGET_DIR=$HOME/.cache/ctf-rs-target OPENBLAS_NUM_THREADS=1
cargo build --release --example sparse_gemm_bench
mpirun --oversubscribe --tag-output -n 4 /usr/bin/time -f peak_rss_kib=%M \
  "$CARGO_TARGET_DIR/release/examples/sparse_gemm_bench"
```

The first launcher attempt lost a shell loop variable before starting any
benchmark process. Passing rank counts directly from PowerShell corrected
the invocation; each listed configuration then ran once. These small single
samples characterize this run only, not a performance guarantee or comparison
against C++ CTF. Windows compilation/linking succeeded, but runtime is still
unaccepted because the native MPI DLL is missing.
