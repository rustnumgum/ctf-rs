# Representative CPU/MPI measurement

2026-09-06, WSL Ubuntu-26.04, release profile, OpenBLAS threads=1.
One run per required rank count, no warmup or statistical speedup claim.
Operation: 128x192 times 192x160 f64 GEMM through `Tensor::gemm_2d`.
Elapsed interval includes distribution alignment/restoration and panel execution,
with barriers around the operation, but excludes process startup and input fill.
Peak RSS is GNU time's process-lifetime maximum for each **worker**, not mpirun.

| Ranks | Grid | Elapsed seconds | Maximum worker peak RSS (KiB) |
|---:|---|---:|---:|
| 1 | 1x1 | 0.018715 | 27736 |
| 2 | 2x1 | 0.010924 | 27084 |
| 4 | 2x2 | 0.006744 | 26848 |

These are observations for this small case, not a scaling guarantee or a
comparison against C++ CTF. RSS includes MPI/BLAS runtime overhead and is not
the tensor allocator's peak. Per-process peaks must not be summed and described
as a simultaneous total-memory peak.

Runner: `bash scripts/bench-wsl.sh`.
Raw local reports: `/home/xylxp/.cache/ctf-rs-bench/ranks-{1,2,4}/`.
This measurement was made before switching source reads to the Linux work copy;
Cargo build artifacts were already in the Linux filesystem.
