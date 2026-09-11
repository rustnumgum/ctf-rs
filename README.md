# ctf: cc4s CTF Rust port

Pinned reference: https://gitlab.cc4s.org/cc4s/ctf at
`f69cbb46e23bc2f39cda5722ce096f56301dab4f`.

This is an independent, **in-progress** CPU/MPI port, not yet a replacement
for the full reference. No muffin-tin integration, C++ CTF linkage, Python/C++
compatibility layer, or CUDA implementation is included.

The implementation uses a single `ctf` crate organized by upstream responsibilities.
Native calls belong in internal FFI modules. BLAS/LAPACK provide local numerical
kernels initially; their internal Rust interface must remain replaceable by faer
(pending confirmation of the requested name “fear”). ScaLAPACK remains the
distributed decomposition implementation, independently of local kernel choice.
Do not introduce a plugin framework or replace distributed algorithms with gather.

The C++ expression-template layer's automatic ordering of multi-term
contraction chains (`Sum_Term::estimate_time`/`execute`, `term.cxx:426,486-518`)
has no Rust equivalent: `interface/{term,idx_tensor,fun_term}` are replaced by
direct operations, not ported, so a chain written as successive
`contract_from`/`sum_from` calls executes in the order the caller writes it.
This is a scope limit, not a pending port item; see `docs/coverage.md`.

## Validation contract

Class R: preserve the pinned CPU algorithms, layouts and communication steps.
Integer/index/layout/discrete-algebra checks are exact. Floating-point checks
must preserve each upstream test's metric and tolerance. Decompositions compare
orthogonality/residual/reconstruction, not individual eigenvectors.

Run the completed acceptance set once at 1, 2, 4 MPI ranks in WSL Ubuntu-26.04,
with the Cargo target directory on the Linux filesystem, then Windows native.
Close passing checks immediately. At most three extra diagnostic computations
for numerical failures, with affected checks rerun only after actual fixes.
Compilation failures do not justify numerical tolerance changes.

Completion requires all five phases and the full source/test mapping, not just
the currently implemented foundations. See `docs/coverage.md` and
`docs/upstream-inventory.tsv` for scope and progress.

## Running the current subset

### MPI ownership

The host owns an rsmpi `Universe`, initializes with at least
`mpi::Threading::Funneled`, and checks the provided thread level. Construct
`Context::world(&universe)` or
`Context::from_communicator(&universe, &communicator)` on the MPI main thread;
ctf checks both requirements and never initializes or finalizes MPI.
Contexts borrow the universe and any host communicator. Drop tensors first,
explicitly `close()` split contexts, then release host communicators and the
universe. Only ctf-owned splits are freed by `close()`; dropping an unclosed
split leaks its handle rather than performing a collective on Drop.
The rsmpi dependency has default features disabled; custom reductions do not
use libffi.

### Platforms

`scripts/acceptance-wsl.sh` is host-portable: it runs on any Unix host with
`mpirun` on `PATH`, including WSL Ubuntu-26.04 and macOS.

- WSL Ubuntu-26.04: install the Rust toolchain, MPI development files,
  `libclang-dev`, BLAS/LAPACK and ScaLAPACK development libraries, and use
  the Linux filesystem for Cargo's target directory.
- macOS: `brew install open-mpi openblas scalapack`. `build.rs` adds the
  Homebrew prefix (`HOMEBREW_PREFIX`, else `brew --prefix`, else
  `/opt/homebrew`) to the link search path; ScaLAPACK/BLAS link as
  `scalapack`/`openblas`, the same names native Windows uses (Linux keeps
  `scalapack-openmpi`/`blas`/`lapack`).
- Native Windows: `scripts/acceptance-native.ps1 -MsMpiBin <path>`; see
  `docs/native-windows.md`.

Both scripts read one manifest, `scripts/acceptance-targets.tsv` (columns
target, class mpi/local/lib/excluded, ranks, sets, note); `scripts/check-targets.sh`
fails if a Cargo test target is not triaged in it. Each target runs once per
rank count, printing `RUN_EXIT <target> ranks=<n> exit=<code>` and a
per-target log under `$CTF_ACCEPTANCE_LOG_DIR` (default
`$HOME/.cache/ctf-rs-acceptance/<timestamp>/`); a failing target does not
stop the run, and the script exits nonzero at the end if any target failed.
`CTF_ACCEPTANCE_ONLY` and `CTF_ACCEPTANCE_RANKS` restrict a run to a subset
of targets or ranks.

The manifest now includes the thirteen S1 targets, `dgtog_redistribution`
and `model_io`; `upstream_bench_contraction` (informational) and
`upstream_model_trainer` (needs `-write`) are listed as deliberate
exclusions. The S1 close ran those thirteen targets at WSL and native 1/2/4
under the fixed plan.v4 contract; every target passed at every rank and no
HANDOFF question remains (`docs/validation.md`, section "S1d and S1 close").
This record is not permission to rerun a closed numerical check; the
acceptance scripts now cover the same targets going forward.

To avoid Windows-drive source I/O, run `bash scripts/sync-wsl.sh` from the
Windows-backed repository in WSL. Build/test in `/home/xylxp/ctf-rs-work` after
syncing; keep edits and Git commits in `D:\projects\ctf-rs`. The sync does not
copy `.git`, remove files, or change the delivery repository. Passing acceptance
sets need not be repeated merely because the work copy location changed.

## Commit cadence

Commit each completed implementation/validation batch with its scope and evidence.
Batch commits do not imply that an entire numbered phase or the full port is done.
Keep native build artifacts outside the source tree; use repository-local excludes
for machine-specific state. Do not push without a separate request.
