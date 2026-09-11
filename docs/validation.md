# Validation evidence

## Index

One row per acceptance driver or gate family, naming by exact heading text
the section that carries its current, authoritative result. Later sections
supersede earlier ones for the same driver: the S1d/S1 close row below
supersedes the S1a/S1b/S1c outcomes for the same thirteen S1 targets, and
each `Dn`/R1/C1 "close" section is the final record for its scope. Nothing
in this document is edited, moved, or removed to build this index; every
row below only points into the existing chronological record.

| Driver / gate family | Authoritative section | Status |
|---|---|---|
| C1 (model I/O, contraction-path symmetrization, dense custom folded kernels, selector agreement, graph I/O native runtime) | "C1 acceptance (2026-09-11)" | CLOSED, DIGIT / PASS |
| R1 (rsmpi Universe/Context ownership, splits, parity worlds) | "Initial R1 acceptance" | CLOSED, DIGIT / PASS |
| D1 dense scaling and strip | "D1 dense scaling and strip close (2026-09-08)" | CLOSED |
| D2 optimized dense redistribution | "D2 optimized dense redistribution close (2026-09-08)" | CLOSED |
| D3 dense contraction | "D3 dense contraction close (2026-09-08)" | CLOSED |
| D4 shared infrastructure | "D4 shared infrastructure close (2026-09-08)" | CLOSED |
| D5 native interface and FFT | "D5 native interface and FFT close (2026-09-08)" | CLOSED |
| D6 dense drivers and native runtime | "D6 dense drivers and native runtime close (2026-09-08)" | CLOSED |
| S1a sparse contraction planning (historical) | "S1a sparse contraction planning" | superseded, see "S1d and S1 close" |
| S1b compressed-symmetry and custom sparse kernels (historical) | "S1b compressed-symmetry and custom sparse kernels" | superseded, see "S1d and S1 close" |
| S1c sparse summation, communication, persistence (historical) | "S1c sparse summation, communication, persistence" | superseded, see "S1d and S1 close" |
| S1d and S1 close (final contract for all thirteen S1 targets) | "S1d and S1 close" | CLOSED, DIGIT / PASS at WSL 1/2/4; three targets also native 1/2/4 |
| A1 audit remediation (plan.v5: scripts, dead code, panel executor merge, SAFETY comments, macOS build, documents) | "A1 audit remediation" | CLOSED on macOS, DIGIT / PASS 622/622 with every baseline quantity reproduced; WSL and native reruns pending on MSI |
| upstream_apsp | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| upstream_algebraic_multigrid | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| upstream_block_sparse | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| upstream_force_integration_sparse | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| upstream_btwn_central | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| upstream_checkpoint_sparse | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 and native 1/2/4 |
| upstream_mis | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| upstream_mis2 | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| sparse_einsum_hadamard | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 and native 1/2/4 |
| sparse_scaled_expression | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 and native 1/2/4 |
| sparse_complex | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| sparse_sy | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| sparse_sample | "S1d and S1 close" > "Final-tree target outcomes" | DIGIT / PASS, WSL 1/2/4 |
| Checkpoint n=3 fixture (historical, not a source failure) | "Checkpoint n=3 history (not a source failure)" | superseded by the n=7 default fixture under "S1d and S1 close" |
| All other dated per-feature milestones (2026-09-06 through 2026-09-08, roughly 140 sections interleaved with the batches above) | not individually indexed here; each is unedited chronological history that the C1/R1/D1-D6/S1 close sections above summarize | see `docs/coverage.md` for the current phase-level status of that work |

## A1 audit remediation

### Fixed plan.v5 contract (2026-09-12)

Accepted immutable harness `plans/ctf-rs/plan.v5.md`, gate `G-CTF-A1`,
class R against pinned cc4s/ctf `f69cbb46e23bc2f39cda5722ce096f56301dab4f`:
every gating driver keeps its own quantity, reference, and bound, and a
driver whose printed result changes is a failure of the refactor. The
audit that fixed the list is harness `evidence/2026-09-12-ctf-rs-audit/findings.md`
(evt-0073). Executor: Claude with five subagents on the Mac, each in its
own git worktree on a disjoint file set, reviewed and merged in order; the
plan names the macOS run as the regression and leaves the WSL and native
runs on MSI for a later request.

| Item | Commits | Audit IDs |
|---|---|---|
| macOS link names and Homebrew search path (`build.rs`) | b47cec8 | A5 |
| Hygiene: `DIGIT / ` prefix, rustdoc links, dead stores, `.gitignore`, `rust-version = "1.89"`, five examples declared, dead `symmetric_reshuffle::plan` and its tests removed | 5ab3a75 to 40f818b | A1, A3, A4, E3 |
| `#[must_use]` on `Context::split`/`split_shared`, unused imports, `// SAFETY:` at all 94 unsafe sites, `redistribute_ror` decision recorded, macOS `memcontrol` branch | 155687f, 9214e4c, e9b6bba, e34146e | D2, D3, D4, A1 |
| Acceptance manifest `scripts/acceptance-targets.tsv`, `scripts/check-targets.sh`, both scripts rewritten to record every target once per rank and continue past failures | 8bbd97b and its five predecessors | E1, E2, E4, E5 |
| Untested sparse-dense-sparse executors and `Pattern::SparseDenseSparse` removed; panel executors shared (one sparse-output and one dense-output generic, the custom executor on the shared helpers); Hadamard-index elimination factored; label metadata built once | bc3a264 to c92915b | C1, C2, C3, C4 |
| Documents to the S1-close state, inventory corrections, README platforms section, expression-chain ordering scope limit, this file's index | 0efac66 to 562aaba | B, F |
| `model_io` defaults to a temporary output directory so the scripts can run it | 7ffd5f2 | E1 |

### Regression on macOS (2026-09-12)

Host: macOS 15.7.3, rustc 1.97.1, Homebrew Open MPI 5.0.10, OpenBLAS,
ScaLAPACK; `OPENBLAS_NUM_THREADS=1`, `mpirun --oversubscribe`. Two runs
of `bash scripts/acceptance-wsl.sh` (the manifest-driven script, every
`mpi` target once at 1, 2, 4 ranks, `upstream_strassen` also at 7, the
five `lib` filters and 46 `local` targets once):

| Run | Revision | Invocations | Result |
|---|---|---|---|
| baseline, before the sparse refactor | e9b6bba | 623 | 617 pass; the six failures are `d4_memcontrol` (no macOS branch yet, `UnsupportedPlatform`) and `model_io` (required an argument) at 1, 2, 4 |
| final | 7ffd5f2 | 622 | **DIGIT / PASS 622/622**; the `symmetric_reshuffle` lib row is gone with its tests, hence one invocation fewer |

Comparison: the 577 `DIGIT` lines of the baseline appear identically in
the final run after stripping wall-clock fields; the final run adds the
six lines of the two newly passing targets. No driver quantity moved.
Named digits, identical to the WSL and native values of "S1d and S1 close":
`upstream_checkpoint_sparse` Q = 1.659570583342333e-6, 1.730313117560421e-6,
1.6343449060870917e-6 at 1, 2, 4; `upstream_algebraic_multigrid` rnorm =
0.004937970528833717, 0.005085930671471991, 0.005189487439309569.
`cargo check --all-targets` and `RUSTDOCFLAGS="-D warnings" cargo doc
--no-deps` are clean; `scripts/check-targets.sh` accounts for all 238 test
targets and 5 lib filters. `scripts/acceptance-native.ps1` parses under
Windows PowerShell on MSI and reads the manifest there (243 rows; sets d6
50, c1 2) but has not been executed; the WSL and native runs of the plan
are pending on MSI. Logs: harness `evidence/2026-09-12-ctf-rs-audit/`
(`baseline-run.log`, `final-run.log`, the normalized `DIGIT` lists).

Not done under this plan, by name: the clippy classes outside the
argument-count allows (238 warnings, style), and a single generic panel
executor for the sparse-output and dense-output cases, kept as two because
their moving-output combination differs in when beta is applied.

## C1 close

### C1.2 contraction-path symmetrization correspondence

Pinned `CTF_int::desymmetrize` (`symmetry/symmetrization.cxx:12-238`)
is implemented by `SymmetricTensor::desymmetrized` in
`src/symmetric_sy_sum.rs`. Pinned `CTF_int::symmetrize`
(`symmetry/symmetrization.cxx:240-399`) is implemented by
`SymmetricTensor::symmetrize_from` in the same Rust module. Both are called
by the broken-symmetry contraction orchestration in `src/symmetric_contract.rs`:
inputs use `desymmetrized(..., false)`, output uses
`desymmetrized(..., true)`, and final output uses `symmetrize_from`.
This preserves SY diagonal/coincidence factors, AS signs, a zero relaxed
output, and the source's active accumulation branch rather than its disabled
alternative.

The existing `upstream_gemm4d` driver exercises both functions through its
NS/SY/AS/SH branches at WSL 1, 2, and 4 ranks. The C1 full script also includes
`upstream_sy_times_ns`, `upstream_multi_tsr_sym`, `upstream_diag_sym`, and
`upstream_weigh4d`. These are the unchanged C1.2 acceptance drivers; the
implementation already exists, so C1.2 closes by correspondence documentation.
The C1 acceptance result is recorded below only after execution.

### C1.4 working selector agreement

The unconditional `ContractionSelector::allgather` stub is not the working
source protocol. `contraction::evaluate_mappings` uses rank-local cost gathers,
winner-rank broadcast and topology/time/memory broadcasts in its normal and
exhaustive branches. `dense_search::select_global` implements that protocol:
its source mapping ID reconstructs the winning topology and distributions.
`Selector::select_best` likewise broadcasts the complete winning plan payload.
The existing `dense_search`, `selector`, and `selection_objective` drivers
exercise this agreement at WSL 1, 2, and 4 ranks in the full C1 script.

C1.4 adds the missing valid-mapping count in `dense_search::exhaustive_pass`:
count after mapping preflight and before memory/cost filtering, then perform
an i64 Allreduce before winner selection. A globally empty enumeration returns
no refinement. This implements the quantity behind the source DEBUG count
without introducing a logging subsystem or the source's undeclared variable.
The prescribed `upstream_gemm4d`, `upstream_ccsdt_map`, `upstream_subworld_gemm`,
and NS `upstream_permute_multiworld` remain unchanged in the full C1 run;
explicit-mapping drivers are not mislabeled as standalone selector tests.

### C1 acceptance (2026-09-11)

Implementation series: `9e84575`, `8da79b1`, `f53220a`, `7155249`,
`cbc8fb1`, `0968f16`, in C1.1 through C1.6 order. The C1.1 coefficient
bound was committed before the first numerical run. C1.3 implements custom
CPU folded GEMM (alpha one, one batch, including transposed output), ordinary
dense/packed folded summation and mapped non-inner custom summation.
C1.5 selects only the two prescribed native I/O drivers. C1.6 corrects the
README and propagates sparse `.cxx` status to all eleven matching headers.

```text
DIGIT / PASS — model_io
Q: every registered model coefficient; class: A
ref: in-memory coefficients before write, tests/model_io.rs
bound: 0.5*10^(floor(log10(abs(ref)))-4), zero exact; source %1.4E
Delta: max normalized difference d=0.9940440900019784 at each rank count
runs: WSL 1/2/4 once each; no diagnostics; closed

INFO — upstream_model_trainer
existing dense workload, time=5, iterations=5, time_jump=1.5
one four-rank invocation with explicit write and load; seconds=24.90146141
informational only; not a speedup or model-training accuracy gate

DIGIT / PASS — full WSL acceptance
Q: each driver's unchanged metric and required invariants; class: R
ref: pinned f69cbb46 and the existing Rust driver assertions
bound: each driver's existing exact/absolute/relative/norm rule, unchanged
Delta: all assertions passed; emitted per-driver values/stamps are in wsl.log
runs: acceptance-wsl.sh once, all 175 MPI drivers at 1/2/4 once each,
      then its prescribed local/library checks and seven-rank Strassen run
exit: 0; diagnostics: 0; numerical verification closed

DIGIT / PASS — native compile/link
Q: all test/example targets compile and link; ref: C1 native gate
bound: successful exit; Delta: no compile/link failure
runs: acceptance-native.ps1 -BuildOnly once; exit 0

DIGIT / PASS — sparse_text_io (native)
Q: four-type coordinate I/O, duplicates, reverse/no-value, empty/tiny files
ref: unchanged driver assertions; bound: exact; Delta: 0
runs: native 1/2/4 once each, world+parity; exit 0; closed

DIGIT / PASS — distributed_sparse_io (native)
Q: sparse indexed I/O, redistribution, views, custom identity/order
ref: unchanged driver assertions; bound: exact; Delta: 0
runs: native 1/2/4 once each, world+parity; exit 0; closed
```

Commands from `D:/projects/ctf-rs`:

```powershell
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-s1/C1/model.sh
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-s1/C1/acceptance.sh
$env:CARGO_BUILD_JOBS='2'
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -BuildOnly > D:\projects\runs\ctf-rs-s1\C1\native-build.log 2>&1"
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -C1Only > D:\projects\runs\ctf-rs-s1\C1\native-runtime.log 2>&1"
```

Logs: `D:/projects/runs/ctf-rs-s1/C1/{model,wsl,native-build,native-runtime}.log`;
adjacent `commands.md`, `model.sh`, and `acceptance.sh` retain nested commands.
The existing keepalive was reused. No tolerance, existing driver, fixture,
or assertion changed; no numerical failure, diagnostic, or rerun occurred.
One accidental broad formatter invocation was removed outside the owned C1
files before acceptance. C1 is closed; S1a follows. ctf-rs is not pushed.

## S1d and S1 close

### Fixed plan.v4 contract (2026-09-12)

Accepted immutable harness `plans/ctf-rs/plan.v4.md`, class R, reference
cc4s/ctf `f69cbb46e23bc2f39cda5722ce096f56301dab4f`. Each of the thirteen
targets runs once at WSL 1/2/4 on the final code tree. That same set supplies
S1d, CK7, AMG and the other ten targets' regression evidence. The unchanged
full `scripts/acceptance-wsl.sh` runs once; native `-BuildOnly` runs once,
then the two S1d drivers and CK7 run once at native 1/2/4. At most the three
named plan.v4 diagnostics per failing driver; a pass is closed.

| Gate | Fixture / work | Q and reference | Fixed bound |
|---|---|---|---|
| G-CTF-S1d | n=11 density .1 `ijk,jkl->ijkl`, sparse/sparse and sparse/dense; n=5 density .1 `ijl,kjl->ijk` in the unchanged scaled expression | sum(abs(diff)) against the same dense Rust expression | <1e-14, source Python allclose |
| G-CTF-S1-CK7 | source default n=7, density .1, rank-seeded fill, six-decimal text round trip | norm2(v-u) against original values | <1e-7*n*n*.1*n = 3.43e-6 |
| G-CTF-S1-AMG | n=4, nlvl=2, ndiv=2, nsmooth=3, repaired layout | V-cycle rnorm against twice-smoothed Jacobi rnorm_alt | rnorm < rnorm_alt |
| G-CTF-S1 | all thirteen targets below, unchanged fixtures and driver criteria | each driver's source quantity/reference; full WSL script without failure | every target PASS at WSL 1/2/4 |

The earlier S1a/S1c description of sparse ABC expressions as a static source
restriction is withdrawn: `contraction.cxx:5417-5527` eliminates Hadamard
indices before folding. Missing that entry step was a Rust port gap, not a
source contradiction. Historical outcome tables below retain their original
run evidence, superseded by this section's final-tree results.

### Final-tree target outcomes

Code revision `d5861de92977809ee0a09a41ba6c5dbe1de904ed`, following CK7
`5a5edaaff004c0148135c8dbe8569988505ecfdf`. Every row below carries
**DIGIT / PASS** at WSL 1/2/4. Each target/rank ran once (39 invocations),
including its unchanged world and parity checks. All nine prescribed native
invocations also passed once. Native compile/link succeeded once. No numerical
diagnostics, changed criteria, repeated checks or additional studies occurred.

| Target | Stamp | Q / reference / bound | Delta or result at WSL ranks 1; 2; 4 |
|---|---|---|---|
| upstream_apsp | DIGIT / PASS | differing path weights / dense tropical result / exact 0 | 0; 0; 0 |
| upstream_algebraic_multigrid | DIGIT / PASS | V-cycle residual / twice-smoothed Jacobi / Q < ref | Q-ref = -0.000609640654155111; -0.000751315910721963; -0.000689413644143350 |
| upstream_block_sparse | DIGIT / PASS | residual norm / source flattened product / <=1e-4 | 0; 0; 0 |
| upstream_force_integration_sparse | DIGIT / PASS | source Boolean displacement/restore criterion / original particles / any initial displacement >1e-6 and all restored within1e-6 | criterion true at all ranks; numerical maxima not printed |
| upstream_btwn_central | DIGIT / PASS | norm2 difference / dense naive result / <=6e-6 | 0; 0; 0 |
| upstream_checkpoint_sparse | DIGIT / PASS | norm2(v-u) / original pre-round-trip values / <3.43e-6 | 1.659570583342333e-6; 1.730313117560421e-6; 1.6343449060870917e-6 |
| upstream_mis | DIGIT / PASS | overlap and uncovered count / source SH graph / both exactly 0 | both zero by passing assertions; counts not separately printed |
| upstream_mis2 | DIGIT / PASS | stored-entry counts >1.1 and <.9 / source sparse checker / both exactly 0 | (0,0); (0,0); (0,0) |
| sparse_einsum_hadamard | DIGIT / PASS | sum(abs(diff)) / same dense expressions / <1e-14 | maximum printed over d1/d2 and world/parity: 0; 0; 0 |
| sparse_scaled_expression | DIGIT / PASS | sum(abs(diff)) / grouped dense source expression / <1e-14 | maximum printed over dense/sparse and world/parity: 2.220446049250313e-16; 2.220446049250313e-16; 5.551115123125783e-16 |
| sparse_complex | DIGIT / PASS | sum(abs(diff)) / real arange indexed coefficient expression / <1e-14 | 0; 0; 0 |
| sparse_sy | DIGIT / PASS | three source comparisons, six shapes/symmetries / dense packed expressions / <1e-14 | every printed delta 0 at all ranks |
| sparse_sample | DIGIT / PASS | norm2 sequence / source zero fixture / nonincreasing | norms (0,0,0), both differences 0 at all ranks |

Every row uses the pinned source reference above. Reported maxima only
summarize already printed comparisons; each original assertion remains the
gate. APSP, block, CK7 and AMG print world Q only; their parity assertions
passed but parity Q is not reconstructed or claimed as measured.

| AMG WSL ranks | Q: rnorm | ref: rnorm_alt |
|---|---|---|
| 1 | 0.004937970528833717 | 0.005547611182988828 |
| 2 | 0.005085930671471991 | 0.005837246582193954 |
| 4 | 0.005189487439309569 | 0.005878901083452919 |

| Native target | Stamp at 1/2/4 | Q / delta at ranks 1; 2; 4 | Reference / bound |
|---|---|---|---|
| sparse_einsum_hadamard | DIGIT / PASS | 0; 0; 0 | same dense expressions / <1e-14 |
| sparse_scaled_expression | DIGIT / PASS | 2.220446049250313e-16; 2.220446049250313e-16; 5.551115123125783e-16 | grouped dense source expression / <1e-14 |
| upstream_checkpoint_sparse | DIGIT / PASS | 1.659570583342333e-6; 1.730313117560421e-6; 1.6343449060870917e-6 | original round-trip values / <3.43e-6 |

CK7's printed computed floating-point bound is 3.4299999999999998e-6,
the unchanged source expression at n=7. The other ten targets' native S1c
passes were not repeated. All prescribed native stderr files are empty.

### S1 close verdict

**DIGIT / PASS — G-CTF-S1d, G-CTF-S1-CK7, G-CTF-S1-AMG and G-CTF-S1.**
The unchanged full WSL script completed once with exit 0, 542
`DIGIT / PASS` lines and no failure line, including its prescribed rank
1/2/4 and local checks. Log: `D:/projects/runs/ctf-rs-s1/S1d/full-wsl.log`.
Native BuildOnly completed once with exit 0 and no missing MS-MPI symbol.
All thirteen targets and the three native targets passed as tabulated above;
diagnostic budget used 0/3 for every driver. No HANDOFF question remains.
Numerical verification is closed; only records/publication follow.

### Commands and run provenance

All paths below are under `D:/projects/runs/ctf-rs-s1/S1d/`. Per target,
`<target>-<ranks>.log` contains its exact WSL command, output, PASS and exit;
`native-<target>-<ranks>.command.txt` accompanies native `.log` and `.err`.
`wsl.log` and `native-runtime.log` preserve run order and all exit statuses;
`build.json` and `native-build.json` resolve the executed artifact paths.
Run count for **each target row**: one invocation per WSL rank count; for
the three native rows, one invocation per native rank count. These scripts
and `commands.md` preserve the exact nested commands and environment.

```powershell
$env:CARGO_BUILD_JOBS='2'
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\projects\ctf-rs\scripts\acceptance-native.ps1 -BuildOnly > D:\projects\runs\ctf-rs-s1\S1d\native-build.log 2>&1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\projects\runs\ctf-rs-s1\S1d\native-executables.ps1
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-s1/S1d/acceptance.sh
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-s1/S1d/full-wsl.sh
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\projects\runs\ctf-rs-s1\S1d\native-runtime.ps1 > D:\projects\runs\ctf-rs-s1\S1d\native-runtime.log 2>&1"
```

The existing `libmuffintin-wsl-keepalive` pid322 was reused; no second
keepalive was started. MPI initialization/finalization remains host-owned.
No planner or kernel redesign, dense X path, gather-based replacement,
collective Drop, libmuffintin/fftw source change or ctf-rs push occurred.

### Checkpoint n=3 history (not a source failure)

The previous six-decimal n=3 fixture had bound 2.7e-7. WSL one rank gave
3.9769591334653147e-7; two ranks and four-rank parity gave
3.4024306287098844e-7. Four-rank world passed without printing Q. The native
results agreed. One prescribed rank-one dense twin gave the identical
3.9769591334653147e-7, separating format rounding from sparse storage.
Diagnostic budget then used 1/3. No precision or tolerance changed.
Evidence remains under `D:/projects/runs/ctf-rs-s1/S1c/` in
`upstream_checkpoint_sparse-<ranks>.log`, native counterparts and
`checkpoint-dense-twin.{rs,sh,log}`. Plan.v4 replaces that undersized fixture
gate with the source default n=7, not a relaxed source criterion.

## S1a sparse contraction planning

### Fixed acceptance contract

Class R, pinned f69cbb46; WSL 1/2/4 once per driver and native compile/link
once. `block_sparse` moves to S1b for its matrix-of-tensors custom folded
kernel dependency (harness evt-1006); it remains required for the S1 close.

| Driver | Fixture | Q and reference | Unchanged bound |
|---|---|---|---|
| upstream_apsp | n=9, rank-seeded integer adjacency, dense and augmented sparse path doubling | differing path-weight entries versus dense tropical result | exact zero |
| upstream_algebraic_multigrid | n=4, original nlvl=2, ndiv=2, nsmooth=3, source Poisson/transfer formulas | V-cycle residual versus twice-smoothed fine-grid Jacobi residual | rnorm < rnorm_alt |
| sparse_einsum_hadamard | n=11, density .1, ijk,jkl->ijkl, sparse/sparse and sparse/dense | each dense Rust expression comparison | sum(abs(diff)) < 1e-14 |
| sparse_scaled_expression | n=5, density .1, literal 2.3/7/-1/-1/-2 expression and accumulated old C | source grouped dense expression versus dense and sparse executions | sum(abs(diff)) < 1e-14 |
| sparse_complex | real C-order arange(27), 3x3x3, aliased kij indexed view | .2*b0 + .7*a0.transpose([1,2,0]) | sum(abs(diff)) < 1e-14 |

The Python file defines its own strict sum-absolute `allclose`; the brief's
requirement to retain that rule takes precedence over plan.v3's mistaken
description as NumPy defaults. No tolerance is chosen from observed results.
Before execution, source tracing identified that the sparse/sparse Hadamard
and scaled expressions contain ABC weigh indices: pinned sparse `can_fold`
rejects them, and its non-inner constructor asserts dense B and C. The Rust
planner retains this boundary instead of admitting a dense-eligibility fold
or inventing an unsupported sparse-output leaf. Their runtime outcomes are
not pre-labeled as passes.

New raw sparse selections carry the storage/COO capability, selected original
distributions and sparse fold descriptor through execution. COO/CSR/CCSR
leaves reuse the source panel, replication, virtual and key-pinning paths;
sparse output is reduced, depinned and returned to its original layout.
Cache hits retain mappings, not values or nonzero counts. No C1 acceptance
check is rerun under S1a.

### S1a outcome (2026-09-11)

Prescribed WSL runs: each of the five drivers once at 1/2/4, revision
`bebe3a4`. Native compile/link of all five passed once successfully (two WSL
and one native build-only type-error attempts preceded it; no numerical runs
occurred in those attempts). Native runtime is reserved for S1c.

| Driver | Stamp | Quantity / delta at 1, 2, 4 ranks |
|---|---|---|
| upstream_apsp | DIGIT / PASS | differing weights = 0, 0, 0; exact bound 0 |
| sparse_complex | DIGIT / PASS | sum(abs(diff)) = 0, 0, 0; strict bound 1e-14 |
| sparse_einsum_hadamard | DIGIT / HANDOFF | selected=None at all ranks; Q and delta uncomputed; source rejects sparse ABC weigh indices |
| sparse_scaled_expression | DIGIT / HANDOFF | selected=None at all ranks; Q and delta uncomputed; same source restriction |
| upstream_algebraic_multigrid | DIGIT / HANDOFF | original runs stop at matricization row-order assertion; residual uncomputed |

Static AMG diagnosis found the selected inner ordering, not original tensor
dimension order, defines the row prefix. Source-derived fix `1285771` changes
only that metadata interpretation. One named rank-count split diagnostic at
1 rank then passed: rnorm=0.004937970528833717 versus
rnorm_alt=0.005547611182988828, strict less-than bound, difference
-0.000609640654155111; one recorded V-cycle timing 0.022667 s. This closes
that diagnostic, not the failed original 2/4-rank acceptance. No further
numerical verification or repeated passing driver was run. AMG diagnostic
budget used 1/3; each Python failure used 0/3 (decisive static source boundary).

Commands and logs under `D:/projects/runs/ctf-rs-s1/S1a/`:
`wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-s1/S1a/acceptance.sh`,
`native-build.ps1` via the exact command in `commands.md`, and
`wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-s1/S1a/amg-rank-split.sh`.
`wsl.log`, `build.json`, each `<driver>-<ranks>.log`, `native-build.log`,
`amg-rank-split.log` and its build JSON retain the complete evidence.
HANDOFF: may source-forbidden sparse ABC expressions remain unsupported, and
does the repaired AMG layout also pass the remaining distributed ranks?
S1b/S1c continue; their GEMM-shaped drivers do not require sparse ABC weigh
support, but custom folded work uses the repaired layout metadata. Final
native execution still includes all thirteen S1 drivers/semantics.

## S1b compressed-symmetry and custom sparse kernels

### Fixed acceptance contract

Class R, pinned f69cbb46, WSL 1/2/4 once for each driver, then native
compile/link once. Source fixture/metric rules below are fixed before runs.
No S1a passing numerical check is repeated under this batch.

| Driver | Fixture and reference | Q / unchanged bound |
|---|---|---|
| upstream_block_sparse | source n=7, r=10, matrix of distributed Tensor blocks, source flattening and dense flattened product | difference norm2 <= 1e-4 |
| upstream_force_integration_sparse | bounded n=5, source Drand48 particles, AS forces, cutoff .708, F2=F+F and two inverse applications | some dx/dy changes >1e-6 initially; every dx/dy restored within 1e-6 |
| upstream_btwn_central | n=6, sp=.2, bsize=2, test=1, sparse B/C, all three batches; fast Bellman/Brandes versus source naive dense closure | difference norm2 <= n*1e-6 = 6e-6 |
| sparse_sy | six exact source shapes/symmetries, fill_random(1,1), sparsify(0), X/Y, X-Y/0, vecnorm(X)/vecnorm(Y) | each source sum(abs(diff)) < 1e-14 |

`block_sparse` is the S1a-to-S1b move in evt-1006, not a dropped driver.
The source force-key transpose and duplicate AS canonical additions are
preserved. Sparse SY conversions keep canonical primary sparse entries;
neither planned force accumulation nor custom contraction gathers an operand.

### S1b outcome (2026-09-11)

DIGIT / PASS, revision `40e6350`: all four drivers ran once at WSL 1/2/4,
world/parity. Block sparse residual norm=0 at all ranks (bound <=1e-4);
betweenness residual norm=0 at all ranks (bound <=6e-6); every sparse SY
sum-absolute delta=0 (strict <1e-14). Force's exact source Boolean criterion
is 1 at all ranks: some particle changed by >1e-6, and every dx/dy was
restored within 1e-6; the source does not report a numerical maximum delta.
Native compile/link of all four passed once successfully. Two WSL and one
native build-only attempts failed on Rust type/literal syntax before any
numerical execution; they consumed no numerical acceptance run.

Runs: 12 prescribed WSL invocations, 0 diagnostics, no repeated passing
checks. Numerical verification is closed. Commands and logs:
`D:/projects/runs/ctf-rs-s1/S1b/{commands.md,acceptance.sh,native-build.ps1}`,
`wsl.log`, `build.json`, `<driver>-<ranks>.log`, `native-build.log`.
Exact entry commands:
`wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-s1/S1b/acceptance.sh`
and the redirected PowerShell native-build command in `commands.md`.
HANDOFF: none for S1b; S1a's three open drivers retain their recorded status.
The source block flattening quirk is retained without opening an extra study.
Continue S1c; the final native set still includes all thirteen S1 targets.

## S1c sparse summation, communication, persistence

### Fixed acceptance contract

Class R, pinned f69cbb46. The four S1c drivers run once at WSL 1/2/4;
native compile/link once, followed by the explicitly required full thirteen
S1 targets once each at native 1/2/4. This cross-platform gate is prescribed,
not an added comparison after earlier WSL passes. No earlier WSL pass reruns.

| Driver | Source fixture/reference | Q / unchanged bound |
|---|---|---|
| upstream_checkpoint_sparse | bounded n=3, sparse 3D density .1, rank-seeded CTF MT, original six-decimal sparse text write/read | norm2(v-u) < 1e-8*n^3 = 2.7e-7 |
| upstream_mis | n=16, density .1, SH sparse graph, canonical lower-triangle NS copy for directed algorithm | exact s^T A s=0 and number of zero entries in dense s+A*s=0 |
| upstream_mis2 | same source graph, max-monoid two-hop algorithm; source sparse t=s+A*s | count of stored t values >1.1 is 0; count of stored t values <.9 is 0 |
| sparse_sample | exact zero-initialized (4,3,5), sample .5 then .3 | norm_after1 <= norm_before; norm_after2 <= norm_after1 |

MIS2's source sparse unary sum visits stored entries only: its lower-bound
check does not count absent vertices. This is preserved rather than replaced
by a stronger graph-theoretic maximality assertion. MIS's target t is dense
and does count all vertices. The sample fixture is also intentionally the
source's zero tensor; neither fixture is strengthened or tuned after a run.
The checkpoint source uses fixed six-decimal text; no precision change may
be made to force its bounded residual gate to pass.

### S1c and full native S1 outcome (2026-09-11)

Run revision `1a0d822`. The four S1c WSL drivers ran once at each of 1/2/4
(12 invocations). MIS, MIS2 and sampling passed; checkpoint remains HANDOFF.
Native compile/link of all thirteen S1 targets passed once successfully.
One WSL and one native build-only attempt failed on shared helper lifetimes
before numerical execution; no fixture, assertion or criterion was changed.

| S1c driver | WSL 1/2/4 stamp | Q / delta and reference |
|---|---|---|
| upstream_mis | DIGIT / PASS | source f32 overlap=0; dense uncovered-vertex count=0, exact bounds |
| upstream_mis2 | DIGIT / PASS | stored-entry counts >1.1 and <.9 both 0; source checker, not a stronger graph property |
| sparse_sample | DIGIT / PASS | norms (0,0,0) at every rank count; both differences=0, nonincreasing bounds |
| upstream_checkpoint_sparse | DIGIT / HANDOFF | Q=3.9769591334653147e-7 at 1, 3.4024306287098844e-7 at 2; 4-rank world bound passes without a printed Q but parity Q=3.4024306287098844e-7 fails; strict bound 2.7e-7 |

The one permitted checkpoint **dense twin** at 1 rank preserved the generated
sparse fixture and used dense text I/O/subtraction. Q was identically
3.9769591334653147e-7, above 2.7e-7. This separates selected sparse key/sum
storage from the source six-decimal format expectation. Diagnostic budget
used 1/3; no precision change, fixture enlargement or further diagnostic.
The known bounded source-format discrepancy is recorded in
`upstream-known-failures.md` and remains unaccepted.

The required **whole native S1 set** ran once per target at 1/2/4:
39 invocations, 30 exits 0 and 9 exits 101, no timeout or skipped target.

| Native target | Stamp at 1/2/4 | Q / reference / bound |
|---|---|---|
| upstream_apsp | DIGIT / PASS | differing weights=0 / dense tropical / exact 0 |
| upstream_algebraic_multigrid | DIGIT / PASS | source V-cycle rnorm < twice-smoothed Jacobi; values below |
| upstream_block_sparse | DIGIT / PASS | residual norm=0 / flattened source product / <=1e-4 |
| upstream_force_integration_sparse | DIGIT / PASS | Boolean criterion=1 / original particles / initial any >1e-6, all restored within1e-6; no maximum delta printed |
| upstream_btwn_central | DIGIT / PASS | residual norm=0 / source naive / <=6e-6 |
| upstream_checkpoint_sparse | DIGIT / HANDOFF | same Q=3.9769591334653147e-7 (1), 3.4024306287098844e-7 (2 and 4-rank parity) / roundtrip / <2.7e-7 |
| upstream_mis | DIGIT / PASS | overlap=0, uncovered=0 / source graph / exact0 |
| upstream_mis2 | DIGIT / PASS | both stored-entry violation counts=0 / source graph / exact0 |
| sparse_einsum_hadamard | DIGIT / HANDOFF | selected=None; Q/delta uncomputed / source expression / <1e-14 unchanged |
| sparse_scaled_expression | DIGIT / HANDOFF | selected=None; Q/delta uncomputed / source expression / <1e-14 unchanged |
| sparse_complex | DIGIT / PASS | sum-absolute delta=0 / same dense expression / <1e-14 |
| sparse_sy | DIGIT / PASS | six source shapes, every comparison passes / dense packed expression / <1e-14 |
| sparse_sample | DIGIT / PASS | norms (0,0,0) / source zero fixture / nonincreasing |

Native AMG (one timing per invocation; informational):

| ranks | rnorm | rnorm_alt | rnorm-rnorm_alt | V-cycle seconds |
|---|---|---|---|---|
| 1 | 0.004937970528833717 | 0.005547611182988828 | -0.000609640654155111 | 0.142621 |
| 2 | 0.005085930671471991 | 0.005837246582193954 | -0.000751315910721963 | 0.108778 |
| 4 | 0.005189487439309569 | 0.005878901083452919 | -0.000689413644143350 | 0.046671 |

These native passes close the native AMG gate; they do not retroactively
replace S1a's original WSL 2/4 failures. That repaired distributed WSL gate
remains unaccepted under the brief's once-only rule. No passing driver was
repeated for confidence. Across S1: all 39 prescribed WSL invocations and all
39 prescribed native invocations occurred; only two extra diagnostics total
(AMG rank-count split in S1a and checkpoint dense twin in S1c).

Exact commands and evidence are under `D:/projects/runs/ctf-rs-s1/S1c/`:

```powershell
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-s1/S1c/acceptance.sh
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\projects\runs\ctf-rs-s1\S1c\native-build.ps1 > D:\projects\runs\ctf-rs-s1\S1c\native-build.log 2>&1"
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-s1/S1c/checkpoint-dense-twin.sh
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\projects\runs\ctf-rs-s1\S1c\native-runtime.ps1 > D:\projects\runs\ctf-rs-s1\S1c\native-runtime.log 2>&1"
```

Logs: `wsl.log`, `build.json`, `<driver>-<ranks>.log`, `native-build.log`,
`native-build.json`, `native-runtime.log`, `native-<driver>-<ranks>.{log,err}`,
and `checkpoint-dense-twin.log`; scripts and `commands.md` preserve nested
commands. No ctf-rs push; no libmuffintin/fftw code edit; existing WSL
keepalive reused. All four milestone commit series and records are delivered,
but **G-CTF-S1 remains HANDOFF**, not a full numerical close.
HANDOFF question: should a future plan change the two source-forbidden ABC
expressions or checkpoint's source precision contract, and authorize repaired
AMG WSL2/4 acceptance? No later batch remains pending on these items.

## rsmpi binding

### CTF-R1-4 direct replica restoration (2026-09-09)

The user stopped B1's history search; its bisect scratch copies, build patches
and targets were discarded.

Fix `a558e30c57bcb0e12abf18c8d2906237d41ccf90` changes only `src/tensor.rs`.
The optimized equal-phase block reshuffle and DGTOG ROR primitives populate
primary physical layers only. `Tensor::redistribute` returned those buffers
without restoring the public tensor invariant: every rank for which
`Distribution::global_key` returns a key must hold that key's value, including
replicas; only padding slots are zero. The legacy exchange already populated
all owning ranks.

One scratch-only two-rank trace named the mechanism: for shape `[3,5]`, cyclic
to fully unmapped redistribution, rank 1 has `receive_root=false`, receive
counts `[10,5]`, and 15 unpopulated valid local slots. Rank 0 unpacks the data;
rank 1 returns zero at offset 0 although that slot is key 0, whose i8 value is
`-5`. This is replica loss, not a wrong MPI exchange or padding index.

After either optimized path, the fix groups ranks by their mapped physical
residues, uses original-rank ordering to put the canonical owner at subgroup
rank 0, broadcasts that local block, and explicitly closes the subgroup.
This also restores scalar and partially mapped replicas while preserving
padding, the low-level primary-only primitives and the higher-order legacy
path. No test, fixture, driver, metric or tolerance changed.

```text
DIGIT / PASS
Q: cyclic_reshuffle exact values/padding/replicas, virtual phases, empty/scalar,
   i8/bool/complex/non-Copy Wire, world and parity contexts; class: R
ref: existing unchanged test assertions; bound: exact equality, 0
Delta: prior rank 1 offset 0 difference 5 (actual 0, expected -5);
       after fix every unchanged assertion passed, exact difference 0
runs: one pre-fix two-rank diagnostic (exit 124 after assertion panic);
      one post-fix confirmation at each of 1/2/4 ranks (exit 0)
closed: the single-test confirmation; no further diagnostic after the fix
```

The diagnostic was instrumented only in a scratch archive of `622ef10`, with
unchanged assertions. The current-tree layout review and trace agree on the
mechanism; no history result was used to choose the fix. Evidence and exact
commands are under `D:/projects/runs/ctf-rs-r1-4/`: `diagnostic.log`,
`confirmation.log`, `diagnostic.sh`, `confirm.sh`, and `commands.md`.

```powershell
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-4/diagnostic.sh
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-4/confirm.sh
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-4/acceptance.sh
$env:CARGO_BUILD_JOBS='2'
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -BuildOnly > D:\projects\runs\ctf-rs-r1-4\native-build.log 2>&1"
cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -D6Only > D:\projects\runs\ctf-rs-r1-4\native-d6.log 2>&1"
```

Native commands run from `D:/projects/ctf-rs`; all runtime logs are outside
the source tree. The acceptance scripts are unchanged, including their own
rank loops and complete driver selections.

```text
DIGIT / PASS
Q: every WSL acceptance driver's own metric and required invariants; class: R
ref: pinned upstream f69cbb46; bound: exact or upstream per driver, unchanged
Delta: every unchanged assertion passed; emitted numerical metrics remain in wsl.log
checks: all 175 MPI drivers completed at each of 1/2/4 ranks, followed by
        all prescribed library/local checks and the seven-rank Strassen run
runs: acceptance-wsl.sh once, each configured rank set once; exit 0
closed: WSL numerical verification; no extra diagnostics or precision checks

DIGIT / PASS
Q: native Windows GNU compile/link of all tests and examples; class: R
ref: plan.v2 native build gate; bound: successful compile/link, exit 0
Delta: no compile/link failure; no numerical delta applies
runs: acceptance-native.ps1 -BuildOnly once
```

```text
DIGIT / PASS
Q: native D6 drivers' own metrics and invariants; class: R
ref: pinned upstream f69cbb46; bound: exact or upstream per driver, unchanged
Delta: every unchanged assertion passed; emitted metrics remain in native-d6.log
checks: all 50 dense drivers completed at each of 1/2/4 ranks, plus four
        local scaling tests; 150 driver PASS stamps and the scaling target passed
runs: acceptance-native.ps1 -D6Only once, each rank configuration once; exit 0
closed: native numerical verification; no diagnostics or additional checks
```

**G-CTF-R1 is closed: DIGIT / PASS.** The direct replica-restoration defect
is fixed, its 1/2/4 confirmation passes, and the entire prescribed WSL/native
acceptance set passes. One diagnostic, three single-test confirmations, and
the three prescribed acceptance-script invocations were used. The separate
confirmation and full acceptance repetitions were explicitly requested; no
additional passing check was rerun. S1 was not started and ctf-rs was not pushed.

The separate `dgtog_redistribution` test contract is now closed under BRIEF-5
(2026-09-09), test commit `cccd5e8ef8bf2ebd66148dfeb8e7070bae240f63`.
The scalar assertion now requires `[29]` on every rank, not zero on nonroots;
the local-storage helper requires `key + 11` for every valid global key and
zero only for padding. Empty storage, high-order values and all other exact
assertions remain intact. No `src/` file or low-level primitive test changed.

```text
DIGIT / PASS
Q: every dgtog_redistribution assertion under the public Tensor replica contract
class: R; ref: test committed at cccd5e8; bound: exact equality, 0
Delta: all unchanged or authorized replica-contract assertions passed, difference 0
checks: populated replicas, padding, virtual/scalar/empty/high-order storage;
        world and parity contexts at each rank configuration
runs: WSL 1/2/4 once each, three runs total; exit 0; no diagnostics
closed: dgtog_redistribution contract; prior G-CTF-R1 passes remain closed
```

Command:
`wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-dgtog-test/confirm.sh`.
Log: `D:/projects/runs/ctf-dgtog-test/confirmation.log`; the adjacent
`confirm.sh` and `commands.md` record the exact nested commands and supervisor.
No full R1 rerun, native runtime run, or ctf-rs push was performed for this item.

### CTF-R1-2 diagnostic 3 (2026-09-09)

Harness evt-0055 and BRIEF-2 authorize exactly two diagnostic runs: trace
`Comm::exchange` at `813d90a` and raw-MPI baseline `f2039d3`, then fix only
the translation identified by the first divergent line. Identical traces
require handoff without a fix or acceptance rerun. The five R1 commits stand.

```text
DIGIT / HANDOFF
Q: per-call/rank exchange send_counts, recv_counts, and each sent/received
   bucket's FNV-1a-64 hash; existing exact i8 cyclic_reshuffle assertions
class: R; ref: f2039d3; candidate: 813d90a
bound: exact trace equality; exact driver equality, unchanged
Delta: zero differing trace fields across two rows (call 0, ranks 0 and 1)
driver result: both revisions fail at rank 1 local offset 0, actual 0,
               expected -5; absolute difference 5 against exact bound 0
runs: diagnostic 3 only, one two-rank run per revision, two runs total
exits: current 124 at the 60-second supervisor after assertion panic;
       baseline 101 after the same assertion panic and MPI process failure
finding: no first divergent exchange line and no wrapper identified;
         the same driver failure is reproduced on the pre-R1 baseline
scope: no fix(mpi) commit, no acceptance rerun, no further diagnostics;
       diagnostic 2 not applicable; R1 remains open, S1 held
open: what scope is authorized for the baseline failure with identical
      exchange traces, rather than an R1 wrapper translation defect?
```

Both traces, sorted by call and rank:

```text
call=0 rank=0 send_counts=[10, 0] recv_counts=[10, 0] sent_hashes=[3bc129cffea72d2d, cbf29ce484222325] received_hashes=[3bc129cffea72d2d, cbf29ce484222325]
call=0 rank=1 send_counts=[0, 10] recv_counts=[0, 10] sent_hashes=[cbf29ce484222325, 4a98c824fe6e7321] received_hashes=[cbf29ce484222325, 4a98c824fe6e7321]
```

Instrumentation and i8-only call selection were applied only to two scratch
archives, with separate Linux target directories. No assertions, layouts,
initialization, split, cleanup, or delivery code were changed. FNV-1a uses
offset `0xcbf29ce484222325` and wrapping multiplier `0x100000001b3`.
The raw logs' PowerShell line wrapping was joined when extracting complete
trace tuples; no count or hash field was discarded. This establishes the
baseline failure, not a claim that all R1 behavior is equivalent.

Exact commands (each once):

```powershell
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-2/diagnostic-3.sh current > D:/projects/runs/ctf-rs-r1-2/current.log 2>&1
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-2/diagnostic-3.sh baseline > D:/projects/runs/ctf-rs-r1-2/baseline.log 2>&1
```

`D:/projects/runs/ctf-rs-r1-2/commands.md` and `diagnostic-3.sh` record the
scratch revisions, narrowing, instrumentation, no-run builds, exact nested
commands and supervisor. `current.trace` and `baseline.trace` contain the
two complete rows each; `current.log` and `baseline.log` preserve both failed
assertions. Copies accompany harness evd-1010 under
`evidence/2026-09-09-ctf-rs-r1-2/`. The conditional acceptance rerun was **not
executed**, as recorded by evd-1011; the initial native passes below remain
closed and are not relabeled as new evidence.

### Initial R1 acceptance

R1, 2026-09-09, implementation `f072971` (series starts after `f2039d3`).
The host owns MPI initialization/finalization; contexts borrow its rsmpi
universe and communicator, check Funneled/main-thread support, and free only
their own split communicators through explicit close. Thread markers remain;
default features and libffi are disabled. MPI-IO and nonblocking ROR traffic
retain their raw calls through `mpi::ffi`. No metric, tolerance, fixture,
algorithm, acceptance script, or driver body changed. The 184 entry-point
edits matched the mechanical lifecycle substitutions plus formatting.

```text
DIGIT / HANDOFF
gate: G-CTF-R1; class: R
Q: each WSL driver's existing metric and required invariants
ref: pinned upstream f69cbb46; bound: each driver's unchanged upstream bound
Delta: cyclic_reshuffle at two ranks, rank 1 local offset 0: actual 0,
       expected -5, absolute difference 5 against exact bound 0
checks: all 175 one-rank MPI drivers completed; at two ranks eight drivers
        passed before the ninth, cyclic_reshuffle, failed its assertion
runs: acceptance-wsl.sh invoked once; rank 1 once, rank 2 partial once;
      rank 4 and subsequent local/seven-rank checks not reached
diagnostics: one computation, plan diagnostic 1, cyclic_reshuffle at two ranks;
             both ranks reported size 2 and Funneled and passed the main-thread
             check; the same exact mismatch recurred before explicit close/drop
diagnostic exit: 124 at the 60-second supervisor after the rank-local panic
                left MPI peers blocked; original failed launcher terminated,
                acceptance script exit 1
scope: no numerical repair, baseline rerun, sweep, changed bound, or repeated pass
open: what R1-scoped resolution is authorized for cyclic_reshuffle's exact
      replica/layout mismatch (0 versus -5) with correct rank/thread setup?
```

The assertion occurs during the world-context test, before any explicit close
or universe drop. Further diagnostics were not spent: initialization is ruled
out, and the accepted plan requires handoff rather than numerical/driver repair.
The diagnostic print was applied only to the Linux work copy and then removed.
No passing numerical result was reopened.

```text
DIGIT / PASS
Q: native Windows GNU compile/link of all tests and examples; class: R
ref: plan.v2 native build gate; bound: successful compilation/link, exit 0
Delta: no compile/link failure; no numerical delta applies
runs: acceptance-native.ps1 -BuildOnly once
```

```text
DIGIT / PASS
Q: native D6 dense drivers' existing metrics and invariants; class: R
ref: pinned upstream f69cbb46; bound: exact or upstream per driver, unchanged
Delta: every invoked driver's unchanged assertion passed; numerical deltas
       where emitted are preserved in native-d6.log, not replaced by a new metric
checks: 50 dense drivers at each of 1/2/4 ranks and four local scaling tests passed
runs: acceptance-native.ps1 -D6Only once, each rank configuration once; exit 0
diagnostics: none; native numerical verification closed
```

Overall R1 remains **DIGIT / HANDOFF** because the WSL gate is incomplete.
Native passing results do not replace the failed WSL result. Total execution:
three acceptance-script invocations and one named diagnostic computation;
no passing driver was repeated at the same platform/rank configuration.

Commands (from `D:/projects/ctf-rs`) and logs, all outside the source tree:

```powershell
wsl -d Ubuntu-26.04 -- bash -lc 'cd /mnt/d/projects/ctf-rs && bash scripts/sync-wsl.sh && cd /home/xylxp/ctf-rs-work && bash -x scripts/acceptance-wsl.sh > /mnt/d/projects/runs/ctf-rs-r1/wsl.log 2>&1'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -BuildOnly > D:/projects/runs/ctf-rs-r1/native-build.log 2>&1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/acceptance-native.ps1 -D6Only > D:/projects/runs/ctf-rs-r1/native-d6.log 2>&1
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1/diagnostic-1.sh > D:/projects/runs/ctf-rs-r1/diagnostic-1.log 2>&1
```

`D:/projects/runs/ctf-rs-r1/commands.md` records the exact commands, temporary
diagnostic line, supervisor, stale-copy cleanup, and single keepalive startup;
`diagnostic-1.sh` records the nested diagnostic command. The unchanged acceptance
scripts contain their own rank loops; neither was wrapped in another rank loop.

## D6 dense drivers and native runtime close (2026-09-08)

The pinned dense `test_suite.cxx` subset now asserts every active dense
criterion directly instead of trusting the source program's unconditional
zero exit status. Its intended `n*n` dimensions replace the C++ `n^2` XOR,
while the already bounded readwrite/readall fixtures remain n=3 and n=2/3.
The driver covers dense NS/SY/AS/SH products, CCSDT, scalar/trace/diagonal,
fast symmetric paths, subworld and recursive multiplication, one-level
Strassen, repack, transforms, FFT/DFT, force/particle operations and the
source double-precision QR/SVD/eigh criteria. Sparse cases are excluded.

The source examples `matmul`, `recursive_matmul`, `ccsd`, `ao_mo_transf`,
`neural_network`, `bitonic_sort`, `checkpoint`, `force_integration`,
`particle_interaction`, `qinformatics` and four-type `mttkrp` have direct Rust
drivers. Their source sizes and criteria were retained: matmul norm <=1e-6,
recursive norm <1e-9, nonzero neural output, exact bitonic order, both
checkpoint norms <=1e-9*n, force modification/restoration at 1e-6, particle
norm <1e-6 and each MTTKRP norm/size <1e-5. CCSD, AO-MO and qinformatics have
no source numerical gate, so acceptance is normal completion. Particle
replication uses owner broadcasts, and recursive multiplication serializes
parent/child transfers by subworld; neither path gathers a distributed
operand.

All twelve D6 drivers plus `dense_low_memory` passed once at WSL 1/2/4 ranks
with world/parity contexts. One AO-MO diagnostic replaced an invalid group-only
relabel with the source NS-to-AS summation; one test-suite diagnostic restored
the fixed n=3 and n=2/3 read fixtures; the rank-two force layout and recursive
subworld failures were corrected from their explicit phase/orientation
assertions. No fixture size, metric or tolerance changed. At four ranks,
`bench_contraction` reported 0.00014722066666666666 s/iteration and the dense
`model_trainer` subset reported 29.087425941 s; each timing ran once.

The full native Windows GNU test/example set compiled and linked once. After
Microsoft MPI was installed, the prescribed native dense runtime gate passed
once at 1/2/4 ranks. Every invoked driver reported DIGIT / PASS within its
fixed exact or numerical bound, including `d4_blas_flops`, `dense_low_memory`,
`distributed_symmetric_repack`, the dense test-suite subset and all eleven D6
examples. No tolerance, fixture or fallback changed. DIGIT / PASS; open: none.

## D5 native interface and FFT close (2026-09-08)

The remaining pinned interface surface is represented by native Rust types and
operations rather than C++ compatibility wrappers. `Partition` and
`IdxPartition` bind explicit process-grid dimensions to tensor labels; a
one-dimensional `Tensor` plus typed `arange` replaces `Vector`; zero-order
`Tensor` set/read operations replace `Scalar`; and the existing `Context` and
`Runtime` remain the native world/communicator owners. The algebra traits now
cover the pinned set, monoid and ring responsibilities, including wrapping
`u32`/`u64` arithmetic. Common index, prefix, permutation and allocation
helpers have direct Rust owners.

`d5_partition`, `d5_algebra_interfaces`, `d5_value_interfaces` and `d5_common`
passed exactly at WSL 1/2/4 ranks with world/parity contexts. The source-faithful
`upstream_fft_with_idx_partition`, `upstream_fft`, `upstream_dft_3d`,
`upstream_endomorphism_cust_sp`, `upstream_endomorphism_cust`,
`upstream_endomorphism`, `upstream_univar_function`, `upstream_bivar_function`
and `upstream_bivar_transform` drivers passed once at the same rank counts and
contexts. Their original sizes and bounds were retained: n=6/logm=8 and
`n*n*m*1e-6` for indexed FFT, n=16 and 1e-6 component residuals for FFT,
n=6 and per-element real error below 1e-9 for three-dimensional DFT, and the
existing exact or 1e-6 transform criteria.

The first multi-rank three-dimensional DFT attempt exposed that a symmetric
fixture mapped only one of the linked axes. One source-directed diagnostic
corrected the test distribution to give all linked axes the same virtual
phase; no tolerance, metric or fixture size changed. The full native Windows
GNU test and example set then compiled/linked once; native runtime remains
deferred to D6. DIGIT / PASS; D5 verification closed.

## D4 shared infrastructure close (2026-09-08)

The low-memory planner now has an explicit process budget derived from the
pinned `memcap * physical / processes-per-machine - used` rule. Linux uses the
smaller active cgroup limit and Windows uses the native physical-memory/process
working-set APIs; resident set replaces the source allocator registry because
Rust tensor storage is not centrally allocated. Rank-local budgets are reduced
to the communicator minimum before selection. The existing folded low-memory
execution remains the production consumer.

Named timers retain inclusive/exclusive/call accumulation and MPI totals with
explicit registry ownership. Source util recurrences cover packed sizes/index
decoding, factorization, permutations and strided/ragged copies while fixing
the pinned `sy_calc_idx_arr` copy-width and `socopy` allocation defects. Four-
type BLAS SYR is integrated into the existing `linalg` boundary; complex SYR
uses plain transpose, not HER. A process-global atomic flop total is sampled by
`FlopCounter`, and the production four-type GEMM center records `2*m*n*k`.

`d4_memcontrol`, `d4_timer_util` and `d4_blas_flops` passed once at WSL 1/2/4
ranks with world/parity contexts. All arithmetic, layout, copy, SYR and flop
checks were exact; timer checks were limited to finite nonnegative accumulation
and nesting invariants. The first timer/util attempt exposed an unsigned index
translation and the second exposed an incorrect expected ragged-copy fixture;
the two distinct diagnostics corrected those issues without changing a bound.

`mpi_low_memory_bench` ran once at four ranks with m=128,k=192,n=160 and
reported 0.002395 s, source memory 397312 bytes and process budget 3094485248
bytes. This is informational only. The full native Windows GNU test and example
set compiled/linked once; native runtime remains deferred to D6. DIGIT / PASS;
D4 verification closed.

## D3 dense contraction close (2026-09-08)

Dense mapped contractions now derive the source replication fibers from all
three operands, broadcast missing A/B axes, apply beta only on the C root,
execute the recursive 2D/virtual child, reduce C to its roots and clear input
replicas before communicator release. The ordinary and folded paths share this
wrapper; nested packed panels retain f32/f64/complex32/complex64 dispatch and
the transposed-C operand swap. Automatic symmetric contraction and canonical
sum planning select a compressed aligned topology in pinned candidate order,
then use packed broadcast/reduce execution without gathering.

`upstream_gemm4d` (NS/SY/AS/SH), `upstream_weigh4d`,
`upstream_sy_times_ns`, `upstream_ccsdt_t3_to_t2`, `upstream_ccsdt_map`,
`upstream_multi_tsr_sym`, and all seven `upstream_fast_*` studies passed once
at WSL 1/2/4 ranks with world/parity contexts. Fixture sizes and source bounds
were unchanged: n=7 for gemm_4D, n=6 for fast_sym_4D, n=13 for fast_sym,
n=5 for fast_3mm/fast_diagram, and n=4,s=t=v=1 for the tensor studies.

The largest reported fast-driver residuals were 4.865e-15 (fast_3mm),
2.990e-13 (fast_diagram), 6.882e-14 (fast_sym_4D), 5.169e-14 (fast_sym),
2.665e-15 for the tensor-study internal symmetry checks, and 1.876e-14 for
their final residuals. `sy_times_ns` reported at most 2.260e-16 and
`multi_tsr_sym` reported zero. The first fast_3mm and fast_diagram attempts
exposed packed coincidence normalization differences; one source-directed
diagnostic per driver identified the diagonal-only SY factor and whole packed
AS/SH factor respectively, after which their prescribed metrics passed. No
tolerance, metric or fixture changed.

The full native Windows GNU test and example set compiled/linked once; native
runtime remains deferred to D6. DIGIT / PASS; D3 numerical verification closed.

## D2 optimized dense redistribution close (2026-09-08)

Dense orders through twelve now use the pinned default DGTOG ROR path: closed-
form physical-phase LCM buckets, exact counts/displacements and value offsets,
old/new replica-root selection, receive-before-send nonblocking MPI traffic,
root unpack and additive-identity new nonroot replicas. Equal-phase layouts use
whole virtual-block reshuffling first; higher orders retain the legacy cyclic
path. The dense slice rank shift, contiguous-prefix nonsymmetric transpose,
canonical packed pad/depad/zeroing and arbitrary-phase global reshuffle are
direct Rust operations with no global tensor gather.

`upstream_readwrite`, `upstream_readall`, `distributed_symmetric_repack`, the
NS `upstream_permute_multiworld`, `upstream_reduce_bcast` and
`upstream_subworld_gemm` passed once at WSL 1/2/4 ranks with their existing
world/parity contexts. Exact integer/layout checks and the original 1e-10,
1e-9 and 1e-6 driver bounds all passed. No numerical failure, diagnostic,
tolerance change or repeated passing driver run occurred.

The release `bench_redistribution` and `bench_nosym_transp` drivers each ran
once at four ranks after exact probes. Their single informational timings were
0.000077 s for a 48x40 redistribution and 0.000656 s for a 192x160 transpose;
no speedup or statistical claim is made. The full native Windows GNU test and
example set compiled/linked once; runtime remains deferred to D6.

The disabled post-pack block in pinned `glb_cyclic_reshuffle` and the disabled/
double-increment padding code are recorded source defects, not preserved as
executable semantics; the Rust paths complete exchange/unpack and visit every
allocated virtual block. DIGIT / PASS; D2 numerical verification closed.

## D1 dense scaling and strip close (2026-09-08)

The `scaling` target passed four exact local checks once in WSL. It preserves
source right-multiplication before a custom transform, packed canonical
SY/AS/SH traversal, repeated-label virtual strides, physical-rank strip
selection, compact strip copy and restore, and the corresponding block-size
reductions. Fixtures remain bounded at n=3/4/5 and use exact String/i64,
index, offset and layout comparisons.

`upstream_scalar`, `upstream_diag_sym`, `upstream_weigh4d` and `upstream_dft`
then passed once at WSL 1/2/4 ranks with their existing world/parity contexts.
Their source criteria remain unchanged: scalar zero-edge/norm inequalities,
diag_sym norm <1e-10, weigh_4D's signed relative 1e-10 rule after the 1e-10
cutoff, and DFT real error <1e-9. All observed values were within those bounds;
no failure, diagnostic computation, tolerance change or repeated passing run
occurred. The full native Windows GNU test set compiled and linked once; native
runtime remains deferred to D6. DIGIT / PASS; D1 numerical verification closed.

## Generic BLAS declarations and fractional node costs (2026-09-08)

The existing four-type local GEMM dispatch now declares the s/c/z BLAS symbols
that its f32/complex implementations already call. Replicated communication
cost trees store the source `CommData::comm_nodes` average as `f64`, preserving
fractional peer-node counts instead of requiring an integer. This repairs the
compile failures that rejected the B0 WIP; it changes no numerical metric,
fixture or algorithm. WSL and native Windows GNU compile/link passed once; no
runtime or numerical driver was executed. DIGIT / PASS for compilation only.

## B0 sparse WIP disposition and expression inventory (2026-09-08)

After syncing `f22da3a` to the Linux work copy, the single prescribed
`distributed_sparse_search_cache` acceptance attempt stopped during compilation
before any 1/2/4-rank execution. The library lacked the `sgemm_`, `cgemm_` and
`zgemm_` declarations and passed `f64` node counts to five `usize` fields. Per
the B0 fallback, the commit is retained on `wip/sparse-search-cache` and the
delivery branch returned to `da5354b`; no tolerance or fixture was changed.

The six `term`, `idx_tensor` and `fun_term` source/header inventory rows are now
`replaced`: direct Rust operations replace the expression interface, matching
the README contract. Native compile/link is not applicable because B0 retained
no source batch. DIGIT / FAIL for the rejected WIP only; B0 bookkeeping closed.

## Unfolded sparse raw execution and automatic selection (2026-09-08)

sparse_mapped_cost passed its three local WSL checks once after adding shared
execution descriptors. distributed_sparse_raw and distributed_sparse_search,
plus affected aligned distributed_sparse_general/function/plan, passed once
at WSL 1/2/4 ranks with world/parity contexts. Exact i64 checks cover uneven
mapped GEMM, shared-axis mismatches, virtual factors, nonempty/empty sparse A,
alpha/beta and restored C distribution. Search covers normal, weighted and
enabled exhaustive passes, collective winner agreement and strict zero-memory
rejection; selected raw mappings execute rather than being replaced by aligned
ones. No numerical diagnostics or tolerance changes were needed. DIGIT / PASS.
All six targets compiled/linked on Windows GNU; native runtime remains
unaccepted. Folded/sparse-output/compressed automatic planning is still pending.

## Sparse randomized matrix/tensor SVD (2026-09-08)

typed_randomized_svd and typed_tensor_svd passed once at WSL 1/2/4 ranks,
world/parity, covering all four scalar types. The sparse matrix remains sparse
during Gram and projection products. Tests cover a missing row (rank 1 has an
empty sparse shard at four ranks), real automatic rank-two fixtures, complex
plain-transpose projection from a supplied guess, oversampled guess writeback,
and exact zero-iteration guess/distribution preservation plus reconstruction.
The tensor test adds sparse randomized input permutation and unchanged stored
pairs. Existing Frobenius m*n*n*1e-6 and tensor normalized <1e-6/factor criteria
remain unchanged. DIGIT / PASS; no failures or additional precision studies.
Both targets compiled/linked once on Windows GNU; native MPI runtime acceptance
remains outstanding.

## Sparse tensor SVD and sequential HOSVD (2026-09-08)

typed_tensor_svd and upstream_hosvd passed once each at WSL 1/2/4 ranks,
including parity subcommunicators. Four scalar types cover sparse input
permutation/reshape, unchanged stored pairs, dense factor output, and existing
dense truncated/randomized behavior after sharing index/factor layout code.
The existing normalized reconstruction <1e-6 and factor criteria are unchanged.
The HOSVD driver covers dense and sparse [2,3,4,5] tensors, R=1/2, four
successive source mode SVDs and singular scaling, with the original finite
residual bound input_norm*(1-(R/n)^4)+1e-4. No vector phase comparison,
tolerance adjustment or diagnostic computation was needed. DIGIT / PASS.
Both targets compiled/linked on Windows GNU; native runtime acceptance remains
outstanding. Sparse randomized SVD remains unimplemented, not densified behind
the new explicitly truncated sparse API.

## Sparse density/redistribution estimates and Jacobi (2026-09-08)

redist_cost (three tests) and sparse_cost (four tests) passed once in WSL.
New checks use exact representable density/model values and integer workspace
counts, including physical replicas, explicit output-density override, equal
phase sparse redistribution, fractional-byte truncation and empty sparse input.
upstream_jacobi passed once at WSL 1/2/4 ranks with world/parity contexts.
It preserves source residual stopping <1e-4 (maximum 100 iterations) and final
dense/sparse solution difference norm <=1e-6; residuals must be finite.
N=3 and a rank-local MT stream are explicit bounded-fixture adaptations.
All three targets compiled/linked on Windows GNU; no native MPI runtime pass
is claimed. DIGIT / PASS; no additional precision verification.

The subsequently completed sparse_mapped_cost assembly passed its three local
WSL checks once and compiled/linked on Windows GNU. Exact model coefficients
check tree order, panel strides, virtual factors, recursive time/workspace and
redistribution totals. These are metadata/formula checks, not a claim of
automatic sparse candidate execution or measured peak memory.

## Sparse plan reuse and spectral element (2026-09-08)

distributed_sparse_plan, upstream_spectral_element, distributed_sparse_general
and distributed_sparse_function passed once at WSL 1/2/4 ranks, world/parity.
An initial compilation required an explicit `[Vec<usize>; 3]` annotation;
no numerical diagnostic runs were needed. Sparse results, distribution restore
and cache statistics use exact integer equality. The spectral-element source
sequence retains its finite norm2 >= 1e-6 criterion, not a residual-accuracy
oracle; bounded N=3 and an explicit rank-local MT stream replace source N=16
and its global RNG lifetime. DIGIT / PASS; numerical verification closed.
All four targets compiled and linked on Windows GNU. Native MPI runtime
acceptance is still outstanding; compilation is not execution evidence.

## Custom sparse GEMM integration and representative measurement (2026-09-08)

distributed_sparse_gemm_function, distributed_sparse_function_output and
distributed_sparse_fold_function passed once at WSL 1/2/4 ranks, world/parity,
after replacing their custom sparse-left panel loops with matricization and
sparse_2d. Existing exact i64/key criteria cover non-distributive functions,
stored/missing zeros, beta-zero merging, high-order batches, permutations
and repeated labels. DIGIT / PASS; no precision studies. All three targets
and sparse_gemm_bench compiled/linked on Windows GNU, not runtime-accepted.

The explicitly requested time/memory measurement was performed once per
1/2/4 ranks on a fixed release-build sparse contraction. Results and the
per-rank lifetime RSS interpretation are in sparse-performance.md; no
speedup claim or additional numerical verification was made.

## Sparse GEMM integration and self mapping (2026-09-08)

The existing distributed_sparse_gemm, distributed_sparse_dense_output,
distributed_sparse_fold, distributed_sparse_gemm_function and
distributed_sparse_function_output suites passed once at WSL 1/2/4 ranks,
world/parity, after the ordinary paths were connected to matricization and
sparse_2d. Existing exact integer/key criteria cover dense/sparse outputs,
CSR/CCSR, beta, empty panels/k, rectangular grids, high-order folding,
repeated/input-only labels, custom paths and restored output distributions.

self_mapping's three local tests and mapping_preflight's two existing tests
passed once after fixing an Option index type at compilation. They cover the
source first-pass virtual-map behavior, phase coordination, physical/repeated
map rejection and calc_dim floor divisions. All seven targets compiled and
linked once on Windows GNU. DIGIT / PASS; no numerical failures, extra
precision runs or native runtime pass claimed.

## Upstream one-level Strassen (2026-09-08)

upstream_strassen passed once at WSL 1/2/4 ranks plus one 7-rank run covering
the source divisible-by-seven branch. World/parity runs include all four
NS/AS/SY/SH inputs. Seven half-size products, source quadrant coefficients
and off-diagonal sign/transpose constructions are checked against the
ordinary distributed product with the original strict squared-error/n^2
<1e-10 rule and finite results. Child products run independently after
explicit parent quadrant transfers; no full input gather is used. The source
is one level only, not recursive. DIGIT / PASS; no precision study. Native
Windows compilation/linking passed; missing MPI runtime remains unaccepted.

## Mixed sparse output and typed COO communication (2026-09-08)

mixed_sparse_output passed two local WSL tests once: exact symbolic structure,
first-product assignment, order-sensitive subsequent accumulation/old merge,
duplicate old coordinates, stored cancellations and empty product. The updated
COO 2D executor passed distributed_mixed_coo once at WSL 1/2/4 ranks,
world/parity, using i32 sparse A, bool dense B and i64 output. Checks cover
stored zeros, empty A panels, A broadcast and simultaneous B broadcast/C
cyclic reduction. All comparisons are exact; DIGIT / PASS, no precision
study. Both targets compiled/linked once on Windows GNU; native runtime
remains unaccepted due to the previously confirmed missing MPI DLL.

## Mixed CPU kernels and sparse matricization (2026-09-07)

mixed_kernel passed two local WSL tests once after replacing unsupported
Transpose equality at compile time with variant matching. Exact integer
checks cover both operand transposes, mixed scalar types, order-sensitive
accumulation, CSR dense/sparse structure, zeros, duplicates and strided xpy.
sparse_matricize passed three local WSL tests once for phase-aware reordered
coordinates, sorted reverse keys, folded SY binomial rank, phased folded
groups, scalar and empty inputs. Both new targets compiled/linked once on
Windows GNU. DIGIT / PASS; no numerical failures or precision studies.
These local kernels/layout transforms introduce no collectives, so no MPI
rank sweep was added; full native runtime remains blocked as previously noted.

## Upstream sparse shortest paths (2026-09-07)

upstream_sssp passed once at WSL 1/2/4 ranks, world/parity. It keeps source
n=7, finite tropical infinity=n*n, min/plus algebra, diagonal infinity and
strict <5*n sparsification. The integer RNG is explicitly adapted to seeded
MT19937-64; no cycle edges are reserved or removed before the positive case.
Source cycle entries are overwritten through write_scaled with one/zero
coefficients in the tropical algebra, not accumulated with the old edges.
The ordinary integer SUM of distances must never increase; positive graph
convergence and injected negative-cycle nonconvergence retain the n+1 limit.
Source P["ij"] is an order-one indexed term (extra label ignored by C++)
converted through the default integer ring, not a tropical-min reduction.
DIGIT / PASS with exact integers. Native compilation/linking passed; Windows
MPI execution remains blocked by the missing runtime DLL.

## Native COO CPU kernels and 2D execution (2026-09-07)

sparse_coo passed two local WSL tests once after fixing explicit Rust
PartialEq bounds at compile time. Exact integer checks cover duplicates,
stored zeros, generic/default beta paths, mixed A/B/C types, order-sensitive
custom accumulators and empty COO. distributed_coo_2d passed once at WSL
1/2/4 ranks, world/parity, covering variable COO A broadcasts, dense B
broadcasts, cyclic output reduction and contiguous/strided dense scatter.
DIGIT / PASS; no numerical failures or precision studies. Both targets
compiled/linked once on Windows GNU; missing MPI runtime remains unaccepted.

## Sparse key pin/depin and mapped production integration (2026-09-07)

sparse_keys passed three local WSL tests once: exact virtual residues,
partial-phase padding removal, divisible shapes, stored zeros, empty blocks,
zero extent, scalar keys and source pin-cost switch fallthrough. After
integrating pinning into mapped sparse contractions, distributed_sparse_general
and distributed_sparse_function passed once at WSL 1/2/4 ranks, world/parity,
with their existing exact i64 criteria. These cover input/output-only labels,
physical i/j/k mappings, virtual factors, sparse replicas, custom stored-zero
evaluation, empty shards and output distribution restoration. DIGIT / PASS;
no failures or precision studies. All three targets compiled/linked once on
Windows GNU; runtime remains blocked by the previously confirmed missing MPI
DLL, not claimed as passing.

## Sparse CPU time and memory models (2026-09-07)

sparse_cost passed its three local tests once in WSL. Exact constructed
coefficients and dyadic fractions check all twelve CPU k0..k5 ordinary/custom
models, raw/folded flop counts, 10x/30x traffic heuristics, per-operand integer
truncation, sparse/dense panel payloads, layer scaling, CSR-vs-dense reduction
model selection, replica copy thresholds, temporary-memory maxima and virtual
composition. No MPI run is needed for these metadata-only functions. DIGIT /
PASS for the model formulas; no precision study or repeated passing checks.

Windows GNU compilation/linking passed. Native execution was attempted but
failed before tests loaded: with the native library PATH set, exit status was
0xc0000135 (DLL not found). The executable imports msmpi.dll, and
C:\Windows\System32\msmpi.dll is absent. This is not a native runtime pass.

## Raw sparse-pair 2D communication (2026-09-07)

distributed_sparse_2d_pairs passed once at WSL 1/2/4 ranks, world/parity.
It connects raw pair blocks directly to the 3D A[ikl]*B[kj]->C[ijl] local
kernel: non-matrix local keys, empty panels, explicit zeros, A/B broadcast,
strided stationary and cyclic moving dense output, beta=0/2, stationary
beta-once, layer subsets and nested levels. Exact i64 results agree with
the stated indexed products and source accumulation rules. The target
compiled and linked once on Windows GNU. DIGIT / PASS; no additional
precision runs. Native MPI runtime acceptance remains pending.

## Mixed sparse 2D dense output (2026-09-07)

distributed_sparse_2d_dense passed once at WSL 1/2/4 ranks, world/parity.
Exact i64 checks cover moving CSR A, moving dense/CSR B, variable empty sparse
panels, moving dense C with cyclic Reduce, one/two strips and consecutive
blocks, two-element output blocks, beta=0/2, stationary whole beta-once,
all layer schedules and a recursive dense-output child. References follow
the source dense new + beta*old rule, not the distinct sparse-C convention.
The target compiled and linked once on Windows GNU. DIGIT / PASS; numerical
verification closed, native MPI runtime acceptance still pending.

## Sparse 2D levels and upstream cyclic trace (2026-09-07)

distributed_sparse_2d and upstream_trace passed once at WSL 1/2/4 ranks,
world/parity. Exact i64 checks cover moving A/B/C, variable sparse payloads,
fixed dense broadcasts, CSR/CCSR cyclic output reductions, contiguous/strided
scatter, all layer-scheduling branches and an actual recursive child level.
Raw sparse beta expectations were fixed from the source before execution:
moving C adds unscaled old C; strided fresh output replaces it; stationary
whole output applies beta only on the first executed step.

The bounded n=3 f64 trace driver retains four NS ABCD cyclic orderings,
rank-seeded drand48 recurrence, diagonal extraction and reduction. Adjacent
trace relative errors retain the source <=1e-10 bound and denominator choice,
with finite results. Both targets compiled/linked once on native Windows GNU.
DIGIT / PASS; no failures, extra precision checks or repeated passing runs.
Native MPI runtime acceptance remains pending.

## Upstream recursive scan (2026-09-07)

upstream_scan passed once at WSL 1/2/4 ranks, world/parity, with bounded logn=3
and an explicitly seeded f64 MT fixture. It retains local vector-to-tensor
writes, recursive first-axis summation, AS->SH->NS canonical repacks for the
shift matrix, contractions, broadcast addition and tensor-to-vector writes.
Only the source acceptance exports use all-rank data. The original adjacent
prefix relation has strict absolute error <1e-9*N; finite values are required.
The f64 conversion's remaining integer literals were corrected after a compile
error and before execution. Native Windows GNU compile/link passed once.
DIGIT / PASS; no numerical failures or extra precision checks. Native MPI
runtime acceptance remains pending.

## Sparse-output replicated/virtual communication (2026-09-07)

distributed_sparse_replicate passed once at WSL 1/2/4 ranks, world/parity.
Exact i64 values and sparse structure cover CSR and CCSR input broadcasts,
fixed-size dense B broadcasts, virtual contracted blocks, multiple virtual C
blocks, beta=0/3 applied once, explicit stored zeros and both input cleanup
rules. Single and two orthogonal output-fiber reductions execute the new sparse
matrix reduction API; no-output-fiber cases retain each rank's local result.
Native Windows GNU compile/link passed once. DIGIT / PASS; no failures or
extra numerical verification. Native MPI runtime acceptance remains pending.

## Distributed CSR/CCSR reduction (2026-09-07)

distributed_sparse_reduce passed once at WSL 1/2/4 ranks, world/parity and a
three-rank subgroup of the four-rank run. Exact checks cover roots 0 and last,
CSR/CCSR cyclic row pieces, nonuniform/empty pieces, zero-row matrices, stored
zeros, root-only results and a million-row CCSR matrix with only a few entries.
An associative noncommutative matrix-product monoid checks the source binary
rank order. Initial compilation required explicit Clone bounds on concrete
format/COO helpers; these were added before any numerical execution.
Windows GNU compile/link passed once. DIGIT / PASS; no numerical failures or
additional precision runs. Native MPI runtime acceptance remains pending.

## Upstream sparse matrix-vector product (2026-09-07)

upstream_spmv passed once at WSL 1/2/4 ranks, world/parity. The bounded n=5
fixture retains source matrix sparsity 0.5/n and initial-vector sparsity 0.5,
with an explicit Rust MT seed adaptation. Both dense and sparse output execute
the source two half-weighted sparse/dense operand orders and compare against
the dense reference update. Original criteria remain initial norm>=1e-6 and
residual norm<=1e-6, with finite values. An ambiguous inferred Arithmetic type
was resolved explicitly before execution; no tolerance change was made.
Native Windows GNU compile/link passed once. DIGIT / PASS; no extra numerical
checks. Native MPI runtime acceptance remains pending.

## Remaining ordinary sparse-output storage dispatch (2026-09-07)

distributed_sparse_storage_dispatch passed once at WSL 1/2/4 ranks with
world/parity subcommunicators. Both matrix and indexed entry points are covered.
Exact keys/values verify dense+dense sparse-output retention of every valid
zero, repeated-output diagonal isolation, original off-diagonal values and
virtual distribution restoration. A noncommutative 2x2 matrix semiring verifies
the pinned ordinary dense-A/sparse-B swap computes B*A, with square/rectangular
process grids and empty local pieces. Initial compilation rejected replacing
the lifetime-bearing sparse tensor from shortened dense input borrows; the fix
transfers only owned output blocks into the unchanged context/distribution.
No numerical run occurred before that compile fix. Native Windows GNU
compile/link passed once; native runtime acceptance remains pending.
DIGIT / PASS; no numerical failures or extra precision checks.

## Sparse-A/dense-B native sparse output (2026-09-07)

distributed_sparse_dense_output and the affected distributed_sparse_gemm,
distributed_sparse_function_output, distributed_sparse_fold targets passed once
at WSL 1/2/4 ranks, world/parity. Exact i64 results and stored key sets cover
CCSR represented rows, explicit zeros, old-only keys with beta=0/3, multiple k
panels, square/rectangular grids, uneven/empty local pieces and zero inner extent.
The high-order case includes repeated A/C labels, both A-only and B-only sums,
off-diagonal preservation and original virtual output distribution restoration.
CCSR output converts directly to coordinates, without a full dense C or full
CSR row-pointer array. All four targets compiled/linked once on Windows GNU.
DIGIT / PASS; no failures or extra numerical computations. Native runtime
acceptance remains pending.

## Integer dense/packed random filling (2026-09-07)

integer_random passed once at WSL 1/2/4 ranks, world/parity. Exact i32/i64
checks cover dense, SY/AS/SH packed storage, virtual padding, empty local
fragments, scalar/zero extents, reversed and constant intervals. Expected values
use the source order: double sample times integer span, integer truncation,
then integer minimum. The subsequent generator value confirms every allocated
slot consumes a draw before padding cleanup. No floating tolerance is used.
Windows GNU compile/link passed once; native runtime acceptance is pending.
DIGIT / PASS; no failed checks or extra numerical runs.

## Compressed custom-function CPU/MPI contraction (2026-09-07)

The new function path shares ordinary packed symmetry/diagonal orchestration,
canonical redistribution, replicated communication and virtualized traversal.
No global gather, dense unpack fallback or replacement algorithm is used.

Eleven affected targets passed once at WSL 1/2/4 ranks, world/parity:
upstream_weigh4d, upstream_dft, distributed_symmetric_function,
distributed_packed_contraction, distributed_canonical_contraction,
distributed_symmetric_contraction, upstream_multi_tsr_sym, upstream_sy_times_ns,
upstream_ccsdt_t3_to_t2, upstream_readwrite and upstream_scalar.

weigh4d retains NS/SY/AS, n=3, rank-seeded drand48 values and the literal source
signed-denominator predicate abs(actual-expected)/expected >1e-10 when
abs(expected)>1e-10; it does not replace that predicate by absolute relative
error. DFT uses bounded n=8, SY complex DFT/inverse, the source factor 0.5,
active custom scalar reduction and strict real-component error <1e-9.
Exact integer checks cover a non-distributive polynomial, repeated input/output
labels, physical i/j/k mappings and retained off-diagonal output values/layouts.
Raw packed custom checks retain whole-buffer beta prescale and scalar execution.
All existing affected checks retain their original bounds. An initial test-only
SymmetricDistribution equality compile error was fixed by comparing its explicit
distribution and links; no numerical execution occurred before that fix.

All eleven targets compiled/linked once on native Windows GNU. Native MPI
runtime acceptance is still pending. DIGIT / PASS; no failed numerical criteria,
extra precision checks or additional diagnostic computations.

## Packed symmetric random filling (2026-09-07)

Added local SymmetricTensor::fill_random for f32/f64/Complex32/Complex64.
Every packed allocation slot consumes one MT19937-64 draw in offset order;
padding/noncanonical slots are then cleared without further draws. The exact
typed sequence and subsequent generator state passed once in symmetric_random
at WSL 1/2/4 ranks, world/parity, covering SY/AS/SH, virtual padding, empty local
fragments, scalar and zero extents. Windows GNU compile/link passed once.
DIGIT / PASS; exact discrete sequence closed without precision studies.
Integer packed filling and native runtime acceptance are not established here.

## Upstream CCSDT mapping smoke (2026-09-07)

The separate upstream_ccsdt_t3_to_t2 target passed at WSL 1/2/4 ranks,
world/parity, using bounded n=3,m=4 deterministic inputs. Both the fully AS
and explicitly partially expanded reference retain their original compressed
layouts: NS_B still has its first AS pair, as does NS_C. Indexed summation
initializes those references; two reference contractions antisymmetrize i/j.
Dot-product versus norm comparisons retain <1e-6, and the final residual
retains <=1e-6. No global gather or fully dense reference substitution is used.
The source random fixture/CLI remains distinct from this bounded equation port.
Windows GNU compile/link passed once for this target as well; no native runtime
acceptance is claimed. DIGIT / PASS; numerical verification closed.

upstream_ccsdt_map passed once at WSL 1/2/4 ranks, world/parity, with the source
n=4 and zero-initialized W/T/Z. The source six-order update Z[hijmno]+=W[hijk]*T[kmno]
executes through grid mapping and native BLAS. Exact retained zeros are checked;
neither the original driver nor this result establishes nonzero CCSDT accuracy.
The source niter option is unused. Windows GNU compile/link passed once;
native MPI execution remains pending. DIGIT / PASS; no further precision checks.

## Upstream scalar/sparse identity and baseline permutation (2026-09-07)

upstream_scalar and upstream_speye passed once at WSL 1/2/4 ranks, including
world and parity subcommunicators. Scalar uses bounded n=3 and retains source
zero-extent SY metadata, resets E before the diagonal operation, checks retained
off-diagonal SUMABS >1e-10, then full contraction SUMABS <1e-10. Scalar assignment
and subtraction retain the source inequalities and norm tolerances. Sparse identity
uses order=3, n=4 and actual indexed scalar broadcast A[iii]=1; both distinct and
repeated-index reductions satisfy the original strict absolute error <1e-9.
An initial missing Monoid import was a compile failure only and was corrected.
DIGIT / PASS; no additional numerical computations after passing. Both targets
compiled and linked on Windows GNU; native MPI runtime acceptance remains pending.

Separately, one unmodified pinned C++ permute_multiworld diagnostic at n=3/rank=1
passed NS then failed the SY expected-copy assertion. AS/SH were not reached.
This is recorded in upstream-known-failures.md, not counted as a Rust pass or
used to relax any threshold. The reference library remains development-only.

## Mixed-storage coordinate permutation (2026-09-07)

`symmetric_permuted_io` and `sparse_permuted_io` passed once at 1/2/4 WSL ranks,
world/parity. Exact i64 checks cover all thirteen newly implemented packed/sparse
combinations: signed full-orbit gather, canonical scatter, implicit versus stored
zeros, masked/reversed coordinates, beta-once, root-only/reordered children and
replicas. No numerical failures or extra precision runs. DIGIT / PASS. The source
multiworld driver's compressed expected-copy assertions remain a separate open
reconciliation item; these tests do not mislabel them as passing. Windows native
MPI runtime acceptance remains pending.
Both mixed-storage targets compiled and linked once on Windows GNU.

## Coordinate multiworld permutation and indexed-write order (2026-09-07)

`upstream_permute_multiworld`, `indexed_write_order`, `dense_semantics`,
`distributed_symmetric_operations`, `binary_io` and `upstream_readwrite` passed
once at 1/2/4 WSL ranks. The new multiworld NS driver uses exact reads and source
abs<1e-9 writes, nonuniform/empty blocks, skipped maps and dense-zero scatter
omission. Exact noncommutative matrix fixtures prove the corrected left-sided
indexed-write coefficients, duplicate beta-once and AS signs. Existing affected
numeric checks retain their original bounds. DIGIT / PASS; no diagnostic runs.
Windows MPI runtime acceptance remains pending.
All six affected integration targets compiled and linked once on Windows GNU.

## Upstream read/write, sparse sum and subworld GEMM drivers (2026-09-07)

`upstream_readall`, `upstream_readwrite`, `upstream_sptensor_sum` and
`upstream_subworld_gemm` passed once at 1/2/4 WSL ranks, world/parity. The first
two retain source abs<=1e-10, sparse sum abs<=1e-9, and subworld GEMM Frobenius
error<1e-9. Random fixtures use the source POSIX 48-bit recurrence in Rust.
Read/write uses real distributed NS and compressed self-contractions, not
all-gathered serial arithmetic. Its source `shape_AS4` is literally SH, preserved
and documented rather than falsely counted as a new AS case. Sparse sum keeps
the complete original key/value fixture at n2. GEMM keeps default m17/n23/k31
plus tiny shards and divisor1/2/4 cases. DIGIT / PASS; no diagnostics or precision
reruns. Windows native runtime remains pending; compile/link is separate evidence.
All four driver targets compiled and linked once on Windows GNU.

## Sparse virtual execution (2026-09-07)

Four exact `sparse_virtual` unit tests passed once: block order, beta-first
tracking, repeated output diagonals, scalar and changing sparse output buckets.
`distributed_sparse_general` and `distributed_sparse_function` passed once at
1/2/4 WSL ranks, world/parity, with explicit no-virtual and multi-label virtual
factors. Exact i64 checks include contracted/output-only virtual indices, empty
sparse blocks, stored zeros and non-distributive custom functions. DIGIT / PASS;
no numerical diagnostic runs. Windows native MPI execution remains pending.
The library and both sparse integration targets compiled/linked once on Windows GNU.

## Value-only subworld streams (2026-09-07)

`subworld_transfer`, `symmetric_subworld`, `typed_distributed_eigh` and `schedule`
passed once at 1/2/4 WSL ranks. Dense tests now also cover scalar and zero-length
global domains. Existing exact noncommutative matrix/i64, bounded complex and
four-type eigendecomposition quantities are unchanged. Coverage includes reversed
odd/even children, virtual/full replicas, canonical packed SY/AS/SH streams and
empty messages. DIGIT / PASS; no diagnostics or precision reruns. Windows native
MPI execution remains pending; this stage performs compile/link acceptance only.
All four integration targets compiled and linked once on Windows GNU.

## Compressed subworld accumulation (2026-09-07)

`symmetric_subworld` passed once at 1/2/4 WSL ranks, world/parity, after fixing a
test-only moved coefficient before execution. Exact integer/complex checks cover
SY/AS/SH, alpha/beta updates in both directions, unchanged source storage,
reversed noncontiguous child ranks, physical/virtual/full replicas, empty regions
and one-element edge lengths. DIGIT / PASS; no numerical diagnostics. Windows
GNU compiled/linked the target once; native MPI runtime acceptance remains pending.

## Packed value-only reshuffle (2026-09-07)

Three exact canonical-stream unit tests and `symmetric_reshuffle`,
`distributed_symmetric_contraction`, `distributed_symmetric_repack` passed once
at 1/2/4 WSL ranks, world/parity. Exact integer/complex fixtures cover SY/AS/SH,
higher-order and mixed groups, physical/virtual/full/mixed-replica switches,
padding, packed holes, empty/scalar domains and unchanged source storage.
Contraction retained its existing 1e-6 absolute bound. DIGIT / PASS; no numerical
diagnostics or precision reruns. Windows native runtime acceptance is pending.
The library and all three integration targets compiled/linked once on Windows GNU.

## Boolean infinity norm (2026-09-07)

`bool_norm` passed once at 1/2/4 WSL ranks, world/parity. Exact checks cover
dense/sparse true and explicit-false records, replicas, empty local shards and
empty global tensors. The acceptance values are the pinned same-type MAXABS
results 0.0/1.0. Source audit distinguishes unsafe bool norm1 from this supported
operation; see `source-runtime-boundaries.md`. DIGIT / PASS.
Windows GNU compiled/linked `bool_norm` once; MPI runtime remains pending.

## Value-only dense cyclic reshuffle (2026-09-07)

Three exact offset-stream unit tests passed once. `cyclic_reshuffle`,
`dense_low_memory`, `typed_distributed_svd` and `schedule` passed once at 1/2/4
WSL ranks, including subcontexts. New exact tests cover i8/bool/Complex64 and
non-Copy custom three-byte Wire values, changing physical/virtual mappings,
full/mixed replicas, padding, empty dimensions and scalar tensors. Affected
low-memory and four-type SVD checks retained their existing bounds. DIGIT / PASS;
no diagnostic runs. Native compile/link results are recorded with the stage;
Windows MPI runtime execution remains pending.

## Distributed schedule execution (2026-09-07)

Seven schedule graph/partition unit tests passed once. `schedule` passed once
at 1/2/4 WSL ranks, world/parity, with exact i64 sums and contractions. Coverage
includes proper child communicator execution, RAW/WAR/WAW and in-place updates,
no-input roots, subsecond cost allocation, deterministic replay, restoration of
parent distributions and finite nonnegative timing fields. DIGIT / PASS; no
diagnostic runs. Windows GNU compiled/linked the library and schedule target once;
native MPI runtime execution remains pending. This does not close the full port
or automatic operation-cost integration backlog.

## Binary tensor MPI-IO (2026-09-07)

`binary_io` passed once at 1/2/4 WSL ranks on world/parity contexts. Exact byte
and value checks cover i8/i16/i32/i64/f32/f64/Complex32/Complex64, dense and sparse
storage, virtual and replicated layouts, SY/AS/SH expansion and canonical-only
overwrite, nonzero byte offsets, preserved prefix/suffix, empty local chunks,
empty global dimensions and scalar tensors. DIGIT / PASS; no diagnostics or
extra precision runs. Windows GNU compiled/linked the target once. Native MPI
execution remains pending because the runtime is not installed.

## Explicit all-rank extraction (2026-09-07)

`pair_read` passed once at 1/2/4 WSL ranks, world and parity subcommunicators.
Exact i64/Complex64 fixtures cover sorted dense pairs/data, virtual and replicated
layouts, sparse stored zeros versus implicit zeros, SY/AS/SH packed and expanded
results, nonzero-only ignoring symmetry unpack, empty local shards, empty global
dimensions and all-empty Allgatherv payloads. Sparse text roundtrip preserves
explicit zero records. DIGIT / PASS; no numerical diagnostic runs.
Windows GNU compiled and linked this target once. `msmpi.dll` and `mpiexec`
remain unavailable, so native MPI runtime acceptance remains pending.

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

## Raw distributed folded execution (2026-09-07)

`dense_folded_execution`, `dense_execution`, and `dense_execution_algebra`
passed at WSL 1/2/4 ranks, including world and parity subcommunicators.
New coverage connects actual folded SearchCache selections to BLAS with fresh
values on reuse, six normal mapping permutations, partial residual indices,
virtual batches, tiny padded fragments and nested input/output panel levels.
Output distributions and cache hit/miss counts are exact; finite f64 results
retain abs(error)<1e-6. The four affected local partial_fold_kernel tests passed.
An initial Rust slice-iterator compilation error was fixed before numerical
execution; there were no numerical failures or extra precision runs.
All four targets compiled and linked on Windows GNU once. Native MPI runtime
acceptance remains outstanding. DIGIT / PASS for this bounded WSL change only.

## Dense node-aware remapping (2026-09-07)

distributed_node_fold and the affected dense_folded_execution passed WSL
1/2/4 ranks with world/parity contexts. Node tests check six raw GEMM mappings,
packed block exchanges, output backmapping, exact non-involutive rank maps,
and strict source-volume selection (32 to 8), no-improvement and ppn=1 cases.
Finite numerical results retain abs(error)<1e-6; no tolerance changes.
The initial selection fixture incorrectly expected communication on a wholly
unused axis; its first correction used a physically mapped output-only label
rejected by source preflight. Replacing it with a valid vector-matrix fixture
resolved both test-construction errors. Only the affected four-rank target
was rerun after those concrete fixture fixes; unchanged passing runs stayed closed.
Both targets compiled/linked on Windows GNU; the corrected node target was
recompiled after its fixture edits. Native MPI execution remains outstanding.
Logical ppn=2 on one WSL host tests permutation semantics, not network speed.

## Fractional node peer accounting (2026-09-07)

node_peer_counts passed exact source cases including [3,4]/ppn4 -> [0.5,2],
fractional original volume 72, reordered volume 32 and strict no-improvement.
The affected plan_cost, grid_plan_cost, mapped_cost, folded_cost and redist_cost
targets passed once. dense_search, dense_execution, dense_folded_execution and
distributed_node_fold passed once at WSL 1/2/4 ranks, world/parity. Existing
exact model/discrete checks and finite abs(error)<1e-6 tensor checks are unchanged.
All ten targets compiled and linked once on Windows GNU; MPI native runtime
acceptance is still outstanding. No failures or extra precision studies.
DIGIT / PASS for this change; whole-project acceptance remains incomplete.

## Dense folded low-memory execution (2026-09-07)

dense_low_memory, dense_folded_execution and distributed_node_fold passed once
at WSL 1/2/4 ranks, including world/parity contexts. Low-memory coverage checks
exact A/B data and A/B/C distribution restoration, six normal raw mappings,
partial folded residual axes, nonuniform/padded fragments, node rank backmapping
and weighted SearchCache reuse with fresh values. Output remains finite with
abs(error)<1e-6. No failures or further numerical verification. All three tests
and the benchmark compiled/linked on Windows GNU; native MPI runtime is pending.

One requested representative release measurement used mpi_low_memory_bench,
OpenBLAS threads=1 and four WSL ranks: m=128,k=192,n=160, contraction elapsed
0.010644 s (planning excluded; redistribution, packing and input/output restore
included), source estimated memory 397312 bytes. Per-process lifetime peak RSS
reported by `/usr/bin/time` was 27216,27268,27548,27552 KiB (maximum 27552 KiB).
RSS includes MPI/runtime and planning, is not tensor workspace alone, and is not
the source memory estimator. This single sample makes no speedup or memory-ratio
claim. DIGIT / PASS for the acceptance set; benchmark was not a precision study.

## Generic semiring low-memory execution (2026-09-07)

dense_execution and the extended dense_execution_algebra passed once at WSL
1/2/4 ranks, world/parity. Both immutable-home and mutable-low-memory modes
cover all six raw normal mappings and scalar coefficient-side cases, with exact
input restoration and layout checks. Noncommutative 2x2 integer matrix products
and i64 results are exact; f32/complex checks retain their existing 1e-6 bounds.
Both targets compiled/linked once on Windows GNU. No failures, tolerance changes
or extra numerical runs. DIGIT / PASS; native MPI runtime and overall port
acceptance remain outstanding.

## Generic node-aware Wire exchange (2026-09-07)

dense_execution and dense_execution_algebra passed once at WSL 1/2/4 ranks,
world/parity, after adding optional node ordering to both generic ownership
modes. Four-rank custom non-Copy matrix values, i64, f32 and complex values
exercise serialized forward/back exchanges on all six raw normal mappings.
Input data/layout restoration and discrete results are exact; existing floating
1e-6 checks and scalar coefficient-side cases are unchanged. No failures or
additional precision checks. Both targets compiled/linked on Windows GNU;
native MPI execution remains pending. DIGIT / PASS for this change.

## Four native BLAS scalar types (2026-09-07)

scalar_blas passed native SGEMM/DGEMM/CGEMM/ZGEMM cases: N/T combinations,
padded leading dimensions with exact untouched padding, zero reduction length,
complex coefficients and typed partial-fold residual traversal. Every floating
component must be finite and within the existing absolute 1e-6 bound.
Affected local_linalg, folded_contraction, partial_fold_kernel, folding and
fold_selection targets passed once. dense_folded_execution, dense_low_memory
and distributed_node_fold passed once at WSL 1/2/4 ranks, world/parity.
Initial compilation caught two missing Monoid imports and one renamed call in
tensor.rs; these were fixed before numerical execution. There were no numerical
failures or added precision runs. All nine targets compiled/linked on Windows
GNU. Native runtime remains pending. DIGIT / PASS for this scalar-kernel stage,
not a claim of typed distributed folded or f32/complex LAPACK completion.

## Typed distributed folded orchestration (2026-09-07)

typed_folded_execution passed once at WSL 1/2/4 ranks, world/parity, for f32,
f64, complex-f32 and complex-f64 native BLAS execution. Cases cover six raw
normal mappings, partial residual dimensions, nonuniform/padded shards,
node ordering, immutable/home and mutable/low-memory paths, and weighted
SearchCache reuse at each scalar's actual Wire width. Input data and layouts
are exact; all output components are finite with abs(error)<1e-6.
The affected dense_folded_execution, dense_low_memory and distributed_node_fold
targets passed at the same rank counts. All four targets compiled/linked on
Windows GNU once. No failures or further precision runs. DIGIT / PASS;
native MPI runtime and typed decomposition coverage remain incomplete.

## Four-type distributed positive-definite and triangular factors (2026-09-07)

typed_matrix_factors, distributed_matrix and distributed_spd passed at WSL
1/2/4 ranks, world/parity. Native S/D/C/Z POTRF, POSV and TRSM are covered,
including n=1 empty local rows, n=5 uneven partition, identity padding/virtual
columns, upper/lower factors, left/right solves and plain N/T operation.
Complex inputs are genuinely Hermitian rather than real-only fixtures.
Cholesky is checked by factor reconstruction, solutions by A*X or X*T residuals;
the existing upstream L1 absolute-or-relative 1e-3 and triangle 1e-6 criteria
are retained, with finite results. Original output distributions are exact.
A missing Group import in the new test was corrected before numerical execution.
No numerical failures or further precision runs. All three targets compiled and
linked on Windows GNU once; native MPI runtime remains pending. DIGIT / PASS.
Typed QR/SVD/eigenvalue decomposition and whole-port acceptance remain unfinished.

## Four-type distributed thin QR (2026-09-07)

typed_distributed_qr passed once at WSL 1/2/4 ranks, world/parity, with all four
native scalar types and tall 13x7, wide 5x8, and scalar 1x1 cases. Inputs remain
exactly unchanged. Finite QR reconstruction uses the existing Frobenius bound
m*n*n*1e-6; finite Q^H Q-I uses m*n*1e-6. Complex tests use genuine imaginary
entries and conjugate adjoints. Factor elements are not directly compared.
There were no failures or additional precision runs. The new target and the
existing combined distributed_qr_svd target compiled/linked on Windows GNU;
unchanged SVD numerical checks were not rerun. Native MPI runtime is pending.
DIGIT / PASS for typed QR; typed SVD/eigh and full-port acceptance remain open.

## Four-type distributed thin SVD (2026-09-07)

typed_distributed_svd passed once at WSL 1/2/4 ranks, world/parity. All four
native GESVD types cover 13x7, 5x8, 1x1 and 1x3 matrices, including zero local
shards and one-row native grid adjustment/restoration. Inputs are exact;
singular values are finite, nonnegative, descending and have zero imaginary
part. Matrix factors remain distributed; only the singular-value vector is read.
Finite reconstruction uses m*n*n*1e-6 and U/V orthogonality uses m*n*1e-6,
the existing source Frobenius criteria. No failures or extra precision checks.
The new target and existing distributed_qr_svd/distributed_svd_paths compiled
and linked once on Windows GNU. Native MPI runtime is pending. DIGIT / PASS;
typed truncated/randomized SVD, eigensolve and full-port acceptance remain open.

## Typed SVD truncation semantics (2026-09-07)

typed_svd_truncation passed once at WSL 1/2/4 ranks, world/parity, for all four
native scalar types. Exact dimensions cover rank-only, inclusive threshold,
combined rank/threshold, rank-zero, oversized rank and above-spectrum threshold
branches, retaining the source zero-rank/full-factor quirk. A fixed threshold
3+1e-8 checks source f32 conversion versus f64 comparison; it is a discrete
branch test, not an accuracy refinement. Distributed sliced reconstruction uses
the existing m*n*n*1e-6 Frobenius bound and finite results. No failures or extra
precision runs. New target and existing distributed_svd_paths compiled/linked
on Windows GNU once. Native MPI runtime remains pending. DIGIT / PASS.

## Randomized-SVD input/output guess (2026-09-07)

randomized_guess passed once at WSL 1/2/4 ranks, world/parity. A supplied
nonorthogonal guess is exactly unchanged for zero iterations. After one
iteration its full oversampled 5x3 shape is retained and its column Gram has
Frobenius residual <=5*3*1e-6, while returned factors have the requested rank 2.
This specifically checks the source in/out side effect, not an approximation
claim for the deliberately nonorthogonal zero-iteration fixture. No failures
or extra numerical studies. New target and updated distributed_svd_paths caller
compiled/linked on Windows GNU once; native MPI runtime remains pending.
DIGIT / PASS for the guess contract; random-generator fidelity remains open.

## Compressed unpack, norms and coordinate text I/O (2026-09-07)

symmetric_norms and symmetric_text_io passed once at WSL 1/2/4 ranks,
world/parity. Distributed unpack checks exact SY/AS/SH values and signs on
virtual/padded layouts without a root gather. Real norms cover i8/i16/i32/i64/
f32/f64; complex norm2 covers both precisions. All-NS storage keeps manual f64
accumulation, while compressed norms expand, square and sum in the original
scalar precision before the source final conversion/sqrt. Existing finite
abs<1e-6 bounds pass, including hollow singleton/empty local cases.

Four-type symmetric text I/O exports only canonical nonzero pairs, matching
get_local_pairs(nonzeros_only=true, unpack_sym=false). Exact tests cover packed
round trips, reverse indices, signed accumulation of equivalent input permutations
and reading the packed export into an ordinary dense tensor. Temporary files
were removed. Both targets compiled and linked on Windows GNU once. No failures
or precision studies. DIGIT / PASS; bool norm contracts, native MPI execution
and full-port acceptance remain outstanding.

## Communicator-scoped sparse text MPI-IO (2026-09-07)

Five sparse_text codec unit tests passed for coordinate order, absent values,
reversed indices, six-decimal real formatting and typed integer parsing.
sparse_text_io passed once at WSL 1/2/4 ranks, world/parity subcontexts, for
f32/f64/i32/i64 dense and sparse tensors. Exact fixtures cover additive reads,
duplicate keys, no-value reads/writes, reversed coordinates, overwrite of longer
existing files, virtual layouts, tiny files shorter than the source overlap,
and empty files. Temporary test files were removed by the successful tests.

The first compile exposed a missing Monoid import after removal of forwarding
helpers; the import was fixed before any numerical execution. No numerical
failures or precision studies occurred. Library tests and the integration target
compiled/linked on Windows GNU once. MPI-IO uses the supplied communicator,
clamped EOF reads and typed value parsing rather than reproducing the source's
MPI_COMM_WORLD and scanf pointer bugs. DIGIT / PASS; native MPI execution,
compressed-symmetry text export and full-port acceptance remain outstanding.

## Generic cross-world accumulation and eigensolver integration (2026-09-07)

subworld_transfer passed once at WSL 1/2/4 ranks, world/parity. Both directions
preserve incoming*alpha + old*beta with exact i64/noncommutative matrix results
and finite complex componentwise abs<1e-6. Tests cover reversed odd/even child
membership, inactive parent participants, cyclic/virtual/replicated distributions,
empty local shards, unchanged inputs and explicit child-context closure.

The four-type eigensolver now uses the reusable transfer APIs for input and both
outputs instead of separate hand-written exchanges. typed_distributed_eigh
passed once at WSL 1/2/4 with unchanged n*n*1e-6 reconstruction/orthogonality
criteria, including degenerate spectra. Child tensors are released before
explicit context closure on success and native errors. An unused Wire import
reported by compilation was removed; passing numerical checks were not repeated.
Three corresponding/affected targets compiled and linked on Windows GNU once.
The current host still lacks C:/Windows/System32/msmpi.dll, so native execution
remains unaccepted. DIGIT / PASS for this stage; optimized cyclic-reshuffle
buffers, sparse/compressed cross-world paths and full-port completion remain open.

## Ownership storage conversion and sparse random fill (2026-09-07)

storage_conversion and sparse_random_fill passed once at WSL 1/2/4 ranks,
world/parity, with exact keys, values, RNG progression and filter call order.
into_sparse consumes dense values, evaluates the predicate over primary-layer
storage including padding, then discards padding and orders retained virtual
blocks. Nonzero, signed/absolute strict thresholds and zero-retaining predicates
are covered. into_dense is explicitly collective and writes canonical source
pairs to restore all mapped dense replicas; no communicating destructor exists.

Sparse-pattern random fill supports both dense and sparse storage for all seven
pinned scalar families. Tests cover the literal exponential candidate count,
distributed candidate-key union, duplicate coalescing before value draws,
pre-scaling casts, retained sampled zeros, reverse bool bounds and zero-density
clearing. Dense value sampling includes unselected valid entries, but not padding;
sparse sampling visits only stored post-dedup entries. No statistical density or
extra precision study. Both targets compiled and linked on Windows GNU once.
DIGIT / PASS; native MPI runtime and full-port acceptance remain outstanding.

## Dense/sparse source norms and narrow scalar algebra (2026-09-07)

tensor_norms passed once at WSL 1/2/4 ranks, world/parity: norm1/norm_infty/
norm2 for i8/i16/i32/i64/f32/f64, and norm2 for bool and both complex precisions.
Finite analytic results satisfy abs<1e-6; inputs remain exact. The pinned NS
manual_norm2 accumulates every local storage slot or stored sparse pair in f64,
including replicas, rather than selecting logical owners. That source behavior
is tested explicitly on replicated layouts; norm1 and norm_infty use logical
canonical-owner reductions. Complex magnitude is formed in source precision
before conversion to f64. This is not the unported compressed-symmetry norm2
branch, which expands and squares in the tensor scalar precision.

narrow_algebra passed exact i8/i16 promoted-then-narrowed arithmetic and Wire
encoding, and Boolean OR-add/AND-multiply. Both targets compiled and linked on
Windows GNU once. No failures or precision studies. DIGIT / PASS for this stage.
Bool norm1/norm_infty remain open pending resolution of the pinned cross-algebra
Term::operator double path; compressed norms and native MPI runtime remain open.

## TTTP budget-selected auxiliary blocking (2026-09-07)

Matrix TTTP now takes TttpBlocking::Divisions or AvailableBytes. The latter
uses the pinned per-mode integer-truncated two-factor-buffer estimate, adds
the local pair accumulator only beyond one division, doubles divisions capped
at k, and collectively selects their maximum. Available bytes are supplied by
the caller per rank; this is not OS-memory discovery or a process peak cap.
Dense accumulator allocation now counts only valid local entries, not padding.

Five exact tttp_blocking unit tests passed once: term-wise integer truncation,
doubling/cap, accumulator accounting, equality boundary and insufficient memory.
tttp_memory and the affected typed_multilinear target passed once at WSL 1/2/4
ranks, world/parity. Rank zero forces four blocks while others admit one;
dense/sparse results remain exact on uneven virtual layouts, both factor
orientations and stored sparse zeros. The generic regression retains finite
abs<1e-6 for all four floating types and exact integer/matrix-semiring results.
The library unit-test target and four corresponding integration targets compiled
and linked on Windows GNU. No failures or additional precision studies.
DIGIT / PASS; native MPI execution, OS/process memory accounting, vector TTTP's
source memory diagnostic and full-port completion remain outstanding.

## Generic semiring dense/sparse TTTP and MTTKRP (2026-09-07)

typed_multilinear passed once at WSL 1/2/4 ranks, world/parity subcontexts.
The same dense and sparse production paths cover f32/f64/Complex<f32>/
Complex<f64>, exact i64 and a non-Copy noncommutative 2x2 integer matrix
semiring. Floating components are finite with abs<1e-6; discrete results and
stored sparse keys are exact. Vector TTTP skips a mode; matrix TTTP uses both
factor orientations and uneven 5-column/2-block auxiliary partitions. MTTKRP
checks every output mode with vector and matrix factors. Input tensor layouts
include virtual blocks, uneven physical shards and a replicated layer at four
ranks. Factor data and TTTP output distributions are preserved.

Arithmetic, factor buffers, mode-fiber broadcasts and final reductions now use
the supplied algebra/Wire types. Source multiplication order and fiber reuse
remain unchanged; there is no complex conjugation or global tensor gather.
Five corresponding/affected targets compiled and linked on Windows GNU once.
No failures or extra precision checks. DIGIT / PASS; automatic auxiliary memory
selection, remaining multilinear routines, native MPI runtime and full-port
acceptance remain incomplete.

## Four-type distributed eigensolve and indexed tensor SVD (2026-09-07)

typed_distributed_eigh and typed_tensor_svd passed once at WSL 1/2/4 ranks,
world/parity subcontexts. Real symmetric and complex Hermitian eigensolves use
the largest-square-grid subworld, restore the original vector distribution and
retain real eigenvalues (zero imaginary component for complex tensor scalars).
Fixtures include n=5 indefinite and degenerate spectra and n=1/empty local
shards. Reconstruction and orthogonality Frobenius norms satisfy n*n*1e-6;
input data are exactly unchanged. No eigenvector entry/phase comparisons.

The complex native HEEVX binding deliberately passes the queried real LRWORK
for its separately allocated RWORK. The pinned C++ pheevx wrapper incorrectly
forwards complex LWORK in that position, unlike its caller's separate buffer
allocation. The Rust binding does not reproduce that unsafe size mismatch.

Four-type indexed tensor SVD exercises noncanonical auxiliary/output index
orders, full truncated SVD with complex inputs and randomized rank-one real
fixtures. Matrix intermediates stay distributed. Normalized reconstruction
is <1e-6; orthogonality retains the existing L1 <=1e-3 or relative <=1e-3 rule.
Four corresponding/affected targets compiled and linked on Windows GNU once.
No failures or precision studies. DIGIT / PASS for this stage; native MPI
runtime and full-port acceptance remain outstanding.

## Four-type explicit-grid BLAS and randomized SVD (2026-09-07)

typed_grid_blas and typed_randomized_svd passed once at WSL 1/2/4 ranks,
world/parity subcontexts, for f32/f64/Complex<f32>/Complex<f64>. Explicit-grid
GEMM and reordered batched folding cover rectangular and square grids, uneven
dimensions and empty local shards. Inputs/layouts are exact and outputs finite
with componentwise abs<1e-6. The five existing folding/fold_selection local tests
also passed once after updating the generic Plan::execute<T,K> calls.

Randomized SVD preserves the fixed source's plain transpose, including complex
inputs, real-only automatic guesses and full oversampled in/out guess writeback
before rank cropping. The automatic rank-two fixture reconstructs A; the complex
supplied-guess case reconstructs the literal source projection Q_r*(Q_r^T*A),
not a silently substituted Hermitian projection. Both use the existing source
Frobenius reconstruction bound m*n*n*1e-6. No extra precision studies or failures.

Eight corresponding/affected test targets compiled and linked on Windows GNU;
an initial command used the nonexistent target tensor_svd and was corrected to
distributed_tensor_svd before compilation. Native MPI runtime acceptance remains
pending. DIGIT / PASS for this stage, not whole-port completion.

## MT19937-64 and typed dense random fill (2026-09-07)

random_generator passed exact u64 vectors for seeds 0,1,5489 across positions
0,1,311,312,623,624 (two twist boundaries), plus exact source interval conversion.
Vectors came from one development-only std::mt19937_64 oracle execution, matching
the pinned engine parameters; no C++ file or executable is a project dependency.
distributed_random_fill passed WSL 1/2/4 ranks, world/parity: all four scalar
fills and padding draw consumption are exact. It also verifies the changed
f64 randomized-SVD auto-guess path on a rank-two matrix using the existing
5*4*4*1e-6 reconstruction bound and finite results. No failures or extra studies.
Both targets compiled/linked on Windows GNU once. Native MPI runtime remains
pending. DIGIT / PASS for this change; whole-port acceptance remains incomplete.

## Normal mapping search (2026-09-07)

normal_mapping: two exact local tests passed for explicit 2D paired maps,
retained layouts, all six common-index permutations and source physical-map
rejections. distributed_normal_mapping passed WSL ranks 1/2/4 with world/parity
communicators: old-layout subsets, fresh choices, source traversal/IDs and unique
rank ownership. The initial rank-1 run exposed premature Distribution validation
of a raw rejected candidate (duplicate physical axis). Raw construction now
preserves source preflight ordering; only this failed target was rerun, followed
by previously unrun ranks 2/4. No floating-point tolerance study was involved.
Both targets compiled/linked on Windows GNU; MPI runtime execution remains
pending the missing MS-MPI runtime, not a claimed native pass.

## Selected map reconstruction and raw 2D cost trees (2026-09-07)

selected_mapping passed once with WSL 1/2/4 ranks, world/parity. Every rank
independently reconstructed every accepted normal/exhaustive ID announced by its
owning rank; topology, shapes and complete mapping chains compared exactly.
mapped_cost passed two local tests with source-derived fixed integer/formula
oracles: 2x2 input-moving GEMM, output-moving custom reduction, and 2x3 LCM
virtualized GEMM. Unit model coefficients isolate local work and communicated
bytes; these are model checks, not measured seconds. The 2x2 inner estimate is
208 model units/128 workspace bytes and dense redistribution total is 560 model
units/272 bytes. No failures or extra precision checks. Both new targets compiled
and linked once on Windows GNU; native MPI execution remains pending.

## Collective dense-unfolded search (2026-09-07)

dense_search passed once at WSL 1/2/4 ranks, world/parity. A test-only serial
reference collects the rank-partitioned candidate stream and checks exact
selected namespace/ID/time/memory against production winner-only communication.
Cases cover time-only and weighted two-pass selection, optional exhaustive
refinement, source 0.01 cutoff with small synthetic coefficients, and rejection
when the strict memory limit admits no candidate. Returned layouts also pass
mapping preflight. All comparisons passed without diagnostic reruns or precision
studies. Windows GNU compilation/linking passed once; native MPI runtime remains
pending. These checks validate selection/model logic, not contraction execution
or performance speedup.

## Raw dense execution and search cache (2026-09-07)

dense_execution passed WSL 1/2/4 ranks with world/parity communicators. It checks
all six normal mapping permutations, input/output-moving raw variants, nested
panel levels including a rectangular physical pair, virtual batch traversal,
padding/empty true fragments, original output layout, and finite reconstruction
values at abs(error)<1e-6. Cached search reuses a plan with changed input values
and alpha/beta, recognizes index alpha-renaming, misses after input redistribution,
and re-searches after clear; hit/miss counts are exact.

Rank 1 passed initially. Rank 2 first rejected the manually constructed nested
test layout: its [2,1] physical pair lacked B's required virtual factor 2. The test
mapping was corrected without changing execution or tolerance, rank 2 then passed,
and previously unrun rank 4 passed. Rank 1 and old suites were not rerun. Windows
GNU compilation/linking passed once; MS-MPI runtime acceptance remains pending.

## Generic raw dense panel execution (2026-09-07)

The changed shared executor was verified with dense_execution_algebra,
dense_execution, ctr_2d and tensor_gemm at WSL 1/2/4 ranks. All passed. The new
test covers a noncommuting 2x2 matrix algebra and i64 exactly, f32 and Complex<f64>
at abs(error)<1e-6, all normal operand permutations, and scalar beta-side behavior
with an unused physical axis versus topology order zero. Existing core callers
cover nested/strided panels, cache execution, and BLAS GEMM after the API change.
An initial compilation error was a missing Clone bound on the private generic
panel operand helper; it was fixed before any test ran. No numerical failures or
post-pass precision checks occurred. All four targets compiled/linked on Windows
GNU; native MPI runtime execution remains pending.

## Dense fold permutation selection (2026-09-07)

fold_selection passed two local tests covering all six source transpose layouts
with actual BLAS execution, last-tie selection, the first-three restriction, and
batch-first layouts with virtual multiplicities [2,3,5]. The latter selects
permutation 5 with exact modeled per-original-operand costs [96,120,150], proving
that the permuted third operand (original A) is doubled. Output uses the fixed
finite abs(error)<1e-6 acceptance. Existing folding tests passed once and affected
tensor_blas_fold passed WSL 1/2/4 ranks. No failures or extra precision studies.
All three targets compiled/linked on Windows GNU; native runtime remains pending.

## Partial/symmetry fold metadata (2026-09-07)

fold_indices and fold_layout passed four local exact tests in WSL. Cases cover
partial NS folds, sparse/custom/repeated-index decisions, SY/AS/SH matching and
reversed-group rejection, common three-operand groups, compressed group lengths,
fold-list index positions, stable residual order, selected-prefix permutation and
scalar metadata. An eligible partial SY contraction also feeds FoldLayout.
No floating-point computation or MPI execution is introduced by these modules,
so no old MPI/numerical suite was rerun. Both targets compiled/linked once on
Windows GNU. Full partial/symmetric folded execution remains unaccepted.

## Partial fold descriptors and packed storage conversion (2026-09-07)

partial_fold and fold_storage passed five exact local WSL tests. Forward and
backward conversions cover independent virtual blocks, packed SY/AS groups,
non-Copy String elements, scalar and zero-size storage. Partial NS selection
includes the residual x dimension of xik/kj/ij in its source transpose cost
(permutation1, [24,0,0]) and verifies the resulting storage offsets. SY/AS/SH
contracted pairs produce packed k lengths 6/3/3 with unchanged zero-cost layouts.
That AS/SH oracle was subsequently found to use logical packed_size rather than
the source fold storage's sy_packed_size; the correction is recorded below.
No failures, floating-point studies or old MPI reruns. Both targets compiled and
linked on Windows GNU. These are local fold/storage stages, not proof of complete
partial-folded contraction execution.

## Local partial-fold BLAS kernel and source storage correction (2026-09-07)

partial_fold_kernel passed four local WSL tests: singleton residual dimensions,
shared residual SY coordinates, residual SY output, folded batches, packed
SY/AS/SH contraction and beta=0 with NaN prior C. Output must be finite and within
abs(error)<1e-6 of the fixed canonical-inner reference. Full symmetry multiplicity
is deliberately not claimed by an inner-kernel test.

Source sy_packed_size inspection exposed and corrected the earlier AS/SH storage
capacity oracle: a linked 3x3 fold group occupies 6 local slots for all three
kinds, with structural diagonal holes for AS/SH. Affected fold_layout,
fold_storage and partial_fold tests passed once after correction, with exact
layout/offset checks. No other passing numerical suite was rerun. All four targets
compiled/linked on Windows GNU. Distributed partial-fold execution and native MPI
runtime acceptance remain pending.

## Raw dense folded cost estimates (2026-09-07)

folded_cost passed two local exact model tests in WSL. Raw 2x2 GEMM retains its
panel tree and inner cost/workspace (208 model units/128 bytes), adds 128 fold
buffer bytes, and yields total 560 model units/336 memory bytes after original
cyclic redistribution. Partial NS residual work and transpose costs are included.
A two-batch case explicitly checks the source model's omitted l multiplier while
retaining full 752-byte fold residency; these figures are not measured timings or
RSS. Scalar fold ineligibility returns None. No failures, precision studies or
old test reruns. Windows GNU compilation/linking passed; distributed folded
execution and native MPI runtime acceptance remain pending.

## Folded cost integration into collective search (2026-09-07)

dense_search passed WSL 1/2/4 ranks, world/parity, after integration. The serial
reference now verifies folded and unfolded candidate winners for time-only and
weighted searches, with/without exhaustive refinement, the source 0.01 cutoff,
strict memory rejection, dense-custom and scalar no-fold cases. Returned fold
descriptors and cache reuse across index alpha-renaming are checked exactly.
No failures or precision studies. Windows GNU compiled/linked dense_search and
the API-updated dense_execution target; unchanged execution tests were not rerun.
The selected folded descriptor is available, but distributed folded execution is
not claimed by this selection test.

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

## Dense redistribution and unfolded candidate cost (2026-09-07)

redist_cost passed once in WSL: no-op maps, equal-phase physical-axis permutation,
and synthetic-coefficient unfolded estimates for 2/4-process topologies. Exact
source cost results are 520/636, input residency 192/96 bytes, temporary memory
408/252 bytes, total source memory 600/348 bytes. These isolate formulas, not
measured wall times or RSS. DIGIT / PASS; no failed checks or extra numerical runs.
The test compiled and linked on Windows GNU. No MPI execution was required for
this pure model stage; full automatic planning remains incomplete.

## Exhaustive raw mappings and preflight (2026-09-07)

mapping_variants and mapping_preflight passed once in WSL (four local tests).
Exact checks cover the six GEMM variants on a 2x2 topology, source choice-zero
duplication, all three 2D orientations including the empty AC case, rectangular
2x3 shared-phase LCM=6, legal 2D mismatches, phase mismatch rejection, three-way
map equality and singleton rules. No MPI execution or floating tolerance was
needed. Both targets compiled/linked on Windows GNU. DIGIT / PASS; automatic
candidate enumeration/selection/execution integration remains incomplete.

## Canonical topology and distributed exhaustive IDs (2026-09-07)

Two topology_canonicalization checks passed once: folded physical-pair reordering
and conflict rejection without candidate mutation. distributed_exhaustive_mapping
passed at 1/2 ranks initially; four ranks exposed unsigned intermediate underflow
in the prior get_choice port's dimension-group+1 expression. Reordering it as
dimension+1-group preserves the source signed arithmetic result. Only the affected
four-rank run was repeated and passed, including parity subcommunicators.
Exact checks cover catalog order, all GEMM raw IDs, modulo rank partitioning and
singleton rejection holes. DIGIT / PASS; no precision study. Both targets compiled
and linked on Windows GNU; native runtime acceptance remains outstanding.

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
