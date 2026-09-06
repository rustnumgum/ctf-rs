# Native Windows GNU build and acceptance

This route uses the installed x86_64-pc-windows-gnu Rust toolchain and MSYS2
MINGW64 native Windows libraries. It is not WSL or Cygwin execution.
Install the matching MINGW64 packages (not mixed UCRT64/CLANG64 binaries):

```
pacman -S --needed mingw-w64-x86_64-scalapack mingw-w64-x86_64-msmpi mingw-w64-x86_64-clang-libs mingw-w64-x86_64-pkgconf
```

The MSYS2 msmpi package supplies the SDK/import library, not the MPI runtime.
Install the runtime separately from [Microsoft MPI 10.1.3](https://www.microsoft.com/en-us/download/details.aspx?id=105289).
Its installer requires Windows elevation. Do not treat SDK installation or
successful linking as proof that msmpi.dll and mpiexec are available.

From the project root in PowerShell:

```powershell
./scripts/acceptance-native.ps1 -BuildOnly
./scripts/acceptance-native.ps1
```

The script sets process-local compiler/header/library paths, keeps build output
outside the repository, and runs the same current MPI/local test selection as
the WSL script. A failure surfaces immediately; no silent fallback to WSL or a
single-process replacement is used. ScaLAPACK links as scalapack on Windows and
scalapack-openmpi on Linux; Windows BLAS/LAPACK symbols come from openblas.

## Observed status, 2026-09-07

All current Cargo test targets compiled and linked successfully on native
Windows GNU with default native-linalg/native-scalapack features. The initial
link failed because MINGW64 provides openblas rather than separate blas/lapack
import names; platform-specific native link names corrected that failure.

Runtime acceptance is **not passed**. The Microsoft MPI installer returned
0x800704c7 (canceled by user); no second installation attempt was made.
System32/msmpi.dll was absent. A direct local_linalg executable launch exited
1 without test output; import inspection confirmed it requires msmpi.dll as
well as OpenBLAS/ScaLAPACK. No numerical result was produced, so there is no
precision comparison to accept. After the runtime is installed, execute the
native acceptance script; do not repeat already-passing WSL checks merely for
the unchanged Linux link attributes.
