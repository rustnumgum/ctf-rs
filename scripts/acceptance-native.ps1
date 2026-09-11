# ctf-rs acceptance run (native Windows, MSYS2 mingw64 + MS-MPI).
#
# Reads scripts\acceptance-targets.tsv (the same manifest scripts\check-
# targets.sh checks on Linux/macOS) for the set of gating cargo test
# targets, builds every needed test binary once, then runs each one,
# recording a per-target RUN_EXIT line and log so a failure partway through
# does not void the evidence of everything that ran after it (audit
# findings E1, E2, E4, E5). mpi targets run the built executable directly
# under mpiexec via Start-Process with a 600s wait, bypassing Cargo's
# per-host-triple runner variable. This script has no bash available, so
# the manifest-completeness check below is a native PowerShell equivalent
# of scripts\check-targets.sh, built on `cargo metadata | ConvertFrom-Json`
# instead of calling bash.
param(
    [string]$MingwRoot = 'C:\msys64\mingw64',
    [string]$TargetDir = 'D:\ctf-rs-native-target',
    [string]$MsMpiBin = 'C:\Program Files\Microsoft MPI\Bin',
    [switch]$BuildOnly,
    [switch]$D6Only,
    [switch]$C1Only
)
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PATH = "$HOME\.cargo\bin;$MingwRoot\bin;$MsMpiBin;" + $env:PATH
$env:MSMPI_INC = "$MingwRoot\include"
$env:MSMPI_LIB64 = "$MingwRoot\lib"
$env:LIBCLANG_PATH = "$MingwRoot\bin"
$env:RUSTFLAGS = "-L native=$($MingwRoot.Replace('\','/'))/lib"
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = $TargetDir }
$env:OPENBLAS_NUM_THREADS = '1'

$manifestPath = Join-Path $PSScriptRoot 'acceptance-targets.tsv'
if (-not (Test-Path $manifestPath)) {
    Write-Host "acceptance-native: missing $manifestPath"
    exit 1
}

$mpiexecPath = Join-Path $MsMpiBin 'mpiexec.exe'
if (-not (Test-Path $mpiexecPath)) {
    Write-Host "acceptance-native: mpiexec.exe not found at $mpiexecPath"
    exit 1
}

function Test-AcceptanceManifest {
    param([string]$ManifestPath)

    $metaText = cargo metadata --no-deps --format-version 1
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'acceptance-native: cargo metadata failed'
        exit $LASTEXITCODE
    }
    $meta = $metaText | ConvertFrom-Json

    $metaNames = @(
        $meta.packages | ForEach-Object { $_.targets } |
            Where-Object { $_.kind -contains 'test' } |
            ForEach-Object { $_.name }
    ) | Sort-Object -Unique

    $rows = Import-Csv -Delimiter "`t" -Path $ManifestPath
    $manifestNames = @(
        $rows | Where-Object { $_.class -in @('mpi', 'local', 'excluded') } |
            ForEach-Object { $_.target }
    ) | Sort-Object -Unique

    $diff = Compare-Object -ReferenceObject $metaNames -DifferenceObject $manifestNames
    $missing = @($diff | Where-Object { $_.SideIndicator -eq '<=' } | ForEach-Object { $_.InputObject })
    $extra = @($diff | Where-Object { $_.SideIndicator -eq '=>' } | ForEach-Object { $_.InputObject })

    if ($missing.Count -gt 0 -or $extra.Count -gt 0) {
        if ($missing.Count -gt 0) {
            Write-Host "acceptance-native: test targets missing from $ManifestPath (not listed as mpi/local/excluded):"
            foreach ($m in $missing) { Write-Host "  $m" }
        }
        if ($extra.Count -gt 0) {
            Write-Host "acceptance-native: manifest rows (class mpi/local/excluded) name no existing target:"
            foreach ($e in $extra) { Write-Host "  $e" }
        }
        exit 1
    }

    Write-Host "acceptance-native: OK, $($metaNames.Count) cargo test targets all accounted for in $ManifestPath"
    return $rows
}

$rows = Test-AcceptanceManifest -ManifestPath $manifestPath

# $ErrorActionPreference = 'Stop' would otherwise turn ordinary stderr
# output from a native command (cargo's own build warnings included) into
# a terminating exception the moment that stream is redirected to a file
# or merged with *>/2>&1; every cargo invocation from here on redirects
# stderr, so relax it before the build and the run loop, where a failing
# target must not stop the run anyway (E2).
$ErrorActionPreference = 'Continue'

if ($BuildOnly) {
    $allMpi = @($rows | Where-Object { $_.class -eq 'mpi' })
    $allLocal = @($rows | Where-Object { $_.class -eq 'local' } | ForEach-Object { $_.target })
    $hasLib = @($rows | Where-Object { $_.class -eq 'lib' }).Count -gt 0
    $buildOnlyArgs = @()
    foreach ($r in $allMpi) { $buildOnlyArgs += @('--test', $r.target) }
    foreach ($t in $allLocal) { $buildOnlyArgs += @('--test', $t) }
    if ($hasLib) { $buildOnlyArgs += '--lib' }
    cargo test --no-run @buildOnlyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    cargo build --examples
    exit $LASTEXITCODE
}

$mpiRows = @($rows | Where-Object { $_.class -eq 'mpi' })
$libFilters = @($rows | Where-Object { $_.class -eq 'lib' } | ForEach-Object { $_.target })
$localTargets = @($rows | Where-Object { $_.class -eq 'local' } | ForEach-Object { $_.target })

if ($D6Only) {
    $mpiRows = @($mpiRows | Where-Object { ($_.sets -split ',') -contains 'd6' })
    $libFilters = @()
    $localTargets = @()
} elseif ($C1Only) {
    $mpiRows = @($mpiRows | Where-Object { ($_.sets -split ',') -contains 'c1' })
    $libFilters = @()
    $localTargets = @()
}

$logDir = Join-Path $env:TEMP "ctf-rs-acceptance\$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Write-Host "acceptance-native: logs in $logDir"

# Build every selected mpi/local test binary once, plus the lib test binary
# if any lib filter is selected, and capture cargo's JSON build artifacts so
# each mpi target's executable path can be resolved without a runner
# env var (keeps this script triple-agnostic, matching scripts\
# acceptance-wsl.sh).
$buildArgs = @()
foreach ($r in $mpiRows) { $buildArgs += @('--test', $r.target) }
foreach ($t in $localTargets) { $buildArgs += @('--test', $t) }
if ($libFilters.Count -gt 0) { $buildArgs += '--lib' }

$buildJsonPath = Join-Path $logDir 'build.json'
$buildErrPath = Join-Path $logDir 'build.stderr.log'
if ($buildArgs.Count -gt 0) {
    cargo test --no-run --message-format=json @buildArgs 1> $buildJsonPath 2> $buildErrPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "acceptance-native: build failed, see $buildErrPath"
        exit $LASTEXITCODE
    }
}

$exeMap = @{}
if (Test-Path $buildJsonPath) {
    Get-Content $buildJsonPath | ForEach-Object {
        if ([string]::IsNullOrWhiteSpace($_)) { return }
        $msg = $null
        try { $msg = $_ | ConvertFrom-Json } catch { return }
        if ($msg.reason -eq 'compiler-artifact' -and $msg.profile -and $msg.profile.test -eq $true -and $msg.executable) {
            $exeMap[$msg.target.name] = $msg.executable
        }
    }
}

foreach ($r in $mpiRows) {
    if (-not $exeMap.ContainsKey($r.target)) {
        Write-Host "acceptance-native: no built executable for mpi target '$($r.target)', see $buildErrPath"
        exit 1
    }
}

function Invoke-MpiAcceptanceTarget {
    param(
        [string]$Target,
        [int]$Ranks,
        [string]$Exe,
        [string]$LogDir,
        [string]$MpiexecPath
    )
    $outPath = Join-Path $LogDir "$Target-$Ranks.out.tmp"
    $errPath = Join-Path $LogDir "$Target-$Ranks.err.tmp"
    $logPath = Join-Path $LogDir "$Target-$Ranks.log"

    $proc = Start-Process -FilePath $MpiexecPath -ArgumentList @('-n', "$Ranks", $Exe) `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput $outPath -RedirectStandardError $errPath

    $finished = $proc.WaitForExit(600000)
    if (-not $finished) {
        & taskkill /PID $proc.Id /T /F | Out-Null
        Add-Content -Path $errPath -Value 'acceptance-native: timed out after 600s'
        $code = 124
    } else {
        $code = $proc.ExitCode
    }

    Get-Content -Path $outPath, $errPath -ErrorAction SilentlyContinue | Set-Content -Path $logPath
    Remove-Item -Path $outPath, $errPath -ErrorAction SilentlyContinue
    return $code
}

# From here a failing target must not stop the run (E2): every RUN_EXIT
# below is recorded regardless of the target's exit code, matching
# scripts\acceptance-wsl.sh's `set +e` after its own build step.
$passCount = 0
$failCount = 0
$failing = @()

foreach ($r in $mpiRows) {
    $ranksList = $r.ranks -split ' '
    $exe = $exeMap[$r.target]
    foreach ($ranksStr in $ranksList) {
        $ranks = [int]$ranksStr
        $code = Invoke-MpiAcceptanceTarget -Target $r.target -Ranks $ranks -Exe $exe -LogDir $logDir -MpiexecPath $mpiexecPath
        Write-Host "RUN_EXIT $($r.target) ranks=$ranks exit=$code"
        if ($code -eq 0) {
            $passCount++
        } else {
            $failCount++
            $failing += "$($r.target) ranks=$ranks exit=$code"
        }
    }
}

foreach ($filter in $libFilters) {
    $logPath = Join-Path $logDir "lib-$filter.log"
    cargo test --lib $filter --no-fail-fast -- --nocapture *> $logPath
    $code = $LASTEXITCODE
    Write-Host "RUN_EXIT $filter ranks=- exit=$code"
    if ($code -eq 0) {
        $passCount++
    } else {
        $failCount++
        $failing += "$filter exit=$code"
    }
}

foreach ($target in $localTargets) {
    $logPath = Join-Path $logDir "$target.log"
    cargo test --test $target --no-fail-fast -- --nocapture *> $logPath
    $code = $LASTEXITCODE
    Write-Host "RUN_EXIT $target ranks=- exit=$code"
    if ($code -eq 0) {
        $passCount++
    } else {
        $failCount++
        $failing += "$target exit=$code"
    }
}

$total = $passCount + $failCount
Write-Host "acceptance-native: summary $passCount/$total passed"
if ($failCount -gt 0) {
    Write-Host "acceptance-native: failing ($failCount):"
    foreach ($f in $failing) { Write-Host "  $f" }
    exit 1
}
exit 0
