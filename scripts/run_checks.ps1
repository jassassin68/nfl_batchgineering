# run_checks.ps1 - Combined test gate for the NFL prediction system.
#
# Runs the Python unit/look-ahead test suite, then the dbt build (run + test).
# Exits non-zero if either layer fails. This is the "make test" equivalent for
# this Windows repo.
#
# Usage (from the repo root):
#   .\scripts\run_checks.ps1

$ErrorActionPreference = 'Stop'

# Resolve repo root (parent of this script's directory).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$failed = $false

Write-Host '=== Python tests (pytest) ===' -ForegroundColor Cyan
python -m pytest
if ($LASTEXITCODE -ne 0) {
    Write-Host 'pytest FAILED' -ForegroundColor Red
    $failed = $true
}

Write-Host ''
Write-Host '=== dbt build (run + test) ===' -ForegroundColor Cyan

# Locate the venv's dbt. .venv usually lives at the repo root; when running
# from a git worktree it may live one or more parents up, so walk upward.
$DbtExe = $null
$Dir = $RepoRoot
while ($Dir -and -not $DbtExe) {
    $candidate = Join-Path $Dir '.venv\Scripts\dbt.exe'
    if (Test-Path $candidate) {
        $DbtExe = $candidate
        break
    }
    $parent = Split-Path -Parent $Dir
    if ($parent -eq $Dir) { break }
    $Dir = $parent
}

if (-not $DbtExe) {
    Write-Host "dbt not found in any .venv\Scripts\dbt.exe from $RepoRoot upward - run 'pip install -r requirements.txt' in the repo-root venv" -ForegroundColor Red
    $failed = $true
} else {
    & $DbtExe build --project-dir dbt_project
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'dbt build FAILED' -ForegroundColor Red
        $failed = $true
    }
}

Write-Host ''
if ($failed) {
    Write-Host 'CHECKS FAILED' -ForegroundColor Red
    exit 1
}
Write-Host 'ALL CHECKS PASSED' -ForegroundColor Green
exit 0
