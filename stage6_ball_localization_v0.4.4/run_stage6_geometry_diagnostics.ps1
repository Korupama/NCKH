param(
    [Parameter(Mandatory=$true)][string]$Csv,
    [Parameter(Mandatory=$true)][string]$Benchmark2DDir,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$PrimaryReport = "",
    [string]$Python = "python",
    [double]$BallRadiusM = 0.11,
    [double]$PitchMarginM = 6.0,
    [double]$MaxHeightM = 30.0
)

$arguments = @(
    "benchmark_stage6_protocol.py", "geometry-diagnostics",
    "--csv", $Csv,
    "--benchmark-2d-dir", $Benchmark2DDir,
    "--output-dir", $OutputDir,
    "--split", "test",
    "--ball-radius-m", "$BallRadiusM",
    "--pitch-margin-m", "$PitchMarginM",
    "--max-height-m", "$MaxHeightM"
)
if ($PrimaryReport -ne "") {
    $arguments += @("--primary-report", $PrimaryReport)
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
