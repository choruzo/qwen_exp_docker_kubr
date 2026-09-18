[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [int]$GeneratorPid,
    [Parameter(Mandatory)]
    [int]$LlamaServerPid
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = (Resolve-Path (Join-Path $root "venv\Scripts\python.exe")).Path
$compose = Join-Path $root "compose.train.yaml"
$fullOutput = Join-Path $root "data\interim\normalized\reverse_generated_full.jsonl"

function Invoke-Checked {
    param(
        [Parameter(Mandatory)]
        [string]$Executable,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE"
    }
}

$generator = Get-Process -Id $GeneratorPid -ErrorAction Stop
if ($generator.ProcessName -notlike "python*") {
    throw "PID $GeneratorPid is not a Python generator: $($generator.ProcessName)"
}
Write-Host "Waiting for reverse generation PID $GeneratorPid"
$generator.WaitForExit()
if ($generator.ExitCode -ne 0) {
    throw "Reverse generation failed with exit code $($generator.ExitCode)"
}
if (-not (Test-Path -LiteralPath $fullOutput -PathType Leaf)) {
    throw "Reverse generation exited successfully but output is missing: $fullOutput"
}

$server = Get-Process -Id $LlamaServerPid -ErrorAction Stop
if ($server.ProcessName -ne "llama-server") {
    throw "PID $LlamaServerPid is not llama-server: $($server.ProcessName)"
}
$server | Stop-Process -Force
$server.WaitForExit()
Write-Host "Stopped llama-server PID $LlamaServerPid"

Push-Location $root
try {
    Invoke-Checked $python @("-m", "docker_k8s_finetune.cli", "dedupe", "--mode", "exact")
    Invoke-Checked "docker" @(
        "compose", "-f", $compose, "run", "--rm", "train",
        "python", "-m", "docker_k8s_finetune.cli", "dedupe", "--mode", "approximate"
    )
    Invoke-Checked $python @("-m", "docker_k8s_finetune.cli", "validate-dataset")
    Invoke-Checked "docker" @(
        "compose", "-f", $compose, "run", "--rm", "train",
        "python", "-m", "docker_k8s_finetune.cli", "split"
    )
    Invoke-Checked $python @("-m", "docker_k8s_finetune.cli", "pipeline-status")
} finally {
    Pop-Location
}

Write-Host "Post-generation data pipeline completed successfully"
