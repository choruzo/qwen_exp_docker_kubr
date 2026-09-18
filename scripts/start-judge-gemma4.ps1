[CmdletBinding()]
param(
    [string]$LlamaServer = "D:\Archivos\Javier\Scritp_python\Agente\llama_cpp_server\build\bin\Release\llama-server.exe",
    [string]$Model = "G:\models\gemma-4-12b-it-UD-Q6_K_XL.gguf",
    [string]$ExpectedSha256 = "eb0f252863d14f7782122a4ac7e8744ed6e4a9fc132584d94686a9441f7c5d35",
    [int64]$ExpectedBytes = 10685011360,
    [ValidateRange(1, 65535)]
    [int]$Port = 8081,
    [ValidateRange(4096, 32768)]
    [int]$Context = 16384,
    [ValidateRange(1, 2)]
    [int]$Parallel = 1,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LlamaServer -PathType Leaf)) {
    throw "llama-server no existe: $LlamaServer"
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "El GGUF del juez no existe: $Model"
}
$modelFile = Get-Item -LiteralPath $Model
if ($modelFile.Length -ne $ExpectedBytes) {
    throw "El tamano del GGUF del juez no coincide: $($modelFile.Length) != $ExpectedBytes"
}
$actualHash = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "El SHA-256 del GGUF del juez no coincide: $actualHash"
}

$serverArgs = @(
    "--model", $Model,
    "--alias", "gemma-4-12b-judge",
    "--jinja",
    "--reasoning", "off",
    "--ctx-size", $Context,
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--flash-attn", "on",
    "--n-gpu-layers", "all",
    "--fit", "off",
    "--no-host",
    "--split-mode", "none",
    "--main-gpu", "0",
    "--parallel", $Parallel,
    "--batch-size", "512",
    "--ubatch-size", "256",
    "--cont-batching",
    "--threads", "4",
    "--threads-batch", "4",
    "--host", "127.0.0.1",
    "--port", $Port,
    "--metrics"
)

Write-Host "GGUF del juez verificado: $Model"
if ($VerifyOnly) {
    exit 0
}
Write-Host "Juez local: http://127.0.0.1:$Port/v1 (gemma-4-12b-judge)"
& $LlamaServer @serverArgs
exit $LASTEXITCODE
