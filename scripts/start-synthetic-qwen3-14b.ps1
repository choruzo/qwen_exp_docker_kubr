[CmdletBinding()]
param(
    [string]$LlamaServer = "D:\Archivos\Javier\Scritp_python\Agente\llama_cpp_server\build\bin\Release\llama-server.exe",
    [string]$Model = "G:\models\Qwen3-14B-Q5_K_M.gguf",
    [ValidateRange(1, 65535)]
    [int]$Port = 8080,
    [ValidateRange(4096, 40960)]
    [int]$Context = 16384,
    [ValidateRange(1, 4)]
    [int]$Parallel = 2
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LlamaServer -PathType Leaf)) {
    throw "llama-server no existe: $LlamaServer"
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "El GGUF sintetico no existe: $Model"
}

$serverArgs = @(
    "--model", $Model,
    "--alias", "qwen3-14b",
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

Write-Host "Servidor sintetico local: http://127.0.0.1:$Port/v1 (qwen3-14b)"
& $LlamaServer @serverArgs
exit $LASTEXITCODE
