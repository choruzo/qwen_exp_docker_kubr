[CmdletBinding()]
param(
    [string]$LlamaServer = "D:\Archivos\Javier\Scritp_python\Agente\llama_cpp_server\build\bin\Release\llama-server.exe",
    [string]$Manifest = "artifacts\export_manifest.json",
    [string]$Quantization = "q4_k_m",
    [string]$Alias = "qwen-docker-k8s",
    [ValidateRange(1, 65535)]
    [int]$Port = 8080,
    [ValidateRange(4096, 32768)]
    [int]$Context = 16384,
    [ValidateRange(1, 4)]
    [int]$Parallel = 1,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$manifestPath = if ([IO.Path]::IsPathRooted($Manifest)) {
    [IO.Path]::GetFullPath($Manifest)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $Manifest))
}

if (-not (Test-Path -LiteralPath $LlamaServer -PathType Leaf)) {
    throw "llama-server no existe: $LlamaServer"
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "El manifiesto de exportacion no existe: $manifestPath"
}

$payload = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$entries = @($payload.artifacts.gguf)
if ($entries.Count -eq 0) {
    throw "El manifiesto no contiene artifacts.gguf"
}
$normalizedQuantization = $Quantization.ToLowerInvariant().Replace("_", "").Replace("-", "")
$matches = @($entries | Where-Object {
    $leaf = [IO.Path]::GetFileName(([string]$_.path).Replace("/", [IO.Path]::DirectorySeparatorChar))
    $leaf.EndsWith(".gguf", [StringComparison]::OrdinalIgnoreCase) -and
        $leaf.ToLowerInvariant().Replace("_", "").Replace("-", "").Contains($normalizedQuantization)
})
if ($matches.Count -ne 1) {
    throw "Se esperaba un unico GGUF $Quantization en el manifiesto; encontrados: $($matches.Count)"
}

$entry = $matches[0]
$relativePath = ([string]$entry.path).Replace("/", [IO.Path]::DirectorySeparatorChar)
$modelPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $relativePath))
$projectPrefix = $projectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $modelPath.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "La ruta GGUF sale del proyecto: $modelPath"
}
if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
    throw "El GGUF manifestado no existe: $modelPath"
}
$actual = Get-Item -LiteralPath $modelPath
if ($actual.Length -ne [int64]$entry.bytes) {
    throw "El tamano del GGUF no coincide con el manifiesto: $modelPath"
}
$actualHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne ([string]$entry.sha256).ToLowerInvariant()) {
    throw "El SHA-256 del GGUF no coincide con el manifiesto: $modelPath"
}

$serverArgs = @(
    "--model", $modelPath,
    "--alias", $Alias,
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

Write-Host "GGUF verificado: $modelPath"
if ($VerifyOnly) {
    exit 0
}
Write-Host "Endpoint local: http://127.0.0.1:$Port/v1 ($Alias)"
Write-Host "Desde Docker: GGUF_LLM_BASE_URL=http://host.docker.internal:$Port/v1"
& $LlamaServer @serverArgs
exit $LASTEXITCODE
