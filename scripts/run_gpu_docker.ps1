[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Image = "igea-tensorflow-gpu:2.12",
    [switch]$Build,
    [string]$CommandLine = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ContainerCommand
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dockerfile = Join-Path $RepoRoot "docker\tensorflow-gpu\Dockerfile"

if ($CommandLine) {
    $ContainerCommand = @("bash", "-lc", $CommandLine)
} elseif (-not $ContainerCommand -or $ContainerCommand.Count -eq 0) {
    $ContainerCommand = @("python", "scripts/check_tf_gpu.py")
}

if ($ContainerCommand.Count -gt 0 -and $ContainerCommand[0] -eq "--") {
    if ($ContainerCommand.Count -eq 1) {
        $ContainerCommand = @("python", "scripts/check_tf_gpu.py")
    } else {
        $ContainerCommand = $ContainerCommand[1..($ContainerCommand.Count - 1)]
    }
}

$imageExists = $false
try {
    docker image inspect $Image *> $null
    $imageExists = $true
} catch {
    $imageExists = $false
}

if ($Build -or -not $imageExists) {
    docker build -f $Dockerfile -t $Image $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed with exit code $LASTEXITCODE"
    }
}

$dockerArgs = @(
    "run",
    "--rm",
    "--gpus", "all",
    "--add-host", "host.docker.internal:host-gateway",
    "-v", "${RepoRoot}:/workspace",
    "-w", "/workspace",
    "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true",
    "-e", "PYTHONUNBUFFERED=1"
)

if ($env:OPENAI_API_KEY) {
    $dockerArgs += @("-e", "OPENAI_API_KEY=$env:OPENAI_API_KEY")
}

$dockerArgs += @($Image)
$dockerArgs += $ContainerCommand

Write-Host "docker $($dockerArgs -join ' ')"
& docker @dockerArgs
if ($LASTEXITCODE -ne 0) {
    throw "Docker run failed with exit code $LASTEXITCODE"
}
