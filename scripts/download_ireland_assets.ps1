[CmdletBinding()]
param(
    [switch]$SkipPbf,
    [switch]$SkipFastText,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$RawDir = Join-Path $Root "data\raw"
$FastTextDir = Join-Path $Root "models\fasttext"
$PbfPath = Join-Path $RawDir "ireland-and-northern-ireland-latest.osm.pbf"
$FastTextPath = Join-Path $FastTextDir "cc.en.300.bin"
$PythonPath = Join-Path $Root "venv\Scripts\python.exe"

function New-DirectoryIfMissing {
    param([string]$Path)
    if (-not (Test-Path $Path -PathType Container)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Test-NonEmptyFile {
    param([string]$Path)
    return (Test-Path $Path -PathType Leaf) -and ((Get-Item $Path).Length -gt 0)
}

New-DirectoryIfMissing $RawDir
New-DirectoryIfMissing $FastTextDir

if (-not (Test-Path $PythonPath -PathType Leaf)) {
    throw "venv Python was not found at $PythonPath. Create/install the venv before downloading fastText."
}

if (-not $SkipPbf) {
    $PbfUrl = "https://download.geofabrik.de/europe/ireland-and-northern-ireland-latest.osm.pbf"
    if ((Test-NonEmptyFile $PbfPath) -and -not $Force) {
        Write-Host "PBF already exists: $PbfPath"
    }
    else {
        Write-Host "Downloading Ireland/Northern Ireland OSM PBF..."
        Write-Host "Source: $PbfUrl"
        & curl.exe -L --fail --continue-at - --output $PbfPath $PbfUrl
        if ($LASTEXITCODE -ne 0) {
            throw "PBF download failed with exit code $LASTEXITCODE."
        }
        if (-not (Test-NonEmptyFile $PbfPath)) {
            throw "PBF download finished but the file is missing or empty: $PbfPath"
        }
    }
}

if (-not $SkipFastText) {
    if ((Test-NonEmptyFile $FastTextPath) -and -not $Force) {
        Write-Host "fastText model already exists: $FastTextPath"
    }
    else {
        Write-Host "Downloading fastText English model into $FastTextDir..."
        Write-Host "This is a large download and decompression step."
        Push-Location $FastTextDir
        try {
            & $PythonPath -c "import fasttext.util; fasttext.util.download_model('en', if_exists='ignore')"
            if ($LASTEXITCODE -ne 0) {
                throw "fastText download failed with exit code $LASTEXITCODE."
            }
        }
        finally {
            Pop-Location
        }
        if (-not (Test-NonEmptyFile $FastTextPath)) {
            throw "fastText download finished but cc.en.300.bin was not found at $FastTextPath"
        }
    }
}

Write-Host ""
Write-Host "Download check complete."
Write-Host "PBF:      $PbfPath"
Write-Host "fastText: $FastTextPath"
Write-Host ""
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\import_ireland_osm.ps1"
