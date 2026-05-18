[CmdletBinding()]
param(
    [string]$PbfPath = "data\raw\ireland-and-northern-ireland-latest.osm.pbf",
    [string]$Osm2pgsqlImage = "iboates/osm2pgsql:latest",
    [string]$TargetTable = "ireland_features",
    [switch]$KeepExistingTable,
    [switch]$SkipImagePull
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$PwExamplePath = Join-Path $Root "config\pw.example.txt"
$PwPath = Join-Path $Root "config\pw.txt"
$StyleDir = (Resolve-Path (Join-Path $Root "docker\osm2pgsql")).Path
$StyleFile = Join-Path $StyleDir "ireland_features.lua"
$StyleName = Split-Path $StyleFile -Leaf

if (-not (Test-Path $PbfPath -PathType Leaf)) {
    throw "PBF file not found: $PbfPath. Run scripts\download_ireland_assets.ps1 first."
}

if (-not (Test-Path $StyleFile -PathType Leaf)) {
    throw "osm2pgsql style file not found: $StyleFile"
}

if (-not (Test-Path $PwPath -PathType Leaf)) {
    if (-not (Test-Path $PwExamplePath -PathType Leaf)) {
        throw "Neither config\pw.txt nor config\pw.example.txt exists."
    }
    Copy-Item $PwExamplePath $PwPath
    Write-Host "Created config\pw.txt from config\pw.example.txt."
}

$PbfFullPath = (Resolve-Path $PbfPath).Path
$PbfDir = Split-Path $PbfFullPath -Parent
$PbfName = Split-Path $PbfFullPath -Leaf
$ContainerPbfPath = "/data/$PbfName"

Write-Host "Starting PostGIS..."
& docker compose up -d postgis
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up failed with exit code $LASTEXITCODE. Check that Docker Desktop is running."
}

Write-Host "Waiting for PostGIS readiness..."
$Ready = $false
for ($i = 1; $i -le 60; $i++) {
    & docker compose exec -T postgis pg_isready -U user -d db *> $null
    if ($LASTEXITCODE -eq 0) {
        $Ready = $true
        break
    }
    Start-Sleep -Seconds 2
}

if (-not $Ready) {
    & docker compose logs postgis
    throw "PostGIS did not become ready. Check Docker Desktop and the postgis container logs above."
}

$NetworkJson = (& docker inspect igea-postgis --format "{{json .NetworkSettings.Networks}}")
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect igea-postgis container network."
}
$NetworkObject = $NetworkJson | ConvertFrom-Json
$NetworkName = ($NetworkObject.PSObject.Properties | Select-Object -First 1).Name
if (-not $NetworkName) {
    throw "Could not resolve Docker network for igea-postgis."
}

if (-not $SkipImagePull) {
    Write-Host "Pulling osm2pgsql Docker image: $Osm2pgsqlImage"
    & docker pull $Osm2pgsqlImage
    if ($LASTEXITCODE -ne 0) {
        throw "docker pull failed for $Osm2pgsqlImage with exit code $LASTEXITCODE."
    }
}

if (-not $KeepExistingTable) {
    Write-Host "Dropping old tempview/$TargetTable objects if they exist..."
    & docker compose exec -T postgis psql -U user -d db -c "DROP MATERIALIZED VIEW IF EXISTS tempview; DROP TABLE IF EXISTS $TargetTable;"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to drop existing tempview or $TargetTable table."
    }
}

Write-Host "Importing $PbfName into PostGIS table $TargetTable..."
Write-Host "This can take a while depending on CPU, disk, and Docker memory."
& docker run --rm `
    --network $NetworkName `
    -e PGPASSWORD=igea `
    -v "${PbfDir}:/data:ro" `
    -v "${StyleDir}:/style:ro" `
    $Osm2pgsqlImage `
    -H postgis `
    -P 5432 `
    -U user `
    -d db `
    -O flex `
    -S "/style/$StyleName" `
    $ContainerPbfPath

if ($LASTEXITCODE -ne 0) {
    throw "osm2pgsql import failed with exit code $LASTEXITCODE."
}

Write-Host "Creating/refreshing helpful indexes..."
& docker compose exec -T postgis psql -U user -d db -c "CREATE INDEX IF NOT EXISTS ${TargetTable}_way_idx ON $TargetTable USING GIST (way); CREATE INDEX IF NOT EXISTS ${TargetTable}_tags_idx ON $TargetTable USING GIN (tags); CREATE INDEX IF NOT EXISTS ${TargetTable}_uid_idx ON $TargetTable (osm_uid);"
if ($LASTEXITCODE -ne 0) {
    throw "Index creation failed."
}

Write-Host "Verifying imported table..."
& docker compose exec -T postgis psql -U user -d db -c "SELECT COUNT(*) AS feature_count, COUNT(*) FILTER (WHERE osm_type = 'N') AS nodes, COUNT(*) FILTER (WHERE osm_type = 'W') AS ways, COUNT(*) FILTER (WHERE osm_type = 'R') AS relations, COUNT(*) FILTER (WHERE tags ? 'wikidata') AS wikidata_features, COUNT(*) FILTER (WHERE tags ? 'wikipedia') AS wikipedia_features FROM $TargetTable;"
if ($LASTEXITCODE -ne 0) {
    throw "Import verification query failed."
}

Write-Host ""
Write-Host "Import complete."
Write-Host ""
Write-Host "Next:"
Write-Host "  venv\Scripts\python scripts\build_nca_vocab.py data\raw\ireland-and-northern-ireland-latest.osm.pbf --input-format pbf --link-keys both --tag-output config\osmTagKeyWiki.csv --key-output config\osmKeyWiki.csv"
