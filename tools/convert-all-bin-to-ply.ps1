param(
  [string]$SourceDir = (Join-Path (Get-Location) 'data'),
  [string]$OutDir = (Join-Path (Get-Location) 'data_ply'),
  [switch]$Ascii
)

function Resolve-CloudCompare {
  $c = Get-Command CloudCompare.exe -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  $default = 'C:\\Program Files\\CloudCompare\\CloudCompare.exe'
  if (Test-Path $default) { return $default }
  throw 'CloudCompare.exe not found in PATH or default install path.'
}

if (-not (Test-Path -LiteralPath $SourceDir)) {
  throw "SourceDir not found: $SourceDir"
}
if (-not (Test-Path -LiteralPath $OutDir)) {
  New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$cc = Resolve-CloudCompare
$fmt = if ($Ascii) { 'ASCII' } else { 'BINARY_LE' }

Write-Host "Using CloudCompare: $cc" -ForegroundColor Cyan
Write-Host "Source: $SourceDir" -ForegroundColor Cyan
Write-Host "Output: $OutDir (PLY $fmt)" -ForegroundColor Cyan

Get-ChildItem -LiteralPath $SourceDir -File -Filter *.bin | ForEach-Object {
  $src = $_.FullName
  $dst = Join-Path $OutDir ($_.BaseName + '.ply')
  Write-Host "Converting: $src -> $dst"
  & $cc -SILENT -AUTO_SAVE OFF -NO_TIMESTAMP -O $src -C_EXPORT_FMT PLY -PLY_EXPORT_FMT $fmt -PREC 10 -SAVE_CLOUDS FILE $dst | Out-Null
}

Write-Host 'Done.' -ForegroundColor Green

