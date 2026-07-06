$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& "D:\miniforge3\python.exe" -m taiko_diffusion.data.prepare_manifest `
  --ese-root "..\ESE-master\ese" `
  --rating-xlsx "..\rating计算工具_11.25.xlsx" `
  --output-dir "data\manifests"

