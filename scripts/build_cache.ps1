$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& "D:\miniforge3\python.exe" -m taiko_diffusion.data.build_cache `
  --config "configs\encoder_v0.yaml" `
  --output-dir "data\cache\encoder_v0"

