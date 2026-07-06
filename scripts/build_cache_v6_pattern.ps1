$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& "D:\miniforge3\python.exe" -m taiko_diffusion.data.build_v5_cache `
  --config "configs\encoder_v6_pattern.yaml" `
  --source-index "data\cache\encoder_v1\index.csv" `
  --source-train "data\splits\encoder_v1\train.csv" `
  --output-dir "data\cache\encoder_v6_pattern"
