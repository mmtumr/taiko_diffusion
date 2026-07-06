$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& "D:\miniforge3\python.exe" -m taiko_diffusion.data.split_stats `
  --config "configs\encoder_v7_speed.yaml" `
  --index "data\cache\encoder_v7_speed\index.csv"
