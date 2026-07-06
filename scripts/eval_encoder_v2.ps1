$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& "D:\miniforge3\envs\diffSPHEnv\python.exe" -m taiko_diffusion.eval_encoder `
  --checkpoint "checkpoints\encoder_v2\best.pt" `
  --split-dir "data\splits\encoder_v1" `
  --stats "data\splits\encoder_v1\label_stats.json" `
  --output-dir "eval\encoder_v2"
