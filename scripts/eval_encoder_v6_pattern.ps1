$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& "D:\miniforge3\envs\diffSPHEnv\python.exe" -m taiko_diffusion.eval_encoder `
  --checkpoint "checkpoints\encoder_v6_pattern\best.pt" `
  --split-dir "data\splits\encoder_v6_pattern" `
  --stats "data\splits\encoder_v6_pattern\label_stats.json" `
  --output-dir "eval\encoder_v6_pattern"
