$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& "D:\miniforge3\envs\diffSPHEnv\python.exe" -m taiko_diffusion.eval_encoder `
  --checkpoint "checkpoints\encoder_v0\best.pt" `
  --split-dir "data\splits\encoder_v0" `
  --stats "data\splits\encoder_v0\label_stats.json" `
  --output-dir "eval\encoder_v0"

