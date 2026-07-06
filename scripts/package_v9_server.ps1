param(
  [string]$Output = "taiko-diffusion-v9-server.tar.gz"
)

$ErrorActionPreference = "Stop"

$items = @(
  "taiko_diffusion",
  "configs",
  "pyproject.toml",
  "requirements.txt",
  "README.md",
  "PROJECT_STATUS_AND_SERVER_PLAN.md",
  "SERVER_TRANSFER_MANIFEST.txt",
  "data/cache/diffusion_v9_donka",
  "data/cache/audio_v0",
  "checkpoints/autoencoder_kl_v9_donka",
  "checkpoints/latent_diffusion_v9_mug_scale_donka",
  "logs/latent_diffusion_v9_mug_scale_donka",
  "eval/full_test_latent_v9_donka_epoch22_g25.json",
  "eval/full_test_latent_v9_donka_epoch22_g30.json",
  "eval/videos/gameplay_v9_sample_row0_seed0_g25.mp4",
  "eval/videos/gameplay_v9_low_row1089_seed1_g25.mp4",
  "eval/videos/gameplay_v9_mid_row98_seed2_g25.mp4",
  "eval/videos/gameplay_v9_high_row641_seed3_g25.mp4"
)

$missing = @()
foreach ($item in $items) {
  if (-not (Test-Path $item)) {
    $missing += $item
  }
}

if ($missing.Count -gt 0) {
  Write-Error ("Missing transfer items:`n" + ($missing -join "`n"))
}

if (Test-Path $Output) {
  Remove-Item -LiteralPath $Output -Force
}

tar -czf $Output @items
Write-Host "Wrote $Output"
