# Taiko Diffusion 项目进度与服务器训练计划

更新时间：2026-06-25

本文档记录当前 `taiko-diffusion` 项目的可用状态、关键文件、迁移服务器所需文件，以及下一阶段训练计划。当前主线是基于 Mug-Diffusion 思路的太鼓谱面 latent diffusion。

## 1. 当前结论

当前最值得继续推进的是：

```text
v9 Mug-style latent diffusion + don/ka split target
```

也就是：

```text
音频 mel/onset/rms 条件
+ 谱面条件向量
+ KL autoencoder latent
+ latent U-Net diffusion
+ DDIM + CFG guidance
+ 输出 don/ka 两轨
```

当前 diffusion 还没有直接接入我们训练出来的 encoder。现在训练和采样用的是缓存里的真值/统计条件向量，不是 encoder 预测值。因此目前的效果主要代表 diffusion 本身的上限，不包含 encoder 误差。

## 2. 当前主线模型

### 2.1 v9 目标格式

v9 已经从旧的：

```text
note_event + ka_probability
```

改回：

```text
don
ka
```

大音符在生成目标里折叠到普通音符：

```text
don = don + big_don
ka  = ka + big_ka
```

这更符合太鼓街机实际击打逻辑：大音符只是分数差别，不需要生成独立击打轨。

### 2.2 当前最佳 checkpoint

```text
latent diffusion:
  checkpoints/latent_diffusion_v9_mug_scale_donka/best.pt

autoencoder:
  checkpoints/autoencoder_kl_v9_donka/best.pt

latent stats:
  checkpoints/autoencoder_kl_v9_donka/latent_stats.json
```

v9 latent diffusion 训练到 24 epoch，best 在 epoch 22：

```text
epoch 22 train_loss 0.1850
epoch 22 val_loss   0.1938
epoch 23 val_loss   0.2055
epoch 24 val_loss   0.1971
```

说明 epoch 22 后有轻微回摆，继续训练需要保存更多 checkpoint 或按生成指标挑选，而不是只看 diffusion val loss。

## 3. 数据集状态

当前 v9 diffusion cache：

```text
data/cache/diffusion_v9_donka
```

来源链路：

```text
ESE-master 谱面
-> 与 rating计算工具_11.25.xlsx 严格匹配
-> data/manifests/strict_matched_dataset.csv
-> encoder_v1 tensor cache
-> data/splits/encoder_final_main train/val/test
-> diffusion_v9_donka 512-frame window cache
```

当前数据量：

```text
train: 9861 windows, 1017 charts
val:   1230 windows, 127 charts
test:  1209 windows, 128 charts
```

窗口设置：

```text
window_frames: 512
stride_frames: 256
frame_ms: 46.4399
window_duration: about 23.8 sec
overlap: about 11.9 sec
```

音频 cache：

```text
data/cache/audio_v0
```

音频与谱面全局对齐审计结果：

```text
train: best shift 0 frames, onset_mean 0.4576, top25 hit 0.4701
val:   best shift 0 frames, onset_mean 0.4616, top25 hit 0.4682
test:  best shift 0 frames, onset_mean 0.4619, top25 hit 0.4834
```

结论：整体对齐可用，但少量单曲仍可能有噪声。

## 4. 当前条件向量

v9 diffusion 当前使用 11 维条件：

```text
const
complex
hs_change
bpm_rhythm_bin
note_type
note_type_high
avg_density
peak_density
big_note_ratio
balloon_roll_ratio
ka_ratio
```

注意：

```text
当前是用 cache 中的真值/统计条件，不是 encoder 预测条件。
```

后续接 encoder 后，输入条件会变成：

```text
谱面/片段 -> encoder -> 这些条件 -> diffusion
```

这会引入 encoder 误差，需要单独评估。

## 5. 当前效果

v8 旧格式：

```text
target: note_event + ka_probability
```

v9 新格式：

```text
target: don + ka
```

全测试 DDIM-50 指标：

```text
v8 epoch 18, guidance 2.5:
  onset_mean   0.4628 / target 0.4727
  top25 hit    0.4916 / target 0.5087
  ka_ratio_mae 0.1304

v9 epoch 22, guidance 2.5:
  onset_mean   0.4685 / target 0.4727
  top25 hit    0.5016 / target 0.5087
  ka_ratio_mae 0.0846

v9 epoch 22, guidance 3.0:
  onset_mean   0.4791 / target 0.4727
  top25 hit    0.5182 / target 0.5087
  ka_ratio_mae 0.0840
```

当前推荐采样参数：

```text
checkpoint: checkpoints/latent_diffusion_v9_mug_scale_donka/best.pt
sampler: DDIM
sample_steps: 50
guidance_scale: 2.5
onset_mix: 0.0
```

`guidance_scale=3.0` 可以更贴 onset，但已经略微过引导。

## 6. 已知问题

当前视频预览暴露的问题：

```text
1. 低密度、原本几乎全 don 的窗口，模型仍会生成一些 ka。
2. 高密度窗口有时 ka 偏多，红蓝分布还不够像人谱。
3. 当前只生成 23.8 秒窗口，还没有做整曲拼接和跨窗口一致性。
4. 当前不生成 roll/balloon，只生成 don/ka 普通音符。
5. 当前 diffusion 未接入 encoder，仍使用 cache 真值条件。
6. 当前生成评价主要是 onset/top25/ka_ratio，缺少 pattern-level 人类谱面质量指标。
```

## 7. 当前可视化

已新增 gameplay 风格预览脚本：

```text
taiko_diffusion/render_gameplay_sample.py
```

已生成示例视频：

```text
eval/videos/gameplay_v9_sample_row0_seed0_g25.mp4
eval/videos/gameplay_v9_low_row1089_seed1_g25.mp4
eval/videos/gameplay_v9_mid_row98_seed2_g25.mp4
eval/videos/gameplay_v9_high_row641_seed3_g25.mp4
```

这些视频只显示生成谱面，不显示 target，对主观判断更有用。

## 8. 服务器迁移：最小必需文件

如果只是继续训练 v9，不需要带所有历史实验。最小需要：

```text
taiko_diffusion/
configs/
pyproject.toml
requirements.txt
README.md
PROJECT_STATUS_AND_SERVER_PLAN.md

data/cache/diffusion_v9_donka/
data/cache/audio_v0/

checkpoints/autoencoder_kl_v9_donka/
checkpoints/latent_diffusion_v9_mug_scale_donka/

logs/latent_diffusion_v9_mug_scale_donka/
eval/full_test_latent_v9_donka_epoch22_g25.json
eval/full_test_latent_v9_donka_epoch22_g30.json
```

可选但建议带：

```text
eval/videos/gameplay_*.mp4
eval/videos/v9_*.npz
```

如果服务器上要重新构建数据和音频 cache，还需要带原始数据：

```text
../ESE-master/
../rating计算工具_11.25.xlsx
```

但如果只继续从现有 cache 训练，原始 ESE 和 Excel 不是必需。

## 9. 当前关键目录体积

本机统计：

```text
taiko_diffusion/                                  0.69 MB
configs/                                         0.09 MB
checkpoints/latent_diffusion_v9_mug_scale_donka 67.42 MB
checkpoints/autoencoder_kl_v9_donka              5.42 MB
data/cache/diffusion_v9_donka                   24.92 MB
data/cache/audio_v0                           1359.79 MB
eval/videos                                     12.89 MB
```

所以最小迁移包主要由 `audio_v0` 决定，大约 1.5GB 级别。

## 10. 服务器环境建议

建议 Python：

```text
Python >= 3.10
CUDA PyTorch + torchaudio
ffmpeg
```

Conda/Miniforge 示例：

```bash
conda create -n taiko-diffusion python=3.11 -y
conda activate taiko-diffusion

# 按服务器 CUDA 版本安装 PyTorch，例如 CUDA 12.1：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -e .
pip install pillow matplotlib imageio
```

检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

ffmpeg -version
```

## 11. 服务器继续训练命令

从当前 v9 best 继续训练到 40 epoch：

```bash
python -m taiko_diffusion.train_latent_diffusion \
  --config configs/latent_diffusion_v9_mug_scale_donka.yaml \
  --epochs 40 \
  --resume-checkpoint checkpoints/latent_diffusion_v9_mug_scale_donka/best.pt
```

如果显存足够，建议先提高 batch size。修改：

```text
configs/latent_diffusion_v9_mug_scale_donka.yaml
```

当前：

```yaml
training:
  batch_size: 12
  num_workers: 0
```

服务器建议试：

```yaml
training:
  batch_size: 32
  num_workers: 4
```

如果显存还很多，再试：

```yaml
training:
  batch_size: 48 或 64
```

注意：batch size 改大后，学习率是否调整需要观察。第一轮建议保持：

```yaml
learning_rate: 0.0002
```

## 12. 服务器评估命令

全测试 guidance 2.5：

```bash
python -m taiko_diffusion.eval_latent_full_test \
  --checkpoint checkpoints/latent_diffusion_v9_mug_scale_donka/best.pt \
  --split data/cache/diffusion_v9_donka/test.csv \
  --stats data/cache/diffusion_v9_donka/stats.json \
  --batch-size 16 \
  --sample-steps 50 \
  --guidance-scale 2.5 \
  --output eval/full_test_latent_v9_server_g25.json
```

全测试 guidance 3.0：

```bash
python -m taiko_diffusion.eval_latent_full_test \
  --checkpoint checkpoints/latent_diffusion_v9_mug_scale_donka/best.pt \
  --split data/cache/diffusion_v9_donka/test.csv \
  --stats data/cache/diffusion_v9_donka/stats.json \
  --batch-size 16 \
  --sample-steps 50 \
  --guidance-scale 3.0 \
  --output eval/full_test_latent_v9_server_g30.json
```

生成单个 sample：

```bash
python -m taiko_diffusion.sample_latent_diffusion \
  --checkpoint checkpoints/latent_diffusion_v9_mug_scale_donka/best.pt \
  --split data/cache/diffusion_v9_donka/test.csv \
  --stats data/cache/diffusion_v9_donka/stats.json \
  --audio-split data/cache/audio_v0/test.csv \
  --audio-stats data/cache/audio_v0/stats.json \
  --output eval/videos/v9_server_sample.npz \
  --sample-steps 50 \
  --guidance-scale 2.5 \
  --seed 0 \
  --row-index 0
```

渲染 gameplay 视频：

```bash
python -m taiko_diffusion.render_gameplay_sample \
  --sample eval/videos/v9_server_sample.npz \
  --audio-split data/cache/audio_v0/test.csv \
  --output eval/videos/gameplay_v9_server_sample.mp4 \
  --fps 30
```

## 13. 下一步训练计划

### 阶段 A：确认服务器复现

目标：

```text
1. 能加载 v9 checkpoint。
2. 能跑一轮 sample。
3. 能跑 full-test eval。
4. 指标与本机接近。
```

通过标准：

```text
guidance 2.5:
  generated_top25_hit 接近 0.50
  ka_ratio_mae 接近 0.085
```

### 阶段 B：更大 batch 继续训练

目标：

```text
把 v9 从 epoch 22 best 继续训练到 40/60 epoch。
```

注意事项：

```text
1. 每隔 5-10 epoch 做 full-test 或抽样视频。
2. 不只看 val_loss，要看 onset/top25/ka_ratio 和 gameplay 主观效果。
3. 保存多个 checkpoint，避免 best.pt 被 loss 更低但生成更差的模型覆盖。
```

当前代码只保存 `best.pt`，后续建议改成同时保存：

```text
best.pt
last.pt
epoch_XX.pt
```

### 阶段 C：改进红蓝分布

当前首要质量问题不是音频对齐，而是局部红蓝配置。

计划：

```text
1. 加 ka_ratio 局部损失或 red/blue count loss。
2. 加 pattern-level 统计评估，例如连续 ka、ka run length、don/ka transition rate。
3. 在采样/解码阶段加入更温和的 ka_ratio calibration。
4. 对低 ka_ratio 条件样本强化训练，避免低密度全 don 曲乱出 ka。
```

### 阶段 D：整曲生成

当前只生成 512-frame window。

计划：

```text
1. 做 sliding-window generation。
2. 对重叠区域做概率融合。
3. 统一全曲 note count / ka_ratio / density。
4. 导出完整 TJA。
5. 渲染完整 gameplay 预览。
```

### 阶段 E：接入 encoder

当前 diffusion 使用真值条件。

计划：

```text
1. 先继续使用手动/真值条件把 diffusion 做稳。
2. 再接 encoder 预测条件。
3. 对比真值条件 vs encoder 条件生成质量。
4. 必要时 fine-tune diffusion 适配 encoder 预测分布。
```

推荐的 encoder 入口：

```text
main encoder:
  checkpoints/encoder_final_main/best.pt

note encoder:
  checkpoints/encoder_v8_note_type_log1p_handtiming_solo/best.pt
```

## 14. 打包建议

Linux/macOS 或 Git Bash：

```bash
tar -czf taiko-diffusion-v9-server.tar.gz \
  taiko_diffusion \
  configs \
  pyproject.toml \
  requirements.txt \
  README.md \
  PROJECT_STATUS_AND_SERVER_PLAN.md \
  data/cache/diffusion_v9_donka \
  data/cache/audio_v0 \
  checkpoints/autoencoder_kl_v9_donka \
  checkpoints/latent_diffusion_v9_mug_scale_donka \
  logs/latent_diffusion_v9_mug_scale_donka \
  eval/full_test_latent_v9_donka_epoch22_g25.json \
  eval/full_test_latent_v9_donka_epoch22_g30.json \
  eval/videos/gameplay_v9_sample_row0_seed0_g25.mp4 \
  eval/videos/gameplay_v9_low_row1089_seed1_g25.mp4 \
  eval/videos/gameplay_v9_mid_row98_seed2_g25.mp4 \
  eval/videos/gameplay_v9_high_row641_seed3_g25.mp4
```

Windows PowerShell 在项目根目录：

```powershell
tar -czf taiko-diffusion-v9-server.tar.gz `
  taiko_diffusion `
  configs `
  pyproject.toml `
  requirements.txt `
  README.md `
  PROJECT_STATUS_AND_SERVER_PLAN.md `
  data/cache/diffusion_v9_donka `
  data/cache/audio_v0 `
  checkpoints/autoencoder_kl_v9_donka `
  checkpoints/latent_diffusion_v9_mug_scale_donka `
  logs/latent_diffusion_v9_mug_scale_donka `
  eval/full_test_latent_v9_donka_epoch22_g25.json `
  eval/full_test_latent_v9_donka_epoch22_g30.json `
  eval/videos/gameplay_v9_sample_row0_seed0_g25.mp4 `
  eval/videos/gameplay_v9_low_row1089_seed1_g25.mp4 `
  eval/videos/gameplay_v9_mid_row98_seed2_g25.mp4 `
  eval/videos/gameplay_v9_high_row641_seed3_g25.mp4
```

如果服务器要重新构建 cache，再额外传：

```text
D:/taiko/ESE-master
D:/taiko/rating计算工具_11.25.xlsx
```

## 15. 当前优先级

推荐顺序：

```text
1. 迁移 v9 最小包到服务器。
2. 在服务器复现 sample + full-test。
3. 改训练脚本保存 last/epoch checkpoint。
4. 大 batch 继续训练 v9。
5. 同步看 gameplay 视频，不只看 loss。
6. 针对 ka_ratio 和局部 pattern 改损失。
7. 做整曲 generation/export。
8. 最后接 encoder 条件。
```

