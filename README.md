# Taiko Diffusion

## One-Click Audio To TJA

The deployable v15 pipeline generates a TJA directly from an audio file. It
requires the v15 diffusion checkpoint, the deployable neural hold-span
checkpoint, and FFmpeg on `PATH`; it does not read ESE charts or training
caches at inference time.

```bash
python -m taiko_diffusion.generate_audio_tja \
  --audio /path/to/song.ogg \
  --output generated_song.tja \
  --set-condition const=7 \
  --set-condition subdivision_bin=1 \
  --set-condition avg_density_bin=1
```

The command also writes `generated_song.npz`, containing the generated
probabilities and neural hold spans. Default checkpoints are:

```text
checkpoints/latent_diffusion_v15_full_catalog/best.pt
checkpoints/autoencoder_kl_v13_mug_holds/best.pt
checkpoints/autoencoder_kl_v13_mug_holds/latent_stats.json
checkpoints/hold_span_v2_deploy/best.pt
```

## Included Inference Assets

The repository includes the complete v15 inference asset set, approximately
88 MB in total. No training charts, audio cache samples, or source audio are
included.

```text
checkpoints/latent_diffusion_v15_full_catalog/best.pt
checkpoints/autoencoder_kl_v13_mug_holds/best.pt
checkpoints/autoencoder_kl_v13_mug_holds/latent_stats.json
checkpoints/hold_span_v2_deploy/best.pt
data/cache/diffusion_v15_full_catalog/stats.json
data/cache/audio_v15_full_catalog/stats.json
```

Install runtime dependencies and provide FFmpeg on the system `PATH`; no ESE
dataset, source chart, or cache rebuild is needed for one-click inference:

```bash
python -m pip install -r requirements.txt
```

Optional `--set-condition NAME=VALUE` values are `const`, `complex_bin`,
`subdivision_bin`, `hs_change_bin`, `bpm_rhythm_bin`, `note_type_bin`,
`avg_density_bin`, `peak_density_bin`, `big_note_ratio`,
`balloon_roll_ratio`, and `ka_ratio`. Use `--bpm VALUE` to override the
audio tempo detector with a static BPM. Set `bpm_rhythm_bin=0` for a static
BPM chart. Set `bpm_rhythm_bin=1` or `2` to enable conservative audio tempo
map detection and `#BPMCHANGE` export: only stable, bar-aligned changes are
written, while weak tempo evidence falls back to a static BPM. The current
full-length model accepts at most 8192 frames, approximately 380 seconds;
longer input is truncated and reported by the command.

Taiko Diffusion is an experimental project for Taiko no Tatsujin chart
representation learning and future audio-conditioned chart generation.

The first milestone is a neural chart encoder:

```text
.tja chart course -> time-grid tensor -> encoder -> chart attributes
```

Initial supervised targets come from `rating计算工具_11.25.xlsx`:

- `const`
- `complex`
- `avg_density`
- `peak_density`
- `note_type` for v1+
- `bpm_change`
- `hs_change`
- `rhythm`
- `combo`
- `roll_time`
- `balloon_num`

The project intentionally starts with conservative data alignment:

- Rating rows that cannot be matched to ESE are kept out of training.
- Ambiguous title matches are kept out of training.
- Courses containing branch commands are discarded for now.
- Matches with suspicious combo differences are written to a review file.

## Local Inputs

Expected workspace layout:

```text
D:\taiko
  ESE-master\ese\
  rating计算工具_11.25.xlsx
  taiko-diffusion\
```

## Miniforge

Use the local Miniforge Python:

```powershell
D:\miniforge3\python.exe --version
```

Prepare the first manifest:

```powershell
cd D:\taiko\taiko-diffusion
D:\miniforge3\python.exe -m taiko_diffusion.data.prepare_manifest `
  --ese-root ..\ESE-master\ese `
  --rating-xlsx ..\rating计算工具_11.25.xlsx `
  --output-dir data\manifests
```

The main training input for the first encoder should be:

```text
data\manifests\strict_matched_dataset.csv
```

Build the first numpy tensor cache:

```powershell
cd D:\taiko\taiko-diffusion
D:\miniforge3\python.exe -m taiko_diffusion.data.build_cache `
  --config configs\encoder_v0.yaml `
  --output-dir data\cache\encoder_v0
```

Each cache sample is a compressed `.npz` with:

```text
x: [8192, 16] float32 time-grid tensor
y: [10] float32 target vector
channels: channel names
label_names: target names
duration_frames: active chart length
```

The first version uses these channels:

```text
don, ka, big_don, big_ka,
roll_start, roll_body, roll_end,
balloon_start, balloon_body, balloon_end,
gogo, barline, bpm, scroll, measure, active
```

The current recommended encoder is `encoder_v1`. It adds explicit BPM/scroll
change event and delta channels:

```text
bpm_change_event, bpm_delta, scroll_change_event, scroll_delta
```

Its cache and split are:

```text
data\cache\encoder_v1
data\splits\encoder_v1
```

Create deterministic train/validation/test splits and label normalization stats:

```powershell
cd D:\taiko\taiko-diffusion
D:\miniforge3\python.exe -m taiko_diffusion.data.split_stats `
  --config configs\encoder_v0.yaml `
  --index data\cache\encoder_v0\index.csv
```

This writes:

```text
data\splits\encoder_v0\train.csv
data\splits\encoder_v0\val.csv
data\splits\encoder_v0\test.csv
data\splits\encoder_v0\label_stats.json
```

Train the first encoder after PyTorch is installed:

```powershell
cd D:\taiko\taiko-diffusion
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.train_encoder `
  --config configs\encoder_v0.yaml
```

Train/evaluate the current recommended encoder:

```powershell
cd D:\taiko\taiko-diffusion
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.train_encoder `
  --config configs\encoder_v1.yaml

D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.eval_encoder `
  --checkpoint checkpoints\encoder_v1\best.pt `
  --split-dir data\splits\encoder_v1 `
  --stats data\splits\encoder_v1\label_stats.json `
  --output-dir eval\encoder_v1
```

## Current Results

`encoder_v1` is the best default checkpoint for now:

```text
checkpoint: checkpoints\encoder_v1\best.pt
best_epoch: 100
test const        MAE 0.406, Spearman 0.955
test complex      MAE 6.404, Spearman 0.925
test avg_density  MAE 3.275, Spearman 0.971
test peak_density MAE 6.430, Spearman 0.921
test note_type    MAE 10.957, Spearman 0.549
test bpm_change   MAE 4.874, Spearman 0.636
test hs_change    MAE 7.509, Spearman 0.763
test rhythm       MAE 11.551, Spearman 0.698
test combo        MAE 35.609, Spearman 0.983
test roll_time    MAE 0.553, Spearman 0.967
test balloon_num  MAE 11.088, Spearman 0.890
```

Compared with `encoder_v0`, v1 keeps the strong chart-size/density targets and
substantially improves the previously weak `bpm_change`, `hs_change`, and
`rhythm` targets. This supports the basic plan: the chart tensor contains
enough information for a neural encoder to recover most rating-table attributes.

`encoder_v2` is an experiment with grouped output heads and extra positive
sample weighting for sparse labels. It improves `hs_change` MAE but hurts the
overall encoder, especially combo/roll/balloon and note-type ranking, so it is
not the default:

```text
checkpoint: checkpoints\encoder_v2\best.pt
best_epoch: 92
test bpm_change MAE 5.342, Spearman 0.664
test hs_change  MAE 6.036, Spearman 0.729
test rhythm     MAE 11.813, Spearman 0.675
```

`encoder_v3` is a sparse-label multitask experiment. It keeps the same chart
grid as v1, but adds binary classification heads for:

```text
has_bpm_change
has_hs_change
```

During training, `bpm_change` and `hs_change` regression losses are applied only
to positive samples, while the classifier learns whether the score should be
non-zero. During evaluation, the predicted score is scaled by the classifier
probability. This works better for the sparse BPM/HS targets:

```text
checkpoint: checkpoints\encoder_v3\best.pt
best_epoch: 44
test bpm_change MAE 3.605, Spearman 0.708
test hs_change  MAE 5.665, Spearman 0.832
```

However, v3 is not a full replacement for v1. The sparse heads pull capacity
away from the shared chart representation and reduce several core metrics:

```text
test combo        v1 MAE 35.609, Spearman 0.983 -> v3 MAE 58.527, Spearman 0.953
test avg_density  v1 MAE 3.275,  Spearman 0.971 -> v3 MAE 3.941,  Spearman 0.944
test rhythm       v1 MAE 11.551, Spearman 0.698 -> v3 MAE 12.074, Spearman 0.584
```

Use v1 as the default general-purpose chart encoder. Use v3 as a BPM/HS
specialist or as evidence for the next architecture: separate the sparse
BPM/HS heads from the core style/difficulty representation, instead of forcing
all targets through one shared bottleneck.

`encoder_v4` implements that separation directly: the first downsampling blocks
are shared, then the model splits into a core branch and a BPM/HS event branch.
This recovers part of the v1 general-style performance while keeping the v3
sparse-label idea:

```text
checkpoint: checkpoints\encoder_v4\best.pt
best_epoch: 69
test bpm_change MAE 3.655, Spearman 0.747
test hs_change  MAE 7.035, Spearman 0.790
test combo       MAE 42.901, Spearman 0.977
test balloon_num MAE 9.454, Spearman 0.911
```

`encoder_v4_fullcore` restores the core branch to three Transformer layers. It
improves `note_type`, `hs_change`, `roll_time`, and `balloon_num`, but hurts
`complex`, density, and combo enough that it should not become the default:

```text
checkpoint: checkpoints\encoder_v4_fullcore\best.pt
best_epoch: 52
test note_type   MAE 10.466, Spearman 0.637
test bpm_change  MAE 3.987,  Spearman 0.735
test hs_change   MAE 5.976,  Spearman 0.811
test combo       MAE 64.832, Spearman 0.945
```

Current practical recommendation:

```text
general chart/style encoder: checkpoints\encoder_v1\best.pt
BPM control/ranking helper:  checkpoints\encoder_v4\best.pt
HS specialist helper:        checkpoints\encoder_v3\best.pt or encoder_v4_fullcore
note_type experiment:        checkpoints\encoder_v4_fullcore\best.pt
```

`encoder_v5` tests a cleaner label design:

```text
remove raw note_type / rhythm regression
add objective note controls:
  big_note_ratio
  ka_ratio
  alternation_rate
add subjective_complexity_bin:
  residualize note_type and rhythm against const/log(combo)/avg_density/peak_density
  combine the two residuals
  split train samples into low/mid/high tertiles
add event-derived input channels:
  bpm_abs_delta
  bpm_event_local_density
  bpm_event_cumsum
  scroll_abs_delta
  scroll_event_local_density
  scroll_event_cumsum
```

The cache is derived from `encoder_v1` tensors, so it reuses the same matched
charts and split size:

```text
cache: data\cache\encoder_v5
split: data\splits\encoder_v5
checkpoint: checkpoints\encoder_v5\best.pt
best_epoch: 90
```

Test results:

```text
big_note_ratio              MAE 0.016, Spearman 0.806
ka_ratio                    MAE 0.038, Spearman 0.814
alternation_rate            MAE 0.049, Spearman 0.437
bpm_change                  MAE 4.363, Spearman 0.736
hs_change                   MAE 7.902, Spearman 0.843
subjective_complexity_bin   MAE 0.565, Spearman 0.424
subjective_complexity_bin rounded accuracy: 52.3%
within one bin: 97.7%
```

Interpretation:

```text
Objective note controls are much cleaner than raw note_type.
big_note_ratio and ka_ratio are already useful.
alternation_rate needs a better local color-pattern representation.
BPM ranking stays strong but MAE is worse than v4.
HS ranking is the best so far, but score calibration is high.
subjective_complexity_bin is learnable, but ordinal regression biases toward the middle bin.
The next improvement should use a real classification head for subjective_complexity_bin.
```

`encoder_v5_class` adds that CrossEntropy classification head directly on the
core branch. It improves the `subjective_complexity_bin` test accuracy, but the
classification loss damages the shared representation too much:

```text
checkpoint: checkpoints\encoder_v5_class\best.pt
best_epoch: 30
subjective_complexity_bin accuracy: 60.2%
big_note_ratio Spearman: 0.422
ka_ratio Spearman: 0.426
roll_time Spearman: 0.253
balloon_num Spearman: 0.153
```

`encoder_v5_class_detached` keeps the same class head, but detaches the core
features before the class head so the classification loss trains only the class
head and does not pull the shared chart representation away from the regression
targets:

```text
checkpoint: checkpoints\encoder_v5_class_detached\best.pt
best_epoch: 60
subjective_complexity_bin accuracy: 56.3%
within one bin: 89.8%
big_note_ratio MAE 0.008, Spearman 0.948
ka_ratio MAE 0.021, Spearman 0.933
alternation_rate MAE 0.037, Spearman 0.758
bpm_change MAE 3.754, Spearman 0.734
hs_change MAE 6.457, Spearman 0.858
roll_time MAE 0.965, Spearman 0.952
balloon_num MAE 10.528, Spearman 0.898
```

Current v5-family recommendation:

```text
use encoder_v5_class_detached for objective note controls:
  big_note_ratio
  ka_ratio
  alternation_rate
  hs_change

use encoder_v5_class only if subjective_complexity_bin classification is the only concern

do not replace encoder_v1 as the general difficulty/style encoder yet
```

`encoder_v6_pattern` extends `encoder_v5_class_detached` with local pattern
input tracks:

```text
note_density_short
note_density_long
color_change_event
color_change_density
prev_note_interval
next_note_interval
bar_phase
```

After retraining with the same split seed as v5, the result is mixed:

```text
checkpoint: checkpoints\encoder_v6_pattern\best.pt
best_epoch: 51
avg_density MAE 2.985, Spearman 0.969
big_note_ratio MAE 0.008, Spearman 0.962
bpm_change MAE 3.766, Spearman 0.740
hs_change MAE 6.490, Spearman 0.852
combo MAE 48.636, Spearman 0.970
roll_time MAE 0.887, Spearman 0.962
subjective_complexity_bin accuracy: 57.8%
```

Compared with `encoder_v5_class_detached`, v6 improves average density, combo,
roll time, big-note ratio, and subjective-bin accuracy slightly. It hurts
complex, ka ratio, alternation rate, peak density, and the subjective bin's
within-one-bin rate. Keep it as a pattern-track experiment, not as the new
default.

`encoder_v7_speed` tests merging `bpm_change` and `hs_change` into one target:

```text
speed_change = max(bpm_change, hs_change)
```

This matches the data distribution better than `bpm_change` alone:

```text
train speed_change zero rate: 46.6%
test speed_change zero rate: 50.8%
```

Result:

```text
checkpoint: checkpoints\encoder_v7_speed\best.pt
best_epoch: 23
speed_change MAE 7.482, Spearman 0.858
speed_change > 25 detection:
  accuracy 89.1%
  precision 71.4%
  recall 93.8%
speed_change > 50 detection:
  accuracy 90.6%
  precision 68.8%
  recall 61.1%
```

Interpretation:

```text
Merging BPM and HS is viable for generation conditioning.
speed_change is better as a unified "significant speed/readability change"
control than as two separate noisy controls.
This v7 checkpoint is not a full replacement for v6 because note-composition
targets degrade:
  big_note_ratio Spearman 0.962 -> 0.424
  ka_ratio Spearman 0.769 -> 0.425
  alternation_rate Spearman 0.642 -> 0.387
Use v7 only for speed_change, or build the final condition vector by combining
v6/v5 note controls with v7 speed_change.
```

`encoder_v8_neural` tests the cleaner generation-conditioning split:

```text
neural targets:
  const
  complex
  hs_change
  rhythm_processing_bin

direct chart statistics, not neural targets:
  avg_density
  peak_density
  big_note_ratio
  balloon_roll_ratio
```

The v8 cache collapses big notes into the normal red/blue lanes before the
network sees the chart:

```text
don = don + big_don
ka = ka + big_ka
```

`rhythm_processing_bin` is built from the Excel rhythm-processing value:

```text
rhythm = (0.9 * note_type^10 + 0.1 * bpm_change^10)^(1/10)
rhythm_processing_bin = tertile(residualize(rhythm, const + log(combo) + avg_density + peak_density))
```

The residualization must use the original Excel density labels before replacing
`avg_density` and `peak_density` with physical chart statistics.

Result:

```text
checkpoint: checkpoints\encoder_v8_neural\best.pt
best_epoch: 24
const MAE 0.431, Spearman 0.946
complex MAE 5.008, Spearman 0.955
hs_change MAE 6.789, Spearman 0.828
rhythm_processing_bin accuracy: 56.3%
```

Interpretation:

```text
v8 is the cleanest current encoder layout for generation conditions.
complex improves strongly compared with v6.
const and hs_change remain usable.
rhythm_processing_bin is semantically cleaner than subjective_complexity_bin,
but it is still only a weak three-class signal.
big_note_ratio should be computed directly, because v8 intentionally removes
big-note information from the neural input.
```

`encoder_v8_abs_bin` replaces the residual rhythm bin with an absolute
Excel-rhythm bin:

```text
rhythm_processing_abs_bin = tertile(rhythm)
train thresholds: 5.360, 25.819
```

The label is statistically cleaner than the residual bin, because it no longer
puts `rhythm = 0` charts into the high class. However, the first balanced
training run did not improve classification:

```text
checkpoint: checkpoints\encoder_v8_abs_bin\best.pt
best_epoch: 16
const MAE 0.356, Spearman 0.953
complex MAE 5.408, Spearman 0.947
hs_change MAE 6.857, Spearman 0.844
rhythm_processing_abs_bin accuracy: 46.1%
```

A stronger classification run lets the class loss update the core branch and
uses a higher class-loss weight:

```text
checkpoint: checkpoints\encoder_v8_abs_bin_classstrong\best.pt
best_epoch: 29
const MAE 0.541, Spearman 0.906
complex MAE 8.720, Spearman 0.889
hs_change MAE 7.175, Spearman 0.837
rhythm_processing_abs_bin accuracy: 59.4%
```

Interpretation:

```text
Absolute rhythm bin is a cleaner label than residual rhythm bin.
It can be pushed to better classification accuracy, but doing so creates clear
multi-task interference and hurts const/complex.
Use encoder_v8_abs_bin_classstrong only if rhythm-bin classification is the
priority.
Use encoder_v8_neural or encoder_v8_abs_bin when const/complex quality matters
more.
```

`encoder_v8_abs_bin_solo` trains only `rhythm_processing_abs_bin`:

```text
checkpoint: checkpoints\encoder_v8_abs_bin_solo\best.pt
best_epoch: 24
val accuracy: 63.0%
test accuracy: 55.5%
test Spearman: 0.584
```

The solo model does not beat the class-strong multi-task version on test. It
overpredicts the middle class:

```text
test true counts: 45, 47, 36
test pred counts: 20, 77, 31
```

Keep `encoder_v8_abs_bin_classstrong` as the current best rhythm-bin specialist.

`encoder_v8_semantic_bin_solo` uses semantic absolute rhythm thresholds instead
of tertiles:

```text
rhythm_processing_semantic_bin:
  0: rhythm < 10
  1: 10 <= rhythm < 40
  2: rhythm >= 40

train counts: 473, 272, 272
test counts: 57, 42, 29
```

Solo result:

```text
checkpoint: checkpoints\encoder_v8_semantic_bin_solo\best.pt
best_epoch: 16
val accuracy: 52.0%
test accuracy: 59.4%
test Spearman: 0.569
```

Test confusion:

```text
true low  -> 46 low, 10 mid, 1 high
true mid  -> 16 low, 16 mid, 10 high
true high -> 5 low, 10 mid, 14 high
```

This semantic split is cleaner and gives a stable low class, but mid/high are
still hard. It is comparable to `encoder_v8_abs_bin_classstrong` for overall
accuracy while avoiding the equal-tertile boundary around very low rhythm.

`encoder_v8_semantic50_bin_solo` moves the high threshold upward:

```text
rhythm_processing_semantic50_bin:
  0: rhythm < 10
  1: 10 <= rhythm < 50
  2: rhythm >= 50

train counts: 473, 350, 194
test counts: 57, 51, 20
```

Solo result:

```text
checkpoint: checkpoints\encoder_v8_semantic50_bin_solo\best.pt
best_epoch: 15
val accuracy: 53.5%
test accuracy: 59.4%
test Spearman: 0.536
```

Test confusion:

```text
true low  -> 42 low, 14 mid, 1 high
true mid  -> 19 low, 26 mid, 6 high
true high -> 1 low, 11 mid, 8 high
```

Compared with the 10/40 split, 10/50 improves the middle class but loses high
recall because the high class becomes small. It is not a clear overall upgrade.

`encoder_v8_split_rhythm_bins` splits rhythm into separate note/BPM sources:

```text
note_rhythm_bin:
  0: note_type < 10
  1: 10 <= note_type < 40
  2: note_type >= 40

bpm_rhythm_bin:
  0: bpm_change = 0
  1: 0 < bpm_change < 25
  2: bpm_change >= 25
```

Result:

```text
checkpoint: checkpoints\encoder_v8_split_rhythm_bins\best.pt
best_epoch: 21

note_rhythm_bin test accuracy: 57.8%
bpm_rhythm_bin test accuracy: 90.6%
```

Test confusion:

```text
note_rhythm_bin:
  true low  -> 63 low, 0 mid, 7 high
  true mid  -> 26 low, 1 mid, 10 high
  true high -> 10 low, 1 mid, 10 high

bpm_rhythm_bin:
  true none -> 83 none, 8 mid, 0 high
  true mid  -> 2 none, 21 mid, 0 high
  true high -> 0 none, 2 mid, 12 high
```

Interpretation:

```text
Splitting proves BPM rhythm is highly learnable from the current event tracks.
The note/叩き分け side is still weak and collapses away from the middle class.
For generation conditions, use bpm_rhythm_bin as a reliable speed/rhythm source
control, but do not trust note_rhythm_bin yet as a replacement for note_type.
```

`encoder_v8_note_high25_solo` tests a binary note-side rhythm label:

```text
note_rhythm_high25:
  0: note_type < 25
  1: note_type >= 25

train counts: 732, 285
test counts: 103, 25
```

Result:

```text
checkpoint: checkpoints\encoder_v8_note_high25_solo\best.pt
best_epoch: 2
test argmax accuracy: 79.7%
test Spearman: 0.104
```

This accuracy is misleading because the model predicts almost everything as
low:

```text
test true counts: 103, 25
test pred counts: 123, 5
default threshold high recall: 8.0%
```

Lowering the probability threshold improves recall only by creating many false
positives. The note-side rhythm label remains weak without audio or a better
label design.

`encoder_v8_note_high25_multitask` trains `note_rhythm_high25` together with
`const`, `complex`, `hs_change`, and `bpm_rhythm_bin`:

```text
checkpoint: checkpoints\encoder_v8_note_high25_multitask\best.pt
best_epoch: 14

note_rhythm_high25 test argmax accuracy: 83.6%
note_rhythm_high25 test Spearman: 0.361
bpm_rhythm_bin test accuracy: 89.8%
```

Default argmax still favors low, but it is better than solo:

```text
test true counts: 103 low, 25 high
test pred counts: 120 low, 8 high
high precision: 75.0%
high recall: 24.0%
```

With a lower high-probability threshold:

```text
threshold 0.30:
  high precision: 50.0%
  high recall: 52.0%
  F1: 0.510
```

Multitask context helps the note-side binary label, but it remains much weaker
than `bpm_rhythm_bin` and also hurts `const`/`complex` quality.

`encoder_v8_note_residual_bin_solo` residualizes `note_type` against
`const + log(combo) + avg_density + peak_density`, then bins the residual into
tertiles:

```text
note_rhythm_residual_bin thresholds: -10.518, 3.896
train counts: 339, 339, 339
test counts: 40, 53, 35
```

Result:

```text
checkpoint: checkpoints\encoder_v8_note_residual_bin_solo\best.pt
best_epoch: 6
test accuracy: 50.0%
test Spearman: 0.262
```

Test confusion:

```text
true low  -> 34 low, 5 mid, 1 high
true mid  -> 22 low, 30 mid, 1 high
true high -> 18 low, 17 mid, 0 high
```

Residualizing the note side makes the label more abstract and does not help the
chart-only model. It completely misses the high residual class on test.

`encoder_v8_note_high25_hand_solo` adds alternating-hand surface-change tracks:

```text
left_hand_note
right_hand_note
left_hand_change_event
right_hand_change_event
left_hand_change_density
right_hand_change_density
hand_change_total_density
hand_change_imbalance
```

The tracks assume full alternating hands and mark a hand-change event when:

```text
seq[i] != seq[i - 2]
```

Result:

```text
checkpoint: checkpoints\encoder_v8_note_high25_hand_solo\best.pt
best_epoch: 31
test argmax accuracy: 84.4%
test Spearman: 0.446
```

Test confusion:

```text
true low  -> 97 low, 6 high
true high -> 14 low, 11 high
```

This is the first chart-only note-side setup with a useful default high recall:

```text
high precision: 64.7%
high recall: 44.0%
```

Compared with the old solo binary setup:

```text
old solo high recall: 8.0%
hand-track solo high recall: 44.0%
```

Separating left/right hand-change timing is materially useful for note-side
rhythm. It still is not as reliable as `bpm_rhythm_bin`, but it is a real
improvement and worth carrying into the next encoder.

`encoder_v8_note_high25_hand_multitask` combines:

```text
const
complex
hs_change
bpm_rhythm_bin
note_rhythm_high25
```

with the same alternating-hand tracks. The non-detached version trains the
classification losses through the shared core representation:

```text
checkpoint: checkpoints\encoder_v8_note_high25_hand_multitask\best.pt
best_epoch: 8
```

Test result:

```text
const               MAE 0.695  Spearman 0.852
complex             MAE 9.289  Spearman 0.845
hs_change           MAE 7.857  Spearman 0.851
bpm_rhythm_bin      accuracy 88.3%  Spearman 0.830
note_rhythm_high25  accuracy 78.1%  Spearman 0.304
```

`note_rhythm_high25` confusion:

```text
true low  -> 89 low, 14 high
true high -> 14 low, 11 high
```

At threshold 0.30, the note high class gets:

```text
precision 34.0%
recall    72.0%
F1        0.462
```

This version hurts the core encoder and is not better than the hand solo note
classifier.

`encoder_v8_note_high25_hand_multitask_detach` uses the same targets/cache/split
but sets:

```text
class_detach: true
```

so the classification heads read the core representation without pushing their
losses back through it.

```text
checkpoint: checkpoints\encoder_v8_note_high25_hand_multitask_detach\best.pt
best_epoch: 16
```

Test result:

```text
const               MAE 0.485  Spearman 0.922
complex             MAE 5.964  Spearman 0.933
hs_change           MAE 6.895  Spearman 0.823
bpm_rhythm_bin      accuracy 85.9%  Spearman 0.819
note_rhythm_high25  accuracy 85.2%  Spearman 0.450
```

`bpm_rhythm_bin` confusion:

```text
true none -> 87 none, 4 mid, 0 high
true mid  -> 7 none, 16 mid, 0 high
true high -> 0 none, 7 mid, 7 high
```

`note_rhythm_high25` confusion:

```text
true low  -> 100 low, 3 high
true high -> 16 low, 9 high
```

At threshold 0.20, the note high class gets:

```text
precision 37.0%
recall    80.0%
F1        0.506
```

At threshold 0.50:

```text
precision 75.0%
recall    36.0%
F1        0.486
```

`encoder_v8_note_high25_hand_multitask_classbranch` adds an independent
classification branch:

```text
class_detach: true
class_branch_downsample_layers: 2
class_transformer_layers: 1
```

This lets classification train its own high-level branch without backpropagating
into the shared stem/core path.

```text
checkpoint: checkpoints\encoder_v8_note_high25_hand_multitask_classbranch\best.pt
best_epoch: 12
```

Test result:

```text
const               MAE 0.471  Spearman 0.933
complex             MAE 5.711  Spearman 0.939
hs_change           MAE 6.804  Spearman 0.838
bpm_rhythm_bin      accuracy 82.8%  Spearman 0.738
note_rhythm_high25  accuracy 82.0%  Spearman 0.332
```

`bpm_rhythm_bin` confusion:

```text
true none -> 81 none, 9 mid, 1 high
true mid  -> 7 none, 14 mid, 2 high
true high -> 0 none, 3 mid, 11 high
```

`note_rhythm_high25` confusion:

```text
true low  -> 97 low, 6 high
true high -> 17 low, 8 high
```

Conclusion: for a single practical chart-style encoder, the detached hand
multitask version is still the better balanced candidate. The classbranch
variant is slightly better for `const/complex/hs_change`, but its classification
heads are weaker. For pure note-side detection, the hand solo model is still
slightly cleaner at the default threshold, while the detached multitask model
gives better low-threshold recall.

`encoder_v8_note_type_hand_solo` removes the binary `note_rhythm_high25` label
and trains the raw continuous Excel `note_type` / `叩き分け` value directly with
the alternating-hand tracks.

```text
checkpoint: checkpoints\encoder_v8_note_type_hand_solo\best.pt
best_epoch: 5
```

Test result:

```text
note_type MAE 13.592  Pearson 0.512  Spearman 0.550
```

The model learns ordering better than the binary target, but raw regression
compresses the long tail:

```text
true quantiles: 0.000, 0.000, 0.310, 7.655, 18.097, 52.494, 92.440
pred quantiles: 5.130, 5.456, 5.509, 13.171, 22.860, 35.466, 49.228
```

If the raw prediction is converted back into `note_type >= 25`, the best tested
threshold is around predicted value `30`:

```text
accuracy 82.0%
precision 55.0%
recall    44.0%
F1        0.489
```

`encoder_v8_note_type_log1p_hand_solo` uses the same raw label but trains it with
a `log1p` transform to reduce the long-tail effect.

```text
checkpoint: checkpoints\encoder_v8_note_type_log1p_hand_solo\best.pt
best_epoch: 11
```

Test result:

```text
note_type MAE 12.128  Pearson 0.561  Spearman 0.574
```

Predictions still underestimate extreme highs, but less badly than identity
training:

```text
true quantiles: 0.000, 0.000, 0.310, 7.655, 18.098, 52.494, 92.440
pred quantiles: 0.575, 1.275, 2.636, 8.163, 21.492, 33.308, 46.677
```

Derived `note_type >= 25` with predicted threshold `30`:

```text
accuracy 85.9%
precision 70.6%
recall    48.0%
F1        0.571
```

Conclusion: raw continuous `note_type` with `log1p` is a better note-side signal
than the hard `high25` binary label if we want a style condition. It provides a
usable continuous axis and can still be thresholded later, although the absolute
score is not precise enough to treat as a calibrated rating-table value.

`encoder_v8_note_type_log1p_hand_multitask_detach` trains the full current
neural target set together:

```text
const
complex
hs_change
bpm_rhythm_bin
note_type
```

with continuous `note_type` using `log1p`, and `bpm_rhythm_bin` kept as a
3-class classification head. The detached version keeps the BPM classification
head from backpropagating into the shared core representation.

```text
checkpoint: checkpoints\encoder_v8_note_type_log1p_hand_multitask_detach\best.pt
best_epoch: 11
```

Test result:

```text
const           MAE 0.526   Spearman 0.922
complex         MAE 5.518   Spearman 0.932
hs_change       MAE 6.750   Spearman 0.833
bpm_rhythm_bin  accuracy 75.8%  Spearman 0.649
note_type       MAE 11.601  Spearman 0.585
```

This version improves `note_type`, but it is not suitable as the balanced
encoder because `bpm_rhythm_bin` predicts no high class:

```text
true none -> 85 none, 6 mid, 0 high
true mid  -> 11 none, 12 mid, 0 high
true high -> 2 none, 12 mid, 0 high
```

`encoder_v8_note_type_log1p_hand_multitask` uses the same full target set but
sets:

```text
class_detach: false
```

so the BPM classification loss can shape the core representation.

```text
checkpoint: checkpoints\encoder_v8_note_type_log1p_hand_multitask\best.pt
best_epoch: 13
```

Test result:

```text
const           MAE 0.543   Spearman 0.902
complex         MAE 7.912   Spearman 0.904
hs_change       MAE 6.978   Spearman 0.841
bpm_rhythm_bin  accuracy 88.3%  Spearman 0.824
note_type       MAE 11.948  Spearman 0.576
```

`bpm_rhythm_bin` confusion:

```text
true none -> 80 none, 11 mid, 0 high
true mid  -> 2 none, 21 mid, 0 high
true high -> 0 none, 2 mid, 12 high
```

Derived `note_type >= 25` from the continuous prediction:

```text
threshold 18:
  accuracy 80.5%
  precision 50.0%
  recall    52.0%
  F1        0.510

threshold 25:
  accuracy 84.4%
  precision 66.7%
  recall    40.0%
  F1        0.500
```

Conclusion: for the current all-in-one neural encoder, the non-detached
continuous-note multitask version is the better balanced candidate. It restores
`bpm_rhythm_bin` to the old strong level while keeping continuous `note_type`
slightly better than the previous binary note label. The detached version is
useful as an analysis reference but should not be the default because it loses
BPM high cases.

`encoder_v8_note_type_log1p_halfhand_solo` adds half-alternating hand tracks on
top of the full-alternating hand tracks. The half-alternating assumption is:

```text
8th-note spacing or slower: do not switch hands
denser than 8th-note spacing: switch hands
```

The local 8th-note threshold is computed from the BPM track. The added channels
are:

```text
half_left_hand_note
half_right_hand_note
half_left_hand_change_event
half_right_hand_change_event
half_left_hand_change_density
half_right_hand_change_density
half_hand_change_total_density
half_hand_change_imbalance
```

Solo result:

```text
checkpoint: checkpoints\encoder_v8_note_type_log1p_halfhand_solo\best.pt
best_epoch: 20
note_type MAE 11.291  Pearson 0.644  Spearman 0.632
```

This is a clear improvement over the previous full-alternating-only solo model:

```text
full-hand solo:
  MAE 12.128
  Pearson 0.561
  Spearman 0.574

full+half-hand solo:
  MAE 11.291
  Pearson 0.644
  Spearman 0.632
```

Derived `note_type >= 25` from the solo half-hand prediction:

```text
threshold 10:
  accuracy 78.1%
  precision 46.3%
  recall    76.0%
  F1        0.576

threshold 15:
  accuracy 83.6%
  precision 59.1%
  recall    52.0%
  F1        0.553
```

`encoder_v8_note_type_log1p_halfhand_multitask` adds the same half-hand tracks
to the all-in-one encoder.

```text
checkpoint: checkpoints\encoder_v8_note_type_log1p_halfhand_multitask\best.pt
best_epoch: 19
```

Test result:

```text
const           MAE 0.514   Spearman 0.885
complex         MAE 8.361   Spearman 0.911
hs_change       MAE 6.654   Spearman 0.709
bpm_rhythm_bin  accuracy 87.5%  Spearman 0.807
note_type       MAE 12.410  Spearman 0.543
```

For continuous `note_type`, the all-in-one half-hand model is worse than the
previous all-in-one model:

```text
without half-hand: MAE 11.948  Spearman 0.576
with half-hand:    MAE 12.410  Spearman 0.543
```

However, its derived `note_type >= 25` signal is stronger at a tuned threshold:

```text
threshold 18:
  accuracy 85.2%
  precision 59.4%
  recall    76.0%
  F1        0.667
```

Conclusion: half-hand tracks are genuinely useful for note-side information.
They clearly improve the solo continuous `note_type` encoder. In the all-in-one
encoder, they do not improve the calibrated continuous axis, but they do improve
the ability to detect high note-side complexity after thresholding. Keep the
tracks available, but treat the all-in-one result as a tradeoff rather than an
unconditional replacement.

`encoder_v8_note_type_log1p_handtiming_solo` adds explicit hand-change timing
regularity tracks on top of the full-hand and half-hand tracks:

```text
hand_change_irregularity
hand_change_burstiness
half_hand_change_irregularity
half_hand_change_burstiness
```

Definitions:

```text
irregularity:
  local coefficient of variation of adjacent hand-change intervals.
  Higher means the change positions are less regular.

burstiness:
  short-window hand-change density divided by long-window density.
  Higher means local hand changes arrive as a burst rather than evenly.
```

Solo result:

```text
checkpoint: checkpoints\encoder_v8_note_type_log1p_handtiming_solo\best.pt
best_epoch: 22
note_type MAE 11.157  RMSE 16.673  Pearson 0.658  Spearman 0.669
```

Comparison:

```text
full-hand solo:
  MAE 12.128
  Pearson 0.561
  Spearman 0.574

full+half-hand solo:
  MAE 11.291
  Pearson 0.644
  Spearman 0.632

full+half-hand+timing solo:
  MAE 11.157
  Pearson 0.658
  Spearman 0.669
```

Derived `note_type >= 25` from the timing solo prediction:

```text
threshold 25:
  accuracy 84.4%
  precision 60.0%
  recall    60.0%
  F1        0.600

threshold 28:
  accuracy 85.9%
  precision 68.4%
  recall    52.0%
  F1        0.591
```

Conclusion: the user's regularity hypothesis is supported. Explicit timing
irregularity/burstiness tracks improve the solo continuous note-side encoder
more than half-hand tracks alone. This is currently the best `note_type`
encoder.

`encoder_v8_note_type_logit100_handtiming_multitask` changes the all-in-one
encoder's `note_type` transform from `log1p` to a bounded sigmoid/logit
transform:

```text
train transform: logit((note_type + 0.5) / 101)
eval inverse:    101 * sigmoid(x) - 0.5, clipped to 0..100
```

Targets:

```text
const
complex
hs_change
bpm_rhythm_bin
note_type
```

Inputs use full-hand, half-hand, and hand-timing tracks.

```text
checkpoint: checkpoints\encoder_v8_note_type_logit100_handtiming_multitask\best.pt
best_epoch: 12
```

Test result:

```text
const           MAE 0.561   Spearman 0.881
complex         MAE 7.742   Spearman 0.888
hs_change       MAE 7.707   Spearman 0.847
bpm_rhythm_bin  accuracy 88.3%  Spearman 0.824
note_type       MAE 11.918  Pearson 0.533  Spearman 0.579
```

Compared with the previous all-in-one `log1p` model:

```text
log1p all-in-one:
  note_type MAE 11.948  Spearman 0.576

logit100 all-in-one:
  note_type MAE 11.918  Spearman 0.579
```

So the bounded sigmoid/logit transform is only a tiny improvement for
`note_type`. It does let the model predict a higher maximum value:

```text
logit100 prediction quantiles:
  min 0.928, median 5.705, 75% 15.248, 90% 31.575, max 70.691
```

but high true values are still heavily underpredicted:

```text
true 60..100:
  true_mean 71.37
  pred_mean 31.79
  MAE 39.58
```

`encoder_v8_allfeatures_logit100_handtiming_multitask` adds the direct physical
statistics as auxiliary targets:

```text
avg_density
peak_density
big_note_ratio
balloon_roll_ratio
```

These are still intended to be computed directly in the final pipeline, but
this experiment tested whether predicting them as auxiliary tasks improves the
shared representation.

```text
checkpoint: checkpoints\encoder_v8_allfeatures_logit100_handtiming_multitask\best.pt
best_epoch: 21
```

Test result:

```text
const               MAE 0.524   Spearman 0.891
complex             MAE 9.756   Spearman 0.815
avg_density         MAE 0.594   Spearman 0.841
peak_density        MAE 1.244   Spearman 0.848
big_note_ratio      MAE 0.028   Spearman 0.289
balloon_roll_ratio  MAE 0.335   Spearman 0.109
hs_change           MAE 7.363   Spearman 0.894
bpm_rhythm_bin      accuracy 89.1%  Spearman 0.847
note_type           MAE 12.924  Pearson 0.411  Spearman 0.503
```

Conclusion: adding physical-stat auxiliary targets does not help `note_type`;
it makes note-side prediction substantially worse and also weakens `complex`.
The only clear gains are `bpm_rhythm_bin` and `hs_change`, which are not worth
the note-side regression loss. Keep physical stats as direct computed
conditions, not auxiliary neural targets.

## Final Encoder Split

The final condition pipeline is split into three parts:

```text
1. Direct physical statistics
2. Main neural style encoder
3. Separate note-side neural encoder
```

Direct physical statistics are computed from the chart and are not neural
targets:

```text
avg_density
peak_density
big_note_ratio
balloon_roll_ratio
```

The selected main encoder predicts:

```text
const
complex
hs_change
bpm_rhythm_bin
```

Selected checkpoint:

```text
config:     configs\encoder_final_main.yaml
checkpoint: checkpoints\encoder_final_main\best.pt
eval:       eval\encoder_final_main
```

Test result:

```text
const           MAE 0.525  Spearman 0.897
complex         MAE 7.813  Spearman 0.894
hs_change       MAE 6.989  Spearman 0.827
bpm_rhythm_bin  accuracy 85.9%  Spearman 0.832
```

`bpm_rhythm_bin` confusion:

```text
true none -> 80 none, 11 mid, 0 high
true mid  -> 2 none, 17 mid, 4 high
true high -> 0 none, 1 mid, 13 high
```

The final note-side encoder is kept separate and predicts continuous
`note_type` / `叩き分け`:

```text
selected config:     configs\encoder_v8_note_type_log1p_handtiming_solo.yaml
selected checkpoint: checkpoints\encoder_v8_note_type_log1p_handtiming_solo\best.pt
selected eval:       eval\encoder_v8_note_type_log1p_handtiming_solo
```

This existing checkpoint is selected over the newly trained
`checkpoints\encoder_final_note\best.pt`, because it performs better on the same
test split.

Selected note encoder test result:

```text
note_type MAE 11.157  RMSE 16.673  Pearson 0.658  Spearman 0.669
```

Derived `note_type_high`:

```text
threshold 25:
  accuracy 84.4%
  precision 60.0%
  recall    60.0%
  F1        0.600

threshold 28:
  accuracy 85.9%
  precision 68.4%
  recall    52.0%
  F1        0.591
```

The final default is:

```text
note_type_high = predicted_note_type >= 25
```

Final condition schema:

```text
physical:
  avg_density
  peak_density
  big_note_ratio
  balloon_roll_ratio

main_style:
  const
  complex
  hs_change
  bpm_rhythm_bin

note_style:
  note_type
  note_type_high
```

Machine-readable summary:

```text
eval\final_encoder_summary.json
```

Export final conditions for the test split:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.export_final_conditions
```

Default output:

```text
eval\final_conditions_test.jsonl
```

Each JSONL row combines direct physical statistics, main style predictions, and
the selected separate note-side prediction:

```json
{
  "sample_id": "...",
  "title": "...",
  "physical": {
    "avg_density": 0.0,
    "peak_density": 0.0,
    "big_note_ratio": 0.0,
    "balloon_roll_ratio": 0.0
  },
  "main_style": {
    "const": 0.0,
    "complex": 0.0,
    "hs_change": 0.0,
    "bpm_rhythm_bin": 0,
    "bpm_rhythm_probs": [1.0, 0.0, 0.0]
  },
  "note_style": {
    "note_type": 0.0,
    "note_type_high": false,
    "note_type_high_threshold": 25.0
  }
}
```

## Current Dependency Status

The current Miniforge environment already has:

```text
Python 3.12
numpy
PyYAML
openpyxl
```

PyTorch is not installed in the current environment. It is only required once
we add model training code.

For training on this machine, use:

```text
D:\miniforge3\envs\diffSPHEnv\python.exe
```

That environment has:

```text
torch 2.9.1+cu128
CUDA available: true
```

The base Miniforge Python is still useful for manifest/cache preparation because
it already has `openpyxl` and `PyYAML`.

## Chart-Only Diffusion V0

The first Taiko diffusion experiment uses an 8-channel chart target:

```text
don
ka
roll_start
roll_body
roll_end
balloon_start
balloon_body
balloon_end
```

It is chart-only: no audio condition is used yet, and BPM/HS are not generated
as output tracks. The conditioning vector contains:

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
```

Build the fixed-window cache:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.data.build_diffusion_cache --config configs\diffusion_v0.yaml
```

Current cache size:

```text
train 9862 windows
val   1230 windows
test  1209 windows
```

Train the v0 model:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.train_diffusion --config configs\diffusion_v0.yaml
```

Short 8-epoch result:

```text
best checkpoint: checkpoints\diffusion_v0\best.pt
epoch 8 train_loss 0.0728 val_loss 0.0682
```

Sample one test-window condition:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.sample_diffusion --checkpoint checkpoints\diffusion_v0\best.pt --output eval\diffusion_v0_sample.npz
```

Export a rough inspection TJA. The density-topk mode uses the `avg_density`
condition to choose the expected number of normal notes in the sampled window:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.export_sample_tja --sample eval\diffusion_v0_sample.npz --output eval\diffusion_v0_sample_density_topk.tja --density-topk
```

Current v0 caveat: the pipeline trains and samples successfully, but the raw
0.5 threshold overproduces notes. Density-topk fixes note count for inspection,
while color balance and roll/balloon legality still need stronger decoding or
model changes.

## Notes-Only Diffusion V1

The v1 experiment removes roll and balloon output tracks and predicts only:

```text
note_event
ka_probability
```

This separates normal-note timing from note color. It is still chart-only and
still uses the same 10-dimensional condition vector as v0.

Build cache:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.data.build_diffusion_cache --config configs\diffusion_v1_notes_only.yaml
```

Current cache size:

```text
train 9861 windows
val   1230 windows
test  1209 windows
```

Train:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.train_diffusion --config configs\diffusion_v1_notes_only.yaml
```

Short 8-epoch result:

```text
best checkpoint: checkpoints\diffusion_v1_notes_only\best.pt
best epoch: 5
epoch 5 train_loss 0.1762 val_loss 0.1563
```

Sample:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.sample_diffusion --checkpoint checkpoints\diffusion_v1_notes_only\best.pt --split data\cache\diffusion_v1_notes_only\test.csv --stats data\cache\diffusion_v1_notes_only\stats.json --output eval\diffusion_v1_notes_only_sample.npz
```

Export with density and color-ratio calibration:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.export_sample_tja --sample eval\diffusion_v1_notes_only_sample.npz --output eval\diffusion_v1_notes_only_sample_density_topk_karatio.tja --density-topk --ka-ratio 0.446
```

The `0.446` value is the current train-window global ka ratio.

First checked test window:

```text
target: don 63, ka 48, total 111
v0 density-topk: don 103, ka 10, total 113
v1 density-topk: don 39, ka 74, total 113
v1 density-topk + ka-ratio 0.446: don 63, ka 50, total 113
```

Interpretation: v1 is a better base for normal-note generation than v0 because
note count and color can be decoded separately. The next real model improvement
should make the note-count and color-ratio constraints part of training or
conditioning instead of relying on inspection-time calibration.

## Notes-Constrained Diffusion V2

The v2 experiment keeps the v1 target representation:

```text
note_event
ka_probability
```

Changes from v1:

```text
1. Adds ka_ratio to the condition vector.
2. Adds x0 reconstruction loss during diffusion training.
3. Adds per-channel count loss during diffusion training.
4. Density-topk export automatically uses condition ka_ratio when available.
```

Build cache:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.data.build_diffusion_cache --config configs\diffusion_v2_notes_constrained.yaml
```

Current cache size:

```text
train 9861 windows
val   1230 windows
test  1209 windows
```

Train:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.train_diffusion --config configs\diffusion_v2_notes_constrained.yaml
```

Short 8-epoch result:

```text
best checkpoint: checkpoints\diffusion_v2_notes_constrained\best.pt
best epoch: 8
epoch 8 train_loss 0.1967 val_loss 0.1864
```

The v2 loss includes extra terms, so its numeric loss is not directly
comparable with v1.

Sample:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.sample_diffusion --checkpoint checkpoints\diffusion_v2_notes_constrained\best.pt --split data\cache\diffusion_v2_notes_constrained\test.csv --stats data\cache\diffusion_v2_notes_constrained\stats.json --output eval\diffusion_v2_notes_constrained_sample.npz
```

Export:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.export_sample_tja --sample eval\diffusion_v2_notes_constrained_sample.npz --output eval\diffusion_v2_notes_constrained_sample_density_topk.tja --density-topk
```

First checked test window:

```text
target: don 63, ka 48, total 111
v1 density-topk + manual ka-ratio 0.446: don 63, ka 50, total 113
v2 density-topk + condition ka_ratio 0.438: don 64, ka 49, total 113
```

Interpretation: v2 mainly cleans up the interface. It no longer needs a manual
global `--ka-ratio` for this sample because `ka_ratio` is part of the condition.
Raw thresholding is still too dense, so top-k note decoding remains necessary.

## Audio Feature Cache V0

Audio processing is prepared for the next `v3_audio_notes` model. The cache is
aligned to the existing `diffusion_v2_notes_constrained` chart windows.

Audio source resolution:

```text
1. Read WAVE from the TJA header.
2. Resolve the audio path relative to the TJA file.
3. Fall back to same-folder .ogg/.mp3/.wav if WAVE is missing.
```

Alignment rule:

```text
chart frame i -> audio time i * frame_ms - OFFSET
```

At the current settings:

```text
sample_rate 22050
frame_ms    46.4399
hop_length  1024
```

Each chart frame maps exactly to one audio hop. Each cached audio window has:

```text
shape: [512, 66]
features:
  64 log-mel bins
  onset
  rms
```

Build audio cache:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.data.build_audio_cache --config configs\audio_v0.yaml
```

Smoke-test only:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.data.build_audio_cache --config configs\audio_v0.yaml --limit 4
```

Current full cache:

```text
data\cache\audio_v0
train 9861
val   1230
test  1209
errors 0
size about 1.43 GB
```

The audio cache matches the v2 chart cache one-to-one by `chunk_id`.

For future audio-conditioned training, use:

```python
from taiko_diffusion.data.diffusion_dataset import TaikoAudioDiffusionDataset

dataset = TaikoAudioDiffusionDataset(
    "data/cache/diffusion_v2_notes_constrained/train.csv",
    "data/cache/diffusion_v2_notes_constrained/stats.json",
    "data/cache/audio_v0/train.csv",
    "data/cache/audio_v0/stats.json",
)
```

Returned tensors:

```text
chart      [2, 512]
audio      [66, 512]
condition  [11]
```

## Audio-Conditioned Notes Diffusion V3

The v3 experiment keeps the v2 note representation:

```text
note_event
ka_probability
```

but conditions the denoising U-Net on cached audio features:

```text
chart      [2, 512]
audio      [66, 512]
condition  [11]
```

Model change:

```text
audio [66, 512] -> Conv1d audio stem -> added to the U-Net chart stem
```

Config:

```text
configs\diffusion_v3_audio_notes.yaml
```

Train:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.train_diffusion --config configs\diffusion_v3_audio_notes.yaml
```

Short 6-epoch result:

```text
best checkpoint: checkpoints\diffusion_v3_audio_notes\best.pt
best epoch: 6
epoch 6 train_loss 0.1854 val_loss 0.1781
```

This uses the same constrained loss as v2, so the loss is comparable with v2.
The short v3 run reached lower validation loss than the short v2 run:

```text
v2 8 epochs: val_loss 0.1864
v3 6 epochs: val_loss 0.1781
```

Sample with audio:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.sample_diffusion --checkpoint checkpoints\diffusion_v3_audio_notes\best.pt --split data\cache\diffusion_v2_notes_constrained\test.csv --stats data\cache\diffusion_v2_notes_constrained\stats.json --audio-split data\cache\audio_v0\test.csv --audio-stats data\cache\audio_v0\stats.json --output eval\diffusion_v3_audio_notes_sample.npz
```

Deterministic seed-1 sample:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.sample_diffusion --checkpoint checkpoints\diffusion_v3_audio_notes\best.pt --split data\cache\diffusion_v2_notes_constrained\test.csv --stats data\cache\diffusion_v2_notes_constrained\stats.json --audio-split data\cache\audio_v0\test.csv --audio-stats data\cache\audio_v0\stats.json --output eval\diffusion_v3_audio_notes_sample_seed1.npz --seed 1
```

Export:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.export_sample_tja --sample eval\diffusion_v3_audio_notes_sample_seed1.npz --output eval\diffusion_v3_audio_notes_sample_seed1_density_topk.tja --density-topk
```

First checked test window:

```text
target: don 63, ka 48, total 111
v3 seed1 density-topk + condition ka_ratio: don 64, ka 49, total 113
```

Onset alignment check for the first test window, using 5 sampling seeds:

```text
target onset_mean_at_notes: 0.461
v2 generated mean:         0.079
v3 generated mean:         0.232

target top25 onset hit:    36.0%
v2 generated mean:         24.8%
v3 generated mean:         29.9%
```

Interpretation: v3 is the first version where audio measurably affects note
placement. It is still not close to the real chart alignment, but it is clearly
better than the chart-only v2 baseline on this first-window check.

Useful evaluation command:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.eval_sample_seeds --checkpoint checkpoints\diffusion_v3_audio_notes\best.pt --seeds 0,1,2,3,4
```

## Audio Multiscale Onset Diffusion V4

The v4 experiment keeps the same notes-only output as v3:

```text
note_event
ka_probability
```

Changes from v3:

```text
1. Adds multiscale audio projections into the U-Net down/up paths.
2. Adds onset-guided note_event reconstruction loss.
3. Adds optional onset-guided density top-k decoding.
```

Config:

```text
configs\diffusion_v4_audio_multiscale_onset.yaml
```

Train:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.train_diffusion --config configs\diffusion_v4_audio_multiscale_onset.yaml
```

Short 6-epoch result:

```text
best checkpoint: checkpoints\diffusion_v4_audio_multiscale_onset\best.pt
best epoch: 6
epoch 6 train_loss 0.1996 val_loss 0.1939
```

The v4 loss includes the extra onset-guided term, so its numeric loss is not
directly comparable with v3.

Sample seed 1:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.sample_diffusion --checkpoint checkpoints\diffusion_v4_audio_multiscale_onset\best.pt --split data\cache\diffusion_v2_notes_constrained\test.csv --stats data\cache\diffusion_v2_notes_constrained\stats.json --audio-split data\cache\audio_v0\test.csv --audio-stats data\cache\audio_v0\stats.json --output eval\diffusion_v4_audio_multiscale_onset_sample_seed1.npz --seed 1
```

Export without extra decode onset mixing:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.export_sample_tja --sample eval\diffusion_v4_audio_multiscale_onset_sample_seed1.npz --output eval\diffusion_v4_audio_multiscale_onset_sample_seed1_density_topk.tja --density-topk
```

Export with light onset-guided top-k:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.export_sample_tja --sample eval\diffusion_v4_audio_multiscale_onset_sample_seed1.npz --output eval\diffusion_v4_audio_multiscale_onset_sample_seed1_density_topk_onset001.tja --density-topk --onset-mix 0.01
```

First checked test window:

```text
target: don 63, ka 48, total 111
v4 seed1 density-topk: don 64, ka 49, total 113
v4 seed1 density-topk + onset_mix 0.01: don 64, ka 49, total 113
```

Onset alignment, first test window, 5 seeds:

```text
target onset_mean_at_notes: 0.461
v2 chart-only:              0.079
v3 audio stem:              0.232
v4 multiscale, no mix:      0.341
v4 multiscale, mix 0.01:    0.488

target top25 onset hit:     36.0%
v2 chart-only:              24.8%
v3 audio stem:              29.9%
v4 multiscale, no mix:      34.7%
v4 multiscale, mix 0.01:    40.4%
```

Interpretation: v4 makes a clear improvement over v3. The model itself is
closer to onset-aligned placement, and a very small decode-time onset mix can
match or slightly exceed the real chart's onset alignment on this first-window
check. Larger onset mixes such as 0.1, 0.2, 0.5, or 1.0 are too strong and
over-concentrate notes on onset peaks.

Continued training to 20 total epochs:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.train_diffusion --config configs\diffusion_v4_audio_multiscale_onset.yaml --epochs 20 --resume-checkpoint checkpoints\diffusion_v4_audio_multiscale_onset\best.pt
```

The resume loads model weights from the previous best checkpoint and starts a
fresh optimizer. The best checkpoint after continued training is:

```text
best checkpoint: checkpoints\diffusion_v4_audio_multiscale_onset\best.pt
best epoch: 18
epoch 18 train_loss 0.1619 val_loss 0.1727
epoch 20 train_loss 0.1587 val_loss 0.1819
```

Training speed for the resumed run:

```text
14 epochs: about 1086 seconds
average:   about 77.6 seconds / epoch
throughput: about 127 windows / second
```

Updated first-window onset alignment, 5 seeds, no decode onset mix:

```text
target onset_mean_at_notes: 0.461
v4 epoch 6:                 0.341
v4 epoch 18:                0.449

target top25 onset hit:     36.0%
v4 epoch 6:                 34.7%
v4 epoch 18:                38.1%
```

With the epoch-18 checkpoint, extra decode-time onset mixing is no longer
needed on the first-window check. `--onset-mix 0.01` now over-emphasizes onset
peaks:

```text
v4 epoch 18 no mix:       onset_mean 0.449, top25 hit 38.1%
v4 epoch 18 mix 0.01:     onset_mean 0.659, top25 hit 46.0%
target:                   onset_mean 0.461, top25 hit 36.0%
```

Updated seed-0 inspection sample:

```text
sample: eval\diffusion_v4_audio_multiscale_onset_epoch18_sample_seed0.npz
tja:    eval\diffusion_v4_audio_multiscale_onset_epoch18_sample_seed0_density_topk.tja

target: don 63, ka 48, total 111
sample: don 64, ka 49, total 113

target onset_mean_at_notes: 0.461
sample onset_mean_at_notes: 0.464
```

## Diffusion v5: Hybrid Mug-Like Audio Attention

v5 changes the audio fusion path toward the Mug-Diffusion idea while keeping
the stable v4 path:

```text
v4:
audio features -> per-scale Conv1d projections -> add into U-Net feature maps

v5 hybrid:
audio features -> per-scale Conv1d projections -> add into U-Net feature maps
audio features -> compressed audio context tokens -> cross-attention in U-Net blocks
audio global token -> added to timestep/condition embedding
```

The pure cross-attention variant was also tested, but it was weaker early in
training and did not preserve the v4 onset alignment. The selected v5 hybrid
variant initializes all matching weights from the v4 best checkpoint and trains
only the added attention/audio-token path plus normal fine-tuning.

Config and checkpoint:

```text
config: configs\diffusion_v5_audio_hybrid_attention.yaml
best:   checkpoints\diffusion_v5_audio_hybrid_attention\best.pt
```

Initialization command:

```powershell
D:\miniforge3\envs\diffSPHEnv\python.exe -m taiko_diffusion.train_diffusion --config configs\diffusion_v5_audio_hybrid_attention.yaml --epochs 4 --init-checkpoint checkpoints\diffusion_v4_audio_multiscale_onset\best.pt
```

Training result:

```text
v4 best: epoch 18 val_loss 0.1727, params 6.89M
v5 hybrid best: epoch 4 val_loss 0.1667, params 8.20M

epoch 1 train_loss 0.1679 val_loss 0.1742
epoch 2 train_loss 0.1591 val_loss 0.1723
epoch 3 train_loss 0.1583 val_loss 0.1672
epoch 4 train_loss 0.1563 val_loss 0.1667
epoch 5 train_loss 0.1487 val_loss 0.1749
epoch 6 train_loss 0.1453 val_loss 0.1739
epoch 7 train_loss 0.1449 val_loss 0.1690
epoch 8 train_loss 0.1432 val_loss 0.1697
```

Epoch 4 remains the best checkpoint; later epochs start to overfit.

First-window onset alignment, 5 seeds, no decode onset mix:

```text
target onset_mean_at_notes: 0.461
v4 epoch 18:                0.449
v5 pure cross-attention:    0.105
v5 hybrid epoch 4:          0.364

target top25 onset hit:     36.0%
v4 epoch 18:                38.1%
v5 pure cross-attention:    27.1%
v5 hybrid epoch 4:          35.6%
```

Full test DDPM evaluation:

```text
v4 epoch 18 no mix:
  onset_mean 0.236, top25 hit 0.345, samples/sec 4.62

v4 epoch 18 onset_mix 0.01:
  onset_mean 0.512, top25 hit 0.447, samples/sec 4.32

v5 hybrid epoch 4 no mix:
  onset_mean 0.262, top25 hit 0.355, samples/sec 1.86

v5 hybrid epoch 4 onset_mix 0.01:
  onset_mean 0.517, top25 hit 0.451, samples/sec 1.85

target:
  onset_mean 0.639, top25 hit 0.509
```

Interpretation: the hybrid attention model improves validation loss and gives a
small full-test onset-alignment gain, but it is much slower and does not remove
the need for light decode-time onset guidance. The next useful step is not more
training of this exact v5 shape, but improving the audio conditioning objective
or adding a stronger Mug-style latent/audio-attention design.

## Audio Alignment Audit And Raw-Onset Metrics

The first full-test reports above used the normalized audio onset channel for
alignment metrics. That made the absolute `onset_mean` values misleading. The
evaluation scripts were updated so the model still consumes normalized audio,
but alignment metrics and decode-time `onset_mix` use the raw cached onset
channel.

Audit command:

```powershell
D:\miniforge3\python.exe -m taiko_diffusion.audit_audio_alignment --output eval\audio_alignment_audit.json
```

Result: chart notes and cached raw onset are globally aligned. The best global
shift is exactly zero frames on all splits:

```text
train: best shift 0 frames, onset_mean 0.4576, top25 hit 0.4701
val:   best shift 0 frames, onset_mean 0.4616, top25 hit 0.4682
test:  best shift 0 frames, onset_mean 0.4619, top25 hit 0.4834
```

There are still some per-song suspects with nonzero best shifts, likely from
individual source audio/OFFSET/version noise:

```text
train: 82 strong suspects / 1017 samples
val:   10 strong suspects / 127 samples
test:   7 strong suspects / 128 samples
```

This means the dataset is not globally misaligned; the remaining problem is
model/decoder behavior plus a small amount of noisy source material.

After switching metrics to raw onset, the first test window is much less
problematic:

```text
target:        onset_mean 0.426, top25 hit 36.0%
v4 epoch 18:   onset_mean 0.423, top25 hit 38.1%
v5 hybrid:     onset_mean 0.401, top25 hit 35.6%
v5 hybrid mix: onset_mean 0.455, top25 hit 44.2%
```

The training loss was also changed so the onset-weighted term uses raw onset
from the dataset instead of the normalized onset channel. This produced v6:

```text
config: configs\diffusion_v6_audio_hybrid_raw_onset.yaml
best:   checkpoints\diffusion_v6_audio_hybrid_raw_onset\best.pt
```

v6 was initialized from v5 hybrid and trained for 4 epochs:

```text
epoch 1 train_loss 0.1459 val_loss 0.1722
epoch 2 train_loss 0.1418 val_loss 0.1738
epoch 3 train_loss 0.1405 val_loss 0.1700
epoch 4 train_loss 0.1387 val_loss 0.1703
```

Best checkpoint is epoch 3.

Current raw-onset full-test DDPM comparison:

```text
v5 hybrid epoch 4, onset_mix 0.01:
  onset_mean 0.4408 / target 0.4727
  top25 hit  0.4510 / target 0.5087
  speed      1.98 samples/sec

v6 raw-onset-loss epoch 3, no mix:
  onset_mean 0.3906 / target 0.4727
  top25 hit  0.3791 / target 0.5087
  speed      1.90 samples/sec

v6 raw-onset-loss epoch 3, onset_mix 0.01:
  onset_mean 0.4508 / target 0.4727
  top25 hit  0.4672 / target 0.5087
  speed      1.91 samples/sec
```

Current interpretation: the audio cache is globally aligned. v6 is the best
current full-test setting with light `onset_mix=0.01`, but the model still
under-hits the strongest onset frames. Further progress should focus on making
note placement explicitly learn a ranked/onset-aware objective, not more
training of the same denoising loss.

## Diffusion v7: Mug-Style Latent Diffusion Prototype

v7 implements the Mug-Diffusion-style pipeline:

```text
chart grid [2, 512]
  -> ChartAutoencoder1D
  -> latent [16, 64]
  -> latent diffusion U-Net
  -> DDIM sampling with classifier-free guidance
  -> autoencoder decoder
  -> chart probability [2, 512]
```

Audio conditioning:

```text
audio [66, 512]
  -> AudioContextEncoder
  -> compressed audio context tokens
  -> cross-attention in latent U-Net down/mid/up blocks
```

Added files:

```text
taiko_diffusion\models\latent_diffusion.py
taiko_diffusion\train_autoencoder.py
taiko_diffusion\compute_latent_stats.py
taiko_diffusion\train_latent_diffusion.py
taiko_diffusion\sample_latent_diffusion.py

configs\autoencoder_v7.yaml
configs\latent_diffusion_v7.yaml
configs\latent_diffusion_v7_norm.yaml
configs\latent_diffusion_v7_norm_x0.yaml
```

Autoencoder:

```text
checkpoint: checkpoints\autoencoder_v7\best.pt
latent stats: checkpoints\autoencoder_v7\latent_stats.json

latent shape: [16, 64]
epoch 8 val_loss 0.000074
```

The autoencoder reconstructs the notes-only chart grid well enough for the
latent experiment.

Latent diffusion v7 without latent normalization:

```text
checkpoint: checkpoints\latent_diffusion_v7\best.pt
best epoch: 11
best val_loss: 0.1702
```

This version trains, but sampled note placement is weakly related to audio.

Latent diffusion v7_norm adds per-channel latent normalization:

```text
checkpoint: checkpoints\latent_diffusion_v7_norm\best.pt
best epoch: 12
best val_loss: 0.2143
```

The normalized loss scale is different, so the val loss is not directly
comparable with unnormalized v7. Training was still improving at epoch 12.

First test-window raw-onset check, seed 0:

```text
target:
  onset_mean 0.426, top25 hit 36.0%

v7_norm epoch 12, guidance 1.0:
  onset_mean 0.328, top25 hit 22.1%

v7_norm epoch 12, guidance 2.0:
  onset_mean 0.328, top25 hit 23.0%
```

`guidance_scale` changes the probabilities slightly but does not yet make the
model strongly follow audio.

v7_norm_x0 adds decoded x0 reconstruction and raw-onset-weighted decoded loss:

```text
checkpoint: checkpoints\latent_diffusion_v7_norm_x0\best.pt
initialized from: checkpoints\latent_diffusion_v7_norm\best.pt
best epoch: 4
best val_loss: 0.2386
```

First test-window raw-onset check:

```text
v7_norm_x0 epoch 4, guidance 1.0:
  onset_mean 0.321, top25 hit 21.2%

v7_norm_x0 epoch 4, guidance 2.0:
  onset_mean 0.324, top25 hit 22.1%

v7_norm_x0 epoch 4, guidance 2.0, onset_mix 0.1:
  onset_mean 0.335, top25 hit 22.1%
```

Current interpretation: the Mug-style latent pipeline is implemented and
trainable, but the first version is not yet competitive with the direct-grid
v6 model. The autoencoder is good; the weak point is latent diffusion learning
audio-conditioned note placement. Since latent top-k note probabilities are
still poorly aligned with onset, the next latent-diffusion step should add a
stronger decoded ranking/contrastive objective or train longer with a smaller
latent bottleneck before full-test evaluation.

## Diffusion v8: Closer Mug-Diffusion Match

v8 fixes the main remaining differences from Mug-Diffusion:

```text
1. AutoencoderKL:
   deterministic AE -> KL posterior AE with mean/logvar and KL regularization

2. Mug-style wave encoder:
   single audio token path -> multi-scale audio feature list

3. U-Net audio fusion:
   token-only cross-attention -> token cross-attention + per-scale audio feature injection

4. CFG/DDIM:
   kept DDIM + condition/audio dropout guidance path
```

The implementation is adapted to the Taiko notes-only grid instead of copying
the whole Mug training stack.

New configs:

```text
configs\autoencoder_kl_v8.yaml
configs\latent_diffusion_v8_mug_scale.yaml
```

New/updated model pieces:

```text
ChartAutoencoderKL1D
DiagonalGaussianDistribution
MugStyleAudioScaleEncoder1D
LatentUNet1D(audio_fusion="mug_scale")
```

KL autoencoder:

```text
checkpoint: checkpoints\autoencoder_kl_v8\best.pt
latent stats: checkpoints\autoencoder_kl_v8\latent_stats.json

epoch 6 val_loss 0.000169
epoch 6 val_bce  0.000161
epoch 6 val_kl   8.515
```

The KL latent has much healthier scale than the previous deterministic AE:

```text
previous deterministic latent std: around 0.36, large per-channel mean offsets
v8 KL latent std: roughly 0.77-1.11 per channel before normalization
```

Latent diffusion v8 Mug-scale:

```text
checkpoint: checkpoints\latent_diffusion_v8_mug_scale\best.pt
best epoch: 12
epoch 12 train_loss 0.2342 val_loss 0.2426
```

Training is still descending at epoch 12, but the model is already much more
audio-responsive than v7.

First test-window raw-onset check, seed 0:

```text
target:
  onset_mean 0.426, top25 hit 36.0%

v8 epoch 4, guidance 1.0:
  onset_mean 0.382, top25 hit 33.6%

v8 epoch 4, guidance 2.0:
  onset_mean 0.413, top25 hit 38.1%

v8 epoch 8, guidance 2.0:
  onset_mean 0.437, top25 hit 40.7%

v8 epoch 8, guidance 3.0:
  onset_mean 0.495, top25 hit 46.9%
```

First test-window 5-seed check at epoch 12:

```text
target:
  onset_mean 0.426, top25 hit 36.0%

guidance 1.5:
  onset_mean 0.383, top25 hit 32.9%

guidance 2.0:
  onset_mean 0.407, top25 hit 36.5%

guidance 2.5:
  onset_mean 0.422, top25 hit 39.1%
```

Current interpretation: v8 is now genuinely Mug-like and the CFG scale behaves
as expected. `guidance_scale=2.0` is the most balanced first-window setting;
`2.5` is stronger and may over-align to obvious onset peaks. Unlike v7, v8 no
longer needs decode-time `onset_mix` on the checked window.

Continued v8 training to 20 epochs:

```text
best checkpoint: checkpoints\latent_diffusion_v8_mug_scale\best.pt
best epoch: 18
epoch 18 train_loss 0.2177 val_loss 0.2180
epoch 19 train_loss 0.2147 val_loss 0.2328
epoch 20 train_loss 0.2138 val_loss 0.2243
```

Epoch 18 is the best point; epochs 19-20 regress on validation.

First test-window 5-seed check at epoch 18:

```text
target:
  onset_mean 0.426, top25 hit 36.0%

guidance 1.5:
  onset_mean 0.387, top25 hit 34.3%

guidance 2.0:
  onset_mean 0.400, top25 hit 35.9%

guidance 2.5:
  onset_mean 0.417, top25 hit 38.6%
```

Full test DDIM-50 evaluation, raw onset metrics:

```text
v8 epoch 18, guidance 2.0:
  onset_mean 0.4437 / target 0.4727
  top25 hit  0.4621 / target 0.5087
  speed      6.15 samples/sec

v8 epoch 18, guidance 2.5:
  onset_mean 0.4628 / target 0.4727
  top25 hit  0.4916 / target 0.5087
  speed      5.89 samples/sec
```

Current recommended v8 generation setting:

```text
checkpoint: checkpoints\latent_diffusion_v8_mug_scale\best.pt
sampler: DDIM
sample_steps: 50
guidance_scale: 2.5
onset_mix: 0.0
```

Compared with the previous direct-grid v6 full-test setting:

```text
v6 + onset_mix 0.01:
  onset_mean 0.4508 / target 0.4727
  top25 hit  0.4672 / target 0.5087
  speed      1.91 samples/sec

v8 + guidance 2.5:
  onset_mean 0.4628 / target 0.4727
  top25 hit  0.4916 / target 0.5087
  speed      5.89 samples/sec
```

So v8 is now both more Mug-like and better than the old v6 setting on full-test
audio alignment, while also sampling faster.

## Diffusion v9: Don/Ka Split Latent Grid

v9 changes the chart target back from:

```text
note_event + ka_probability
```

to a direct two-lane note grid:

```text
don, ka
```

Big notes are still collapsed into the normal don/ka lanes for generation. This
matches the arcade-relevant hit lanes and lets the model learn red/blue
placement directly instead of relying on a separate ka-probability channel.

Configs and cache:

```text
configs\diffusion_v9_donka.yaml
configs\autoencoder_kl_v9_donka.yaml
configs\latent_diffusion_v9_mug_scale_donka.yaml
data\cache\diffusion_v9_donka
```

Cache split:

```text
train 9861
val   1230
test  1209
```

KL autoencoder:

```text
checkpoint: checkpoints\autoencoder_kl_v9_donka\best.pt
latent stats: checkpoints\autoencoder_kl_v9_donka\latent_stats.json

epoch 6 val_loss 0.000238
epoch 6 val_bce  0.000231
epoch 6 val_kl   7.129
```

Latent diffusion v9 Mug-scale don/ka:

```text
checkpoint: checkpoints\latent_diffusion_v9_mug_scale_donka\best.pt
best epoch: 22
epoch 22 train_loss 0.1850 val_loss 0.1938
epoch 23 train_loss 0.1821 val_loss 0.2055
epoch 24 train_loss 0.1803 val_loss 0.1971
```

Full test DDIM-50 evaluation, raw onset metrics:

```text
v9 epoch 8, guidance 2.0:
  onset_mean 0.4653 / target 0.4727
  top25 hit  0.4951 / target 0.5087
  ka_ratio_mae 0.0866
  speed      5.96 samples/sec

v9 epoch 16, guidance 2.5:
  onset_mean 0.4732 / target 0.4727
  top25 hit  0.5081 / target 0.5087
  ka_ratio_mae 0.0837
  speed      5.91 samples/sec

v9 epoch 22, guidance 2.5:
  onset_mean 0.4685 / target 0.4727
  top25 hit  0.5016 / target 0.5087
  ka_ratio_mae 0.0846
  speed      5.68 samples/sec

v9 epoch 22, guidance 3.0:
  onset_mean 0.4791 / target 0.4727
  top25 hit  0.5182 / target 0.5087
  ka_ratio_mae 0.0840
  speed      5.64 samples/sec
```

Comparison with v8:

```text
v8 epoch 18, guidance 2.5:
  onset_mean 0.4628 / target 0.4727
  top25 hit  0.4916 / target 0.5087
  ka_ratio_mae 0.1304

v9 epoch 22, guidance 2.5:
  onset_mean 0.4685 / target 0.4727
  top25 hit  0.5016 / target 0.5087
  ka_ratio_mae 0.0846
```

Current interpretation: don/ka split is better than the previous
`note_event + ka_probability` representation. It keeps Mug-like audio alignment
while substantially improving red/blue distribution. For balanced generation,
use `guidance_scale=2.5`; for stronger onset locking, `3.0` is usable but
slightly over-guided on the full-test aggregate.

Current recommended v9 generation setting:

```text
checkpoint: checkpoints\latent_diffusion_v9_mug_scale_donka\best.pt
sampler: DDIM
sample_steps: 50
guidance_scale: 2.5
onset_mix: 0.0
target_channels: don, ka
```
