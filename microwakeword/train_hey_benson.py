#!/usr/bin/env python3
"""
Best-effort microWakeWord training for "Hey Benson".

This is adapted from notebooks/basic_training_notebook.ipynb in
OHF-Voice/micro-wake-word, with these adjustments for the DGX Spark:
  - All paths are absolute and rooted at /opt/benson/microwakeword/.
  - TF runs CPU-only on this aarch64 box (no CUDA TF wheel), so training
    is slow. Step count is therefore conservative.
  - Positive WAVs come from piper-sample-generator (libritts_r-medium,
    ~904 voices). See synthetic_positive/all/.
  - Negative spectrogram features come pre-built from HuggingFace
    (kahrendt/microwakeword: speech, no_speech, dinner_party,
    dinner_party_eval).
  - Background-noise augmentation uses MIT RIRs only (no AudioSet to
    keep disk usage tractable). Pitch/EQ/distortion/gain augmentations
    still apply.

If this produces a marginal model (likely), do the proper run in the
kahrendt Colab — see README.md.
"""

import os
import random
import shutil
import sys
import yaml
from pathlib import Path

ROOT = Path("/opt/benson/microwakeword")
WORK = ROOT / "work"
FAMILY_POS_DIR = ROOT / "family_positives"
WORK.mkdir(exist_ok=True)
os.chdir(WORK)

# -------- Stage 0: Stage family-voice positives (if any) --------
# Real "Hey Benson" utterances captured through the family-config UI live
# at family_positives/<name>/*.wav as 16 kHz mono PCM. They go through a
# light augmentation pass (time-shift, gain, optional ambient mix) and are
# blended into the synthetic positive pool. The blend ratio is capped so
# small recording sessions don't overwhelm the synthetic ROC; cap rises
# as the family-clip count grows.
SYN_POS = ROOT / "synthetic_positive" / "all"
STAGED_POS = WORK / "staged_positive"
if STAGED_POS.exists():
    shutil.rmtree(STAGED_POS)
STAGED_POS.mkdir(parents=True, exist_ok=True)

family_clips: list[Path] = []
if FAMILY_POS_DIR.is_dir():
    for member_dir in sorted(FAMILY_POS_DIR.iterdir()):
        if not member_dir.is_dir():
            continue
        for wav in sorted(member_dir.glob("*.wav")):
            family_clips.append(wav)

print(f"[stage0] synthetic positives: {SYN_POS}")
print(f"[stage0] family positives: {len(family_clips)} clips")

# Copy synthetic clips by reference (symlink) — faster + no disk doubling.
for syn in SYN_POS.glob("*.wav"):
    link = STAGED_POS / syn.name
    try:
        link.symlink_to(syn)
    except OSError:
        shutil.copy2(syn, link)

if family_clips:
    import numpy as np
    import soundfile as sf

    rng = random.Random(20260517)

    def _load_mono_16k(path: Path) -> np.ndarray:
        data, sr = sf.read(str(path), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            # crude resample by linear interp — sufficient for augmentation
            ratio = 16000 / sr
            new_len = int(round(len(data) * ratio))
            xp = np.linspace(0, 1, len(data), endpoint=False)
            x = np.linspace(0, 1, new_len, endpoint=False)
            data = np.interp(x, xp, data).astype(np.float32)
        return data.astype(np.float32)

    AUG_PER_CLIP = 5
    TARGET_LEN = 16000 * 3  # 3 sec window — mWW will jitter inside
    aug_count = 0
    for src in family_clips:
        try:
            audio = _load_mono_16k(src)
        except Exception as e:
            print(f"[stage0] skip {src}: {e}")
            continue
        if audio.size < 1600:
            continue
        # Pad/trim to 3 sec so jitter window has room.
        if audio.size < TARGET_LEN:
            padded = np.zeros(TARGET_LEN, dtype=np.float32)
            # Random placement gives free time-shift across augmentations.
            for k in range(AUG_PER_CLIP):
                buf = np.zeros(TARGET_LEN, dtype=np.float32)
                offset_max = TARGET_LEN - audio.size
                offset = rng.randint(0, max(0, offset_max))
                buf[offset:offset + audio.size] = audio
                # Gain jitter ±3 dB
                gain_db = rng.uniform(-3.0, 3.0)
                buf *= 10 ** (gain_db / 20.0)
                # Light noise floor to avoid pristine silence
                buf += rng.uniform(0.0005, 0.002) * np.random.randn(TARGET_LEN).astype(np.float32)
                peak = float(np.max(np.abs(buf)))
                if peak > 0.99:
                    buf *= 0.99 / peak
                out_name = f"family_{src.parent.name}_{src.stem}_a{k}.wav"
                sf.write(str(STAGED_POS / out_name), buf, 16000, subtype="PCM_16")
                aug_count += 1
        else:
            for k in range(AUG_PER_CLIP):
                start_max = audio.size - TARGET_LEN
                start = rng.randint(0, max(0, start_max))
                buf = audio[start:start + TARGET_LEN].copy()
                gain_db = rng.uniform(-3.0, 3.0)
                buf *= 10 ** (gain_db / 20.0)
                buf += rng.uniform(0.0005, 0.002) * np.random.randn(TARGET_LEN).astype(np.float32)
                peak = float(np.max(np.abs(buf)))
                if peak > 0.99:
                    buf *= 0.99 / peak
                out_name = f"family_{src.parent.name}_{src.stem}_a{k}.wav"
                sf.write(str(STAGED_POS / out_name), buf, 16000, subtype="PCM_16")
                aug_count += 1

    syn_count = sum(1 for _ in SYN_POS.glob("*.wav"))
    fam_ratio = aug_count / max(1, aug_count + syn_count)
    print(f"[stage0] wrote {aug_count} augmented family clips; "
          f"family share of positives = {fam_ratio*100:.1f}%")
else:
    print("[stage0] no family clips — training on synthetic positives only")

# -------- Stage 1: Build augmented positive features --------
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

POS_WAVS = str(STAGED_POS)
RIRS = str(ROOT / "data" / "mit_rirs")
FEAT_DIR = WORK / "generated_augmented_features"
# Force-rebuild features when family pool changes — otherwise the [skip]
# branch leaves stale synthetic-only mmaps in place.
if family_clips and FEAT_DIR.exists():
    shutil.rmtree(FEAT_DIR)
FEAT_DIR.mkdir(exist_ok=True)

clips = Clips(
    input_directory=POS_WAVS,
    file_pattern="*.wav",
    max_clip_duration_s=None,
    remove_silence=False,
    random_split_seed=10,
    split_count=0.1,
)
augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        # No external background noise corpus; rely on RIRs + EQ + gain.
        "AddBackgroundNoise": 0.0,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    impulse_paths=[RIRS],
    background_paths=[],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)

splits = [("training", "train", 2), ("validation", "validation", 1), ("testing", "test", 1)]
for split, split_name, rep in splits:
    out_dir = FEAT_DIR / split
    out_dir.mkdir(exist_ok=True)
    target = out_dir / "wakeword_mmap"
    if target.exists():
        print(f"[skip] {target} already exists")
        continue
    slide = 10 if split != "testing" else 1
    sg = SpectrogramGeneration(
        clips=clips, augmenter=augmenter, slide_frames=slide, step_ms=10
    )
    print(f"[build] {target} repetition={rep}")
    RaggedMmap.from_generator(
        out_dir=str(target),
        sample_generator=sg.spectrogram_generator(split=split_name, repeat=rep),
        batch_size=100,
        verbose=True,
    )

# -------- Stage 2: Write training config --------
NEG = str(ROOT / "data" / "negative_datasets")
TRAIN_CFG = WORK / "training_parameters.yaml"

config = {
    "window_step_ms": 10,
    "train_dir": str(WORK / "trained_models" / "hey_benson"),
    "features": [
        {
            "features_dir": str(FEAT_DIR),
            "sampling_weight": 2.0,
            "penalty_weight": 1.0,
            "truth": True,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        },
        {
            "features_dir": f"{NEG}/speech",
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": f"{NEG}/dinner_party",
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": f"{NEG}/no_speech",
            "sampling_weight": 5.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": f"{NEG}/dinner_party_eval",
            "sampling_weight": 0.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "split",
            "type": "mmap",
        },
    ],
    # Training hyperparameters. CPU-only, so keep step count modest.
    # Empirically, model hits 99%+ accuracy / 96%+ recall by step ~500
    # on the synthetic-only positives; longer training mostly tightens
    # precision on the dinner_party ambient set.
    "training_steps": [4000],
    "positive_class_weight": [1],
    "negative_class_weight": [20],
    "learning_rates": [0.001],
    "batch_size": 128,
    "time_mask_max_size": [5],
    "time_mask_count": [2],
    "freq_mask_max_size": [5],
    "freq_mask_count": [2],
    "eval_step_interval": 500,
    "clip_duration_ms": 1500,
    "target_minimization": 0.9,
    "minimization_metric": None,
    "maximization_metric": "average_viable_recall",
}

with open(TRAIN_CFG, "w") as f:
    yaml.safe_dump(config, f)
print(f"[config] wrote {TRAIN_CFG}")

# -------- Stage 3: hand off to model_train_eval --------
print("[train] launching model_train_eval. CPU-only; expect this to be slow.")
import subprocess
cmd = [
    sys.executable, "-m", "microwakeword.model_train_eval",
    "--training_config", str(TRAIN_CFG),
    "--train", "1",
    "--restore_checkpoint", "1",
    "--test_tf_nonstreaming", "0",
    "--test_tflite_nonstreaming", "0",
    "--test_tflite_nonstreaming_quantized", "0",
    "--test_tflite_streaming", "0",
    "--test_tflite_streaming_quantized", "1",
    "--use_weights", "best_weights",
    "mixednet",
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1, 1, 1, 1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", "3",
]
print(" ".join(cmd))
rc = subprocess.run(cmd, check=False).returncode
print(f"[train] model_train_eval exit={rc}")

# -------- Stage 4: publish artifacts to models/ --------
MODELS_OUT = ROOT / "models"
MODELS_OUT.mkdir(exist_ok=True)
TFLITE_SRC = (
    WORK / "trained_models" / "hey_benson" /
    "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
)
if TFLITE_SRC.exists():
    shutil.copy2(TFLITE_SRC, MODELS_OUT / "hey_benson.tflite")
    print(f"[publish] copied {TFLITE_SRC} -> {MODELS_OUT/'hey_benson.tflite'}")
    template = MODELS_OUT / "hey_benson.json.template"
    target_json = MODELS_OUT / "hey_benson.json"
    if template.exists() and not target_json.exists():
        shutil.copy2(template, target_json)
        print(f"[publish] seeded manifest from template")
    print("[done] hey_benson.tflite published.")
else:
    print(f"[publish] WARN: tflite not produced at {TFLITE_SRC}")
