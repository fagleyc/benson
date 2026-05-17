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
import sys
import yaml
from pathlib import Path

ROOT = Path("/opt/benson/microwakeword")
WORK = ROOT / "work"
os.chdir(WORK)

# -------- Stage 1: Build augmented positive features --------
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

POS_WAVS = str(ROOT / "synthetic_positive" / "all")
RIRS = str(ROOT / "data" / "mit_rirs")
FEAT_DIR = WORK / "generated_augmented_features"
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
subprocess.run(cmd, check=False)
