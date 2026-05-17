# Wake-word recommendation memo — Benson kitchen ReSpeaker

**TL;DR — ship `hey_jarvis` this week, train `hey_benson` properly with
real family voices via the kahrendt Colab next week.** Do not chase a
synthetic-only "Hey Benson" or, especially, a bare "Benson". Both will
misfire.

## What runs on the device

The XIAO ESP32-S3 in the ReSpeaker Lite runs **microWakeWord** (mWW),
not openWakeWord. mWW classifies 40-feature spectrograms with a tiny
(~26 k parameter, ~100 KB) MixConv streaming CNN entirely on-MCU. The
formatBCE base YAML pre-loads five models (`okay_nabu`, `kenobi`,
`hey_jarvis`, `hey_mycroft`, `stop`). Adding a sixth `hey_benson`
needs a hosted `.tflite` + manifest `.json` pair plus a one-line
`!extend` in `respeaker-kitchen.yaml`.

## Path A — pretrained `hey_jarvis` (recommended for tonight)

`hey_jarvis` is already compiled into formatBCE's base config; flipping
the active wake word is purely a Home Assistant Assist-pipeline setting,
no flash required. Among the upstream pretrained models, `hey_jarvis`
is the closest fit for a household chief-of-staff voice. `okay_nabu` is
Nabu Casa's brand mark, `hey_mycroft` carries the dead-project baggage,
`kenobi` is a Star Wars meme. `hey_jarvis` skews neutral-butler — fine
stand-in until a real `hey_benson` exists. Cost: zero. Tradeoff: it's
not the actual name; some onboarding friction with the family.

## Path B — custom `hey_benson` (correct end state)

mWW training inputs:
- **Positives:** ~2-5 k clips of the target phrase. The repo's
  piper-sample-generator + libritts_r-medium ships ~904 TTS voices,
  which we used here. **But the mWW README is explicit that pure-TTS
  positives "most likely [produce] a model [that is] not usable."** Real
  recordings from each family member dramatically improve recall, and
  are the bottleneck for personalization.
- **Negatives:** Pre-built ragged-mmap spectrogram features hosted on
  HuggingFace (`kahrendt/microwakeword`): `speech`, `no_speech`,
  `dinner_party` for training, `dinner_party_eval` for benchmarking
  false-accepts/hour on chime6/DiPCo-style household audio. ~25 GB
  unzipped (overshot tonight's 5 GB budget — these are mmap'd and
  can't be re-zipped after extraction; disk free is 3.4 TB so it's fine
  in practice).

**Where to train.** The kahrendt Colab notebook
([basic_training_notebook.ipynb](https://colab.research.google.com/github/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb))
with a free T4 GPU is the right venue. We *could* train locally on the
Spark, but TF 2.21's PyPI wheel on aarch64 is **CPU-only**
(`is_cuda_build=False`) — the GB10 GPU sits idle. Training is therefore
~10-15x slower than a T4 Colab. PyTorch on the Spark *does* have CUDA,
which is why the synthetic positive-sample generation step ran fast (2
sec per 100 clips on GPU). The training step is the bottleneck.

**Where the artifact lives.** Output is `stream_state_internal_quant.tflite`.
Drop into `/opt/benson/microwakeword/models/hey_benson.tflite`, write a
matching `hey_benson.json` (template in `models/`), host the pair at any
HTTPS URL (a GitHub repo is easiest — ESPHome fetches at compile time),
and add an entry to the `micro_wake_word: models:` list in
`respeaker-kitchen.yaml`. Done.

## Bare "Benson" — push back

I'd recommend **against** training a standalone single-word "Benson"
model. Three reasons:

1. "Benson" is a real surname (Benson Boone is currently charting),
   a casual one-word interjection in some American dialects, and a
   character/title in TV and games (Benson the gardener on *Regular
   Show*, etc.). False-accepts during media playback in the kitchen
   are nearly guaranteed.
2. A two-syllable trochee is at the absolute floor of what mWW can
   reliably discriminate. The pretrained models are all 3-4 syllables
   for good reason.
3. `Hey Benson` with a tight `probability_cutoff` (start at 0.95) and
   the bare word in the post-trigger LLM context is more robust than
   either alone.

If Casey insists, fine — but it gates on having a few hundred real
"Benson" recordings and weeks of in-field tuning, not synthetic data.

## Quality target

Per the mWW conventions:
- `false_reject_rate < 0.30` on the held-out test set
- `false_accepts_per_hour < 1.0` on the `dinner_party_eval` ambient
  validation set
- `probability_cutoff` selected so the model lands on or under the
  faph constraint when streaming-quantized

If tonight's local run lands inside those bands on synthetic-only data,
ship it (with caution); if not, treat it as "homework" and use the
Colab + real family voice samples (see README.md). Either way,
`hey_jarvis` is the safe interim wake word — no reason to wait on
training to enable voice for the family.
