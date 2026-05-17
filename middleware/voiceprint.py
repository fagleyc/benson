"""ECAPA-TDNN voiceprint storage + matching.

Lazy-loads SpeechBrain's `speechbrain/spkrec-ecapa-voxceleb` (Apache-2.0,
192-dim, ~25 MB) on first use. Pins to CUDA when available. Mirrors the
lazy-load pattern in kokoro_tts.py so the model isn't held until the first
enrollment runs.

Storage layout under /opt/benson/memory/voiceprints/:
  - <name>.npy   averaged 192-dim float32 embedding
  - <name>.json  metadata + sample list + interview map
  - raw/<name>/<uuid>.wav  16 kHz mono PCM source samples

Re-enrollment APPENDS samples and updates the running mean weighted by
sample_count, so the voiceprint converges as more samples arrive.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("benson.voiceprint")

VOICEPRINT_DIR = Path("/opt/benson/memory/voiceprints")
RAW_DIR = VOICEPRINT_DIR / "raw"
VOICEPRINT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 192

_model = None
_model_lock = threading.Lock()


def _device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from speechbrain.inference.speaker import EncoderClassifier
        import time
        t0 = time.time()
        dev = _device()
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/opt/benson/middleware/venv/share/speechbrain-ecapa",
            run_opts={"device": dev},
        )
        logger.info(f"ECAPA loaded on {dev} in {time.time() - t0:.2f}s")
        return _model


def extract_embedding(wav_path: Path) -> np.ndarray:
    """Run ECAPA over a WAV and return a 192-dim float32 embedding."""
    import torch
    import soundfile as sf
    model = _load_model()
    # torchaudio 2.11 dropped the soundfile/sox backends in favor of
    # torchcodec, which doesn't ship an aarch64 wheel — bypass it by
    # loading via soundfile directly. The upload pipeline already
    # normalizes everything to 16 kHz mono PCM with ffmpeg, so resampling
    # is rarely needed; do it via torchaudio.functional only if the WAV
    # came in at a different rate.
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    signal = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
    if sr != 16000:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, 16000)
        signal = resampler(signal)
    signal = signal.to(_device())
    with torch.no_grad():
        emb = model.encode_batch(signal)
    emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)
    if emb.ndim > 1:
        emb = emb.reshape(-1)
    if emb.shape[0] != EMBEDDING_DIM:
        raise RuntimeError(
            f"ECAPA returned {emb.shape[0]}-dim, expected {EMBEDDING_DIM}"
        )
    return _l2_normalize(emb)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return v
    return (v / n).astype(np.float32)


def _meta_path(name: str) -> Path:
    return VOICEPRINT_DIR / f"{name}.json"


def _emb_path(name: str) -> Path:
    return VOICEPRINT_DIR / f"{name}.npy"


def load_meta(name: str) -> dict:
    p = _meta_path(name)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    import io
    buf = io.BytesIO()
    np.save(buf, arr)
    _atomic_write_bytes(path, buf.getvalue())


def write_meta(name: str, meta: dict) -> None:
    _atomic_write_bytes(_meta_path(name), json.dumps(meta, indent=2).encode("utf-8"))


def merge_voiceprint(
    name: str, new_embs: list[np.ndarray]
) -> tuple[np.ndarray, int]:
    """Fold a batch of new (L2-normalized) embeddings into the stored mean.

    Mean over normalized vectors, then re-normalize so cosine math stays
    well-conditioned. Returns (averaged_embedding, new_sample_count).
    """
    if not new_embs:
        raise ValueError("merge_voiceprint called with empty list")
    emb_path = _emb_path(name)
    meta = load_meta(name)
    prev_count = int(meta.get("sample_count", 0))
    new_count = prev_count + len(new_embs)
    summed = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    if prev_count > 0 and emb_path.exists():
        prev = np.load(emb_path).astype(np.float64)
        summed += prev * prev_count
    for e in new_embs:
        summed += e.astype(np.float64)
    avg = (summed / float(new_count)).astype(np.float32)
    avg = _l2_normalize(avg)
    _atomic_save_npy(emb_path, avg)
    return avg, new_count


def update_voiceprint(name: str, new_emb: np.ndarray) -> tuple[np.ndarray, int]:
    """Single-sample variant of merge_voiceprint (kept for callers)."""
    return merge_voiceprint(name, [new_emb])


def load_all() -> dict[str, np.ndarray]:
    """All persisted voiceprints, keyed by lowercase name. Slice 2 calls
    this once at startup and keeps the dict in memory for matching."""
    out: dict[str, np.ndarray] = {}
    for p in VOICEPRINT_DIR.glob("*.npy"):
        try:
            out[p.stem.lower()] = np.load(p).astype(np.float32)
        except Exception as e:
            logger.warning(f"failed to load {p}: {e}")
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def identify_with_confidence(
    query_emb: np.ndarray,
    threshold: float = 0.45,
    gap: float = 0.08,
) -> tuple[Optional[str], float, float]:
    """Cosine-match against every stored voiceprint.

    Returns (name|None, best_similarity, margin) where margin is
    best-second (0.0 when only one enrolled). `name` is None when the
    best similarity is below threshold OR (with >=2 enrolled) the margin
    is below gap.
    """
    all_emb = load_all()
    if not all_emb:
        return None, 0.0, 0.0
    scores = sorted(
        ((name, _cosine(query_emb, emb)) for name, emb in all_emb.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best = scores[0]
    margin = (best - scores[1][1]) if len(scores) > 1 else 0.0
    if best < threshold:
        return None, best, margin
    if len(scores) > 1 and margin < gap:
        return None, best, margin
    return best_name, best, margin


def identify(
    query_emb: np.ndarray,
    threshold: float = 0.45,
    gap: float = 0.08,
) -> Optional[str]:
    """Convenience wrapper returning just the matched name (or None)."""
    name, _best, _margin = identify_with_confidence(query_emb, threshold, gap)
    return name


def list_enrolled() -> list[dict]:
    """Summary list for the dashboard card grid."""
    out: list[dict] = []
    for meta_file in VOICEPRINT_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue
        name = meta_file.stem
        out.append({
            "name": meta.get("name") or name,
            "role": meta.get("role"),
            "photo": meta.get("photo"),
            "sample_count": int(meta.get("sample_count", 0)),
            "enrolled_at": meta.get("enrolled_at"),
            "last_updated_at": meta.get("last_updated_at"),
        })
    out.sort(key=lambda r: (r["name"] or "").lower())
    return out


def delete(name: str) -> dict:
    """Remove .npy + .json + raw samples. Leaves the memory .md alone."""
    removed: list[str] = []
    for p in (_emb_path(name), _meta_path(name)):
        if p.exists():
            p.unlink()
            removed.append(p.name)
    raw_subdir = RAW_DIR / name
    if raw_subdir.exists():
        for f in raw_subdir.glob("*"):
            try:
                f.unlink()
                removed.append(f"raw/{name}/{f.name}")
            except OSError:
                pass
        try:
            raw_subdir.rmdir()
        except OSError:
            pass
    return {"name": name, "removed": removed}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
