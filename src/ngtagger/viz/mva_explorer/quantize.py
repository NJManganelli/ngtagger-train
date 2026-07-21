"""Quantization used by every exporter: strictly-positive quantities (sigmas,
ratios, probabilities in (0, 1]) are stored as int16 round(log10(v) * SCALE)
exactly like eval_spixel's builder (SCALE=3000 => ~0.077% relative step,
representable range 10^-10.9 .. 10^10.9); anything that can be <= 0 falls back
to raw float32.  Which one was used is recorded per block in the meta JSON
("quant": "log10_i16" | "f32")."""

from __future__ import annotations

import numpy as np

LOG10_SCALE = 3000
_CLIP_LO = 1e-9
_CLIP_HI = 1e9


def quantize_log10_int16(values: np.ndarray, scale: int = LOG10_SCALE) -> np.ndarray:
    """v > 0 assumed (caller checks); clips to [1e-9, 1e9] first."""
    v = np.clip(np.asarray(values, dtype=np.float64), _CLIP_LO, _CLIP_HI)
    return np.round(np.log10(v) * scale).astype("<i2")


def dequantize_log10_int16(q: np.ndarray, scale: int = LOG10_SCALE) -> np.ndarray:
    return np.power(10.0, np.asarray(q, dtype=np.float64) / scale)


def choose_block(values: np.ndarray, scale: int = LOG10_SCALE):
    """Return (bytes_le, quant_tag). log10-int16 when every finite value is
    strictly positive (NaNs are not allowed in exported blocks); else float32."""
    v = np.asarray(values)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite values in export block")
    if v.size and np.min(v) > 0:
        return quantize_log10_int16(v, scale).tobytes(), "log10_i16"
    return np.asarray(v, dtype="<f4").tobytes(), "f32"
