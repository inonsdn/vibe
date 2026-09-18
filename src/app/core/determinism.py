"""Seed derivation.

A render must be reproducible frame by frame, and resuming a job must produce
exactly the pixels a fresh run would have. Both properties come from *derived*
seeds: a per-frame seed is a pure function of (job seed, garment id, frame
index), never of wall-clock time or iteration order.
"""

from __future__ import annotations

import hashlib

UINT32_MASK = 0xFFFFFFFF
UINT64_MASK = 0xFFFFFFFFFFFFFFFF


def derive_seed(*parts: object, bits: int = 32) -> int:
    """Derive a stable non-negative integer seed from arbitrary parts."""
    if bits not in (32, 64):
        raise ValueError("bits must be 32 or 64")
    digest = hashlib.sha256(b"\x1f".join(str(part).encode("utf-8") for part in parts)).digest()
    width = bits // 8
    value = int.from_bytes(digest[:width], "big")
    return value & (UINT32_MASK if bits == 32 else UINT64_MASK)


def frame_seed(base_seed: int, frame_index: int, *extra: object) -> int:
    """Per-frame seed: stable across resume, unique per frame."""
    return derive_seed("frame", base_seed, frame_index, *extra)


def window_seed(base_seed: int, start_frame: int, end_frame: int, *extra: object) -> int:
    """Per-window seed for backends that render frame windows."""
    return derive_seed("window", base_seed, start_frame, end_frame, *extra)


__all__ = ["UINT32_MASK", "UINT64_MASK", "derive_seed", "frame_seed", "window_seed"]
