"""
Chunk N — Beat Engine

A tiny sample-based step sequencer for the "cool beats" menu.  Loads the drum
one-shots from samples/ and loops a genre pattern through its own sounddevice
OutputStream — fully independent of ChordEngine / FluidSynth, so a beat can run
under any chord backend.

Patterns are 16-step (one bar) grids, one row per sample:
    'x'      → hit at full velocity
    '1'-'9'  → hit at velocity n/9
    '.'      → rest
Odd 16th steps are pushed late by `swing` (fraction of a step) for a shuffle.

Public API:
    BeatEngine()                  — loads samples on construction
    .play(genre: str)             — start/replace the looping beat
    .stop()                       — silence
    .is_playing  / .current       — state
    .set_gain(v)                  — master gain 0..1
    GENRES                        — list[str] of pattern names (menu order)
"""

from __future__ import annotations

import glob
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

try:
    import soundfile as sf
except Exception:                       # pragma: no cover
    sf = None

# ── Audio config ────────────────────────────────────────────────────────────

SR_BEAT   = 44100        # output rate; samples are resampled to this on load
BLOCKSIZE = 256
_MAX_VOICES = 64

_SAMPLES_DIR = Path(__file__).parent / "samples"

# Per-instrument mix gains so hats don't bury the kick/snare.
def _mix_gain(name: str) -> float:
    if name.startswith("hh"):    return 0.50
    if name.startswith("kick"):  return 1.00
    if name.startswith("snare"): return 0.85
    if name.startswith("clap"):  return 0.80
    return 0.80


# ── Genre patterns (my interpretation of each kit) ───────────────────────────
# bpm + 16-step grids.  Kit choice per genre is deliberate:
#   Trap kit  → trap / drill / phonk / jersey (punchy, modern)
#   R&B kit   → boom bap / lo-fi          (warm, dusty)
#   Pop kit   → house / pop / amapiano / afrobeat / reggaeton (clean, tight)
# amapiano/afrobeat/phonk/reggaeton are approximations — we lack log-drum/808/
# cowbell/shaker, so they're built from the closest available drums.

PATTERNS: dict[str, dict] = {
    "jersey club": {  # bouncy 3-kick "bed-squeak" bounce
        "bpm": 140, "swing": 0.0,
        "tracks": {
            "kickTrap": "x...x...x..x..x.",
            "clapTrap": "............x...",
            "hhPop":    "x.x.x.x.x.x.x.xx",
        },
    },
    "boom bap": {  # swung, kick on 1 + and-of-3, snare on 2 & 4
        "bpm": 90, "swing": 0.18,
        "tracks": { 
            "kickR&B":  "x.......x.x.....",
            "snareR&B": "....x.......x...",
            "hhR&B":    "x.x.x.x.x.x.x.x.",
        },
    },
    "trap": {  # half-time backbeat, hi-hat roll into the bar end
        "bpm": 140, "swing": 0.0,
        "tracks": {
            "kickTrap":  "x.....x...x.....",
            "snareTrap": "........x.......",
            "clapTrap":  "........x.......",
            "hhTrap":    "x.x.x.x.x.xxx.x.",
        },
    },
    "drill": {  # sliding syncopated kicks, triplet-feel hats
        "bpm": 140, "swing": 0.0,
        "tracks": {
            "kickTrap":  "x.....x.....x..x",
            "snareTrap": "........x.......",
            "clapTrap":  "........x.......",
            "hhTrap":    "x..x..x.x..x..x.", # tresillo drum pattern 
        },
    },
    "house": {  # four-on-the-floor, clap on 2 & 4, offbeat hats
        "bpm": 124, "swing": 0.0,
        "tracks": {
            "kickPop":  "x...x...x...x...",
            "clapTrap": "....x.......x...",
            "hhPop":    "..x...x...x...x.",
        },
    },
    "amapiano": {  # 4-on-floor + offbeat clap, busy shaker-ish hats (approx)
        "bpm": 112, "swing": 0.08,
        "tracks": {
            "kickPop":  "x...x...x...x...",
            "clapTrap": "......x.......x.",
            "hhPop":    "x.xxx.x.x.xxx.x.",
        },
    },
    "afrobeat": {  # syncopated kick, off-beat hats (approx)
        "bpm": 105, "swing": 0.0,
        "tracks": {
            "kickPop":  "x..x...x..x.....",
            "clapTrap": "....x.......x...",
            "hhPop":    "..x.x..x..x.x..x",
        },
    },
    "lo-fi": {  # sparse, swung, soft
        "bpm": 80, "swing": 0.20,
        "tracks": {
            "kickR&B":  "x.......x.......",
            "snareR&B": "....x.......x...",
            "hhR&B":    "5.5.5.5.5.5.5.5.",
        },
    },
    "phonk": {  # driving half-time, relentless hats (approx — no cowbell/808)
        "bpm": 135, "swing": 0.0,
        "tracks": {
            "kickTrap":  "x.....x...x....x",
            "snareTrap": "........x.......",
            "hhTrap":    "x.x.x.x.x.x.xxx.",
        },
    },
    "reggaeton": {  # dembow — kick on 1 & 3, 3-3-2 snare/clap
        "bpm": 95, "swing": 0.0,
        "tracks": {
            "kickPop":   "x.......x.......",
            "clapTrap":  "...x..x.x..x..x.",
            "hhPop":     "x.x.x.x.x.x.x.x.",
        },
    },
}

# Menu order (matches the genres the "cool beats" list shows).
GENRES = [
    "jersey club", "boom bap", "trap", "drill", "house",
    "amapiano", "afrobeat", "lo-fi", "phonk", "reggaeton",
]


# ── Sample loading ──────────────────────────────────────────────────────────

def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data
    n_out = max(1, int(round(data.shape[0] * sr_out / sr_in)))
    x_old = np.linspace(0.0, 1.0, data.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.stack(
        [np.interp(x_new, x_old, data[:, c]) for c in range(data.shape[1])],
        axis=1,
    ).astype(np.float32)


def _load_samples() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if sf is None or not _SAMPLES_DIR.exists():
        return out
    for path in sorted(glob.glob(str(_SAMPLES_DIR / "*.wav"))):
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            print(f"  [beat] skip {os.path.basename(path)}: {e}")
            continue
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        data = _resample(data, sr, SR_BEAT)
        name = os.path.splitext(os.path.basename(path))[0]
        out[name] = np.ascontiguousarray(data, dtype=np.float32)
    return out


# ── Engine ──────────────────────────────────────────────────────────────────

class BeatEngine:
    def __init__(self) -> None:
        self._samples = _load_samples()
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None
        self._events: Optional[list[tuple[int, np.ndarray, float]]] = None
        self._loop_len = 0
        self._pos = 0
        self._voices: list[list] = []      # [array, pos, gain, start_offset]
        self._master_gain = 0.9
        self._current: Optional[str] = None
        print(f"  BeatEngine: loaded {len(self._samples)} drum samples "
              f"({', '.join(sorted(self._samples)) or 'none'})")

    # ── Pattern → scheduled sample events ────────────────────────────────────
    def _build_events(self, genre: str):
        pat = PATTERNS[genre]
        bpm   = pat["bpm"]
        swing = pat.get("swing", 0.0)
        sps   = SR_BEAT * 60.0 / (bpm * 4.0)        # samples per 16th
        steps = max(len(g) for g in pat["tracks"].values())
        loop_len = int(round(steps * sps))
        events: list[tuple[int, np.ndarray, float]] = []
        for name, grid in pat["tracks"].items():
            arr = self._samples.get(name)
            if arr is None:
                continue
            base_gain = _mix_gain(name)
            for step, ch in enumerate(grid):
                if ch == "." or ch == " ":
                    continue
                vel = 1.0 if ch == "x" else (int(ch) / 9.0 if ch.isdigit() else 1.0)
                off = step * sps + (swing * sps if step % 2 == 1 else 0.0)
                events.append((int(round(off)) % loop_len, arr, base_gain * vel))
        return events, loop_len

    # ── Public control ───────────────────────────────────────────────────────
    def play(self, genre: str) -> None:
        if genre not in PATTERNS:
            return
        events, loop_len = self._build_events(genre)
        with self._lock:
            self._events = events
            self._loop_len = loop_len
            self._pos = 0
            self._voices = []
            self._current = genre
        self._ensure_stream()

    def stop(self) -> None:
        with self._lock:
            self._events = None
            self._voices = []
            self._current = None

    @property
    def is_playing(self) -> bool:
        return self._current is not None

    @property
    def current(self) -> Optional[str]:
        return self._current

    def set_gain(self, v: float) -> None:
        self._master_gain = max(0.0, min(1.0, float(v)))

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ── Audio callback ───────────────────────────────────────────────────────
    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=SR_BEAT, blocksize=BLOCKSIZE, channels=2,
                dtype="float32", callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"  [beat] audio stream unavailable: {e}")
            self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        out = np.zeros((frames, 2), dtype=np.float32)
        with self._lock:
            events, loop_len, pos = self._events, self._loop_len, self._pos
            # Schedule any hits that fall inside this block.
            if events and loop_len > 0:
                for off, arr, g in events:
                    rel = (off - pos) % loop_len
                    if rel < frames and len(self._voices) < _MAX_VOICES:
                        self._voices.append([arr, 0, g, rel])
                self._pos = (pos + frames) % loop_len
            # Mix active one-shots.
            alive = []
            for v in self._voices:
                arr, vpos, g, start = v
                avail = arr.shape[0] - vpos
                n = min(frames - start, avail)
                if n > 0:
                    out[start:start + n] += arr[vpos:vpos + n] * g
                    v[1] = vpos + n
                v[3] = 0                      # start offset only applies first block
                if v[1] < arr.shape[0]:
                    alive.append(v)
            self._voices = alive
            gain = self._master_gain
        out *= gain
        np.clip(out, -1.0, 1.0, out)
        outdata[:] = out


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, time
    be = BeatEngine()
    genre = sys.argv[1] if len(sys.argv) > 1 else "trap"
    print(f"playing '{genre}' for 6s …")
    be.play(genre)
    time.sleep(6)
    be.close()
