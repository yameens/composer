"""
Custom oscillator synthesizer — pure numpy + sounddevice, no external audio files.

Public API (used by chord_engine.py and the future front-end):

    SynthPatch / OscSpec / FilterSpec / EnvSpec
    default_patch()                    -- factory: a usable pad preset
    SynthEngine.start() / stop()
    SynthEngine.note_on(note, vel)
    SynthEngine.note_off(note)
    SynthEngine.all_notes_off()
    SynthEngine.set_patch(patch)       -- change sound for future notes
    SynthEngine.audition(patch, notes) -- play a test chord with any patch
    PatchStore.save / load / list / all / delete   -- JSON at patches.json
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

# ── Constants ─────────────────────────────────────────────────────────────────

SR         = 44100          # sample rate
BLOCKSIZE  = 256            # audio callback block size (~5.8 ms per block)
MAX_VOICES = 24             # hard polyphony cap
MIDI_A4    = 69             # MIDI note 69 = 440 Hz

# All waveform shape names the front end may use
WAVE_NAMES = ("sine", "saw", "square", "triangle", "wavetable")

# ── MIDI helpers ──────────────────────────────────────────────────────────────

def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - MIDI_A4) / 12.0))


# ── Oscillator waveform generators ────────────────────────────────────────────
# All functions accept a *phase* array (float, 0..1, already incremented) and
# return a sample block of the same length, values in [-1, 1].

def _osc_sine(phase: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * phase)


def _osc_saw(phase: np.ndarray) -> np.ndarray:
    # Band-limited via PolyBLEP correction would be ideal; this sounds good
    # enough for pads and can be upgraded later without changing the API.
    return 2.0 * (phase - np.floor(phase + 0.5))


def _osc_square(phase: np.ndarray, pw: float = 0.5) -> np.ndarray:
    return np.where((phase % 1.0) < pw, 1.0, -1.0).astype(np.float32)


def _osc_triangle(phase: np.ndarray) -> np.ndarray:
    p = phase % 1.0
    return np.where(p < 0.5, 4.0 * p - 1.0, 3.0 - 4.0 * p).astype(np.float32)


def _osc_wavetable(phase: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Linear-interpolated wavetable lookup."""
    n = len(table)
    idx = (phase % 1.0) * n
    i0 = idx.astype(np.int32) % n
    i1 = (i0 + 1) % n
    frac = idx - np.floor(idx)
    return table[i0] * (1.0 - frac) + table[i1] * frac


def _render_wave(
    wave: str,
    phase: np.ndarray,
    morph: float = 0.0,
    morph_target: str = "sine",
    wavetable: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render one oscillator block.

    `morph` (0..1) crossfades from `wave` towards `morph_target`.
    """
    _dispatch = {
        "sine":     _osc_sine,
        "saw":      _osc_saw,
        "square":   _osc_square,
        "triangle": _osc_triangle,
    }
    if wave == "wavetable":
        tbl = wavetable if wavetable is not None else np.sin(
            2.0 * np.pi * np.linspace(0, 1, 2048, endpoint=False)
        ).astype(np.float32)
        base = _osc_wavetable(phase, tbl)
    else:
        base = _dispatch.get(wave, _osc_sine)(phase)

    if morph > 0.0:
        if morph_target == "wavetable":
            tbl2 = wavetable if wavetable is not None else np.sin(
                2.0 * np.pi * np.linspace(0, 1, 2048, endpoint=False)
            ).astype(np.float32)
            target = _osc_wavetable(phase, tbl2)
        else:
            target = _dispatch.get(morph_target, _osc_sine)(phase)
        return (1.0 - morph) * base + morph * target

    return base


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class OscSpec:
    """One oscillator layer within a patch."""
    wave:         str   = "saw"       # sine | saw | square | triangle | wavetable
    morph:        float = 0.0         # 0 = pure wave, 1 = pure morph_target
    morph_target: str   = "sine"      # wave to morph towards
    detune_cents: float = 0.0         # ± cents from root pitch
    octave:       int   = 0           # octave shift from root (−2..+2)
    level:        float = 0.7         # mix level for this layer (0..1)
    # Optional user-defined wavetable (list of floats, length power-of-2).
    # None = ignored (falls through to morph_target or base wave).
    wavetable:    Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "wave": self.wave,
            "morph": self.morph,
            "morph_target": self.morph_target,
            "detune_cents": self.detune_cents,
            "octave": self.octave,
            "level": self.level,
            "wavetable": self.wavetable,
        }

    @staticmethod
    def from_dict(d: dict) -> "OscSpec":
        return OscSpec(
            wave         = d.get("wave", "saw"),
            morph        = float(d.get("morph", 0.0)),
            morph_target = d.get("morph_target", "sine"),
            detune_cents = float(d.get("detune_cents", 0.0)),
            octave       = int(d.get("octave", 0)),
            level        = float(d.get("level", 0.7)),
            wavetable    = d.get("wavetable"),
        )


@dataclass
class FilterSpec:
    """Simple one-pole lowpass per voice."""
    cutoff_hz:  float = 1800.0   # Hz
    resonance:  float = 0.0      # 0..1 (mild Q lift; full resonance not implemented)

    def to_dict(self) -> dict:
        return {"cutoff_hz": self.cutoff_hz, "resonance": self.resonance}

    @staticmethod
    def from_dict(d: dict) -> "FilterSpec":
        return FilterSpec(
            cutoff_hz = float(d.get("cutoff_hz", 1800.0)),
            resonance = float(d.get("resonance", 0.0)),
        )


@dataclass
class EnvSpec:
    """ADSR in seconds / level."""
    attack:  float = 0.06    # s
    decay:   float = 0.10    # s
    sustain: float = 0.75    # level 0..1
    release: float = 0.18    # s — "slight release"

    def to_dict(self) -> dict:
        return {
            "attack":  self.attack,
            "decay":   self.decay,
            "sustain": self.sustain,
            "release": self.release,
        }

    @staticmethod
    def from_dict(d: dict) -> "EnvSpec":
        return EnvSpec(
            attack  = float(d.get("attack",  0.06)),
            decay   = float(d.get("decay",   0.10)),
            sustain = float(d.get("sustain", 0.75)),
            release = float(d.get("release", 0.18)),
        )


@dataclass
class SynthPatch:
    """A complete synthesizer preset.

    Each patch has a name, one or more oscillator layers, a filter, an
    amplitude envelope, and a master gain.  The oscillator list is kept in a
    Python list so the front end can append / remove / reorder freely.
    """
    name:        str              = "Untitled"
    oscillators: list[OscSpec]    = field(default_factory=list)
    flt:         FilterSpec       = field(default_factory=FilterSpec)
    env:         EnvSpec          = field(default_factory=EnvSpec)
    gain:        float            = 0.55

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "oscillators": [o.to_dict() for o in self.oscillators],
            "filter":      self.flt.to_dict(),
            "env":         self.env.to_dict(),
            "gain":        self.gain,
        }

    @staticmethod
    def from_dict(d: dict) -> "SynthPatch":
        return SynthPatch(
            name        = d.get("name", "Untitled"),
            oscillators = [OscSpec.from_dict(o) for o in d.get("oscillators", [])],
            flt         = FilterSpec.from_dict(d.get("filter", {})),
            env         = EnvSpec.from_dict(d.get("env", {})),
            gain        = float(d.get("gain", 0.55)),
        )


def default_patch() -> SynthPatch:
    """Factory: a warm detuned-saw pad — usable immediately, sounds good."""
    return SynthPatch(
        name = "Default Pad",
        oscillators = [
            OscSpec(wave="saw", detune_cents= -7.0, level=0.65),
            OscSpec(wave="saw", detune_cents=  0.0, level=0.70),
            OscSpec(wave="saw", detune_cents= +7.0, level=0.65),
            OscSpec(wave="triangle", octave=1, level=0.25, morph=0.15, morph_target="sine"),
        ],
        flt  = FilterSpec(cutoff_hz=1600.0, resonance=0.08),
        env  = EnvSpec(attack=0.08, decay=0.12, sustain=0.72, release=0.22),
        gain = 0.50,
    )


# ── Voice ─────────────────────────────────────────────────────────────────────

_ADSR_ATTACK  = 0
_ADSR_DECAY   = 1
_ADSR_SUSTAIN = 2
_ADSR_RELEASE = 3
_ADSR_DONE    = 4


class Voice:
    """A single sounding note.

    Owns:
    - Per-oscillator phase accumulators (click-free block boundaries)
    - ADSR state machine running at sample rate
    - One-pole lowpass filter state
    """

    __slots__ = (
        "note", "freq", "patch", "vel_scale",
        "_phases", "_adsr_stage", "_adsr_pos",
        "_filt_y",
    )

    def __init__(self, note: int, velocity: int, patch: SynthPatch) -> None:
        self.note      = note
        self.freq      = midi_to_hz(note)
        self.patch     = patch
        self.vel_scale = velocity / 127.0
        n_osc          = len(patch.oscillators)
        self._phases   = np.zeros(n_osc, dtype=np.float64)
        self._adsr_stage = _ADSR_ATTACK
        self._adsr_pos   = 0.0          # samples elapsed in current stage
        self._filt_y     = 0.0          # lowpass IIR state

    # ── Public ────────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._adsr_stage == _ADSR_DONE

    def note_off(self) -> None:
        if self._adsr_stage < _ADSR_RELEASE:
            self._adsr_stage = _ADSR_RELEASE
            self._adsr_pos   = 0.0

    def render(self, frames: int) -> np.ndarray:
        """Return a (frames,) float32 block, mono — mixed and filtered."""
        patch  = self.patch
        env    = patch.env
        flt    = patch.flt

        # ── ADSR envelope ─────────────────────────────────────────────────
        env_block = self._render_adsr(frames, env)

        if self._adsr_stage == _ADSR_DONE:
            return np.zeros(frames, dtype=np.float32)

        # ── Oscillator mix ────────────────────────────────────────────────
        mix = np.zeros(frames, dtype=np.float64)
        t   = np.arange(frames, dtype=np.float64) / SR

        for k, ospec in enumerate(patch.oscillators):
            freq_k = self.freq * (2.0 ** (ospec.octave)) * (
                2.0 ** (ospec.detune_cents / 1200.0)
            )
            phase_block = self._phases[k] + freq_k * t
            sig = _render_wave(
                ospec.wave, phase_block,
                morph        = ospec.morph,
                morph_target = ospec.morph_target,
                wavetable    = (
                    np.array(ospec.wavetable, dtype=np.float32)
                    if ospec.wavetable else None
                ),
            )
            mix += sig * ospec.level
            # Advance phase accumulator (keep fractional part to avoid drift)
            self._phases[k] = (self._phases[k] + freq_k * frames / SR) % 1.0

        # ── One-pole lowpass filter ────────────────────────────────────────
        # coefficient: c = exp(-2π * fc / SR)
        fc = max(20.0, min(flt.cutoff_hz, SR / 2.0 - 1.0))
        c  = math.exp(-2.0 * math.pi * fc / SR)
        a  = 1.0 - c
        # Mild resonance: boost around cutoff by mixing a tiny derivative term
        res_gain = 1.0 + flt.resonance * 3.0
        y = self._filt_y
        out = np.empty(frames, dtype=np.float64)
        for i in range(frames):
            y     = a * mix[i] * res_gain + c * y
            out[i] = y
        self._filt_y = y

        # ── Apply envelope + velocity ──────────────────────────────────────
        result = (out * env_block * self.vel_scale * patch.gain).astype(np.float32)
        return result

    # ── ADSR internals ────────────────────────────────────────────────────

    def _render_adsr(self, frames: int, env: EnvSpec) -> np.ndarray:
        """Per-sample ADSR — returns a (frames,) float64 envelope curve."""
        atk  = max(1, int(env.attack  * SR))
        dec  = max(1, int(env.decay   * SR))
        sus  = env.sustain
        rel  = max(1, int(env.release * SR))
        out  = np.empty(frames, dtype=np.float64)
        pos  = self._adsr_pos
        stage = self._adsr_stage

        for i in range(frames):
            if stage == _ADSR_ATTACK:
                out[i] = pos / atk
                pos += 1.0
                if pos >= atk:
                    stage = _ADSR_DECAY
                    pos   = 0.0
            elif stage == _ADSR_DECAY:
                out[i] = 1.0 - (1.0 - sus) * (pos / dec)
                pos += 1.0
                if pos >= dec:
                    stage = _ADSR_SUSTAIN
                    pos   = 0.0
            elif stage == _ADSR_SUSTAIN:
                out[i] = sus
            elif stage == _ADSR_RELEASE:
                # Release from current sustain level
                out[i] = sus * max(0.0, 1.0 - pos / rel)
                pos += 1.0
                if pos >= rel:
                    stage = _ADSR_DONE
                    pos   = 0.0
            else:  # _ADSR_DONE
                out[i] = 0.0

        self._adsr_pos   = pos
        self._adsr_stage = stage
        return out


# ── SynthEngine ───────────────────────────────────────────────────────────────

class SynthEngine:
    """Real-time polyphonic synthesizer backed by sounddevice.

    Thread-safe: the audio callback and the chord engine run on different
    threads; all voice mutations are guarded by _lock.
    """

    def __init__(self) -> None:
        self._lock:   threading.Lock        = threading.Lock()
        self._voices: dict[int, Voice]      = {}   # note → Voice
        self._patch:  SynthPatch            = default_patch()
        self._stream: Optional[sd.OutputStream] = None
        self.master_volume: float           = 1.0  # 0..1, applied in the audio callback

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the audio output stream.  Safe to call more than once."""
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate = SR,
            blocksize  = BLOCKSIZE,
            channels   = 2,
            dtype      = "float32",
            callback   = self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Silence all voices and close the stream."""
        with self._lock:
            self._voices.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ── Patch ──────────────────────────────────────────────────────────────

    def set_patch(self, patch: SynthPatch) -> None:
        """Set the active patch; new note_on calls will use it."""
        with self._lock:
            self._patch = patch

    # ── Note control ───────────────────────────────────────────────────────

    def note_on(self, note: int, velocity: int = 82) -> None:
        with self._lock:
            # If note already sounding, retrigger rather than stack
            if note in self._voices:
                del self._voices[note]
            # Hard polyphony cap: steal the oldest released voice first,
            # then the oldest held voice
            if len(self._voices) >= MAX_VOICES:
                released = [k for k, v in self._voices.items() if v._adsr_stage == _ADSR_RELEASE]
                if released:
                    del self._voices[released[0]]
                else:
                    # remove any voice
                    del self._voices[next(iter(self._voices))]
            self._voices[note] = Voice(note, velocity, self._patch)

    def note_off(self, note: int) -> None:
        with self._lock:
            if note in self._voices:
                self._voices[note].note_off()

    def all_notes_off(self) -> None:
        with self._lock:
            for v in self._voices.values():
                v.note_off()

    def audition(
        self,
        patch: SynthPatch,
        notes: tuple[int, ...] = (48, 52, 55, 60),
        duration: float = 1.5,
    ) -> None:
        """Trigger a brief test chord with `patch` so the designer can hear edits.

        Non-blocking — the chord plays in the background through the already-
        running stream; a daemon timer sends note-offs after `duration` seconds.
        """
        with self._lock:
            prev_patch = self._patch
            self._patch = patch
            for n in notes:
                if n in self._voices:
                    del self._voices[n]
                self._voices[n] = Voice(n, 82, patch)

        def _release():
            with self._lock:
                for n in notes:
                    if n in self._voices:
                        self._voices[n].note_off()
                self._patch = prev_patch   # restore previous patch

        t = threading.Timer(duration, _release)
        t.daemon = True
        t.start()

    # ── Callback (audio thread) ─────────────────────────────────────────────

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        mix = np.zeros(frames, dtype=np.float32)
        dead: list[int] = []
        with self._lock:
            for note, voice in self._voices.items():
                mix += voice.render(frames)
                if voice.done:
                    dead.append(note)
            for note in dead:
                del self._voices[note]

        # Soft clip to prevent harsh digital distortion when many voices sum
        mix = np.tanh(mix * 1.2) * 0.85

        # Apply master volume
        mix *= float(self.master_volume)

        # Mono → stereo
        outdata[:, 0] = mix
        outdata[:, 1] = mix


# ── PatchStore ────────────────────────────────────────────────────────────────

_PATCHES_PATH = Path(__file__).parent / "patches.json"


class PatchStore:
    """JSON-backed persistent store for SynthPatch presets.

    Patches are identified by name (unique; saving with an existing name
    overwrites).  The JSON file is read fresh on every call so multiple
    processes (or a quick restart) always see the current state.
    """

    def __init__(self, path: Path = _PATCHES_PATH) -> None:
        self._path = path

    # ── Internal ───────────────────────────────────────────────────────────

    def _load_raw(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_raw(self, records: list[dict]) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    # ── Public API ─────────────────────────────────────────────────────────

    def list(self) -> list[str]:
        """Return saved patch names in save order."""
        return [r["name"] for r in self._load_raw() if "name" in r]

    def all(self) -> list[SynthPatch]:
        """Return all saved patches as SynthPatch objects."""
        return [SynthPatch.from_dict(r) for r in self._load_raw()]

    def save(self, patch: SynthPatch) -> None:
        """Persist patch (overwrite if same name exists)."""
        records = [r for r in self._load_raw() if r.get("name") != patch.name]
        records.append(patch.to_dict())
        self._save_raw(records)

    def load(self, name: str) -> Optional[SynthPatch]:
        """Load a named patch, or None if not found."""
        for r in self._load_raw():
            if r.get("name") == name:
                return SynthPatch.from_dict(r)
        return None

    def delete(self, name: str) -> bool:
        """Delete a patch by name.  Returns True if it existed."""
        records = self._load_raw()
        kept = [r for r in records if r.get("name") != name]
        if len(kept) == len(records):
            return False
        self._save_raw(kept)
        return True
