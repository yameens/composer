"""
Chunk 3 — Chord Engine (MIDI via IAC Driver → Logic Pro)
Run this file directly to test: python chord_engine.py
Requires: pip install mido python-rtmidi
          IAC Driver enabled in macOS Audio MIDI Setup
"""

import threading
import time
from typing import Optional

import fluidsynth
import mido
from pathlib import Path

from synth_engine import SynthEngine, SynthPatch, default_patch, PatchStore

# ── Beat soundfont (drums only — independent of Logic) ───────────────────────

_BEAT_SF2 = Path(__file__).parent / "soundfonts" / "FluidR3_GM.sf2"

# ── Timing ────────────────────────────────────────────────────────────────────

RELEASE_MS = 80     # ms before old notes are released
VELOCITY   = 82     # MIDI velocity (0–127)

# ── MIDI channels ─────────────────────────────────────────────────────────────
# mido uses 0-indexed channels (0=ch1, 9=ch10 drums)
# Logic shows these as channels 1–16 in the track inspector.

MAIN_CHANNEL = 0   # Logic ch 1 — main synth instrument

# Three instrument layers toggled by the three buttons.
# Logic ch 2, 3, 4 — assign any instrument you like on each track.
# When any layer is active the main channel is muted (layers replace, not stack).

LAYERS = [
    {"ch": 1, "notes": "chord"},     # Logic ch 2
    {"ch": 2, "notes": "chord"},     # Logic ch 3
    {"ch": 3, "notes": "chord"},     # Logic ch 4
]

# ── Jersey beat (130 BPM, 16-step grid) ───────────────────────────────────────
# Pattern: kick on 1 2 3 a-of-3 &-of-4
#   16-step indices (0-based): 0  4  8  11  14
#
#  step:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
#  kick:  ✔           ✔           ✔        ✔        ✔

BEAT_BPM       = 130
DRUM_CHANNEL   = 9    # Logic ch 10 — Drum Kit Designer / UltraBeat
SYNTH_CHANNEL  = 0    # FluidSynth channel for chords (drums stay on ch 9)
SYNTH_PROGRAMS = [0, 48, 89]   # Piano, String Ensemble 1, Pad 2 Warm

GM_INSTRUMENTS = [   # (friendly name, GM program 0-127)
    # pads
    ("warm pad", 89), ("analog pad", 90), ("halo pad", 94), ("sweep pad", 95),
    # strings
    ("violin", 40), ("cello", 42), ("pizzicato strings", 45), ("harp", 46),
    ("warm strings", 48), ("slow strings", 49), ("synth strings", 50), ("choir aahs", 52),
    # organs
    ("drawbar organ", 16), ("rock organ", 18), ("church organ", 19), ("accordion", 21),
    # the rest (original order)
    ("grand piano", 0), ("electric grand", 2), ("rhodes", 4), ("harpsichord", 6),
    ("clavinet", 7), ("vibraphone", 11), ("marimba", 12), ("nylon guitar", 24),
    ("steel guitar", 25), ("jazz guitar", 26), ("clean guitar", 27), ("distortion guitar", 30),
    ("acoustic bass", 32), ("finger bass", 33), ("fretless bass", 35), ("slap bass", 36),
    ("synth bass", 38), ("trumpet", 56), ("trombone", 57), ("french horn", 60),
    ("brass section", 61), ("alto sax", 65), ("tenor sax", 66), ("flute", 73),
    ("square lead", 80), ("saw lead", 81),
]
KICK_NOTE      = 36   # Bass Drum 1 (GM)
KICK_SUB_NOTE  = 35   # Acoustic Bass Drum — layered for low-end body
KICK_VELOCITY  = 122  # primary kick velocity (0–127)
KICK_SUB_VELOCITY = 78   # softer layer under main kick
KICK_RELEASE_S = 0.05
KICK_STEPS     = frozenset({0, 4, 8, 11, 14})
_STEP_S        = 60.0 / (BEAT_BPM * 4)   # ≈ 0.1154 s per 16th note

# ── Chord tables ──────────────────────────────────────────────────────────────

ROOT_MIDI: dict[str, int] = {
    "A": 69, "B": 71, "C": 60, "D": 62, "E": 64, "F": 65, "G": 67,
    # flat counterparts (natural − 1 semitone)
    "Ab": 68, "Bb": 70, "Cb": 59, "Db": 61, "Eb": 63, "Fb": 64, "Gb": 66,
}

CHORD_INTERVALS: dict[str, list[int]] = {
    "Maj":  [0, 4, 7],
    "Maj7": [0, 4, 7, 11],
    "7":    [0, 4, 7, 10],
    "dim":  [0, 3, 6, 10],   # half-diminished (m7b5)
    "Min":  [0, 3, 7],
    "min7": [0, 3, 7, 10],
    "sus4": [0, 5, 7],
}

# chord types that accept a 9th or b9 extension (all except dim)
NINE_COMPATIBLE_TYPES = {"Maj", "Maj7", "7", "Min", "min7", "sus4"}

def voiced_chord_notes(
    root: str,
    chord_type: str,
    add_nine: bool = False,
    add_flat_nine: bool = False,
) -> list[int]:
    """
    Build the full voiced chord (bottom-up):
      [bass, pad, root, ...optional 9/b9..., 3rd, 5th, 7th]

    bass = root - 24  (two octaves below)
    pad  = bass + chord's fifth interval  (consonant fifth above the bass)

    9th / b9 are only inserted for NINE_COMPATIBLE_TYPES chords.
    b9 is only available on dominant "7" (the G7 easter egg).
    """
    base = ROOT_MIDI[root]
    iv   = CHORD_INTERVALS[chord_type]
    bass = base - 24
    pad  = bass + iv[2]      # fifth above bass (uses b5 for half-dim automatically)

    extra: int | None = None
    if chord_type in NINE_COMPATIBLE_TYPES:
        if add_flat_nine and chord_type == "7":
            extra = base + 1   # b9 = root + minor 2nd
        elif add_nine:
            extra = base + 2   # 9  = root + major 2nd

    if extra is not None:
        # insert 9 between root and 3rd: [root, 9, 3rd, 5th, 7th]
        top = [base, extra] + [base + i for i in iv[1:]]
    else:
        top = [base + i for i in iv]

    return [bass, pad] + top


def chord_notes(root: str, chord_type: str) -> list[int]:
    """Backward-compatible thin wrapper around voiced_chord_notes."""
    return voiced_chord_notes(root, chord_type)


def _layer_notes(layer: dict, notes: list[int]) -> list[int]:
    if layer.get("notes") == "root_low":
        return [notes[0] - 12]
    return notes

# ── Chord Engine ──────────────────────────────────────────────────────────────

class ChordEngine:

    def __init__(self):
        self._port: Optional[mido.ports.BaseOutput] = None
        self._lock = threading.Lock()

        self._active: dict[int, list[int]] = {
            MAIN_CHANNEL: [],
            **{L["ch"]: [] for L in LAYERS},
        }
        self._layer_on: list[bool] = [False] * len(LAYERS)
        self._current_chord: Optional[tuple] = None

        self._beat_running = False
        self._beat_thread: Optional[threading.Thread] = None
        self._beat_fs: Optional[fluidsynth.Synth] = None
        self._sfid: Optional[int] = None

        self._use_synth    = False
        self._synth_preset = 0
        self._synth_program = 0
        self._synth_active: list[int] = []

        # Oscillator synth backend
        self._osc: Optional[SynthEngine]  = None
        self._use_osc:     bool           = False
        self._active_patch: SynthPatch    = default_patch()
        self._osc_active:  list[int]      = []

        self._voice_lead = False

        # IAC availability — set in start() after the initial open attempt,
        # and updated in set_backend("iac") on every deliberate switch.
        self._iac_available: bool = False
        self._prev_top: Optional[int] = None

        # Master volume — set via set_master_volume(); default preserves
        # previous behaviour (VELOCITY=82).
        self._master_volume: float = 0.7
        self._velocity: int = VELOCITY

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def _open_iac_port(self) -> bool:
        """Scan for an IAC MIDI port and open it.  Returns True on success."""
        iac_ports = [p for p in mido.get_output_names() if "IAC" in p]
        if not iac_ports:
            return False
        print(f"  Opening MIDI port: {iac_ports[0]}")
        self._port = mido.open_output(iac_ports[0])
        return True

    def start(self) -> None:
        _iac_unavailable = not self._open_iac_port()
        if _iac_unavailable:
            print("[WARN] IAC Driver not found — starting in fluid mode.\n"
                  "Enable it: Audio MIDI Setup → Window → Show MIDI Studio → "
                  "double-click IAC Driver → check 'Device is online'.")
            self._port = None
        self._iac_available = not _iac_unavailable

        # FluidSynth instance — drums (ch 9) + built-in chord synth (ch 0)
        self._beat_fs = fluidsynth.Synth(gain=0.7)
        self._beat_fs.start(driver="coreaudio")
        self._sfid = self._beat_fs.sfload(str(_BEAT_SF2))
        self._beat_fs.program_select(DRUM_CHANNEL, self._sfid, 128, 0)   # GM Standard Kit
        self._beat_fs.program_select(SYNTH_CHANNEL, self._sfid, 0, SYNTH_PROGRAMS[0])  # Piano
        time.sleep(0.25)   # let CoreAudio settle before first note
        print(f"  FluidSynth ready (sfid={self._sfid}) — drums ch9, chord synth ch0")

        if _iac_unavailable:
            self._use_synth = True
            print("  → Fluid mode (IAC unavailable)")

        # Oscillator synth — stream stays open and silent until backend = "osc"
        self._osc = SynthEngine()
        self._osc.set_patch(self._active_patch)
        self._osc.start()
        print("  OscSynth ready")

    def stop(self) -> None:
        self._beat_running = False
        with self._lock:
            self._kill_all()
            for n in self._synth_active:
                if self._beat_fs:
                    self._beat_fs.noteoff(SYNTH_CHANNEL, n)
            self._synth_active = []
        time.sleep(0.1)
        if self._port:
            self._port.close()
            self._port = None
        if self._beat_fs:
            self._beat_fs.delete()
            self._beat_fs = None
        if self._osc:
            self._osc.all_notes_off()
            self._osc.stop()
            self._osc = None

    # ── Output mode ────────────────────────────────────────────────────────

    def active_backend(self) -> str:
        """Return the engine's current backend: 'osc' | 'fluid' | 'iac'."""
        if self._use_osc:
            return "osc"
        if self._use_synth:
            return "fluid"
        return "iac"

    def iac_available(self) -> bool:
        """True when an IAC MIDI port was successfully opened."""
        return self._iac_available

    def set_backend(self, name: str) -> str:
        """Switch active audio backend.  name: 'iac' | 'fluid' | 'osc'.

        Returns the backend that was actually activated.  Switching to 'iac'
        always resolves to 'iac' regardless of port availability — play() is a
        silent no-op when no port is open, so there is never a crash.
        """
        with self._lock:
            self._silence_current()
            self._use_synth = (name == "fluid")
            self._use_osc   = (name == "osc")
            self._current_chord = None

            if name == "fluid" and self._beat_fs and self._sfid is not None:
                self._beat_fs.program_select(
                    SYNTH_CHANNEL, self._sfid, 0, SYNTH_PROGRAMS[self._synth_preset]
                )
                print(f"  → Synth mode (GM program {SYNTH_PROGRAMS[self._synth_preset]})")
            elif name == "osc":
                if self._osc:
                    self._osc.set_patch(self._active_patch)
                print("  → Osc mode (custom synth)")
            else:  # "iac"
                if self._port is not None or self._open_iac_port():
                    self._iac_available = True
                    print("  → logic mode (iac midi)")
                else:
                    self._iac_available = False
                    print("  → iac selected but no iac port — silent until enabled")
                # leave _use_synth = False and _use_osc = False either way

            return self.active_backend()

    def set_output_mode(self, use_synth: bool) -> None:
        """Back-compat shim for main.py toggle (SYN button)."""
        self.set_backend("fluid" if use_synth else "iac")

    def set_active_patch(self, patch: SynthPatch) -> None:
        """Set the patch used for future osc notes (takes effect immediately)."""
        self._active_patch = patch
        if self._osc:
            self._osc.set_patch(patch)

    def preview_patch(
        self,
        patch: SynthPatch,
        notes: tuple[int, ...] = (48, 52, 55, 60),
    ) -> None:
        """Play a short audition chord so the designer can hear edits live."""
        if self._osc:
            self._osc.audition(patch, notes)

    # ── Internal silence helper ─────────────────────────────────────────────

    def _silence_current(self) -> None:
        """Kill notes on the currently selected backend (caller must hold lock)."""
        if self._use_osc and self._osc:
            for n in self._osc_active:
                self._osc.note_off(n)
            self._osc_active = []
        elif self._use_synth:
            for n in self._synth_active:
                if self._beat_fs:
                    self._beat_fs.noteoff(SYNTH_CHANNEL, n)
            self._synth_active = []
        else:
            self._kill_all()
            for ch in self._active:
                self._active[ch] = []

    # ── Layer toggle ───────────────────────────────────────────────────────

    def set_layer(self, idx: int, active: bool) -> None:
        """Enable or disable one instrument layer / GM preset slot."""
        if idx < 0 or idx >= len(LAYERS):
            return
        with self._lock:
            self._layer_on[idx] = active

            # ── Logic mode: original MIDI channel routing ─────────────────────
            ch = LAYERS[idx]["ch"]
            any_layer_on = any(self._layer_on)

            if not active:
                self._kill_channel(ch)
                self._active[ch] = []
                # No layers remain — restore main channel
                if not any_layer_on and self._current_chord is not None:
                    root, ctype, a9, ab9, _vl = self._current_chord
                    full  = voiced_chord_notes(root, ctype, a9, ab9)
                    notes = [full[0]] + self._lead_voicing(full[1:]) if self._voice_lead else full
                    for n in notes:
                        self._noteon(MAIN_CHANNEL, n, self._velocity)
                    self._active[MAIN_CHANNEL] = notes
            else:
                # Mute main channel
                if self._active[MAIN_CHANNEL]:
                    self._kill_channel(MAIN_CHANNEL)
                    self._active[MAIN_CHANNEL] = []
                # Start playing on this layer immediately
                if self._current_chord is not None:
                    root, ctype, a9, ab9, _vl = self._current_chord
                    full  = voiced_chord_notes(root, ctype, a9, ab9)
                    notes = [full[0]] + self._lead_voicing(full[1:]) if self._voice_lead else full
                    ln = _layer_notes(LAYERS[idx], notes)
                    for n in ln:
                        self._noteon(ch, n, self._velocity)
                    self._active[ch] = ln

    def set_synth_program(self, program: int) -> None:
        """Switch the FluidSynth chord channel to a new GM program number.

        Stores the new program, calls program_select, then re-triggers the
        currently sounding chord so the timbre change is immediately audible.
        """
        with self._lock:
            self._synth_program = program
            if self._beat_fs and self._sfid is not None:
                self._beat_fs.program_select(SYNTH_CHANNEL, self._sfid, 0, program)
            # Re-trigger chord with the new sound
            if self._current_chord is not None:
                root, ctype, a9, ab9, _vl = self._current_chord
                full  = voiced_chord_notes(root, ctype, a9, ab9)
                notes = [full[0]] + self._lead_voicing(full[1:]) if self._voice_lead else full
                for n in self._synth_active:
                    if self._beat_fs:
                        self._beat_fs.noteoff(SYNTH_CHANNEL, n)
                for n in notes:
                    if self._beat_fs:
                        self._beat_fs.noteon(SYNTH_CHANNEL, n, self._velocity)
                self._synth_active = notes

    # ── Jersey beat ────────────────────────────────────────────────────────

    def set_beat(self, active: bool) -> None:
        if active and not self._beat_running:
            self._beat_running = True
            self._beat_thread = threading.Thread(
                target=self._beat_loop, daemon=True, name="jersey-beat"
            )
            self._beat_thread.start()
        elif not active:
            self._beat_running = False

    def _beat_loop(self) -> None:
        step = 0
        kick_count = 0
        while self._beat_running:
            t0 = time.perf_counter()
            if step in KICK_STEPS and self._beat_fs is not None:
                self._beat_fs.noteon(DRUM_CHANNEL, KICK_NOTE, KICK_VELOCITY)
                self._beat_fs.noteon(DRUM_CHANNEL, KICK_SUB_NOTE, KICK_SUB_VELOCITY)
                threading.Timer(KICK_RELEASE_S, self._release_kick).start()
                kick_count += 1
                if kick_count <= 3:   # confirm first 3 kicks in terminal
                    print(f"  KICK {kick_count} (step {step})")
            step = (step + 1) % 16
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, _STEP_S - elapsed))

    def _release_kick(self) -> None:
        if self._beat_fs is not None:
            self._beat_fs.noteoff(DRUM_CHANNEL, KICK_NOTE)
            self._beat_fs.noteoff(DRUM_CHANNEL, KICK_SUB_NOTE)

    # ── Voice leading ──────────────────────────────────────────────────────

    def set_master_volume(self, vol: float) -> None:
        """Set master volume (0..1).

        Scales the MIDI note velocity used in chord/layer note-on calls.
        Does not touch KICK_VELOCITY / KICK_SUB_VELOCITY (beat is independent).
        Also adjusts FluidSynth gain and OSC engine master_volume.
        """
        vol = max(0.0, min(1.0, vol))
        self._master_volume = vol
        # Map 0..1 → 30..120 velocity
        self._velocity = max(1, min(127, round(30 + vol * 90)))
        # FluidSynth gain (0.2..1.0 range to stay audible)
        if self._beat_fs is not None:
            try:
                self._beat_fs.set_gain(0.2 + vol * 0.8)
            except Exception:
                pass
        # OSC engine
        if self._osc is not None:
            self._osc.master_volume = vol

    def set_voice_lead(self, on: bool) -> None:
        """Enable or disable strict-closest voice-leading mode."""
        with self._lock:
            self._voice_lead = on
            self._prev_top   = None
            self._current_chord = None   # force retrigger on next chord event

    def _lead_voicing(self, chord_top: list[int]) -> list[int]:
        """
        Given the top-register notes of a chord (no bass/pad), return a
        rotation + octave-shift whose highest note is strictly closest in
        semitones to the previous chord's highest note.

        On the very first call (no previous chord), the notes are returned
        as-is and the top note is remembered for the next call.
        """
        if self._prev_top is None:
            self._prev_top = chord_top[-1]
            return chord_top

        target = self._prev_top
        n = len(chord_top)
        best      = chord_top
        best_dist = abs(chord_top[-1] - target)

        # chord_top is already sorted ascending (intervals are non-negative).
        # Each inversion is built by taking a rotation of the pitch-classes and
        # stacking them in strictly ascending order, then globally shifting by
        # an octave until the top note is as close as possible to target.
        pitch_classes = [p % 12 for p in chord_top]
        for k in range(n):
            # Build an ascending stack starting from pitch class k
            rot_pc = pitch_classes[k:] + pitch_classes[:k]
            notes: list[int] = []
            base_oct = chord_top[k] - pitch_classes[k]   # octave offset for first note
            current = base_oct + rot_pc[0]
            notes.append(current)
            for pc in rot_pc[1:]:
                candidate = base_oct + pc
                while candidate <= notes[-1]:
                    candidate += 12
                notes.append(candidate)
            # Try shifting the whole voicing by ±1 octave to find closest top
            for shift in (-12, 0, 12):
                cand = [x + shift for x in notes]
                d = abs(cand[-1] - target)
                if d < best_dist:
                    best_dist = d
                    best      = cand

        self._prev_top = best[-1]
        return best

    # ── Playback ───────────────────────────────────────────────────────────

    def play(
        self,
        root: str,
        chord_type: str,
        *,
        add_nine: bool = False,
        add_flat_nine: bool = False,
    ) -> None:
        if self._port is None and not self._use_synth and not self._use_osc:
            return   # no usable backend yet — stay silent instead of crashing

        with self._lock:
            new_chord = (root, chord_type, add_nine, add_flat_nine, self._voice_lead)
            if new_chord == self._current_chord:
                return

            full = voiced_chord_notes(root, chord_type, add_nine, add_flat_nine)
            if self._voice_lead:
                # Keep the very-low root fixed; invert everything above (pad + chord)
                new_notes = [full[0]] + self._lead_voicing(full[1:])
            else:
                new_notes = full
            new_set   = set(new_notes)
            self._current_chord = new_chord

            # ── Osc synth path ────────────────────────────────────────────────
            if self._use_osc and self._osc is not None:
                new_osc_set = set(new_notes)
                for n in new_notes:
                    if n not in self._osc_active:
                        self._osc.note_on(n, self._velocity)
                for n in self._osc_active:
                    if n not in new_osc_set:
                        self._osc.note_off(n)
                self._osc_active = new_notes
                return

            # ── Synth path ────────────────────────────────────────────────────
            if self._use_synth:
                if self._beat_fs is None:
                    return
                old_synth = self._synth_active
                self._synth_active = new_notes
                for n in new_notes:
                    self._beat_fs.noteon(SYNTH_CHANNEL, n, self._velocity)
                to_kill = [n for n in old_synth if n not in new_set]
                if to_kill:
                    threading.Timer(
                        RELEASE_MS / 1000.0,
                        self._synth_noteoff_notes,
                        args=(to_kill,),
                    ).start()
                return

            # ── Logic/MIDI path ───────────────────────────────────────────────
            # Main channel — only when no layer is active
            if not any(self._layer_on):
                old_main = self._active[MAIN_CHANNEL]
                self._active[MAIN_CHANNEL] = new_notes
                for n in new_notes:
                    self._noteon(MAIN_CHANNEL, n, self._velocity)
                to_kill = [n for n in old_main if n not in new_set]
                if to_kill:
                    threading.Timer(
                        RELEASE_MS / 1000.0,
                        self._kill_channel_notes,
                        args=(MAIN_CHANNEL, to_kill),
                    ).start()

            # Active layers
            for i, L in enumerate(LAYERS):
                if not self._layer_on[i]:
                    continue
                ch     = L["ch"]
                ln     = _layer_notes(L, new_notes)
                ln_set = set(ln)
                old_ln = self._active[ch]
                self._active[ch] = ln
                for n in ln:
                    self._noteon(ch, n, self._velocity)
                to_kill = [n for n in old_ln if n not in ln_set]
                if to_kill:
                    threading.Timer(
                        RELEASE_MS / 1000.0,
                        self._kill_channel_notes,
                        args=(ch, to_kill),
                    ).start()

    def all_notes_off(self) -> None:
        with self._lock:
            self._kill_all()
            self._panic_all_channels()
            for ch in self._active:
                self._active[ch] = []
            for n in self._synth_active:
                if self._beat_fs:
                    self._beat_fs.noteoff(SYNTH_CHANNEL, n)
            self._synth_active = []
            if self._osc:
                self._osc.all_notes_off()
            self._osc_active = []
            self._current_chord = None

    # ── Internals ──────────────────────────────────────────────────────────

    def _noteon(self, ch: int, note: int, vel: int) -> None:
        if self._port:
            self._port.send(mido.Message("note_on", channel=ch, note=note, velocity=vel))

    def _noteoff(self, ch: int, note: int) -> None:
        if self._port:
            self._port.send(mido.Message("note_off", channel=ch, note=note, velocity=0))

    def _kill_all(self) -> None:
        for ch in self._active:
            self._kill_channel(ch)

    def _kill_channel(self, ch: int) -> None:
        if self._port:
            for n in self._active.get(ch, []):
                self._noteoff(ch, n)
            self._port.send(mido.Message("control_change", channel=ch, control=64,  value=0))
            self._port.send(mido.Message("control_change", channel=ch, control=123, value=0))
            self._port.send(mido.Message("control_change", channel=ch, control=120, value=0))

    def _kill_channel_notes(self, ch: int, notes: list[int]) -> None:
        for n in notes:
            self._noteoff(ch, n)
        if self._port:
            self._port.send(mido.Message("control_change", channel=ch, control=64, value=0))
            self._port.send(mido.Message("control_change", channel=ch, control=123, value=0))

    def _panic_all_channels(self) -> None:
        if not self._port:
            return
        for ch in range(16):
            self._port.send(mido.Message("control_change", channel=ch, control=64,  value=0))
            self._port.send(mido.Message("control_change", channel=ch, control=123, value=0))
            self._port.send(mido.Message("control_change", channel=ch, control=120, value=0))

    def _synth_noteoff_notes(self, notes: list[int]) -> None:
        if self._beat_fs:
            for n in notes:
                self._beat_fs.noteoff(SYNTH_CHANNEL, n)

    @property
    def current_chord(self) -> Optional[tuple]:
        return self._current_chord

# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Available MIDI ports:", mido.get_output_names())
    engine = ChordEngine()
    print("Connecting to IAC…")
    engine.start()
    engine.set_beat(True)
    print("Beat running — 130 BPM jersey pattern (Logic should play kick on ch 10)")
    time.sleep(4.0)

    for root, ctype in [("C", "Maj"), ("A", "Min"), ("F", "Maj7"), ("G", "7")]:
        print(f"  {root} {ctype} → {chord_notes(root, ctype)}")
        engine.play(root, ctype)
        time.sleep(2.0)

    print("Done.")
    engine.stop()
