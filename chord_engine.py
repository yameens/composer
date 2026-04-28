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
KICK_NOTE      = 36   # Bass Drum 1
KICK_VELOCITY  = 110
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

def chord_notes(root: str, chord_type: str) -> list[int]:
    base = ROOT_MIDI[root]
    return [base + i for i in CHORD_INTERVALS[chord_type]]

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
        self._current_chord: Optional[tuple[str, str]] = None

        self._beat_running = False
        self._beat_thread: Optional[threading.Thread] = None
        self._beat_fs: Optional[fluidsynth.Synth] = None
        self._sfid: Optional[int] = None

        self._use_synth    = False
        self._synth_preset = 0
        self._synth_active: list[int] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        available = mido.get_output_names()
        iac_ports = [p for p in available if "IAC" in p]
        if not iac_ports:
            raise RuntimeError(
                "IAC Driver not found.\n"
                "Enable it: Audio MIDI Setup → Window → Show MIDI Studio → "
                "double-click IAC Driver → check 'Device is online'."
            )
        print(f"  Opening MIDI port: {iac_ports[0]}")
        self._port = mido.open_output(iac_ports[0])

        # FluidSynth instance — drums (ch 9) + built-in chord synth (ch 0)
        self._beat_fs = fluidsynth.Synth(gain=0.7)
        self._beat_fs.start(driver="coreaudio")
        self._sfid = self._beat_fs.sfload(str(_BEAT_SF2))
        self._beat_fs.program_select(DRUM_CHANNEL, self._sfid, 128, 0)   # GM Standard Kit
        self._beat_fs.program_select(SYNTH_CHANNEL, self._sfid, 0, SYNTH_PROGRAMS[0])  # Piano
        time.sleep(0.25)   # let CoreAudio settle before first note
        print(f"  FluidSynth ready (sfid={self._sfid}) — drums ch9, chord synth ch0")

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

    # ── Output mode ────────────────────────────────────────────────────────

    def set_output_mode(self, use_synth: bool) -> None:
        """Switch between Logic/IAC (use_synth=False) and built-in FluidSynth (use_synth=True)."""
        with self._lock:
            # Kill whatever is currently playing
            if self._use_synth:
                for n in self._synth_active:
                    if self._beat_fs:
                        self._beat_fs.noteoff(SYNTH_CHANNEL, n)
                self._synth_active = []
            else:
                self._kill_all()
                for ch in self._active:
                    self._active[ch] = []

            self._use_synth   = use_synth
            self._current_chord = None   # force re-play on next chord event

            if use_synth and self._beat_fs and self._sfid is not None:
                self._beat_fs.program_select(
                    SYNTH_CHANNEL, self._sfid, 0, SYNTH_PROGRAMS[self._synth_preset]
                )
                print(f"  → Synth mode (GM program {SYNTH_PROGRAMS[self._synth_preset]})")
            else:
                print("  → Logic mode (IAC MIDI)")

    # ── Layer toggle ───────────────────────────────────────────────────────

    def set_layer(self, idx: int, active: bool) -> None:
        """Enable or disable one instrument layer / GM preset slot."""
        if idx < 0 or idx >= len(LAYERS):
            return
        with self._lock:
            self._layer_on[idx] = active

            # ── Synth mode: buttons select GM program, not MIDI channel ──────
            if self._use_synth:
                if active:
                    self._synth_preset = idx
                    if self._beat_fs and self._sfid is not None:
                        self._beat_fs.program_select(
                            SYNTH_CHANNEL, self._sfid, 0, SYNTH_PROGRAMS[idx]
                        )
                    # Re-trigger chord with the new sound
                    if self._current_chord is not None:
                        root, ctype = self._current_chord
                        notes = chord_notes(root, ctype)
                        for n in self._synth_active:
                            if self._beat_fs:
                                self._beat_fs.noteoff(SYNTH_CHANNEL, n)
                        for n in notes:
                            if self._beat_fs:
                                self._beat_fs.noteon(SYNTH_CHANNEL, n, VELOCITY)
                        self._synth_active = notes
                else:
                    # Deselected — revert to Piano
                    self._synth_preset = 0
                    if self._beat_fs and self._sfid is not None:
                        self._beat_fs.program_select(
                            SYNTH_CHANNEL, self._sfid, 0, SYNTH_PROGRAMS[0]
                        )
                return

            # ── Logic mode: original MIDI channel routing ─────────────────────
            ch = LAYERS[idx]["ch"]
            any_layer_on = any(self._layer_on)

            if not active:
                self._kill_channel(ch)
                self._active[ch] = []
                # No layers remain — restore main channel
                if not any_layer_on and self._current_chord is not None:
                    root, ctype = self._current_chord
                    notes = chord_notes(root, ctype)
                    for n in notes:
                        self._noteon(MAIN_CHANNEL, n, VELOCITY)
                    self._active[MAIN_CHANNEL] = notes
            else:
                # Mute main channel
                if self._active[MAIN_CHANNEL]:
                    self._kill_channel(MAIN_CHANNEL)
                    self._active[MAIN_CHANNEL] = []
                # Start playing on this layer immediately
                if self._current_chord is not None:
                    root, ctype = self._current_chord
                    notes = chord_notes(root, ctype)
                    ln = _layer_notes(LAYERS[idx], notes)
                    for n in ln:
                        self._noteon(ch, n, VELOCITY)
                    self._active[ch] = ln

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

    # ── Playback ───────────────────────────────────────────────────────────

    def play(self, root: str, chord_type: str) -> None:
        if self._port is None and not self._use_synth:
            raise RuntimeError("Call start() before play().")

        new_chord = (root, chord_type)
        if new_chord == self._current_chord:
            return

        new_notes = chord_notes(root, chord_type)
        new_set   = set(new_notes)

        with self._lock:
            self._current_chord = new_chord

            # ── Synth path ────────────────────────────────────────────────────
            if self._use_synth:
                if self._beat_fs is None:
                    return
                old_synth = self._synth_active
                self._synth_active = new_notes
                for n in new_notes:
                    self._beat_fs.noteon(SYNTH_CHANNEL, n, VELOCITY)
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
                    self._noteon(MAIN_CHANNEL, n, VELOCITY)
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
                    self._noteon(ch, n, VELOCITY)
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
            for ch in self._active:
                self._active[ch] = []
            for n in self._synth_active:
                if self._beat_fs:
                    self._beat_fs.noteoff(SYNTH_CHANNEL, n)
            self._synth_active = []
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
            # Individual note-offs for each tracked note (reliable with all DAWs)
            for n in self._active.get(ch, []):
                self._noteoff(ch, n)
            # CC 123 as backup
            self._port.send(mido.Message("control_change", channel=ch, control=123, value=0))

    def _kill_channel_notes(self, ch: int, notes: list[int]) -> None:
        for n in notes:
            self._noteoff(ch, n)
        if self._port:
            self._port.send(mido.Message("control_change", channel=ch, control=64, value=0))
            self._port.send(mido.Message("control_change", channel=ch, control=123, value=0))

    def _synth_noteoff_notes(self, notes: list[int]) -> None:
        if self._beat_fs:
            for n in notes:
                self._beat_fs.noteoff(SYNTH_CHANNEL, n)

    @property
    def current_chord(self) -> Optional[tuple[str, str]]:
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
