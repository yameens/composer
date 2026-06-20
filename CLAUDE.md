# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Conductor is a webcam-based, hands-free chord instrument: a MediaPipe hand tracker turns finger positions into a root + chord-type selection, which is voiced and sent to one of three audio backends. It's also an educational tool — a theory overlay teaches chord/circle-of-fifths concepts. macOS-first (CoreAudio, IAC MIDI, FaceTime-camera preference), though it falls back on other platforms.

## Running

```bash
./run.sh                              # = .venv/bin/python main.py
.venv/bin/python main.py              # direct
CONDUCTOR_CAM_INDEX=1 .venv/bin/python main.py   # force camera device (try 0/1/2)
```

There is **no test suite, linter, or build step**. Each engine module is independently runnable for manual testing — `python hand_tracker.py`, `python chord_engine.py`, `python synth_engine.py` each have a `__main__` smoke test.

Dependencies live in `.venv/` (gitignored). `requirements.txt` lists opencv-python, mediapipe, pyfluidsynth, Pillow, sounddevice; `mido` + `python-rtmidi` are also required (for IAC MIDI). `soundfonts/` (large `.sf2` files) and `assets/hand_landmarker.task` are gitignored — they must exist locally.

### macOS camera gotcha
On macOS the app auto-picks the built-in FaceTime camera and avoids an iPhone "Continuity Camera." If the wrong device opens, use `CONDUCTOR_CAM_INDEX`. See `_resolve_camera_index()` in `main.py`.

## Architecture

The whole app is a single synchronous render loop in `main.py:main()`. Each frame: read camera → (optionally pixelate the camera image, see Settings) → run hand tracking → map fingers to chord segments → draw UI → handle keys/mouse → drive the audio engine. There is no event/callback architecture except the OpenCV mouse callback (`on_mouse`), which drives the synth editor, the settings overlay (slider drag / arrow / tab hits), and cursor clicks on the on-screen mode/beat buttons. The callback only sets request flags / draft state; the actual engine calls happen back in the loop body.

### Signal flow: hand → chord → sound
1. **`hand_tracker.py`** (`HandTracker`) runs MediaPipe on a downscaled frame (`INFERENCE_SCALE`), returns up to two `HandData` (pixel landmarks + debounced gesture flags). Gestures use frame-count hysteresis (`_DebouncedSignal`) and prefer world-space landmark distances, falling back to normalized. Gestures: **pinch** (index+middle, left hand → flatten root), **triple pinch** (index+middle+ring, right hand → b9), **index_extended** (fist guard — a segment only registers when the index finger is actually pointing).
2. **`ui_circles.py`** `angle_to_segment()` converts an index-fingertip angle around a circle center into one of **7 segments**. Left circle = root (`ROOT_LABELS`, circle-of-fifths order), right circle = chord type (`TYPE_LABELS`). This module owns nearly all drawing (PIL text → cached BGRA sprites alpha-blitted onto the cv2 frame; fonts in `assets/`).
3. **`chord_engine.py`** `ChordEngine.play(root, type, add_nine, add_flat_nine)` voices the chord (`voiced_chord_notes` — bass two octaves down + consonant fifth pad + chord tones, with optional 9/b9) and routes it to the active backend. Dedupes against `_current_chord` so unchanged frames are no-ops.

### Three audio backends (cycled by the on-screen MODE button → `set_backend`)
- **`iac`** — MIDI note on/off to Logic Pro via the macOS IAC Driver (the default; requires IAC enabled in Audio MIDI Setup). Supports 3 instrument **layers** (Logic ch 2–4, radio-selected by the instrument buttons; main ch 1 mutes when a layer is on).
- **`fluid`** (HUD label "SYN") — built-in FluidSynth GM synth (ch 0), same FluidSynth instance that always plays the **drum beat** (ch 9). The Jersey beat (130 BPM, 16-step kick pattern) is an independent toggle running in `_beat_thread`. Instruments are chosen from a keyboard-navigated picker (`ui_instruments.py`) backed by the curated `GM_INSTRUMENTS` list in `chord_engine.py` (~40 entries). The picker auto-opens on entering SYN mode; reopen anytime with the `I` key. `ChordEngine.set_synth_program(program)` applies the change and re-triggers the current chord immediately.
- **`osc`** — custom numpy/sounddevice synthesizer in **`synth_engine.py`** (`SynthEngine`, `SynthPatch`/`OscSpec`/`FilterSpec`/`EnvSpec`, up to 4 oscillators + filter + ADSR, `MAX_VOICES=24`). Patches persist as JSON in `patches.json` via `PatchStore`.

### UI overlay modules
`main.py` orchestrates overlays via state flags. `osc_ui` is the single **modal-overlay state machine** — a string in `{"off","editor","sounds","settings","instruments","iac_help"}`; when it's not `"off"`, hand tracking/chord logic is skipped and the loop draws a cached dimmed backdrop plus the active overlay. The rest are booleans (`theory_mode`, `practice_mode`, `sync_mode`, `voice_lead`):
- **`ui_buttons.py`** — beat button and mode-cycle button (hover-to-activate via fingertip). The former instrument layer buttons have been removed; the module no longer owns those.
- **`ui_instruments.py`** — SYN-mode instrument picker overlay (`I` key, or auto-opens on entering SYN). `draw_instrument_picker(frame, instruments, sel_idx)` draws a translucent-black rounded panel with a scrollable list of GM instrument names and a gold highlight bar on the selected row.
- **`ui_theory.py`** — theory lesson overlay (`T` key). Lessons/progressions are data tables (`LESSONS`, `PROGRESSIONS`); navigation is arrow-key + dwell (`update_dwell`). The "practice box" (`X` key, only on the `PRACTICE_SLIDE`) overlays a live circle-of-fifths drill.
- **`ui_synth.py`** — the oscillator synth editor + saved-sounds list (`O` key). Driven by the mouse callback in `main()`; `editor_hit`/`sounds_hit`/`slider_value_at` map clicks to patch edits. `OSC_PARAMS`/`GLOBAL_PARAMS` are the param schema shared with `main.py`.
- **`ui_settings.py` + `settings.py`** — the settings card, opened by the top-right **gear** (`draw_settings_gear`/`get_hovered_settings_gear`; `osc_ui == "settings"`). `settings.py` is the persisted `Settings` dataclass (`tracker_color`, `circle_radius`, `master_volume`, `resolution`) loaded from / saved to `settings.json`. The card has a single styled **"controls"** tab with four rows (tracker color, circle size, volume, resolution); `settings_hit`/`settings_slider_value_at` map mouse/fingertip positions to controls, and edits go to a `settings_draft` that only commits on **Apply** (Cancel reverts to a snapshot). Apply also pushes live values (`set_circle_size`, `_set_landmark_colour`, `engine.set_master_volume`, `render_resolution`). The tab strip is kept for style even though there is only one tab.
- **`ui_iac_help.py`** — the IAC/FL-Studio setup help card (`osc_ui == "iac_help"`), shown when IAC MIDI isn't available; left/right arrows switch the Logic vs FL Studio tab.

### Keyboard map (handled at the bottom of the main loop)
`Q`/ESC quit (ESC only closes overlays when one is open) · `T` theory · `V` voice-leading (`set_voice_lead` — keeps the low root fixed, inverts upper voices for smooth movement) · `S` sync (avoids transient chords while root/type update one finger at a time) · `O` open synth editor / hold for sounds list · `I` open/close SYN instrument picker (only active in SYN mode). Arrow-key handling uses raw OpenCV keycodes (`_LEFT_KEYS` etc.) which differ across platforms.

## Conventions
- **Performance:** all static text is rendered once into BGRA sprites and cached, then alpha-blitted per frame — never re-render PIL text in the hot loop. `_prewarm_caches()` builds these up front. New persistent UI text should follow the same sprite-cache pattern.
- Theory overlay text is **all lowercase** by design (per `theory/lessonDesign.txt`).
- **Resolution / pixelation:** the `resolution` setting (1.0 = full) drives `_pixelate()` (downscale → `INTER_NEAREST` upscale). It's applied twice per frame: a heavy pass on the raw camera image *before* overlays (so only the person pixelates, UI stays crisp), and a gentle whole-frame pass before `imshow` (only when no overlay is open).
- Module docstrings call files "Chunk N" — historical build-order labels, not a dependency ordering.
- The osc synth feature is documented in `sounds-logic.txt`/`oscillator.txt`; note its caveat: the osc path does **not** yet respect voice-leading / b9.
