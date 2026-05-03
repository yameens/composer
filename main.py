"""
Conductor — main entry point.
Run: python main.py  |  Press Q or ESC to quit.
"""

import os
import platform
import re
import subprocess
import sys

import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from hand_tracker import HandTracker
from ui_circles   import (
    draw_circles,
    draw_brand_wordmark,
    angle_to_segment,
    ROOT_LABELS,
    TYPE_LABELS,
    _draw_text_pil,
    _FONT_LG,
    _FONT_SM,
    _FONT_HUD_KEYS,
)
from ui_buttons   import (draw_buttons, get_hovered_button,
                           draw_beat_button, get_hovered_beat_button,
                           draw_mode_button, get_hovered_mode_button)
from ui_theory    import (TheoryState, draw_theory_overlay, update_dwell, navigate,
                           draw_practice_box, PRACTICE_SLIDE)
from chord_engine  import ChordEngine

# ── Config ─────────────────────────────────────────────────────────────────────

WINDOW_NAME        = "Conductor"
# Fallback when auto-pick fails (Linux / Windows). On macOS we prefer built-in FaceTime over iPhone.
CAM_INDEX          = 0
TARGET_W, TARGET_H = 1280, 720
SILENCE_FRAMES     = 3      # frames both segments invalid before auto-silence

# Left-hand pinch → flatten the root by one semitone
FLAT_MAP = {"A": "Ab", "B": "Bb", "C": "Cb", "D": "Db",
            "E": "Eb", "F": "Fb", "G": "Gb"}

# Bottom key row (Q / T / V / S): flush-right block inset from frame edge
HUD_KEYS_MARGIN_RIGHT = 20
# Translucent strip behind bottom wordmark + key hints
HUD_BOTTOM_STRIP_H     = 52
HUD_BOTTOM_STRIP_ALPHA = 0.24

# Arrow-key raw codes for theory navigation (macOS / Linux / other OpenCV builds)
_LEFT_KEYS  = frozenset({63234, 65361, 2, 81})
_RIGHT_KEYS = frozenset({63235, 65363, 3, 83})


def _resolve_camera_index() -> int:
    """Pick OpenCV camera index: built-in FaceTime first, skip iPhone Continuity Camera.

    Uses ffmpeg AVFoundation device listing, which reports cameras in the EXACT same
    order as OpenCV's VideoCapture index numbering on macOS.
    Override with env CONDUCTOR_CAM_INDEX (integer). On non-macOS returns CAM_INDEX.
    """
    raw = os.environ.get("CONDUCTOR_CAM_INDEX", "").strip()
    if raw.isdigit():
        return int(raw)
    if platform.system() != "Darwin":
        return CAM_INDEX
    try:
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=8, check=False,
        )
        # ffmpeg prints device list on stderr
        lines = (proc.stderr or "").splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        lines = []

    video_section = False
    devices: list[tuple[int, str]] = []   # (avf_index, name)
    for line in lines:
        if "AVFoundation video devices" in line:
            video_section = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if not video_section:
            continue
        m = re.search(r"\[(\d+)\]\s+(.*)", line)
        if m:
            devices.append((int(m.group(1)), m.group(2).strip()))

    if not devices:
        return CAM_INDEX

    # Pass 1 — explicit FaceTime / MacBook / built-in HD camera
    for idx, name in devices:
        low = name.lower()
        if "iphone" in low or "continuity" in low or "capture screen" in low:
            continue
        if "facetime" in low or "built-in" in low or "macbook" in low or "hd camera" in low:
            return idx
    # Pass 2 — any non-iPhone device
    for idx, name in devices:
        low = name.lower()
        if "iphone" not in low and "continuity" not in low and "capture screen" not in low:
            return idx
    return CAM_INDEX

# ── HUD ────────────────────────────────────────────────────────────────────────

def _hud_text_width(text: str, font: ImageFont.FreeTypeFont = _FONT_HUD_KEYS, bold: bool = False) -> int:
    """Measure pixel width for the bottom HUD key hints (Jacquard 24 Regular)."""
    stroke = 1 if bold else 0
    dummy  = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox   = dummy.textbbox((0, 0), text, font=font, stroke_width=stroke)
    return bbox[2] - bbox[0]


def _draw_hud(frame: np.ndarray, left_seg: int, right_seg: int,
              root_override: str = "", use_synth: bool = False,
              theory_mode: bool = False,
              voice_lead: bool = False, sync_mode: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    root_label = root_override if root_override else (ROOT_LABELS[left_seg] if left_seg != -1 else "—")
    type_label = TYPE_LABELS[right_seg] if right_seg != -1 else "—"
    mode_label = "SYN" if use_synth else "IAC"
    mode_col   = (50, 230, 255) if use_synth else (0, 215, 255)

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 52), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

    frame = _draw_text_pil(frame, f"{root_label}  {type_label}", (w // 2, 26), _FONT_LG, (230, 230, 230))
    frame = _draw_text_pil(frame, mode_label, (w - 48, 26), _FONT_SM, mode_col)

    # Bottom tray — lightly tinted translucent band behind wordmark + key row
    y_strip = h - HUD_BOTTOM_STRIP_H
    strip_overlay = frame.copy()
    cv2.rectangle(strip_overlay, (0, y_strip), (w, h), (14, 16, 20), thickness=-1, lineType=cv2.LINE_AA)
    blended_strip = cv2.addWeighted(
        strip_overlay[y_strip:h], HUD_BOTTOM_STRIP_ALPHA,
        frame[y_strip:h], 1.0 - HUD_BOTTOM_STRIP_ALPHA, 0,
    )
    frame = frame.copy()
    frame[y_strip:h] = blended_strip

    # Bottom HUD — right-aligned block; Jacquard; underline T / V / S when active.
    # Order: T = theory | V = voice lead | S = sync | Q = quit
    t_txt  = "T = theory"
    mid1   = "   |   "
    v_txt  = "V = voice lead"
    mid2   = "   |   "
    s_txt  = "S = sync"
    mid3   = "   |   "
    q_suf  = "Q = quit"
    t_w  = _hud_text_width(t_txt)
    m1w  = _hud_text_width(mid1)
    v_w  = _hud_text_width(v_txt)
    m2w  = _hud_text_width(mid2)
    s_w  = _hud_text_width(s_txt)
    m3w  = _hud_text_width(mid3)
    q_w  = _hud_text_width(q_suf)
    total_w = t_w + m1w + v_w + m2w + s_w + m3w + q_w
    left_x = w - HUD_KEYS_MARGIN_RIGHT - total_w
    y = h - HUD_BOTTOM_STRIP_H // 2   # vertically centred in the footer strip
    cx = left_x
    _footer_text = (255, 255, 255)
    frame = _draw_text_pil(frame, t_txt, (cx + t_w // 2, y), _FONT_HUD_KEYS, _footer_text,
                           underline=theory_mode)
    cx += t_w
    frame = _draw_text_pil(frame, mid1, (cx + m1w // 2, y), _FONT_HUD_KEYS, _footer_text)
    cx += m1w
    frame = _draw_text_pil(frame, v_txt, (cx + v_w // 2, y), _FONT_HUD_KEYS, _footer_text,
                           underline=voice_lead, underline_gap=6)
    cx += v_w
    frame = _draw_text_pil(frame, mid2, (cx + m2w // 2, y), _FONT_HUD_KEYS, _footer_text)
    cx += m2w
    frame = _draw_text_pil(frame, s_txt, (cx + s_w // 2, y), _FONT_HUD_KEYS, _footer_text,
                           underline=sync_mode)
    cx += s_w
    frame = _draw_text_pil(frame, mid3, (cx + m3w // 2, y), _FONT_HUD_KEYS, _footer_text)
    cx += m3w
    frame = _draw_text_pil(frame, q_suf, (cx + q_w // 2, y), _FONT_HUD_KEYS, _footer_text)
    frame = draw_brand_wordmark(frame, margin_x=14, strip_h=HUD_BOTTOM_STRIP_H)
    return frame

# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("Starting chord engine…")
    engine = ChordEngine()
    try:
        engine.start()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    print("  MIDI ready — sending to Logic via IAC.")

    cam_idx = _resolve_camera_index()
    if platform.system() == "Darwin":
        print(f"  Camera index {cam_idx} (built-in laptop cam preferred over iPhone / Continuity).")
        print("  Tip: CONDUCTOR_CAM_INDEX=1 python main.py  to force a different device.")

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        engine.stop()
        sys.exit(1)

    # Validate that the chosen index delivers real frames.
    # If the first read fails, try the next index before giving up.
    _ok, _test = cap.read()
    if not _ok:
        print(f"  [WARN] Camera {cam_idx} returned no frame — trying index {cam_idx + 1}…")
        cap.release()
        cap = cv2.VideoCapture(cam_idx + 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
        if not cap.isOpened() or not cap.read()[0]:
            print("[ERROR] No working camera found.")
            if platform.system() == "Darwin":
                print("  Check: System Settings → Privacy & Security → Camera → allow Terminal/Python")
                print("  Override: CONDUCTOR_CAM_INDEX=<n> python main.py")
            engine.stop()
            sys.exit(1)
        print(f"  Using camera index {cam_idx + 1}.")

    tracker = HandTracker(max_hands=2)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, TARGET_W, TARGET_H)
    cv2.waitKey(1)   # prime the window event loop before the main loop starts

    # Chord follow-finger state
    prev_left_seg   = -1
    prev_right_seg  = -1
    prev_mods       = (False, False, False)   # (left_flat, right_nine, right_b9)
    silence_counter = 0

    # Voice-leading mode (toggled by V key)
    voice_lead = False

    # Theory overlay (T key) and live practice box (also T, on trigger slide)
    theory_mode   = False
    theory_state  = TheoryState()
    practice_mode = False

    # Sync mode — avoid intermediate chords when root/type update one-at-a-time (S key)
    sync_mode = False
    committed_ls: int | None = None   # last segments passed to engine.play (sound identity)
    committed_rs: int | None = None

    # Instrument buttons — radio: only one layer active at a time (-1 = none)
    active_layer  = -1
    btn_was_in    = [False, False, False]   # edge-detection per instrument button

    # Jersey beat button — independent toggle
    beat_active  = False
    beat_was_in  = False

    # Output mode toggle — Logic (IAC) vs built-in Synth
    synth_mode   = False
    mode_was_in  = False

    print("Conductor running — point fingers at circles to play, hover buttons (bottom-right) to layer instruments.")
    print("Press Q or ESC to quit.\n")

    _bad_frames = 0   # consecutive failed reads
    while True:
        ok, frame = cap.read()
        if not ok:
            _bad_frames += 1
            if _bad_frames > 120:   # ~4 s of failures at 30 fps
                print("[ERROR] Camera stopped delivering frames. Shutting down.")
                break
            cv2.waitKey(1)   # keep the window event loop alive during failures
            continue
        _bad_frames = 0

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Hand tracking ──────────────────────────────────────────────
        frame, hands = tracker.process(frame)
        left  = tracker.get_hand(hands, "Left")
        right = tracker.get_hand(hands, "Right")

        # ── Circle centres ─────────────────────────────────────────────
        lcx, lcy = w // 4,     h // 2
        rcx, rcy = 3 * w // 4, h // 2

        # ── Chord segment detection ────────────────────────────────────
        left_seg  = angle_to_segment(lcx, lcy, *left.index_tip)  if (left  and left.index_extended)  else -1
        right_seg = angle_to_segment(rcx, rcy, *right.index_tip) if (right and right.index_extended) else -1

        # ── Button hover + toggle (synth mode only) ───────────────────
        finger_tips = [hd.index_tip for hd in hands]
        btn_hovered = [False, False, False]

        if synth_mode:
            for tip in finger_tips:
                bi = get_hovered_button(tip, w, h)
                if bi != -1:
                    btn_hovered[bi] = True
                    if not btn_was_in[bi]:                      # rising edge
                        if bi == active_layer:
                            # Re-hover active button → turn layer off
                            engine.set_layer(active_layer, False)
                            active_layer = -1
                            print("  Layer OFF")
                        else:
                            # Switch to a different layer
                            if active_layer != -1:
                                engine.set_layer(active_layer, False)
                            active_layer = bi
                            engine.set_layer(bi, True)
                            print(f"  Layer {bi} ON")

        for i in range(3):
            btn_was_in[i] = btn_hovered[i]

        # ── Jersey beat button ─────────────────────────────────────────
        beat_hovered = any(get_hovered_beat_button(tip, w, h) for tip in finger_tips)
        if beat_hovered and not beat_was_in:          # rising edge
            beat_active = not beat_active
            engine.set_beat(beat_active)
            print(f"  Beat {'ON' if beat_active else 'OFF'}")
        beat_was_in = beat_hovered

        # ── Mode toggle button ─────────────────────────────────────────
        mode_hovered = any(get_hovered_mode_button(tip, w, h) for tip in finger_tips)
        if mode_hovered and not mode_was_in:          # rising edge
            synth_mode = not synth_mode
            engine.set_output_mode(synth_mode)
            prev_left_seg  = -1   # force immediate re-trigger
            prev_right_seg = -1
        mode_was_in = mode_hovered

        # ── Pinch / modifier state ─────────────────────────────────────
        left_flat     = left.pinch_active  if left  else False
        right_nine    = right.pinch_active if right else False
        right_b9      = right.pinch_triple if right else False
        add_nine      = right_nine and not right_b9
        add_flat_nine = right_b9

        # ── Chord audio logic ──────────────────────────────────────────
        # Only hard-clear when both hands vanish; one-hand flicker uses segment silence below.
        if theory_mode:
            pass   # audio suppressed while theory overlay is open
        elif left is None and right is None:
            engine.all_notes_off()
            silence_counter = 0
            committed_ls = committed_rs = None
        elif left_seg == -1 or right_seg == -1:
            silence_counter += 1
            if silence_counter >= SILENCE_FRAMES:
                engine.all_notes_off()
                committed_ls = committed_rs = None
        else:
            silence_counter = 0
            seg_changed   = left_seg != prev_left_seg or right_seg != prev_right_seg
            mods_changed  = (left_flat, right_nine, right_b9) != prev_mods
            if seg_changed or mods_changed:
                root  = ROOT_LABELS[left_seg]
                root  = FLAT_MAP[root] if left_flat else root
                ctype = TYPE_LABELS[right_seg]

                do_play = True
                if sync_mode and committed_ls is not None:
                    root_diff = left_seg != committed_ls
                    typ_diff  = right_seg != committed_rs
                    partial   = root_diff ^ typ_diff   # XOR — only one axis vs committed
                    if partial and not mods_changed:
                        do_play = False

                if do_play:
                    engine.play(root, ctype, add_nine=add_nine, add_flat_nine=add_flat_nine)
                    committed_ls = left_seg
                    committed_rs = right_seg
                    suffix = " (b9)" if add_flat_nine else (" (9)" if add_nine else "")
                    print(f"  {root} {ctype}{suffix}")

        prev_left_seg  = left_seg
        prev_right_seg = right_seg
        prev_mods      = (left_flat, right_nine, right_b9)

        # ── Draw ───────────────────────────────────────────────────────
        frame = draw_circles(
            frame,
            left_hover_idx    = left_seg,
            right_hover_idx   = right_seg,
            left_confirm_idx  = left_seg,
            right_confirm_idx = right_seg,
        )
        btn_active_flags = [i == active_layer for i in range(3)]
        if synth_mode:
            frame = draw_buttons(frame, btn_active_flags, btn_hovered)
        frame = draw_beat_button(frame, beat_active, beat_hovered)
        frame = draw_mode_button(frame, synth_mode, mode_hovered)

        hud_root = FLAT_MAP[ROOT_LABELS[left_seg]] if (left_seg != -1 and left_flat) else ""
        frame = _draw_hud(frame, left_seg, right_seg, root_override=hud_root,
                          use_synth=synth_mode, theory_mode=theory_mode,
                          voice_lead=voice_lead, sync_mode=sync_mode)

        if theory_mode:
            left_fired, right_fired = update_dwell(theory_state, finger_tips, w, h)
            if left_fired:
                navigate(theory_state, -1)
            if right_fired:
                navigate(theory_state, +1)
            frame = draw_theory_overlay(frame, theory_state, finger_tips)
        if practice_mode:
            frame = draw_practice_box(frame)

        cv2.imshow(WINDOW_NAME, frame)
        raw_key = cv2.waitKey(1)
        key     = raw_key & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("t"), ord("T")):
            if practice_mode:
                # Close practice box, reopen theory overlay on same slide
                practice_mode = False
                theory_mode   = True
                print("  Practice box OFF — theory restored")
            else:
                theory_mode = not theory_mode
                if theory_mode:
                    theory_state = TheoryState()
                    engine.all_notes_off()
                prev_left_seg  = -1
                prev_right_seg = -1
                committed_ls = committed_rs = None
                print(f"  Theory {'ON' if theory_mode else 'OFF'}")
        if key in (ord("x"), ord("X")):
            if practice_mode:
                # Cancel everything — close practice box and theory entirely
                practice_mode = False
                theory_mode   = False
                engine.all_notes_off()
                print("  Practice box cancelled")
            elif (theory_mode
                  and theory_state.screen     == "LESSON"
                  and theory_state.lesson_idx == PRACTICE_SLIDE[0]
                  and theory_state.step_idx   == PRACTICE_SLIDE[1]):
                # On the trigger slide — open practice box
                practice_mode = True
                theory_mode   = False
                print("  Practice mode ON")
        # Theory overlay navigation
        if theory_mode:
            if theory_state.screen == "SELECTION":
                if key == ord("0"):
                    theory_state.screen     = "LESSON"
                    theory_state.lesson_idx = 0
                    theory_state.step_idx   = 0
                elif key == ord("1"):
                    theory_state.screen     = "LESSON"
                    theory_state.lesson_idx = 1
                    theory_state.step_idx   = 0
            elif theory_state.screen == "LESSON":
                if raw_key in _LEFT_KEYS:
                    navigate(theory_state, -1)
                elif raw_key in _RIGHT_KEYS:
                    navigate(theory_state, +1)
        if key in (ord("v"), ord("V")):
            voice_lead = not voice_lead
            engine.set_voice_lead(voice_lead)
            prev_left_seg  = -1   # force immediate re-trigger with new voicing mode
            prev_right_seg = -1
            print(f"  Voice lead {'ON' if voice_lead else 'OFF'}")
        if key in (ord("s"), ord("S")):
            sync_mode = not sync_mode
            prev_left_seg   = -1
            prev_right_seg  = -1
            committed_ls = committed_rs = None
            print(f"  Sync {'ON' if sync_mode else 'OFF'}")
        if key == ord(" "):
            engine.all_notes_off()
            committed_ls = committed_rs = None
            print("  PANIC — all notes off")

    print("\nShutting down…")
    engine.stop()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
