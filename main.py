"""
Conductor — main entry point.
Run: python main.py  |  Press Q or ESC to quit.
"""

import sys

import cv2
import numpy as np

from hand_tracker import HandTracker
from ui_circles   import draw_circles, angle_to_segment, ROOT_LABELS, TYPE_LABELS, _draw_text_pil, _FONT_LG, _FONT_SM
from ui_buttons   import (draw_buttons, get_hovered_button,
                           draw_beat_button, get_hovered_beat_button)
from chord_engine  import ChordEngine

# ── Config ─────────────────────────────────────────────────────────────────────

WINDOW_NAME        = "Conductor"
CAM_INDEX          = 1
TARGET_W, TARGET_H = 1280, 720
SILENCE_FRAMES     = 6    # frames before auto-silence (~200ms at 30fps)

# ── HUD ────────────────────────────────────────────────────────────────────────

def _draw_hud(frame: np.ndarray, left_seg: int, right_seg: int) -> np.ndarray:
    h, w = frame.shape[:2]
    root_label = ROOT_LABELS[left_seg]  if left_seg  != -1 else "—"
    type_label = TYPE_LABELS[right_seg] if right_seg != -1 else "—"

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 52), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

    frame = _draw_text_pil(frame, f"{root_label}  {type_label}", (w // 2, 26), _FONT_LG, (230, 230, 230))
    frame = _draw_text_pil(frame, "Point index finger into a segment to play   |   Q = quit",
                           (w // 2, h - 18), _FONT_SM, (0, 0, 0))
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

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        engine.stop()
        sys.exit(1)

    tracker = HandTracker(max_hands=2)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, TARGET_W, TARGET_H)

    # Chord follow-finger state
    prev_left_seg   = -1
    prev_right_seg  = -1
    silence_counter = 0

    # Instrument buttons — radio: only one layer active at a time (-1 = none)
    active_layer  = -1
    btn_was_in    = [False, False, False]   # edge-detection per instrument button

    # Jersey beat button — independent toggle
    beat_active  = False
    beat_was_in  = False

    print("Conductor running — point fingers at circles to play, hover buttons (bottom-right) to layer instruments.")
    print("Press Q or ESC to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

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
        left_seg  = angle_to_segment(lcx, lcy, *left.index_tip)  if left  else -1
        right_seg = angle_to_segment(rcx, rcy, *right.index_tip) if right else -1

        # ── Button hover + toggle ──────────────────────────────────────
        finger_tips = [hd.index_tip for hd in hands]
        btn_hovered = [False, False, False]

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

        # ── Chord audio logic ──────────────────────────────────────────
        if left_seg == -1 and right_seg == -1:
            silence_counter += 1
            if silence_counter >= SILENCE_FRAMES:
                engine.all_notes_off()
        else:
            silence_counter = 0
            if left_seg != prev_left_seg or right_seg != prev_right_seg:
                if left_seg != -1 and right_seg != -1:
                    root  = ROOT_LABELS[left_seg]
                    ctype = TYPE_LABELS[right_seg]
                    engine.play(root, ctype)
                    print(f"  {root} {ctype}")

        prev_left_seg  = left_seg
        prev_right_seg = right_seg

        # ── Draw ───────────────────────────────────────────────────────
        frame = draw_circles(
            frame,
            left_hover_idx    = left_seg,
            right_hover_idx   = right_seg,
            left_confirm_idx  = left_seg,
            right_confirm_idx = right_seg,
        )
        btn_active_flags = [i == active_layer for i in range(3)]
        frame = draw_buttons(frame, btn_active_flags, btn_hovered)
        frame = draw_beat_button(frame, beat_active, beat_hovered)
        frame = _draw_hud(frame, left_seg, right_seg)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    print("\nShutting down…")
    engine.stop()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
