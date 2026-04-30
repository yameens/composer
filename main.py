"""
Conductor — main entry point.
Run: python main.py  |  Press Q or ESC to quit.
"""

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
from chord_engine  import ChordEngine

# ── Config ─────────────────────────────────────────────────────────────────────

WINDOW_NAME        = "Conductor"
CAM_INDEX          = 1
TARGET_W, TARGET_H = 1280, 720
SILENCE_FRAMES     = 3      # frames both segments invalid before auto-silence

# Left-hand pinch → flatten the root by one semitone
FLAT_MAP = {"A": "Ab", "B": "Bb", "C": "Cb", "D": "Db",
            "E": "Eb", "F": "Fb", "G": "Gb"}

# Bottom key row (Q / T / V / S): flush-right block inset from frame edge
HUD_KEYS_MARGIN_RIGHT = 20
# Translucent strip behind bottom wordmark + key hints
HUD_BOTTOM_STRIP_H     = 52
HUD_BOTTOM_STRIP_ALPHA = 0.14

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
    mode_col   = (100, 220, 100) if use_synth else (40, 160, 235)

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 52), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

    frame = _draw_text_pil(frame, f"{root_label}  {type_label}", (w // 2, 26), _FONT_LG, (230, 230, 230))
    frame = _draw_text_pil(frame, mode_label, (w - 48, 26), _FONT_SM, mode_col)

    # Bottom tray — lightly tinted translucent band behind wordmark + key row
    y_strip = h - HUD_BOTTOM_STRIP_H
    strip_overlay = frame.copy()
    cv2.rectangle(strip_overlay, (0, y_strip), (w, h), (22, 24, 28), thickness=-1, lineType=cv2.LINE_AA)
    blended_strip = cv2.addWeighted(
        strip_overlay[y_strip:h], HUD_BOTTOM_STRIP_ALPHA,
        frame[y_strip:h], 1.0 - HUD_BOTTOM_STRIP_ALPHA, 0,
    )
    frame = frame.copy()
    frame[y_strip:h] = blended_strip

    # Bottom HUD — right-aligned block; Jacquard; underline T / V / S when active.
    q_pref = "Q = quit   |   "
    t_txt  = "T = theory"
    mid1   = "   |   "
    v_txt  = "V = voice lead"
    mid2   = "   |   "
    s_txt  = "S = sync"
    q_w = _hud_text_width(q_pref)
    t_w = _hud_text_width(t_txt)
    m1w = _hud_text_width(mid1)
    v_w = _hud_text_width(v_txt)
    m2w = _hud_text_width(mid2)
    s_w = _hud_text_width(s_txt)
    total_w = q_w + t_w + m1w + v_w + m2w + s_w
    left_x = w - HUD_KEYS_MARGIN_RIGHT - total_w
    y = h - 18
    cx = left_x
    frame = _draw_text_pil(frame, q_pref, (cx + q_w // 2, y), _FONT_HUD_KEYS, (0, 0, 0))
    cx += q_w
    frame = _draw_text_pil(frame, t_txt, (cx + t_w // 2, y), _FONT_HUD_KEYS, (0, 0, 0),
                           underline=theory_mode)
    cx += t_w
    frame = _draw_text_pil(frame, mid1, (cx + m1w // 2, y), _FONT_HUD_KEYS, (0, 0, 0))
    cx += m1w
    frame = _draw_text_pil(frame, v_txt, (cx + v_w // 2, y), _FONT_HUD_KEYS, (0, 0, 0),
                           underline=voice_lead)
    cx += v_w
    frame = _draw_text_pil(frame, mid2, (cx + m2w // 2, y), _FONT_HUD_KEYS, (0, 0, 0))
    cx += m2w
    frame = _draw_text_pil(frame, s_txt, (cx + s_w // 2, y), _FONT_HUD_KEYS, (0, 0, 0),
                           underline=sync_mode)
    frame = draw_brand_wordmark(frame, margin_x=14, margin_bottom=14)
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
    prev_mods       = (False, False, False)   # (left_flat, right_nine, right_b9)
    silence_counter = 0

    # Voice-leading mode (toggled by V key)
    voice_lead = False

    # Theory overlay mode (toggled by T key — hook real theory UI later)
    theory_mode = False

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
        if left is None and right is None:
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
        frame = draw_buttons(frame, btn_active_flags, btn_hovered)
        frame = draw_beat_button(frame, beat_active, beat_hovered)
        frame = draw_mode_button(frame, synth_mode, mode_hovered)

        hud_root = FLAT_MAP[ROOT_LABELS[left_seg]] if (left_seg != -1 and left_flat) else ""
        frame = _draw_hud(frame, left_seg, right_seg, root_override=hud_root,
                          use_synth=synth_mode, theory_mode=theory_mode,
                          voice_lead=voice_lead, sync_mode=sync_mode)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("t"), ord("T")):
            theory_mode = not theory_mode
            prev_left_seg   = -1
            prev_right_seg  = -1
            committed_ls = committed_rs = None
            print(f"  Theory {'ON' if theory_mode else 'OFF'}")
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
