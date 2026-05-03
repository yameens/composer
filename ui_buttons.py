"""
Instrument toggle buttons — top-right corner (3 dots, radio-button behaviour).
Jersey beat button      — top-left  corner (1 sky-blue dot, independent toggle).

Visual states:
  Off + not hovered : faint ring, no fill
  Off + hovered     : brighter ring, very slight fill
  On  (active)      : solid coloured fill
  On  + hovered     : slightly darker fill (signals toggleable off)
"""

import math

import cv2
import numpy as np

from ui_circles import RADIUS, _draw_text_pil, _FONT_MODE

# ── Shared geometry ───────────────────────────────────────────────────────────

BTN_RADIUS  = 28
BTN_MARGIN  = 58    # horizontal inset: edge → button centre (left beat / right stack)
BTN_SPACING = 65    # vertical gap between instrument buttons (top-right)

# Top HUD bar in main.py is 52 px tall; first row sits fully below it with air gap.
HEADER_BAR_H          = 52
BTN_BELOW_HEADER_GAP  = 14   # px between bar bottom and top of circle (north padding)
FIRST_ROW_CY          = HEADER_BAR_H + BTN_BELOW_HEADER_GAP + BTN_RADIUS

# ── Instrument buttons — top-right ────────────────────────────────────────────

NUM_BUTTONS = 3

def _btn_centres(w: int, h: int) -> list[tuple[int, int]]:
    """(cx, cy) for each instrument button, top to bottom."""
    return [
        (w - BTN_MARGIN, FIRST_ROW_CY + i * BTN_SPACING)
        for i in range(NUM_BUTTONS)
    ]

# Colours BGR
_C_OFF_RING      = (255, 255, 255)
_C_ON_FILL       = (  0,   0,   0)   # black
_C_ON_HOVER_FILL = ( 40,  40,  40)   # dark gray (slight lift on hover)

def get_hovered_button(tip: tuple[int, int], w: int, h: int) -> int:
    """Return index 0-2 of the instrument button under the fingertip, or -1."""
    for i, (cx, cy) in enumerate(_btn_centres(w, h)):
        if math.hypot(tip[0] - cx, tip[1] - cy) <= BTN_RADIUS:
            return i
    return -1

def draw_buttons(
    frame:   np.ndarray,
    active:  list[bool],
    hovered: list[bool],
) -> np.ndarray:
    """Draw the three instrument toggle buttons and return the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for i, (cx, cy) in enumerate(_btn_centres(w, h)):
        on  = active[i]
        hov = hovered[i]

        if on and hov:
            _filled_circle(overlay, cx, cy, _C_ON_HOVER_FILL, alpha=0.82)
            cv2.circle(overlay, (cx, cy), BTN_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
        elif on:
            _filled_circle(overlay, cx, cy, _C_ON_FILL, alpha=0.78)
            cv2.circle(overlay, (cx, cy), BTN_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
        elif hov:
            _filled_circle(overlay, cx, cy, _C_OFF_RING, alpha=0.18)
            cv2.circle(overlay, (cx, cy), BTN_RADIUS, _C_OFF_RING, 2, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (cx, cy), BTN_RADIUS, _C_OFF_RING, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)
    return frame

# ── Jersey beat button — top-left ─────────────────────────────────────────────

_BEAT_CX = BTN_MARGIN
_BEAT_CY = FIRST_ROW_CY   # same vertical row as top instrument button (below header)

# Gold: BGR for RGB #FFD700
_C_BEAT_ON     = (  0, 215, 255)
_C_BEAT_ON_HOV = (  0, 175, 210)
_C_BEAT_RING   = (  0, 215, 255)

def get_hovered_beat_button(tip: tuple[int, int], w: int, h: int) -> bool:
    """Return True if the fingertip is inside the jersey beat button."""
    return math.hypot(tip[0] - _BEAT_CX, tip[1] - _BEAT_CY) <= BTN_RADIUS

def draw_beat_button(
    frame:   np.ndarray,
    active:  bool,
    hovered: bool,
) -> np.ndarray:
    """Draw the jersey beat button (top-left) and return the frame."""
    cx, cy = _BEAT_CX, _BEAT_CY
    overlay = frame.copy()

    if active and hovered:
        _filled_circle(overlay, cx, cy, _C_BEAT_ON_HOV, alpha=0.82)
        cv2.circle(overlay, (cx, cy), BTN_RADIUS, _C_BEAT_RING, 2, cv2.LINE_AA)
    elif active:
        _filled_circle(overlay, cx, cy, _C_BEAT_ON, alpha=0.78)
        cv2.circle(overlay, (cx, cy), BTN_RADIUS, _C_BEAT_RING, 2, cv2.LINE_AA)
    elif hovered:
        _filled_circle(overlay, cx, cy, _C_BEAT_RING, alpha=0.18)
        cv2.circle(overlay, (cx, cy), BTN_RADIUS, _C_BEAT_RING, 2, cv2.LINE_AA)
    else:
        cv2.circle(overlay, (cx, cy), BTN_RADIUS, _C_BEAT_RING, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)
    return frame

# ── Mode toggle button — left of root circle ──────────────────────────────────
# Position: horizontally centred between screen left edge and the left circle's
# left edge; vertically aligned with the circle centre (h//2).
# L = Logic/IAC mode   S = Synth/FluidSynth mode

MODE_BTN_RADIUS = 22

_C_MODE_LOGIC      = (  0, 215, 255)   # gold — IAC mode
_C_MODE_LOGIC_HOV  = (  0, 175, 210)
_C_MODE_SYNTH      = ( 20, 200, 245)   # slightly lighter gold — SYN mode
_C_MODE_SYNTH_HOV  = (  0, 160, 200)


def _mode_btn_centre(w: int, h: int) -> tuple[int, int]:
    """Return (cx, cy) for the mode button: left of the root circle."""
    cx = (w // 4 - RADIUS) // 2
    cy = h // 2
    return cx, cy


def get_hovered_mode_button(tip: tuple[int, int], w: int, h: int) -> bool:
    """Return True if the fingertip is inside the mode toggle button."""
    cx, cy = _mode_btn_centre(w, h)
    return math.hypot(tip[0] - cx, tip[1] - cy) <= MODE_BTN_RADIUS

def draw_mode_button(
    frame:     np.ndarray,
    use_synth: bool,
    hovered:   bool,
) -> np.ndarray:
    """Draw the Logic/Synth mode toggle button to the left of the root circle."""
    h, w   = frame.shape[:2]
    cx, cy = _mode_btn_centre(w, h)
    overlay = frame.copy()

    if use_synth:
        fill = _C_MODE_SYNTH_HOV if hovered else _C_MODE_SYNTH
        label = "SYN"
    else:
        fill = _C_MODE_LOGIC_HOV if hovered else _C_MODE_LOGIC
        label = "IAC"

    _filled_circle(overlay, cx, cy, fill, alpha=0.85, radius=MODE_BTN_RADIUS)
    cv2.circle(overlay, (cx, cy), MODE_BTN_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)
    frame = _draw_text_pil(frame, label, (cx, cy), _FONT_MODE, (255, 255, 255))
    return frame

# ── Helper ────────────────────────────────────────────────────────────────────

def _filled_circle(
    img: np.ndarray,
    cx: int, cy: int,
    colour: tuple[int, int, int],
    alpha: float,
    radius: int = BTN_RADIUS,
) -> None:
    """Draw a semi-transparent filled circle in-place on img."""
    temp = img.copy()
    cv2.circle(temp, (cx, cy), radius, colour, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(temp, alpha, img, 1 - alpha, 0, img)
