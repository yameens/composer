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
from pathlib import Path

import cv2
import numpy as np

import ui_circles
from ui_circles import (
    _FONT_MODE,
    _make_text_sprite,
    _blit_text_centered,
    _blit_bgra,
    _to_premul,
)

# Cached sprites for the two mode-button labels — static text, never changes.
_MODE_LABEL_CACHE: dict[str, tuple] = {}


def _get_mode_label_sprite(label: str) -> tuple:
    ts = _MODE_LABEL_CACHE.get(label)
    if ts is None:
        ts = _make_text_sprite(label, _FONT_MODE, (255, 255, 255))
        _MODE_LABEL_CACHE[label] = ts
    return ts

# ── Drummer sprite loader ─────────────────────────────────────────────────────

_ASSETS    = Path(__file__).parent / "assets"
_BEAT_PNG  = _ASSETS / "beat_drummer.png"
_beat_sprite_cache: dict[int, object] = {}   # diameter -> premultiplied BGRA sprite (or None)


def _get_beat_sprite(diam: int):
    """Load and cache the circular drummer sprite at the given pixel diameter."""
    if diam in _beat_sprite_cache:
        return _beat_sprite_cache[diam]
    img = cv2.imread(str(_BEAT_PNG), cv2.IMREAD_UNCHANGED) if _BEAT_PNG.exists() else None
    if img is None:
        _beat_sprite_cache[diam] = None
        return None
    # Ensure BGRA
    if img.ndim == 3 and img.shape[2] == 3:
        a = np.full((*img.shape[:2], 1), 255, np.uint8)
        img = np.concatenate([img, a], axis=2)
    # Trim transparent margin (square, centred) so the artwork fills the button
    # circle exactly instead of sitting inset from the ring.
    alpha = img[..., 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) and len(ys):
        ccx = (int(xs.min()) + int(xs.max())) // 2
        ccy = (int(ys.min()) + int(ys.max())) // 2
        side = max(int(xs.max()) - int(xs.min()), int(ys.max()) - int(ys.min())) + 1
        half = side // 2
        sx0 = max(0, ccx - half); sy0 = max(0, ccy - half)
        sx1 = min(img.shape[1], sx0 + side); sy1 = min(img.shape[0], sy0 + side)
        img = img[sy0:sy1, sx0:sx1]
    # Slight overscale + centre-crop so the artwork's own dark rim is pushed to
    # (and just under) the gold ring rather than peeking inside it — and the
    # drummer reads a touch bigger in the circle.
    big = max(diam + 2, int(round(diam * 1.12)))
    img = cv2.resize(img, (big, big), interpolation=cv2.INTER_AREA)
    off = (big - diam) // 2
    img = img[off:off + diam, off:off + diam]
    # Pixelate — downscale then nearest-upscale for a chunky retro skeleton.
    blocks = max(6, diam // 5)
    img = cv2.resize(img, (blocks, blocks), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (diam, diam), interpolation=cv2.INTER_NEAREST)
    # Circular alpha mask — clip square/white corners to the circle
    mask = np.zeros((diam, diam), np.uint8)
    cv2.circle(mask, (diam // 2, diam // 2), diam // 2, 255, -1, cv2.LINE_AA)
    img[..., 3] = (img[..., 3].astype(np.uint16) * mask // 255).astype(np.uint8)
    spr = _to_premul(img.copy())
    _beat_sprite_cache[diam] = spr
    return spr


# ── Shared geometry ───────────────────────────────────────────────────────────

BTN_RADIUS  = 38
BTN_MARGIN  = 58    # horizontal inset: edge → button centre (left beat / right stack)
BTN_SPACING = 65    # vertical gap between instrument buttons (top-right)

# Top HUD bar in main.py is 52 px tall; first row sits fully below it with air gap.
HEADER_BAR_H          = 52
BTN_BELOW_HEADER_GAP  = 14   # px between bar bottom and top of circle (north padding)
FIRST_ROW_CY          = HEADER_BAR_H + BTN_BELOW_HEADER_GAP + BTN_RADIUS

# ── Instrument buttons — top-right ────────────────────────────────────────────

NUM_BUTTONS = 3

def _btn_centres(w: int, h: int) -> list[tuple[int, int]]:
    """(cx, cy) for each instrument button, bottom-right stack (stacking upward)."""
    return [
        (w - BTN_MARGIN, h - BTN_MARGIN - i * BTN_SPACING)
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
    """Draw the three instrument toggle buttons and return the frame.

    Drawing happens directly on ``frame`` — _filled_circle alpha-blends inside
    a small ROI, so the previous full-frame copy + ``addWeighted(_, 1.0, _, 0.0)``
    round-trip was pure waste.
    """
    h, w = frame.shape[:2]

    for i, (cx, cy) in enumerate(_btn_centres(w, h)):
        on  = active[i]
        hov = hovered[i]

        if on and hov:
            _filled_circle(frame, cx, cy, _C_ON_HOVER_FILL, alpha=0.82)
            cv2.circle(frame, (cx, cy), BTN_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
        elif on:
            _filled_circle(frame, cx, cy, _C_ON_FILL, alpha=0.78)
            cv2.circle(frame, (cx, cy), BTN_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
        elif hov:
            _filled_circle(frame, cx, cy, _C_OFF_RING, alpha=0.18)
            cv2.circle(frame, (cx, cy), BTN_RADIUS, _C_OFF_RING, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), BTN_RADIUS, _C_OFF_RING, 1, cv2.LINE_AA)

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
    """Draw the beat button (top-left) with drummer sprite and return the frame."""
    cx, cy = _BEAT_CX, _BEAT_CY
    diam = BTN_RADIUS * 2
    spr  = _get_beat_sprite(diam)

    if spr is None:
        # Fallback: old coloured ring behaviour
        if active and hovered:
            _filled_circle(frame, cx, cy, _C_BEAT_ON_HOV, alpha=0.82)
            cv2.circle(frame, (cx, cy), BTN_RADIUS, _C_BEAT_RING, 2, cv2.LINE_AA)
        elif active:
            _filled_circle(frame, cx, cy, _C_BEAT_ON, alpha=0.78)
            cv2.circle(frame, (cx, cy), BTN_RADIUS, _C_BEAT_RING, 2, cv2.LINE_AA)
        elif hovered:
            _filled_circle(frame, cx, cy, _C_BEAT_RING, alpha=0.18)
            cv2.circle(frame, (cx, cy), BTN_RADIUS, _C_BEAT_RING, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), BTN_RADIUS, _C_BEAT_RING, 1, cv2.LINE_AA)
        return frame

    if hovered:
        _filled_circle(frame, cx, cy, _C_BEAT_RING, alpha=0.18)
    _blit_bgra(frame, spr, cx - BTN_RADIUS, cy - BTN_RADIUS)
    cv2.circle(frame, (cx, cy), BTN_RADIUS, (0, 0, 0), 3 if hovered else 2, cv2.LINE_AA)
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
    """Return (cx, cy) for the mode button: left of the root circle.

    Reads ui_circles.RADIUS dynamically so changes from set_circle_size()
    are reflected immediately without a stale import-time capture.
    """
    cx = (w // 4 - ui_circles.RADIUS) // 2
    cy = h // 2
    return cx, cy


def get_hovered_mode_button(tip: tuple[int, int], w: int, h: int) -> bool:
    """Return True if the fingertip is inside the mode toggle button."""
    cx, cy = _mode_btn_centre(w, h)
    return math.hypot(tip[0] - cx, tip[1] - cy) <= MODE_BTN_RADIUS

def draw_mode_button(
    frame:      np.ndarray,
    mode_state: str,
    hovered:    bool,
) -> np.ndarray:
    """Draw the IAC / SYN / OSC mode button to the left of the root circle.

    mode_state: "iac" | "syn" | "osc"
    """
    h, w   = frame.shape[:2]
    cx, cy = _mode_btn_centre(w, h)

    if mode_state == "osc":
        fill  = _C_MODE_SYNTH_HOV if hovered else _C_MODE_SYNTH
        label = "OSC"
    elif mode_state == "syn":
        fill  = _C_MODE_SYNTH_HOV if hovered else _C_MODE_SYNTH
        label = "SYN"
    else:
        fill  = _C_MODE_LOGIC_HOV if hovered else _C_MODE_LOGIC
        label = "IAC"

    _filled_circle(frame, cx, cy, fill, alpha=0.85, radius=MODE_BTN_RADIUS)
    cv2.circle(frame, (cx, cy), MODE_BTN_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
    _blit_text_centered(frame, _get_mode_label_sprite(label), cx, cy)
    return frame

# ── Helper ────────────────────────────────────────────────────────────────────

def _filled_circle(
    img: np.ndarray,
    cx: int, cy: int,
    colour: tuple[int, int, int],
    alpha: float,
    radius: int = BTN_RADIUS,
) -> None:
    """Draw a semi-transparent filled circle in-place on img.

    Operates on a tight ROI around the circle so we never copy the full frame
    just to alpha-blend a ~57-px button.
    """
    h, w = img.shape[:2]
    pad = 1                                # 1-px halo for AA edge
    x1 = max(0, cx - radius - pad)
    y1 = max(0, cy - radius - pad)
    x2 = min(w, cx + radius + pad + 1)
    y2 = min(h, cy + radius + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.circle(
        overlay, (cx - x1, cy - y1), radius, colour,
        thickness=-1, lineType=cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
