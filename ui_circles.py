"""
Chunk 2 — Circle UI & Section Selection
Run this file directly to test: python ui_circles.py
Move your mouse over the circles to simulate fingertip hover.
Press Q to quit.
"""

import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Layout constants ──────────────────────────────────────────────────────────

RADIUS       = 200       # outer radius
INNER_RADIUS = 30        # hole radius
NUM_SEGMENTS = 7
SEG_ANGLE    = 360 / NUM_SEGMENTS   # ≈ 51.43°
START_OFFSET = -90       # top of circle = segment 0

# Translucency — near-transparent fills, just lines visible
ALPHA_BASE    = 0.0    # invisible fill; only divider lines show
ALPHA_HOVER   = 0.30   # subtle fill on hovered segment
ALPHA_CONFIRM = 0.45   # slightly stronger on active/playing segment

# Segment label sets
ROOT_LABELS = ["A", "D", "G", "C", "F", "B", "E"]   # circle of fifths order
TYPE_LABELS = ["Maj", "Maj7", "7", "dim", "Min", "min7", "sus4"]

# Colour palette (BGR)
PALETTE = {
    "left":  {
        "base":    ( 40, 140, 255),
        "hover":   ( 80, 190, 255),
        "confirm": (160, 230, 255),
        "border":  (255, 255, 255),
        "text":    (255, 255, 255),
    },
    "right": {
        "base":    (200,  60, 255),
        "hover":   (230, 120, 255),
        "confirm": (255, 200, 255),
        "border":  (255, 255, 255),
        "text":    (255, 255, 255),
    },
}

# Font
_FONT_PATH = Path(__file__).parent / "assets" / "FiraMono-Regular.ttf"

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(_FONT_PATH), size)
    except Exception:
        return ImageFont.load_default()

_FONT_LG = _load_font(28)   # root labels (single letters)
_FONT_SM = _load_font(20)   # type labels (short strings)

# ── Core geometry helpers ─────────────────────────────────────────────────────

def _seg_angles(seg_idx: int) -> tuple[float, float]:
    """Return (start_deg, end_deg) for a segment in OpenCV's angle convention."""
    start = START_OFFSET + seg_idx * SEG_ANGLE
    end   = start + SEG_ANGLE
    return start, end

def _seg_mid_angle_rad(seg_idx: int) -> float:
    """Midpoint angle of a segment in radians (standard math convention)."""
    mid_deg = START_OFFSET + (seg_idx + 0.5) * SEG_ANGLE
    return math.radians(mid_deg)

def angle_to_segment(cx: int, cy: int, tx: int, ty: int) -> int:
    """
    Map a fingertip (tx, ty) to the segment index it falls in,
    relative to circle centre (cx, cy).
    Returns -1 if the tip is inside the inner dead-zone or outside the outer radius.
    """
    dx, dy = tx - cx, ty - cy
    dist = math.hypot(dx, dy)
    if dist < INNER_RADIUS or dist > RADIUS:
        return -1
    raw_deg  = math.degrees(math.atan2(dy, dx))          # –180..180, 0=right
    norm_deg = (raw_deg - START_OFFSET) % 360             # 0 = top, clockwise
    return int(norm_deg / SEG_ANGLE) % NUM_SEGMENTS

# ── PIL text helper ───────────────────────────────────────────────────────────

def _draw_text_pil(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    colour_bgr: tuple[int, int, int],
) -> np.ndarray:
    """Render anti-aliased text onto an OpenCV frame via PIL (centred on pos)."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    bbox    = draw.textbbox((0, 0), text, font=font)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    xy      = (pos[0] - tw // 2, pos[1] - th // 2)
    r, g, b = colour_bgr[2], colour_bgr[1], colour_bgr[0]
    draw.text(xy, text, font=font, fill=(r, g, b, 255))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ── Single circle renderer ────────────────────────────────────────────────────

def _draw_circle(
    frame:      np.ndarray,
    cx:         int,
    cy:         int,
    labels:     list[str],
    palette:    dict,
    hover_idx:  int,       # segment under fingertip (-1 = none)
    confirm_idx: int,      # currently selected segment (-1 = none)
    font:       ImageFont.FreeTypeFont,
) -> np.ndarray:
    overlay = frame.copy()

    for i, label in enumerate(labels):
        sa, ea = _seg_angles(i)

        if i == confirm_idx:
            colour = palette["confirm"]
            alpha  = ALPHA_CONFIRM
        elif i == hover_idx:
            colour = palette["hover"]
            alpha  = ALPHA_HOVER
        else:
            colour = palette["base"]
            alpha  = ALPHA_BASE

        # ── Filled pie slice (skip if fully transparent) ──────────────────
        if alpha > 0.0:
            seg_overlay = frame.copy()
            cv2.ellipse(
                seg_overlay, (cx, cy), (RADIUS, RADIUS),
                0, sa, ea, colour, thickness=-1, lineType=cv2.LINE_AA,
            )
            cv2.circle(seg_overlay, (cx, cy), INNER_RADIUS, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.addWeighted(seg_overlay, alpha, overlay, 1 - alpha, 0, overlay)

        # ── Segment divider lines ─────────────────────────────────────────
        border_rad = math.radians(sa)
        bx = int(cx + RADIUS * math.cos(border_rad))
        by = int(cy + RADIUS * math.sin(border_rad))
        ix = int(cx + INNER_RADIUS * math.cos(border_rad))
        iy = int(cy + INNER_RADIUS * math.sin(border_rad))
        cv2.line(overlay, (ix, iy), (bx, by), palette["border"], 1, cv2.LINE_AA)

    # ── Outer and inner circle rings ──────────────────────────────────────
    cv2.circle(overlay, (cx, cy), RADIUS,       palette["border"], 2, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), INNER_RADIUS, palette["border"], 2, cv2.LINE_AA)

    # ── Bright arc highlight on hovered segment ────────────────────────
    if hover_idx != -1:
        sa, ea = _seg_angles(hover_idx)
        # Thick bright arc on outer edge
        cv2.ellipse(overlay, (cx, cy), (RADIUS - 4, RADIUS - 4),
                    0, sa, ea, (255, 255, 255), 5, cv2.LINE_AA)
        # Thick arc on inner edge
        cv2.ellipse(overlay, (cx, cy), (INNER_RADIUS + 4, INNER_RADIUS + 4),
                    0, sa, ea, (255, 255, 255), 4, cv2.LINE_AA)

    # ── Labels via PIL (rendered after blending so they stay crisp) ───────
    mid_r = (RADIUS + INNER_RADIUS) // 2
    for i, label in enumerate(labels):
        ang = _seg_mid_angle_rad(i)
        lx  = int(cx + mid_r * math.cos(ang))
        ly  = int(cy + mid_r * math.sin(ang))

        text_col = (255, 255, 255)   # white — visible on transparent/dark bg

        overlay = _draw_text_pil(overlay, label, (lx, ly), font, text_col)

    return overlay

# ── Public API ────────────────────────────────────────────────────────────────

def draw_circles(
    frame:          np.ndarray,
    left_hover_idx:   int = -1,
    right_hover_idx:  int = -1,
    left_confirm_idx: int = -1,
    right_confirm_idx: int = -1,
) -> np.ndarray:
    """
    Draw both circles onto frame and return the annotated frame.

    Centres:
      Left  circle: (W//4,    H//2)
      Right circle: (3*W//4,  H//2)

    Args:
        frame:              BGR frame from OpenCV
        left_hover_idx:     segment under left-hand index tip (-1 = none)
        right_hover_idx:    segment under right-hand index tip (-1 = none)
        left_confirm_idx:   currently selected root segment (-1 = none)
        right_confirm_idx:  currently selected type segment (-1 = none)

    Returns:
        Annotated frame (same shape as input)
    """
    h, w = frame.shape[:2]
    lcx, lcy = w // 4,     h // 2
    rcx, rcy = 3 * w // 4, h // 2

    # Left circle — chord roots (A–G)
    frame = _draw_circle(
        frame, lcx, lcy, ROOT_LABELS, PALETTE["left"],
        left_hover_idx, left_confirm_idx, _FONT_LG,
    )

    # Right circle — chord types
    frame = _draw_circle(
        frame, rcx, rcy, TYPE_LABELS, PALETTE["right"],
        right_hover_idx, right_confirm_idx, _FONT_SM,
    )

    # ── Circle title + hand assignment labels ─────────────────────────────
    frame = _draw_text_pil(frame, "ROOT",      (lcx, lcy - RADIUS - 38), _FONT_SM, PALETTE["left"]["hover"])
    frame = _draw_text_pil(frame, "Left Hand", (lcx, lcy - RADIUS - 16), _FONT_SM, (200, 200, 200))
    frame = _draw_text_pil(frame, "TYPE",       (rcx, rcy - RADIUS - 38), _FONT_SM, PALETTE["right"]["hover"])
    frame = _draw_text_pil(frame, "Right Hand", (rcx, rcy - RADIUS - 16), _FONT_SM, (200, 200, 200))

    return frame

# ── Standalone mouse-driven test ──────────────────────────────────────────────

if __name__ == "__main__":
    W, H = 1280, 720
    blank = np.zeros((H, W, 3), dtype=np.uint8)

    left_confirm  = -1
    right_confirm = -1
    mouse_x, mouse_y = W // 2, H // 2

    def _on_mouse(event, x, y, flags, param):
        global mouse_x, mouse_y, left_confirm, right_confirm
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            lcx, lcy = W // 4,     H // 2
            rcx, rcy = 3 * W // 4, H // 2
            li = angle_to_segment(lcx, lcy, x, y)
            ri = angle_to_segment(rcx, rcy, x, y)
            if li != -1:
                left_confirm = li
            if ri != -1:
                right_confirm = ri

    cv2.namedWindow("Conductor — Chunk 2: Circle UI")
    cv2.setMouseCallback("Conductor — Chunk 2: Circle UI", _on_mouse)

    print("Circle UI test — hover to see highlights, click to confirm. Press Q to quit.")

    while True:
        frame = blank.copy()
        lcx, lcy = W // 4,     H // 2
        rcx, rcy = 3 * W // 4, H // 2

        lhover = angle_to_segment(lcx, lcy, mouse_x, mouse_y)
        rhover = angle_to_segment(rcx, rcy, mouse_x, mouse_y)

        frame = draw_circles(frame, lhover, rhover, left_confirm, right_confirm)

        # Show confirmed selection
        root = ROOT_LABELS[left_confirm]  if left_confirm  != -1 else "—"
        typ  = TYPE_LABELS[right_confirm] if right_confirm != -1 else "—"
        frame = _draw_text_pil(
            frame, f"{root}  {typ}",
            (W // 2, H // 2),
            _FONT_LG, (220, 220, 220),
        )

        cv2.imshow("Conductor — Chunk 2: Circle UI", frame)
        if cv2.waitKey(16) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
