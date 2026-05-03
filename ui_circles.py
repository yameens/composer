"""
Chunk 2 — Circle UI & Section Selection
Run this file directly to test: python ui_circles.py
Move your mouse over the circles to simulate fingertip hover.
Press Q to quit.
"""

import math
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Layout constants ──────────────────────────────────────────────────────────

RADIUS       = 190       # outer radius — sized to clear top-right / top-left UI at 1280px
INNER_RADIUS = 60        # hole radius — wider dead-zone for easier sweeps between segments
NUM_SEGMENTS = 7
SEG_ANGLE    = 360 / NUM_SEGMENTS   # ≈ 51.43°
START_OFFSET = -90       # top of circle = segment 0

# Translucency
ALPHA_BASE    = 0.42   # translucent black bg on all segments — makes labels readable
ALPHA_HOVER   = 0.60   # gold fill on hovered segment
ALPHA_CONFIRM = 0.75   # brighter gold on active/playing segment

# Segment label sets
ROOT_LABELS = ["A", "D", "G", "C", "F", "B", "E"]   # circle of fifths order
TYPE_LABELS = ["Maj", "Maj7", "7", "dim", "Min", "min7", "sus4"]

# Gold: BGR for RGB #FFD700
_GOLD         = (  0, 215, 255)
_GOLD_HOVER   = (  0, 180, 215)
_GOLD_CONFIRM = ( 20, 230, 255)

# Colour palette (BGR) — black & gold theme
PALETTE = {
    "left":  {
        "base":    (  0,   0,   0),   # black — translucent bg behind labels
        "hover":   _GOLD_HOVER,
        "confirm": _GOLD_CONFIRM,
        "border":  (255, 255, 255),
        "text":    (255, 255, 255),
    },
    "right": {
        "base":    (  0,   0,   0),
        "hover":   _GOLD_HOVER,
        "confirm": _GOLD_CONFIRM,
        "border":  (255, 255, 255),
        "text":    (255, 255, 255),
    },
}

def _resolve_accent_font_path() -> Optional[Path]:
    """Jacquard 24 — bottom wordmark + keyboard hints only."""
    assets_dir = Path(__file__).parent / "assets"
    for name in ("Jacquard24-Regular.ttf", "Jacquard-24-Regular.ttf"):
        p = assets_dir / name
        if p.exists():
            return p
    return None


def _resolve_body_font_path() -> Optional[Path]:
    """Georgia for main UI (circles, top HUD, buttons); then Calibri / Fira."""
    assets_dir = Path(__file__).parent / "assets"
    candidates: list[Path] = [
        assets_dir / "Georgia.ttf",
        assets_dir / "georgia.ttf",
    ]
    if sys.platform == "darwin":
        candidates.extend([
            Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
            Path("/Library/Fonts/Georgia.ttf"),
            Path("/Library/Fonts/Microsoft/Georgia.ttf"),
        ])
    elif sys.platform == "win32":
        wind = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates.extend([wind / "georgia.ttf", wind / "Georgia.ttf"])
    candidates.extend([
        assets_dir / "calibri.ttf",
        assets_dir / "Calibri.ttf",
    ])
    if sys.platform == "darwin":
        candidates.append(Path("/Library/Fonts/Microsoft/Calibri.ttf"))
    elif sys.platform == "win32":
        candidates.append(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "calibri.ttf")
    candidates.append(assets_dir / "FiraMono-Regular.ttf")
    for p in candidates:
        if p.exists():
            return p
    return None


_FONT_PATH_ACCENT = _resolve_accent_font_path()
_FONT_PATH_BODY = _resolve_body_font_path()


def _load_body_font(size: int) -> ImageFont.FreeTypeFont:
    if _FONT_PATH_BODY:
        try:
            return ImageFont.truetype(str(_FONT_PATH_BODY), size)
        except Exception:
            pass
    return ImageFont.load_default()


def _load_accent_font(size: int) -> ImageFont.FreeTypeFont:
    if _FONT_PATH_ACCENT:
        try:
            return ImageFont.truetype(str(_FONT_PATH_ACCENT), size)
        except Exception:
            pass
    return _load_body_font(size)


def app_font(size: int) -> ImageFont.FreeTypeFont:
    """Body / UI typeface (Georgia when available)."""
    return _load_body_font(size)


_FONT_LG    = _load_body_font(28)   # root letters, HUD chord readout
_FONT_SM    = _load_body_font(20)   # segment labels inside circles (right ring)
_FONT_MODE  = _load_body_font(15)   # small buttons
_FONT_XS    = _load_body_font(14)   # reserved / small UI
_FONT_BRAND = _load_accent_font(30)   # bottom-left wordmark (Jacquard)
_FONT_HUD_KEYS = _load_accent_font(24)   # footer Q/T/V/S row (Jacquard)
_FONT_CIRCLE_HEADINGS = _load_accent_font(32)   # “root” / “type” above circles

# Uppercase wordmark fallback (Jacquard); change if you regenerate PNG.
BRAND_TEXT           = "COMPOSER"
BRAND_LETTER_SPACING = 7

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
    bold: bool = False,
    underline: bool = False,
    anchor: str = "center",
    letter_spacing: int = 0,
    underline_gap: int = 2,
) -> np.ndarray:
    """Render anti-aliased text onto an OpenCV frame via PIL.

    ``anchor``: ``center`` — ``pos`` is the centre (default); ``frame_bl`` —
    ``pos`` is ``(margin_left, margin_bottom)`` insets from the frame bottom-left;
    ``tl`` — ``pos`` is the top-left corner of the tight text bbox.

    When ``bold=True``, stroke matching the fill colour thickens the glyphs.
    When ``underline=True``, a 2-px line is drawn just below the glyphs.
    ``letter_spacing``: extra pixels between glyphs (0 = default kerning only).
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    stroke  = 1 if bold else 0
    r, g, b = colour_bgr[2], colour_bgr[1], colour_bgr[0]
    fh      = frame.shape[0]

    if letter_spacing <= 0:
        bbox   = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if anchor == "frame_bl":
            xy = (pos[0], fh - pos[1] - th)
        elif anchor == "tl":
            xy = (pos[0] - bbox[0], pos[1] - bbox[1])
        else:
            xy = (pos[0] - tw // 2, pos[1] - th // 2)
        draw.text(
            xy, text, font=font, fill=(r, g, b, 255),
            stroke_width=stroke, stroke_fill=(r, g, b, 255),
        )
    else:
        th = 0
        advances: list[int] = []
        for ch in text:
            bb = draw.textbbox((0, 0), ch, font=font, stroke_width=stroke)
            advances.append(bb[2] - bb[0])
            th = max(th, bb[3] - bb[1])
        tw = sum(advances) + letter_spacing * max(0, len(text) - 1)
        if anchor == "frame_bl":
            x0, y0 = pos[0], fh - pos[1] - th
        elif anchor == "tl":
            bb0 = draw.textbbox((0, 0), text[0], font=font, stroke_width=stroke)
            x0 = pos[0] - bb0[0]
            y0 = pos[1] - bb0[1]
        else:
            x0, y0 = pos[0] - tw // 2, pos[1] - th // 2
        x_cur = x0
        for i, ch in enumerate(text):
            draw.text(
                (x_cur, y0), ch, font=font, fill=(r, g, b, 255),
                stroke_width=stroke, stroke_fill=(r, g, b, 255),
            )
            x_cur += advances[i] + (letter_spacing if i < len(text) - 1 else 0)
        xy = (x0, y0)

    if underline:
        line_y = xy[1] + th + underline_gap
        draw.line([(xy[0], line_y), (xy[0] + tw, line_y)],
                  fill=(r, g, b, 255), width=2)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


_BRAND_PNG = Path(__file__).parent / "assets" / "conductor_brand.png"


def draw_brand_wordmark(
    frame: np.ndarray,
    margin_x: int = 14,
    margin_bottom: int = 14,
    strip_h: int = 0,
) -> np.ndarray:
    """
    Bottom-left titling: use assets/conductor_brand.png if present (e.g. TeX export),
    else letter-spaced vector text with BRAND_TEXT / _FONT_BRAND (Jacquard accent).
    When strip_h > 0, the text is vertically centred inside that strip instead of
    using margin_bottom as a bottom inset.
    """
    fh = frame.shape[0]
    if strip_h > 0:
        # compute margin_bottom so text centre lands at strip centre
        _dummy_img = Image.new("RGB", (1, 1))
        _dummy_draw = ImageDraw.Draw(_dummy_img)
        _bbox = _dummy_draw.textbbox((0, 0), BRAND_TEXT, font=_FONT_BRAND, stroke_width=0)
        _th = _bbox[3] - _bbox[1]
        margin_bottom = strip_h // 2 - _th // 2
        margin_bottom = max(margin_bottom, 4)
    if _BRAND_PNG.exists():
        badge = cv2.imread(str(_BRAND_PNG), cv2.IMREAD_UNCHANGED)
        if badge is None:
            return _draw_text_pil(
                frame,
                BRAND_TEXT,
                (margin_x, margin_bottom),
                _FONT_BRAND,
                (255, 255, 255),
                bold=False,
                anchor="frame_bl",
                letter_spacing=BRAND_LETTER_SPACING,
            )
        fw = frame.shape[1]
        bh, bw = badge.shape[:2]
        y1 = fh - margin_bottom - bh
        x1 = margin_x
        if y1 >= 0 and x1 + bw <= fw and x1 >= 0:
            frame = frame.copy()
            roi = frame[y1 : y1 + bh, x1 : x1 + bw]
            if badge.ndim == 3 and badge.shape[2] == 4:
                a = badge[:, :, 3:4].astype(np.float32) / 255.0
                bgr = badge[:, :, :3]
                blended = (a * bgr + (1.0 - a) * roi).astype(np.uint8)
                frame[y1 : y1 + bh, x1 : x1 + bw] = blended
            elif badge.ndim == 3 and badge.shape[2] == 3:
                frame[y1 : y1 + bh, x1 : x1 + bw] = badge
            return frame
    return _draw_text_pil(
        frame,
        BRAND_TEXT,
        (margin_x, margin_bottom),
        _FONT_BRAND,
        (255, 255, 255),
        bold=False,
        anchor="frame_bl",
        letter_spacing=BRAND_LETTER_SPACING,
    )

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
            cv2.addWeighted(seg_overlay, alpha, overlay, 1 - alpha, 0, overlay)

        # ── Segment divider lines ─────────────────────────────────────────
        border_rad = math.radians(sa)
        bx = int(cx + RADIUS * math.cos(border_rad))
        by = int(cy + RADIUS * math.sin(border_rad))
        ix = int(cx + INNER_RADIUS * math.cos(border_rad))
        iy = int(cy + INNER_RADIUS * math.sin(border_rad))
        cv2.line(overlay, (ix, iy), (bx, by), palette["border"], 1, cv2.LINE_AA)

    # ── Restore camera pixels inside the inner circle (true transparency)──
    # Done before drawing the inner ring so the ring outline remains crisp.
    inner_mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
    cv2.circle(inner_mask, (cx, cy), INNER_RADIUS - 1, 255, -1, cv2.LINE_AA)
    overlay[inner_mask > 0] = frame[inner_mask > 0]

    # ── Outer and inner circle rings ──────────────────────────────────────
    cv2.circle(overlay, (cx, cy), RADIUS,       palette["border"], 2, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), INNER_RADIUS, palette["border"], 2, cv2.LINE_AA)

    # ── Gold arc highlight on hovered segment ─────────────────────────
    if hover_idx != -1:
        sa, ea = _seg_angles(hover_idx)
        cv2.ellipse(overlay, (cx, cy), (RADIUS - 4, RADIUS - 4),
                    0, sa, ea, _GOLD, 5, cv2.LINE_AA)
        cv2.ellipse(overlay, (cx, cy), (INNER_RADIUS + 4, INNER_RADIUS + 4),
                    0, sa, ea, _GOLD, 4, cv2.LINE_AA)

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

    # ── Circle titles (Jacquard, same family as footer key row) ─────────────
    title_y = -RADIUS - 26
    frame = _draw_text_pil(
        frame, "root", (lcx, lcy + title_y), _FONT_CIRCLE_HEADINGS, _GOLD,
        letter_spacing=3,
    )
    frame = _draw_text_pil(
        frame, "type", (rcx, rcy + title_y), _FONT_CIRCLE_HEADINGS, _GOLD,
        letter_spacing=3,
    )

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
