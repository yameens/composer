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


def _resolve_mono_font_path() -> Optional[Path]:
    """FiraMono — bundled monospace font that contains arrow glyphs."""
    p = Path(__file__).parent / "assets" / "FiraMono-Regular.ttf"
    return p if p.exists() else None


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
_FONT_PATH_BODY   = _resolve_body_font_path()
_FONT_PATH_MONO   = _resolve_mono_font_path()


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


def _load_mono_font(size: int) -> ImageFont.FreeTypeFont:
    """FiraMono — contains arrow glyphs missing from Georgia and Jacquard."""
    if _FONT_PATH_MONO:
        try:
            return ImageFont.truetype(str(_FONT_PATH_MONO), size)
        except Exception:
            pass
    return _load_body_font(size)


def app_font(size: int) -> ImageFont.FreeTypeFont:
    """Body / UI typeface (Georgia when available)."""
    return _load_body_font(size)


def load_accent_font(size: int) -> ImageFont.FreeTypeFont:
    """Public wrapper — load the Jacquard accent font at any size."""
    return _load_accent_font(size)


def load_mono_font(size: int) -> ImageFont.FreeTypeFont:
    """Public wrapper — load the FiraMono font at any size (contains arrow glyphs)."""
    return _load_mono_font(size)


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


# ── Sprite cache & alpha-composite helpers ────────────────────────────────────
# Static UI is rendered to BGRA sprites once (PIL roundtrip is paid up-front),
# then alpha-blitted onto the camera frame each tick. Per-frame text/circle
# rendering is replaced with cheap memory copies + per-pixel premultiplied
# alpha math, so the main loop stops paying ~30 full-frame BGR↔RGB conversions
# and ~14 full-frame copies that previously bottlenecked drawing.

_CIRCLE_PAD = 6                                         # AA padding around each circle bbox
_CIRCLE_SPRITE_SIZE = 2 * RADIUS + 2 * _CIRCLE_PAD


def _premul_color(bgr: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    """Return a 4-tuple BGRA in *premultiplied* form for cv2 drawing."""
    a8 = max(0, min(255, int(round(alpha * 255))))
    return (
        int(bgr[0]) * a8 // 255,
        int(bgr[1]) * a8 // 255,
        int(bgr[2]) * a8 // 255,
        a8,
    )


def _to_premul(bgra: np.ndarray) -> np.ndarray:
    """Convert a non-premultiplied BGRA sprite to premultiplied form, in place."""
    a = bgra[..., 3:4].astype(np.uint16)
    bgra[..., :3] = (bgra[..., :3].astype(np.uint16) * a // 255).astype(np.uint8)
    return bgra


def _blit_bgra(frame: np.ndarray, sprite: Optional[np.ndarray], x: int, y: int) -> None:
    """Alpha-composite a *premultiplied* BGRA sprite onto a BGR frame in place.

    (x, y) is the top-left of the sprite in frame coordinates. Pixels falling
    outside the frame are clipped. No-op if sprite is None or fully transparent.
    """
    if sprite is None:
        return
    sh, sw = sprite.shape[:2]
    fh, fw = frame.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2 = min(x + sw, fw)
    y2 = min(y + sh, fh)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    sub_sprite = sprite[sy1:sy2, sx1:sx2]
    sub_frame = frame[y1:y2, x1:x2]
    a = sub_sprite[..., 3]
    if a.max() == 0:
        return
    a16 = a.astype(np.uint16)[..., None]
    inv = (np.uint16(255) - a16)
    bgr_pre = sub_sprite[..., :3].astype(np.uint16)
    sub_frame[:] = (bgr_pre + sub_frame.astype(np.uint16) * inv // 255).astype(np.uint8)


def _make_text_sprite(
    text: str,
    font: ImageFont.FreeTypeFont,
    color_bgr: tuple[int, int, int],
    *,
    bold: bool = False,
    underline: bool = False,
    letter_spacing: int = 0,
    underline_gap: int = 2,
) -> tuple[np.ndarray, int, int, int]:
    """Render text into a tightly-cropped premultiplied BGRA sprite.

    Returns ``(sprite, tw, th, pad)`` — text origin within the sprite is at
    ``(pad, pad)`` after bbox correction. Use :func:`_blit_text_centered` /
    :func:`_blit_text_frame_bl` to position the result on a frame.
    """
    stroke = 1 if bold else 0
    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]

    if letter_spacing <= 0:
        dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=font, stroke_width=stroke)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 4
        ext_h = (underline_gap + 4) if underline else 0
        sw = max(1, tw + 2 * pad)
        sh = max(1, th + 2 * pad + ext_h)
        img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        x_text = pad - bbox[0]
        y_text = pad - bbox[1]
        draw.text(
            (x_text, y_text), text, font=font, fill=(r, g, b, 255),
            stroke_width=stroke, stroke_fill=(r, g, b, 255),
        )
        if underline:
            ly = pad + th + underline_gap
            draw.line([(pad, ly), (pad + tw, ly)], fill=(r, g, b, 255), width=2)
        rgba = np.array(img)
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        return _to_premul(bgra), tw, th, pad

    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    th = 0
    advances: list[int] = []
    bb_lefts: list[int] = []
    bb_tops: list[int] = []
    for ch in text:
        bb = d.textbbox((0, 0), ch, font=font, stroke_width=stroke)
        advances.append(bb[2] - bb[0])
        bb_lefts.append(bb[0])
        bb_tops.append(bb[1])
        th = max(th, bb[3] - bb[1])
    tw = sum(advances) + letter_spacing * max(0, len(text) - 1)
    pad = 4
    sw = max(1, tw + 2 * pad)
    sh = max(1, th + 2 * pad)
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x_cur = pad
    for i, ch in enumerate(text):
        draw.text(
            (x_cur - bb_lefts[i], pad - bb_tops[i]), ch, font=font,
            fill=(r, g, b, 255), stroke_width=stroke, stroke_fill=(r, g, b, 255),
        )
        x_cur += advances[i] + (letter_spacing if i < len(text) - 1 else 0)
    rgba = np.array(img)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return _to_premul(bgra), tw, th, pad


def _blit_text_centered(
    frame: np.ndarray,
    ts: tuple[np.ndarray, int, int, int],
    cx: int,
    cy: int,
) -> None:
    sprite, tw, th, pad = ts
    _blit_bgra(frame, sprite, cx - tw // 2 - pad, cy - th // 2 - pad)


def _blit_text_frame_bl(
    frame: np.ndarray,
    ts: tuple[np.ndarray, int, int, int],
    margin_x: int,
    margin_bottom: int,
) -> None:
    sprite, _tw, th, pad = ts
    fh = frame.shape[0]
    _blit_bgra(frame, sprite, margin_x - pad, fh - margin_bottom - th - pad)


def _blit_text_tl(
    frame: np.ndarray,
    ts: tuple[np.ndarray, int, int, int],
    x: int,
    y: int,
) -> None:
    sprite, _tw, _th, pad = ts
    _blit_bgra(frame, sprite, x - pad, y - pad)


# ── Mixed-font hint renderer ──────────────────────────────────────────────────

_ARROW_CHARS = set("↑↓←→▲▼◄►")


def make_hint_sprite(
    text: str,
    base_font: ImageFont.FreeTypeFont,
    color_bgr: tuple[int, int, int],
    arrow_font: ImageFont.FreeTypeFont,
) -> tuple[np.ndarray, int, int, int]:
    """Render a hint string into a premultiplied BGRA sprite, using arrow_font
    for chars in _ARROW_CHARS and base_font for all other chars.

    Returns the same ``(sprite, tw, th, pad)`` tuple as ``_make_text_sprite``,
    compatible with ``_blit_text_centered`` / ``_blit_text_frame_bl``.
    """
    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
    pad = 4

    # Split text into consecutive runs keyed by (is_arrow).
    if not text:
        dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        bgra = cv2.cvtColor(np.array(dummy), cv2.COLOR_RGBA2BGRA)
        return _to_premul(bgra), 0, 0, pad

    runs: list[tuple[str, bool]] = []
    cur_is_arrow = text[0] in _ARROW_CHARS
    buf = text[0]
    for ch in text[1:]:
        is_arr = ch in _ARROW_CHARS
        if is_arr == cur_is_arrow:
            buf += ch
        else:
            runs.append((buf, cur_is_arrow))
            buf = ch
            cur_is_arrow = is_arr
    runs.append((buf, cur_is_arrow))

    # Measure runs on a dummy canvas to compute total width and baseline.
    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    run_widths: list[float] = []
    max_ascent  = 0
    max_descent = 0
    for run_text, is_arrow in runs:
        font = arrow_font if is_arrow else base_font
        w_run = d.textlength(run_text, font=font)
        run_widths.append(w_run)
        ascent, descent = font.getmetrics()
        if ascent  > max_ascent:
            max_ascent  = ascent
        if descent > max_descent:
            max_descent = descent

    tw = int(round(sum(run_widths)))
    th = max_ascent + max_descent   # consistent with textbbox height convention

    sw = max(1, tw + 2 * pad)
    sh = max(1, th + 2 * pad)
    img  = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    baseline_y = pad + max_ascent   # y coordinate of the shared baseline
    x_cur = float(pad)
    for (run_text, is_arrow), rw in zip(runs, run_widths):
        font = arrow_font if is_arrow else base_font
        draw.text((x_cur, baseline_y), run_text, font=font,
                  fill=(r, g, b, 255), anchor="ls")
        x_cur += rw

    rgba = np.array(img)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return _to_premul(bgra), tw, th, pad


# ── Per-circle sprite cache ───────────────────────────────────────────────────
# Three layers per circle, blitted in order:
#   1. fills_sprite   — 7 translucent segment fills + AA inner cutout
#   2. decor_sprite   — dividers + outer/inner rings + segment labels
#   3. arcs_sprite    — gold hover arcs (only when hover_idx != -1)
# Layers 1 and 3 depend on (hover_idx, confirm_idx); layer 2 is fully static.

_CIRCLE_DECOR_LEFT: Optional[np.ndarray]  = None
_CIRCLE_DECOR_RIGHT: Optional[np.ndarray] = None
_FILLS_CACHE: dict[str, tuple[Optional[tuple[int, int]], Optional[np.ndarray]]] = {
    "left":  (None, None),
    "right": (None, None),
}
_ARCS_CACHE: dict[str, tuple[Optional[int], Optional[np.ndarray]]] = {
    "left":  (None, None),
    "right": (None, None),
}


def _build_circle_decorations(
    labels: list[str],
    font: ImageFont.FreeTypeFont,
    palette: dict,
) -> np.ndarray:
    """Static layer: dividers, outer/inner rings, labels (alpha=255 ink only)."""
    pad = _CIRCLE_PAD
    size = _CIRCLE_SPRITE_SIZE
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = cy = RADIUS + pad
    border = palette["border"]
    border_rgba = (border[2], border[1], border[0], 255)

    for i in range(NUM_SEGMENTS):
        sa, _ea = _seg_angles(i)
        theta = math.radians(sa)
        bx = cx + RADIUS * math.cos(theta)
        by = cy + RADIUS * math.sin(theta)
        ix = cx + INNER_RADIUS * math.cos(theta)
        iy = cy + INNER_RADIUS * math.sin(theta)
        draw.line([(ix, iy), (bx, by)], fill=border_rgba, width=1)

    draw.ellipse(
        [(cx - RADIUS, cy - RADIUS), (cx + RADIUS, cy + RADIUS)],
        outline=border_rgba, width=2,
    )
    draw.ellipse(
        [(cx - INNER_RADIUS, cy - INNER_RADIUS), (cx + INNER_RADIUS, cy + INNER_RADIUS)],
        outline=border_rgba, width=2,
    )

    mid_r = (RADIUS + INNER_RADIUS) // 2
    text_rgba = (255, 255, 255, 255)
    for i, label in enumerate(labels):
        ang = _seg_mid_angle_rad(i)
        lx = cx + mid_r * math.cos(ang)
        ly = cy + mid_r * math.sin(ang)
        bbox = draw.textbbox((0, 0), label, font=font, stroke_width=0)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = lx - tw / 2 - bbox[0]
        y = ly - th / 2 - bbox[1]
        draw.text((x, y), label, font=font, fill=text_rgba)

    rgba = np.array(img)
    return _to_premul(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def _get_circle_decorations(side: str) -> np.ndarray:
    global _CIRCLE_DECOR_LEFT, _CIRCLE_DECOR_RIGHT
    if side == "left":
        if _CIRCLE_DECOR_LEFT is None:
            _CIRCLE_DECOR_LEFT = _build_circle_decorations(
                ROOT_LABELS, _FONT_LG, PALETTE["left"],
            )
        return _CIRCLE_DECOR_LEFT
    if _CIRCLE_DECOR_RIGHT is None:
        _CIRCLE_DECOR_RIGHT = _build_circle_decorations(
            TYPE_LABELS, _FONT_SM, PALETTE["right"],
        )
    return _CIRCLE_DECOR_RIGHT


def _build_fills_sprite(
    palette: dict,
    hover_idx: int,
    confirm_idx: int,
) -> np.ndarray:
    """Translucent segment fills (premultiplied) + AA inner cutout."""
    pad = _CIRCLE_PAD
    size = _CIRCLE_SPRITE_SIZE
    sprite = np.zeros((size, size, 4), dtype=np.uint8)
    cx = cy = RADIUS + pad

    for i in range(NUM_SEGMENTS):
        sa, ea = _seg_angles(i)
        if i == confirm_idx:
            color, alpha = palette["confirm"], ALPHA_CONFIRM
        elif i == hover_idx:
            color, alpha = palette["hover"], ALPHA_HOVER
        else:
            color, alpha = palette["base"], ALPHA_BASE
        c4 = _premul_color(color, alpha)
        cv2.ellipse(
            sprite, (cx, cy), (RADIUS, RADIUS), 0, sa, ea,
            c4, thickness=-1, lineType=cv2.LINE_AA,
        )

    cv2.circle(sprite, (cx, cy), INNER_RADIUS - 1, (0, 0, 0, 0), -1, cv2.LINE_AA)
    return sprite


def _get_fills_sprite(
    side: str,
    palette: dict,
    hover_idx: int,
    confirm_idx: int,
) -> np.ndarray:
    cached_key, cached_sprite = _FILLS_CACHE[side]
    key = (hover_idx, confirm_idx)
    if cached_key != key or cached_sprite is None:
        sprite = _build_fills_sprite(palette, hover_idx, confirm_idx)
        _FILLS_CACHE[side] = (key, sprite)
        return sprite
    return cached_sprite


def _build_arcs_sprite(hover_idx: int) -> np.ndarray:
    pad = _CIRCLE_PAD
    size = _CIRCLE_SPRITE_SIZE
    sprite = np.zeros((size, size, 4), dtype=np.uint8)
    if hover_idx == -1:
        return sprite
    cx = cy = RADIUS + pad
    sa, ea = _seg_angles(hover_idx)
    gold_c4 = (_GOLD[0], _GOLD[1], _GOLD[2], 255)
    cv2.ellipse(
        sprite, (cx, cy), (RADIUS - 4, RADIUS - 4), 0, sa, ea,
        gold_c4, 5, cv2.LINE_AA,
    )
    cv2.ellipse(
        sprite, (cx, cy), (INNER_RADIUS + 4, INNER_RADIUS + 4), 0, sa, ea,
        gold_c4, 4, cv2.LINE_AA,
    )
    return sprite


def _get_arcs_sprite(side: str, hover_idx: int) -> np.ndarray:
    cached_key, cached_sprite = _ARCS_CACHE[side]
    if cached_key != hover_idx or cached_sprite is None:
        sprite = _build_arcs_sprite(hover_idx)
        _ARCS_CACHE[side] = (hover_idx, sprite)
        return sprite
    return cached_sprite


# ── Static text sprites (lazy) ────────────────────────────────────────────────

_TITLE_ROOT_TS: Optional[tuple[np.ndarray, int, int, int]] = None
_TITLE_TYPE_TS: Optional[tuple[np.ndarray, int, int, int]] = None
_BRAND_TEXT_TS: Optional[tuple[np.ndarray, int, int, int]] = None
_BRAND_BADGE_BGRA: Optional[np.ndarray] = None


def _get_title_root_ts() -> tuple[np.ndarray, int, int, int]:
    global _TITLE_ROOT_TS
    if _TITLE_ROOT_TS is None:
        _TITLE_ROOT_TS = _make_text_sprite(
            "root", _FONT_CIRCLE_HEADINGS, _GOLD, letter_spacing=3,
        )
    return _TITLE_ROOT_TS


def _get_title_type_ts() -> tuple[np.ndarray, int, int, int]:
    global _TITLE_TYPE_TS
    if _TITLE_TYPE_TS is None:
        _TITLE_TYPE_TS = _make_text_sprite(
            "type", _FONT_CIRCLE_HEADINGS, _GOLD, letter_spacing=3,
        )
    return _TITLE_TYPE_TS


def _get_brand_text_ts() -> tuple[np.ndarray, int, int, int]:
    global _BRAND_TEXT_TS
    if _BRAND_TEXT_TS is None:
        _BRAND_TEXT_TS = _make_text_sprite(
            BRAND_TEXT, _FONT_BRAND, (255, 255, 255),
            letter_spacing=BRAND_LETTER_SPACING,
        )
    return _BRAND_TEXT_TS


def _get_brand_badge() -> Optional[np.ndarray]:
    """Load assets/conductor_brand.png once as a premultiplied BGRA sprite."""
    global _BRAND_BADGE_BGRA
    if _BRAND_BADGE_BGRA is not None:
        return _BRAND_BADGE_BGRA
    if not _BRAND_PNG.exists():
        return None
    badge = cv2.imread(str(_BRAND_PNG), cv2.IMREAD_UNCHANGED)
    if badge is None:
        return None
    if badge.ndim == 3 and badge.shape[2] == 3:
        bh, bw = badge.shape[:2]
        alpha_ch = np.full((bh, bw, 1), 255, dtype=np.uint8)
        badge = np.concatenate([badge, alpha_ch], axis=2)
    elif badge.ndim != 3 or badge.shape[2] != 4:
        return None
    _BRAND_BADGE_BGRA = _to_premul(badge.copy())
    return _BRAND_BADGE_BGRA


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
    badge = _get_brand_badge()
    if badge is not None:
        bh, _bw = badge.shape[:2]
        if strip_h > 0:
            margin_bottom = max(strip_h // 2 - bh // 2, 4)
        _blit_bgra(frame, badge, margin_x, fh - margin_bottom - bh)
        return frame

    ts = _get_brand_text_ts()
    _sprite, _tw, th, _pad = ts
    if strip_h > 0:
        margin_bottom = max(strip_h // 2 - th // 2, 4)
    _blit_text_frame_bl(frame, ts, margin_x, margin_bottom)
    return frame

# ── Single circle renderer ────────────────────────────────────────────────────

def _draw_circle(
    frame:       np.ndarray,
    cx:          int,
    cy:          int,
    side:        str,            # "left" | "right" — picks the cached decoration sprite
    palette:     dict,
    hover_idx:   int,             # segment under fingertip (-1 = none)
    confirm_idx: int,              # currently selected segment (-1 = none)
) -> np.ndarray:
    """Blit the three cached layers onto frame in place: fills → decor → arcs."""
    pad = _CIRCLE_PAD
    bx = cx - RADIUS - pad
    by = cy - RADIUS - pad
    fills = _get_fills_sprite(side, palette, hover_idx, confirm_idx)
    decor = _get_circle_decorations(side)
    arcs  = _get_arcs_sprite(side, hover_idx)
    _blit_bgra(frame, fills, bx, by)
    _blit_bgra(frame, decor, bx, by)
    _blit_bgra(frame, arcs,  bx, by)
    return frame

# ── Runtime circle-size adjustment ───────────────────────────────────────────

def set_circle_size(radius: int) -> None:
    """Update RADIUS / INNER_RADIUS at runtime and invalidate sprite caches.

    Call this whenever the user changes the circle-size slider.  The sprite
    caches (decor, fills, arcs) are invalidated so they rebuild lazily at the
    new size on the next draw_circles() call.  size changes happen at most once
    per settings-apply, so the rebuild cost is irrelevant.
    """
    global RADIUS, INNER_RADIUS, _CIRCLE_SPRITE_SIZE
    global _CIRCLE_DECOR_LEFT, _CIRCLE_DECOR_RIGHT
    RADIUS       = int(radius)
    INNER_RADIUS = round(radius * 60 / 190)
    _CIRCLE_SPRITE_SIZE = 2 * RADIUS + 2 * _CIRCLE_PAD
    # Invalidate cached sprites so they rebuild at the new size
    _CIRCLE_DECOR_LEFT  = None
    _CIRCLE_DECOR_RIGHT = None
    _FILLS_CACHE["left"]  = (None, None)
    _FILLS_CACHE["right"] = (None, None)
    _ARCS_CACHE["left"]   = (None, None)
    _ARCS_CACHE["right"]  = (None, None)
    # Also reset the cached title sprites (centred above circles — position depends
    # on radius if the caller recomputes title_y from RADIUS)
    global _TITLE_ROOT_TS, _TITLE_TYPE_TS
    _TITLE_ROOT_TS = None
    _TITLE_TYPE_TS = None


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

    _draw_circle(frame, lcx, lcy, "left",  PALETTE["left"],
                 left_hover_idx,  left_confirm_idx)
    _draw_circle(frame, rcx, rcy, "right", PALETTE["right"],
                 right_hover_idx, right_confirm_idx)

    title_y = -RADIUS - 26
    _blit_text_centered(frame, _get_title_root_ts(), lcx, lcy + title_y)
    _blit_text_centered(frame, _get_title_type_ts(), rcx, rcy + title_y)

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
