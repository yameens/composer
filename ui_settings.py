"""
Settings overlay for Conductor — gear icon + three-row settings table.

Public API consumed by main.py:

    NAMED_COLORS           — list[(name, bgr)] in display order
    SLIDER_MIN_RADIUS      — int
    SLIDER_MAX_RADIUS      — int

    draw_settings_gear(frame, hovered) -> frame
    get_hovered_settings_gear(pt, w, h) -> bool    (mirror of beat button)

    draw_settings_overlay(frame, draft, mouse_xy) -> frame
    settings_hit(x, y) -> tuple | None
    settings_slider_value_at(x, which) -> value    which: "radius"|"volume"
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ui_circles import (
    _make_text_sprite,
    _blit_text_centered,
    _blit_bgra,
    _to_premul,
    load_accent_font,
    load_mono_font,
)
from ui_buttons import BTN_MARGIN, BTN_RADIUS, FIRST_ROW_CY

# ── Tab names ────────────────────────────────────────────────────────────────

TABS = ["controls"]   # single tab — header kept for style

# ── Named colors (name, BGR) ──────────────────────────────────────────────────

NAMED_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("shadow",  (  0,   0,   0)),   # black
    ("ruby",    (  0,   0, 200)),   # deep red
    ("lapis",   (200,  40,   0)),   # deep blue
    ("dove",    (255, 255, 255)),   # white
    ("gold",    (  0, 215, 255)),   # gold / amber
    ("blade",   (192, 192, 192)),   # silver
    ("garland", ( 30,  90, 140)),   # teal-green
    ("oak",     ( 40, 140,  40)),   # mid green
]

_COLOR_MAP: dict[str, tuple[int, int, int]] = {n: bgr for n, bgr in NAMED_COLORS}


def color_bgr(name: str) -> tuple[int, int, int]:
    """Return the BGR tuple for a named color, falling back to black."""
    return _COLOR_MAP.get(name, (0, 0, 0))


def color_index(name: str) -> int:
    """Return the index of a color name in NAMED_COLORS, or 0."""
    for i, (n, _) in enumerate(NAMED_COLORS):
        if n == name:
            return i
    return 0

# ── Slider range ──────────────────────────────────────────────────────────────

SLIDER_MIN_RADIUS = 175
SLIDER_MAX_RADIUS = 250

# ── Fonts ─────────────────────────────────────────────────────────────────────

_ASSETS = Path(__file__).parent / "assets"

_FONT_JACQUARD  = load_accent_font(42)     # title "settings"
_FONT_JACQ_LBL  = load_accent_font(28)     # row labels
_FONT_JACQ_BTN  = load_accent_font(26)     # apply / cancel pills
_FONT_JACQ_ARR  = load_accent_font(30)     # arrows ◀ ▶ (Jacquard, lacks glyphs)
_FONT_MONO_ARR  = load_mono_font(30)       # FiraMono at same size — has ◄► glyphs


def _load_times_new_roman(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


_FONT_TIMES = _load_times_new_roman(22)

# ── Card geometry (centred, designed for 1280×720) ────────────────────────────

_CARD_W  = 800
_CARD_H  = 470
_CARD_BG = (20, 20, 20)      # very dark — matches synth editor spirit
_CARD_BORDER = (80, 80, 80)

# Row centres inside the card (relative to card top-left, not frame)
_TITLE_DY  = 54    # y inside card for title
_TABS_DY   = 110   # y inside card for tab strip
_BTN_DY    = _CARD_H - 50   # apply / cancel pills (divider sits 22px above)

# Body region (between tab strip and the buttons divider).  Rows for the active
# tab are spread evenly inside this band so each tab's controls stay vertically
# centred regardless of how many rows it has.
_BODY_TOP_DY = _TABS_DY + 36
_BODY_BOT_DY = _BTN_DY - 34

_LABEL_RX  = 220   # right edge of left-column label (inside card)
_CTRL_CX   = 540   # centre of right-column control (inside card)

# Slider bar (inside card coords)
_SL_X1 = 340
_SL_X2 = 680
_SL_H  = 8

# Arrow hit targets (inside card)
_ARR_W = 40    # half-width of arrow hit zone
_ARR_H = 22    # half-height

# Pill buttons
_PIL_W  = 120
_PIL_H  = 34

# ── Sprite caches ─────────────────────────────────────────────────────────────

_TITLE_TS: Optional[tuple] = None
_LABEL_TS: dict[str, tuple] = {}        # label text → sprite
_COLOR_NAME_TS: dict[tuple, tuple] = {} # (name, bgr) → sprite
_ARROW_L_TS: Optional[tuple] = None
_ARROW_R_TS: Optional[tuple] = None
_APPLY_TS:   Optional[tuple] = None
_CANCEL_TS:  Optional[tuple] = None
_TAB_ACTIVE_TS:  dict[str, tuple] = {}  # tab name → gold sprite


def _get_title_ts() -> tuple:
    global _TITLE_TS
    if _TITLE_TS is None:
        _TITLE_TS = _make_text_sprite("settings", _FONT_JACQUARD, (220, 220, 220))
    return _TITLE_TS


def _get_label_ts(text: str) -> tuple:
    ts = _LABEL_TS.get(text)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_JACQ_LBL, (180, 180, 180))
        _LABEL_TS[text] = ts
    return ts


def _make_color_name_sprite(name: str, bgr: tuple[int, int, int]) -> tuple:
    """Times New Roman color name tinted to its BGR, with opposite-luminance outline
    so 'shadow' (black text) stays legible on a dark card."""
    lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]   # weighted BGR lum
    outline_lum = 255 if lum < 128 else 0
    outline = (outline_lum, outline_lum, outline_lum)
    r, g, b = bgr[2], bgr[1], bgr[0]   # BGR→RGB
    or_, og, ob = outline[2], outline[1], outline[0]

    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d     = ImageDraw.Draw(dummy)
    bbox  = d.textbbox((0, 0), name, font=_FONT_TIMES)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad   = 6
    sw    = tw + 2 * pad
    sh    = th + 2 * pad

    img  = Image.new("RGBA", (max(1, sw), max(1, sh)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x0   = pad - bbox[0]
    y0   = pad - bbox[1]
    # Draw outline by offsetting ±1 px in 8 directions
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x0 + dx, y0 + dy), name, font=_FONT_TIMES,
                      fill=(or_, og, ob, 200))
    # Draw fill
    draw.text((x0, y0), name, font=_FONT_TIMES, fill=(r, g, b, 255))

    rgba = np.array(img)
    bgra_arr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return _to_premul(bgra_arr), tw, th, pad


def _get_color_name_ts(name: str, bgr: tuple[int, int, int]) -> tuple:
    key = (name, bgr)
    ts  = _COLOR_NAME_TS.get(key)
    if ts is None:
        ts = _make_color_name_sprite(name, bgr)
        _COLOR_NAME_TS[key] = ts
    return ts


def _get_arrow_ts(direction: str) -> tuple:
    global _ARROW_L_TS, _ARROW_R_TS
    if direction == "L":
        if _ARROW_L_TS is None:
            _ARROW_L_TS = _make_text_sprite("◄", _FONT_MONO_ARR, (200, 200, 200))
        return _ARROW_L_TS
    else:
        if _ARROW_R_TS is None:
            _ARROW_R_TS = _make_text_sprite("►", _FONT_MONO_ARR, (200, 200, 200))
        return _ARROW_R_TS


def _get_apply_ts() -> tuple:
    global _APPLY_TS
    if _APPLY_TS is None:
        _APPLY_TS = _make_text_sprite("apply", _FONT_JACQ_BTN, (0, 0, 0))
    return _APPLY_TS


def _get_cancel_ts() -> tuple:
    global _CANCEL_TS
    if _CANCEL_TS is None:
        _CANCEL_TS = _make_text_sprite("cancel", _FONT_JACQ_BTN, (200, 200, 200))
    return _CANCEL_TS


def _get_tab_active_ts(text: str) -> tuple:
    """Gold sprite for the active tab label."""
    ts = _TAB_ACTIVE_TS.get(text)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_JACQ_LBL, (0, 215, 255))
        _TAB_ACTIVE_TS[text] = ts
    return ts


def prewarm_settings_sprites() -> None:
    """Pre-render all static settings sprites at startup."""
    _get_title_ts()
    _get_label_ts("tracker color")
    _get_label_ts("circle size")
    _get_label_ts("volume")
    _get_label_ts("resolution")
    _get_arrow_ts("L")
    _get_arrow_ts("R")
    _get_apply_ts()
    _get_cancel_ts()
    for name, bgr in NAMED_COLORS:
        _get_color_name_ts(name, bgr)
    for tab in TABS:
        _get_label_ts(tab)
        _get_tab_active_ts(tab)


# ── Card coordinate helpers ────────────────────────────────────────────────────

def _card_origin(frame_w: int, frame_h: int) -> tuple[int, int]:
    """Top-left of the settings card in frame coords."""
    return (frame_w - _CARD_W) // 2, (frame_h - _CARD_H) // 2


def _card_row_frame(frame_w: int, frame_h: int, row_dy: int) -> tuple[int, int]:
    """Frame coords of the centre of a card row."""
    ox, oy = _card_origin(frame_w, frame_h)
    return ox + _CARD_W // 2, oy + row_dy


def _tab_centers(card_cx: int) -> list[int]:
    """X-centres for the tab headers — a lone tab is centred, two are spread."""
    if len(TABS) <= 1:
        return [card_cx]
    return [card_cx - _CARD_W // 4, card_cx + _CARD_W // 4]


def _tab_n_rows(active_tab: int) -> int:
    """Control rows on the controls tab: tracker color, circle size, volume, resolution."""
    return 4


def _row_dy(active_tab: int, idx: int) -> int:
    """Y (inside card) of row `idx`, spread to vertically centre the tab's rows
    inside the body band between the tab strip and the buttons divider."""
    n    = _tab_n_rows(active_tab)
    slot = (_BODY_BOT_DY - _BODY_TOP_DY) / n
    return int(_BODY_TOP_DY + slot * (idx + 0.5))


# ── Hit-testing ───────────────────────────────────────────────────────────────

def settings_hit(x: int, y: int, frame_w: int = 1280, frame_h: int = 720,
                 active_tab: int = 0):
    """
    Return which element (x, y) hits, or None.

      ("tab",    0)                 — tab strip click (single "controls" tab)
      ("arrow",  "color", -1|+1)   — left / right color-cycle arrow
      ("slider", "radius")          — circle-size slider bar area
      ("slider", "volume")          — volume slider bar area
      ("slider", "resolution")      — resolution slider bar area
      ("apply",)
      ("cancel",)
      None
    """
    ox, oy = _card_origin(frame_w, frame_h)

    # ── Tab strip — always tested first ───────────────────────────────────────
    tab_y   = oy + _TABS_DY
    tab_hit_h = 20   # generous vertical half-height
    card_cx = ox + _CARD_W // 2
    tab_hw  = _CARD_W // 5   # horizontal half-width hit zone
    if abs(y - tab_y) <= tab_hit_h:
        for i, tcx in enumerate(_tab_centers(card_cx)):
            if abs(x - tcx) <= tab_hw:
                return ("tab", i)

    # ── Controls rows: tracker color / circle size / volume / resolution ───────
    sl_margin = 18
    sl_x1f = ox + _SL_X1
    sl_x2f = ox + _SL_X2
    cx_ctrl = ox + _CTRL_CX

    row0_fy = oy + _row_dy(0, 0)   # tracker color
    row1_fy = oy + _row_dy(0, 1)   # circle size
    row2_fy = oy + _row_dy(0, 2)   # volume
    row3_fy = oy + _row_dy(0, 3)   # resolution

    # Tracker-color arrows
    lax = cx_ctrl - 100
    if abs(x - lax) <= _ARR_W and abs(y - row0_fy) <= _ARR_H:
        return ("arrow", "color", -1)
    rax = cx_ctrl + 100
    if abs(x - rax) <= _ARR_W and abs(y - row0_fy) <= _ARR_H:
        return ("arrow", "color", +1)

    if sl_x1f <= x <= sl_x2f and abs(y - row1_fy) <= sl_margin:
        return ("slider", "radius")
    if sl_x1f <= x <= sl_x2f and abs(y - row2_fy) <= sl_margin:
        return ("slider", "volume")
    if sl_x1f <= x <= sl_x2f and abs(y - row3_fy) <= sl_margin:
        return ("slider", "resolution")

    # Pills — Cancel LEFT, Apply RIGHT
    # Vertically centered between div2 and card bottom edge
    div2_y = oy + _BTN_DY - 22
    btn_fy = (div2_y + (oy + _CARD_H)) // 2
    # Cancel — left pill
    cx1 = ox + _CARD_W // 2 - _PIL_W - 20
    cx2 = cx1 + _PIL_W
    if cx1 <= x <= cx2 and abs(y - btn_fy) <= _PIL_H // 2 + 4:
        return ("cancel",)
    # Apply — right pill
    ax1 = ox + _CARD_W // 2 + 20
    ax2 = ax1 + _PIL_W
    if ax1 <= x <= ax2 and abs(y - btn_fy) <= _PIL_H // 2 + 4:
        return ("apply",)

    return None


def settings_slider_value_at(x: int, which: str, frame_w: int = 1280, frame_h: int = 720) -> float:
    """Map cursor x → radius (int), volume (float), or resolution (float) based on slider geometry."""
    ox, _oy = _card_origin(frame_w, frame_h)
    x1f = ox + _SL_X1
    x2f = ox + _SL_X2
    bar = max(1, x2f - x1f)
    norm = max(0.0, min(1.0, (x - x1f) / bar))
    if which == "radius":
        return int(round(SLIDER_MIN_RADIUS + norm * (SLIDER_MAX_RADIUS - SLIDER_MIN_RADIUS)))
    elif which == "resolution":
        return round(norm, 3)
    else:  # volume
        return round(norm, 3)


# ── Gear drawing (also used by ui_buttons; re-exported from here for main.py) ─

def draw_settings_gear(
    frame: np.ndarray,
    hovered: bool,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> np.ndarray:
    """Draw a black gear icon in the top-right corner (mirror of beat button).

    Gear is drawn with cv2 directly (no PIL round-trip needed per frame).
    Alpha states match draw_beat_button style.
    """
    h_fr, w_fr = frame.shape[:2]
    cx = w_fr - BTN_MARGIN
    cy = FIRST_ROW_CY

    alpha = 0.55 if hovered else 0.35

    # Semi-transparent circle background
    pad = 1
    x1 = max(0, cx - BTN_RADIUS - pad)
    y1 = max(0, cy - BTN_RADIUS - pad)
    x2 = min(w_fr, cx + BTN_RADIUS + pad + 1)
    y2 = min(h_fr, cy + BTN_RADIUS + pad + 1)
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.circle(overlay, (cx - x1, cy - y1), BTN_RADIUS, (20, 20, 20),
               thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    # White ring
    ring_col = (200, 200, 200) if hovered else (140, 140, 140)
    cv2.circle(frame, (cx, cy), BTN_RADIUS, ring_col, 1, cv2.LINE_AA)

    # Gear: center hub + outer ring + teeth — scaled to BTN_RADIUS
    f           = BTN_RADIUS / 28.0
    hub_r       = max(3, int(6  * f))
    body_r      = max(6, int(13 * f))
    tooth_outer = max(10, int(20 * f))
    n_teeth = 8
    gear_col = (200, 200, 200) if hovered else (150, 150, 150)

    # Outer toothed ring
    for i in range(n_teeth):
        angle = math.radians(i * 360 / n_teeth)
        tooth_w_angle = math.radians(12)
        pts = []
        for da in (-tooth_w_angle, tooth_w_angle):
            a = angle + da
            pts.append((int(cx + body_r * math.cos(a)), int(cy + body_r * math.sin(a))))
            pts.append((int(cx + tooth_outer * math.cos(a)), int(cy + tooth_outer * math.sin(a))))
        # Tooth as a filled quad
        quad = np.array([
            [int(cx + body_r    * math.cos(angle - tooth_w_angle)),
             int(cy + body_r    * math.sin(angle - tooth_w_angle))],
            [int(cx + tooth_outer * math.cos(angle - tooth_w_angle)),
             int(cy + tooth_outer * math.sin(angle - tooth_w_angle))],
            [int(cx + tooth_outer * math.cos(angle + tooth_w_angle)),
             int(cy + tooth_outer * math.sin(angle + tooth_w_angle))],
            [int(cx + body_r    * math.cos(angle + tooth_w_angle)),
             int(cy + body_r    * math.sin(angle + tooth_w_angle))],
        ], dtype=np.int32)
        cv2.fillConvexPoly(frame, quad, gear_col, cv2.LINE_AA)

    # Outer body ring fill
    cv2.circle(frame, (cx, cy), body_r, gear_col, -1, cv2.LINE_AA)

    # Center hole — punch out using background color
    # We use the darkened version to approximate the camera background
    cv2.circle(frame, (cx, cy), hub_r, (25, 25, 25), -1, cv2.LINE_AA)

    # Small ring around center hole
    cv2.circle(frame, (cx, cy), hub_r, gear_col, 1, cv2.LINE_AA)

    return frame


def get_hovered_settings_gear(pt: tuple[int, int], w: int, h: int) -> bool:
    """Return True if pt is within the gear button hit area."""
    cx = w - BTN_MARGIN
    cy = FIRST_ROW_CY
    return math.hypot(pt[0] - cx, pt[1] - cy) <= BTN_RADIUS


# ── Settings overlay draw ────────────────────────────────────────────────────

def draw_settings_overlay(
    frame: np.ndarray,
    draft,           # Settings dataclass
    mouse_xy: tuple[int, int],
    active_tab: int = 0,
    dim: bool = True,
) -> np.ndarray:
    """Draw the settings card over the (dimmed) frame and return the result."""
    h_fr, w_fr = frame.shape[:2]

    # Dim the background
    if dim:
        frame = (frame.astype(np.float32) * 0.18).astype(np.uint8)

    ox, oy = _card_origin(w_fr, h_fr)

    # Card background
    cv2.rectangle(frame, (ox, oy), (ox + _CARD_W, oy + _CARD_H), _CARD_BG, -1)
    cv2.rectangle(frame, (ox, oy), (ox + _CARD_W, oy + _CARD_H), _CARD_BORDER, 2, cv2.LINE_AA)

    card_cx = ox + _CARD_W // 2
    hx, hy  = mouse_xy

    # ── Title ──────────────────────────────────────────────────────────────────
    _blit_text_centered(frame, _get_title_ts(), card_cx, oy + _TITLE_DY)

    # Divider below title
    div_y = oy + _TITLE_DY + 22
    cv2.line(frame, (ox + 30, div_y), (ox + _CARD_W - 30, div_y), (60, 60, 60), 1)

    # ── Tab strip ─────────────────────────────────────────────────────────────
    tab_y = oy + _TABS_DY

    hit = settings_hit(hx, hy, w_fr, h_fr, active_tab)

    for i, (tab_name, tab_cx) in enumerate(zip(TABS, _tab_centers(card_cx))):
        if i == active_tab:
            _blit_text_centered(frame, _get_tab_active_ts(tab_name), tab_cx, tab_y)
            # Gold underline
            ts = _get_tab_active_ts(tab_name)
            hw = ts[1] // 2
            cv2.line(frame, (tab_cx - hw, tab_y + 14), (tab_cx + hw, tab_y + 14),
                     (0, 215, 255), 2)
        else:
            _blit_text_centered(frame, _get_label_ts(tab_name), tab_cx, tab_y)

    # ── Tab content ───────────────────────────────────────────────────────────
    ctrl_cx = ox + _CTRL_CX

    def _draw_arrows(row_y, neg_hit, pos_hit):
        """Draw a clickable ◀ ▶ pair around ctrl_cx; gold on hover."""
        lax = ctrl_cx - 100
        rax = ctrl_cx + 100
        larr_col = (255, 215, 0) if hit == neg_hit else (200, 200, 200)
        rarr_col = (255, 215, 0) if hit == pos_hit else (200, 200, 200)
        arm = 10
        l_pts = np.array([[lax + arm, row_y - arm], [lax - arm, row_y], [lax + arm, row_y + arm]], np.int32)
        r_pts = np.array([[rax - arm, row_y - arm], [rax + arm, row_y], [rax - arm, row_y + arm]], np.int32)
        cv2.fillConvexPoly(frame, l_pts, larr_col, cv2.LINE_AA)
        cv2.fillConvexPoly(frame, r_pts, rarr_col, cv2.LINE_AA)

    # ── Controls: tracker color / circle size / volume / resolution ───────────
    row0_y = oy + _row_dy(0, 0)
    row1_y = oy + _row_dy(0, 1)
    row2_y = oy + _row_dy(0, 2)
    row3_y = oy + _row_dy(0, 3)

    # Row 0: tracker color
    _blit_text_centered(frame, _get_label_ts("tracker color"), ox + _LABEL_RX - 10, row0_y)
    _draw_arrows(row0_y, ("arrow", "color", -1), ("arrow", "color", +1))
    ci   = color_index(draft.tracker_color)
    cname, cbgr = NAMED_COLORS[ci]
    _blit_text_centered(frame, _get_color_name_ts(cname, cbgr), ctrl_cx, row0_y)

    # Row 1: circle size
    _blit_text_centered(frame, _get_label_ts("circle size"), ox + _LABEL_RX - 10, row1_y)
    _draw_slider(frame, ox, row1_y, draft.circle_radius,
                 SLIDER_MIN_RADIUS, SLIDER_MAX_RADIUS, hit == ("slider", "radius"))

    # Row 2: volume
    _blit_text_centered(frame, _get_label_ts("volume"), ox + _LABEL_RX - 10, row2_y)
    _draw_slider(frame, ox, row2_y, draft.master_volume,
                 0.0, 1.0, hit == ("slider", "volume"))

    # Row 3: resolution
    _blit_text_centered(frame, _get_label_ts("resolution"), ox + _LABEL_RX - 10, row3_y)
    _draw_slider(frame, ox, row3_y, draft.resolution,
                 0.0, 1.0, hit == ("slider", "resolution"))

    # Divider above buttons
    div2_y = oy + _BTN_DY - 22
    cv2.line(frame, (ox + 30, div2_y), (ox + _CARD_W - 30, div2_y), (60, 60, 60), 1)

    # ── Pills: Cancel LEFT, Apply RIGHT — vertically centered in gap ──────────
    btn_y = (div2_y + (oy + _CARD_H)) // 2

    # Cancel (dark fill, light text) — LEFT
    cx1 = card_cx - _PIL_W - 20
    cx2 = cx1 + _PIL_W
    cancel_hov = (hit == ("cancel",))
    cancel_col = (50, 50, 50) if cancel_hov else (35, 35, 35)
    cv2.rectangle(frame, (cx1, btn_y - _PIL_H // 2), (cx2, btn_y + _PIL_H // 2),
                  cancel_col, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (cx1, btn_y - _PIL_H // 2), (cx2, btn_y + _PIL_H // 2),
                  (120, 120, 120), 1, cv2.LINE_AA)
    _blit_text_centered(frame, _get_cancel_ts(), (cx1 + cx2) // 2, btn_y)

    # Apply (gold fill, black text) — RIGHT
    ax1 = card_cx + 20
    ax2 = ax1 + _PIL_W
    apply_hov = (hit == ("apply",))
    apply_col = (0, 225, 255) if apply_hov else (0, 195, 230)
    cv2.rectangle(frame, (ax1, btn_y - _PIL_H // 2), (ax2, btn_y + _PIL_H // 2),
                  apply_col, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (ax1, btn_y - _PIL_H // 2), (ax2, btn_y + _PIL_H // 2),
                  (255, 255, 255), 1, cv2.LINE_AA)
    _blit_text_centered(frame, _get_apply_ts(), (ax1 + ax2) // 2, btn_y)

    return frame


def _draw_slider(
    frame: np.ndarray,
    card_ox: int,
    row_y: int,
    value: float,
    vmin: float,
    vmax: float,
    active: bool,
) -> None:
    x1f = card_ox + _SL_X1
    x2f = card_ox + _SL_X2
    by1 = row_y - _SL_H // 2
    by2 = row_y + _SL_H // 2

    # Track background
    cv2.rectangle(frame, (x1f, by1), (x2f, by2), (55, 55, 55), -1)

    # Filled portion
    rng  = vmax - vmin
    norm = max(0.0, min(1.0, (value - vmin) / rng)) if rng > 0 else 0.0
    fill_x = x1f + int((x2f - x1f) * norm)
    fill_col = (0, 215, 255) if active else (0, 150, 190)
    if fill_x > x1f:
        cv2.rectangle(frame, (x1f, by1), (fill_x, by2), fill_col, -1)

    # Scrubber knob
    knob_col = (0, 225, 255) if active else (200, 200, 200)
    cv2.rectangle(frame, (fill_x - 4, row_y - 11), (fill_x + 4, row_y + 11),
                  knob_col, -1)

    # Numeric value label to the right
    if vmax - vmin > 1.0:
        label = str(int(round(value)))
    else:
        label = f"{value:.2f}"
    font_size = 18
    val_x = x2f + 36
    # Render small text with cv2 (no PIL for this value label to avoid hot path)
    cv2.putText(frame, label, (val_x - 20, row_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
