"""
SYN-mode instrument picker overlay for Conductor.

Public API consumed by main.py:

    draw_instrument_picker(frame, instruments, sel_idx) -> frame

instruments: list of (name: str, program: int) — e.g. GM_INSTRUMENTS from chord_engine.
sel_idx:     index of the currently highlighted row.

Styled as a translucent-black rounded-rectangle panel (like draw_practice_box in
ui_theory.py), with a gold highlight bar on the selected row and a centered
scrolling window (like draw_sounds_list in ui_synth.py).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ui_circles import (
    _FONT_LG,
    _FONT_SM,
    _make_text_sprite,
    _blit_text_centered,
    load_accent_font,
    load_mono_font,
    make_hint_sprite,
)

# ── Colors (BGR) ──────────────────────────────────────────────────────────────

_GOLD      = (  0, 215, 255)
_WHITE     = (255, 255, 255)
_BLACK     = (  0,   0,   0)
_TEXT_DARK = ( 28,  28,  28)
_TEXT_LIGHT = (235, 235, 235)
_DIVIDER   = (180, 180, 180)

# ── Fonts ─────────────────────────────────────────────────────────────────────

_FONT_TITLE  = load_accent_font(42)
_FONT_ROW    = _FONT_LG          # same size as sounds-list rows
_FONT_FOOTER = _FONT_SM          # plain body font — accent font lacks arrow glyphs
_FONT_FOOTER_MONO = load_mono_font(20)  # FiraMono at same size — has arrow glyphs

# ── Layout ────────────────────────────────────────────────────────────────────

_PANEL_W       = 420   # panel width (px)
_PANEL_H       = 500   # panel height (px)
_MAX_ROWS      = 10    # max visible rows at once
_ROW_H         = 36    # vertical spacing per row
_TITLE_Y_OFF   = 38    # title centre y from panel top
_DIV_Y_OFF     = 58    # divider y from panel top
_FOOTER_Y_OFF  = 14    # footer centre y from panel bottom (upward)
_PANEL_RADIUS  = 12    # rounded-rectangle corner radius

# ── Sprite cache ──────────────────────────────────────────────────────────────
# Keyed by (text, color_bgr_tuple) — same convention as ui_synth._list_item.

_SPRITE_CACHE: dict[tuple, tuple] = {}

# Lazily cached static sprites
_TITLE_SPRITE:  tuple | None = None
_FOOTER_SPRITE: tuple | None = None


def _get_sprite(text: str, color: tuple) -> tuple:
    key = (text, color)
    ts = _SPRITE_CACHE.get(key)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_ROW, color)
        _SPRITE_CACHE[key] = ts
    return ts


# ── Public draw function ───────────────────────────────────────────────────────

def draw_instrument_picker(
    frame: np.ndarray,
    instruments: list[tuple[str, int]],
    sel_idx: int,
) -> np.ndarray:
    """Draw the SYN instrument picker overlay and return the updated frame.

    instruments: list of (name, program) — only names are displayed.
    sel_idx:     index of the highlighted row (0-based).
    """
    global _TITLE_SPRITE, _FOOTER_SPRITE

    fh, fw = frame.shape[:2]

    # ── Translucent-black rounded panel via PIL RGBA composite ────────────
    panel_x1 = (fw - _PANEL_W) // 2
    panel_y1 = (fh - _PANEL_H) // 2
    panel_x2 = panel_x1 + _PANEL_W
    panel_y2 = panel_y1 + _PANEL_H

    pil       = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    box_layer = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    box_draw  = ImageDraw.Draw(box_layer)
    box_draw.rounded_rectangle(
        [panel_x1, panel_y1, panel_x2, panel_y2],
        radius=_PANEL_RADIUS,
        fill=(0, 0, 0, 205),
    )
    pil   = Image.alpha_composite(pil, box_layer).convert("RGB")
    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    cx = (panel_x1 + panel_x2) // 2

    # ── Title ─────────────────────────────────────────────────────────────
    if _TITLE_SPRITE is None:
        _TITLE_SPRITE = _make_text_sprite("instruments", _FONT_TITLE, _WHITE)
    _blit_text_centered(frame, _TITLE_SPRITE, cx, panel_y1 + _TITLE_Y_OFF)

    # ── Divider ───────────────────────────────────────────────────────────
    div_y = panel_y1 + _DIV_Y_OFF
    cv2.line(
        frame,
        (panel_x1 + 20, div_y),
        (panel_x2 - 20, div_y),
        _DIVIDER, 1,
    )

    # ── Rows (centered-scrolling window) ──────────────────────────────────
    n = len(instruments)
    if n > 0:
        # Center the window on sel_idx
        start = max(0, sel_idx - _MAX_ROWS // 2)
        end   = min(n, start + _MAX_ROWS)
        start = max(0, end - _MAX_ROWS)   # re-clamp if near the bottom

        n_visible  = end - start
        region_top = _DIV_Y_OFF + 12
        region_bot = _PANEL_H - _FOOTER_Y_OFF - 16
        block_h    = n_visible * _ROW_H
        rows_y0    = region_top + (region_bot - region_top - block_h) // 2 + _ROW_H // 2

        for j, idx in enumerate(range(start, end)):
            row_cy = panel_y1 + rows_y0 + j * _ROW_H
            sel    = (idx == sel_idx)

            if sel:
                # Gold highlight bar
                bar_h = _ROW_H - 8
                cv2.rectangle(
                    frame,
                    (panel_x1 + 12, row_cy - bar_h // 2),
                    (panel_x2 - 12, row_cy + bar_h // 2),
                    _GOLD, -1,
                )

            name   = instruments[idx][0]
            color  = _TEXT_DARK if sel else _TEXT_LIGHT
            _blit_text_centered(frame, _get_sprite(name, color), cx, row_cy)

    # ── Footer hint ───────────────────────────────────────────────────────
    if _FOOTER_SPRITE is None:
        _FOOTER_SPRITE = make_hint_sprite(
            "↑↓ navigate    Enter select    ESC close",
            _FONT_FOOTER, _WHITE, _FONT_FOOTER_MONO,
        )
    _blit_text_centered(
        frame, _FOOTER_SPRITE, cx, panel_y2 - _FOOTER_Y_OFF,
    )

    return frame
