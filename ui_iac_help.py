"""
Chunk 9 — IAC Setup Help Card
Renders a centered white rounded-rectangle card with numbered setup steps
for connecting the macOS IAC Driver to Logic or FL Studio.

public api:
    draw_iac_help(frame, tab_idx) -> frame
    prewarm_iac_help()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ui_circles import (
    app_font,
    load_mono_font,
    _blit_bgra,
    _to_premul,
    _ARROW_CHARS,
)

# ── Layout constants (match ui_theory card) ───────────────────────────────────

CARD_W_FRAC   = 0.70
CARD_H_FRAC   = 0.70
CARD_ALPHA    = 0.93
DIM_ALPHA     = 0.48
CARD_CORNER_R = 12

# ── Fonts ──────────────────────────────────────────────────────────────────────

_FONT_TITLE = app_font(32)
_FONT_TAB   = app_font(21)
_FONT_BODY  = app_font(22)
_FONT_HINT  = app_font(14)
_FONT_MONO  = load_mono_font(14)   # FiraMono — contains arrow glyphs

# ── Colors (PIL RGBA) ──────────────────────────────────────────────────────────

_BLACK = (10,  10,  10,  255)
_GRAY  = (160, 160, 160, 255)
_DGRAY = (100, 100, 100, 255)
_LGRAY = (200, 200, 200, 255)
_GOLD  = (170, 130,   0, 255)

# ── Tab labels and content ────────────────────────────────────────────────────

_TABS = ("logic", "fl studio")

_TAB_STEPS: dict[int, list[str]] = {
    0: [
        "1. open audio midi setup, found in applications then utilities",
        "2. open the window menu and choose show midi studio",
        "3. double click the iac driver icon",
        "4. tick the box that says device is online",
        "5. open logic and add a software instrument track",
        "6. record arm that track so it listens for midi",
        "7. come back here and play, logic makes the sound",
    ],
    1: [
        "1. open audio midi setup, found in applications then utilities",
        "2. open the window menu and choose show midi studio",
        "3. double click the iac driver icon",
        "4. tick the box that says device is online",
        "5. open fl studio and press f10 to open midi settings",
        "6. find the iac driver bus in the input list and turn it on",
        "7. come back here and play, fl studio makes the sound",
    ],
}

# ── Text layout helpers (replicated from ui_theory — same pattern) ────────────

def _wrap_text(
    text:     str,
    font:     ImageFont.FreeTypeFont,
    max_w_px: int,
    draw:     ImageDraw.ImageDraw,
    stroke:   int = 0,
) -> list[str]:
    """Word-wrap text to max_w_px, preserving explicit newlines."""
    result: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            result.append("")
            continue
        words, line = para.split(), ""
        for word in words:
            candidate = (line + " " + word).strip()
            bb = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke)
            if bb[2] - bb[0] <= max_w_px:
                line = candidate
            else:
                if line:
                    result.append(line)
                line = word
        if line:
            result.append(line)
    return result


def _line_h(font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw) -> int:
    bb = draw.textbbox((0, 0), "Ag", font=font)
    return bb[3] - bb[1]


# ── Card rect helper ──────────────────────────────────────────────────────────

def _card_rect(fw: int, fh: int) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for the centered modal card."""
    cw = int(fw * CARD_W_FRAC)
    ch = int(fh * CARD_H_FRAC)
    x1 = (fw - cw) // 2
    y1 = (fh - ch) // 2
    return x1, y1, x1 + cw, y1 + ch


# ── Static card sprite cache ──────────────────────────────────────────────────
# Each tab's full card (at 1280×720) is rendered once and cached as a
# premultiplied BGRA sprite; draw_iac_help just blits it — no PIL per frame.

_CARD_SPRITE: dict[int, Optional[np.ndarray]] = {0: None, 1: None}
_CARD_W: int = 1280
_CARD_H: int = 720


def _build_card_sprite(tab_idx: int, fw: int, fh: int) -> np.ndarray:
    """Render the full card for tab_idx into a BGRA sprite the size of the frame."""
    x1, y1, x2, y2 = _card_rect(fw, fh)
    pad = 52
    cx  = (x1 + x2) // 2

    # ── Base: transparent canvas ──────────────────────────────────────────
    base = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))

    # ── White card with rounded corners ───────────────────────────────────
    card_layer = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    card_draw  = ImageDraw.Draw(card_layer)
    card_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=CARD_CORNER_R,
        fill=(245, 245, 245, int(255 * CARD_ALPHA)),
    )
    card_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=CARD_CORNER_R,
        outline=(190, 190, 190, 255), width=1,
    )
    card = Image.alpha_composite(base, card_layer)
    draw = ImageDraw.Draw(card)

    # ── Title ─────────────────────────────────────────────────────────────
    tb     = draw.textbbox((0, 0), "connect iac", font=_FONT_TITLE, stroke_width=1)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ty     = y1 + pad
    draw.text(
        (cx - tw // 2, ty), "connect iac",
        font=_FONT_TITLE, fill=_BLACK, stroke_width=1, stroke_fill=_BLACK,
    )

    # ── Tab row (mirrors _draw_progressions in ui_theory.py) ─────────────
    tab_y   = ty + th + 22
    tab_gap = 40
    widths  = []
    for label in _TABS:
        lb = draw.textbbox((0, 0), label, font=_FONT_TAB)
        widths.append(lb[2] - lb[0])
    total_w = sum(widths) + tab_gap
    tx      = cx - total_w // 2
    tab_h   = _line_h(_FONT_TAB, draw)
    for i, label in enumerate(_TABS):
        active = (i == tab_idx)
        draw.text((tx, tab_y), label, font=_FONT_TAB,
                  fill=_BLACK if active else _GRAY)
        if active:
            draw.line(
                [(tx, tab_y + tab_h + 5), (tx + widths[i], tab_y + tab_h + 5)],
                fill=_GOLD, width=2,
            )
        tx += widths[i] + tab_gap

    # Separator under tabs
    sep_y = tab_y + tab_h + 16
    draw.line([(x1 + pad, sep_y), (x2 - pad, sep_y)], fill=_LGRAY, width=1)

    # ── Hint line (footer, centered) ──────────────────────────────────────
    hint_text = "← → switch tab    esc close"
    hb = draw.textbbox((0, 0), hint_text, font=_FONT_HINT)
    hh = hb[3] - hb[1]
    hint_y = y2 - pad - hh

    # Render hint via runs so arrow glyphs use the mono font (same as ui_theory)
    _hint_runs: list[tuple[str, bool]] = []
    _buf = hint_text[0]; _is_arr = hint_text[0] in _ARROW_CHARS
    for _ch in hint_text[1:]:
        _a = _ch in _ARROW_CHARS
        if _a == _is_arr:
            _buf += _ch
        else:
            _hint_runs.append((_buf, _is_arr))
            _buf = _ch; _is_arr = _a
    _hint_runs.append((_buf, _is_arr))
    _run_ws = [
        draw.textlength(rt, font=(_FONT_MONO if ia else _FONT_HINT))
        for rt, ia in _hint_runs
    ]
    hw   = int(sum(_run_ws))
    _hx  = cx - hw // 2
    for (rt, ia), rw in zip(_hint_runs, _run_ws):
        _font = _FONT_MONO if ia else _FONT_HINT
        draw.text((_hx, hint_y), rt, font=_font, fill=_DGRAY)
        _hx += int(rw)

    # ── Body — numbered steps, left-aligned (mirrors _draw_lesson body) ───
    inner_w  = (x2 - x1) - pad * 2
    list_top = sep_y + 18
    list_bot = hint_y - 14
    steps    = _TAB_STEPS[tab_idx]
    lh       = _line_h(_FONT_BODY, draw) + 6

    total_body_h = 0
    all_lines: list[list[str]] = []
    for step in steps:
        wrapped = _wrap_text(step, _FONT_BODY, inner_w, draw)
        all_lines.append(wrapped)
        total_body_h += len(wrapped) * lh

    avail_h = list_bot - list_top
    body_y  = list_top + max(0, (avail_h - total_body_h) // 2)

    for wrapped in all_lines:
        for line in wrapped:
            if body_y + lh > list_bot:
                break
            if line:
                draw.text((x1 + pad, body_y), line, font=_FONT_BODY, fill=_BLACK)
            body_y += lh

    # ── Convert to premultiplied BGRA sprite ──────────────────────────────
    rgba = np.array(card)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return _to_premul(bgra)


def prewarm_iac_help() -> None:
    """Build both tab card sprites up front so the first switch into iac is free."""
    global _CARD_SPRITE, _CARD_W, _CARD_H
    for tab_idx in (0, 1):
        if _CARD_SPRITE[tab_idx] is None:
            _CARD_SPRITE[tab_idx] = _build_card_sprite(tab_idx, _CARD_W, _CARD_H)


def draw_iac_help(frame: np.ndarray, tab_idx: int) -> np.ndarray:
    """Alpha-composite the IAC setup card onto frame and return the result.

    The card sprites are built once and cached; no PIL roundtrip per frame.
    tab_idx: 0 = logic, 1 = fl studio.
    """
    global _CARD_W, _CARD_H

    fh, fw = frame.shape[:2]

    # Rebuild cache once if frame size differs from cached size (e.g. first call
    # on a camera whose resolution isn't the 1280×720 prewarm default). Update
    # the cached dims so this never re-renders PIL on the hot path again.
    if _CARD_SPRITE[tab_idx] is None or fw != _CARD_W or fh != _CARD_H:
        _CARD_SPRITE[0] = _build_card_sprite(0, fw, fh)
        _CARD_SPRITE[1] = _build_card_sprite(1, fw, fh)
        _CARD_W, _CARD_H = fw, fh

    sprite = _CARD_SPRITE[tab_idx]
    if sprite is None:
        return frame

    _blit_bgra(frame, sprite, 0, 0)
    return frame
