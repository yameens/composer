"""
Cool beats menu overlay for Conductor.

Parchment-card style matching draw_theory_overlay / _draw_progressions.
Non-functional placeholder — selection navigates but does not trigger playback.

Public API:
    BEAT_NAMES          — list[str]  placeholder beat names
    draw_beats_menu(frame, names, sel_idx, dim=False) -> frame
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ui_theory import (
    _FONT_TITLE, _FONT_ITEM, _FONT_HINT, _FONT_HINT_MONO,
    _BLACK, _GOLD, _GRAY, _DGRAY, _LGRAY,
)
from ui_circles import _ARROW_CHARS

# ── Beat names (placeholders, all lowercase) ─────────────────────────────────

BEAT_NAMES = [
    "jersey club", "boom bap", "trap", "drill", "house",
    "amapiano", "afrobeat", "lo-fi", "phonk", "reggaeton",
]

# ── Card geometry ─────────────────────────────────────────────────────────────

_CARD_W   = 560
_CARD_H   = 440
_CORNER_R = 18

# ── Module-level scroll state ─────────────────────────────────────────────────

_scroll: int = 0


def _line_h(font, draw: ImageDraw.ImageDraw) -> int:
    bb = draw.textbbox((0, 0), "Ag", font=font)
    return bb[3] - bb[1]


def draw_beats_menu(
    frame:       np.ndarray,
    names:       list[str],
    sel_idx:     int,
    playing_idx: int = -1,
    dim:         bool = False,
) -> np.ndarray:
    """Render the 'cool beats' parchment menu over *frame* and return the result.

    `playing_idx` marks the row of the currently-looping beat with a gold ▶.
    """
    global _scroll

    fh, fw = frame.shape[:2]

    # Card rect — centred in the frame
    x1 = (fw - _CARD_W) // 2
    y1 = (fh - _CARD_H) // 2
    x2 = x1 + _CARD_W
    y2 = y1 + _CARD_H

    # Optional background dimming (caller pre-dims the backdrop like draw_settings_overlay)
    if dim:
        frame = cv2.addWeighted(
            np.zeros_like(frame), 0.45,
            frame, 1.0 - 0.45,
            0,
        )

    # ── Parchment card via PIL RGBA composite (mirrors draw_theory_overlay) ──
    pil       = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    card_layer = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    card_draw  = ImageDraw.Draw(card_layer)
    card_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=_CORNER_R,
        fill=(245, 245, 245, 235),
    )
    card_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=_CORNER_R,
        outline=(190, 190, 190, 255), width=1,
    )
    pil  = Image.alpha_composite(pil, card_layer).convert("RGB")
    draw = ImageDraw.Draw(pil)

    pad = 40
    cx  = (x1 + x2) // 2

    # ── Title ─────────────────────────────────────────────────────────────────
    tb     = draw.textbbox((0, 0), "cool beats", font=_FONT_TITLE, stroke_width=1)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ty     = y1 + pad
    draw.text(
        (cx - tw // 2, ty), "cool beats",
        font=_FONT_TITLE, fill=_BLACK, stroke_width=1, stroke_fill=_BLACK,
    )

    # Gold underline — full title width, centred under "cool beats"
    underline_y = ty + th + 6
    draw.line([(cx - tw // 2, underline_y), (cx + tw // 2, underline_y)],
              fill=_GOLD, width=2)

    # Separator across the card
    sep_y = underline_y + 10
    draw.line([(x1 + pad, sep_y), (x2 - pad, sep_y)], fill=_LGRAY, width=1)

    # ── Hint line at the bottom ───────────────────────────────────────────────
    hint = "↑ ↓ move   •   enter   •   esc to close"
    # Split into runs: arrow chars use _FONT_HINT_MONO, rest use _FONT_HINT
    _hint_runs: list[tuple[str, bool]] = []
    _buf    = hint[0]
    _is_arr = hint[0] in _ARROW_CHARS
    for _ch in hint[1:]:
        _a = _ch in _ARROW_CHARS
        if _a == _is_arr:
            _buf += _ch
        else:
            _hint_runs.append((_buf, _is_arr))
            _buf = _ch; _is_arr = _a
    _hint_runs.append((_buf, _is_arr))
    _run_ws = [
        draw.textlength(rt, font=(_FONT_HINT_MONO if ia else _FONT_HINT))
        for rt, ia in _hint_runs
    ]
    hw  = int(sum(_run_ws))
    hh  = _FONT_HINT.getmetrics()[0] + _FONT_HINT.getmetrics()[1]
    hint_y = y2 - pad - hh
    _hx    = cx - hw // 2
    for (rt, ia), rw in zip(_hint_runs, _run_ws):
        _font = _FONT_HINT_MONO if ia else _FONT_HINT
        draw.text((_hx, hint_y), rt, font=_font, fill=_DGRAY)
        _hx += int(rw)

    # ── List of beats ─────────────────────────────────────────────────────────
    list_top = sep_y + 18
    list_bot = hint_y - 14
    rh       = _line_h(_FONT_ITEM, draw) + 12
    visible  = max(1, (list_bot - list_top) // rh)

    # Keep selected row inside the visible window
    if sel_idx < _scroll:
        _scroll = sel_idx
    elif sel_idx >= _scroll + visible:
        _scroll = sel_idx - visible + 1
    _scroll = max(0, min(_scroll, max(0, len(names) - visible)))

    iy = list_top
    for abs_i in range(_scroll, min(len(names), _scroll + visible)):
        selected = (abs_i == sel_idx)
        playing  = (abs_i == playing_idx)
        if selected:
            draw.rounded_rectangle(
                [x1 + pad - 14, iy - 4, x2 - pad + 14, iy + rh - 10],
                radius=6, fill=(235, 225, 190, 255),
            )
        # Gold ▶ marker on the currently-playing beat.
        if playing:
            ty0 = iy + 3
            tsz = max(6, rh // 4)
            draw.polygon(
                [(x1 + pad - 26, ty0), (x1 + pad - 26, ty0 + tsz),
                 (x1 + pad - 26 + tsz, ty0 + tsz // 2)],
                fill=_GOLD,
            )
        draw.text(
            (x1 + pad, iy), names[abs_i],
            font=_FONT_ITEM,
            fill=(90, 70, 0, 255) if selected else _BLACK,
        )
        iy += rh

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
