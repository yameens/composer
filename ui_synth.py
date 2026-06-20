"""
Oscillator synth UI overlays for Conductor.

Public API consumed by main.py:

    OSC_PARAMS         — list[dict], per-oscillator params
                         get(patch, osc_idx)  /  set(patch, osc_idx, value)
    GLOBAL_PARAMS      — list[dict], patch-level params
                         get(patch)  /  set(patch, value)
    EDITOR_PARAMS      — alias for GLOBAL_PARAMS (backward-compat for tests)
    EDITOR_N_ROWS      — 1 + len(OSC_PARAMS) + len(GLOBAL_PARAMS)

    draw_synth_editor(frame, patch, osc_idx, sel_idx, hover=None,
                      name_edit=False, name_buf="") -> frame
    draw_sounds_list(frame, names, sel_idx) -> frame

    editor_hit(x, y, patch, osc_idx, frame_w=1280) -> tuple | None
    slider_value_at(x, group, param_i) -> float
    sounds_hit(x, y, names, sel_idx, frame_w=1280) -> tuple | None
"""

from __future__ import annotations

import cv2
import numpy as np

from ui_circles import (
    _FONT_LG,
    _FONT_SM,
    _FONT_HUD_KEYS,
    _make_text_sprite,
    _blit_text_centered,
    app_font,
    load_accent_font,
    load_mono_font,
    make_hint_sprite,
)
from synth_engine import WAVE_NAMES

# ── Colors (BGR) ──────────────────────────────────────────────────────────────

_GOLD       = (  0, 215, 255)
_GOLD_DIM   = (  0, 130, 180)
_WHITE      = (255, 255, 255)
_BLACK      = (  0,   0,   0)
_MID_GRAY   = ( 55,  55,  55)
_HOVER_GRAY = ( 72,  72,  72)
_LIGHT_GRAY = (120, 120, 120)
_TEXT_MAIN  = (235, 235, 235)
_TEXT_DIM   = (130, 130, 130)
_TEXT_DARK  = ( 28,  28,  28)
_BOX_BG     = (245, 245, 245)

# ── Fonts ─────────────────────────────────────────────────────────────────────

_FONT_TITLE      = load_accent_font(46)
_FONT_PARAM      = app_font(22)
_FONT_HINT       = _FONT_HUD_KEYS
_FONT_HINT_MONO  = load_mono_font(24)   # FiraMono at same size as _FONT_HUD_KEYS — has arrow glyphs
_FONT_WAVE       = _FONT_SM
_FONT_LIST       = _FONT_LG
_FONT_NAME       = app_font(16)
_FONT_SECT       = app_font(13)
_FONT_TAB        = app_font(15)

# ── Layout constants (1280 × 720 target) ─────────────────────────────────────

_FRAME_W = 1280   # expected frame width; used for centring and hit-testing

# Editor vertical positions
_ED_TITLE_Y    = 58
_ED_NAME_Y     = 92
_ED_TAB_Y      = 126   # osc tab row centre
_ED_WAVE_Y     = 163   # waveform pill row centre
_ED_OSC_LBL_Y  = 200   # "— OSCILLATOR —" section label
_ED_OSC_Y0     = 222   # first OSC param row centre
_ED_OSC_DY     = 40    # spacing between OSC param rows
_ED_GLOB_LBL_Y = 388   # "— GLOBAL —" section label
_ED_GLOB_Y0    = 410   # first GLOBAL param row centre
_ED_GLOB_DY    = 38    # spacing between GLOBAL param rows
_ED_FOOTER_Y   = 676   # footer hint

# Osc tab geometry
_TAB_H     = 28
_TAB_W     = 82
_TAB_GAP   = 8
_TAB_ADD_W = 38

# Waveform pills
_N_WAVES  = len(WAVE_NAMES)
_PILL_W   = 140
_PILL_H   = 28
_PILL_GAP = 8

# Slider bar geometry (shared by OSC and GLOBAL rows)
_LABEL_RX = 344    # right edge of label text
_BAR_X1   = 360
_BAR_X2   = 904
_BAR_H    = 10
_VAL_CX   = 964    # centre of numeric value text

# Sounds list
_SL_BX1      = 320
_SL_BY1      = 88
_SL_BY2      = 622
_SL_TITLE_Y  = 132
_SL_DIV_Y    = 162
_SL_ROW_Y0   = 200
_SL_ROW_H    = 52
_SL_MAX_ROWS = 6
_SL_ACT_Y    = 546   # [ACTIVATE] button centre
_SL_FOOTER_Y = 598   # footer hint text

# ── OSC_PARAMS ────────────────────────────────────────────────────────────────
# Per-oscillator params.  All getters: (patch, osc_idx) → float.
#                         All setters: (patch, osc_idx, value) → None.

OSC_PARAMS: list[dict] = [
    {
        "name": "morph",
        "get":  lambda p, i: p.oscillators[i].morph         if i < len(p.oscillators) else 0.0,
        "set":  lambda p, i, v: setattr(p.oscillators[i], "morph",
                    max(0.0, min(1.0, v)))                   if i < len(p.oscillators) else None,
        "min": 0.0, "max": 1.0, "step": 0.05, "snap_int": False,
        "fmt": lambda v: f"{v:.2f}",
    },
    {
        "name": "detune",
        "get":  lambda p, i: p.oscillators[i].detune_cents  if i < len(p.oscillators) else 0.0,
        "set":  lambda p, i, v: setattr(p.oscillators[i], "detune_cents",
                    max(-50.0, min(50.0, v)))                if i < len(p.oscillators) else None,
        "min": -50.0, "max": 50.0, "step": 1.0, "snap_int": False,
        "fmt": lambda v: f"{v:+.0f}c",
    },
    {
        "name": "octave",
        "get":  lambda p, i: float(p.oscillators[i].octave) if i < len(p.oscillators) else 0.0,
        "set":  lambda p, i, v: setattr(p.oscillators[i], "octave",
                    int(round(max(-2.0, min(2.0, v)))))      if i < len(p.oscillators) else None,
        "min": -2.0, "max": 2.0, "step": 1.0, "snap_int": True,
        "fmt": lambda v: f"{int(v):+d}",
    },
    {
        "name": "level",
        "get":  lambda p, i: p.oscillators[i].level          if i < len(p.oscillators) else 0.0,
        "set":  lambda p, i, v: setattr(p.oscillators[i], "level",
                    max(0.0, min(1.0, v)))                    if i < len(p.oscillators) else None,
        "min": 0.0, "max": 1.0, "step": 0.05, "snap_int": False,
        "fmt": lambda v: f"{v:.2f}",
    },
]

# ── GLOBAL_PARAMS ─────────────────────────────────────────────────────────────
# Patch-level params.  All getters: (patch) → float.
#                       All setters: (patch, value) → None.

GLOBAL_PARAMS: list[dict] = [
    {
        "name": "cutoff Hz",
        "get":  lambda p: p.flt.cutoff_hz,
        "set":  lambda p, v: setattr(p.flt, "cutoff_hz", max(20.0, min(8000.0, v))),
        "min": 20.0, "max": 8000.0, "step": 100.0, "snap_int": False,
        "fmt": lambda v: f"{v:.0f}",
    },
    {
        "name": "resonance",
        "get":  lambda p: p.flt.resonance,
        "set":  lambda p, v: setattr(p.flt, "resonance", max(0.0, min(1.0, v))),
        "min": 0.0, "max": 1.0, "step": 0.05, "snap_int": False,
        "fmt": lambda v: f"{v:.2f}",
    },
    {
        "name": "attack",
        "get":  lambda p: p.env.attack,
        "set":  lambda p, v: setattr(p.env, "attack", max(0.01, min(2.0, v))),
        "min": 0.01, "max": 2.0, "step": 0.01, "snap_int": False,
        "fmt": lambda v: f"{v:.2f}s",
    },
    {
        "name": "decay",
        "get":  lambda p: p.env.decay,
        "set":  lambda p, v: setattr(p.env, "decay", max(0.01, min(2.0, v))),
        "min": 0.01, "max": 2.0, "step": 0.01, "snap_int": False,
        "fmt": lambda v: f"{v:.2f}s",
    },
    {
        "name": "sustain",
        "get":  lambda p: p.env.sustain,
        "set":  lambda p, v: setattr(p.env, "sustain", max(0.0, min(1.0, v))),
        "min": 0.0, "max": 1.0, "step": 0.05, "snap_int": False,
        "fmt": lambda v: f"{v:.2f}",
    },
    {
        "name": "release",
        "get":  lambda p: p.env.release,
        "set":  lambda p, v: setattr(p.env, "release", max(0.01, min(3.0, v))),
        "min": 0.01, "max": 3.0, "step": 0.01, "snap_int": False,
        "fmt": lambda v: f"{v:.2f}s",
    },
    {
        "name": "gain",
        "get":  lambda p: p.gain,
        "set":  lambda p, v: setattr(p, "gain", max(0.0, min(1.0, v))),
        "min": 0.0, "max": 1.0, "step": 0.05, "snap_int": False,
        "fmt": lambda v: f"{v:.2f}",
    },
]

# Legacy alias so tests that imported EDITOR_PARAMS still work.
# GLOBAL_PARAMS has the same 2-arg get/set signature as the old EDITOR_PARAMS.
EDITOR_PARAMS = GLOBAL_PARAMS

EDITOR_N_ROWS = 1 + len(OSC_PARAMS) + len(GLOBAL_PARAMS)   # = 12

# ── Sprite caches ─────────────────────────────────────────────────────────────

_LABEL_CACHE: dict[tuple, tuple] = {}
_VALUE_CACHE: dict[str,   tuple] = {}
_WAVE_CACHE:  dict[tuple, tuple] = {}
_HINT_CACHE:  dict[str,   tuple] = {}
_LIST_CACHE:  dict[tuple, tuple] = {}
_TAB_CACHE:   dict[tuple, tuple] = {}
_SECT_CACHE:  dict[str,   tuple] = {}

_EDITOR_TITLE_TS: tuple | None = None
_SL_TITLE_TS:     tuple | None = None
_SL_FOOTER_TS:    tuple | None = None
_NAME_SPRITE_CACHE: dict[str, tuple] = {}


def _lbl(text: str, color: tuple) -> tuple:
    key = (text, color)
    ts = _LABEL_CACHE.get(key)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_PARAM, color)
        _LABEL_CACHE[key] = ts
    return ts


def _val(text: str) -> tuple:
    ts = _VALUE_CACHE.get(text)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_PARAM, _TEXT_MAIN)
        _VALUE_CACHE[text] = ts
    return ts


def _wave_lbl(text: str, color: tuple) -> tuple:
    key = (text, color)
    ts = _WAVE_CACHE.get(key)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_WAVE, color)
        _WAVE_CACHE[key] = ts
    return ts


def _hint(text: str) -> tuple:
    ts = _HINT_CACHE.get(text)
    if ts is None:
        ts = make_hint_sprite(text, _FONT_HINT, _TEXT_DIM, _FONT_HINT_MONO)
        _HINT_CACHE[text] = ts
    return ts


def _list_item(text: str, color: tuple) -> tuple:
    key = (text, color)
    ts = _LIST_CACHE.get(key)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_LIST, color)
        _LIST_CACHE[key] = ts
    return ts


def _tab_lbl(text: str, color: tuple) -> tuple:
    key = (text, color)
    ts = _TAB_CACHE.get(key)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_TAB, color)
        _TAB_CACHE[key] = ts
    return ts


def _sect_lbl(text: str) -> tuple:
    ts = _SECT_CACHE.get(text)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_SECT, _TEXT_DIM)
        _SECT_CACHE[text] = ts
    return ts


# ── Layout helpers (shared by draw functions and hit-testers) ──────────────────

def _tab_rects(n_oscs: int, frame_w: int = _FRAME_W) -> list:
    """Returns (x1,y1,x2,y2) for each osc tab followed by the [+] add button."""
    total = n_oscs * (_TAB_W + _TAB_GAP) + _TAB_ADD_W if n_oscs > 0 else _TAB_ADD_W
    x0 = (frame_w - total) // 2
    y1 = _ED_TAB_Y - _TAB_H // 2
    y2 = _ED_TAB_Y + _TAB_H // 2
    rects = []
    x = x0
    for _ in range(n_oscs):
        rects.append((x, y1, x + _TAB_W, y2))
        x += _TAB_W + _TAB_GAP
    rects.append((x, y1, x + _TAB_ADD_W, y2))
    return rects


def _pill_rects(frame_w: int = _FRAME_W) -> list:
    """Returns (x1,y1,x2,y2) for each waveform pill."""
    total = _N_WAVES * _PILL_W + (_N_WAVES - 1) * _PILL_GAP
    x0 = (frame_w - total) // 2
    y1 = _ED_WAVE_Y - _PILL_H // 2
    y2 = _ED_WAVE_Y + _PILL_H // 2
    rects = []
    x = x0
    for _ in range(_N_WAVES):
        rects.append((x, y1, x + _PILL_W, y2))
        x += _PILL_W + _PILL_GAP
    return rects


def _bar_rect(row_y: int) -> tuple:
    """Clickable region for a slider bar (extended vertically for easy clicking)."""
    margin = 14
    return (_BAR_X1, row_y - margin, _BAR_X2, row_y + margin)


# ── draw_synth_editor ─────────────────────────────────────────────────────────

def draw_synth_editor(
    frame: np.ndarray,
    patch,
    osc_idx: int,
    sel_idx: int,
    hover=None,
    name_edit: bool = False,
    name_buf: str = "",
    dim: bool = True,
) -> np.ndarray:
    """Draw the full-screen synth editor overlay and return the modified frame.

    Parameters
    ----------
    osc_idx   : which oscillator tab is selected (0-based)
    sel_idx   : keyboard-selected row  (0=wave, 1-4=OSC param, 5-11=GLOBAL param)
    hover     : result of editor_hit() for the current mouse position, or None
    name_edit : True while the user is editing the patch name
    name_buf  : current text buffer when name_edit is True
    """
    global _EDITOR_TITLE_TS
    h, w = frame.shape[:2]

    # Dark overlay
    if dim:
        frame = (frame.astype(np.float32) * 0.10).astype(np.uint8)

    # Title
    if _EDITOR_TITLE_TS is None:
        _EDITOR_TITLE_TS = _make_text_sprite("sound editor", _FONT_TITLE, _GOLD)
    _blit_text_centered(frame, _EDITOR_TITLE_TS, w // 2, _ED_TITLE_Y)

    # Patch name — gold with cursor when being edited, dim otherwise
    if name_edit:
        display_name = f'"{name_buf}|"'
        name_ts = _make_text_sprite(display_name, _FONT_NAME, _GOLD)
    else:
        name_ts = _NAME_SPRITE_CACHE.get(patch.name)
        if name_ts is None:
            name_ts = _make_text_sprite(f'"{patch.name}"', _FONT_NAME, _TEXT_DIM)
            _NAME_SPRITE_CACHE[patch.name] = name_ts
    _blit_text_centered(frame, name_ts, w // 2, _ED_NAME_Y)

    # Osc tab row
    _draw_tab_row(frame, patch, osc_idx, hover, w)

    # Waveform pills
    _draw_waveform_row(frame, patch, osc_idx, selected=(sel_idx == 0), hover=hover, frame_w=w)

    # OSC section
    _blit_text_centered(frame, _sect_lbl("— OSCILLATOR —"), w // 2, _ED_OSC_LBL_Y)
    for i, param in enumerate(OSC_PARAMS):
        row_y  = _ED_OSC_Y0 + i * _ED_OSC_DY
        is_sel = (sel_idx == 1 + i)
        is_hov = (hover == ("slider", "osc", i))
        val    = param["get"](patch, osc_idx) if osc_idx < len(patch.oscillators) else param["min"]
        _draw_param_row(frame, val, param, row_y, is_sel, is_hov)

    # GLOBAL section
    _blit_text_centered(frame, _sect_lbl("— GLOBAL —"), w // 2, _ED_GLOB_LBL_Y)
    for i, param in enumerate(GLOBAL_PARAMS):
        row_y  = _ED_GLOB_Y0 + i * _ED_GLOB_DY
        is_sel = (sel_idx == 1 + len(OSC_PARAMS) + i)
        is_hov = (hover == ("slider", "global", i))
        val    = param["get"](patch)
        _draw_param_row(frame, val, param, row_y, is_sel, is_hov)

    # Footer hint
    _blit_text_centered(
        frame,
        _hint("Tab osc   \u2191\u2193 row   \u2190\u2192 adjust   A audition   S save   ESC close"),
        w // 2,
        _ED_FOOTER_Y,
    )
    return frame


def _draw_tab_row(
    frame: np.ndarray,
    patch,
    osc_idx: int,
    hover,
    frame_w: int,
) -> None:
    n     = len(patch.oscillators)
    rects = _tab_rects(n, frame_w)

    for i in range(n):
        x1, y1, x2, y2 = rects[i]
        active = (i == osc_idx)
        h_tab  = hover in (("tab", i), ("remove", i))

        if active:
            bg, border, txt = _GOLD, _GOLD, _BLACK
        elif h_tab:
            bg, border, txt = (38, 38, 38), _GOLD, _TEXT_MAIN
        else:
            bg, border, txt = (22, 22, 22), _MID_GRAY, _TEXT_DIM

        cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
        _blit_text_centered(frame, _tab_lbl(f"OSC {i + 1}", txt), (x1 + x2) // 2, _ED_TAB_Y)

        # × remove button in top-right of tab (only when >1 osc)
        if n > 1:
            xm, ym = x2 - 10, y1 + 7
            arm    = 4
            xcol   = _GOLD if hover == ("remove", i) else (_TEXT_MAIN if active else _TEXT_DIM)
            cv2.line(frame, (xm - arm, ym - arm), (xm + arm, ym + arm), xcol, 1, cv2.LINE_AA)
            cv2.line(frame, (xm + arm, ym - arm), (xm - arm, ym + arm), xcol, 1, cv2.LINE_AA)

    # [+] add button
    ax1, ay1, ax2, ay2 = rects[-1]
    enabled = n < 4
    h_add   = (hover == ("add",))
    add_bg  = (42, 42, 42) if h_add and enabled else (22, 22, 22)
    add_col = _GOLD if enabled else _MID_GRAY
    cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), add_bg, -1)
    cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), add_col, 1, cv2.LINE_AA)
    acx, acy = (ax1 + ax2) // 2, (ay1 + ay2) // 2
    arm = 8
    cv2.line(frame, (acx - arm, acy), (acx + arm, acy), add_col, 2, cv2.LINE_AA)
    cv2.line(frame, (acx, acy - arm), (acx, acy + arm), add_col, 2, cv2.LINE_AA)


def _draw_waveform_row(
    frame: np.ndarray,
    patch,
    osc_idx: int,
    selected: bool,
    hover,
    frame_w: int,
) -> None:
    current = patch.oscillators[osc_idx].wave if osc_idx < len(patch.oscillators) else "saw"
    rects   = _pill_rects(frame_w)

    for i, wname in enumerate(WAVE_NAMES):
        x1, y1, x2, y2 = rects[i]
        is_active  = (wname == current)
        is_hovered = (hover == ("wave", i))

        if is_active:
            bg, border, txt = _GOLD, _GOLD, _BLACK
        elif is_hovered:
            bg, border, txt = (38, 38, 38), _GOLD, _TEXT_MAIN
        elif selected:
            bg, border, txt = (22, 22, 22), _MID_GRAY, _LIGHT_GRAY
        else:
            bg, border, txt = (15, 15, 15), (35, 35, 35), _TEXT_DIM

        cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
        _blit_text_centered(frame, _wave_lbl(wname, txt), (x1 + x2) // 2, _ED_WAVE_Y)

    if selected:
        bx = rects[0][0] - 22 if rects else _BAR_X1 - 22
        _draw_bracket(frame, bx, _ED_WAVE_Y)


def _draw_param_row(
    frame: np.ndarray,
    val: float,
    param: dict,
    row_y: int,
    is_sel: bool,
    is_hov: bool,
) -> None:
    # Label (gold when selected, bright when hovered, dim otherwise)
    col = _GOLD if is_sel else (_TEXT_MAIN if is_hov else _TEXT_DIM)
    lts = _lbl(param["name"], col)
    _blit_text_centered(frame, lts, _LABEL_RX - lts[1] // 2, row_y)

    # Bar background
    by1   = row_y - _BAR_H // 2
    by2   = row_y + _BAR_H // 2
    bg    = _HOVER_GRAY if (is_hov and not is_sel) else _MID_GRAY
    cv2.rectangle(frame, (_BAR_X1, by1), (_BAR_X2, by2), bg, -1)

    # Filled portion
    rng  = param["max"] - param["min"]
    norm = max(0.0, min(1.0, (val - param["min"]) / rng)) if rng > 0 else 0.0
    fill = _BAR_X1 + int((_BAR_X2 - _BAR_X1) * norm)
    bcol = _GOLD if is_sel else _GOLD_DIM
    if fill > _BAR_X1:
        cv2.rectangle(frame, (_BAR_X1, by1), (fill, by2), bcol, -1)

    # Scrubber handle — thin upright rectangle (shown when selected or hovered)
    if is_sel or is_hov:
        dot_col = _GOLD if is_sel else _GOLD_DIM
        cv2.rectangle(frame, (fill - 3, row_y - 10), (fill + 3, row_y + 10), dot_col, -1)

    # Numeric value text
    _blit_text_centered(frame, _val(param["fmt"](val)), _VAL_CX, row_y)


def _draw_bracket(frame: np.ndarray, bx: int, cy: int, arm: int = 10) -> None:
    """Small gold ']' selection indicator."""
    cv2.line(frame, (bx, cy - arm), (bx, cy + arm),     _GOLD, 2, cv2.LINE_AA)
    cv2.line(frame, (bx, cy - arm), (bx + arm - 2, cy - arm), _GOLD, 2, cv2.LINE_AA)
    cv2.line(frame, (bx, cy + arm), (bx + arm - 2, cy + arm), _GOLD, 2, cv2.LINE_AA)


# ── editor_hit ────────────────────────────────────────────────────────────────

def editor_hit(x: int, y: int, patch, osc_idx: int, frame_w: int = _FRAME_W):
    """
    Return the editor element under pixel (x, y), or None.

    Return values:
      ("tab",    i)            — osc tab i
      ("remove", i)            — × on osc tab i   (only when n_oscs > 1)
      ("add",)                 — [+] add-oscillator button
      ("wave",   i)            — waveform pill i
      ("slider", "osc",    i)  — OSC param i slider bar
      ("slider", "global", i)  — GLOBAL param i slider bar
      ("name",)                — patch name area
      None                     — no hit
    """
    n = len(patch.oscillators)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    for i, (x1, y1, x2, y2) in enumerate(_tab_rects(n, frame_w)):
        if x1 <= x <= x2 and y1 <= y <= y2:
            if i == n:                                  # [+] button
                return ("add",)
            if n > 1 and x >= x2 - 20 and y <= y1 + 16:   # × remove zone
                return ("remove", i)
            return ("tab", i)

    # ── Wave pills ────────────────────────────────────────────────────────────
    for i, (x1, y1, x2, y2) in enumerate(_pill_rects(frame_w)):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return ("wave", i)

    # ── OSC param bars ────────────────────────────────────────────────────────
    if osc_idx < n:
        for i in range(len(OSC_PARAMS)):
            bx1, by1, bx2, by2 = _bar_rect(_ED_OSC_Y0 + i * _ED_OSC_DY)
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                return ("slider", "osc", i)

    # ── GLOBAL param bars ─────────────────────────────────────────────────────
    for i in range(len(GLOBAL_PARAMS)):
        bx1, by1, bx2, by2 = _bar_rect(_ED_GLOB_Y0 + i * _ED_GLOB_DY)
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            return ("slider", "global", i)

    # ── Patch name area ───────────────────────────────────────────────────────
    name_x1, name_x2 = _BAR_X1, _BAR_X2
    name_y1, name_y2 = _ED_NAME_Y - 14, _ED_NAME_Y + 14
    if name_x1 <= x <= name_x2 and name_y1 <= y <= name_y2:
        return ("name",)

    return None


# ── slider_value_at ───────────────────────────────────────────────────────────

def slider_value_at(x: int, group: str, param_i: int) -> float:
    """Convert a bar x-pixel position to a real parameter value.

    Clamps to [min, max] and applies integer-snap for params marked snap_int.
    """
    params = OSC_PARAMS if group == "osc" else GLOBAL_PARAMS
    if param_i >= len(params):
        return 0.0
    p    = params[param_i]
    bar  = max(1, _BAR_X2 - _BAR_X1)
    norm = max(0.0, min(1.0, (x - _BAR_X1) / bar))
    v    = p["min"] + norm * (p["max"] - p["min"])
    v    = max(p["min"], min(p["max"], v))
    if p.get("snap_int", False):
        v = float(round(v))
    return v


# ── draw_sounds_list ──────────────────────────────────────────────────────────

def draw_sounds_list(
    frame: np.ndarray,
    names: list,
    sel_idx: int,
    dim: bool = True,
) -> np.ndarray:
    """Draw the saved-sounds list overlay and return the frame."""
    global _SL_TITLE_TS, _SL_FOOTER_TS
    h, w = frame.shape[:2]

    # Dark overlay
    if dim:
        frame = (frame.astype(np.float32) * 0.14).astype(np.uint8)

    bx1, bx2 = _SL_BX1, w - _SL_BX1
    by1, by2  = _SL_BY1, _SL_BY2

    # White box
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), _BOX_BG, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), _GOLD, 2, cv2.LINE_AA)

    cx = (bx1 + bx2) // 2

    # Title
    if _SL_TITLE_TS is None:
        _SL_TITLE_TS = _make_text_sprite("YOUR SOUNDS", _FONT_TITLE, _TEXT_DARK)
    _blit_text_centered(frame, _SL_TITLE_TS, cx, _SL_TITLE_Y)

    # Divider
    cv2.line(frame, (bx1 + 20, _SL_DIV_Y), (bx2 - 20, _SL_DIV_Y), (190, 190, 190), 1)

    # Sound name rows
    if not names:
        _blit_text_centered(
            frame, _list_item("No saved sounds yet.", (160, 160, 160)),
            cx, _SL_ROW_Y0 + _SL_ROW_H,
        )
    else:
        start = max(0, sel_idx - _SL_MAX_ROWS // 2)
        end   = min(len(names), start + _SL_MAX_ROWS)
        start = max(0, end - _SL_MAX_ROWS)
        for j, idx in enumerate(range(start, end)):
            ry  = _SL_ROW_Y0 + j * _SL_ROW_H
            sel = (idx == sel_idx)
            if sel:
                rh = _SL_ROW_H - 8
                cv2.rectangle(
                    frame,
                    (bx1 + 12, ry - rh // 2),
                    (bx2 - 12, ry + rh // 2),
                    _GOLD, -1,
                )
            _blit_text_centered(
                frame,
                _list_item(names[idx], _BLACK if sel else _TEXT_DARK),
                cx, ry,
            )

    # [ACTIVATE] button
    abw, abh = 130, 32
    cv2.rectangle(
        frame,
        (cx - abw // 2, _SL_ACT_Y - abh // 2),
        (cx + abw // 2, _SL_ACT_Y + abh // 2),
        _GOLD, -1,
    )
    _blit_text_centered(frame, _tab_lbl("ACTIVATE", _BLACK), cx, _SL_ACT_Y)

    # Footer hint
    if _SL_FOOTER_TS is None:
        _SL_FOOTER_TS = make_hint_sprite(
            "\u2191\u2193 navigate    Enter / click activate    ESC back",
            _FONT_HINT, _TEXT_DARK, _FONT_HINT_MONO,
        )
    _blit_text_centered(frame, _SL_FOOTER_TS, cx, _SL_FOOTER_Y)

    return frame


# ── sounds_hit ────────────────────────────────────────────────────────────────

def sounds_hit(x: int, y: int, names: list, sel_idx: int, frame_w: int = _FRAME_W):
    """
    Hit-test (x, y) against sounds list elements. Returns one of:
      ("row",      idx)   — name row (not currently selected)
      ("activate",)       — [ACTIVATE] button, or already-selected row
      ("close",)          — clicked outside the white box
      None
    """
    bx1 = _SL_BX1
    bx2 = frame_w - _SL_BX1

    # Outside box → close overlay
    if not (_SL_BX1 <= x <= bx2 and _SL_BY1 <= y <= _SL_BY2):
        return ("close",)

    # [ACTIVATE] button
    cx   = (bx1 + bx2) // 2
    abw, abh = 130, 32
    if (cx - abw // 2 <= x <= cx + abw // 2 and
            _SL_ACT_Y - abh // 2 <= y <= _SL_ACT_Y + abh // 2):
        return ("activate",)

    # Name rows (using the same scroll window as draw_sounds_list)
    if names:
        start = max(0, sel_idx - _SL_MAX_ROWS // 2)
        end   = min(len(names), start + _SL_MAX_ROWS)
        start = max(0, end - _SL_MAX_ROWS)
        for j, idx in enumerate(range(start, end)):
            ry = _SL_ROW_Y0 + j * _SL_ROW_H
            if abs(y - ry) <= _SL_ROW_H // 2:
                return ("activate",) if idx == sel_idx else ("row", idx)

    return None


def _SL_BOX_X2_FN(w: int) -> int:
    return w - _SL_BX1
