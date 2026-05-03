"""Theory overlay — lesson selection and stepped content display.

Renders a glass-card modal on top of the OpenCV frame.
All text is lowercase per lessonDesign.txt; Georgia font via app_font().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ui_circles import app_font, RADIUS

# ── Constants ─────────────────────────────────────────────────────────────────

DWELL_FRAMES    = 15          # ~0.5 s at 30 fps to trigger arrow navigation
ARROW_RADIUS    = 34          # dwell hit radius (px)
CARD_W_FRAC     = 0.70        # modal width as fraction of frame width
CARD_H_FRAC     = 0.70        # modal height as fraction of frame height
CARD_ALPHA      = 0.93        # modal card opacity
DIM_ALPHA       = 0.48        # background darkening factor
CARD_CORNER_R   = 12          # rounded corner radius (px)

# Identifies the slide where pressing T opens practice mode
PRACTICE_SLIDE  = (1, 2)      # (lesson_idx, step_idx) — "circle of fifths: goes like this"

# Footer strip height (must match HUD_BOTTOM_STRIP_H in main.py)
_FOOTER_H = 52

# ── Fonts (Georgia, loaded once at import) ─────────────────────────────────────

_FONT_TITLE  = app_font(32)
_FONT_HEADER = app_font(34)
_FONT_BODY   = app_font(22)
_FONT_ITEM   = app_font(21)
_FONT_HINT   = app_font(14)
_FONT_ARROW  = app_font(30)
_FONT_PRAC   = app_font(20)   # practice box body
_FONT_PRAC_H = app_font(16)   # practice box hint line

# ── Colors (PIL RGBA) ──────────────────────────────────────────────────────────

_BLACK = (10,  10,  10,  255)
_WHITE = (255, 255, 255, 255)
_GRAY  = (160, 160, 160, 255)
_DGRAY = (100, 100, 100, 255)
_LGRAY = (200, 200, 200, 255)
_GOLD  = (170, 130,   0, 255)   # muted gold for dwell arc

# ── Lesson data ───────────────────────────────────────────────────────────────
# Each step: {"header": str, "body": str}
# (smiley face) → ":)"  per lessonDesign.txt

LESSONS: dict[int, list[dict[str, str]]] = {
    0: [
        {
            "header": "lesson zero",
            "body": (
                "navigate the theory pages by keyboard."
            ),
        },
        {
            "header": "preface",
            "body": (
                "this is not a learning tool for proper music theory. "
                "it's a tool for discovering cool chord progressions that "
                "will carry over when you jam on an instrument. if you have "
                "no music theory experience, that's completely fine, though "
                "some of the relationships between notes and chords may not "
                "feel immediately intuitive. if you'd like to deepen your "
                "understanding of music theory, i'd highly recommend "
                "musictheory.net."
            ),
        },
        {
            "header": "preface",
            "body": (
                "i'll do my best to explain the intuition behind these chord "
                "progressions, but if you learn better by doing, feel free to "
                "skip straight to the progressions hand guide and try them out."
            ),
        },
        {
            "header": "root notes",
            "body": (
                "press \"t\" on the keyboard anytime to practice your newfound "
                "knowledge. don't worry, we will save your progress :)\n\n"
                "each root note is the foundation of your chord. in western "
                "music theory, there are 12 root notes: a, a#, b, c, c#, d, "
                "d#, e, f, f#, g, and g#. each represents a different pitch, "
                "a sound produced by a specific frequency. as you move from "
                "a to g, the pitch increases. after g, you cycle back to a, "
                "but in a higher octave: the same note, just pitched up."
            ),
        },
        {
            "header": "root notes",
            "body": (
                "you may notice that the \"roots\" circle only shows 7 notes. "
                "to access the sharp notes, hover and pinch with your middle "
                "and pointer finger together. give it a try now."
            ),
        },
        {
            "header": "chord types",
            "body": (
                "the circle on the right controls the chord type. so, what is "
                "a chord? a chord is a set of notes built on top of the root. "
                "the way the notes are built produces a different sound and "
                "type of chord. for example, if the root is c, the major "
                "chord, which has a happy sound, is c, e, and g. different "
                "chord types have distinct feelings: happy, eerie, sad, "
                "mysterious, or funky, each with its own name. with your left "
                "pointer finger hovering over c, try cycling through the chord "
                "types and listen to how each one feels."
            ),
        },
        {
            "header": "chord types",
            "body": (
                "that covers the basics. head to lesson one to start learning "
                "about chord progressions."
            ),
        },
    ],
    1: [
        {
            "header": "lesson one",
            "body": (
                "now that you know what a chord is, let's learn how to build "
                "some cool progressions. a chord progression is simply a "
                "sequence of chords. some chords sound great together and "
                "others don't. however, there's no single \"correct\" "
                "progression. different genres and cultures focus on different "
                "patterns, and that's what makes music really diverse."
            ),
        },
        {
            "header": "the circle of fifths",
            "body": (
                "a lot of music theory treats the circle of fifths as the gold "
                "standard. it sounds naturally pleasing to the brain because it "
                "is the perfect balance between tension and resolution. the "
                "sound waves also progress in a 3:2 ratio that the brain "
                "processes very easily. i couldn't tell you exactly why, but "
                "it works. let's practice it in c major."
            ),
        },
        {
            "header": "the circle of fifths",
            "body": (
                "the circle of fifths goes like this:\n\n"
                "c major  |  f major  |  b dim  |  e major"
                "  |  a minor  |  d minor  |  g major"
                "\n\n(press x and try for yourself!)"
            ),
        },
        {
            "header": "the circle of fifths",
            "body": (
                "each chord moves in a perfect fifth. if you look closely, "
                "each root note is actually a fourth apart from the next. "
                "c is four intervals away from f, f is four intervals away "
                "from b, and so on. it sounds like a lot, but once you play "
                "through it a few times, it will begin to feel intuitive."
            ),
        },
        {
            "header": "the two-five-one",
            "body": (
                "the two-five-one is a sequence pulled from the end of the "
                "circle of fifths. in c major, that's d minor  |  g major  |  "
                "c major. these three chords flow into each other beautifully "
                "and form the strong backbone of jazz. start here, get "
                "comfortable, then work your way through the rest of the "
                "circle.\n\n"
                "the circle of fifths can be played in any key and in many "
                "different ways. keep going to continue learning."
            ),
        },
    ],
}

LESSON_TITLES: dict[int, str] = {
    0: "introduction to chords",
    1: "chord progressions & the circle of fifths",
}

# ── Chord progressions directory ───────────────────────────────────────────────
# Each entry: title (lowercase display) + sequence string shown in practice box.

PROGRESSIONS: dict[int, dict[str, str]] = {
    1: {
        "title":    "circle of fifths",
        "sequence": "C + Maj7  |  F + Maj7  |  B + Dim  |  E + 7  |  A + Min7  |  D + Min7  |  G + 7",
    },
    2: {
        "title":    "two five one",
        "sequence": "D + Min7  |  G + 7  |  C + Maj7",
    },
    3: {
        "title":    "falling two five one",
        "sequence": (
            "D + Min  |  G + 7  |  C + Maj7  |  C + Min  |  F + 7  |"
            "  Bb + Maj7  |  Bb + Min  |  Eb + 7  |  Ab + Maj7"
        ),
    },
}

# ── State ──────────────────────────────────────────────────────────────────────

@dataclass
class TheoryState:
    screen:      str = "SELECTION"   # "SELECTION" | "LESSON" | "PROGRESSIONS"
    lesson_idx:  int = 0
    step_idx:    int = 0
    left_dwell:  int = 0             # frames finger has been near left arrow
    right_dwell: int = 0             # frames finger has been near right arrow


def navigate(state: TheoryState, direction: int) -> None:
    """Advance (+1) or rewind (-1) through lesson steps.

    Crossing a boundary returns to the SELECTION screen.
    """
    steps   = LESSONS[state.lesson_idx]
    new_idx = state.step_idx + direction
    if new_idx < 0 or new_idx >= len(steps):
        state.screen   = "SELECTION"
        state.step_idx = 0
    else:
        state.step_idx = new_idx
    state.left_dwell  = 0
    state.right_dwell = 0


# ── Geometry ───────────────────────────────────────────────────────────────────

def _card_rect(fw: int, fh: int) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) of the centred modal card."""
    cw = int(fw * CARD_W_FRAC)
    ch = int(fh * CARD_H_FRAC)
    x1 = (fw - cw) // 2
    y1 = (fh - ch) // 2
    return x1, y1, x1 + cw, y1 + ch


def _arrow_centres(
    x1: int, y1: int, x2: int, y2: int, pad: int = 50
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (left_centre, right_centre) for the navigation arrows."""
    return (x1 + pad, y2 - pad), (x2 - pad, y2 - pad)


# ── Dwell detection ────────────────────────────────────────────────────────────

def update_dwell(
    state:       TheoryState,
    finger_tips: list[tuple[int, int]],
    fw:          int,
    fh:          int,
) -> tuple[bool, bool]:
    """Increment/reset per-frame dwell counters.

    Returns (left_fired, right_fired) — True when the threshold is crossed.
    Counters reset to 0 immediately after firing so the next dwell is fresh.
    """
    if state.screen != "LESSON":
        state.left_dwell = state.right_dwell = 0
        return False, False

    x1, y1, x2, y2 = _card_rect(fw, fh)
    (lx, ly), (rx, ry) = _arrow_centres(x1, y1, x2, y2)

    near_l = any(math.hypot(tx - lx, ty - ly) <= ARROW_RADIUS for tx, ty in finger_tips)
    near_r = any(math.hypot(tx - rx, ty - ry) <= ARROW_RADIUS for tx, ty in finger_tips)

    state.left_dwell  = (state.left_dwell  + 1) if near_l else 0
    state.right_dwell = (state.right_dwell + 1) if near_r else 0

    left_fired  = state.left_dwell  >= DWELL_FRAMES
    right_fired = state.right_dwell >= DWELL_FRAMES
    if left_fired:
        state.left_dwell  = 0
    if right_fired:
        state.right_dwell = 0
    return left_fired, right_fired


# ── Text helpers ───────────────────────────────────────────────────────────────

def _wrap_text(
    text:      str,
    font:      ImageFont.FreeTypeFont,
    max_w_px:  int,
    draw:      ImageDraw.ImageDraw,
    stroke:    int = 0,
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


# ── Main draw entry ────────────────────────────────────────────────────────────

def draw_theory_overlay(
    frame:       np.ndarray,
    state:       TheoryState,
    finger_tips: list[tuple[int, int]],
) -> np.ndarray:
    """Render the theory modal on top of frame and return the result."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = _card_rect(fw, fh)

    # Dim the background
    dim = frame.copy()
    cv2.rectangle(dim, (0, 0), (fw, fh), (0, 0, 0), -1)
    frame = cv2.addWeighted(dim, DIM_ALPHA, frame, 1.0 - DIM_ALPHA, 0)

    # White glass card with rounded corners — draw via PIL RGBA composite
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
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
    pil  = Image.alpha_composite(pil, card_layer).convert("RGB")
    draw = ImageDraw.Draw(pil)

    if state.screen == "SELECTION":
        _draw_selection(draw, x1, y1, x2, y2)
    elif state.screen == "PROGRESSIONS":
        _draw_progressions(draw, x1, y1, x2, y2)
    else:
        _draw_lesson(draw, state, finger_tips, x1, y1, x2, y2)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── Selection screen ───────────────────────────────────────────────────────────

def _draw_selection(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
) -> None:
    pad = 52
    cx  = (x1 + x2) // 2

    # Title
    tb     = draw.textbbox((0, 0), "theory", font=_FONT_TITLE, stroke_width=1)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ty     = y1 + pad
    draw.text(
        (cx - tw // 2, ty), "theory",
        font=_FONT_TITLE, fill=_BLACK, stroke_width=1, stroke_fill=_BLACK,
    )

    # Separator
    sep_y = ty + th + 18
    draw.line([(x1 + pad, sep_y), (x2 - pad, sep_y)], fill=_LGRAY, width=1)

    # Lesson rows — period format: "0. title"
    rows = [
        ("0. introduction to chords",                    _BLACK),
        ("1. chord progressions & the circle of fifths", _BLACK),
        ("2. coming soon",                               _GRAY),
        ("3. coming soon",                               _GRAY),
    ]
    iy = sep_y + 26
    rh = _line_h(_FONT_ITEM, draw) + 18
    for text, colour in rows:
        draw.text((x1 + pad, iy), text, font=_FONT_ITEM, fill=colour)
        iy += rh

    # Separator before progressions row
    iy += 4
    draw.line([(x1 + pad, iy), (x2 - pad, iy)], fill=_LGRAY, width=1)
    iy += 14

    # Progressions shortcut row
    draw.text((x1 + pad, iy), "+  chord progressions", font=_FONT_ITEM, fill=_BLACK)

    # Hint
    hint = "press a number to select   |   +  for progressions   |   t to close"
    hb   = draw.textbbox((0, 0), hint, font=_FONT_HINT)
    hw   = hb[2] - hb[0]
    hh   = hb[3] - hb[1]
    draw.text(
        (cx - hw // 2, y2 - pad - hh),
        hint, font=_FONT_HINT, fill=_DGRAY,
    )


# ── Chord progressions directory screen ───────────────────────────────────────

def _draw_progressions(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
) -> None:
    pad = 52
    cx  = (x1 + x2) // 2

    # Title
    tb     = draw.textbbox((0, 0), "chord progressions", font=_FONT_TITLE, stroke_width=1)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ty     = y1 + pad
    draw.text(
        (cx - tw // 2, ty), "chord progressions",
        font=_FONT_TITLE, fill=_BLACK, stroke_width=1, stroke_fill=_BLACK,
    )

    # Separator
    sep_y = ty + th + 18
    draw.line([(x1 + pad, sep_y), (x2 - pad, sep_y)], fill=_LGRAY, width=1)

    # Progression rows
    iy = sep_y + 26
    rh = _line_h(_FONT_ITEM, draw) + 18
    for idx, prog in PROGRESSIONS.items():
        draw.text(
            (x1 + pad, iy),
            f"{idx}. {prog['title']}",
            font=_FONT_ITEM, fill=_BLACK,
        )
        iy += rh

    # Hint
    hint = "press a number to practice   |   t to close"
    hb   = draw.textbbox((0, 0), hint, font=_FONT_HINT)
    hw   = hb[2] - hb[0]
    hh   = hb[3] - hb[1]
    draw.text(
        (cx - hw // 2, y2 - pad - hh),
        hint, font=_FONT_HINT, fill=_DGRAY,
    )


# ── Lesson content screen ──────────────────────────────────────────────────────

def _draw_lesson(
    draw:        ImageDraw.ImageDraw,
    state:       TheoryState,
    finger_tips: list[tuple[int, int]],
    x1: int, y1: int, x2: int, y2: int,
) -> None:
    pad      = 52
    cx       = (x1 + x2) // 2
    inner_w  = (x2 - x1) - pad * 2
    step     = LESSONS[state.lesson_idx][state.step_idx]
    steps    = LESSONS[state.lesson_idx]

    (lx, ly), (rx, ry) = _arrow_centres(x1, y1, x2, y2)
    body_max_y = ly - 24   # stop body text before arrow row

    # ── Header ────────────────────────────────────────────────────────────
    hb     = draw.textbbox((0, 0), step["header"], font=_FONT_HEADER, stroke_width=1)
    hw, hh = hb[2] - hb[0], hb[3] - hb[1]
    hy     = y1 + pad
    draw.text(
        (cx - hw // 2, hy), step["header"],
        font=_FONT_HEADER, fill=_BLACK, stroke_width=1, stroke_fill=_BLACK,
    )

    # Separator
    sep_y = hy + hh + 14
    draw.line([(x1 + pad, sep_y), (x2 - pad, sep_y)], fill=_LGRAY, width=1)

    # ── Body — vertically centred in available space ───────────────────────
    lines  = _wrap_text(step["body"], _FONT_BODY, inner_w, draw)
    lh     = _line_h(_FONT_BODY, draw) + 6
    avail_h      = body_max_y - (sep_y + 20)
    total_body_h = len(lines) * lh
    body_y = sep_y + 20 + max(0, (avail_h - total_body_h) // 2)

    for line in lines:
        if body_y + lh > body_max_y:
            break
        if line:
            draw.text((x1 + pad, body_y), line, font=_FONT_BODY, fill=_BLACK)
        body_y += lh

    # ── Navigation arrows ─────────────────────────────────────────────────
    left_col = _BLACK if state.step_idx > 0 else _LGRAY
    _draw_arrow(draw, lx, ly, "<", _FONT_ARROW, left_col, bold=True)
    _draw_arrow(draw, rx, ry, ">", _FONT_ARROW, _BLACK,   bold=True)

    # Dwell progress arcs
    if state.left_dwell > 0 and state.step_idx > 0:
        _draw_dwell_arc(draw, lx, ly, state.left_dwell / DWELL_FRAMES)
    if state.right_dwell > 0:
        _draw_dwell_arc(draw, rx, ry, state.right_dwell / DWELL_FRAMES)

    # ── Step counter (centred between arrows) ─────────────────────────────
    counter = f"{state.step_idx + 1} / {len(steps)}"
    cb      = draw.textbbox((0, 0), counter, font=_FONT_HINT)
    cw_     = cb[2] - cb[0]
    ch_     = cb[3] - cb[1]
    draw.text(
        (cx - cw_ // 2, ly - ch_ // 2),
        counter, font=_FONT_HINT, fill=_DGRAY,
    )


# ── Arrow + dwell ring helpers ─────────────────────────────────────────────────

def _draw_arrow(
    draw:   ImageDraw.ImageDraw,
    cx:     int,
    cy:     int,
    glyph:  str,
    font:   ImageFont.FreeTypeFont,
    colour: tuple[int, int, int, int],
    bold:   bool = False,
) -> None:
    stroke = 1 if bold else 0
    bb     = draw.textbbox((0, 0), glyph, font=font, stroke_width=stroke)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(
        (cx - tw // 2, cy - th // 2), glyph,
        font=font, fill=colour, stroke_width=stroke, stroke_fill=colour,
    )


def _draw_dwell_arc(
    draw: ImageDraw.ImageDraw,
    cx:   int,
    cy:   int,
    frac: float,
    r:    int = ARROW_RADIUS + 8,
) -> None:
    """Thin arc that fills clockwise as dwell progresses."""
    end_a = -90 + frac * 360
    bbox  = [cx - r, cy - r, cx + r, cy + r]
    draw.arc(bbox, start=-90, end=end_a, fill=_GOLD, width=3)


# ── Practice box ──────────────────────────────────────────────────────────────

_PRACTICE_HINT = "x to cancel  |  t to go back to theory"


def draw_practice_box(
    frame:    np.ndarray,
    sequence: str = "",
) -> np.ndarray:
    """Overlay a translucent practice reference box below the chord circles.

    The box sits in the band between the circle bottoms and the footer strip,
    horizontally spanning most of the frame width.  White text on a dark
    semi-transparent background gives clear contrast over the live camera feed.
    """
    fh, fw = frame.shape[:2]

    # Vertical bounds — circle bottom → footer top
    box_top    = fh // 2 + RADIUS + 12
    box_bottom = fh - _FOOTER_H - 12
    box_x1     = int(fw * 0.08)
    box_x2     = int(fw * 0.92)

    if box_bottom <= box_top + 20:
        return frame   # not enough room; skip drawing

    # Translucent dark background via PIL RGBA composite
    pil        = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    box_layer  = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    box_draw   = ImageDraw.Draw(box_layer)
    box_draw.rounded_rectangle(
        [box_x1, box_top, box_x2, box_bottom],
        radius=10,
        fill=(0, 0, 0, 200),
    )
    pil  = Image.alpha_composite(pil, box_layer).convert("RGB")
    draw = ImageDraw.Draw(pil)

    # Layout — sequence text centred, hint at bottom
    inner_w = box_x2 - box_x1 - 48
    cx      = (box_x1 + box_x2) // 2

    seq_lines = _wrap_text(sequence or PROGRESSIONS[1]["sequence"], _FONT_PRAC, inner_w, draw)
    seq_lh    = _line_h(_FONT_PRAC, draw) + 6

    hint_bb = draw.textbbox((0, 0), _PRACTICE_HINT, font=_FONT_PRAC_H)
    hint_h  = hint_bb[3] - hint_bb[1]
    hint_w  = hint_bb[2] - hint_bb[0]

    total_h = len(seq_lines) * seq_lh + 14 + hint_h
    box_mid = (box_top + box_bottom) // 2
    seq_y   = box_mid - total_h // 2

    for line in seq_lines:
        if line:
            lb  = draw.textbbox((0, 0), line, font=_FONT_PRAC)
            lw  = lb[2] - lb[0]
            draw.text((cx - lw // 2, seq_y), line, font=_FONT_PRAC, fill=_WHITE)
        seq_y += seq_lh

    hint_y = seq_y + 10
    draw.text((cx - hint_w // 2, hint_y), _PRACTICE_HINT, font=_FONT_PRAC_H, fill=_WHITE)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
