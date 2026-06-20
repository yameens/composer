"""
Chunk 1 — Hand Tracking & Visualization (MediaPipe Tasks API)
Run this file directly to test: python hand_tracker.py
Press Q to quit.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ui_circles import _draw_text_pil, _make_text_sprite, _blit_text_centered, app_font

_FONT_WRIST = app_font(20)
_FONT_DEBUG = app_font(18)

# Cached wrist label sprites — "left" / "right" never change, so we render once
# into BGRA sprites at module init and just alpha-blit them onto the frame each
# tick instead of paying ~2 full-frame BGR↔RGB PIL roundtrips per hand.
_WRIST_SPRITE_CACHE: dict[tuple[str, tuple[int, int, int]], tuple] = {}


def _get_wrist_sprite(label: str, color_bgr: tuple[int, int, int]) -> tuple:
    key = (label, color_bgr)
    ts = _WRIST_SPRITE_CACHE.get(key)
    if ts is None:
        ts = _make_text_sprite(label, _FONT_WRIST, color_bgr)
        _WRIST_SPRITE_CACHE[key] = ts
    return ts

# ── Constants ─────────────────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).parent / "assets" / "hand_landmarker.task"

# Index–middle pinch in world space (meters) when hand_world_landmarks is available
WORLD_PINCH_FLOOR_M      = 0.028   # max tip separation (index 8 ↔ middle 12) for “touching”
WORLD_PINCH_PALM_FRAC    = 0.15    # threshold scales with wrist→middle-MCP length

# Fallback: normalized landmark x,y,z
NORM_PINCH_FLOOR         = 0.035
NORM_PINCH_PALM_FRAC     = 0.15

# Pointing guard: index MCP→tip should be long enough we are not a closed fist
WORLD_INDEX_EXTEND_FLOOR_M   = 0.028
WORLD_INDEX_EXTEND_PALM_FRAC = 0.32
NORM_INDEX_EXTEND_FLOOR      = 0.042
NORM_INDEX_EXTEND_PALM_FRAC  = 0.32

# Ring-cluster (b9 gesture): middle tip (12) ↔ ring tip (16)
WORLD_RING_FLOOR_M   = 0.030
WORLD_RING_PALM_FRAC = 0.18
NORM_RING_FLOOR      = 0.040
NORM_RING_PALM_FRAC  = 0.18

PINCH_ON_FRAMES        = 2
PINCH_OFF_FRAMES       = 2

# If the wrist (landmark 0) moves more than this many pixels between consecutive
# frames, treat the hand as freshly re-acquired and skip EMA smoothing for one
# frame.  Prevents EMA "snap" artefacts when Left/Right labels swap as hands
# cross the midline, or when a hand re-enters the frame in a different spot.
_LM_EMA_RESET_DIST     = 80

INFERENCE_SCALE = 0.5   # run MediaPipe on this fraction of the full frame size
LANDMARK_DOT_RADIUS = 5
ACTIVE_TIP_RADIUS   = 12
PINCH_COLOUR        = (0, 215, 255)    # BGR gold — pinch highlight

# Hand bone connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17), (2, 5), (5, 9), (9, 13), (13, 17),  # palm
]

# Per-hand colours (BGR) — black/dark theme
COLOUR = {
    "Left":  {"line": (30, 30, 30), "dot": (55, 55, 55), "tip": (55, 55, 55)},
    "Right": {"line": (30, 30, 30), "dot": (55, 55, 55), "tip": (55, 55, 55)},
}

# ── Landmark colour API ───────────────────────────────────────────────────────

def set_landmark_colour(bgr: tuple[int, int, int]) -> None:
    """Set the dot/tip colour for both hands and derive a darkened line colour.

    PINCH_COLOUR (gold) is left unchanged.
    """
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    # Darken for the connecting lines (multiply by ~0.45)
    line_bgr = (b * 45 // 100, g * 45 // 100, r * 45 // 100)
    for hand in ("Left", "Right"):
        COLOUR[hand]["dot"]  = bgr
        COLOUR[hand]["tip"]  = bgr
        COLOUR[hand]["line"] = line_bgr
    # Invalidate wrist sprite cache so it rebuilds at the new colour
    _WRIST_SPRITE_CACHE.clear()


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class HandData:
    label:        str                     # screen-side: Left = left half / leftmost of two
    index_tip:    tuple[int, int]         # landmark 8  (pixel coords)
    middle_tip:   tuple[int, int]         # landmark 12 (pixel coords)
    ring_tip:     tuple[int, int]         # landmark 16 (pixel coords)
    landmarks:    list[tuple[int, int]]   # all 21 landmarks (pixel coords)
    pinch_active:   bool = False          # debounced index+middle pinch (+ pointing/curl guard)
    pinch_triple:   bool = False          # index+middle+ring cluster — b9 on right hand
    index_extended: bool = False          # True when index MCP→tip length clears the fist threshold


@dataclass
class _DebouncedSignal:
    """Frame-count hysteresis so brief threshold flicker does not toggle output."""
    stable: bool = False
    on_streak: int = 0
    off_streak: int = 0


# ── Tracker ───────────────────────────────────────────────────────────────────

class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        detection_conf: float = 0.7,
        presence_conf:  float = 0.7,
        tracking_conf:  float = 0.6,
    ):
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {_MODEL_PATH}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=presence_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._start_ms = time.monotonic()
        self._last_ts_ms: int = -1

        self._debounce_pinch = {
            "Left":  {"active": _DebouncedSignal(), "triple": _DebouncedSignal()},
            "Right": {"active": _DebouncedSignal(), "triple": _DebouncedSignal()},
        }

        # Per-hand exponential moving average on the 21 pixel-space landmarks.
        # alpha = 0.55 means "55% new + 45% previous" — kills high-frequency
        # MediaPipe jitter (which causes hover flicker on segment boundaries)
        # without adding noticeable latency. State resets on hand loss or when
        # the wrist jumps more than _LM_EMA_RESET_DIST px (handles label swaps
        # when both hands cross the screen midline).
        self._lm_ema_alpha: float = 0.55
        self._lm_ema_prev: dict[str, Optional[list[tuple[int, int]]]] = {
            "Left":  None,
            "Right": None,
        }

    def process(self, frame_bgr: Any, draw_skeleton: bool = True) -> tuple[Any, list[HandData]]:
        """
        Process a BGR frame (already mirrored).
        Left/Right are by screen position: two hands → leftmost wrist = Left;
        one hand → Left if wrist on left half of frame, else Right.
        Returns (annotated_frame, list[HandData]).

        draw_skeleton : when False, bone lines, landmark dots, pinch highlights,
                        and wrist labels are all suppressed.  HandData is still
                        built and returned for every detected hand.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if INFERENCE_SCALE < 1.0:
            small = cv2.resize(rgb, (int(w * INFERENCE_SCALE), int(h * INFERENCE_SCALE)))
        else:
            small = rgb
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)

        timestamp_ms = max(int((time.monotonic() - self._start_ms) * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = timestamp_ms
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        hand_data_list: list[HandData] = []
        world_list: list[Any] = list(result.hand_world_landmarks or [])

        entries: list[tuple[int, int, list, list[tuple[int, int]], Any]] = []
        for i, landmarks_norm in enumerate(result.hand_landmarks):
            lm_pixels: list[tuple[int, int]] = [
                (int(lm.x * w), int(lm.y * h))
                for lm in landmarks_norm
            ]
            wx = lm_pixels[0][0]
            wl = (
                world_list[i]
                if i < len(world_list) and world_list[i] and len(world_list[i]) >= 21
                else None
            )
            entries.append((wx, i, landmarks_norm, lm_pixels, wl))

        entries.sort(key=lambda e: e[0])
        labeled: list[tuple[tuple, str]] = []
        if len(entries) == 1:
            wx, *_rest = entries[0]
            lbl = "Left" if wx < w // 2 else "Right"
            labeled.append((entries[0], lbl))
        elif len(entries) >= 2:
            labeled.append((entries[0], "Left"))
            labeled.append((entries[1], "Right"))

        seen_labels: set[str] = set()

        for (_wx, _i, landmarks_norm, lm_pixels, world_chunk), label in labeled:
            seen_labels.add(label)
            col = COLOUR[label]

            # ── Landmark EMA smoothing ────────────────────────────────────
            # Smoothing happens in pixel space only — pinch/world-distance
            # detection still runs on raw MediaPipe landmarks below.
            prev_smoothed = self._lm_ema_prev[label]
            if prev_smoothed is not None and len(prev_smoothed) == len(lm_pixels):
                pwx, pwy = prev_smoothed[0]
                cwx, cwy = lm_pixels[0]
                if math.hypot(cwx - pwx, cwy - pwy) > _LM_EMA_RESET_DIST:
                    prev_smoothed = None
            if prev_smoothed is None:
                smoothed = lm_pixels
            else:
                a = self._lm_ema_alpha
                inv = 1.0 - a
                smoothed = [
                    (int(a * x + inv * px), int(a * y + inv * py))
                    for (x, y), (px, py) in zip(lm_pixels, prev_smoothed)
                ]
            self._lm_ema_prev[label] = smoothed
            lm_pixels = smoothed

            index_tip  = lm_pixels[8]
            middle_tip = lm_pixels[12]
            ring_tip   = lm_pixels[16]

            if world_chunk is not None:
                metrics = _pinch_metrics_world(world_chunk)
            else:
                metrics = _pinch_metrics_norm(landmarks_norm)

            d_im, d_mr, thresh_im, thresh_ring, pinch_ok = metrics

            raw_pinch  = (d_im < thresh_im) and pinch_ok
            raw_triple = raw_pinch and (d_mr < thresh_ring) and label == "Right"

            deb = self._debounce_pinch[label]
            pinch_active = _debounce_signal(
                deb["active"], raw_pinch, PINCH_ON_FRAMES, PINCH_OFF_FRAMES,
            )
            pinch_triple = _debounce_signal(
                deb["triple"], raw_triple, PINCH_ON_FRAMES, PINCH_OFF_FRAMES,
            )

            if draw_skeleton:
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame_bgr, lm_pixels[a], lm_pixels[b],
                             col["line"], 2, cv2.LINE_8)

                if pinch_triple:
                    skip = {8, 12, 16}
                elif pinch_active:
                    skip = {8, 12}
                else:
                    skip = {8}
                for li, pt in enumerate(lm_pixels):
                    if li in skip:
                        continue
                    cv2.circle(frame_bgr, pt, LANDMARK_DOT_RADIUS,
                               col["dot"], -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, pt, LANDMARK_DOT_RADIUS,
                               (255, 255, 255), 1, cv2.LINE_AA)

                if pinch_triple:
                    for pt in (index_tip, middle_tip, ring_tip):
                        cv2.circle(frame_bgr, pt, ACTIVE_TIP_RADIUS,
                                   PINCH_COLOUR, -1, cv2.LINE_AA)
                        cv2.circle(frame_bgr, pt, ACTIVE_TIP_RADIUS,
                                   (255, 255, 255), 2, cv2.LINE_AA)
                elif pinch_active:
                    for pt in (index_tip, middle_tip):
                        cv2.circle(frame_bgr, pt, ACTIVE_TIP_RADIUS,
                                   PINCH_COLOUR, -1, cv2.LINE_AA)
                        cv2.circle(frame_bgr, pt, ACTIVE_TIP_RADIUS,
                                   (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.circle(frame_bgr, index_tip, ACTIVE_TIP_RADIUS,
                               col["tip"], -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, index_tip, ACTIVE_TIP_RADIUS,
                               (255, 255, 255), 2, cv2.LINE_AA)

                wrist = lm_pixels[0]
                wrist_ts = _get_wrist_sprite(label.lower(), tuple(col["dot"]))
                _blit_text_centered(frame_bgr, wrist_ts, wrist[0], wrist[1] + 26)

            hand_data_list.append(HandData(
                label=label,
                index_tip=index_tip,
                middle_tip=middle_tip,
                ring_tip=ring_tip,
                landmarks=lm_pixels,
                pinch_active=pinch_active,
                pinch_triple=pinch_triple,
                index_extended=pinch_ok,
            ))

        for lbl in ("Left", "Right"):
            if lbl not in seen_labels:
                _reset_debounced_signal(self._debounce_pinch[lbl]["active"])
                _reset_debounced_signal(self._debounce_pinch[lbl]["triple"])
                self._lm_ema_prev[lbl] = None

        return frame_bgr, hand_data_list

    def get_hand(self, hands: list[HandData], label: str) -> Optional[HandData]:
        for h in hands:
            if h.label == label:
                return h
        return None

    def close(self) -> None:
        self._detector.close()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _lm_xyz_dist(a: object, b: object) -> float:
    """3D distance between landmarks with .x .y .z (world or normalized)."""
    return math.sqrt(
        (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
    )


def _pinch_metrics_world(world_lms: list) -> tuple[float, float, float, float, bool]:
    """
    Returns (d_im, d_mr, thresh_im, thresh_ring, pinch_ok).
    d_im  = index tip (8) ↔ middle tip (12) distance (IM pinch).
    d_mr  = middle tip (12) ↔ ring tip (16) distance (ring-cluster b9).
    pinch_ok = index extended enough that we are not in a closed fist.
    """
    palm = _lm_xyz_dist(world_lms[0], world_lms[9])
    thresh_im   = max(WORLD_PINCH_FLOOR_M, WORLD_PINCH_PALM_FRAC * palm)
    thresh_ring = max(WORLD_RING_FLOOR_M,  WORLD_RING_PALM_FRAC  * palm)
    d_im = _lm_xyz_dist(world_lms[8],  world_lms[12])
    d_mr = _lm_xyz_dist(world_lms[12], world_lms[16])
    idx_len = _lm_xyz_dist(world_lms[5], world_lms[8])
    extend_min = max(
        WORLD_INDEX_EXTEND_FLOOR_M,
        WORLD_INDEX_EXTEND_PALM_FRAC * palm,
    )
    index_ok = (idx_len >= extend_min) and (world_lms[8].z < world_lms[6].z)
    pinch_ok = index_ok
    return d_im, d_mr, thresh_im, thresh_ring, pinch_ok


def _pinch_metrics_norm(norm_lms: list) -> tuple[float, float, float, float, bool]:
    palm = _lm_xyz_dist(norm_lms[0], norm_lms[9])
    thresh_im   = max(NORM_PINCH_FLOOR, NORM_PINCH_PALM_FRAC * palm)
    thresh_ring = max(NORM_RING_FLOOR,  NORM_RING_PALM_FRAC  * palm)
    d_im = _lm_xyz_dist(norm_lms[8],  norm_lms[12])
    d_mr = _lm_xyz_dist(norm_lms[12], norm_lms[16])
    idx_len = _lm_xyz_dist(norm_lms[5], norm_lms[8])
    extend_min = max(
        NORM_INDEX_EXTEND_FLOOR,
        NORM_INDEX_EXTEND_PALM_FRAC * palm,
    )
    index_ok = (idx_len >= extend_min) and (norm_lms[8].y < norm_lms[6].y)
    pinch_ok = index_ok
    return d_im, d_mr, thresh_im, thresh_ring, pinch_ok


def _debounce_signal(state: _DebouncedSignal, raw: bool, on_n: int, off_n: int) -> bool:
    """Return hysteresis-stable boolean from noisy raw input."""
    if raw:
        state.off_streak = 0
        if not state.stable:
            state.on_streak += 1
            if state.on_streak >= on_n:
                state.stable = True
                state.on_streak = 0
    else:
        state.on_streak = 0
        if state.stable:
            state.off_streak += 1
            if state.off_streak >= off_n:
                state.stable = False
                state.off_streak = 0
    return state.stable


def _reset_debounced_signal(state: _DebouncedSignal) -> None:
    state.stable = False
    state.on_streak = 0
    state.off_streak = 0


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os as _os, platform as _platform, subprocess as _subprocess, re as _re

    def _pick_cam() -> int:
        raw = _os.environ.get("CONDUCTOR_CAM_INDEX", "").strip()
        if raw.isdigit():
            return int(raw)
        if _platform.system() != "Darwin":
            return 0
        try:
            proc = _subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True, text=True, timeout=8, check=False,
            )
            lines = (proc.stderr or "").splitlines()
        except (FileNotFoundError, _subprocess.TimeoutExpired):
            return 0
        section = False
        devices: list[tuple[int, str]] = []
        for line in lines:
            if "AVFoundation video devices" in line:
                section = True; continue
            if "AVFoundation audio devices" in line:
                break
            if not section:
                continue
            m = _re.search(r"\[(\d+)\]\s+(.*)", line)
            if m:
                devices.append((int(m.group(1)), m.group(2).strip()))
        has_iphone = any(
            "iphone" in n.lower() or "continuity" in n.lower() for _, n in devices
        )
        if not has_iphone:
            return 0
        # iPhone present — probe OpenCV indices, prefer the narrower (non-iPhone) camera
        probe: list[tuple[int, int]] = []
        for ci in range(4):
            cap_p = cv2.VideoCapture(ci)
            if cap_p.isOpened():
                w = int(cap_p.get(cv2.CAP_PROP_FRAME_WIDTH))
                if w > 0:
                    probe.append((ci, w))
            cap_p.release()
        for ci, w in probe:
            if w < 1920:
                return ci
        return 0

    cap = cv2.VideoCapture(_pick_cam())
    tracker = HandTracker()
    print("Hand Tracker running — press Q to quit")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame, hands = tracker.process(frame)

        y = 28
        for hd in hands:
            line = f"{hd.label.lower()}: pinch={'Y' if hd.pinch_active else 'n'}  b9={'Y' if hd.pinch_triple else 'n'}"
            frame = _draw_text_pil(
                frame,
                line,
                (10, y),
                _FONT_DEBUG,
                (200, 200, 200),
                anchor="tl",
            )
            y += 24

        cv2.imshow("Conductor — Chunk 1: Hand Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
