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

from ui_circles import _draw_text_pil, app_font

_FONT_WRIST = app_font(20)
_FONT_DEBUG = app_font(18)

# ── Constants ─────────────────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).parent / "assets" / "hand_landmarker.task"

# Index–middle pinch in world space (meters) when hand_world_landmarks is available
WORLD_PINCH_FLOOR_M      = 0.028   # max tip separation (index 8 ↔ middle 12) for “touching”
WORLD_PINCH_PALM_FRAC    = 0.15    # threshold scales with wrist→middle-MCP length

# Fallback: normalized landmark x,y,z
NORM_PINCH_FLOOR         = 0.035
NORM_PINCH_PALM_FRAC     = 0.15

# Pointing guard: index MCP→tip should be long enough we are not a closed fist…
WORLD_INDEX_EXTEND_FLOOR_M   = 0.028
WORLD_INDEX_EXTEND_PALM_FRAC = 0.26
NORM_INDEX_EXTEND_FLOOR      = 0.042
NORM_INDEX_EXTEND_PALM_FRAC  = 0.26

# …unless index+middle are already tight (bypass ratio of thresh).
PINCH_TIGHT_BYPASS_FRAC = 0.72

# Middle MCP→tip longer than this ⇒ middle is “pointing” with index → ignore IM pinch unless tight.
WORLD_MIDDLE_MAX_LEN_M    = 0.054
WORLD_MIDDLE_MAX_LEN_FRAC = 0.48
NORM_MIDDLE_MAX_LEN       = 0.078
NORM_MIDDLE_MAX_LEN_FRAC  = 0.48

# b9: index–middle pinch plus middle–ring cluster
TRIPLE_THRESH_MULT = 1.08

PINCH_ON_FRAMES        = 2
PINCH_OFF_FRAMES       = 5
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

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class HandData:
    label:        str                     # screen-side: Left = left half / leftmost of two
    index_tip:    tuple[int, int]         # landmark 8  (pixel coords)
    middle_tip:   tuple[int, int]         # landmark 12 (pixel coords)
    ring_tip:     tuple[int, int]         # landmark 16 (pixel coords)
    landmarks:    list[tuple[int, int]]   # all 21 landmarks (pixel coords)
    pinch_active: bool = False            # debounced index+middle pinch (+ pointing/curl guard)
    pinch_triple: bool = False            # IM pinch + middle–ring tight (b9 on right hand)


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
        self._start_ms = time.time()

        self._debounce_pinch = {
            "Left":  {"active": _DebouncedSignal(), "triple": _DebouncedSignal()},
            "Right": {"active": _DebouncedSignal(), "triple": _DebouncedSignal()},
        }

    def process(self, frame_bgr: Any) -> tuple[Any, list[HandData]]:
        """
        Process a BGR frame (already mirrored).
        Left/Right are by screen position: two hands → leftmost wrist = Left;
        one hand → Left if wrist on left half of frame, else Right.
        Returns (annotated_frame, list[HandData]).
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - self._start_ms) * 1000)
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

            index_tip  = lm_pixels[8]
            middle_tip = lm_pixels[12]
            ring_tip   = lm_pixels[16]

            if world_chunk is not None:
                metrics = _pinch_metrics_world(world_chunk)
            else:
                metrics = _pinch_metrics_norm(landmarks_norm)

            d_im, d_mr, thresh, pinch_ok = metrics
            thr_triple = thresh * TRIPLE_THRESH_MULT

            raw_pinch = (d_im < thresh) and pinch_ok
            raw_triple = raw_pinch and (d_mr < thr_triple)

            deb = self._debounce_pinch[label]
            pinch_active = _debounce_signal(
                deb["active"], raw_pinch, PINCH_ON_FRAMES, PINCH_OFF_FRAMES,
            )
            pinch_triple = _debounce_signal(
                deb["triple"], raw_triple, PINCH_ON_FRAMES, PINCH_OFF_FRAMES,
            )

            for a, b in HAND_CONNECTIONS:
                cv2.line(frame_bgr, lm_pixels[a], lm_pixels[b],
                         col["line"], 2, cv2.LINE_AA)

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
            frame_bgr = _draw_text_pil(
                frame_bgr,
                label,
                (wrist[0], wrist[1] + 26),
                _FONT_WRIST,
                col["dot"],
                anchor="center",
            )

            hand_data_list.append(HandData(
                label=label,
                index_tip=index_tip,
                middle_tip=middle_tip,
                ring_tip=ring_tip,
                landmarks=lm_pixels,
                pinch_active=pinch_active,
                pinch_triple=pinch_triple,
            ))

        for lbl in ("Left", "Right"):
            if lbl not in seen_labels:
                _reset_debounced_signal(self._debounce_pinch[lbl]["active"])
                _reset_debounced_signal(self._debounce_pinch[lbl]["triple"])

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


def _pinch_metrics_world(world_lms: list) -> tuple[float, float, float, bool]:
    """
    Index–middle gap, middle–ring gap, pinch threshold, pinch_ok (guards vs fist / parallel point).
    """
    palm = _lm_xyz_dist(world_lms[0], world_lms[9])
    thresh = max(WORLD_PINCH_FLOOR_M, WORLD_PINCH_PALM_FRAC * palm)
    d_im = _lm_xyz_dist(world_lms[8], world_lms[12])
    d_mr = _lm_xyz_dist(world_lms[12], world_lms[16])
    idx_len = _lm_xyz_dist(world_lms[5], world_lms[8])
    mid_len = _lm_xyz_dist(world_lms[9], world_lms[12])
    extend_min = max(
        WORLD_INDEX_EXTEND_FLOOR_M,
        WORLD_INDEX_EXTEND_PALM_FRAC * palm,
    )
    mid_max = max(WORLD_MIDDLE_MAX_LEN_M, WORLD_MIDDLE_MAX_LEN_FRAC * palm)
    tight = d_im < thresh * PINCH_TIGHT_BYPASS_FRAC
    index_ok = (idx_len >= extend_min) or tight
    middle_ok = (mid_len <= mid_max) or tight
    pinch_ok = index_ok and middle_ok
    return d_im, d_mr, thresh, pinch_ok


def _pinch_metrics_norm(norm_lms: list) -> tuple[float, float, float, bool]:
    palm = _lm_xyz_dist(norm_lms[0], norm_lms[9])
    thresh = max(NORM_PINCH_FLOOR, NORM_PINCH_PALM_FRAC * palm)
    d_im = _lm_xyz_dist(norm_lms[8], norm_lms[12])
    d_mr = _lm_xyz_dist(norm_lms[12], norm_lms[16])
    idx_len = _lm_xyz_dist(norm_lms[5], norm_lms[8])
    mid_len = _lm_xyz_dist(norm_lms[9], norm_lms[12])
    extend_min = max(
        NORM_INDEX_EXTEND_FLOOR,
        NORM_INDEX_EXTEND_PALM_FRAC * palm,
    )
    mid_max = max(NORM_MIDDLE_MAX_LEN, NORM_MIDDLE_MAX_LEN_FRAC * palm)
    tight = d_im < thresh * PINCH_TIGHT_BYPASS_FRAC
    index_ok = (idx_len >= extend_min) or tight
    middle_ok = (mid_len <= mid_max) or tight
    pinch_ok = index_ok and middle_ok
    return d_im, d_mr, thresh, pinch_ok


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
    cap = cv2.VideoCapture(0)
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
            line = f"{hd.label}: pinch={'Y' if hd.pinch_active else 'n'}  b9={'Y' if hd.pinch_triple else 'n'}"
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
