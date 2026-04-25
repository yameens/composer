"""
Chunk 1 — Hand Tracking & Visualization (MediaPipe Tasks API)
Run this file directly to test: python hand_tracker.py
Press Q to quit.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Constants ─────────────────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).parent / "assets" / "hand_landmarker.task"

PINCH_THRESHOLD_PX  = 28   # fingertips must actually touch
LANDMARK_DOT_RADIUS = 5
ACTIVE_TIP_RADIUS   = 12

# Hand bone connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17), (2, 5), (5, 9), (9, 13), (13, 17),  # palm
]

# Per-hand colours (BGR)
COLOUR = {
    "Left":  {"line": (255, 180,  60), "dot": (255, 220, 120), "tip": ( 60, 220, 255)},
    "Right": {"line": ( 60, 180, 255), "dot": (120, 200, 255), "tip": (255,  80, 160)},
}

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class HandData:
    label:      str                       # 'Left' or 'Right'
    index_tip:  tuple[int, int]           # landmark 8  (pixel coords)
    middle_tip: tuple[int, int]           # landmark 12 (pixel coords)
    landmarks:  list[tuple[int, int]]     # all 21 landmarks (pixel coords)
    pinch_active: bool = False

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

    def process(self, frame_bgr: "np.ndarray") -> tuple["np.ndarray", list[HandData]]:
        """
        Process a BGR frame (already mirrored).
        Returns (annotated_frame, list[HandData]).
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - self._start_ms) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        hand_data_list: list[HandData] = []

        for landmarks_norm in result.hand_landmarks:
            # Pixel coords first so we can use wrist position
            lm_pixels: list[tuple[int, int]] = [
                (int(lm.x * w), int(lm.y * h))
                for lm in landmarks_norm
            ]

            # Position-based classification: after cv2.flip, wrist left of
            # centre = user's left hand, wrist right of centre = user's right.
            # This is more reliable than MediaPipe's anatomical classifier on
            # a mirrored frame.
            wrist_x = lm_pixels[0][0]
            label   = "Left" if wrist_x < w // 2 else "Right"
            col     = COLOUR[label]

            index_tip  = lm_pixels[8]
            middle_tip = lm_pixels[12]   # kept for HandData API compatibility

            # Draw bone connections
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame_bgr, lm_pixels[a], lm_pixels[b],
                         col["line"], 2, cv2.LINE_AA)

            # Draw all 21 landmark dots (including middle tip as normal dot)
            for i, pt in enumerate(lm_pixels):
                if i == 8:
                    continue  # drawn separately below
                cv2.circle(frame_bgr, pt, LANDMARK_DOT_RADIUS,
                           col["dot"], -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pt, LANDMARK_DOT_RADIUS,
                           (255, 255, 255), 1, cv2.LINE_AA)

            # Overdraw index tip only — the active pointer
            cv2.circle(frame_bgr, index_tip, ACTIVE_TIP_RADIUS,
                       col["tip"], -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, index_tip, ACTIVE_TIP_RADIUS,
                       (255, 255, 255), 2, cv2.LINE_AA)

            # Wrist label
            wrist = lm_pixels[0]
            cv2.putText(
                frame_bgr, label,
                (wrist[0] - 20, wrist[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col["dot"], 2, cv2.LINE_AA,
            )

            hand_data_list.append(HandData(
                label=label,
                index_tip=index_tip,
                middle_tip=middle_tip,
                landmarks=lm_pixels,
            ))

        return frame_bgr, hand_data_list

    def get_hand(self, hands: list[HandData], label: str) -> Optional[HandData]:
        for h in hands:
            if h.label == label:
                return h
        return None

    def close(self) -> None:
        self._detector.close()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

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

        y = 30
        for hd in hands:
            cv2.putText(frame,
                f"{hd.label}: index={hd.index_tip}  pinch={'YES' if hd.pinch_active else 'no'}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22

        cv2.imshow("Conductor — Chunk 1: Hand Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
