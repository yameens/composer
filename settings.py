"""
Persistent user settings for Conductor.

Stores tracker_color, circle_radius, and master_volume in settings.json
alongside patches.json.  Tolerates a missing or corrupt file by returning
defaults.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_SETTINGS_PATH = Path(__file__).parent / "settings.json"


@dataclass
class Settings:
    tracker_color: str   = "shadow"   # key into NAMED_COLORS in ui_settings.py
    circle_radius: int   = 190        # pixels — SLIDER_MIN_RADIUS..SLIDER_MAX_RADIUS
    master_volume: float = 0.7        # 0..1
    resolution:    float = 1.0        # 0..1; 1.0 = full (no pixelation), lower = more pixelated


def load_settings() -> Settings:
    """Read settings.json; return defaults on any error."""
    try:
        if _SETTINGS_PATH.exists():
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return Settings(
                    tracker_color  = str(raw.get("tracker_color", "shadow")),
                    circle_radius  = int(raw.get("circle_radius", 190)),
                    master_volume  = float(raw.get("master_volume", 0.7)),
                    resolution     = max(0.0, min(1.0, float(raw.get("resolution", 1.0)))),
                )
    except Exception:
        pass
    return Settings()


def save_settings(s: Settings) -> None:
    """Write settings to settings.json."""
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(s), f, indent=2)
