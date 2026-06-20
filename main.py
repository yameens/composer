"""
Conductor — main entry point.
Run: python main.py  |  Press Q or ESC to quit.
"""

import copy
import os
import platform
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

import ui_circles as ui_circles_mod
from hand_tracker import HandTracker, set_landmark_colour as _set_landmark_colour
from settings     import Settings, load_settings, save_settings
from ui_settings  import (
    NAMED_COLORS,
    color_bgr as _settings_color_bgr, color_index as _settings_color_idx,
    draw_settings_overlay, draw_settings_gear, get_hovered_settings_gear,
    settings_hit, settings_slider_value_at, prewarm_settings_sprites,
)
from ui_circles   import (
    draw_circles,
    draw_brand_wordmark,
    angle_to_segment,
    ROOT_LABELS,
    TYPE_LABELS,
    BRAND_TEXT,
    BRAND_LETTER_SPACING,
    _FONT_LG,
    _FONT_SM,
    _FONT_HUD_KEYS,
    _make_text_sprite,
    _blit_text_centered,
    load_accent_font,
    set_circle_size,
)
from ui_buttons      import (draw_beat_button, get_hovered_beat_button,
                              draw_mode_button, get_hovered_mode_button)
from ui_theory    import (TheoryState, draw_theory_overlay, update_dwell, navigate,
                           draw_practice_box, PRACTICE_SLIDE, PROGRESSIONS,
                           prog_move_cursor, prog_switch_tab, prog_selected_sequence)
from chord_engine  import ChordEngine, GM_INSTRUMENTS
from ui_instruments import draw_instrument_picker
from ui_iac_help    import draw_iac_help, prewarm_iac_help
from ui_beats       import draw_beats_menu, BEAT_NAMES
from beat_engine     import BeatEngine
from synth_engine  import PatchStore, OscSpec, WAVE_NAMES as _OSC_WAVE_NAMES
from ui_synth      import (
    draw_synth_editor, draw_sounds_list,
    OSC_PARAMS, GLOBAL_PARAMS, EDITOR_N_ROWS,
    editor_hit, slider_value_at, sounds_hit,
)

# ── Config ─────────────────────────────────────────────────────────────────────

WINDOW_NAME        = "Conductor"
# Fallback when auto-pick fails (Linux / Windows). On macOS we prefer built-in FaceTime over iPhone.
CAM_INDEX          = 0
TARGET_W, TARGET_H = 1280, 720
SILENCE_FRAMES     = 3      # frames both segments invalid before auto-silence

# Left-hand pinch → flatten the root by one semitone
FLAT_MAP = {"A": "Ab", "B": "Bb", "C": "Cb", "D": "Db",
            "E": "Eb", "F": "Fb", "G": "Gb"}

# Bottom key row (Q / T / V / S): flush-right block inset from frame edge
HUD_KEYS_MARGIN_RIGHT = 20
# Translucent strip behind bottom wordmark + key hints
HUD_BOTTOM_STRIP_H     = 52
HUD_BOTTOM_STRIP_ALPHA = 0.24

# Arrow-key raw codes for theory navigation (macOS / Linux / other OpenCV builds)
_LEFT_KEYS  = frozenset({63234, 65361, 2, 81})
_RIGHT_KEYS = frozenset({63235, 65363, 3, 83})
_UP_KEYS    = frozenset({63232, 65362, 0, 82})
_DOWN_KEYS  = frozenset({63233, 65364, 1, 84})



def _prefer_builtin_camera() -> None:
    """Pin macOS AVCaptureDevice.userPreferredCamera to the built-in wide-angle camera.

    This prevents Continuity Camera from hijacking OpenCV's index 0 (the macOS
    default device).  Must be called before any cv2.VideoCapture() call.
    No-op on non-Darwin platforms.  Any failure is silenced so the app still starts.
    """
    if platform.system() != "Darwin":
        return
    try:
        import AVFoundation  # type: ignore

        # Pass 1: use a discovery session constrained to the built-in type.
        device = None
        try:
            session = AVFoundation.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
                [AVFoundation.AVCaptureDeviceTypeBuiltInWideAngleCamera],
                AVFoundation.AVMediaTypeVideo,
                AVFoundation.AVCaptureDevicePositionUnspecified,
            )
            devs = session.devices() if session else []
            if devs:
                device = devs[0]
        except Exception:
            pass

        # Pass 2: iterate all video devices, pick first built-in wide-angle.
        if device is None:
            try:
                all_devs = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
                    AVFoundation.AVMediaTypeVideo
                )
                builtin_type = AVFoundation.AVCaptureDeviceTypeBuiltInWideAngleCamera
                for d in (all_devs or []):
                    if d.deviceType() == builtin_type:
                        device = d
                        break
            except Exception:
                pass

        # Pass 3: name-based fallback.
        if device is None:
            try:
                all_devs = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
                    AVFoundation.AVMediaTypeVideo
                )
                _good = {"facetime", "built-in", "macbook", "hd camera"}
                _bad  = {"iphone", "continuity", "desk view"}
                for d in (all_devs or []):
                    low = (d.localizedName() or "").lower()
                    if any(g in low for g in _good) and not any(b in low for b in _bad):
                        device = d
                        break
            except Exception:
                pass

        if device is None:
            print("  [camera] Could not identify built-in camera — using macOS default.")
            return

        AVFoundation.AVCaptureDevice.setUserPreferredCamera_(device)
        print(f"  Pinned macOS default camera to: {device.localizedName()}")

    except Exception as exc:
        print(f"  [camera] AVFoundation pin skipped ({exc})")


def _resolve_camera_index() -> int:
    """Return the OpenCV camera index to open.

    Honours the CONDUCTOR_CAM_INDEX env override (integer).  Otherwise returns
    CAM_INDEX (0) on all platforms — on macOS the built-in camera has already
    been set as the system default by _prefer_builtin_camera(), so index 0 is
    correct.  A runtime fallback to index+1 lives in main() if the first read
    fails.
    """
    raw = os.environ.get("CONDUCTOR_CAM_INDEX", "").strip()
    if raw.isdigit():
        return int(raw)
    return CAM_INDEX

# ── HUD ────────────────────────────────────────────────────────────────────────
# All static HUD text is rendered into BGRA sprites once and alpha-blitted each
# frame; the only dynamic text is the chord readout + mode badge, both cached
# by content so a typical frame does zero PIL roundtrips.

_HUD_TOP_FILL          = (15, 15, 15)
_HUD_TOP_DARK_ALPHA    = 0.65
_HUD_BOT_FILL          = (14, 16, 20)
_HUD_TOP_FILL_CONST: dict[int, np.ndarray] = {}
_HUD_BOT_FILL_CONST: dict[int, np.ndarray] = {}

_CHORD_SPRITE_CACHE: dict[str, tuple] = {}
_MODE_SPRITE_CACHE:  dict[tuple, tuple] = {}
_FOOTER_ITEM_CACHE:  dict[tuple, tuple] = {}

_FOOTER_TEXT_COLOR = (255, 255, 255)


def _get_hud_top_const(w: int) -> np.ndarray:
    arr = _HUD_TOP_FILL_CONST.get(w)
    if arr is None:
        arr = np.full((52, w, 3), _HUD_TOP_FILL, dtype=np.uint8)
        _HUD_TOP_FILL_CONST[w] = arr
    return arr


def _get_hud_bot_const(w: int) -> np.ndarray:
    arr = _HUD_BOT_FILL_CONST.get(w)
    if arr is None:
        arr = np.full((HUD_BOTTOM_STRIP_H, w, 3), _HUD_BOT_FILL, dtype=np.uint8)
        _HUD_BOT_FILL_CONST[w] = arr
    return arr


def _get_chord_sprite(text: str) -> tuple:
    ts = _CHORD_SPRITE_CACHE.get(text)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_LG, (230, 230, 230))
        _CHORD_SPRITE_CACHE[text] = ts
    return ts


def _get_mode_sprite(text: str, color_bgr: tuple[int, int, int]) -> tuple:
    key = (text, color_bgr)
    ts = _MODE_SPRITE_CACHE.get(key)
    if ts is None:
        ts = _make_text_sprite(text, _FONT_SM, color_bgr)
        _MODE_SPRITE_CACHE[key] = ts
    return ts


def _get_footer_item(text: str, underline: bool, *, underline_gap: int = 2) -> tuple:
    key = (text, underline, underline_gap)
    ts = _FOOTER_ITEM_CACHE.get(key)
    if ts is None:
        ts = _make_text_sprite(
            text, _FONT_HUD_KEYS, _FOOTER_TEXT_COLOR,
            underline=underline, underline_gap=underline_gap,
        )
        _FOOTER_ITEM_CACHE[key] = ts
    return ts


def _draw_hud(frame: np.ndarray, left_seg: int, right_seg: int,
              root_override: str = "", mode_state: str = "iac",
              theory_mode: bool = False,
              voice_lead: bool = False, sync_mode: bool = False,
              instruments_open: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    root_label = root_override if root_override else (ROOT_LABELS[left_seg] if left_seg != -1 else "—")
    type_label = TYPE_LABELS[right_seg] if right_seg != -1 else "—"
    mode_label = {"iac": "IAC", "syn": "SYN", "osc": "OSC"}.get(mode_state, "IAC")
    mode_col   = (0, 215, 255) if mode_state == "iac" else (50, 230, 255)

    # Top bar — translucent dark band, blended in place on the top 52-px ROI only.
    top_roi = frame[0:52, 0:w]
    cv2.addWeighted(_get_hud_top_const(w), _HUD_TOP_DARK_ALPHA,
                    top_roi, 1.0 - _HUD_TOP_DARK_ALPHA, 0, top_roi)

    _blit_text_centered(frame, _get_chord_sprite(f"{root_label}  {type_label}"), w // 2, 26)
    _blit_text_centered(frame, _get_mode_sprite(mode_label, mode_col), w - 48, 26)

    # Bottom strip — translucent tint band, blended in place on bottom strip ROI only.
    y_strip = h - HUD_BOTTOM_STRIP_H
    bot_roi = frame[y_strip:h, 0:w]
    cv2.addWeighted(_get_hud_bot_const(w), HUD_BOTTOM_STRIP_ALPHA,
                    bot_roi, 1.0 - HUD_BOTTOM_STRIP_ALPHA, 0, bot_roi)

    # Footer keys — cached sprites, blit at right-aligned positions.
    items = []
    if mode_state == "syn":
        items += [
            _get_footer_item("I = instruments", instruments_open),
            _get_footer_item("   |   ",         False),
        ]
    items += [
        _get_footer_item("T = theory",      theory_mode),
        _get_footer_item("   |   ",         False),
        _get_footer_item("V = voice lead",  voice_lead, underline_gap=6),
        _get_footer_item("   |   ",         False),
        _get_footer_item("S = sync",        sync_mode),
        _get_footer_item("   |   ",         False),
        _get_footer_item("Q = quit",        False),
    ]
    widths   = [ts[1] for ts in items]
    total_w  = sum(widths)
    left_x   = w - HUD_KEYS_MARGIN_RIGHT - total_w
    y_center = h - HUD_BOTTOM_STRIP_H // 2
    cx = left_x
    for ts, tw in zip(items, widths):
        _blit_text_centered(frame, ts, cx + tw // 2, y_center)
        cx += tw

    frame = draw_brand_wordmark(frame, margin_x=14, strip_h=HUD_BOTTOM_STRIP_H)
    return frame

_OVERLAY_DIM = {"editor": 0.10, "sounds": 0.14, "settings": 0.18, "instruments": 0.18, "iac_help": 0.18, "beats": 0.18}

def _fast_dim(img, f):
    return cv2.convertScaleAbs(img, alpha=f)


def _pixelate(frame: np.ndarray, resolution: float, floor: float) -> np.ndarray:
    """Downscale-then-nearest-upscale. resolution 1.0 => no-op; lower => blockier.
    `floor` is the smallest scale factor at resolution 0.0."""
    if resolution >= 0.999:
        return frame
    h, w = frame.shape[:2]
    factor = floor + (1.0 - floor) * max(0.0, min(1.0, resolution))
    sw, sh = max(1, int(w * factor)), max(1, int(h * factor))
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# ── Threaded camera reader ─────────────────────────────────────────────────────
# AVFoundation on macOS serialises VideoCapture.read() to the camera period
# (~33 ms at 30 fps). When the main loop blocks on read(), every key press has
# to wait a full camera period before the next frame can render the new state,
# which is what makes T / V / S toggles feel sluggish even though the actual
# UI work is fast. Reading frames in a background thread decouples camera I/O
# from rendering: the main loop just grabs the latest available frame and runs
# at processing speed.

class _CameraReader:
    """Background thread that pulls frames from a `cv2.VideoCapture` and keeps
    only the freshest one available to the main loop. ``read()`` never blocks
    on camera I/O once the first frame has arrived."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap   = cap
        self._lock  = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq   = 0
        self._stop  = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="conductor-cam-reader",
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                # Brief retry; matches the main loop's tolerance.
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._seq  += 1
            self._ready.set()

    def read(self, timeout: float = 1.0) -> tuple[bool, Optional[np.ndarray], int]:
        """Return ``(ok, frame, seq)``. ``seq`` increments per camera frame so
        the caller can detect duplicate reads (main loop running faster than
        the camera) and skip redundant work."""
        if not self._ready.wait(timeout):
            return False, None, -1
        with self._lock:
            frame = self._frame
            seq   = self._seq
        if frame is None:
            return False, None, -1
        return True, frame, seq

    def release(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()


def _prewarm_caches() -> None:
    """Render every static UI sprite once at startup so no PIL roundtrip ever
    fires on the hot path. Without this the very first T / V / S toggle pays
    a one-shot ~5–10 ms cost to render the underlined footer variant; cheap
    but visible as a hitch right when the user wants the snappiest feedback."""
    for text in ("I = instruments", "T = theory", "V = voice lead", "S = sync", "Q = quit", "   |   "):
        for underline in (False, True):
            ug = 6 if text == "V = voice lead" else 2
            _get_footer_item(text, underline, underline_gap=ug)
    _get_chord_sprite("—  —")
    _get_mode_sprite("IAC", (0, 215, 255))
    _get_mode_sprite("SYN", (50, 230, 255))
    _get_mode_sprite("OSC", (50, 230, 255))
    # Settings gear + overlay static sprites
    prewarm_settings_sprites()
    # IAC setup card (both tabs — static content, built once)
    prewarm_iac_help()


# ── Splash screen ─────────────────────────────────────────────────────────────

def _run_splash(cam_reader: "_CameraReader", w: int, h: int) -> None:
    """Show 'COMPOSER' on black, then crossfade to the live camera feed.

    Hold ~0.8 s → fade ~0.7 s. Any key press skips the sequence.
    `cam_reader` must already be running so real frames are available during
    the fade phase.
    """
    font  = load_accent_font(120)
    ts    = _make_text_sprite(BRAND_TEXT, font, (255, 255, 255),
                               letter_spacing=BRAND_LETTER_SPACING)
    title = np.zeros((h, w, 3), dtype=np.uint8)
    _blit_text_centered(title, ts, w // 2, h // 2)

    HOLD_S = 2.3
    FADE_S = 1.0

    t0     = time.perf_counter()
    phase  = "hold"
    fade_t = 0.0
    while True:
        elapsed = time.perf_counter() - t0
        if phase == "hold":
            frame = title
            if elapsed >= HOLD_S:
                phase  = "fade"
                fade_t = time.perf_counter()
        else:
            alpha = min(1.0, (time.perf_counter() - fade_t) / FADE_S)
            ok, src, _ = cam_reader.read(timeout=0.1)
            cam_frame  = cv2.flip(src, 1) if (ok and src is not None) else title
            frame      = cv2.addWeighted(title, 1.0 - alpha, cam_frame, alpha, 0)
            if alpha >= 1.0:
                break
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(16) & 0xFF != 255:   # any key skips
            break


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("Starting chord engine…")
    engine = ChordEngine()
    engine.start()
    beat_engine = BeatEngine()
    print("  Engine ready.")

    # ── Load and apply persisted settings ──────────────────────────────────────
    _settings = load_settings()
    set_circle_size(_settings.circle_radius)
    _set_landmark_colour(_settings_color_bgr(_settings.tracker_color))
    engine.set_master_volume(_settings.master_volume)
    render_resolution = _settings.resolution   # live; assignable from mouse callback
    print(f"  Settings loaded: color={_settings.tracker_color!r}  "
          f"radius={_settings.circle_radius}  volume={_settings.master_volume:.2f}")

    _prefer_builtin_camera()
    cam_idx = _resolve_camera_index()
    if platform.system() == "Darwin":
        print(f"  Camera index {cam_idx} (built-in cam pinned via AVFoundation).")
        print("  Tip: CONDUCTOR_CAM_INDEX=1 python main.py  to force a different device.")

    def _configure_capture(cap_obj: cv2.VideoCapture) -> None:
        """Match capture to TARGET_W×TARGET_H@30 with a 1-deep buffer.

        BUFFERSIZE=1 prevents AVFoundation from queueing frames; without it the
        main loop reads stale frames whenever rendering briefly falls behind,
        which manifests as bursty / shaky playback on macOS.
        """
        cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
        cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
        cap_obj.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap_obj.set(cv2.CAP_PROP_FPS,          30)

    cap = cv2.VideoCapture(cam_idx)
    _configure_capture(cap)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        engine.stop()
        sys.exit(1)

    # Validate that the chosen index delivers real frames.
    # If the first read fails, try the next index before giving up.
    _ok, _test = cap.read()
    if not _ok:
        print(f"  [WARN] Camera {cam_idx} returned no frame — trying index {cam_idx + 1}…")
        cap.release()
        cap = cv2.VideoCapture(cam_idx + 1)
        _configure_capture(cap)
        if not cap.isOpened() or not cap.read()[0]:
            print("[ERROR] No working camera found.")
            if platform.system() == "Darwin":
                print("  Check: System Settings → Privacy & Security → Camera → allow Terminal/Python")
                print("  Override: CONDUCTOR_CAM_INDEX=<n> python main.py")
            engine.stop()
            sys.exit(1)
        print(f"  Using camera index {cam_idx + 1}.")

    tracker = HandTracker(max_hands=2)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, TARGET_W, TARGET_H)
    cv2.waitKey(1)   # prime the window event loop before the main loop starts

    # Start camera thread before the splash so live frames are ready for the fade.
    cam_reader = _CameraReader(cap)

    # Title splash — "COMPOSER" on black, crossfades to camera.
    _run_splash(cam_reader, TARGET_W, TARGET_H)

    # Render every static UI sprite once so the first T / V / S toggle is instant.
    _prewarm_caches()

    # ── Mouse callback (registered once; stays active for the whole session) ───
    # The callback runs on the main thread during cv2.waitKey, so it can safely
    # mutate main() locals via nonlocal.  Audio engine calls happen AFTER
    # waitKey in the "drain flags" block — not inside the callback — to keep the
    # callback non-blocking.
    def on_mouse(event, x, y, flags, param):
        nonlocal osc_ui, edit_patch, osc_idx, editor_sel
        nonlocal drag, mouse_xy, preview_dirty, save_req, activate_name
        nonlocal sounds_names, sounds_sel
        nonlocal name_edit, name_buf
        nonlocal mode_click_req, beat_click_req
        nonlocal settings_draft, settings_snapshot, settings_slider_drag
        nonlocal settings_tab, render_resolution

        mouse_xy = (x, y)

        # ── MOUSE MOVE — update slider value only; preview fires on release ───
        if event == cv2.EVENT_MOUSEMOVE:
            if settings_slider_drag is not None and osc_ui == "settings":
                fw, fh = frame_wh
                v = settings_slider_value_at(x, settings_slider_drag, fw, fh)
                if settings_slider_drag == "radius":
                    settings_draft.circle_radius = int(v)
                elif settings_slider_drag == "resolution":
                    settings_draft.resolution = float(v)
                else:
                    settings_draft.master_volume = float(v)
            elif drag is not None and osc_ui == "editor" and edit_patch is not None:
                group, pi = drag
                v = slider_value_at(x, group, pi)
                if group == "osc":
                    OSC_PARAMS[pi]["set"](edit_patch, osc_idx, v)
                    editor_sel = 1 + pi
                else:
                    GLOBAL_PARAMS[pi]["set"](edit_patch, v)
                    editor_sel = 1 + len(OSC_PARAMS) + pi
                # Do NOT set preview_dirty here — wait for LBUTTONUP

        # ── LEFT BUTTON DOWN ─────────────────────────────────────────────────
        elif event == cv2.EVENT_LBUTTONDOWN:
            if osc_ui == "editor" and edit_patch is not None:
                h = editor_hit(x, y, edit_patch, osc_idx)
                if h is None:
                    pass
                elif h[0] == "name":
                    name_edit = True
                    name_buf  = ""
                elif h[0] == "tab":
                    osc_idx    = h[1]
                    editor_sel = 0
                elif h[0] == "remove":
                    ri = h[1]
                    if len(edit_patch.oscillators) > 1:
                        edit_patch.oscillators.pop(ri)
                        osc_idx       = min(osc_idx, len(edit_patch.oscillators) - 1)
                        preview_dirty = True
                elif h[0] == "add":
                    if len(edit_patch.oscillators) < 4:
                        edit_patch.oscillators.append(OscSpec(wave="sine", level=0.5))
                        osc_idx       = len(edit_patch.oscillators) - 1
                        preview_dirty = True
                elif h[0] == "wave":
                    wi = h[1]
                    if osc_idx < len(edit_patch.oscillators):
                        edit_patch.oscillators[osc_idx].wave = _OSC_WAVE_NAMES[wi]
                        editor_sel    = 0
                        preview_dirty = True
                elif h[0] == "slider":
                    group, pi = h[1], h[2]
                    v = slider_value_at(x, group, pi)
                    if group == "osc":
                        OSC_PARAMS[pi]["set"](edit_patch, osc_idx, v)
                        editor_sel = 1 + pi
                    else:
                        GLOBAL_PARAMS[pi]["set"](edit_patch, v)
                        editor_sel = 1 + len(OSC_PARAMS) + pi
                    drag = (group, pi)
                    # preview fires on LBUTTONUP, not here

            elif osc_ui == "sounds":
                h = sounds_hit(x, y, sounds_names, sounds_sel)
                if h is None:
                    pass
                elif h[0] == "close":
                    osc_ui = "off"
                elif h[0] == "row":
                    sounds_sel = h[1]
                elif h[0] == "activate":
                    if sounds_names:
                        activate_name = sounds_names[sounds_sel]
                    osc_ui = "off"

            elif osc_ui == "settings":
                fw, fh = frame_wh
                sh = settings_hit(x, y, fw, fh, settings_tab)
                if sh is None:
                    pass
                elif sh[0] == "tab":
                    settings_tab = sh[1]
                elif sh[0] == "arrow" and sh[1] == "color":
                    ci  = _settings_color_idx(settings_draft.tracker_color)
                    ci  = (ci + sh[2]) % len(NAMED_COLORS)
                    settings_draft.tracker_color = NAMED_COLORS[ci][0]
                elif sh[0] == "slider":
                    settings_slider_drag = sh[1]
                    v = settings_slider_value_at(x, sh[1], fw, fh)
                    if sh[1] == "radius":
                        settings_draft.circle_radius = int(v)
                    elif sh[1] == "resolution":
                        settings_draft.resolution = float(v)
                    else:
                        settings_draft.master_volume = float(v)
                elif sh[0] == "apply":
                    set_circle_size(settings_draft.circle_radius)
                    _set_landmark_colour(_settings_color_bgr(settings_draft.tracker_color))
                    engine.set_master_volume(settings_draft.master_volume)
                    render_resolution = settings_draft.resolution
                    save_settings(settings_draft)
                    osc_ui = "off"
                elif sh[0] == "cancel":
                    settings_draft = copy.deepcopy(settings_snapshot)
                    settings_slider_drag = None
                    osc_ui = "off"

            # ── Normal mode: click the on-screen buttons with the cursor ─────
            # Hand hover still works; this just adds a mouse path.  We only flag
            # the request here — the actual engine calls run in the drain block.
            elif osc_ui == "off":
                fw, fh = frame_wh
                # Check gear first (top-right)
                if get_hovered_settings_gear((x, y), fw, fh):
                    # Snapshot the current live state so Cancel can revert
                    settings_snapshot = copy.deepcopy(settings_draft)
                    osc_ui = "settings"
                elif get_hovered_mode_button((x, y), fw, fh):
                    mode_click_req = True
                elif get_hovered_beat_button((x, y), fw, fh):
                    beat_click_req = True

        # ── LEFT BUTTON UP — release drag and fire a single preview ──────────
        elif event == cv2.EVENT_LBUTTONUP:
            if drag is not None:
                preview_dirty = True
            drag = None
            settings_slider_drag = None

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # Chord follow-finger state
    prev_left_seg   = -1
    prev_right_seg  = -1
    prev_mods       = (False, False, False)   # (left_flat, right_nine, right_b9)
    silence_counter = 0

    # Voice-leading mode (toggled by V key)
    voice_lead = False

    # Theory overlay (T key) and live practice box (X, on trigger slide / progressions)
    theory_mode         = False
    theory_state        = TheoryState()
    practice_mode       = False
    active_practice_seq = ""

    # Sync mode — avoid intermediate chords when root/type update one-at-a-time (S key)
    sync_mode = False
    committed_ls: int | None = None   # last segments passed to engine.play (sound identity)
    committed_rs: int | None = None

    # SYN instrument picker — cursor index into GM_INSTRUMENTS
    instr_sel = 0

    # Jersey beat button — independent toggle
    beat_active  = False
    beat_was_in  = False
    beats_sel    = 0
    beat_playing: str | None = None   # genre currently looping via BeatEngine, or None

    # Output mode — 3-state: "iac" | "syn" | "osc". Reflect what the engine
    # actually resolved at start() (IAC may have fallen back to fluid).
    mode_state   = "syn" if engine.active_backend() == "fluid" else "iac"
    mode_was_in  = False

    # IAC setup help card tab — 0 = logic, 1 = fl studio
    iac_help_tab  = 0

    # Osc synth UI state (also takes "settings", "instruments", and "iac_help")
    osc_ui        = "off"    # "off" | "editor" | "sounds" | "settings" | "instruments" | "iac_help"
    osc_backend   = False    # True when engine backend == "osc" (for HUD underline)
    edit_patch    = None     # copy of patch being edited
    osc_idx       = 0        # which oscillator tab is active in the editor
    editor_sel    = 0        # 0=wave, 1-4=OSC param, 5-11=GLOBAL param
    drag          = None     # ("osc"|"global", param_i) while mouse button held
    mouse_xy      = (0, 0)   # last reported cursor position
    preview_dirty = False    # call engine.preview_patch once per frame when True
    save_req      = False    # save patch + close editor on next frame
    activate_name: str | None = None   # patch name to load & activate
    sounds_names: list[str] = []
    sounds_sel    = 0
    name_edit     = False    # True while user is editing the patch name
    name_buf      = ""       # text buffer during name edit

    # Settings overlay state
    settings_draft:    Settings = copy.deepcopy(_settings)
    settings_snapshot: Settings = copy.deepcopy(_settings)
    settings_slider_drag: str | None = None   # "radius" | "volume" | "resolution" while dragging
    settings_tab: int = 0   # 0 = controls, 1 = visuals
    _overlay_backdrop = None
    _overlay_backdrop_kind = None

    # Cursor-click parity for the on-screen buttons (mode / beat).
    # The mouse callback only sets these request flags; the engine calls happen
    # in the drain block after waitKey, matching the editor's flag pattern so the
    # callback stays non-blocking.
    frame_wh        = (TARGET_W, TARGET_H)   # live frame size for hit-testing in the callback
    mode_click_req  = False                  # cursor clicked the mode button
    beat_click_req  = False                  # cursor clicked the beat button

    # Shared button actions — invoked by BOTH hand-hover rising edges and cursor
    # clicks so the two input paths can never drift out of sync.
    def _cycle_mode() -> None:
        nonlocal mode_state, osc_backend, osc_ui, sounds_names, sounds_sel
        nonlocal prev_left_seg, prev_right_seg, instr_sel, iac_help_tab
        if mode_state == "iac":
            mode_state  = "syn"
            osc_backend = False
            engine.set_backend("fluid")
            prev_left_seg = prev_right_seg = -1
            # Auto-open the instrument picker when entering SYN
            prog = engine._synth_program
            instr_sel = next(
                (i for i, (_, p) in enumerate(GM_INSTRUMENTS) if p == prog), 0
            )
            osc_ui = "instruments"
            print("  Mode: SYN — instrument picker open")
        elif mode_state == "syn":
            mode_state   = "osc"
            osc_backend  = True
            engine.set_backend("osc")
            sounds_names = PatchStore().list()
            sounds_sel   = 0
            osc_ui       = "sounds"
            print("  Mode: OSC — sounds list open")
        else:  # "osc"
            osc_backend   = False
            engine.set_backend("iac")     # always resolves to iac now
            mode_state    = "iac"
            iac_help_tab  = 0             # default to the logic tab
            osc_ui        = "iac_help"    # show the setup card every time
            prev_left_seg = prev_right_seg = -1
            print("  Mode: IAC" + ("" if engine.iac_available() else "  (iac not set up — silent)"))

    def _toggle_beat() -> None:
        nonlocal beat_active
        beat_active = not beat_active
        engine.set_beat(beat_active)
        print(f"  Beat {'ON' if beat_active else 'OFF'}")

    print("Conductor running — point fingers at circles to play, hover buttons (bottom-right) to layer instruments.")
    print("Press Q or ESC to quit.\n")

    _bad_frames    = 0           # consecutive failed reads
    _last_seq      = -1          # de-dupe identical frames from the camera reader
    _fps_t0        = time.perf_counter()
    _fps_count     = 0
    while True:
        ok, src_frame, seq = cam_reader.read(timeout=1.0)
        if not ok:
            _bad_frames += 1
            if _bad_frames > 120:   # ~4 s of failures at 30 fps
                print("[ERROR] Camera stopped delivering frames. Shutting down.")
                break
            cv2.waitKey(1)   # keep the window event loop alive during failures
            continue
        _bad_frames = 0
        if seq == _last_seq and osc_ui == "off":
            # Render loop ran faster than the camera produced a new frame.
            # Pump the GUI event loop and try again — avoids burning CPU on
            # identical frames without adding any noticeable latency.
            cv2.waitKey(1)
            continue
        _last_seq = seq

        # cv2.flip allocates a fresh array, so the reader's stored buffer is
        # never mutated by the per-frame drawing that happens below.
        frame = cv2.flip(src_frame, 1)
        frame = _pixelate(frame, render_resolution, 0.04)
        h, w  = frame.shape[:2]
        frame_wh = (w, h)   # keep the mouse callback's hit-testing in sync

        # ── Circle centres ─────────────────────────────────────────────
        lcx, lcy = w // 4,     h // 2
        rcx, rcy = 3 * w // 4, h // 2

        # ── Hand tracking + interaction (skipped when overlay is open) ─
        if osc_ui == "off":
            frame, hands = tracker.process(frame)
            left  = tracker.get_hand(hands, "Left")
            right = tracker.get_hand(hands, "Right")

            left_seg  = angle_to_segment(lcx, lcy, *left.index_tip)  if (left  and left.index_extended)  else -1
            right_seg = angle_to_segment(rcx, rcy, *right.index_tip) if (right and right.index_extended) else -1

            finger_tips = [hd.index_tip for hd in hands]

            # ── Jersey beat button ─────────────────────────────────────
            beat_hovered = any(get_hovered_beat_button(tip, w, h) for tip in finger_tips)
            if beat_hovered and not beat_was_in:
                osc_ui = "beats"; beats_sel = 0
            beat_was_in = beat_hovered

            # ── Mode toggle button — 3-state cycle IAC → SYN → OSC ────
            mode_hovered = any(get_hovered_mode_button(tip, w, h) for tip in finger_tips)
            if mode_hovered and not mode_was_in:          # rising edge
                _cycle_mode()
            mode_was_in = mode_hovered

            # ── Pinch / modifier state ─────────────────────────────────
            left_flat     = left.pinch_active  if left  else False
            right_nine    = right.pinch_active if right else False
            right_b9      = right.pinch_triple if right else False
            add_nine      = right_nine and not right_b9
            add_flat_nine = right_b9

            # ── Chord audio logic ──────────────────────────────────────
            if theory_mode:
                pass
            elif left is None and right is None:
                engine.all_notes_off()
                silence_counter = 0
                committed_ls = committed_rs = None
            elif left_seg == -1 or right_seg == -1:
                silence_counter += 1
                if silence_counter >= SILENCE_FRAMES:
                    engine.all_notes_off()
                    committed_ls = committed_rs = None
            else:
                silence_counter = 0
                seg_changed   = left_seg != prev_left_seg or right_seg != prev_right_seg
                mods_changed  = (left_flat, right_nine, right_b9) != prev_mods
                if seg_changed or mods_changed:
                    root  = ROOT_LABELS[left_seg]
                    root  = FLAT_MAP[root] if left_flat else root
                    ctype = TYPE_LABELS[right_seg]

                    do_play = True
                    if sync_mode and committed_ls is not None:
                        root_diff = left_seg != committed_ls
                        typ_diff  = right_seg != committed_rs
                        partial   = root_diff ^ typ_diff
                        if partial and not mods_changed:
                            do_play = False

                    if do_play:
                        engine.play(root, ctype, add_nine=add_nine, add_flat_nine=add_flat_nine)
                        committed_ls = left_seg
                        committed_rs = right_seg
                        suffix = " (b9)" if add_flat_nine else (" (9)" if add_nine else "")
                        print(f"  {root} {ctype}{suffix}")

            prev_left_seg  = left_seg
            prev_right_seg = right_seg
            prev_mods      = (left_flat, right_nine, right_b9)

        else:
            # Overlay is open — skip all hand tracking and chord logic.
            # The last chord continues to ring; no button hover updates.
            hands        = []
            finger_tips  = []
            left_seg = right_seg = -1
            left_flat = False
            beat_hovered = False
            mode_hovered = False

        # ── Draw ───────────────────────────────────────────────────────
        def _draw_base_ui(f):
            f = draw_circles(
                f,
                left_hover_idx    = left_seg,
                right_hover_idx   = right_seg,
                left_confirm_idx  = left_seg,
                right_confirm_idx = right_seg,
            )
            f = draw_beat_button(f, beat_active, beat_hovered)
            f = draw_mode_button(f, mode_state, mode_hovered)
            _ghov = get_hovered_settings_gear(mouse_xy, w, h) and osc_ui == "off"
            f = draw_settings_gear(f, _ghov)
            _hud_root = FLAT_MAP[ROOT_LABELS[left_seg]] if (left_seg != -1 and left_flat) else ""
            f = _draw_hud(
                f, left_seg, right_seg, root_override=_hud_root,
                mode_state=mode_state, theory_mode=theory_mode,
                voice_lead=voice_lead, sync_mode=sync_mode,
                instruments_open=(osc_ui == "instruments"),
            )
            if theory_mode:
                _lf, _rf = update_dwell(theory_state, finger_tips, w, h)
                if _lf:
                    navigate(theory_state, -1)
                if _rf:
                    navigate(theory_state, +1)
                f = draw_theory_overlay(f, theory_state, finger_tips)
            if practice_mode:
                f = draw_practice_box(f, active_practice_seq)
            return f

        if osc_ui != "off":
            if _overlay_backdrop is None or _overlay_backdrop_kind != osc_ui:
                _base = cv2.flip(src_frame, 1)
                _base = _pixelate(_base, render_resolution, 0.04)
                _base = _draw_base_ui(_base)
                _overlay_backdrop      = _fast_dim(_base, _OVERLAY_DIM[osc_ui])
                _overlay_backdrop_kind = osc_ui
            frame = _overlay_backdrop.copy()
            if osc_ui == "editor" and edit_patch is not None:
                _ed_hover = editor_hit(*mouse_xy, edit_patch, osc_idx)
                frame = draw_synth_editor(
                    frame, edit_patch, osc_idx, editor_sel, _ed_hover,
                    name_edit=name_edit, name_buf=name_buf, dim=False,
                )
            elif osc_ui == "sounds":
                frame = draw_sounds_list(frame, sounds_names, sounds_sel, dim=False)
            elif osc_ui == "settings":
                frame = draw_settings_overlay(frame, settings_draft, mouse_xy, settings_tab, dim=False)
            elif osc_ui == "instruments":
                frame = draw_instrument_picker(frame, GM_INSTRUMENTS, instr_sel)
            elif osc_ui == "iac_help":
                frame = draw_iac_help(frame, iac_help_tab)
            elif osc_ui == "beats":
                _play_idx = 0 if beat_playing is None else BEAT_NAMES.index(beat_playing) + 1
                frame = draw_beats_menu(frame, ["(stop)"] + BEAT_NAMES, beats_sel, _play_idx)
        else:
            _overlay_backdrop      = None
            _overlay_backdrop_kind = None
            frame = _draw_base_ui(frame)

        if osc_ui == "off":
            frame = _pixelate(frame, render_resolution, 0.5)
        cv2.imshow(WINDOW_NAME, frame)
        raw_key = cv2.waitKey(1)
        key     = raw_key & 0xFF

        # Lightweight FPS readout every ~2 s — cheap, useful for verifying the
        # render loop is actually keeping up with the camera.
        _fps_count += 1
        _fps_dt = time.perf_counter() - _fps_t0
        if _fps_dt >= 2.0:
            print(f"  fps: {_fps_count / _fps_dt:.1f}")
            _fps_t0    = time.perf_counter()
            _fps_count = 0

        # ── Drain cursor-click button requests ───────────────────────────
        # The mouse callback (during waitKey, above) only sets flags; the
        # engine calls happen here so the callback stays non-blocking. These
        # mirror the hand-hover edges and share the same action helpers.
        if mode_click_req:
            mode_click_req = False
            _cycle_mode()
        if beat_click_req:
            beat_click_req = False
            osc_ui = "beats"; beats_sel = 0

        if not name_edit and key in (ord("q"), ord("Q")):
            break
        if key == 27:   # ESC — close any open overlay, never quit
            if name_edit:
                name_edit = False   # cancel name edit
            elif osc_ui == "settings":
                # Cancel semantics: nothing was applied live, just discard draft
                settings_draft = copy.deepcopy(settings_snapshot)
                settings_slider_drag = None
                osc_ui    = "off"
            elif osc_ui != "off":
                osc_ui    = "off"
                name_edit = False

        # ── 'i' key — toggle SYN instrument picker open / closed ───────
        if not name_edit and key in (ord("i"), ord("I")):
            if mode_state == "syn":
                if osc_ui == "instruments":
                    osc_ui = "off"
                elif osc_ui == "off":
                    prog = engine._synth_program
                    instr_sel = next(
                        (i for i, (_, p) in enumerate(GM_INSTRUMENTS) if p == prog), 0
                    )
                    osc_ui = "instruments"

        # ── 'o' key — toggle editor open / closed ──────────────────────
        if not name_edit and key in (ord("o"), ord("O")):
            if osc_ui in ("editor", "sounds"):
                osc_ui    = "off"
                name_edit = False
            elif osc_ui == "off":
                edit_patch = copy.deepcopy(engine._active_patch)
                editor_sel = 0
                osc_idx    = 0
                name_edit  = False
                osc_ui     = "editor"

        # ── Name-edit keyboard handling ─────────────────────────────────
        if name_edit and osc_ui == "editor":
            if key == 13:                         # Enter — confirm
                if edit_patch is not None:
                    edit_patch.name = name_buf
                name_edit = False
            elif key == 27:
                name_edit = False                 # ESC — cancel (already handled above)
            elif key in (8, 127):                 # Backspace (8 = BS, 127 = DEL/macOS)
                name_buf = name_buf[:-1]
            elif 32 <= key <= 126:                # printable ASCII
                name_buf += chr(key)

        # ── Editor key handling ─────────────────────────────────────────
        if osc_ui == "editor" and edit_patch is not None and not name_edit:
            if key == 9:   # Tab — cycle oscillator
                if edit_patch.oscillators:
                    osc_idx = (osc_idx + 1) % len(edit_patch.oscillators)
                    editor_sel = 0
            elif raw_key in _UP_KEYS:
                editor_sel = max(0, editor_sel - 1)
            elif raw_key in _DOWN_KEYS:
                editor_sel = min(EDITOR_N_ROWS - 1, editor_sel + 1)
            elif raw_key in _LEFT_KEYS or raw_key in _RIGHT_KEYS:
                delta = -1 if raw_key in _LEFT_KEYS else +1
                if editor_sel == 0:
                    # Cycle waveform for the active oscillator
                    if osc_idx < len(edit_patch.oscillators):
                        cur = edit_patch.oscillators[osc_idx].wave
                        wi  = list(_OSC_WAVE_NAMES).index(cur) if cur in _OSC_WAVE_NAMES else 0
                        edit_patch.oscillators[osc_idx].wave = _OSC_WAVE_NAMES[
                            (wi + delta) % len(_OSC_WAVE_NAMES)
                        ]
                        preview_dirty = True
                elif editor_sel <= len(OSC_PARAMS):
                    # OSC param
                    pi = editor_sel - 1
                    p  = OSC_PARAMS[pi]
                    if osc_idx < len(edit_patch.oscillators):
                        old_v = p["get"](edit_patch, osc_idx)
                        p["set"](edit_patch, osc_idx, old_v + delta * p["step"])
                        preview_dirty = True
                else:
                    # GLOBAL param
                    gi = editor_sel - 1 - len(OSC_PARAMS)
                    if gi < len(GLOBAL_PARAMS):
                        p     = GLOBAL_PARAMS[gi]
                        old_v = p["get"](edit_patch)
                        p["set"](edit_patch, old_v + delta * p["step"])
                        preview_dirty = True
            elif key in (ord("a"), ord("A")):
                preview_dirty = True
            elif key in (ord("s"), ord("S")):
                save_req = True

        # ── Sounds list key handling ────────────────────────────────────
        elif osc_ui == "sounds":
            if raw_key in _UP_KEYS:
                sounds_sel = max(0, sounds_sel - 1)
            elif raw_key in _DOWN_KEYS:
                sounds_sel = min(max(0, len(sounds_names) - 1), sounds_sel + 1)
            elif key == 13:   # Enter — queue patch activation
                if sounds_names:
                    activate_name = sounds_names[sounds_sel]
                osc_ui = "off"

        # ── Instrument picker key handling ──────────────────────────────
        elif osc_ui == "instruments":
            if raw_key in _UP_KEYS:
                instr_sel = max(0, instr_sel - 1)
            elif raw_key in _DOWN_KEYS:
                instr_sel = min(len(GM_INSTRUMENTS) - 1, instr_sel + 1)
            elif key == 13:   # Enter — apply instrument and close
                engine.set_synth_program(GM_INSTRUMENTS[instr_sel][1])
                osc_ui = "off"
                print(f"  Instrument: {GM_INSTRUMENTS[instr_sel][0]!r} "
                      f"(GM {GM_INSTRUMENTS[instr_sel][1]})")

        # ── IAC help card tab handling ───────────────────────────────────
        elif osc_ui == "iac_help":
            if raw_key in _LEFT_KEYS:
                iac_help_tab = 0
            elif raw_key in _RIGHT_KEYS:
                iac_help_tab = 1
            elif key == 13:   # enter also closes
                osc_ui = "off"

        # ── Beats menu key handling ──────────────────────────────────────
        elif osc_ui == "beats":
            # Rows = ["(stop)"] + BEAT_NAMES; row 0 stops, rows 1.. are genres.
            if raw_key in _UP_KEYS:
                beats_sel = max(0, beats_sel - 1)
            elif raw_key in _DOWN_KEYS:
                beats_sel = min(len(BEAT_NAMES), beats_sel + 1)
            elif key == 13:   # enter — start / switch / stop (menu stays open)
                if beats_sel == 0:
                    beat_engine.stop(); beat_playing = None
                else:
                    genre = BEAT_NAMES[beats_sel - 1]
                    if beat_playing == genre:
                        beat_engine.stop(); beat_playing = None
                    else:
                        beat_engine.play(genre); beat_playing = genre

        # ── Drain mouse / keyboard flags ────────────────────────────────
        if osc_ui == "editor" and edit_patch is not None:
            if preview_dirty:
                engine.preview_patch(edit_patch)
                preview_dirty = False
            if save_req:
                save_req  = False
                name_edit = False
                if edit_patch.name in ("Default Pad", "Untitled", ""):
                    _n = len(PatchStore().list()) + 1
                    edit_patch.name = f"Sound {_n}"
                PatchStore().save(edit_patch)
                engine.set_active_patch(edit_patch)
                osc_ui = "off"
                print(f"  Saved patch: {edit_patch.name!r}")
        else:
            preview_dirty = False
            save_req      = False

        if activate_name is not None:
            _aname        = activate_name
            activate_name = None
            _loaded = PatchStore().load(_aname)
            if _loaded is not None:
                engine.set_active_patch(_loaded)
                engine.set_backend("osc")
                osc_backend    = True
                prev_left_seg  = -1
                prev_right_seg = -1
                print(f"  Activated patch: {_loaded.name!r}")

        # ── Normal-mode key handlers (suppressed while overlay is open) ─
        if osc_ui != "off":
            pass   # skip all normal hotkeys while editor/sounds is open
        elif key in (ord("t"), ord("T")):
            if practice_mode:
                # Close practice box, reopen theory overlay on same slide
                practice_mode = False
                theory_mode   = True
                print("  Practice box OFF — theory restored")
            else:
                theory_mode = not theory_mode
                if theory_mode:
                    theory_state = TheoryState()
                    engine.all_notes_off()
                prev_left_seg  = -1
                prev_right_seg = -1
                committed_ls = committed_rs = None
                print(f"  Theory {'ON' if theory_mode else 'OFF'}")
        if osc_ui == "off":
            if key in (ord("x"), ord("X")):
                if practice_mode:
                    # Cancel everything — close practice box and theory entirely
                    practice_mode = False
                    theory_mode   = False
                    engine.all_notes_off()
                    print("  Practice box cancelled")
                elif (theory_mode
                      and theory_state.screen     == "LESSON"
                      and theory_state.lesson_idx == PRACTICE_SLIDE[0]
                      and theory_state.step_idx   == PRACTICE_SLIDE[1]):
                    # On the trigger slide — open practice box (circle of fifths)
                    active_practice_seq = PROGRESSIONS[1]["sequence"]
                    practice_mode       = True
                    theory_mode         = False
                    print("  Practice mode ON")
            # Theory overlay navigation
            if theory_mode:
                if theory_state.screen == "SELECTION":
                    if key == ord("0"):
                        theory_state.screen     = "LESSON"
                        theory_state.lesson_idx = 0
                        theory_state.step_idx   = 0
                    elif key == ord("1"):
                        theory_state.screen     = "LESSON"
                        theory_state.lesson_idx = 1
                        theory_state.step_idx   = 0
                    elif key in (ord("+"), ord("=")):
                        theory_state.screen = "PROGRESSIONS"
                elif theory_state.screen == "PROGRESSIONS":
                    if raw_key in _UP_KEYS:
                        prog_move_cursor(theory_state, -1)
                    elif raw_key in _DOWN_KEYS:
                        prog_move_cursor(theory_state, +1)
                    elif raw_key in _LEFT_KEYS:
                        prog_switch_tab(theory_state, -1)
                    elif raw_key in _RIGHT_KEYS:
                        prog_switch_tab(theory_state, +1)
                    elif key in (13, 10, ord(" ")):
                        seq = prog_selected_sequence(theory_state)
                        if seq:
                            active_practice_seq = seq
                            practice_mode       = True
                            theory_mode         = False
                            print("  Practice: progression loaded")
                elif theory_state.screen == "LESSON":
                    if raw_key in _LEFT_KEYS:
                        navigate(theory_state, -1)
                    elif raw_key in _RIGHT_KEYS:
                        navigate(theory_state, +1)
            if key in (ord("v"), ord("V")):
                voice_lead = not voice_lead
                engine.set_voice_lead(voice_lead)
                prev_left_seg  = -1
                prev_right_seg = -1
                print(f"  Voice lead {'ON' if voice_lead else 'OFF'}")
            if key in (ord("s"), ord("S")):
                sync_mode = not sync_mode
                prev_left_seg   = -1
                prev_right_seg  = -1
                committed_ls = committed_rs = None
                print(f"  Sync {'ON' if sync_mode else 'OFF'}")
            if key == ord(" "):
                engine.all_notes_off()
                committed_ls = committed_rs = None
                print("  PANIC — all notes off")

    print("\nShutting down…")
    beat_engine.close()
    engine.stop()
    tracker.close()
    cam_reader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
