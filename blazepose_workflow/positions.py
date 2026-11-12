"""View, handedness and P1-P10 from body landmarks only (no club, no ball).

Position definitions follow HackMotion (see FRAME_RECOGNITION.md section 1).
Without a club, the club-based positions use body proxies:
- P2 / P6: lead arm halfway between its address / impact angle and parallel
  (shaft parallel is not observable).
- P7: the hands' lowest point after the top (ball contact is not observable).
- P8: the lead arm rising back through the P6 level after impact.

Image y points down; "height" is -y. World landmarks are metres around the
hips with y down, so straight down is +y.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from blazepose_workflow.landmarks import J, hands, hip_center, lead_trail, torso_length

POSITIONS = [f"p{i}" for i in range(1, 11)]
# Hands this slow (torso lengths per second) for STILL_SECONDS count as stationary.
STILL_SPEED = 0.6
STILL_SECONDS = 0.2
# A top must be a hand-height peak this prominent, in torso lengths and as a fraction of the range.
TOP_PROMINENCE = 0.5
TOP_SPAN_FRACTION = 0.5
# Hands highest within this time of the first frame: the clip starts at or after the top.
LATE_START_SECONDS = 0.1
ARM_PARALLEL_DEG = 90.0
# P2 / P6 / P8 lead-arm level: this fraction of the way from the address (P2) or
# impact (P6, P8) arm angle to parallel.
ARM_HALFWAY = 0.5
# View: shoulder width over torso length in the image at address.
FACE_ON_WIDTH = 0.35
DOWN_THE_LINE_WIDTH = 0.25
# View: share of the 3D shoulder line across the camera (x) rather than along depth (z).
FACE_ON_ACROSS = 0.7
DOWN_THE_LINE_ACROSS = 0.35
# Handedness: hands' horizontal shift in torso lengths.
HAND_SHIFT = 0.2


class SwingCoverageError(ValueError):
    """The clip does not contain the whole swing (address to finish)."""


@dataclass
class Signals:
    times_ms: np.ndarray
    height: np.ndarray          # hands above the hips, torso lengths
    hands_x: np.ndarray         # hands right of the hips, torso lengths
    hands_rel: np.ndarray       # (n, 2) hands relative to the hips, torso lengths
    speed: np.ndarray           # torso lengths per second
    torso_px: float
    extra: dict = field(default_factory=dict)


def swing_signals(points2d: np.ndarray, times_ms: np.ndarray) -> Signals:
    hips = hip_center(points2d)
    torso = float(np.nanmedian(torso_length(points2d)))
    if not np.isfinite(torso) or torso <= 0:
        raise SwingCoverageError("No frame has both shoulders and hips; the golfer was not found")
    rel = (hands(points2d) - hips) / torso
    seconds = np.asarray(times_ms, float) / 1000
    with np.errstate(invalid="ignore"):
        velocity = np.gradient(rel, seconds, axis=0) if len(seconds) > 1 else np.zeros_like(rel)
    return Signals(times_ms=np.asarray(times_ms, float), height=-rel[:, 1], hands_x=rel[:, 0],
                   hands_rel=rel, speed=np.linalg.norm(velocity, axis=1), torso_px=torso)


def arm_elevation(world: np.ndarray, side: str) -> np.ndarray:
    """Degrees between the shoulder-to-wrist vector and straight down (0 hanging, 90 parallel)."""
    vector = world[:, J[f"{side}_wrist"]] - world[:, J[f"{side}_shoulder"]]
    length = np.linalg.norm(vector, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip(vector[:, 1] / length, -1, 1)))


def elbow_angle(world: np.ndarray, side: str) -> np.ndarray:
    """Inner elbow angle in degrees; 180 is a straight arm."""
    upper = world[:, J[f"{side}_shoulder"]] - world[:, J[f"{side}_elbow"]]
    lower = world[:, J[f"{side}_wrist"]] - world[:, J[f"{side}_elbow"]]
    cosine = (upper * lower).sum(axis=1) / (np.linalg.norm(upper, axis=1) * np.linalg.norm(lower, axis=1))
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def _frames_within(times_ms: np.ndarray, start: int, seconds: float) -> int:
    """Number of frames from `start` spanning `seconds`."""
    limit = times_ms[start] + seconds * 1000
    return int(np.searchsorted(times_ms, limit, side="right") - start)


def find_top(signals: Signals) -> tuple[int, int, dict]:
    """(top, lowest hands after the top).

    The top is the first prominent hand-height peak whose drop to the next low
    is at least half the largest such drop; the finish (also a high point) comes
    later. The low is the minimum before the next peak, so a clip ending soon
    after impact still has one.
    """
    height = _interpolated(signals.height)
    span = float(np.ptp(height))
    prominence = max(TOP_PROMINENCE, TOP_SPAN_FRACTION * span)
    peaks, _ = find_peaks(height, prominence=prominence)
    candidates = []
    for i, peak in enumerate(peaks):
        stop = peaks[i + 1] if i + 1 < len(peaks) else len(height)
        low = _bottom_after(height, int(peak), int(stop), prominence)
        drop = float(height[peak] - height[low])
        if drop >= prominence:
            candidates.append((drop, int(peak), low))
    if not candidates:
        low = int(np.argmin(height))
        highest = int(np.argmax(height[:low + 1]))
        if (_frames_within(signals.times_ms, 0, LATE_START_SECONDS) > highest
                and height[highest] - height[low] >= prominence):
            raise SwingCoverageError(
                f"The clip starts at or after the top of the backswing: the hands are highest on frame "
                f"{highest} and only fall afterwards (lowest on frame {low}). Address, P2 and P3 "
                f"are not in the video.")
        raise SwingCoverageError("No backswing found: the hands never rise and fall back by at least "
                                 f"{prominence:.2f} torso lengths")
    largest = max(drop for drop, _, _ in candidates)
    _, top, low = next(item for item in candidates if item[0] >= 0.5 * largest)
    return top, low, {"hand_height_span_torso": round(span, 2), "top_prominence_torso": round(prominence, 2)}


def _interpolated(series: np.ndarray) -> np.ndarray:
    """Missing values linearly interpolated (held at the ends), so they never look like extremes."""
    good = np.isfinite(series)
    if not good.any():
        raise SwingCoverageError("The hands were never found")
    index = np.arange(len(series))
    return np.interp(index, index[good], series[good])


def _bottom_after(height: np.ndarray, peak: int, stop: int, prominence: float) -> int:
    """Lowest hands of the first dip after `peak`: ends once the hands climb back halfway out of it."""
    low = peak
    for frame in range(peak + 1, stop):
        if height[frame] < height[low]:
            low = frame
        elif (height[peak] - height[low] >= prominence
              and height[frame] - height[low] >= 0.5 * (height[peak] - height[low])):
            break
    return low


def find_address(signals: Signals, top: int) -> tuple[int, str]:
    """Last frame of a still period (hands below STILL_SPEED for STILL_SECONDS) before the takeaway."""
    before = signals.height[:top + 1]
    low = float(np.nanmin(before))
    rise = float(signals.height[top] - low)
    if rise < TOP_SPAN_FRACTION * float(np.nanmax(signals.height) - np.nanmin(signals.height)):
        raise SwingCoverageError(f"The clip starts during the backswing: the hands rise only {rise:.2f} torso "
                                 "lengths before the top. Address is not in the video.")
    still = signals.speed < STILL_SPEED
    near_low = signals.height <= low + 0.25 * rise
    times = signals.times_ms
    for frame in range(top, -1, -1):
        if not (still[frame] and near_low[frame]):
            continue
        start = int(np.searchsorted(times, times[frame] - STILL_SECONDS * 1000))
        if times[frame] - times[start] >= STILL_SECONDS * 1000 - 1 and still[start:frame + 1].all():
            return frame, "hands_still"
    return int(np.nanargmin(before)), "lowest_hands_before_top"


def find_impact(signals: Signals, address: int, top: int, low: int) -> tuple[int, dict]:
    """The hands' lowest point after the top.

    The hands pass their address position on the way down earlier than impact
    (005: 214 vs ~225) because the body lowers and the hands lead the clubhead;
    that frame is reported for reference only.
    """
    window = np.arange(top + 1, low + 1)
    distance = np.linalg.norm(signals.hands_rel[window] - signals.hands_rel[address], axis=1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    back = int(window[int(np.argmin(distance))]) if len(window) else low
    return low, {"hand_height_torso": _round(signals.height[low]), "hands_back_at_address_frame": back,
                 "hands_from_address_at_impact_torso": _round(np.linalg.norm(
                     signals.hands_rel[low] - signals.hands_rel[address]))}


def crossing(signal: np.ndarray, level: float, start: int, stop: int, rising: bool,
             last: bool = False) -> int | None:
    """First (or last) frame in (start, stop] where `signal` crosses `level` in the given direction."""
    hits = []
    for frame in range(max(start + 1, 1), min(stop, len(signal) - 1) + 1):
        a, b = signal[frame - 1], signal[frame]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if (rising and a < level <= b) or (not rising and a > level >= b):
            hits.append(frame)
            if not last:
                break
    return hits[-1] if hits else None


def subframe_ms(signal: np.ndarray, times_ms: np.ndarray, frame: int, level: float) -> float:
    """Time at which `signal` reaches `level` between frame-1 and frame."""
    if frame < 1:
        return float(times_ms[frame])
    a, b = signal[frame - 1], signal[frame]
    if not (np.isfinite(a) and np.isfinite(b)) or a == b:
        return float(times_ms[frame])
    fraction = float(np.clip((level - a) / (b - a), 0, 1))
    return float(times_ms[frame - 1] + fraction * (times_ms[frame] - times_ms[frame - 1]))


def infer_view(points2d: np.ndarray, world: np.ndarray, address: int) -> tuple[str, dict]:
    shoulders = points2d[address, J["right_shoulder"], :2] - points2d[address, J["left_shoulder"], :2]
    torso = float(torso_length(points2d[address]))
    width = float(np.linalg.norm(shoulders) / torso) if torso > 0 else float("nan")
    line = world[address, J["right_shoulder"]] - world[address, J["left_shoulder"]]
    across = float(abs(line[0]) / (abs(line[0]) + abs(line[2]) + 1e-9))
    report = {"shoulder_width_over_torso": round(width, 3), "shoulder_line_across_camera_3d": round(across, 3)}
    # The 3D shoulder line is decisive when clear: an oblique rear camera (005)
    # still shows a wide 2D shoulder line, but the shoulders lie along the depth axis.
    if across >= FACE_ON_ACROSS:
        return "face_on", report
    if across <= DOWN_THE_LINE_ACROSS:
        return "down_the_line", report
    if width >= FACE_ON_WIDTH:
        return "face_on", report
    if width < DOWN_THE_LINE_WIDTH:
        return "down_the_line", report
    return "down_the_line", {**report, "reason": "ambiguous, defaulted"}


def infer_handedness(points2d: np.ndarray, signals: Signals, view: str, address: int, top: int) -> tuple[str, dict]:
    """Face-on: hands move toward the trail side (image left for a right-hander) at the top.
    Down-the-line: at address the hands are on the ball side (image right for a right-hander)."""
    if view == "face_on":
        shift = float(signals.hands_x[top] - signals.hands_x[address])
        report = {"hands_shift_address_to_top_torso": round(shift, 3)}
        if abs(shift) >= HAND_SHIFT:
            return ("right" if shift < 0 else "left"), report
    else:
        ahead = float(signals.hands_x[address])
        report = {"hands_ahead_of_hips_at_address_torso": round(ahead, 3)}
        if abs(ahead) >= HAND_SHIFT:
            return ("right" if ahead > 0 else "left"), report
    nose = (points2d[address, J["nose"], 0] - hip_center(points2d[address])[0]) / signals.torso_px
    report["nose_ahead_of_hips_torso"] = round(float(nose), 3)
    if np.isfinite(nose) and abs(nose) >= HAND_SHIFT and view != "face_on":
        return ("right" if nose > 0 else "left"), {**report, "reason": "nose direction"}
    return "right", {**report, "reason": "ambiguous, defaulted"}


def select_positions(points2d: np.ndarray, world: np.ndarray, times_ms: np.ndarray,
                     view: str = "auto", handedness: str = "auto") -> dict:
    """Frames for P1-P10 with per-position source, signal value and sub-frame time."""
    signals = swing_signals(points2d, times_ms)
    top, low, top_report = find_top(signals)
    address, address_source = find_address(signals, top)
    view_report = {"source": "argument"}
    if view == "auto":
        view, view_report = infer_view(points2d, world, address)
        view_report["source"] = "auto"
    hand_report = {"source": "argument"}
    if handedness == "auto":
        handedness, hand_report = infer_handedness(points2d, signals, view, address, top)
        hand_report["source"] = "auto"
    lead, trail = lead_trail(handedness)
    impact, impact_report = find_impact(signals, address, top, low)
    lead_arm = arm_elevation(world, lead)
    n = len(times_ms)
    detail: dict[str, dict] = {
        "p1": {"source": address_source}, "p4": {"source": "highest_hands", **top_report},
        "p7": {"source": "lowest_hands_after_top", **impact_report},
    }
    frames = {"p1": address, "p4": top, "p7": impact}

    def place(key, level, start, stop, rising, last=False, fallback=None):
        frame = crossing(lead_arm, level, start, stop, rising, last)
        if frame is not None:
            frames[key] = frame
            detail[key] = {"source": f"lead_arm_{'up' if rising else 'down'}_{level:.0f}deg",
                           "time_ms": round(subframe_ms(lead_arm, times_ms, frame, level), 1)}
        elif fallback is not None:
            frames[key] = int(fallback)
            detail[key] = {"source": "nearest_lead_arm_angle"}
        detail[key]["lead_arm_deg"] = None if key not in frames else _round(lead_arm[frames[key]])

    def nearest(level, start, stop):
        segment = lead_arm[start + 1:stop]
        if not len(segment) or not np.isfinite(segment).any():
            return (start + stop) // 2
        return start + 1 + int(np.nanargmin(np.abs(segment - level)))

    # Halfway between the arm angle at address/impact and parallel: GHUM's lead
    # arm is not exactly vertical at address or impact (005: 16 and 31 degrees).
    takeaway = _halfway(lead_arm, address)
    release = _halfway(lead_arm, impact)
    detail["levels_deg"] = {"p2": round(takeaway, 1), "p6_p8": round(release, 1), "parallel": ARM_PARALLEL_DEG}
    place("p2", takeaway, address, top, True, fallback=nearest(takeaway, address, top))
    place("p3", ARM_PARALLEL_DEG, frames["p2"], top, True, fallback=nearest(ARM_PARALLEL_DEG, frames["p2"], top))
    place("p5", ARM_PARALLEL_DEG, top, impact, False, fallback=nearest(ARM_PARALLEL_DEG, top, impact))
    # Last crossing: a one-frame dip of the 3D arm early in the downswing (005: frame 202) must not win.
    place("p6", release, frames["p5"], impact, False, last=True, fallback=nearest(release, frames["p5"], impact))
    p9 = crossing(lead_arm, ARM_PARALLEL_DEG, impact, n - 1, True)
    if p9 is None:
        raise SwingCoverageError(f"The clip ends before the follow-through: the lead arm never rises to parallel "
                                 f"after impact (frame {impact}).")
    frames["p9"] = p9
    detail["p9"] = {"source": "lead_arm_up_90deg_after_impact", "lead_arm_deg": _round(lead_arm[p9]),
                    "time_ms": round(subframe_ms(lead_arm, times_ms, p9, ARM_PARALLEL_DEG), 1)}
    frames["p8"], detail["p8"] = _release(world, lead_arm, times_ms, impact, p9, release)
    frames["p10"], detail["p10"] = _finish(signals, p9)
    for key in POSITIONS:
        detail[key].setdefault("time_ms", round(float(times_ms[frames[key]]), 1))
    detail = {**{key: detail[key] for key in POSITIONS}, "levels_deg": detail["levels_deg"]}
    adjusted = _enforce_order(frames, n)
    for key in adjusted:
        detail[key]["adjusted_for_order"] = True
    return {"positions": {key: int(frames[key]) for key in POSITIONS}, "details": detail,
            "view": view, "view_detection": view_report, "handedness": handedness,
            "handedness_detection": hand_report, "lead_side": lead,
            "limits": "body only: P2/P6/P8 use lead-arm proxies for shaft parallel, P7 uses the hands' "
                      "lowest point instead of ball contact"}


def _halfway(lead_arm: np.ndarray, frame: int) -> float:
    """Arm angle halfway from its value around `frame` (median of +-2 frames) to parallel."""
    near = lead_arm[max(0, frame - 2):frame + 3]
    base = float(np.nanmedian(near)) if np.isfinite(near).any() else 0.0
    return base + ARM_HALFWAY * (ARM_PARALLEL_DEG - base)


def _release(world: np.ndarray, lead_arm: np.ndarray, times_ms: np.ndarray, impact: int, p9: int,
             level: float) -> tuple[int, dict]:
    """First upward crossing of `level` (halfway from the impact angle to parallel) before P9.

    GHUM's elbows rarely read straight after impact (005: at most 149 degrees),
    so the elbow angle is only reported, not used to choose the frame.
    """
    window = np.arange(impact + 1, p9)
    if not len(window):
        return impact + 1, {"source": "no_frames_between_p7_and_p9"}
    elbows = np.minimum(elbow_angle(world, "left"), elbow_angle(world, "right"))
    frame = crossing(lead_arm, level, impact, p9, True)
    if frame is not None:
        report = {"source": f"lead_arm_up_{level:.0f}deg_after_impact",
                  "time_ms": round(subframe_ms(lead_arm, times_ms, frame, level), 1)}
    else:
        segment = np.abs(lead_arm[window] - level)
        if not np.isfinite(segment).any():
            return int(window[len(window) // 2]), {"source": "middle_of_p7_p9"}
        frame = int(window[int(np.nanargmin(segment))])
        report = {"source": "nearest_lead_arm_angle"}
    return frame, {**report, "lead_arm_deg": _round(lead_arm[frame]), "elbow_deg": _round(elbows[frame])}


def _finish(signals: Signals, p9: int) -> tuple[int, dict]:
    """Start of the end hold after P9: hands high and still; else the highest hands."""
    n = len(signals.times_ms)
    after = np.arange(p9 + 1, n)
    if not len(after):
        return min(p9 + 1, n - 1), {"source": "clip_end"}
    height = signals.height[after]
    peak = float(np.nanmax(height))
    high = signals.height >= peak - 0.2 * max(peak - float(signals.height[p9]), 0.1)
    still = signals.speed < STILL_SPEED
    for frame in after:
        stop = frame + _frames_within(signals.times_ms, frame, STILL_SECONDS)
        if stop <= n and high[frame] and still[frame:stop].all():
            return int(frame), {"source": "hands_high_and_still", "hand_height_torso": _round(signals.height[frame])}
    frame = int(after[int(np.nanargmax(height))])
    return frame, {"source": "highest_hands_after_p9", "hand_height_torso": _round(signals.height[frame])}


def _enforce_order(frames: dict[str, int], n: int) -> list[str]:
    """Make P1-P10 strictly increasing by pushing later positions forward; returns adjusted keys."""
    adjusted = []
    for previous, key in zip(POSITIONS, POSITIONS[1:]):
        if frames[key] <= frames[previous]:
            frames[key] = frames[previous] + 1
            adjusted.append(key)
    if frames["p10"] >= n:
        raise SwingCoverageError("Not enough frames after impact for P8-P10")
    return adjusted


def _round(value) -> float | None:
    value = float(value)
    return round(value, 2) if np.isfinite(value) else None
