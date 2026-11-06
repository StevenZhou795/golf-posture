"""Joint P2-P10 selection, sub-frame timing, view/handedness and leveling.

`find_checkpoint_frames` in swing_phases picks each position greedily from the
one before it, so one bad pick shifts every later position. `select_positions`
scores every candidate frame for every position with the same definitions
(lead arm parallel at P3/P5/P9, shaft parallel at P2/P6, straight arms then
shaft parallel at P8, hands by the head at P10) and then picks all positions of a segment together
by dynamic programming under the order constraint. P1, P4 and P7 stay the
phases found from the club and hands; they bound the three segments.

Per-position confidence is the margin between the best total cost and the
best total cost with that position forced at least two frames away.
"""

from __future__ import annotations

import numpy as np

from golf_workflows.swing_phases import ARM_PROGRESS_AT_SHAFT_PARALLEL

SHAFT, HOSEL, HEAD = 17, 18, 10
LEFT_ARM, RIGHT_ARM = (11, 12, 13), (14, 15, 16)
RIGHT_FOOT, LEFT_FOOT, RIGHT_HIP, LEFT_HIP = 3, 6, 1, 4
ROOT, THORAX = 0, 8
# Lead wrist below this confidence is treated as occluded; the trail wrist
# (same grip, a few centimetres away) is mixed in by confidence.
OCCLUDED_WRIST = 0.3
# Arm elevation cost per degree from parallel, in cost units.
DEGREES_PER_COST = 5.0
# P2/P6 from the lead arm instead of a detected shaft costs this much more.
ARM_STAND_IN_PENALTY = 2.0
# Later passes through parallel after the first one (P3, P9) cost this much per frame.
LATER_PASS_PER_FRAME = 1.0
P8_RISE_FRACTION = 0.25
P8_TIE_DEGREES = 5.0
# Arms count as straight at P8 within this of their straightest between P7 and P9.
P8_STRAIGHT_DEGREES = 10.0
# Lead arm progress from hanging (P7) to parallel (P9) when the shaft is parallel
# after impact: 0.58 (w09, labelled), 0.48 (005), 0.40 (008), reviewed frames.
ARM_PROGRESS_AT_RELEASE = 0.45
P8_MIN_ARM_PROGRESS = 0.25
P10_APEX_FRACTION = 0.05
MAX_LEVEL_DEGREES = 10.0
CONFIDENCE_SCALE = 2.0


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 1e-8, vectors / norm, np.nan)


def smooth_signal(values: np.ndarray) -> np.ndarray:
    """Savitzky-Golay (window 5, quadratic) over finite runs; symmetric, so crossings do not shift."""
    from scipy.signal import savgol_filter

    out = np.array(values, dtype=np.float64)
    finite = np.isfinite(out)
    start = 0
    while start < len(out):
        if not finite[start]:
            start += 1
            continue
        stop = start
        while stop < len(out) and finite[stop]:
            stop += 1
        if stop - start >= 5:
            out[start:stop] = savgol_filter(out[start:stop], 5, 2)
        start = stop
    return out


def hand_points(poses: np.ndarray, lead: str = "left", confidence: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lead wrist, or a confidence-weighted mix with the trail wrist where the lead wrist is occluded.

    Returns the per-frame point and a mask of frames where the trail wrist was used.
    """
    lead_wrist, trail_wrist = (13, 16) if lead == "left" else (16, 13)
    hands = np.array(poses[:, lead_wrist], dtype=np.float64)
    used = np.zeros(len(poses), dtype=bool)
    if confidence is None:
        return hands, used
    lead_conf = np.nan_to_num(np.asarray(confidence[:, lead_wrist], dtype=np.float64))
    trail_conf = np.nan_to_num(np.asarray(confidence[:, trail_wrist], dtype=np.float64))
    trail_ok = np.isfinite(poses[:, trail_wrist]).all(axis=1)
    occluded = (lead_conf < OCCLUDED_WRIST) & (trail_conf > lead_conf) & trail_ok
    lead_ok = np.isfinite(hands).all(axis=1)
    for frame in np.flatnonzero(occluded):
        if not lead_ok[frame]:
            hands[frame] = poses[frame, trail_wrist]
        else:
            weight = trail_conf[frame] / (trail_conf[frame] + lead_conf[frame])
            hands[frame] = (1 - weight) * poses[frame, lead_wrist] + weight * poses[frame, trail_wrist]
        used[frame] = True
    return hands, used


def arm_elevation(poses: np.ndarray, hands: np.ndarray, lead: str = "left") -> np.ndarray:
    """Lead-arm elevation in degrees, shoulder to hands; positive above the shoulder (camera y is down)."""
    shoulder = 11 if lead == "left" else 14
    direction = _unit_rows(hands - poses[:, shoulder])
    return np.degrees(np.arcsin(np.clip(-direction[:, 1], -1, 1)))


def shaft_tilt_degrees(poses: np.ndarray) -> np.ndarray:
    direction = _unit_rows(poses[:, HOSEL] - poses[:, SHAFT])
    return np.degrees(np.arcsin(np.clip(np.abs(direction[:, 1]), 0, 1)))


def elbow_extension(poses: np.ndarray) -> np.ndarray:
    angles = []
    for shoulder, elbow, wrist in (LEFT_ARM, RIGHT_ARM):
        upper = _unit_rows(poses[:, shoulder] - poses[:, elbow])
        lower = _unit_rows(poses[:, wrist] - poses[:, elbow])
        cosine = np.clip(np.sum(upper * lower, axis=1), -1, 1)
        angles.append(np.degrees(np.arccos(cosine)))
    return np.mean(angles, axis=0)


def ordered_dp(costs: list[np.ndarray], gap: int = 1) -> tuple[list[int | None], list[float | None]]:
    """Strictly ordered frames (at least `gap` apart) minimising the summed cost.

    `costs[k][f]` is the cost of position k at frame f (inf = not allowed).
    Positions with no finite cost are skipped (None) and the rest still ordered.
    Confidence per position is 1 - exp(-margin / CONFIDENCE_SCALE), where the
    margin is the extra total cost of moving that position two or more frames.
    """
    active = [k for k, cost in enumerate(costs) if np.isfinite(cost).any()]
    frames: list[int | None] = [None] * len(costs)
    confidence: list[float | None] = [None] * len(costs)
    if not active:
        return frames, confidence
    table = [np.asarray(costs[k], dtype=np.float64) for k in active]
    n = len(table[0])
    forward = []
    previous = None
    for cost in table:
        if previous is None:
            total = cost.copy()
        else:
            best_before = np.full(n, np.inf)
            running = np.inf
            for f in range(n):
                if f - gap >= 0:
                    running = min(running, previous[f - gap])
                best_before[f] = running
            total = cost + best_before
        forward.append(total)
        previous = total
    backward = [None] * len(table)
    following = None
    for index in range(len(table) - 1, -1, -1):
        cost = table[index]
        if following is None:
            total = cost.copy()
        else:
            best_after = np.full(n, np.inf)
            running = np.inf
            for f in range(n - 1, -1, -1):
                if f + gap < n:
                    running = min(running, following[f + gap])
                best_after[f] = running
            total = cost + best_after
        backward[index] = total
        following = total
    if not np.isfinite(forward[-1]).any():
        return frames, confidence
    # Backtrack the optimum.
    choice = [0] * len(table)
    choice[-1] = int(np.argmin(forward[-1]))
    for index in range(len(table) - 2, -1, -1):
        limit = choice[index + 1] - gap
        candidates = forward[index][: limit + 1]
        choice[index] = int(np.argmin(candidates)) if len(candidates) and np.isfinite(candidates).any() else limit
    best = float(forward[-1][choice[-1]])
    for index, k in enumerate(active):
        frames[k] = choice[index]
        with np.errstate(invalid="ignore"):  # inf - inf on frames not allowed for this position
            through = forward[index] + backward[index] - table[index]
        far = np.ones(n, dtype=bool)
        far[max(0, choice[index] - 1): choice[index] + 2] = False
        runner_up = float(np.min(through[far])) if far.any() else np.inf
        margin = runner_up - best
        confidence[k] = 1.0 if not np.isfinite(margin) else float(1 - np.exp(-max(0.0, margin) / CONFIDENCE_SCALE))
    return frames, confidence


def _crossing_time(signal: np.ndarray, frame: int, target: float = 0.0) -> float | None:
    """Fractional frame where `signal` crosses `target` next to `frame`."""
    best = None
    for a in (frame - 1, frame):
        b = a + 1
        if a < 0 or b >= len(signal) or not (np.isfinite(signal[a]) and np.isfinite(signal[b])):
            continue
        da, db = signal[a] - target, signal[b] - target
        if da == 0:
            time = float(a)
        elif da * db < 0:
            time = a + da / (da - db)
        else:
            continue
        if best is None or abs(time - frame) < abs(best - frame):
            best = time
    return best


def _extremum_time(signal: np.ndarray, frame: int) -> float | None:
    """Parabola vertex through (frame-1, frame, frame+1), clipped to half a frame."""
    if frame <= 0 or frame >= len(signal) - 1:
        return None
    a, b, c = signal[frame - 1], signal[frame], signal[frame + 1]
    if not np.isfinite([a, b, c]).all():
        return None
    denominator = a - 2 * b + c
    if abs(denominator) < 1e-9:
        return float(frame)
    return float(frame + np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))


def _band(n: int, start: int, stop: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[max(0, start): max(0, min(n, stop + 1))] = True
    return mask


def _crossing_cost(elevation: np.ndarray, allowed: np.ndarray, rising: bool) -> np.ndarray:
    """|elevation| per frame, plus a per-frame penalty after the first pass through parallel."""
    cost = np.where(allowed & np.isfinite(elevation), np.abs(elevation) / DEGREES_PER_COST, np.inf)
    frames = np.flatnonzero(allowed & np.isfinite(elevation))
    first = None
    for a, b in zip(frames, frames[1:]):
        if (elevation[a] < 0 <= elevation[b]) if rising else (elevation[a] > 0 >= elevation[b]):
            first = b
            break
    if first is not None:
        later = np.arange(len(cost)) > first
        cost = cost + np.where(later, LATER_PASS_PER_FRAME * (np.arange(len(cost)) - first), 0.0)
    return cost


def _shaft_or_arm_cost(elevation: np.ndarray, tilt: np.ndarray, observed: np.ndarray | None, allowed: np.ndarray,
                       hanging: float | None,
                       progress: float = ARM_PROGRESS_AT_SHAFT_PARALLEL) -> tuple[np.ndarray, np.ndarray]:
    """Cost of shaft parallel per frame: detected shaft tilt, else lead arm `progress` of the way to parallel.

    Returns the cost and a per-frame flag for frames where the shaft was cheaper.
    """
    n = len(elevation)
    shaft = np.full(n, np.inf)
    if observed is not None:
        usable = allowed & observed & np.isfinite(tilt)
        shaft[usable] = tilt[usable] / DEGREES_PER_COST
    elif np.isfinite(tilt).any():
        usable = allowed & np.isfinite(tilt)
        shaft[usable] = tilt[usable] / DEGREES_PER_COST
    arm = np.full(n, np.inf)
    if hanging is not None and np.isfinite(hanging):
        target = hanging + progress * (0.0 - hanging)
        usable = allowed & np.isfinite(elevation)
        arm[usable] = np.abs(elevation[usable] - target) / DEGREES_PER_COST + ARM_STAND_IN_PENALTY
    return np.minimum(shaft, arm), shaft <= arm


def select_positions(
    poses: np.ndarray,
    address: int | None,
    top: int | None,
    impact: int | None,
    lead: str = "left",
    club_observed: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    reserve_positions: bool = True,
    image_points: np.ndarray | None = None,
) -> tuple[dict[str, int | None], dict]:
    """P1-P10 with per-position confidence, sources and sub-frame times.

    `image_points` (face-on only; 2D pixels padded with z = 0, NaN where not
    detected) replaces the lifted lead arm for P9: face-on, the lead arm points
    at the target there, in the picture plane, so its observed angle is exact,
    while the lifted one lags by about a frame (008). At P3/P5 the lead arm lies
    across the chest, foreshortened with the wrist by the head, and the lifted
    arm is better (010). Down the line the 2D arm is foreshortened throughout.
    """
    positions = {f"p{i}": None for i in range(1, 11)}
    positions.update(p1=address, p4=top, p7=impact)
    details: dict = {"method": "joint_dp", "confidence": {}, "subframe": {}, "sources": {}}
    n = len(poses)
    if address is None or top is None or impact is None or not (0 <= address <= top <= impact < n):
        details["method"] = "joint_dp_unavailable"
        return positions, details
    reserve = int(reserve_positions)
    hands, trail_used = hand_points(poses, lead, confidence)
    details["trail_wrist_frames"] = [int(f) for f in np.flatnonzero(trail_used)]
    elevation = smooth_signal(arm_elevation(poses, hands, lead))
    crossing = elevation
    if image_points is not None:
        image_hands, _ = hand_points(image_points, lead, confidence)
        observed_arm = arm_elevation(image_points, image_hands, lead)
        crossing = smooth_signal(np.where(np.isfinite(observed_arm), observed_arm, elevation))
        details["p9_signal"] = "image_lead_arm"
    tilt = smooth_signal(shaft_tilt_degrees(poses))
    extension = smooth_signal(elbow_extension(poses))
    observed = None if club_observed is None else np.asarray(club_observed, dtype=bool)
    shoulder = 11 if lead == "left" else 14
    finite = np.flatnonzero(np.isfinite(poses).all(axis=(1, 2)))
    last = int(finite[-1]) if reserve and len(finite) else n - 1

    def hanging_at(frame: int) -> float | None:
        values = [elevation[f] for f in (frame - 1, frame, frame + 1) if 0 <= f < n and np.isfinite(elevation[f])]
        return float(np.mean(values)) if values else None

    # Backswing: P2 then P3, strictly between P1 and P4.
    back = _band(n, address + 1, top - 1)
    p3_cost = _crossing_cost(elevation, _band(n, address + 1 + reserve, top - 1), rising=True)
    p2_cost, p2_shaft = _shaft_or_arm_cost(elevation, tilt, observed, back, hanging_at(address))
    # Downswing: P5 then P6, strictly between P4 and P7.
    down = _band(n, top + 1, impact - 1)
    p5_cost = _crossing_cost(elevation, _band(n, top + 1, impact - 1 - reserve), rising=False)
    p6_cost, p6_shaft = _shaft_or_arm_cost(elevation, tilt, observed, down, hanging_at(impact))
    # Follow-through: P8, P9, P10 after P7.
    after = _band(n, impact + 1 + reserve, last)
    p9_cost = _crossing_cost(crossing, after, rising=True)
    height = -(hands[:, 1] - poses[:, shoulder, 1])  # hands above the shoulder, up positive
    risen = np.zeros(n, dtype=bool)
    p9_guess = int(np.argmin(p9_cost)) if np.isfinite(p9_cost).any() else None
    if p9_guess is not None and np.isfinite(height[impact]) and np.isfinite(height[p9_guess]):
        floor = height[impact] + P8_RISE_FRACTION * (height[p9_guess] - height[impact])
        risen = np.isfinite(height) & (height >= floor)
    # P8: arms straight first, then the shaft parallel to the ground. The arms
    # must have started up (a shaft read as parallel while they still hang is a
    # misdetection); with the shaft unseen, the lead arm stands in at the height
    # it has when the shaft is parallel.
    hanging_p7 = hanging_at(impact)
    progress = np.full(n, np.nan)
    if hanging_p7 is not None and hanging_p7 < 0:
        progress = (elevation - hanging_p7) / (0.0 - hanging_p7)
    release = (_band(n, impact + 1, p9_guess - 1 if p9_guess is not None else last) & np.isfinite(extension)
               & np.isfinite(progress) & (progress >= P8_MIN_ARM_PROGRESS))
    p8_cost = np.full(n, np.inf)
    p8_shaft = np.zeros(n, dtype=bool)
    if release.any():
        best = float(np.max(extension[release]))
        straight = release & (extension >= best - P8_STRAIGHT_DEGREES)
        shortfall = np.maximum(0.0, best - extension - P8_TIE_DEGREES)
        parallel, p8_shaft = _shaft_or_arm_cost(elevation, tilt, observed, straight, hanging_p7,
                                                ARM_PROGRESS_AT_RELEASE)
        p8_cost = np.where(straight, parallel + shortfall / DEGREES_PER_COST, np.inf)
        details["p8_straightest_deg"] = round(best, 1)
    if not np.isfinite(p8_cost).any():
        p8_allowed = _band(n, impact + 1, last) & risen & np.isfinite(extension)
        if p8_allowed.any():
            best = float(np.max(extension[p8_allowed]))
            shortfall = np.maximum(0.0, best - extension - P8_TIE_DEGREES)
            p8_cost = np.where(p8_allowed, shortfall / DEGREES_PER_COST, np.inf)
    # P10: hands at their highest after P9 (within 5% of the rise since impact), above the shoulder,
    # nearest the head.
    p10_cost = np.full(n, np.inf)
    post = np.flatnonzero(after & np.isfinite(height))
    if len(post) and np.isfinite(height[impact]):
        apex = post[int(np.argmax(height[post]))]
        floor = height[apex] - P10_APEX_FRACTION * (height[apex] - height[impact])
        block_start = block_end = apex
        while block_start - 1 > impact and np.isfinite(height[block_start - 1]) and height[block_start - 1] >= floor:
            block_start -= 1
        while block_end + 1 <= last and np.isfinite(height[block_end + 1]) and height[block_end + 1] >= floor:
            block_end += 1
        block = _band(n, block_start, block_end) & (height > 0)
        scale = np.nanmedian(np.linalg.norm(poses[:, THORAX] - poses[:, ROOT], axis=1))
        distance = np.linalg.norm(hands - poses[:, HEAD], axis=1) / (scale if scale > 0 else 1.0)
        p10_cost = np.where(block & np.isfinite(distance), distance * DEGREES_PER_COST, np.inf)

    segments = (("p2", "p3"), (p2_cost, p3_cost)), (("p5", "p6"), (p5_cost, p6_cost)), \
               (("p8", "p9", "p10"), (p8_cost, p9_cost, p10_cost))
    for names, costs in segments:
        frames, confidences = ordered_dp(list(costs))
        for name, frame, score in zip(names, frames, confidences):
            positions[name] = frame
            details["confidence"][name] = None if score is None else round(score, 3)
    for name, flags in (("p2", p2_shaft), ("p6", p6_shaft), ("p8", p8_shaft)):
        frame = positions[name]
        if frame is not None:
            details["sources"][name] = ("observed_shaft" if observed is not None else "shaft") if flags[frame] \
                else "lead_arm_progress"
    # Sub-frame times on the smoothed signals; frames stay the nearest index.
    for name, signal in (("p3", elevation), ("p5", elevation), ("p9", crossing)):
        if positions[name] is not None:
            details["subframe"][name] = _crossing_time(signal, positions[name])
    for name, signal in (("p2", tilt), ("p6", tilt), ("p8", tilt)):
        frame = positions[name]
        if frame is None:
            continue
        if details["sources"].get(name) == "lead_arm_progress":
            hanging = hanging_at(address if name == "p2" else impact)
            share = ARM_PROGRESS_AT_RELEASE if name == "p8" else ARM_PROGRESS_AT_SHAFT_PARALLEL
            target_value = hanging + share * (0.0 - hanging)
            details["subframe"][name] = _crossing_time(elevation, frame, target_value)
        else:
            details["subframe"][name] = _extremum_time(signal, frame)
    if positions["p10"] is not None:
        details["subframe"]["p10"] = _extremum_time(np.where(np.isfinite(p10_cost), p10_cost, np.nan),
                                                    positions["p10"])
    details["subframe"] = {key: None if value is None else round(float(value), 3)
                           for key, value in details["subframe"].items()}
    return positions, details


def subframe_ms(subframe: dict, fps: float) -> dict:
    return {key: None if value is None else round(1000.0 * value / fps, 1) for key, value in subframe.items()}


def _early_frames(points: np.ndarray, valid: np.ndarray, joints: tuple[int, ...], count: int = 10) -> np.ndarray:
    good = np.flatnonzero(np.all(valid[:, list(joints)], axis=1))
    return good[:count]


def infer_view(points: np.ndarray, valid: np.ndarray, default: str = "down_the_line") -> tuple[str, dict]:
    """Face-on vs down-the-line from the first confident frames (the address).

    Face-on shows both shoulders apart (width / torso >= 0.35) and a near
    horizontal line between the feet. Down the line shows the shoulders nearly
    stacked (< 0.25) or a steep feet line, because the feet are at different depths.
    """
    frames = _early_frames(points, valid, (LEFT_ARM[0], RIGHT_ARM[0], ROOT, THORAX, LEFT_FOOT, RIGHT_FOOT))
    if len(frames) == 0:
        return default, {"source": "default", "reason": "no confident address frames"}
    torso = np.median(np.linalg.norm(points[frames, THORAX] - points[frames, ROOT], axis=1))
    width = np.median(np.linalg.norm(points[frames, LEFT_ARM[0]] - points[frames, RIGHT_ARM[0]], axis=1))
    feet = np.median(points[frames, LEFT_FOOT] - points[frames, RIGHT_FOOT], axis=0)
    ratio = float(width / torso) if torso > 0 else float("nan")
    feet_angle = float(np.degrees(np.arctan2(abs(feet[1]), abs(feet[0]))))
    evidence = {"shoulder_width_over_torso": round(ratio, 3), "feet_line_from_horizontal_deg": round(feet_angle, 1)}
    if ratio >= 0.35 and feet_angle < 25:
        return "face_on", {"source": "auto", **evidence}
    if ratio < 0.25 or feet_angle >= 30:
        return "down_the_line", {"source": "auto", **evidence}
    return default, {"source": "default", "reason": "ambiguous", **evidence}


def _high_hand_peaks(hands: np.ndarray, both: np.ndarray, address: np.ndarray, torso: float) -> list[int]:
    """Frames where the hands peak at least halfway from address height to their maximum, in time order.

    Peaks must stand out by half a torso length, so waggles and jitter are ignored.
    """
    from scipy.signal import find_peaks

    frames = np.flatnonzero(both)
    if len(frames) < 3:
        return []
    height = np.interp(np.arange(len(hands)), frames, -hands[frames, 1])
    base = float(np.median(height[address]))
    tallest = float(np.max(height))
    found, _ = find_peaks(height, prominence=0.5 * torso)
    found = [int(f) for f in found if height[f] >= base + 0.5 * (tallest - base) and both[f]]
    if not found:
        found = [int(np.argmax(np.where(both, height, -np.inf)))]
    return found


def infer_handedness(points: np.ndarray, valid: np.ndarray, view: str, default: str = "right") -> tuple[str, dict]:
    """Right- or left-handed from the 2D hands.

    Down the line (camera behind the hands): the hands sit toward the ball,
    image right of the hips for a right-hander. Face-on: at the top of the
    backswing the hands have moved to the golfer's trail side, image left for a
    right-hander, and at the finish to the lead side. The finish is often the
    highest the hands get, so the top is the first high peak, not the highest.
    """
    frames = _early_frames(points, valid, (ROOT, THORAX, 13, 16))
    if len(frames) == 0:
        return default, {"source": "default", "reason": "no confident address frames"}
    torso = float(np.median(np.linalg.norm(points[frames, THORAX] - points[frames, ROOT], axis=1)))
    if torso <= 0:
        return default, {"source": "default", "reason": "degenerate torso"}
    hands = (points[:, 13] + points[:, 16]) / 2.0
    both = valid[:, 13] & valid[:, 16]
    if view == "face_on":
        peaks = _high_hand_peaks(hands, both, frames, torso)
        if not peaks:
            return default, {"source": "default", "reason": "no confident top"}
        top = peaks[0]
        shift = float((hands[top, 0] - np.median(hands[frames, 0])) / torso)
        evidence = {"top_hands_shift_over_torso": round(shift, 3), "top_frame": top}
        sign = -shift
        if len(peaks) > 1:
            finish = peaks[-1]
            travel = float((hands[finish, 0] - hands[top, 0]) / torso)
            evidence.update(finish_frame=finish, top_to_finish_shift_over_torso=round(travel, 3))
            sign = travel
    else:
        shift = float(np.median(hands[frames, 0] - points[frames, ROOT, 0]) / torso)
        evidence = {"address_hands_ahead_over_torso": round(shift, 3)}
        sign = shift
    if sign >= 0.2 if view != "face_on" else sign >= 0.3:
        return "right", {"source": "auto", **evidence}
    if sign <= -0.2 if view != "face_on" else sign <= -0.3:
        return "left", {"source": "auto", **evidence}
    return default, {"source": "default", "reason": "ambiguous", **evidence}


def ground_roll(points: np.ndarray, valid: np.ndarray, view: str) -> tuple[float, dict]:
    """Camera roll in degrees from the feet line at address (face-on only).

    Face-on, both feet stand on the ground at the same depth, so their line is
    horizontal in a level picture. Down the line the feet are at different
    depths and their line says nothing about roll, so no correction is made.
    """
    if view != "face_on":
        return 0.0, {"applied": False, "reason": "only measured face-on"}
    frames = _early_frames(points, valid, (LEFT_FOOT, RIGHT_FOOT))
    if len(frames) == 0:
        return 0.0, {"applied": False, "reason": "no confident feet"}
    feet = np.median(points[frames, LEFT_FOOT] - points[frames, RIGHT_FOOT], axis=0)
    if feet[0] < 0:
        feet = -feet
    roll = float(np.degrees(np.arctan2(feet[1], feet[0])))
    if abs(roll) > MAX_LEVEL_DEGREES:
        return 0.0, {"applied": False, "reason": "feet line too steep for camera roll", "feet_roll_deg": round(roll, 2)}
    return roll, {"applied": True, "roll_deg": round(roll, 2)}


def level_poses(poses: np.ndarray, roll_deg: float) -> np.ndarray:
    """Undo a camera roll of `roll_deg` (picture rotated clockwise is positive) about the camera axis."""
    if roll_deg == 0:
        return poses
    angle = np.radians(-roll_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return (poses.astype(np.float64) @ rotation.T).astype(poses.dtype)
