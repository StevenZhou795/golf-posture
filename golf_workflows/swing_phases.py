"""Swing phases from a clubhead path, with a wrist fallback.

The idea follows GolfPosePro (MediaPipe wrist trajectory, a short moving
average, then velocity): address is the start of motion, the top is the
point farthest from address, impact is the fast return to that spot, and
follow-through is when the club slows down again. The clubhead is the
signal when it exists. Wrist height is used only when the club is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

MIN_SAMPLES = 8
MOTION_FRACTION = 0.12
IMPACT_NEAR = 0.35
DECAY_FRACTION = 0.35
SUSTAIN_FRAMES = 3
# Club frames below this landmark confidence are treated as unobserved when a
# position depends on the club (blurred shafts typically score about 0.2).
RELIABLE_CLUB_CONFIDENCE = 0.3
# Fraction of the lead arm's rotation from hanging to parallel at which the
# shaft is parallel (P2 on the way back, P6 on the way down).
ARM_PROGRESS_AT_SHAFT_PARALLEL = 0.3
IMPACT_BEFORE_HANDS_SECONDS = 0.1
IMPACT_AFTER_HANDS_SECONDS = 0.3


@dataclass
class SwingPhases:
    address: int | None
    top: int | None
    impact: int | None
    follow_through: int | None
    signal: str

    def as_dict(self) -> dict:
        return {
            "address": self.address,
            "top": self.top,
            "impact": self.impact,
            "follow_through": self.follow_through,
            "signal": self.signal,
        }


def smooth_window(fps: float) -> int:
    """About 0.15 s, at least five frames, and always odd."""
    size = int(round(max(5.0, 0.15 * float(fps or 30.0))))
    if size % 2 == 0:
        size += 1
    return size


def smooth_series(values: np.ndarray, fps: float) -> np.ndarray:
    """Moving average. NaNs are filled first so the filter does not punch holes."""
    series = np.asarray(values, dtype=np.float64).copy()
    if series.ndim != 1 or len(series) < 3:
        return series
    good = np.isfinite(series)
    if good.sum() < 2:
        return series
    frames = np.arange(len(series))
    series = np.interp(frames, frames[good], series[good])
    window = min(smooth_window(fps), len(series) if len(series) % 2 else len(series) - 1)
    if window < 3:
        return series
    return uniform_filter1d(series, size=window, mode="nearest")


def smooth_keypoints(points: np.ndarray, fps: float) -> np.ndarray:
    """Smooth each joint independently. Shape is (frames, joints, 2)."""
    smoothed = np.asarray(points, dtype=np.float64).copy()
    for joint in range(smoothed.shape[1]):
        for axis in range(smoothed.shape[2]):
            smoothed[:, joint, axis] = smooth_series(smoothed[:, joint, axis], fps)
    return smoothed


def _sustain(mask: np.ndarray, count: int) -> int | None:
    run = 0
    for index, flag in enumerate(mask):
        run = run + 1 if flag else 0
        if run >= count:
            return index - count + 1
    return None


def _prepare_track(track: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return smoothed xy, speed in px/frame, and a finite-count check."""
    points = np.asarray(track, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    finite = np.isfinite(points).all(axis=1)
    if int(finite.sum()) < MIN_SAMPLES:
        return None
    frames = np.arange(len(points))
    filled = np.column_stack(
        [
            np.interp(frames, frames[finite], points[finite, 0]),
            np.interp(frames, frames[finite], points[finite, 1]),
        ]
    )
    smoothed = np.column_stack([smooth_series(filled[:, 0], fps), smooth_series(filled[:, 1], fps)])
    velocity = np.gradient(smoothed, axis=0)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    return smoothed, speed, finite


def detect_phases(
    clubhead: np.ndarray | None,
    wrist_y: np.ndarray | None,
    fps: float,
) -> SwingPhases:
    """Label address, top, impact, and follow-through on one swing."""
    empty = SwingPhases(None, None, None, None, "none")
    prepared = None
    signal = "clubhead"
    if clubhead is not None:
        prepared = _prepare_track(clubhead, fps)
    if prepared is None and wrist_y is not None:
        wrist = np.asarray(wrist_y, dtype=np.float64)
        track = np.column_stack([np.zeros(len(wrist)), wrist])
        prepared = _prepare_track(track, fps)
        signal = "wrist"
    if prepared is None:
        return empty

    smoothed, speed, _finite = prepared
    peak = float(np.max(speed))
    if peak <= 1e-6:
        return empty
    motion = speed >= max(1.5, MOTION_FRACTION * peak)
    onset = _sustain(motion, SUSTAIN_FRAMES)
    if onset is None:
        return SwingPhases(None, None, None, None, signal)
    address = max(0, onset - 1)
    origin = smoothed[address]
    distance = np.linalg.norm(smoothed - origin, axis=1)
    farthest = float(np.max(distance[address:]))
    if farthest <= 1e-6:
        return SwingPhases(address, None, None, None, signal)

    # The club has to travel away from address before a fast return can be impact.
    raised_at = np.flatnonzero(np.arange(len(distance)) > address)
    raised_at = raised_at[distance[raised_at] >= 0.55 * farthest]
    raised = int(raised_at[0]) if len(raised_at) else address
    after_raise = np.arange(len(speed)) >= raised
    near = distance <= IMPACT_NEAR * farthest
    fast = speed >= 0.45 * peak
    candidates = np.flatnonzero(after_raise & near & fast)
    if len(candidates):
        impact = int(candidates[np.argmax(speed[candidates])])
    else:
        later = np.flatnonzero(after_raise)
        impact = int(later[np.argmax(speed[later])]) if len(later) else raised
    if impact <= address:
        impact = min(len(speed) - 1, address + 1)
    top_slice = distance[address:impact + 1]
    top = int(address + np.argmax(top_slice)) if len(top_slice) else address

    decay_at = None
    if impact + 1 < len(speed):
        quiet = speed < DECAY_FRACTION * float(speed[impact])
        quiet[: impact + 1] = False
        decay_at = _sustain(quiet, SUSTAIN_FRAMES)
    follow = decay_at if decay_at is not None else len(speed) - 1
    follow = max(follow, min(len(speed) - 1, impact + 1))
    return SwingPhases(address, top, impact, follow, signal)


def observed_club_track(points, valid):
    """Use a single observed landmark consistently: hosel, then shaft."""
    for joint, name in ((18, 'hosel'), (17, 'shaft')):
        good = valid[:, joint] & np.isfinite(points[:, joint]).all(axis=1)
        if good.sum() >= MIN_SAMPLES:
            track = points[:, joint].copy()
            track[~good] = np.nan
            return track, name
    return np.full((len(points), 2), np.nan), 'unavailable'


def _club_reversal(points, reliable, wrist_top, impact, fps, scale):
    """Top of the backswing as the frame where the shaft stops rotating back.

    The hands often peak before the club finishes loading, so the 2D shaft
    angle (grip to hosel) is searched within 0.15 s of the wrist peak. Only
    reliable detections with a readable projected length count, and the turn
    must be inside the window on both sides; otherwise None.
    """
    radius = max(2, int(round(.15 * fps)))
    vector = points[:, 18] - points[:, 17]
    length = np.linalg.norm(vector, axis=1)
    usable = reliable & (length >= .4 * scale)
    before = [f for f in range(max(0, wrist_top - 4 * radius), wrist_top + 1) if usable[f]]
    window = [f for f in range(max(1, wrist_top - radius), min(impact - 2, wrist_top + radius) + 1) if usable[f]]
    if len(before) < 3 or len(window) < 3:
        return None
    frames = sorted(set(before) | set(window))
    theta = dict(zip(frames, np.degrees(np.unwrap(np.arctan2(vector[frames, 1], vector[frames, 0])))))
    direction = np.sign(np.polyfit(before, [theta[f] for f in before], 1)[0])
    if direction == 0:
        return None
    turn = max(window, key=lambda f: direction * theta[f])
    earlier = [direction * theta[f] for f in window if f < turn]
    later = [direction * theta[f] for f in window if f > turn]
    peak = direction * theta[turn]
    if not earlier or not later or peak - min(earlier) < 3 or peak - min(later) < 3:
        return None
    return int(turn)


def _club_rest_end(club, reliable, origin, top, fps):
    """Last reliable frame before the top with the clubhead still at its address spot.

    `club` is the body-relative clubhead; the rest radius is a tenth of its
    backswing excursion (at least 0.1 torso). Needs two reliable rest frames
    within 0.3 s, so a single noisy detection cannot set the address. A
    detection that jumps more than 0.6 torso per frame from the last accepted
    clubhead is a misdetection (takeaway is slow) and is ignored.
    """
    if top is None or top < 3:
        return None
    displacement = np.linalg.norm(club[:top] - origin, axis=1)
    observed = reliable[:top] & np.isfinite(displacement)
    accepted = None
    for frame in np.flatnonzero(observed):
        if accepted is not None and np.linalg.norm(club[frame] - club[accepted]) > .6 * (frame - accepted):
            observed[frame] = False
        else:
            accepted = frame
    if not observed.any():
        return None
    excursion = float(np.max(displacement[observed]))
    threshold = max(.1, .1 * excursion)
    resting = np.flatnonzero(observed & (displacement < threshold))
    moving = np.flatnonzero(observed & (displacement >= threshold))
    if not len(resting) or not len(moving):
        return None
    candidates = resting[resting < moving[-1]]
    if not len(candidates):
        return None
    last = int(candidates[-1])
    span = max(2, int(round(.3 * fps)))
    if np.count_nonzero((resting > last - span) & (resting <= last)) < 2:
        return None
    return last


def impact_window(wrist_impact, top, fps, frames):
    """Frames searched for contact around the hands' low point.

    With forward shaft lean (lead wrist bowed, clubhead trailing the hands) the
    hands bottom out before the clubhead reaches the ball, so the window reaches
    further after the hands' low point (0.3 s) than before it (0.1 s).
    """
    lower = max(top + 1, wrist_impact - max(2, int(round(IMPACT_BEFORE_HANDS_SECONDS * fps))))
    upper = min(frames - 1, wrist_impact + max(3, int(round(IMPACT_AFTER_HANDS_SECONDS * fps))))
    return int(lower), int(max(lower, upper))


def detect_refined_phases(points, valid, filled, fps, confidence=None, impact_hint=None, impact_hint_source='ball'):
    """Use refined club geometry and body-relative wrists before 3D lifting.

    Wrist rise/return corroborates club impact candidates. Missing club samples
    never become observations. A sustained finish hold, rather than the first
    speed dip, ends the interval. This remains a single-swing heuristic.
    """
    from golf_workflows.club_refinement import prepare_club_inputs
    fps = max(float(fps), 1.0)
    n = len(points)
    wrists = (filled[:,13] + filled[:,16]) / 2
    torso = np.linalg.norm(filled[:,8]-filled[:,0],axis=1)
    body_good = valid[:,0] & valid[:,8] & (valid[:,13] | valid[:,16])
    scale = float(np.median(torso[body_good])) if body_good.any() else 0.0
    raw_club, track_source = observed_club_track(points, valid)
    details = dict(method='refined_club_with_wrist_corroboration',
                   original_club_observations=int(valid[:,18].sum()),
                   original_shaft_observations=int(valid[:,17].sum()), club_track_source=track_source,
                   body_observations=int(body_good.sum()),body_scale_px=scale,
                   ball_impact_hint=impact_hint,ball_impact_hint_accepted=False)
    if body_good.sum() < MIN_SAMPLES or scale < 1e-6:
        result=detect_phases(raw_club,None,fps)
        # Without wrist corroboration, keep uncertain post-impact frames rather
        # than treating a short club speed dip as proof the swing has ended.
        if result.impact is not None:
            result.follow_through=n-1
        details.update(fallback='insufficient_body_observations',finish_reason='clip_end_uncertain')
        return result, details
    phase_valid = valid.copy()
    near_hands = np.linalg.norm(points[:,17]-wrists,axis=1) <= 0.8*scale
    phase_valid[:,17:19] &= near_hands[:,None]
    club_filled,weights=prepare_club_inputs(points,filled,phase_valid,confidence)
    club = (club_filled[:,18]-filled[:,0])/scale
    if track_source == 'shaft':
        club = (filled[:,17]-filled[:,0])/scale
    hands = (wrists-filled[:,0])/scale
    hands = np.column_stack([smooth_series(hands[:,i],fps) for i in range(2)])
    height = -hands[:,1]
    speed = np.linalg.norm(np.gradient(hands,axis=0),axis=1)*fps
    span = float(np.ptp(height))
    details['reliable_club_pairs']=int((weights>0).sum())
    if span < 0.15:
        result=detect_phases(raw_club,None,fps)
        if result.impact is not None: result.follow_through=n-1
        details.update(fallback='insufficient_wrist_height_change',finish_reason='clip_end_uncertain')
        return result,details
    peaks,_=find_peaks(height,prominence=.25*span)
    troughs,_=find_peaks(-height,prominence=.25*span)
    candidates=[]
    for top in peaks:
        after=troughs[troughs>top+1]
        if len(after):
            impact=int(after[0])
            candidates.append((float(height[top]-height[impact]),int(top),impact))
    if not candidates:
        result=detect_phases(club*scale,None,fps)
        if result.impact is not None: result.follow_through=n-1
        details.update(fallback='no_corroborated_wrist_rise_and_return',finish_reason='clip_end_uncertain')
        return result,details
    _,top,wrist_impact=max(candidates)
    wrist_top=top
    reliable=(weights>=RELIABLE_CLUB_CONFIDENCE)&np.isfinite(points[:,17:19]).all(axis=(1,2))
    club_top=_club_reversal(points,reliable,wrist_top,wrist_impact,fps,scale)
    if club_top is not None:
        top=club_top
    # Contact comes first: a ball-based hint (ball leaving its spot, or the
    # clubhead at the ball), then the clubhead back at its address spot, and only
    # then the hands' low point. All are limited to the window around the hands.
    impact=wrist_impact
    impact_source='wrist_return'
    lower,upper=impact_window(wrist_impact,top,fps,n)
    ids=np.arange(n)
    observed=(weights>0)&(ids>=lower)&(ids<=upper)
    baseline_count=max(2,min(8,top//3))
    origin=np.median(club[:baseline_count],axis=0)
    if observed.any():
        options=np.flatnonzero(observed)
        distances=np.linalg.norm(club[options]-origin,axis=1)
        choice=int(np.argmin(distances))
        excursion=float(np.max(np.linalg.norm(club[:top+1]-origin,axis=1)))
        if distances[choice] <= max(.15,.35*excursion):
            impact=int(options[choice])
            impact_source='clubhead_return_to_address'
    if impact_hint is not None and lower <= impact_hint <= upper:
        impact=int(impact_hint)
        impact_source=impact_hint_source
        details['ball_impact_hint_accepted']=True
    details.update(impact_source=impact_source,impact_window=[lower,upper])
    hand_origin=np.median(hands[:baseline_count],axis=0)
    distance=np.linalg.norm(hands-hand_origin,axis=1)
    threshold=max(.06,.1*float(distance[:top+1].max()))
    moving=(distance>=threshold)&(ids<top)
    onset=_sustain(moving,max(2,int(round(.06*fps))))
    address=max(0,(onset if onset is not None else top-1)-int(round(.1*fps)))
    hand_address=address
    # The clubhead resting behind the ball is a cleaner start than hand motion:
    # the clubhead moves first and furthest. The last reliable rest frame before
    # the top skips earlier waggles.
    club_rest=_club_rest_end(club,reliable,origin,top,fps)
    if club_rest is not None:
        address=max(0,min(club_rest+1-int(round(.1*fps)),top-2))
    # Finish requires raised hands plus sustained low wrist speed. Do not use
    # instantaneous speed at impact, where vertical velocity may reverse.
    peak_speed=float(np.percentile(speed,95))
    quiet_threshold=max(.08,.18*peak_speed)
    hold=max(3,int(round(.25*fps)))
    risen=height >= height[wrist_impact]+.5*(height[top]-height[wrist_impact])
    quiet=(speed<quiet_threshold)&risen&(ids>impact+max(2,int(round(.15*fps))))&body_good
    settled=_sustain(quiet,hold)
    # P10 is the held finish with the hands up by the lead ear. The first quiet
    # moment can be the pause where the hands stop rising and wrap around the
    # head, so the window stays open until the raised hands come back down.
    lowered=None
    if settled is None:
        follow=n-1
        finish_reason='clip_end_no_sustained_hold'
    else:
        lowered=_sustain(~risen[settled:],max(2,int(round(.1*fps))))
        follow=n-1 if lowered is None else settled+lowered-1
        finish_reason='raised_hands_lowered' if lowered is not None else 'clip_end_hands_still_raised'
    follow=max(follow,impact+1)
    follow=min(follow,n-1)
    details.update(wrist_top=wrist_top,club_top=club_top,top_source='club_reversal' if club_top is not None else 'wrist_peak',
                   hand_address=hand_address,club_rest_end=club_rest,
                   address_source='club_rest' if club_rest is not None else 'hand_motion_onset',
                   wrist_impact=wrist_impact,quiet_threshold_body_units_per_s=quiet_threshold,
                   finish_hold_frames=hold,finish_hold_start=settled,
                   hands_lowered_at=None if lowered is None else int(settled+lowered),
                   finish_reason=finish_reason)
    return SwingPhases(address,top,impact,follow,'refined_club+wrist'),details


def _finite_row(points: np.ndarray) -> bool:
    return bool(np.isfinite(points).all())


def _unit(start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        return None
    return vector / length


def _up_component(direction: np.ndarray) -> float:
    """Height above the ground. Camera Y points down, so up is -Y.

    The 3D picture draws the ground as its XY plane. A bone parallel to that
    plane has an up-component near zero.
    """
    return abs(float(direction[1]))


def _first_parallel(
    poses: np.ndarray,
    start: int,
    end: int,
    joint_a: int,
    joint_b: int,
    threshold: float,
) -> int | None:
    """First frame in (start, end) where the bone is near parallel to the ground."""
    stop = min(end, len(poses) - 1)
    for frame in range(max(0, start) + 1, stop):
        if not (_finite_row(poses[frame, joint_a]) and _finite_row(poses[frame, joint_b])):
            continue
        direction = _unit(poses[frame, joint_a], poses[frame, joint_b])
        if direction is not None and _up_component(direction) <= threshold:
            return frame
    return None


def _closest_horizontal(poses: np.ndarray, start: int, end: int, joint_a: int, joint_b: int,
                        allowed: np.ndarray | None = None) -> int | None:
    best = None
    best_tilt = float("inf")
    stop = min(end, len(poses) - 1)
    for frame in range(max(0, start), stop + 1):
        if allowed is not None and not allowed[frame]:
            continue
        if not (_finite_row(poses[frame, joint_a]) and _finite_row(poses[frame, joint_b])):
            continue
        direction = _unit(poses[frame, joint_a], poses[frame, joint_b])
        if direction is None:
            continue
        tilt = abs(float(direction[1]))
        if tilt < best_tilt:
            best_tilt = tilt
            best = frame
    return best


def _arm_crossing(
    poses: np.ndarray,
    start: int,
    end: int,
    shoulder: int,
    wrist: int,
    rising: bool,
    threshold: float = 0.25,
) -> int | None:
    """Lead arm passing through shoulder height, in [start, end].

    A tolerance test alone takes the first frame inside the band, which at
    low frame rates can still be well below (rising) or above (falling)
    the shoulder. Take the sign change instead, then keep whichever frame of
    the pair is closer to horizontal. Falls back to the tolerance test when
    the arm starts on the far side or never crosses.
    """
    stop = min(end, len(poses) - 1)
    previous = None
    for frame in range(max(0, start), stop + 1):
        if not (_finite_row(poses[frame, shoulder]) and _finite_row(poses[frame, wrist])):
            continue
        direction = _unit(poses[frame, shoulder], poses[frame, wrist])
        if direction is None:
            continue
        tilt = abs(float(direction[1]))
        above = float(poses[frame, wrist, 1]) <= float(poses[frame, shoulder, 1])
        if above == rising:
            if previous is None:
                break
            return previous[0] if previous[1] < tilt else frame
        previous = (frame, tilt)
    return _first_parallel(poses, start - 1, end + 1, shoulder, wrist, threshold)


def _elbow_extension(poses: np.ndarray, frame: int) -> float | None:
    """Mean of both elbow angles in degrees; 180 is a straight arm."""
    angles = []
    for shoulder, elbow, wrist in ((11, 12, 13), (14, 15, 16)):
        if not all(_finite_row(poses[frame, joint]) for joint in (shoulder, elbow, wrist)):
            return None
        upper = _unit(poses[frame, elbow], poses[frame, shoulder])
        lower = _unit(poses[frame, elbow], poses[frame, wrist])
        if upper is None or lower is None:
            return None
        angles.append(np.degrees(np.arccos(min(1.0, max(-1.0, float(np.dot(upper, lower)))))))
    return float(np.mean(angles))


def _shaft_or_arm(poses: np.ndarray, start: int, end: int, shaft: int, hosel: int, shoulder: int, wrist: int,
                  observed: np.ndarray | None, max_tilt: float = 0.35) -> tuple[int | None, str]:
    """Shaft closest to horizontal, using only frames with a detected club.

    Without `observed` every lifted shaft counts. When no detected club in the
    range is near horizontal (within `max_tilt`), the parallel moment fell in
    frames where the lifted shaft is mostly inferred. The lead arm then stands
    in: 30% of its rotation from the hanging end (P1 for P2, P7 for P6) toward
    the parallel end (P3 or P5). On observed shafts in 005 and frames reviewed
    in 008, the shaft is parallel at 25-30% of that rotation. Lifted wrist
    height relative to the pelvis is not used; at address it can already be
    level with the root.
    """
    if observed is None:
        return _closest_horizontal(poses, start, end, shaft, hosel), "shaft"
    frame = _closest_horizontal(poses, start, end, shaft, hosel, observed)
    if frame is not None:
        direction = _unit(poses[frame, shaft], poses[frame, hosel])
        if direction is not None and abs(float(direction[1])) <= max_tilt:
            return frame, "observed_shaft"

    def elevation(frame: int) -> float | None:
        if not (0 <= frame < len(poses) and _finite_row(poses[frame, shoulder]) and _finite_row(poses[frame, wrist])):
            return None
        direction = _unit(poses[frame, shoulder], poses[frame, wrist])
        return None if direction is None else float(np.degrees(np.arcsin(np.clip(-direction[1], -1, 1))))

    before, after = elevation(start - 1), elevation(end + 1)
    if before is None or after is None:
        return _closest_horizontal(poses, start, end, shaft, hosel), "shaft"
    hanging, parallel = (before, after) if before < after else (after, before)
    target = hanging + ARM_PROGRESS_AT_SHAFT_PARALLEL * (parallel - hanging)
    scored = [(abs(value - target), frame) for frame in range(max(0, start), min(end, len(poses) - 1) + 1)
              if (value := elevation(frame)) is not None]
    return (min(scored)[1] if scored else None), "lead_arm_progress"


def _most_extended(poses: np.ndarray, start: int, end: int, shaft: int, hosel: int, tolerance: float = 5.0,
                   observed: np.ndarray | None = None) -> int | None:
    """Frame in [start, end] with the straightest arms. Frames within
    `tolerance` degrees of the best are tied; the shaft closest to
    horizontal breaks the tie, counting only detected clubs."""
    scored = []
    for frame in range(max(0, start), min(end, len(poses) - 1) + 1):
        extension = _elbow_extension(poses, frame)
        if extension is not None:
            scored.append((frame, extension))
    if not scored:
        return _closest_horizontal(poses, start, end, shaft, hosel)
    best = max(extension for _, extension in scored)
    tied = [frame for frame, extension in scored if extension >= best - tolerance]

    def tilt(frame: int) -> float:
        if observed is not None and not observed[frame]:
            return float("inf")
        if not (_finite_row(poses[frame, shaft]) and _finite_row(poses[frame, hosel])):
            return float("inf")
        direction = _unit(poses[frame, shaft], poses[frame, hosel])
        return float("inf") if direction is None else abs(float(direction[1]))

    return min(tied, key=lambda frame: (tilt(frame), -dict(scored)[frame]))


def _lag_angle(poses: np.ndarray, frame: int, elbow: int, wrist: int, shaft: int, hosel: int) -> float | None:
    if not all(_finite_row(poses[frame, joint]) for joint in (elbow, wrist, shaft, hosel)):
        return None
    forearm = _unit(poses[frame, elbow], poses[frame, wrist])
    club = _unit(poses[frame, shaft], poses[frame, hosel])
    if forearm is None or club is None:
        return None
    cosine = min(1.0, max(-1.0, float(np.dot(forearm, club))))
    return float(np.degrees(np.arccos(cosine)))


def find_checkpoint_frames(
    poses: np.ndarray,
    address: int | None,
    top: int | None,
    impact: int | None,
    lead: str = "left",
    reserve_positions: bool = False,
    club_observed: np.ndarray | None = None,
    sources: dict | None = None,
) -> dict[str, int | None]:
    """P2 through P10 from a 3D skeleton. P1, P4, and P7 are the phases already found.

    Definitions follow the P classification (hackmotion.com/golf-swing-positions):
    shaft parallel at P2/P6/P8, lead arm parallel at P3/P5/P9, hands by the lead
    ear at P10. Camera y points down, so a horizontal bone has a near-zero y component.
    `club_observed` (per frame) limits shaft tests to frames with a detected
    club; `sources`, if given, receives which evidence chose P2 and P6.
    """
    sources = {} if sources is None else sources
    if club_observed is not None:
        club_observed = np.asarray(club_observed, dtype=bool)
    shaft, hosel = 17, 18
    head = 10
    if lead == "left":
        shoulder, wrist = 11, 13
    else:
        shoulder, wrist = 14, 16
    positions = {
        "p1": address,
        "p2": None,
        "p3": None,
        "p4": top,
        "p5": None,
        "p6": None,
        "p7": impact,
        "p8": None,
        "p9": None,
        "p10": None,
    }
    if address is None or top is None or impact is None:
        return positions
    if not (0 <= address <= top <= impact < len(poses)):
        return positions

    # Halfway back is the lead arm rising through parallel, before the top.
    positions["p3"] = _arm_crossing(poses, address + 1 + int(reserve_positions), top - 1, shoulder, wrist, rising=True)
    # Takeaway is the shaft closest to the 3D view's ground, after address and before P3.
    p3 = positions["p3"]
    if p3 is not None:
        positions["p2"], sources["p2"] = _shaft_or_arm(poses, address + 1, p3 - 1, shaft, hosel, shoulder, wrist,
                                                       club_observed)
    # Early downswing mirrors P3: the lead arm comes down through parallel. A shaft
    # test would fire at the top, where a full backswing already has the shaft parallel.
    p5 = _arm_crossing(poses, top + 1, impact - 1 - int(reserve_positions), shoulder, wrist, rising=False)
    positions["p5"] = p5
    # Pre-impact is the shaft closest to the 3D view's ground, after P5 and before the ball.
    if p5 is not None:
        positions["p6"], sources["p6"] = _shaft_or_arm(poses, p5 + 1, impact - 1, shaft, hosel, shoulder, wrist,
                                                       club_observed)

    last = len(poses) - 1
    if reserve_positions:
        finite=np.flatnonzero(np.isfinite(poses).all(axis=(1,2)))
        last=int(finite[-1]) if len(finite) else last
    # P8 needs a frame between impact and P9; P10 needs one after P9.
    # P9 mirrors P3: the lead arm rising through parallel after impact. The first
    # frame inside the tolerance band can still have the hands below the shoulder.
    # Later frames include the finish and the club being lowered, where the arm
    # passes parallel again.
    p9 = _arm_crossing(poses, impact + 1 + int(reserve_positions), last - int(reserve_positions),
                       shoulder, wrist, rising=True)
    if p9 is None:
        rising = [frame for frame in range(impact + 1, last + 1) if _finite_row(poses[frame, wrist])]
        apex = min(rising, key=lambda frame: float(poses[frame, wrist, 1])) if rising else last
        p9 = _closest_horizontal(poses, impact + 1 + int(reserve_positions),
                                 min(apex, last - int(reserve_positions)), shoulder, wrist)
    positions["p9"] = p9
    heights = {frame: -float(poses[frame, wrist, 1]) for frame in range(impact, last + 1)
               if _finite_row(poses[frame, wrist])}
    # Release is both arms fully extended after the ball: the largest elbow angles
    # before P9. The arms are just as straight at impact, so the hands must have
    # started up, a quarter of the way from their impact height to P9. The shaft
    # is often blurred here, so it only breaks near-ties.
    p8 = None
    if p9 is not None:
        start = impact + 1
        if impact in heights and p9 in heights:
            risen = heights[impact] + 0.25 * (heights[p9] - heights[impact])
            start = next((frame for frame in range(impact + 1, p9)
                          if heights.get(frame, -np.inf) >= risen), start)
        p8 = _most_extended(poses, start, p9 - 1, shaft, hosel, observed=club_observed)
    positions["p8"] = p8
    if p9 is None:
        p9 = impact
    # End of swing: hands up by the lead ear while they are held at their highest.
    # Frames within 5% of that peak (relative to the rise since impact) form the
    # finish; after them the golfer is lowering the club. A lead-side dot product
    # is unstable here because the hands wrap around close to the head, so require
    # raised hands.
    finish_start, finish_end = p9 + 1, last
    if impact in heights and any(frame > p9 for frame in heights):
        apex = max((frame for frame in heights if frame > p9), key=heights.get)
        floor = heights[apex] - 0.05 * (heights[apex] - heights[impact])
        finish_start = finish_end = apex
        while finish_start - 1 > p9 and heights.get(finish_start - 1, -np.inf) >= floor:
            finish_start -= 1
        while heights.get(finish_end + 1, -np.inf) >= floor:
            finish_end += 1
    nearest = None
    nearest_distance = float("inf")
    for frame in range(finish_start, finish_end + 1):
        if not (_finite_row(poses[frame, wrist]) and _finite_row(poses[frame, head])):
            continue
        if not _finite_row(poses[frame, shoulder]):
            continue
        if float(poses[frame, wrist, 1]) >= float(poses[frame, shoulder, 1]):
            continue
        distance = float(np.linalg.norm(poses[frame, wrist] - poses[frame, head]))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = frame
    positions["p10"] = nearest
    return positions


def wrist_shaft_degrees(poses: np.ndarray, frame: int | None, lead: str = "left") -> float | None:
    """Angle between the lead forearm and the shaft. This is a picture angle, not a sensor reading."""
    if frame is None or not (0 <= frame < len(poses)):
        return None
    elbow, wrist = (12, 13) if lead == "left" else (15, 16)
    return _lag_angle(poses, frame, elbow, wrist, 17, 18)


def phase_progress(frame: int, phases: SwingPhases) -> float:
    """0 at address, about 0.35 at the top, 0.7 at impact, 1 at follow-through."""
    marks = [
        (phases.address, 0.0),
        (phases.top, 0.35),
        (phases.impact, 0.70),
        (phases.follow_through, 1.0),
    ]
    marks = [(int(frame_index), value) for frame_index, value in marks if frame_index is not None]
    if not marks:
        return 0.0
    if frame <= marks[0][0]:
        return marks[0][1]
    if frame >= marks[-1][0]:
        return marks[-1][1]
    for (left, left_value), (right, right_value) in zip(marks, marks[1:]):
        if left <= frame <= right:
            span = max(right - left, 1)
            return left_value + (right_value - left_value) * ((frame - left) / span)
    return marks[-1][1]


def phase_name(frame: int, phases: SwingPhases) -> str:
    if phases.address is None:
        return ""
    if frame < phases.address:
        return "setup"
    if frame == phases.address:
        return "address"
    if phases.top is not None and frame < phases.top:
        return "backswing"
    if frame == phases.top:
        return "top"
    if phases.impact is not None and frame < phases.impact:
        return "downswing"
    if frame == phases.impact:
        return "impact"
    if phases.follow_through is not None and frame < phases.follow_through:
        return "follow-through"
    if frame == phases.follow_through:
        return "follow-through"
    return "finish"


def save_phase_chart(path, clubhead_y: np.ndarray, phases: SwingPhases, fps: float) -> None:
    """Wrist-style chart: smoothed height and speed, with phase lines."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    prepared = _prepare_track(np.column_stack([np.zeros(len(clubhead_y)), clubhead_y]), fps)
    if prepared is None:
        return
    _smoothed, speed, _finite = prepared
    figure = Figure(figsize=(8, 3.2), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    frames = np.arange(len(clubhead_y))
    axis.plot(frames, smooth_series(clubhead_y, fps), color="#1f77b4", label="clubhead y")
    twin = axis.twinx()
    twin.plot(frames, speed, color="#ff7f0e", alpha=0.8, label="speed")
    colors = {"address": "#2ca02c", "top": "#9467bd", "impact": "#d62728", "follow_through": "#7f7f7f"}
    for name, color in colors.items():
        frame_index = getattr(phases, name)
        if frame_index is not None:
            axis.axvline(frame_index, color=color, linestyle="--", label=name.replace("_", " "))
    axis.set_xlabel("frame")
    axis.set_ylabel("y (px, down)")
    twin.set_ylabel("speed (px/frame)")
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path)


class WristFallback:
    """Fill weak YOLO wrists from MediaPipe Pose. Off unless the task model loads."""

    LEFT = 9
    RIGHT = 10
    MP_LEFT = 15
    MP_RIGHT = 16
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )

    def __init__(self, model_path: str | None = None) -> None:
        self.landmarker = None
        self.replaced = 0
        self._timestamp_ms = 0
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core import base_options as base_options_lib
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
            from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib
        except ImportError:
            print("mediapipe is not installed; wrist fallback is off", flush=True)
            return
        path = model_path or _pose_landmarker_path()
        if not os_path_exists(path):
            print(f"MediaPipe pose model not found: {path}", flush=True)
            return
        options = PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=path),
            running_mode=running_mode_lib.VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._mp = mp

    @property
    def enabled(self) -> bool:
        return self.landmarker is not None

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def fuse(self, frame_bgr, xy: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replace COCO wrists under 0.25 confidence when MediaPipe is confident."""
        xy = np.asarray(xy, dtype=np.float64).copy()
        confidence = np.asarray(confidence, dtype=np.float64).copy()
        if self.landmarker is None:
            return xy, confidence
        if confidence[self.LEFT] >= 0.25 and confidence[self.RIGHT] >= 0.25:
            return xy, confidence
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self.landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.pose_landmarks:
            return xy, confidence
        height, width = frame_bgr.shape[:2]
        landmarks = result.pose_landmarks[0]
        for coco_index, mp_index in ((self.LEFT, self.MP_LEFT), (self.RIGHT, self.MP_RIGHT)):
            if confidence[coco_index] >= 0.25:
                continue
            landmark = landmarks[mp_index]
            visibility = float(getattr(landmark, "visibility", 0.0) or 0.0)
            if visibility < 0.5:
                continue
            xy[coco_index] = (landmark.x * width, landmark.y * height)
            confidence[coco_index] = visibility
            self.replaced += 1
        return xy, confidence


def os_path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).is_file()


def _pose_landmarker_path() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task")
