"""Optional club geometry refinement. Camera projection is weak-perspective.

Length is marker-to-marker, not nominal equipment length. Inferred length is a
MixSTE prior, not a measurement. Missing landmarks are never made valid.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def observation_weights(valid, confidence=None):
    good = np.asarray(valid[:, 17] & valid[:, 18], dtype=bool)
    weights = good.astype(float)
    if confidence is not None:
        weights *= np.clip(np.asarray(confidence, dtype=float), 0, 1)
    return weights


def prepare_club_inputs(raw, filled, valid, confidence=None):
    """Interpolate center, projected length and unwrapped angle separately.

    Preserve detected pairs exactly. Unlike endpoint interpolation, angular gap
    filling does not take a chord through a rotating shaft. No fixed 2D length
    is imposed: foreshortening remains possible.
    """
    out = filled.copy()
    weights = observation_weights(valid, confidence)
    good = (weights > 0) & np.isfinite(raw[:, 17:19]).all(axis=(1, 2))
    indices = np.flatnonzero(good)
    if len(indices) < 2:
        return out, weights
    vec = raw[good, 18] - raw[good, 17]
    lengths = np.linalg.norm(vec, axis=1)
    usable = lengths > 1e-6
    indices, vec, lengths = indices[usable], vec[usable], lengths[usable]
    if len(indices) < 2:
        return out, weights
    frames = np.arange(len(raw))
    centers = (raw[indices, 17] + raw[indices, 18]) / 2
    theta = np.interp(frames, indices, np.unwrap(np.arctan2(vec[:, 1], vec[:, 0])))
    length = np.interp(frames, indices, lengths)
    center = np.column_stack([np.interp(frames, indices, centers[:, axis]) for axis in range(2)])
    vector = length[:, None] * np.column_stack([np.cos(theta), np.sin(theta)])
    pair = np.stack([center-vector/2, center+vector/2], axis=1)
    # Retain a lone observed endpoint and derive only its missing counterpart.
    only_shaft = valid[:,17] & ~valid[:,18]
    only_head = valid[:,18] & ~valid[:,17]
    pair[only_shaft,0] = raw[only_shaft,17]
    pair[only_shaft,1] = raw[only_shaft,17]+vector[only_shaft]
    pair[only_head,1] = raw[only_head,18]
    pair[only_head,0] = raw[only_head,18]-vector[only_head]
    out[:,17:19] = pair
    if raw.shape[1] == 22:
        for joint in range(19,22):
            if not valid[:,joint].any():
                out[:,joint] = out[:,18]
    return out, weights


def refine_club_3d(poses, points2d, valid, confidence=None, length_m=None):
    """Fit a rigid marker segment to 2D vectors, using MixSTE as depth prior.

    A positive body-derived projection scale converts normalized image units to
    model metres. This is an approximation, not calibrated perspective geometry.
    The shaft anchor/body remain unchanged; head markers translate with hosel.
    """
    if length_m is not None and (not np.isfinite(length_m) or length_m <= 0):
        raise ValueError('Club marker length must be finite and positive')
    output = poses.copy()
    frames = np.flatnonzero(np.isfinite(poses).all(axis=(1,2)))
    if not len(frames):
        return output, {'applied': False, 'reason': 'no finite 3D frames'}
    if np.any(np.diff(frames) != 1):
        raise ValueError('Refinement requires one contiguous finite 3D interval')
    p = poses[frames]
    q = points2d[frames]
    weight = observation_weights(valid, confidence)[frames]
    vector = p[:,18]-p[:,17]
    lengths = np.linalg.norm(vector, axis=1)
    reliable = (weight >= .25) & (lengths > 1e-6)
    if reliable.sum() < 3:
        return output, {'applied': False, 'reason': 'fewer than three reliable endpoint pairs'}
    length = float(length_m if length_m is not None else np.median(lengths[reliable]))
    prior = vector / np.maximum(lengths[:,None], 1e-8)
    # Fit positive scale and translation to body XY, then discard translation
    # because club direction/length uses endpoint differences.
    body3 = p[:,:17,:2]-p[:,:17,:2].mean(axis=1,keepdims=True)
    body2 = q[:,:17]-q[:,:17].mean(axis=1,keepdims=True)
    scale = np.sum(body3*body2,axis=(1,2))/np.maximum(np.sum(body3**2,axis=(1,2)),1e-8)
    positive = scale[np.isfinite(scale) & (scale > 1e-6)]
    if not len(positive):
        return output, {'applied': False, 'reason': 'body projection scale unavailable'}
    median_scale = np.median(positive)
    scale = np.clip(scale, .5*median_scale, 2*median_scale)
    target = (q[:,18]-q[:,17]) / (scale[:,None]*length)
    initial = np.column_stack([np.arctan2(prior[:,1],prior[:,0]),
                               np.arctan2(prior[:,2],np.linalg.norm(prior[:,:2],axis=1))])
    n = len(p)

    def direction(x):
        theta, elevation = x.reshape(n,2).T
        return np.column_stack([np.cos(elevation)*np.cos(theta),
                                np.cos(elevation)*np.sin(theta), np.sin(elevation)])

    def residual(x):
        unit = direction(x)
        # Missing 2D pairs exert zero data force; temporal/depth priors fill them.
        image = np.sqrt(weight)[:,None]*(unit[:,:2]-target)
        depth_prior = .3*(unit-prior)
        acceleration = .4*np.diff(unit,n=2,axis=0)
        return np.concatenate([image.ravel(),depth_prior.ravel(),acceleration.ravel()])

    sparsity = lil_matrix((5*n+3*max(n-2,0),2*n),dtype=int)
    for i in range(n):
        sparsity[2*i:2*i+2,2*i:2*i+2] = 1
        sparsity[2*n+3*i:2*n+3*i+3,2*i:2*i+2] = 1
    for i in range(n-2):
        sparsity[5*n+3*i:5*n+3*i+3,2*i:2*i+6] = 1
    fit = least_squares(residual, initial.ravel(), jac_sparsity=sparsity.tocsr(),
                        loss='soft_l1', f_scale=.1, max_nfev=100)
    if not fit.success or not np.isfinite(fit.x).all():
        return output, {'applied': False, 'reason': 'optimizer did not converge', 'message': fit.message}
    new_head = p[:,17] + length*direction(fit.x)
    delta = new_head-p[:,18]
    output[frames,18] = new_head
    if p.shape[1] > 19:
        output[frames,19:] += delta[:,None]
    return output, dict(applied=True, method='rigid_segment_weak_perspective',
        marker_length_m=length, length_source='user_measured' if length_m is not None else 'median_reliable_mixste_prior',
        reliable_pairs=int(reliable.sum()), optimizer_evaluations=int(fit.nfev),
        optimizer_cost=float(fit.cost), projection_scale_median=float(median_scale),
        caveat='Uncalibrated projection; inferred length and depth are not physical accuracy measurements')
