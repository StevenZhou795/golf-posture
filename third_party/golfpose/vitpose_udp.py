# Copyright (c) OpenMMLab. All rights reserved. Apache-2.0.
# Inference decoder extracted from installed MMPose 1.3.2; see VITPOSE_SOURCE.txt.
from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np

def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, f'Invalid shape {heatmaps.shape}'
    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)
    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.0] = -1
    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)
    return (locs, vals)

def gaussian_blur(heatmaps: np.ndarray, kernel: int=11) -> np.ndarray:
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape
    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps

def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int) -> np.ndarray:
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 0.001, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)
    heatmaps_pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()
    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]
        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian, derivative).squeeze()
    return keypoints
