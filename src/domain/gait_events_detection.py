"""
This file contains the logic for detecting gait events (heel strike and toe off) from the cleaned keypoints data.
The main function will take the cleaned keypoints data and compute the relevant gait events based on the anterior-posterior distance of the ankle to the pelvis. Heel strikes are detected as local maxima and
"""

import numpy as np

from typing import Dict
# Simple local extrema detection using numpy
from scipy.signal import find_peaks, argrelextrema

from src.models.joint_model_mapping import BODY25_GAIT_KEYPOINTS, WHOLEBODY_GAIT_KEYPOINTS

def ankle_to_pelvis_distance(model, keypoints_dict):
    """ 
    calculates the relative position of the anterior-posterior location of the ankle to the central pelvis point. 
    """
    keypoints_arr = keypoints_dict['keypoints']  # (N, K, 3)

    if model == "wholebody":
        kp_arr = WHOLEBODY_GAIT_KEYPOINTS
        mid_pelvis_loc = (
            keypoints_arr[:, kp_arr["left"]["hip"], :2]
            + keypoints_arr[:, kp_arr["right"]["hip"], :2]
        ) / 2

    elif model == "body25":
        kp_arr = BODY25_GAIT_KEYPOINTS
        mid_pelvis_loc = keypoints_arr[:, kp_arr["center"]["pelvis"], :2]
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    left_ankle_loc = keypoints_arr[:, kp_arr["left"]["ankle"], :2]
    right_ankle_loc = keypoints_arr[:, kp_arr["right"]["ankle"], :2]

    left_ankle_to_pelvis = left_ankle_loc - mid_pelvis_loc  # (N, 2)
    right_ankle_to_pelvis = right_ankle_loc - mid_pelvis_loc  # (N, 2)

    left_dx = np.diff(left_ankle_to_pelvis[:, 0], prepend=np.nan)
    right_dx = np.diff(right_ankle_to_pelvis[:, 0], prepend=np.nan)

    return {
        "left_ankle_distance": left_ankle_to_pelvis,
        "right_ankle_distance": right_ankle_to_pelvis,
        "left_ankle_dx": left_dx,
        "right_ankle_dx": right_dx,
        "mid_pelvis_loc": mid_pelvis_loc,
        "left_ankle": left_ankle_loc,
        "right_ankle": right_ankle_loc,
    }

def infer_forward_axis_sign(model, keypoints_dict, confidence_threshold=0.2):
    """
    Returns +1 when forward-facing direction is toward increasing image x,
    and -1 when it is toward decreasing image x.

    For treadmill videos the pelvis may drift only slightly, so body orientation
    is a more reliable forward-axis cue than start/end pelvis position.
    """
    keypoints_arr = keypoints_dict["keypoints"]

    if model == "wholebody":
        kp_arr = WHOLEBODY_GAIT_KEYPOINTS
        nose_idx = 0
        pelvis_x = (
            keypoints_arr[:, kp_arr["left"]["hip"], 0]
            + keypoints_arr[:, kp_arr["right"]["hip"], 0]
        ) / 2
        if keypoints_arr.shape[2] > 2:
            confidence = np.minimum(
                keypoints_arr[:, nose_idx, 2],
                np.minimum(
                    keypoints_arr[:, kp_arr["left"]["hip"], 2],
                    keypoints_arr[:, kp_arr["right"]["hip"], 2],
                ),
            )
        else:
            confidence = np.ones(len(keypoints_arr), dtype=float)

    elif model == "body25":
        kp_arr = BODY25_GAIT_KEYPOINTS
        nose_idx = 0
        pelvis_idx = kp_arr["center"]["pelvis"]
        pelvis_x = keypoints_arr[:, pelvis_idx, 0]
        if keypoints_arr.shape[2] > 2:
            confidence = np.minimum(
                keypoints_arr[:, nose_idx, 2],
                keypoints_arr[:, pelvis_idx, 2],
            )
        else:
            confidence = np.ones(len(keypoints_arr), dtype=float)
    else:
        raise ValueError(f"Unsupported model: {model}")

    nose_x = keypoints_arr[:, nose_idx, 0]
    nose_to_pelvis_dx = nose_x - pelvis_x
    valid = np.isfinite(nose_to_pelvis_dx) & (confidence >= confidence_threshold)

    if not np.any(valid):
        valid = np.isfinite(nose_to_pelvis_dx)

    if not np.any(valid):
        return 1.0

    median_dx = np.nanmedian(nose_to_pelvis_dx[valid])
    if not np.isfinite(median_dx) or median_dx == 0:
        return 1.0

    return 1.0 if median_dx > 0 else -1.0

def gait_event_detection(
    gait_distance_data,
    frame_indices,
    heel_strike_extrema="max",
    prominence_mult=0.1,
    shared_prominence=False,
    min_distance_frames=None,
    min_width_frames=None,
    return_diagnostics=False,
) -> Dict[str, np.ndarray]:

    def _fill_nan_1d(x):
        x = np.asarray(x, dtype=float).copy()
        valid = np.isfinite(x)
        if not np.any(valid) or np.all(valid):
            return x
        idx = np.arange(len(x))
        x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
        return x

    def _robust_range(signal):
        finite = signal[np.isfinite(signal)]
        if finite.size == 0:
            return 0.0
        lo, hi = np.nanpercentile(finite, [5, 95])
        signal_range = hi - lo
        if not np.isfinite(signal_range) or signal_range <= 0:
            signal_range = np.nanmax(finite) - np.nanmin(finite)
        return float(max(signal_range, 0.0))

    def _detect_hs_to(signal, hs_prominence):
        if signal.size < 3 or hs_prominence <= 0:
            empty = np.array([], dtype=int)
            return empty, empty

        peak_kwargs = {
            "prominence": hs_prominence,
            "distance": min_distance_frames,
            "width": min_width_frames,
        }

        if heel_strike_extrema == "max":
            hs_idx, _ = find_peaks(signal, **peak_kwargs)
            to_idx, _ = find_peaks(-signal, **peak_kwargs)
        elif heel_strike_extrema == "min":
            hs_idx, _ = find_peaks(-signal, **peak_kwargs)
            to_idx, _ = find_peaks(signal, **peak_kwargs)
        else:
            raise ValueError("heel_strike_extrema must be 'max' or 'min'")

        return hs_idx.astype(int), to_idx.astype(int)

    left_signal = _fill_nan_1d(gait_distance_data["left_ankle_distance"][:, 0])
    right_signal = _fill_nan_1d(gait_distance_data["right_ankle_distance"][:, 0])

    left_range = _robust_range(left_signal)
    right_range = _robust_range(right_signal)

    if shared_prominence:
        base_range = max(left_range, right_range)
        left_prominence = prominence_mult * base_range
        right_prominence = prominence_mult * base_range
    else:
        base_range = None
        left_prominence = prominence_mult * left_range
        right_prominence = prominence_mult * right_range

    lhs_idx, lto_idx = _detect_hs_to(left_signal, left_prominence)
    rhs_idx, rto_idx = _detect_hs_to(right_signal, right_prominence)

    events = {
        "lhs": lhs_idx,
        "lto": lto_idx,
        "rhs": rhs_idx,
        "rto": rto_idx,
    }

    if not return_diagnostics:
        return events

    diagnostics = {
        "ranges": {
            "left": left_range,
            "right": right_range,
            "shared": base_range,
        },
        "prominence": {
            "left": left_prominence,
            "right": right_prominence,
        },
        "min_distance_frames": min_distance_frames,
        "min_width_frames": min_width_frames,
        "heel_strike_extrema": heel_strike_extrema,
        "frame_span": (
            frame_indices[0],
            frame_indices[-1],
        ) if len(frame_indices) else None,
    }

    return events, diagnostics