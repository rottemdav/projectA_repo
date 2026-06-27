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

def gait_event_detection(gait_distance_data, frame_indices, heel_strike_extrema="max") -> Dict[str, np.ndarray]:

    def _fill_nan_1d(x):
        x = np.asarray(x, dtype=float).copy()
        valid = np.isfinite(x)

        if not np.any(valid):
            return x

        if np.all(valid):
            return x

        idx = np.arange(len(x))
        x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
        return x

    def _detect_hs_to(signal):
        prominence = 0.1 * (np.nanmax(signal) - np.nanmin(signal))

        if heel_strike_extrema == "max":
            hs_idx, _ = find_peaks(signal, prominence=prominence)
            to_idx, _ = find_peaks(-signal, prominence=prominence)
        elif heel_strike_extrema == "min":
            hs_idx, _ = find_peaks(-signal, prominence=prominence)
            to_idx, _ = find_peaks(signal, prominence=prominence)
        else:
            raise ValueError("heel_strike_extrema must be 'max' or 'min'")

        return hs_idx.astype(int), to_idx.astype(int)

    left_signal = _fill_nan_1d(gait_distance_data["left_ankle_distance"][:, 0])
    right_signal = _fill_nan_1d(gait_distance_data["right_ankle_distance"][:, 0])

    lhs_idx, lto_idx = _detect_hs_to(left_signal)
    rhs_idx, rto_idx = _detect_hs_to(right_signal)

    return {
        "lhs": lhs_idx,
        "lto": lto_idx,
        "rhs": rhs_idx,
        "rto": rto_idx,
    }