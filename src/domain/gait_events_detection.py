import numpy as np

from typing import Dict
# Simple local extrema detection using numpy
from scipy.signal import find_peaks, argrelextrema

from src.models.keypoints_mapping import BODY25_GAIT_KEYPOINTS, WHOLEBODY_GAIT_KEYPOINTS

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

    return {"left_ankle_distance": left_ankle_to_pelvis,
            "right_ankle_distance": right_ankle_to_pelvis,
            "mid_pelvis_loc": mid_pelvis_loc,
            "left_ankle": left_ankle_loc,
            "right_ankle": right_ankle_loc
            }

def gait_event_detection(gait_distance_data, frame_indices) -> Dict[str, np.ndarray]:
    """
    Detects gait events (heel strike and toe off) from 2D gait data.
    heel strike is defined as the local maxima for the same foot, and toe off is defined as the local minima for the same foot, in the anterior-posterior distance of the ankle to the pelvis.

    Parameters:
    - gait_distance_data: Dictionary containing the distance data for each foot - from the pelvis.
    - frame_indices: (N,) array of frame indices corresponding to the keypoints.

    Returns a dictionary with:
    - lhs: (M,) array of frame indices where left heel strikes are detected.
    - lto: (M,) array of frame indices where left toe offs are detected.
    - rhs: (M,) array of frame indices where right heel strikes are detected.
    - rto: (M,) array of frame indices where right toe offs are detected.
    - toe_offs: (M,) array of frame indices where toe offs are detected.
    """
    left_ankle_to_pelvis = gait_distance_data["left_ankle_distance"][:, 0]  # (N,)
    right_ankle_to_pelvis = gait_distance_data["right_ankle_distance"][:, 0]  # (N,)

    # tuning constant to ignore false positives
    left_prominence = 0.1 * (np.max(left_ankle_to_pelvis) - np.min(left_ankle_to_pelvis))
    right_prominence = 0.1 * (np.max(right_ankle_to_pelvis) - np.min(right_ankle_to_pelvis))

    lhs_idx, _ = find_peaks(left_ankle_to_pelvis, prominence=left_prominence, height=0.0)
    lto_idx, _ = find_peaks(-left_ankle_to_pelvis, prominence=left_prominence, height=0.0)

    rhs_idx, _ = find_peaks(right_ankle_to_pelvis, prominence=right_prominence, height=0.0)
    rto_idx, _ = find_peaks(-right_ankle_to_pelvis, prominence=right_prominence, height=0.0)

    return {
        "lhs": lhs_idx,
        "lto": lto_idx,
        "rhs": rhs_idx,
        "rto": rto_idx,
    }
