"""
This file contains the logic for cleaning and post-processing keypoints obtained from the pose estimation stage.
This stage responsible for filterting and interpolating the keypoints to handle missing detection and smooth out 
the otuput. 
"""
import numpy as np
import sys
import os

from scipy.signal import filtfilt, butter
from project_files.projectA_repo.src.models.joint_model_mapping import BODY25_GAIT_KEYPOINTS, WHOLEBODY_GAIT_KEYPOINTS, WHOLEBODY_KEYPOINTS
from project_files.projectA_repo.src.domain.analysis_config import PostProcessConfig


fs = 60.0  # Sampling frequency in Hz, adjust as needed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from libs.hrnet_classes import Config, KeypointPostProcessor, HAS_MMDET      
from src.io.keypoints_io import save_keypoints_dict_to_json

# ======== analysis functions ========
def fill_missing_keypoints(keypoints: np.ndarray, config: PostProcessConfig) -> np.ndarray:
    """Fill missing keypoints (with confidence < threshold) using interpolation."""
    keypoints_filled = keypoints.copy()
    num_frames, num_keypoints, _ = keypoints.shape
    
    for k in range(num_keypoints):
        # Get confidence scores for this keypoint across all frames
        conf_series = keypoints[:, k, 2]
        mask = conf_series < config.CONF_THRESHOLD  # True where confidence is low
        
        if np.sum(~mask) < 2:
            continue  # Not enough points to interpolate
        
        # Interpolate x and y coordinates
        for d in range(2):  # x and y
            coord_series = keypoints_filled[:, k, d]
            coord_series[mask] = np.interp(
                np.where(mask)[0],
                np.where(~mask)[0],
                coord_series[~mask]
            )
    
    return keypoints_filled

def butterworth_lpf (KP_filled, fc, order=4):
    """Apply Butterworth low-pass filter to keypoint time series."""
    nyq = 0.5 * fs
    cutoff_freq = fc / nyq
    b, a = butter(order, cutoff_freq, btype='low')
    return filtfilt(b, a, KP_filled)

def temporal_filter(KP_filled, fc, order=4):
    """Apply temporal Butterworth low-pass filter to keypoints."""
    KP_filled = KP_filled.astype(np.float64)        
    T, J, D = KP_filled.shape
    KP_filtered = np.empty_like(KP_filled)
    
    for j in range(J):
        for d in range(D):
            KP_filtered[:, j, d] = butterworth_lpf(KP_filled[:, j, d], fc, order)

    return KP_filtered.astype(np.float32)

def calc_fc_residual(keypoints_raw, 
                     filter_func, 
                     score, 
                     config: PostProcessConfig):
    """
    keypoint_raw: (T,J,D)
    score: (T,J) - confidence scores
    joints: list of joint indices to consider, if None use all

    return: knee_fcs - cutoff per joint, fcs - cutoff for all joints, rms_curves, recommended_fc 

    """
    fc_grid = config.FC_GRID
    joints = config.JOINTS[0] + config.JOINTS[1] if joints is None else joints

    keypoints_raw = np.asarray(keypoints_raw, dtype=float)
    T,J,D = keypoints_raw.shape
    fcs = np.array(list(fc_grid), dtype=float)
    
    if joints is None:
        joints = list(range(J))
    else:
        joints = np.array(joints, dtype=int)

    rms_curves = np.full((J,D,len(fcs)), np.nan, dtype=float)
    knee_fcs = np.full((J,D), np.nan, dtype=float)

    if score is not None:
        score = np.asarray(score, dtype=float)
        global_valid_mask = (score >= config.CONF_THRESHOLD) & np.isfinite(score)

    else:
        global_valid_mask = np.ones((T,J), dtype=bool)

    for j in joints:
        for d in range(D):
            keypoint_series = keypoints_raw[:, j, d]
            valid_mask = global_valid_mask[:, j] & np.isfinite(keypoint_series)
            

            if valid_mask.sum() < max(10, int(0.2 * T)):
                continue  # Not enough valid data

            # Extract valid data points
            valid_data = keypoint_series[valid_mask]
            
            residuals = []
            for fc in fcs:
                filtered_series = filter_func(valid_data, fc)
                residual = valid_data - filtered_series
                rms = np.sqrt(np.mean(residual**2))
                residuals.append(rms)

            residuals = np.array(residuals, dtype=float)
            rms_curves[j, d, :] = residuals

            # Find knee point in residual curve using maximum curvature
            # The knee is where the curve changes most dramatically
            if len(residuals) > 2:
                # Calculate second derivative to find maximum curvature
                diffs = np.diff(residuals)
                second_diffs = np.diff(diffs)
                
                # The knee is where curvature is maximum (most negative second derivative)
                if len(second_diffs) > 0:
                    knee_idx = np.argmin(second_diffs) + 1  # +1 to align with fcs
                    knee_fcs[j, d] = fcs[knee_idx]
                
    recommended_fc = np.nanmedian(knee_fcs) if np.any(~np.isnan(knee_fcs)) else None
    return recommended_fc, knee_fcs, fcs, rms_curves

# ======== main function ========

def post_process_keypoints(keypoints_array):
    """
    Post-process the keypoints array to handle missing detections and smooth the output.
    The post-processing includes:
    1. Filling missing keypoints using interpolation.
    2. Applying a smoothing filter to reduce noise.
    """

    # ===== check if mmdet is available =====
    if not HAS_MMDET:
        print("Error: mmdet is required for person detection.")
        print("Install with: pip install mmdet")
        return
    
    config = PostProcessConfig()

# ===== Post-Process Video =====
    #kp_filled = post_processor.fill_missing_keypoints(keypoints_array)
    kp_filled = fill_missing_keypoints(keypoints_array, config)

    # fc = 3.0 -> thumb rule
    fc = calc_fc_residual(keypoints_array, filter_func=butterworth_lpf, score=None, conf_threshold=config.CONF_THRESHOLD)[0]
    print(f"Recommended cutoff frequency based on residual analysis: {fc:.2f} Hz")
    kp_filtered = temporal_filter(kp_filled, fc=fc, order=4)

# ===== return results =====
    return kp_filtered




    