"""

Planar kinetics utilities for 2D pose estimation-based gait analysis.
This file contains functions to compute projected joint and segment angles
from 2D keypoint coordinates in the image plane.

Assumptions:
- Angles are computed in the image plane, in degrees.
- We did not construct body-attached coordinate systems or anatomical axes; angles are purely 
  geometric projections in 2D.
- Spatial quatitites may be reported in normalized pixel units.
                    
This library contains:
- calculate_angles: Computes joint angles (hip, knee, ankle) based on 2D keypoints.
- calculate_gait_parameters: Computes temporal gait parameters (step time, stance time, swing time, double support time) based on detected gait events.
- calculate_spatial_parameters: Computes spatial parameters such as step length and gait speed based on 2D keypoint positions and gait events.

Authors: Leeor Gabbay, Rottem David
"""

from typing import Dict
import numpy as np

"""
COCO-WholeBody outputs 133 keypoints:
    - Body: 17 keypoints (indices 0-16)
    - Feet: 6 keypoints (indices 17-22) 
"""

WHOLEBODY_KEYPOINTS = {
    # --- Body (0–16) ---
    0:  "nose", 1:  "left_eye", 2:  "right_eye", 3:  "left_ear",
    4:  "right_ear", 5:  "left_shoulder", 6:  "right_shoulder",
    7:  "left_elbow", 8:  "right_elbow", 9:  "left_wrist",
    10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle",
    16: "right_ankle",

    # --- Feet (17–22) ---
    17: "left_big_toe", 18: "left_small_toe", 19: "left_heel",
    20: "right_big_toe", 21: "right_small_toe", 22: "right_heel"
}

BODY25_KEYPOINTS = {
    0:  "nose",
    1:  "neck",
    2:  "right_shoulder",
    3:  "right_elbow",
    4:  "right_wrist",
    5:  "left_shoulder",
    6:  "left_elbow",
    7:  "left_wrist",
    8:  "mid_hip",
    9:  "right_hip",
    10: "right_knee",
    11: "right_ankle",
    12: "left_hip",
    13: "left_knee",
    14: "left_ankle",
    15: "right_eye",
    16: "left_eye",
    17: "right_ear",
    18: "left_ear",
    19: "left_big_toe",
    20: "left_small_toe",
    21: "left_heel",
    22: "right_big_toe",
    23: "right_small_toe",
    24: "right_heel",
}

WHOLEBODY_GAIT_KEYPOINTS = {
    "left": {
        "hip": 11,
        "knee": 13,
        "ankle": 15,
        "heel": 19,
        "toe": 17
    },
    "right": {
        "hip": 12,
        "knee": 14,
        "ankle": 16,
        "heel": 22,
        "toe": 20
    }
}

BODY25_GAIT_KEYPOINTS = {
    "left": {
        "hip": 12,
        "knee": 13,
        "ankle": 14,
        "heel": 21,
        "toe": 19,        # big toe
    },
    "right": {
        "hip": 9,
        "knee": 10,
        "ankle": 11,
        "heel": 24,
        "toe": 22,        # big toe
    },
    "center": {
        "pelvis": 8,      # mid_hip
        "neck": 1
    }
}

# ------------------ Kinetics Calculations ------------------ #

# ------------------ Joint Angles ------------------ #
def calculate_angles(model, keypoints):
    def angle_between(p1_idx, p2_idx):
        y_diff = keypoints[:, p1_idx, 1] - keypoints[:, p2_idx, 1]
        x_diff = keypoints[:, p1_idx, 0] - keypoints[:, p2_idx, 0]
        return np.degrees(np.arctan2(y_diff, x_diff))

    angles = {}

    if model == 'BODY25':
        R_Hip, R_Knee, R_Ankle, R_BigToe = 9, 10, 11, 22
        L_Hip, L_Knee, L_Ankle, L_BigToe = 12, 13, 14, 19
    elif model == 'wholeBody':
        R_Hip, R_Knee, R_Ankle, R_BigToe = 12, 14, 16, 20
        L_Hip, L_Knee, L_Ankle, L_BigToe = 11, 13, 15, 17
    else:
        raise ValueError(f"Unsupported model: {model}")

    angles['LHip'] = angle_between(L_Knee, L_Hip) + 90
    angles['RHip'] = angle_between(R_Knee, R_Hip) + 90

    angles['LKnee'] = angle_between(L_Hip, L_Knee) - angle_between(L_Ankle, L_Knee) - 180
    angles['RKnee'] = angle_between(R_Hip, R_Knee) - angle_between(R_Ankle, R_Knee) - 180

    term1_L = angle_between(L_Knee, L_Ankle)
    term2_L = angle_between(L_BigToe, L_Ankle)
    angles['LAnkle'] = - (term1_L - term2_L - 90)

    term1_R = angle_between(R_Knee, R_Ankle)
    term2_R = angle_between(R_BigToe, R_Ankle)
    angles['RAnkle'] = - (term1_R - term2_R - 90)

    return angles

# FIXME: old version - delete
def calculate_gait_parameters(keypoints, time_vector, events, scaling_factor=1.0):
    """
    - LHS - Left Heel Strike
    - RHS - Right Heel Strike
    - LTO - Left Toe Off
    - RTO - Right Toe Off
    - DS - Double Support

    Args:
        keypoints (np.array): Shape (n_frames, 25, 2) or (n_frames, 25, 3).
        time_vector (np.array): Time in seconds for each frame.
        events (dict): Dictionary with numpy arrays of frame indices:
                       {'lhs', 'rhs', 'lto', 'rto'}
                       (Left/Right Heel Strike, Left/Right Toe Off).
        scaling_factor (float): Conversion factor from pixels to meters (default 1.0).

    Returns:
        dict: A dictionary containing all calculated gait parameters.
    """

    # --- Helper Function ---
    def get_time_diff(start_frames, find_in_frames, time_vec, mode='next'):
        """
        Finds the time difference between a start event and the next occurring target event.
        """
        durations = []
        for start_f in start_frames:
            # Find events that happen AFTER the start frame
            future_events = find_in_frames[find_in_frames > start_f]
            
            if len(future_events) > 0:
                # Get the first one (the closest next event)
                next_f = future_events[0]
                durations.append(time_vec[next_f] - time_vec[start_f])
            else:
                durations.append(np.nan) # Handle missing next event
        return np.array(durations)

    # Initialize Output Dictionary
    gait_params = {
        'stepTime': {},
        'stanceTime': {},
        'swingTime': {},
        'dsTime': {}, # Double Support
        'stepLength': {},
        'gaitSpeed': 0
    }

    # --- 1. Step Times (Time between opposite heel strikes) ---
    # Right Step: Time from LHS to next RHS
    gait_params['stepTime']['right'] = get_time_diff(events['lhs'], events['rhs'], time_vector)
    # Left Step: Time from RHS to next LHS
    gait_params['stepTime']['left'] = get_time_diff(events['rhs'], events['lhs'], time_vector)

    # --- 2. Stance Times (Time foot is on ground: HS -> TO) ---
    # Left Stance: LHS -> LTO
    gait_params['stanceTime']['left'] = get_time_diff(events['lhs'], events['lto'], time_vector)
    # Right Stance: RHS -> RTO
    gait_params['stanceTime']['right'] = get_time_diff(events['rhs'], events['rto'], time_vector)

    # --- 3. Swing Times (Time foot is in air: TO -> Next HS) ---
    # Left Swing: LTO -> Next LHS
    # Note: We iterate over LTO frames here to find the next LHS
    gait_params['swingTime']['left'] = get_time_diff(events['lto'], events['lhs'], time_vector)
    # Right Swing: RTO -> Next RHS
    gait_params['swingTime']['right'] = get_time_diff(events['rto'], events['rhs'], time_vector)

    # --- 4. Double Support Times (Both feet on ground) ---
    # DS Left-to-Right: From RHS landing until Left Toe leaves.
    # Logic: We calculate Time(Next LTO) - Time(Next RHS) relative to a cycle start,
    # But simpler logic is: For every RHS, find the closest *following* LTO.
    # Note: This assumes valid walking where overlap exists.
    
    # Calculation: Time(LTO) - Time(RHS) where LTO > RHS
    ds_lr = []
    for rhs_f in events['rhs']:
        future_lto = events['lto'][events['lto'] > rhs_f]
        if len(future_lto) > 0:
            next_lto = future_lto[0]
            # Check if this LTO belongs to the immediate double support phase (short duration)
            dt = time_vector[next_lto] - time_vector[rhs_f]
            ds_lr.append(dt)
    gait_params['dsTime']['left_to_right'] = np.array(ds_lr)

    # DS Right-to-Left: From LHS landing until Right Toe leaves.
    ds_rl = []
    for lhs_f in events['lhs']:
        future_rto = events['rto'][events['rto'] > lhs_f]
        if len(future_rto) > 0:
            next_rto = future_rto[0]
            dt = time_vector[next_rto] - time_vector[lhs_f]
            ds_rl.append(dt)
    gait_params['dsTime']['right_to_left'] = np.array(ds_rl)

    # --- 5. Step Lengths (Spatial) ---
    # OpenPose Indices (0-based): R_Ankle=11, L_Ankle=14
    R_ANKLE, L_ANKLE = 11, 14

    # Right Step Length: Distance between ankles at RHS frame
    # We filter valid frames only (integers)
    valid_rhs = events['rhs'].astype(int)
    if len(valid_rhs) > 0:
        # Calculate X distance at the moment of impact
        diff_x = keypoints[valid_rhs, R_ANKLE, 0] - keypoints[valid_rhs, L_ANKLE, 0]
        gait_params['stepLength']['right'] = np.abs(diff_x * scaling_factor)
    else:
        gait_params['stepLength']['right'] = np.array([])

    # Left Step Length: Distance between ankles at LHS frame
    valid_lhs = events['lhs'].astype(int)
    if len(valid_lhs) > 0:
        diff_x = keypoints[valid_lhs, L_ANKLE, 0] - keypoints[valid_lhs, R_ANKLE, 0]
        gait_params['stepLength']['left'] = np.abs(diff_x * scaling_factor)
    else:
        gait_params['stepLength']['left'] = np.array([])

    # --- 6. Gait Speed ---
    # Speed = Mean(Step Lengths) / Mean(Step Times)
    all_lengths = np.concatenate([gait_params['stepLength']['left'], gait_params['stepLength']['right']])
    all_times = np.concatenate([gait_params['stepTime']['left'], gait_params['stepTime']['right']])
    
    # Avoid division by zero
    if len(all_times) > 0 and np.nanmean(all_times) > 0:
        gait_params['gaitSpeed'] = np.nanmean(all_lengths) / np.nanmean(all_times)
    else:
        gait_params['gaitSpeed'] = 0.0

    return gait_params
# FIXME: end
def calculate_spatial_parameters(model, keypoints, events, scaling_factor=1.0):
    """
    calculates spatial parameters such as step length and gait speed based on 2D keypoint positions and gait events.
    returns a dictionary with calculated spatial parameters.
    """
    if model == 'BODY25':
        IDX_R_FOOT = 11  # Right Ankle
        IDX_L_FOOT = 14  # Left Ankle
        IDX_X = 0        # X coordinate
    elif model == 'COCO-WholeBody':
        IDX_R_FOOT = 16  # Right Ankle
        IDX_L_FOOT = 15  # Left Ankle
        IDX_X = 0        # X coordinate
   
    spatial_parameters = {}

    scaling_factor =scaling_factor
    rhs_frames = events['rhs']
    lhs_frames = events['lhs']

    if 'stepLength' not in spatial_parameters:
        spatial_parameters['stepLength'] = {}

    r_pos_at_rhs = keypoints[rhs_frames, IDX_R_FOOT, IDX_X]
    l_pos_at_rhs = keypoints[rhs_frames, IDX_L_FOOT, IDX_X]
    spatial_parameters['stepLength']['right'] = np.abs(scaling_factor * (r_pos_at_rhs - l_pos_at_rhs))

    l_pos_at_lhs = keypoints[lhs_frames, IDX_L_FOOT, IDX_X]
    r_pos_at_lhs = keypoints[lhs_frames, IDX_R_FOOT, IDX_X]
    spatial_parameters['stepLength']['left'] = np.abs(scaling_factor * (l_pos_at_lhs - r_pos_at_lhs))
    
    return spatial_parameters

def calculate_gait_parameters_2(keypoints, time_vector, events, scaling_factor=1.0):

    #build chronological heel-strike timeline
    hs_events = []

    for e in events['lhs']:
        hs_events.append((int(e), "left"))
    for e in events['rhs']:
        hs_events.append((int(e), "right"))

    hs_events = sorted(hs_events, key=lambda x: x[0]) # sort by frame index

    #helper to get next event index in a sorted array

    def next_event_index(frames, f):
        future_events = frames[frames > f]
        return int(future_events[0]) if len(future_events) else None
    
    lhs = np.array(sorted(events['lhs']), dtype = int)
    rhs = np.array(sorted(events['rhs']), dtype = int)
    lto = np.array(sorted(events['lto']), dtype = int)
    rto = np.array(sorted(events['rto']), dtype = int)

    step_time = []
    stance_time = []
    swing_time = []
    hs_frames = []
    hs_sides = []
    prev_opposite_hs_frames = []
    to_frames = []
    next_same_side_hs_frames = []

    for i in range(1, len(hs_events)):
        prev_f, prev_side = hs_events[i-1]
        curr_f, curr_side = hs_events[i]
        #next_f, next_side = hs_events[i+1]

        # treat a step only if the current side and the next side alternates sides (i.e. left -> right or right -> left)
        if prev_side == curr_side:
            continue

        # calculate step time: current HS -> next opposite HS
        step_time.append(time_vector[curr_f] - time_vector[prev_f])
        hs_frames.append(curr_f)
        hs_sides.append(curr_side)

        if curr_side == "left":
            # calculate the frame index of toe-off and the next heel strike after that toe-off of the same side (lhs -> lto -> lhs)
            to_f = next_event_index(lto, curr_f)
            next_same_hs = next_event_index(lhs, curr_f)
            prev_opposite_hs_frames.append(prev_f)  # the previous opposite-side HS frame
            to_frames.append(to_f if to_f is not None else -1)
            next_same_side_hs_frames.append(next_same_hs if next_same_hs is not None else -1)
        else: 
            # rhs -> rto -> rhs
            to_f = next_event_index(rto, curr_f)
            next_same_hs = next_event_index(rhs, curr_f)
            prev_opposite_hs_frames.append(prev_f)
            to_frames.append(to_f if to_f is not None else -1)
            next_same_side_hs_frames.append(next_same_hs if next_same_hs is not None else -1)

        # calculate stance time: current HS -> next TO (same side) - how much time the foot is on the ground
        if to_f is not None:
            stance_time.append(time_vector[to_f] - time_vector[curr_f])
        else:
            stance_time.append(np.nan)

        # calculate swing time: current TO -> next same-side HS - how much time the foot is in the air
        if to_f is not None and next_same_hs is not None:
            swing_time.append(time_vector[next_same_hs] - time_vector[to_f])
        else:
            swing_time.append(np.nan)

    gait_params = {
        "hs_frames": np.array(hs_frames, dtype=int),
        "hs_sides": np.array(hs_sides, dtype=object),
        "prev_opposite_hs_frames": np.array(prev_opposite_hs_frames, dtype=int),
        "to_frames": np.array(to_frames, dtype=int),
        "next_same_side_hs_frames": np.array(next_same_side_hs_frames, dtype=int),
        "stepTime": np.array(step_time, dtype=float),
        "stanceTime": np.array(stance_time, dtype=float),
        "swingTime": np.array(swing_time, dtype=float),
    }

    return gait_params
# ------------------ Temporal Parameters ------------------ #

def add_step_direction(source_df, distance_data, stationary_threshold_px=1.0):
    steps_df = source_df.copy()

    dx_values = []
    directions = []

    for _, row in steps_df.iterrows():
        side = row["side"]
        hs_frame = int(row["hs_frame"])

        if side == "left":
            step_signal = distance_data["left_ankle_distance"][:, 0]
            stance_signal = distance_data["right_ankle_distance"][:, 0]
        elif side == "right":
            step_signal = distance_data["right_ankle_distance"][:, 0]
            stance_signal = distance_data["left_ankle_distance"][:, 0]
        else:
            dx_values.append(np.nan)
            directions.append("unknown")
            continue

        if hs_frame < 0 or hs_frame >= len(step_signal) or hs_frame >= len(stance_signal):
            dx_values.append(np.nan)
            directions.append("unknown")
            continue

        dx = step_signal[hs_frame] - stance_signal[hs_frame]

        if not np.isfinite(dx):
            dx_values.append(np.nan)
            directions.append("unknown")
            continue

        dx_values.append(float(dx))

        if dx > stationary_threshold_px:
            directions.append("forward")
        elif dx < -stationary_threshold_px:
            directions.append("backward")
        else:
            directions.append("stationary")

    steps_df["relative_foot_dx_px"] = dx_values
    steps_df["step_direction"] = directions

    return steps_df
