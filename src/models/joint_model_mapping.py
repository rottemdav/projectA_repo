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
