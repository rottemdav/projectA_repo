"""
Data objects  for the main data variables we'll use in the project.
Includes:
- PoseSequence: A structured representation of the keypoints and related metadata for a sequence of video frames.
- GaitEvents: A structured representation of detected gait events (e.g., heel strikes, toe-offs) with their corresponding frame indices and confidence scores.
- GaitParameters: A structured representation of calculated gait parameters (e.g., stride length, cadence) with their values and confidence scores.
- SpatialGaitParameters: A structured representation of spatial gait parameters (e.g., step width, foot angle) with their values and confidence scores.

"""

from dataclass import dataclass
import numpy as np

@dataclass
class PoseSequence:
    keypoints: np.ndarray  # Shape: (num_frames, num_keypoints, 3)
    scores: np.ndarray     # Shape: (num_frames, num_keypoints)
    frame_indices: np.ndarray  # Shape: (num_frames,)
    fps: float
    joint_names: list[str]

@dataclass
class GaitEvents:
    lhs: np.ndarray  # Left heel strikes frame indices
    lto: np.ndarray  # Left toe offs frame indices
    rhs: np.ndarray  # Right heel strikes frame indices
    rto: np.ndarray  # Right toe offs frame indices
    ds: np.ndarray  # Double support frame indices 

@dataclass
class SpatialGaitParameters:
    step_time: dict[str, np.ndarray]  # {'left': array, 'right': array}
    stance_time: dict[str, np.ndarray]  # {'left': array, 'right': array}
    swing_time: dict[str, np.ndarray]  # {'left': array, 'right': array}
    stride_time: dict[str, np.ndarray]  # {'left': array, 'right': array}
    step_length: dict[str, np.ndarray]  # {'left': array, 'right': array}
    stride_length: dict[str, np.ndarray]  # {'left': array, 'right': array}
    cadence: float
    gait_speed: float

@dataclass
class KinematicGaitParameters:
    join_angles: np.ndarray  # Shape: (num_frames, num_joints, 1) - (angle value only)
    knee_rom: np.ndarray  # Shape: (num_frames, 1) - knee range of motion
    hip_rom: np.ndarray  # Shape: (num_frames, 1) - hip range of motion
    lateral_com:    np.ndarray  # Shape: (num_frames, 1) - lateral center of mass movement
