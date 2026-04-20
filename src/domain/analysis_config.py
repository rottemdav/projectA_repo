import numpy as np
from project_files.projectA_repo.src.models.joint_model_mapping import WHOLEBODY_KEYPOINTS

class PostProcessConfig:
    FS: float = 60.0  # Sampling frequency in Hz, adjust as needed
    CONF_THRESHOLD: float = 0.2  # Confidence threshold for keypoint validity
    FC_GRID: np.ndarray = np.arange(1.0, 20.5, 0.5)  # Grid of cutoff frequencies to evaluate for residual analysis

    foot_idx = list(WHOLEBODY_KEYPOINTS.keys())[17:23]
    body_idx = list(WHOLEBODY_KEYPOINTS.keys())[:17]
    JOINTS: list = [body_idx, foot_idx]