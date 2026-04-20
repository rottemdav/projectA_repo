import os
import sys
from datetime import datetime

# ==================== Configuration ====================
class Config:
    """
    Main configuration class.
    Contains shared setting and model-specific parameters. 
    """
    
    ACTIVE_MODEL = None

# --- shared parameters ---
    DEVICE = "cuda:0"  # Use "cpu" if no GPU available
    DATE = datetime.now().strftime('%Y%m%d_%H%M%S')

    INPUT_PATH = None
    START_FRAME = None
    END_FRAME = None
    MAX_FRAMES = None

    OUTPUT_DIR = f"/home/projects/sipl-prj10496/project_files/data/output/{DATE}"

# =========== HRNet Model Configuration ============
    class hrnet:
        # Paths - Update these to match your setup
        MMPOSE_ROOT = "/home/projects/sipl-prj10496/project_files/mmpose"
        
        # ============ WholeBody Model Options ============
        # Option 1: HRNet-W48 WholeBody (Best accuracy, 133 keypoints)
        POSE_CONFIG = os.path.join(
            MMPOSE_ROOT, 
            "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288.py"
        )
        POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth"
        
        # Option 2: HRNet-W32 WholeBody (Faster, still good accuracy)
        # POSE_CONFIG = os.path.join(
        #     MMPOSE_ROOT, 
        #     "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py"
        # )
        # POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth"
        
        # Option 3: RTMPose-L WholeBody (Fast and accurate, good for real-time)
        # POSE_CONFIG = os.path.join(
        #     MMPOSE_ROOT, 
        #     "projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py"
        # )
        # POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth"
        
        # Detection model config and checkpoint (RTMDet recommended)
        DET_CONFIG = os.path.join(
            MMPOSE_ROOT, 
            "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
        )
        DET_CHECKPOINT = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
        
        
        # Detection parameters
        DET_CAT_ID = 0  # Category ID for person in COCO
        BBOX_THR = 0.3  # Bounding box score threshold
        NMS_THR = 0.3   # IoU threshold for NMS
        
        # Visualization parameters
        KPT_THR = 0.3   # Keypoint confidence threshold for visualization
        RADIUS = 8      # Keypoint radius (increased for better visibility)
        THICKNESS = 4   # Skeleton line thickness (increased for better visibility)
        DRAW_FACE = False  # Whether to draw face keypoints (set False to skip face)

        JSON_FILENAME_FORMAT = f"{{video_name}}_keypoints_{{DATE}}_{{out_range}}.json"
        VIDEO_FILENAME_FORMAT = f"{{video_name}}_pose_{{DATE}}_{{out_range}}.mp4"
        FILTERED_JSON_FILENAME_FORMAT = f"{{video_name}}_keypoints_filtered_{{DATE}}_{{out_range}}.json"
        FILTERED_VIDEO_FILENAME_FORMAT = f"{{video_name}}_pose_filtered_{{DATE}}_{{out_range}}.mp4"
        RESIDUAL_PLOT_FORMAT = f"{{video_name}}_residuals_{{DATE}}_{{out_range}}.png"

    class openpose:
        # Placeholder for OpenPose configuration if needed in the future
        pass

@classmethod
def set_active_model(cls, model_name):
    if model_name.lower() == "hrnet":
        return cls.hrnet
    elif model_name.lower() == "openpose":
        return cls.openpose
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose 'hrnet' or 'openpose'.")
    
Config.hrnet.OUTPUT_PATH = os.path.join(Config.OUTPUT_DIR, "hrnet")
