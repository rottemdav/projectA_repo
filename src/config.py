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
    VIDEO_NAME = None
    START_FRAME = None
    END_FRAME = None
    MAX_FRAMES = None
    FRAME_RANGE = None

    OUTPUT_DIR = f"/home/projects/sipl-prj10496/project_files/data/outputs/{ACTIVE_MODEL}/{DATE}"

    @classmethod
    def build_frame_range(cls):
        if cls.END_FRAME is not None:
            return f"{cls.START_FRAME}_to_{cls.END_FRAME}"
        if cls.MAX_FRAMES is not None:
            return f"{cls.START_FRAME}_to_{cls.START_FRAME + cls.MAX_FRAMES - 1}"
        return f"{cls.START_FRAME}_to_end"

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
        # ============ OpenPose Paths ============
        OPENPOSE_ROOT = "/home/projects/sipl-prj10496/project_files/openpose"
        OPENPOSE_PYTHON_PATH = os.path.join(OPENPOSE_ROOT, "build/python/openpose")
        MODEL_FOLDER = os.path.join(OPENPOSE_ROOT, "models/")
        
        # ============ OpenPose Engine Options ============
        # Can add other pyopenpose params here
        NUMBER_PEOPLE_MAX = 3
        RENDER_POSE = 1
        
        # ============ Tracking & Filtering Parameters ============
        USE_CAMERA_POSITION_FILTER = True
        
        # Camera spatial limits
        CAM1_Y_MAX = 930
        CAM2_X_MIN = 1860
        CAM2_X_MAX = 2380
        CAM3_Y_MAX = 1200
        
        # Tracking logic
        MIN_HIP_CONF = 0.4                  # Minimum MidHip confidence to start an anchor
        CONF_THRESHOLD = 0.3                # Threshold to keep general keypoints
        HIP_DIST_THRESHOLD = 350            # Max allowed distance jump for the hip anchor
        RESET_LAST_KNOWN_AFTER_MISSES = 5   # Number of missed frames before resetting tracker
        
        # ============ Visualization Parameters ============
        DRAW_PREFILTER_PEOPLE = False
        DRAW_POSTFILTER_SELECTED = True
        DRAW_FACE = True
        DRAW_CAM2_X_LIMITS = True
        DRAW_SKELETON = False               # Decide whether to draw the skeleton or not
        
        # ============ Video Processing Parameters ============
        START_FRAME = None    # Frame to start processing from (0 = beginning)
        MAX_FRAMES = None     # Maximum frames to process (None = all frames)
        END_FRAME = None      # Frame to end processing (inclusive, None = till end)

        # ============ I/O Configuration ============
        DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # filename format and directory configuration 

        JSON_FILENAME_FORMAT = f"{{video_name}}_keypoints_{{DATE}}_{{out_range}}.json"
        VIDEO_FILENAME_FORMAT = f"{{video_name}}_pose_{{DATE}}_{{out_range}}.mp4"
        FILTERED_JSON_FILENAME_FORMAT = f"{{video_name}}_keypoints_filtered_{{DATE}}_{{out_range}}.json"
        FILTERED_VIDEO_FILENAME_FORMAT = f"{{video_name}}_pose_filtered_{{DATE}}_{{out_range}}.mp4"
        RESIDUAL_PLOT_FORMAT = f"{{video_name}}_residuals_{{DATE}}_{{out_range}}.png"

    @classmethod
    def set_active_model(cls, model_name):
        if model_name.lower() == "hrnet":
            cls.ACTIVE_MODEL = cls.hrnet
        elif model_name.lower() == "openpose":
            cls.ACTIVE_MODEL = cls.openpose
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Choose 'hrnet' or 'openpose'.")
        
    @classmethod
    def get_output_dir(cls):
        return f"/home/projects/sipl-prj10496/project_files/data/outputs/{cls.ACTIVE_MODEL.__name__}/{cls.DATE}"
        

    