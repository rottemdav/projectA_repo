"""
This file contains the logic for cleaning and post-processing keypoints obtained from the pose estimation stage.
This stage responsible for filterting and interpolating the keypoints to handle missing detection and smooth out 
the otuput. 
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from libs.hrnet_classes import Config, KeypointPostProcessor, HAS_MMDET      
from src.io.keypoints_io import save_keypoints_dict_to_json

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
    
    post_processor = KeypointPostProcessor(fs=60, conf_threshold=0.3)

# ===== Post-Process Video =====
    video_path = Config.INPUT_PATH
    # FIXME 1 start : move the output file name config to utils or config file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if Config.END_FRAME is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.END_FRAME}"
    elif Config.MAX_FRAMES is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.START_FRAME + Config.MAX_FRAMES - 1}"
    else:
        out_range = f"{Config.START_FRAME}_to_end"
    
    
    filtered_output_path = os.path.join(Config.OUTPUT_DIR,
                                        Config.FILTERED_VIDEO_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range))

    filtered_json_output_path = os.path.join(Config.OUTPUT_DIR,
                                            Config.FILTERED_JSON_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range))
    
    # FIXME 1 end

    kp_filled = post_processor.fill_missing_keypoints(keypoints_array)

    # FIXME 2 start: move the cutoff frequency config to utils or config file
    fc = 3.0  # Cutoff frequency in Hz
    # FIXME 2 end

    kp_filtered = post_processor.temporal_filter(kp_filled, fc=fc, order=4)

    # --- FIXME 3: define in separate function inside the post processor class and move it there, then call it here. ---
    """
    #compute residuals and recommended fc
    fc_grid = np.arange(1.0, 20.5, 0.5)
    foot_idx = WholeBodyPoseProcessor.FOOT_INDICES
    body_idx = WholeBodyPoseProcessor.BODY_INDICES
    main_joints = [ *body_idx, *foot_idx ]
    knee_fcs, fcs, rms_curves, recommended_fc = post_processor.calc_fc_residual(
        keypoints_array, filter_func=post_processor.butterworth_lpf,
        fc_grid=fc_grid, score=keypoints_array[:,:,2], conf_threshold=0.2, joints=main_joints
        )
    """
    # --- FIXME 3 end

# ===== return results =====
    return kp_filtered




    