"""
This file contains the logic for processing videos using the HRNet pose estimation model.
In this stage the execution of handling missing detections and focusing on the main subject in the video is done.
The output of the main function is a formatted keypoints dictionary that can be saved as JSON for later use.

# FIXME: missing of the model flow the focus on the main subject in the video.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from libs.hrnet_classes import Config, WholeBodyPoseProcessor, HAS_MMDET      
from src.io.keypoints_io import save_keypoints_dict_to_json

def hrnet_pose_estimation(input, start_frame, end_frame):
    
    Config.INPUT_PATH = input
    Config.START_FRAME = start_frame
    Config.END_FRAME = end_frame

# ===== check if mmdet is available =====
    if not HAS_MMDET:
        print("Error: mmdet is required for person detection.")
        print("Install with: pip install mmdet")
        return
    
# ===== Initialize Processor =====
    processor = WholeBodyPoseProcessor(
        pose_config=Config.hrnet.POSE_CONFIG,
        pose_checkpoint=Config.hrnet.POSE_CHECKPOINT,
        det_config=Config.hrnet.DET_CONFIG,
        det_checkpoint=Config.hrnet.DET_CHECKPOINT,
        device=Config.DEVICE,
        bbox_thr=Config.hrnet.BBOX_THR,
        vis_kpt_thr=Config.hrnet.VIS_KPT_THR,
    )

# ===== Process Video =====
    video_path = Config.INPUT_PATH
    # FIXME 2 start : move the output file name config to utils or config file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if Config.END_FRAME is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.END_FRAME}"
    elif Config.MAX_FRAMES is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.START_FRAME + Config.MAX_FRAMES - 1}"
    else:
        out_range = f"{Config.START_FRAME}_to_end"
    output_path = os.path.join(Config.OUTPUT_DIR,
                                Config.hrnet.VIDEO_FILENAME_FORMAT.format(video_name=video_name,
                                                                    DATE=Config.DATE,
                                                                    out_range=out_range))

    json_output_path = os.path.join(Config.OUTPUT_DIR,
                                    Config.hrnet.JSON_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range))
    
    # FIXME 2 end

    all_results, all_frames = processor.process_video(
        video_path,
        output_path=output_path,
        start_frame=Config.START_FRAME,  # Start from this frame
        max_frames=Config.MAX_FRAMES,    # None for all frames
        end_frame=Config.END_FRAME,      # None for till end, or set for explicit range
        draw_face=Config.hrnet.DRAW_FACE,      # Set False to hide face keypoints
        show=False,
        json_output_path=json_output_path
    )

# ===== Results Formatting =====
    keypoints_arr = processor.keypoints_to_array(all_frames)

    return keypoints_arr
    



