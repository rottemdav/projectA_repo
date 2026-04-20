"""
This file contains the logic for processing videos using the OpenPose pose estimation model.
In this stage the execution of handling missing detections and focusing on the main subject in the video is done.
The output of the main function is a formatted keypoints dictionary that can be saved as JSON for later use.

# fixme: missing of the model flow the focus on the main subject in the video.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from libs.openPose_classes import Config, OpenPoseProcessor, KeypointPostProcessor

def openpose_pose_estimation(input, start_frame, end_frame):
    
    Config.INPUT_PATH = input
    Config.START_FRAME = start_frame
    Config.END_FRAME = end_frame
    
# ===== Initialize Processor =====
    processor = OpenPoseProcessor()

# ===== Process Video =====
    video_path = Config.INPUT_PATH
    # fixme 2 start : move the output file name config to utils or config file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if Config.END_FRAME is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.END_FRAME}"
    elif Config.MAX_FRAMES is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.START_FRAME + Config.MAX_FRAMES - 1}"
    else:
        out_range = f"{Config.START_FRAME}_to_end"

    output_path = os.path.join(Config.OUTPUT_DIR,
                                Config.VIDEO_FILENAME_FORMAT.format(video_name=video_name,
                                                                    DATE=Config.DATE,
                                                                    out_range=out_range))

    json_output_path = os.path.join(Config.OUTPUT_DIR,
                                    Config.JSON_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range))
    
    # fixme 2 end

    all_results, all_frames = processor.process_video(
        video_path=video_path,
        output_path=output_path,
        start_frame=Config.START_FRAME,  # Start from this frame
        max_frames=Config.MAX_FRAMES,    # None for all frames
        end_frame=Config.END_FRAME,      # None for till end, or set for explicit range
        draw_face=Config.DRAW_FACE,      # Set False to hide face keypoints
        show=False,
        json_output_path=json_output_path
    )

# ===== Results Formatting =====
    keypoints_arr = processor.keypoints_to_array(all_frames)

    return keypoints_arr
