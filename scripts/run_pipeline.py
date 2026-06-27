"""
26/05/26 - first version of the pipeline execution script, which runs the whole pipeline 
from video processing to gait event detection and parameter calculation. 
This is the main entry point for executing the entire workflow.
"""
import json
import sys
import os
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config  
#from src.models.hrnet_video_process import hrnet_pose_estimation, keypoint_post_process, create_filtered_video
#from src.models.openPose_video_process import openpose_pose_estimation
#from src.libs.hrnet_classes import KeypointPostProcessor, WholeBodyPoseProcessor, HAS_MMDET
from src.io.keypoints_io import load_keypoints_dict_from_json, save_keypoints_dict_to_json

def parse_args():
    parser = argparse.ArgumentParser(description="WholeBody Pose Estimation with MMPose")

    parser.add_argument("--input", 
                        type=str, 
                        required=True,
                        help="Path to input video or image")

    parser.add_argument("--start", "--start_frame",
                        type=int,
                        default=0,
                        help="Frame number to start processing from (default: 0)")

    parser.add_argument("--end", "--end_frame",
                        type=int,
                        default=None,
                        help="Frame number to end processing (inclusive, default: None for till end)")
    parser.add_argument("--skip_processing", 
                        action="store_true",
                        help="If set, skip the video processing step and directly run post-processing and gait event detection on existing keypoints json output. This is useful for iterating on post-processing and gait event detection without re-running the whole video processing step.")
    parser.add_argument("--existing_json",
                        type=str,
                        default=None,
                        help="Path to existing keypoints json output to use when --skip_processing is set. If not provided, will look for json output in the default location based on input video name and frame range.")
    parser.add_argument("--model",
                        type=str,
                        choices=["hrnet", "openpose"],
                        default="hrnet",
                        help="Which pose estimation model to use (default: hrnet)")
    return parser.parse_args()


def main():
    """Main function demonstrating WholeBody pose estimation usage."""
    args = parse_args()
    print(f"\n args.model: {args.model}")

    Config.INPUT_PATH = args.input
    Config.START_FRAME = args.start
    Config.END_FRAME = args.end
    Config.set_active_model(args.model)
    Config.VIDEO_NAME = os.path.splitext(os.path.basename(Config.INPUT_PATH))[0]
    Config.FRAME_RANGE = Config.build_frame_range()
    Config.set_active_model(args.model)    
    Config.OUTPUT_DIR = Config.get_output_dir()
    output_dir = Config.get_output_dir()

    output_path = os.path.join(
    output_dir,
    Config.ACTIVE_MODEL.FILTERED_JSON_FILENAME_FORMAT.format(
        video_name=Config.VIDEO_NAME,
        DATE=Config.DATE,
        out_range=Config.FRAME_RANGE,
        ),
    )

    if args.model == "hrnet":
        from src.models.hrnet_video_process import (
            hrnet_pose_estimation, keypoint_post_process, create_filtered_video
    )
    elif args.model == "openpose":
        from src.models.openPose_video_process import openpose_pose_estimation, keypoint_post_process 

# ============ Process Video ============

# ================================= hrnet pose estimation and post-processing =================================
    if args.model == "hrnet":
        print ("Running HRNet pose estimation...")
        print("\n" + "="*60)
        if args.skip_processing:
            print("Skipping video processing step. Loading keypoints from existing json output...")
            # Load keypoints from existing json output
            if args.existing_json:
                json_path = args.existing_json
            else:
                # quit the program if no existing json path is provided
                print("Error: --existing_json must be provided when --skip_processing is set.")
                return

            loaded = load_keypoints_dict_from_json(json_path, model_type="wholebody")
            hrnet_keypoints_arr = loaded["keypoints"]  # shape (N, 133, 3)
            frame_indices = loaded["frame_indices"]
            has_person = loaded["has_person"]

        else:
            hrnet_keypoints_arr, all_frames, frame_indices, has_person,processor = hrnet_pose_estimation(Config.INPUT_PATH, Config.START_FRAME, Config.END_FRAME)
            if hrnet_keypoints_arr is None:
                return    
            
        # post-processing: temporal filtering
        keypoints_filtered = keypoint_post_process(hrnet_keypoints_arr, Config.VIDEO_NAME, Config.FRAME_RANGE)
        if not args.skip_processing:
            create_filtered_video(processor, all_frames, keypoints_filtered, Config.INPUT_PATH, Config.VIDEO_NAME, Config.FRAME_RANGE)
                    
        # save the filtered keypoints to a new json file
            
        save_keypoints_dict_to_json(
            {
                "frame_indices": frame_indices,
                "keypoints": keypoints_filtered,
                "has_person": has_person,
            },
            output_path,
            model_type="wholebody"
        )

# ================================= openpose pose estimation and post-processing =================================
    elif args.model == "openpose":
        print ("Running OpenPose pose estimation...")
        print("\n" + "="*60)
        print("\n")
        print("the process will save the keypoints extracted by OpenPose to a json file to the path: ", output_path)
        if args.skip_processing:
            print("Skipping video processing step. Loading keypoints from existing json output...")
            if args.existing_json:
                json_path = args.existing_json
            else:
                print("Error: --existing_json must be provided when --skip_processing is set.")
                return

            loaded = load_keypoints_dict_from_json(json_path, model_type="body25")
            op_keypoints_arr = loaded["keypoints"]
            frame_indices = loaded["frame_indices"]
            has_person = loaded["has_person"]

            # Mirror HRNet skip behavior by re-running temporal post-processing.
            _, op_keypoints_arr = keypoint_post_process(op_keypoints_arr, Config.VIDEO_NAME, Config.FRAME_RANGE)
        else:
            kp_filtered, kp_unfiltered, frame_indices, has_person = openpose_pose_estimation(
                Config.INPUT_PATH,
                Config.START_FRAME,
                Config.END_FRAME,
            )
            save_keypoints_dict_to_json(
                {
                    "frame_indices": frame_indices,
                    "keypoints": op_keypoints_arr,
                    "has_person": has_person,
                },
                output_path,
                model_type="openpose"
            )
            op_keypoints_arr = kp_filtered
            start = Config.START_FRAME or 0
            frame_indices = list(range(start, start + len(op_keypoints_arr)))
            has_person = [True] * len(op_keypoints_arr)

        if op_keypoints_arr is None:
            return


        save_keypoints_dict_to_json(
            {
                "frame_indices": frame_indices,
                "keypoints": op_keypoints_arr,
                "has_person": has_person,
            },
            output_path,
            model_type="openpose"
        )
    
    print("\n" + "="*60)
    print("WholeBody Pose Processor initialized successfully!")
    print("="*60)
    print("\nCOCO-WholeBody 133 Keypoint Format:")
    print("  Body (0-16):    17 keypoints - standard COCO body") 
    print("  Feet (17-22):    6 keypoints - left/right big toe, small toe, heel")
    print("  Face (23-90):   68 keypoints - facial landmarks")
    print("  Left hand (91-111):  21 keypoints")
    print("  Right hand (112-132): 21 keypoints")
    print("\nBody keypoints (0-16):")
    print("  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear")
    print("  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow")
    print("  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip")
    print("  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle")
    print("\nUsage examples:")
    print("  # Process image:")
    print("  pose_results, vis_img = processor.process_image('image.jpg', 'output.jpg')")
    print("  # Process video:") 
    print("  results = processor.process_video('video.mp4', 'output.mp4')")
    print("  # Extract all keypoints:")
    print("  keypoints = processor.extract_keypoints(pose_results)")
    print("  # Get specific body parts:")
    print("  body_kp, scores = processor.get_body_keypoints(keypoints[0])")
    print("  face_kp, scores = processor.get_face_keypoints(keypoints[0])")
    print("  hands = processor.get_hand_keypoints(keypoints[0], 'both')")

if __name__ == "__main__":
    main()