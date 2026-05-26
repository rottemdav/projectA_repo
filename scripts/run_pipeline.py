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
from src.models.hrnet_video_process import hrnet_pose_estimation
from src.libs.hrnet_classes import KeypointPostProcessor, WholeBodyPoseProcessor, HAS_MMDET
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
    return parser.parse_args()

def main():
    """Main function demonstrating WholeBody pose estimation usage."""
    args = parse_args()

    Config.INPUT_PATH = args.input
    Config.START_FRAME = args.start
    Config.END_FRAME = args.end

    # Check if mmdet is available
    if not HAS_MMDET:
        print("Error: mmdet is required for person detection.")
        print("Install with: pip install mmdet")
        return
    
    # Initialize processor
    processor = WholeBodyPoseProcessor(
        pose_config=Config.hrnet.POSE_CONFIG,
        pose_checkpoint=Config.hrnet.POSE_CHECKPOINT,
        det_config=Config.hrnet.DET_CONFIG,
        det_checkpoint=Config.hrnet.DET_CHECKPOINT,
        device=Config.DEVICE,
        bbox_thr=Config.hrnet.BBOX_THR,
        vis_kpt_thr=Config.hrnet.KPT_THR,
    )

    #Initialize post-processor
    post_processor = KeypointPostProcessor(fs=60, conf_threshold=0.3)
    
    # ============ Process Video ============
    video_path = Config.INPUT_PATH
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # output file name config: use end_frame in output filename if set, else max_frames, else till end
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
    
    filtered_output_path = os.path.join(Config.OUTPUT_DIR,
                                        Config.hrnet.FILTERED_VIDEO_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range))
    
    json_output_path = os.path.join(Config.OUTPUT_DIR,
                                    Config.hrnet.JSON_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range))
    
    filtered_json_output_path = os.path.join(Config.OUTPUT_DIR,
                                            Config.hrnet.FILTERED_JSON_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range))

    #video processing results
    if not args.skip_processing:
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

        # === DEBUG: check how many frames actually have detections ===
        num_total = len(all_frames)
        num_with_person = sum(1 for f in all_frames if len(f["persons"]) > 0)

        print("\n=== Frame sanity check ===")
        print(f"Total processed frames      : {num_total}")
        print(f"Frames with detected person : {num_with_person}")
        print("============================\n")

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
        keypoints_arr = loaded["keypoints"]  # shape (N, 133, 3)
        frame_indices = loaded["frame_indices"]
        has_person = loaded["has_person"]

    #post-processing: temporal filtering
    print("Applying temporal filtering to keypoints...")

    # Extract keypoints into numpy array for processing   
    if args.skip_processing:
        keypoints_array = keypoints_arr
    else:
        keypoints_array = post_processor.keypoints_to_array(all_frames)

    # Fill missing keypoints
    keypoints_filled = post_processor.fill_missing_keypoints(keypoints_array)

    # Apply temporal low-pass filter
    fc = 3.0  # Cutoff frequency in Hz
    keypoints_filtered = post_processor.temporal_filter(keypoints_filled, fc=fc, order=4)

    #compute residuals and recommended fc
    fc_grid = np.arange(1.0, 20.5, 0.5)
    foot_idx = WholeBodyPoseProcessor.FOOT_INDICES
    body_idx = WholeBodyPoseProcessor.BODY_INDICES
    main_joints = [ *body_idx, *foot_idx ]
    knee_fcs, fcs, rms_curves, recommended_fc = post_processor.calc_fc_residual(
        keypoints_array, filter_func=post_processor.butterworth_lpf,
        fc_grid=fc_grid, score=keypoints_array[:,:,2], conf_threshold=0.2, joints=main_joints
        )
    
    print(f"Recommended cutoff frequency from residual analysis: {recommended_fc:.2f} Hz")

    #plot residual curves for body keypoints
    post_processor.plot_residual_curves(
        fcs, 
        np.nanmean(rms_curves[main_joints,:,:], axis=0).squeeze(), 
        save_path=os.path.join(Config.OUTPUT_DIR, 
                               Config.hrnet.RESIDUAL_PLOT_FORMAT.format(video_name=video_name,
                                                                  DATE=Config.DATE,
                                                                  out_range=out_range))
    )
    
    if not args.skip_processing:
        processor.write_and_visualize_filtered_video(
            all_frames=all_frames,
            filtered_keypoints=keypoints_filtered,
            video_path=video_path,
            output_path=filtered_output_path,
            start_frame=Config.START_FRAME,
            end_frame=Config.END_FRAME,
            draw_face=Config.hrnet.DRAW_FACE,
            show=False
        )


        # Extract and print keypoints for the first frame with detections
        if all_results:
            frame_idx, pose_results = all_results[0]
            keypoints = processor.extract_keypoints(pose_results)
            print(f"\nFrame {frame_idx}: Detected {len(keypoints)} person(s)")
            for i, kp_data in enumerate(keypoints):
                print(f"  Person {i}: {kp_data['keypoints'].shape[0]} keypoints")
                
    # save the filtered keypoints to a new json file
    save_keypoints_dict_to_json(
        {
            "frame_indices": frame_indices,
            "keypoints": keypoints_filtered,
            "has_person": has_person,
        },
        filtered_json_output_path,
        model_type="wholebody"
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