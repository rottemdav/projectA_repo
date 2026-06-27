"""
This file contains the logic for processing videos using the HRNet pose estimation model.
In this stage the execution of handling missing detections and focusing on the main subject in the video is done.
The output of the main function is a formatted keypoints dictionary that can be saved as JSON for later use.

# FIXME: missing of the model flow the focus on the main subject in the video.
"""
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from libs.hrnet_classes import Config, WholeBodyPoseProcessor, KeypointPostProcessor, HAS_MMDET      
from src.domain.person_tracking import (
    select_tracked_person_frames,
    tracked_frames_to_keypoints_array,
)
from src.io.keypoints_io import save_keypoints_dict_to_json

def _create_processor():
    # ===== Initialize Processor and Post-Processor =====
    return WholeBodyPoseProcessor(
        pose_config=Config.hrnet.POSE_CONFIG,
        pose_checkpoint=Config.hrnet.POSE_CHECKPOINT,
        det_config=Config.hrnet.DET_CONFIG,
        det_checkpoint=Config.hrnet.DET_CHECKPOINT,
        device=Config.DEVICE,
        bbox_thr=Config.hrnet.BBOX_THR,
        vis_kpt_thr=Config.hrnet.KPT_THR,    
    )

def _create_post_processor():
    return KeypointPostProcessor(fs=60, conf_threshold=0.3)

def hrnet_pose_estimation(input, start_frame, end_frame):
    
    Config.INPUT_PATH = input
    Config.START_FRAME = start_frame
    Config.END_FRAME = end_frame

# ===== check if mmdet is available =====
    if not HAS_MMDET:
        print("Error: mmdet is required for person detection.")
        print("Install with: pip install mmdet")
        return

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
    processor = _create_processor()

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

# ===== Results Formatting =====
    tracked_frames = select_tracked_person_frames(
        all_frames,
        model_type="hrnet",
        video_name=video_name,
        min_conf=0.3,
    )
    keypoints_arr = tracked_frames_to_keypoints_array(
        tracked_frames,
        model_type="hrnet",
        num_keypoints=133,
    )

    frame_indices = [f["frame_index"] for f in tracked_frames]  # or whatever the frame index key is
    has_person = [len(f["persons"]) > 0 for f in tracked_frames]

    # Extract and print keypoints for the first frame with detections
    if all_results:
        frame_idx, pose_results = all_results[0]
        keypoints = processor.extract_keypoints(pose_results)
        print(f"\nFrame {frame_idx}: Detected {len(keypoints)} person(s)")
        for i, kp_data in enumerate(keypoints):
            print(f"  Person {i}: {kp_data['keypoints'].shape[0]} keypoints")

    return keypoints_arr, tracked_frames, frame_indices, has_person, processor    

def keypoint_post_process(keypoints_arr,video_name, out_range):
    post_processor = _create_post_processor()

    #post-processing: temporal filtering
    print("Applying temporal filtering to keypoints...")

    # Fill missing keypoints
    keypoints_filled = post_processor.fill_missing_keypoints(keypoints_arr)

    # Apply temporal low-pass filter
    fc = 3.0  # Cutoff frequency in Hz
    keypoints_filtered = post_processor.temporal_filter(keypoints_filled, fc=fc, order=4)

    #compute residuals and recommended fc
    fc_grid = np.arange(1.0, 20.5, 0.5)
    foot_idx = WholeBodyPoseProcessor.FOOT_INDICES
    body_idx = WholeBodyPoseProcessor.BODY_INDICES
    main_joints = [ *body_idx, *foot_idx ]
    knee_fcs, fcs, rms_curves, recommended_fc = post_processor.calc_fc_residual(
        keypoints_arr, filter_func=post_processor.butterworth_lpf,
        fc_grid=fc_grid, score=keypoints_arr[:,:,2], conf_threshold=0.2, joints=main_joints
        )
    
    if recommended_fc is None:
        print("Recommended cutoff frequency from residual analysis: None")
    else:
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

    return keypoints_filtered

def create_filtered_video(processor,all_frames, keypoints_arr, video_path, video_name, out_range):
    processor.write_and_visualize_filtered_video(
        all_frames=all_frames,
        filtered_keypoints=keypoints_arr,
        video_path=video_path,
        output_path=os.path.join(Config.OUTPUT_DIR,
                                    Config.hrnet.FILTERED_VIDEO_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range)),
        start_frame=Config.START_FRAME,
        end_frame=Config.END_FRAME,
        draw_face=Config.hrnet.DRAW_FACE,
        show=False
    )



