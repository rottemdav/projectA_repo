"""
This file contains the logic for processing videos using the OpenPose pose estimation model.
In this stage the execution of handling missing detections and focusing on the main subject in the video is done.
The output of the main function is a formatted keypoints dictionary that can be saved as JSON for later use.
"""
import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from libs.openPose_classes import Config, OpenPoseProcessor, KeypointPostProcessor
from utils.skeleton_tracking import filter_and_track_person, is_in_roi, is_skeleton_valid
import utils.skeleton_tracking as st; print("skeleton_tracking:", st.__file__)

def openpose_pose_estimation(input, start_frame, end_frame):
    
    Config.INPUT_PATH = input
    Config.START_FRAME = start_frame
    Config.END_FRAME = end_frame
    
# ===== Initialize Processor =====
    processor = OpenPoseProcessor()

# ===== Process Video =====
    video_path = Config.INPUT_PATH
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if Config.END_FRAME is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.END_FRAME}"
    elif Config.MAX_FRAMES is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.START_FRAME + Config.MAX_FRAMES - 1}"
    else:
        out_range = f"{Config.START_FRAME}_to_end"

    output_path = os.path.join(Config.OUTPUT_DIR,
                                Config.openpose.VIDEO_FILENAME_FORMAT.format(video_name=video_name,
                                                                    DATE=Config.DATE,
                                                                    out_range=out_range))

    json_output_path = os.path.join(Config.OUTPUT_DIR,
                                    Config.openpose.JSON_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range))
    
    all_results, all_frames = processor.process_video(
        video_path=video_path,
        output_path=output_path,
        start_frame=Config.START_FRAME,
        max_frames=Config.MAX_FRAMES,
        end_frame=Config.END_FRAME,
        draw_face=Config.openpose.DRAW_FACE,
        show=False,
        json_output_path=json_output_path
    )

    # ===== Results Formatting & Spatial Tracker Shield =====
    # We apply target tracking across all detected people to extract the main subject.
    # This acts as our "shield" against the Frankenstein effect: isolating one person
    # geometrically BEFORE temporal interpolation connects multiple subjects.
    # Naming convention: name_camNum_videoNum - camera is the SECOND
    # underscore-separated token, not the last one (see src/video_process.py).
    try:
        cam_num = int(video_name.split('_')[1])
    except (IndexError, ValueError):
        cam_num = 1
        
    num_frames = len(all_frames)
    num_keypoints = 25 # BODY_25
    # Keep missing frames as NaN in x/y so downstream interpolation does not
    # mistake the top-left corner for a real body location.
    keypoints_arr = np.full((num_frames, num_keypoints, 3), np.nan, dtype=np.float32)
    last_known_hip_x = None

    for t, frame in enumerate(all_frames):
        persons = frame.get('persons', [])
        if not persons:
            continue
            
        # Build a list of (K,3) arrays per person: [x,y,conf]
        all_kps_for_tracker = []
        for p in persons:
            coords = np.array(p.get('keypoints', []), dtype=np.float32)
            scores = np.array(p.get('scores', []), dtype=np.float32)

            # If coords are (K,2) and scores are (K,), combine to (K,3)
            if coords.ndim == 2 and scores.ndim == 1 and coords.shape[0] == scores.shape[0]:
                combined = np.concatenate([coords, scores.reshape(-1, 1)], axis=1)
            else:
                # Fallback: create combined array with NaNs and fill available values
                n = 0
                if coords.ndim == 2:
                    n = coords.shape[0]
                if scores.ndim == 1:
                    n = max(n, scores.shape[0])
                combined = np.full((n, 3), np.nan, dtype=np.float32)
                if coords.ndim == 2:
                    combined[:coords.shape[0], :2] = coords
                if scores.ndim == 1:
                    combined[:scores.shape[0], 2] = scores

            all_kps_for_tracker.append(combined.tolist())

        for i, p in enumerate(persons):
            # use the combined representation for validity checks
            combined = np.array(all_kps_for_tracker[i], dtype=np.float32)
            hip_conf = float(combined[8, 2]) if combined.size and combined.shape[0] > 8 else None
            try:
                roi_ok = is_in_roi(combined, cam_num, model_type='openpose') if cam_num is not None else True
            except Exception as e:
                roi_ok = f"error:{e}"
            try:
                valid_ok = is_skeleton_valid(combined, model_type='openpose', prev_hip_pos=last_known_hip_x, min_conf=0.3)
            except Exception as e:
                valid_ok = f"error:{e}"
            # Keep the per-person inspection only in code (no print) for now; tests use logs elsewhere

        best_idx = filter_and_track_person(
            all_keypoints=all_kps_for_tracker,
            last_known_hip_x=last_known_hip_x,
            cam_num=cam_num,
            model_type="openpose",
            use_roi_filter=True,
            min_conf=0.3
        )
        # best_idx is used downstream; avoid noisy per-frame prints in normal runs
        
        if best_idx != -1:
            best_person = persons[best_idx]
            kp = np.array(best_person['keypoints'], dtype=np.float32)
            sc = np.array(best_person['scores'], dtype=np.float32)
            
            if kp.shape[-1] == 3:
                keypoints_arr[t, :, :2] = kp[:, :2]
            else:
                keypoints_arr[t, :, :2] = kp
            keypoints_arr[t, :, 2] = sc
            last_known_hip_x = kp[8][0] # MidHip X
        else:
            # Preserve a missing frame explicitly: NaN coordinates with zero confidence.
            keypoints_arr[t, :, :2] = np.nan
            keypoints_arr[t, :, 2] = 0.0

    # ===== Post Processing =====
    # Run post processing (Interpolation and Butterworth)
    # Ensure it returns both the intermediate array and final array
    spatial_filtered_unfiltered_time_arr, keypoints_arr = keypoint_post_process(keypoints_arr, video_name, out_range)
    
    create_filtered_video(processor, all_frames, keypoints_arr, video_path, video_name, out_range)

    # Returning both arrays so the caller can extract gait parameters without smoothing effects
    return keypoints_arr, spatial_filtered_unfiltered_time_arr

def _create_post_processor():
    # Exactly match HRNet settings with fs=60 and conf_threshold=0.3
    return KeypointPostProcessor(fs=60, conf_threshold=0.3)

def keypoint_post_process(keypoints_arr, video_name, out_range):
    post_processor = _create_post_processor()

    print("Applying temporal filtering to keypoints...")

    # 1. Fill missing keypoints (Interpolation logic matching HRNet precisely)
    # This preserves the intermediate data AFTER spatial tracker validation but BEFORE Butterworth
    keypoints_filled = post_processor.fill_missing_keypoints(keypoints_arr)

    # Intermediate Step Preservation:
    # Save a separate copy of the array post-interpolation but pre-Butterworth
    spatial_filtered_unfiltered_time_arr = keypoints_filled.copy()

    # 2. Apply temporal low-pass filter (Butterworth exactly matched: fc=3.0, order=4)
    fc = 3.0  # Cutoff frequency in Hz
    keypoints_filtered = post_processor.temporal_filter(keypoints_filled, fc=fc, order=4)

    # 3. Compute residuals specifically pointing to openpose BODY_25 (not WholeBody)
    fc_grid = np.arange(1.0, 20.5, 0.5)
    foot_idx = OpenPoseProcessor.FOOT_INDICES
    body_idx = OpenPoseProcessor.BODY_INDICES
    main_joints = [ *body_idx, *foot_idx ]
    
    knee_fcs, fcs, rms_curves, recommended_fc = post_processor.calc_fc_residual(
        keypoints_arr, filter_func=post_processor.butterworth_lpf,
        fc_grid=fc_grid, score=keypoints_arr[:,:,2], conf_threshold=0.3, joints=main_joints
    )
    
    if recommended_fc is None:
        print("Recommended cutoff frequency from residual analysis: None")
    else:
        print(f"Recommended cutoff frequency from residual analysis: {recommended_fc:.2f} Hz")

    try:
        post_processor.plot_residual_curves(
            fcs, 
            np.nanmean(rms_curves[main_joints,:,:], axis=0).squeeze(), 
            save_path=os.path.join(Config.OUTPUT_DIR, 
                                   Config.openpose.RESIDUAL_PLOT_FORMAT.format(video_name=video_name,
                                                                      DATE=Config.DATE,
                                                                      out_range=out_range))
        )
    except Exception as e:
        print(f"Failed to plot residual curves: {e}")

    return spatial_filtered_unfiltered_time_arr, keypoints_filtered

def create_filtered_video(processor, all_frames, keypoints_arr, video_path, video_name, out_range):
    processor.write_and_visualize_filtered_video(
        all_frames=all_frames,
        filtered_keypoints=keypoints_arr,
        video_path=video_path,
        output_path=os.path.join(Config.OUTPUT_DIR,
                                    Config.openpose.FILTERED_VIDEO_FILENAME_FORMAT.format(video_name=video_name, DATE=Config.DATE, out_range=out_range)),
        start_frame=Config.START_FRAME,
        end_frame=Config.END_FRAME,
        draw_face=Config.openpose.DRAW_FACE,
        show=False
    )
