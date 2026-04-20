import sys
import cv2
import os
import json
import numpy as np
import time

import math

def get_dist(p1, p2):
    """
    Calculates Euclidean distance between two keypoints.
    Returns None if one of the points is missing (confidence = 0).
    """
    if p1[2] == 0 or p2[2] == 0:
        return None
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_skeleton_valid(keypoints, prev_hip_pos=None, min_conf=0.3):
    """
    Validates the skeleton based on anatomical proportions and movement.
    Returns True if valid, False if it's a 'merged' or 'broken' skeleton.
    """
    # Required confidence checks to reject incomplete/broken lower-body skeletons.
    # OpenPose BODY_25 indices used in this project:
    # MidHip=8, RHip=9, RKnee=10, RAnkle=11, LHip=12, LKnee=13, LAnkle=14
    def has_conf(idx):
        return keypoints[idx][2] >= min_conf

    if not has_conf(8):
        return False

    required_leg_points = [9, 10, 11, 12, 13, 14]
    confident_leg_points = sum(1 for idx in required_leg_points if has_conf(idx))

    # Require at least 4/6 lower-limb keypoints, and at least one full leg chain.
    right_leg_complete = has_conf(9) and has_conf(10) and has_conf(11)
    left_leg_complete = has_conf(12) and has_conf(13) and has_conf(14)
    if confident_leg_points < 4 or not (right_leg_complete or left_leg_complete):
        return False

    # Reference Scale: Torso length (Neck=1 to MidHip=8)
    torso_len = get_dist(keypoints[1], keypoints[8])
    
    # If torso is missing, we can't judge proportions reliably
    if torso_len is None or torso_len < 10:
        return True 

    # --- CHECK 1: ARM PROPORTIONS ---
    # Max arm length is usually ~1.3x torso length
    r_arm = get_dist(keypoints[2], keypoints[4]) # Right Shoulder to Wrist
    l_arm = get_dist(keypoints[5], keypoints[7]) # Left Shoulder to Wrist
    
    if r_arm and r_arm > torso_len * 1.5: return False
    if l_arm and l_arm > torso_len * 1.5: return False

    # --- CHECK 2: LEG PROPORTIONS ---
    # Max leg length is usually ~1.8x torso length
    r_leg = get_dist(keypoints[9], keypoints[11]) # Right Hip to Ankle
    l_leg = get_dist(keypoints[12], keypoints[14]) # Left Hip to Ankle
    
    if r_leg and r_leg > torso_len * 2.0: return False
    if l_leg and l_leg > torso_len * 2.0: return False

    # --- CHECK 3: SYMMETRY ---
    # If one arm is 2x longer than the other, it's likely a merged skeleton
    if r_arm and l_arm:
        if r_arm > l_arm * 2.0 or l_arm > r_arm * 2.0:
            return False

    # --- CHECK 4: VELOCITY (IF PREVIOUS POSITION EXISTS) ---
    # At 60 FPS, a human typically cannot move horizontally more than ~50% of their torso length in 1/60th of a second.
    if prev_hip_pos is not None:
        current_hip_x = keypoints[8][0]
        velocity = abs(current_hip_x - prev_hip_pos)
        max_allowed_jump = torso_len * 0.5  
        if velocity > max_allowed_jump:
            return False

    return True


def parse_time_to_seconds(time_str):
    """Parse time formats: SS, MM:SS, HH:MM:SS (supports decimal seconds)."""
    time_str = time_str.strip()
    if not time_str:
        raise ValueError("Time string is empty")

    parts = time_str.split(':')
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"Unsupported time format: {time_str}")

try:
    sys.path.insert(0, '/home/projects/sipl-prj10496/project_files/openpose/build/python/openpose')
    import pyopenpose as op
except ImportError as e:
    print("Error: Could not find OpenPose library.")
    raise e

params = dict()
params["model_folder"] = "/home/projects/sipl-prj10496/project_files/openpose/models/"
#params["tracking"] = 1
params["number_people_max"] = 3
params["render_pose"] = 1

try:
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
except Exception as e:
    print(f"Error starting OpenPose: {e}")
    sys.exit(1)
    
video_path = "project_files/data/source_videos/NL100/NL100_1.MP4"
video_filename = os.path.splitext(os.path.basename(video_path))[0]
cap = cv2.VideoCapture(video_path)

parts = video_filename.split('_')

# Validation: ensure the filename has the expected parts
if len(parts) < 2 or not parts[1].isdigit():
    print(f"Error: Could not extract camera number from filename '{video_filename}'.")
    print("Expected format: name_camNum_videoNum")
    sys.exit(1) # Stop execution

cam_num = int(parts[1])
print(f"Processing video for Camera: {cam_num}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration_sec = total_frames / fps if fps > 0 else 0

print(f"Video FPS: {fps:.3f}, total frames: {total_frames}, duration: {video_duration_sec:.2f}s")

default_start_time = "30"
default_end_time = "60"

start_time_input = input(f"Enter start time [SS / MM:SS / HH:MM:SS] (default {default_start_time}): ").strip()
end_time_input = input(f"Enter end time   [SS / MM:SS / HH:MM:SS] (default {default_end_time}): ").strip()

if not start_time_input:
    start_time_input = default_start_time
if not end_time_input:
    end_time_input = default_end_time

try:
    start_time_sec = parse_time_to_seconds(start_time_input)
    end_time_sec = parse_time_to_seconds(end_time_input)
except ValueError as e:
    print(f"Invalid time input: {e}")
    sys.exit(1)

if start_time_sec < 0 or end_time_sec < 0:
    print("Error: Start/end times must be non-negative.")
    sys.exit(1)

if end_time_sec <= start_time_sec:
    print("Error: End time must be greater than start time.")
    sys.exit(1)

if video_duration_sec > 0:
    start_time_sec = min(start_time_sec, video_duration_sec)
    end_time_sec = min(end_time_sec, video_duration_sec)

start_frame = int(round(start_time_sec * fps))
end_frame = int(round(end_time_sec * fps))

if end_frame <= start_frame:
    print("Error: Calculated frame range is empty. Check times and FPS.")
    sys.exit(1)

print(f"Selected range: {start_time_sec:.2f}s -> {end_time_sec:.2f}s")
print(f"Calculated frames: start={start_frame}, end={end_frame}, total={end_frame - start_frame}")

# Generate unique output paths (avoid overwriting existing files)
output_base = f"/home/projects/sipl-prj10496/project_files/data/output_videos_hip/{video_filename}"
output_video_path = f"{output_base}.mp4"
counter = 1
while os.path.exists(output_video_path):
    output_video_path = f"{output_base}_{counter}.mp4"
    counter += 1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

json_base = f"/home/projects/sipl-prj10496/project_files/data/output_jsons/{video_filename}_output_json"
json_dir = json_base
counter = 1
while os.path.exists(json_dir):
    json_dir = f"{json_base}_{counter}"
    counter += 1
os.makedirs(json_dir)

print(f"Starting processing... Video output: {output_video_path}")

# Process video frames

debug_mode = True
debug_print_every = 30
# Keep for future tuning: set True to draw MidHip candidates and print their coordinates.
draw_hip_debug_overlay = False
# Set False to disable pre-filter drawing.
draw_prefilter_people = False
# Draw selected filtered person on top (green/red), so you can compare before/after.
draw_postfilter_selected = True
draw_cam2_x_limits, cam2_x_min, cam2_x_max = True, 1860, 2380

use_camera_position_filter = True
processed_length_dur = end_time_sec - start_time_sec
print(f"Processing frames {start_frame} to {end_frame} ({processed_length_dur:.2f} seconds at {fps:.1f} fps)...")

# Jump to start position
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

start_time = time.time()
frame_number = start_frame
frames_processed = 0

POSE_PAIRS = [
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
    [1, 0], [0, 15], [15, 17], [0, 16], [16, 18],
    [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]
]

last_known_hip_x = None
# Reset tracking anchor after consecutive frames with no filtered selection.
missed_filtered_frames = 0
reset_last_known_after_misses = 5
# (Ensure start_frame, cap, datum, op, opWrapper, etc. are already initialized)

while cap.isOpened():
    # Check if the current frame exceeds the end frame limit
    if frame_number >= end_frame:
        break

    # Read the next frame from the video capturel
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame using OpenPose wrapper
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    final_image = frame.copy()

    if cam_num == 2 and draw_cam2_x_limits:
        cv2.line(final_image, (cam2_x_min, 0), (cam2_x_min, height - 1), (255, 255, 0), 2)
        cv2.line(final_image, (cam2_x_max, 0), (cam2_x_max, height - 1), (255, 255, 0), 2)
        cv2.putText(final_image, f"CAM2 X LIMITS: [{cam2_x_min}, {cam2_x_max}]",
                    (max(10, cam2_x_min - 220), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    people_list_for_json = []
    debug_info = {
        "raw_people": 0,
        "valid_position_people": 0,
        "conf_pass_people": 0,
        "selected": False,
        "hips": []
    }

    # Check if any people were detected in the current frame
    if datum.poseKeypoints is not None and datum.poseKeypoints.size > 0:
        all_keypoints = datum.poseKeypoints.tolist()
        debug_info["raw_people"] = len(all_keypoints)
        
        best_person_idx = -1
        min_distance = float('inf')

        # Iterate through all detected people to find the target
        for i, keypoints in enumerate(all_keypoints):
            mid_hip_x = keypoints[8][0]
            mid_hip_y = keypoints[8][1]
            mid_hip_conf = keypoints[8][2]

            # Apply spatial filtering based on the specific camera number
            is_valid_position = False
            if cam_num == 1 and mid_hip_y < 930:
                is_valid_position = True
            elif cam_num == 2 and cam2_x_min <= mid_hip_x <= cam2_x_max:
                is_valid_position = True
            elif cam_num == 3 and mid_hip_y < 1200:
                is_valid_position = True

            # Optional debug overlay for MidHip tuning
            if draw_hip_debug_overlay and mid_hip_conf > 0.05:
                hip_point = (int(mid_hip_x), int(mid_hip_y))
                hip_color = (0, 255, 255) if is_valid_position else (0, 165, 255)
                cv2.circle(final_image, hip_point, 6, hip_color, -1)
                cv2.putText(final_image, f"H{i} ({int(mid_hip_x)}, {int(mid_hip_y)}) c={mid_hip_conf:.2f}",
                            (hip_point[0] + 10, max(18, hip_point[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, hip_color, 1, cv2.LINE_AA)

                if len(debug_info["hips"]) < 5:
                    debug_info["hips"].append(f"p{i}=({int(mid_hip_x)},{int(mid_hip_y)}) c={mid_hip_conf:.2f} valid={is_valid_position}")

            if is_valid_position:
                debug_info["valid_position_people"] += 1

            position_ok = is_valid_position or (not use_camera_position_filter)
            if mid_hip_conf > 0.4 and position_ok:
                debug_info["conf_pass_people"] += 1

                if last_known_hip_x is None:
                    best_person_idx = i
                    break 
                else:
                    distance = abs(mid_hip_x - last_known_hip_x)
                    if distance < min_distance and distance < 350:
                        min_distance = distance
                        best_person_idx = i

        # --- VALIDATION AND PROCESSING ---
        if best_person_idx != -1:
            candidate_kp = all_keypoints[best_person_idx]
            
            # Sanity Check: Ensure the skeleton is anatomically logical
            if is_skeleton_valid(candidate_kp, last_known_hip_x):
                # Update tracking and confirm selection
                best_keypoints = candidate_kp
                last_known_hip_x = best_keypoints[8][0]
                debug_info["selected"] = True

                # Clean keypoints (Threshold 0.3)
                cleaned_keypoints = []
                for point in best_keypoints:
                    x, y, conf = point
                    cleaned_keypoints.append([x, y, conf] if conf > 0.3 else [0.0, 0.0, 0.0])

                # Prepare JSON data
                people_list_for_json.append({
                    "person_id": 1, 
                    "pose_keypoints_2d": cleaned_keypoints
                })

                if draw_postfilter_selected:
                    # Draw filtered selected person on top
                    for pair in POSE_PAIRS:
                        partA, partB = pair[0], pair[1]
                        xA, yA, confA = cleaned_keypoints[partA]
                        xB, yB, confB = cleaned_keypoints[partB]
                        if confA > 0 and confB > 0:
                            cv2.line(final_image, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 3)

                    for point_data in cleaned_keypoints:
                        if point_data[2] > 0:
                            cv2.circle(final_image, (int(point_data[0]), int(point_data[1])), 4, (0, 0, 255), -1)
        else:
            # Distortion detected - reset index so no data is saved for this frame
            best_person_idx = -1

    # Reset/maintain tracking anchor based on whether a filtered person was selected.
    if debug_info["selected"]:
        missed_filtered_frames = 0
    else:
        missed_filtered_frames += 1
        if missed_filtered_frames >= reset_last_known_after_misses:
            last_known_hip_x = None

    # --- FINAL OUTPUT FOR THE FRAME ---

    if debug_mode and ((frames_processed % debug_print_every == 0) or (frames_processed < 5)):
        hip_debug_text = f" hips={'; '.join(debug_info['hips']) if debug_info['hips'] else 'none'}" if draw_hip_debug_overlay else ""
        print(f"DEBUG frame={frame_number} raw_people={debug_info['raw_people']} "
              f"valid_pos={debug_info['valid_position_people']} selected={debug_info['selected']} "
              f"last_x={last_known_hip_x}{hip_debug_text}")

    # Write processed frame to output video
    out_video.write(final_image)

    # Save frame metadata and keypoints into JSON
    json_data = {
        "frame_id": frame_number,
        "people_count": len(people_list_for_json),
        "people": people_list_for_json
    }

    json_path = os.path.join(json_dir, f"frame_{frame_number:06d}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    # Log progress
    if frames_processed % 50 == 0:
        elapsed = time.time() - start_time
        fps_proc = frames_processed / elapsed if elapsed > 0 else 0
        print(f"Processed frame {frame_number} ({frames_processed} done)... ({fps_proc:.1f} fps)")

    frame_number += 1
    frames_processed += 1

# Release resources
cap.release()
out_video.release()







