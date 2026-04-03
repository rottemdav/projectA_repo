import sys
import cv2
import os
import json
import numpy as np
import time

try:
    sys.path.insert(0, '/home/projects/sipl-prj10496/project_files/openpose/build/python/openpose')
    import pyopenpose as op
except ImportError as e:
    print("Error: Could not find OpenPose library.")
    raise e

params = dict()
params["model_folder"] = "/home/projects/sipl-prj10496/project_files/openpose/models/"
#params["tracking"] = 1
params["number_people_max"] = 2
params["render_pose"] = 1

try:
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
except Exception as e:
    print(f"Error starting OpenPose: {e}")
    sys.exit(1)
    

video_path = "/home/projects/sipl-prj10496/project_files/data/source_videos/NL129/NL129_3_1.MP4"
video_filename = os.path.splitext(os.path.basename(video_path))[0]

parts = video_filename.split('_') 

camera_view = parts[1]

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video: {video_path}")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if fps <= 0:
    fps = 30  # fallback default
    print(f"Warning: Could not detect FPS from video. Using default: {fps}")

if total_frames <= 0:
    print("Error: Video reports 0 total frames.")
    cap.release()
    sys.exit(1)

# Generate unique output paths (avoid overwriting existing files)
output_base = f"/home/projects/sipl-prj10496/project_files/data/output_videos_hip/{video_filename}_woman"
output_video_path = f"{output_base}.mp4"
counter = 1
while os.path.exists(output_video_path):
    output_video_path = f"{output_base}_{counter}.mp4"
    counter += 1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

json_base = f"/home/projects/sipl-prj10496/project_files/data/output_jsons/{video_filename}_new_output_json"
json_dir = json_base
counter = 1
while os.path.exists(json_dir):
    json_dir = f"{json_base}_{counter}"
    counter += 1
os.makedirs(json_dir)

print(f"Starting processing... Video output: {output_video_path}")

# Process video frames
# Window: 03:55 to 03:59
start_time_sec = 3 * 60 + 55
end_time_sec = 3 * 60 + 59

processed_length_dur = max(0, end_time_sec - start_time_sec)
start_frame = int(fps * start_time_sec)

if start_frame >= total_frames:
    print(f"Warning: start_frame={start_frame} is beyond video length ({total_frames} frames). Resetting start_frame to 0.")
    start_frame = 0

end_frame = min(start_frame + int(fps * processed_length_dur), total_frames)
print(f"Processing frames {start_frame} to {end_frame} ({processed_length_dur} seconds at {fps:.1f} fps, total_frames={total_frames})...")

# Jump to start position
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

start_time = time.time()
frame_number = start_frame
frames_processed = 0

POSE_PAIRS = [
    [1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
    [8,9], [9,10], [10,11], [8,12], [12,13], [13,14],
    [1,0], [0,15], [15,17], [0,16], [16,18],
    [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]
]

# Lower body keypoints (hips, knees, ankles, and foot points)
LOWER_BODY_KEYPOINTS = [8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]

HIP_Y_LIMIT = 1200
PERSON_TRACK_CONF_THRESH = 0.30   # was 0.40
KEYPOINT_KEEP_CONF_THRESH = 0.45  # was 0.60
last_known_hip_x = None

while cap.isOpened():
    if frame_number >= end_frame:
        break

    ret, frame = cap.read()
    if not ret:
        break

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    final_image = frame.copy()
    people_list_for_json = []

    if datum.poseKeypoints is not None and datum.poseKeypoints.size > 0:
        all_keypoints = datum.poseKeypoints.tolist()
        
        best_person_idx = -1
        min_distance = float('inf')

        for i, keypoints in enumerate(all_keypoints):
            mid_hip_x = keypoints[8][0]
            mid_hip_y = keypoints[8][1]
            mid_hip_conf = keypoints[8][2]

            if mid_hip_conf > PERSON_TRACK_CONF_THRESH and mid_hip_y < HIP_Y_LIMIT:
                if last_known_hip_x is None:
                    best_person_idx = i
                    break 
                else:
                    distance = abs(mid_hip_x - last_known_hip_x)
                    if distance < min_distance and distance < 350:
                        min_distance = distance
                        best_person_idx = i

        if best_person_idx != -1:
            best_keypoints = all_keypoints[best_person_idx]
            last_known_hip_x = best_keypoints[8][0]

            cleaned_keypoints = []
            for point in best_keypoints:
                x, y, conf = point
                if conf > KEYPOINT_KEEP_CONF_THRESH:
                    cleaned_keypoints.append([x, y, conf])
                else:
                    cleaned_keypoints.append([0.0, 0.0, 0.0])

            people_list_for_json.append({
                "person_id": 1, 
                "pose_keypoints_2d": cleaned_keypoints
            })

            for pair in POSE_PAIRS:
                partA, partB = pair[0], pair[1]
                xA, yA, confA = cleaned_keypoints[partA]
                xB, yB, confB = cleaned_keypoints[partB]

                if confA > 0 and confB > 0:
                    cv2.line(final_image, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 3)

            for point_index, point_data in enumerate(cleaned_keypoints):
                x, y, conf = point_data
                if conf > 0:
                    cv2.circle(final_image, (int(x), int(y)), 4, (0, 0, 255), -1)

    out_video.write(final_image)

    json_data = {
        "frame_id": frame_number,
        "people_count": len(people_list_for_json),
        "people": people_list_for_json
    }

    json_file = os.path.join(json_dir, f"frame_{frame_number:06d}.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=4)

    if frames_processed % 50 == 0:
        elapsed = time.time() - start_time
        fps_processing = frames_processed / elapsed if elapsed > 0 else 0
        print(
            f"Processed frame {frame_number} "
            f"({frames_processed} frames done)... "
            f"({elapsed:.1f}s elapsed, {fps_processing:.1f} fps)"
        )

    frame_number += 1
    frames_processed += 1

cap.release()
out_video.release()

total_time = time.time() - start_time
avg_fps = frames_processed / total_time if total_time > 0 else 0
print(f"Done. Processed {frames_processed} frames ({start_frame} to {frame_number}) in {total_time:.1f} seconds ({avg_fps:.1f} fps average)")