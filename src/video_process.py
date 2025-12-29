import sys
import cv2
import os
import json
import numpy as np
import time

try:
    sys.path.append('/home/projects/sipl-prj10496/project_files/openpose/build/python/openpose')
    import pyopenpose as op
except ImportError as e:
    print("Error: Could not find OpenPose library.")
    raise e

params = dict()
params["model_folder"] = "/home/projects/sipl-prj10496/project_files/openpose/models/"
params["tracking"] = 1
params["number_people_max"] = 1
params["render_pose"] = 1

try:
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
except Exception as e:
    print(f"Error starting OpenPose: {e}")
    sys.exit(1)

video_path = "/home/projects/sipl-prj10496/project_files/data/source_videos/NL124/4-3336/GX010030[1].MP4"
video_filename = os.path.splitext(os.path.basename(video_path))[0]
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Generate unique output paths (avoid overwriting existing files)
output_base = f"/home/projects/sipl-prj10496/project_files/data/output_videos/{video_filename}"
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

processed_length_dur = 30
start_frame = 1800

end_frame = start_frame + int(fps * processed_length_dur)
print(f"Processing frames {start_frame} to {end_frame} ({processed_length_dur} seconds at {fps:.1f} fps)...")

# Jump to start position
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

start_time = time.time()
frame_number = start_frame
frames_processed = 0

while cap.isOpened():
    if frame_number >= end_frame:
        break

    ret, frame = cap.read()
    if not ret:
        break

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    final_image = datum.cvOutputData if datum.cvOutputData is not None else frame
    
    people_list_for_json = []

    if datum.poseKeypoints is not None and datum.poseKeypoints.size > 0:
        all_keypoints = datum.poseKeypoints.tolist()
        
        if datum.poseIds is not None:
            all_ids = datum.poseIds.tolist()
        else:
            all_ids = list(range(len(all_keypoints)))

        for i, (person_id, keypoints) in enumerate(zip(all_ids, all_keypoints)):
            people_list_for_json.append({
                "person_id": int(person_id),
                "pose_keypoints_2d": keypoints
            })

            for point_index, point_data in enumerate(keypoints):
                x, y, confidence = point_data
                
                if confidence > 0.1:
                    px, py = int(x), int(y)
                    cv2.putText(final_image, str(point_index), (px, py), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            nose_x, nose_y = int(keypoints[0][0]), int(keypoints[0][1])
            if nose_x > 0:
                cv2.putText(final_image, f"ID: {person_id}", (nose_x, nose_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_video.write(final_image)

    json_data = {
        "frame_id": frame_number,
        "people_count": len(people_list_for_json),
        "people": people_list_for_json
    }
    
    json_file = os.path.join(json_dir, f"frame_{frame_number:06d}.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=4)

    if frame_number % 50 == 0:
        elapsed = time.time() - start_time
        fps_processing = frames_processed / elapsed if elapsed > 0 else 0
        print(f"Processed frame {frame_number} ({frames_processed} frames done)... ({elapsed:.1f}s elapsed, {fps_processing:.1f} fps)")
        
    frame_number += 1
    frames_processed += 1

cap.release()
out_video.release()

total_time = time.time() - start_time
avg_fps = frames_processed / total_time if total_time > 0 else 0
print(f"Done. Processed {frames_processed} frames ({start_frame} to {frame_number}) in {total_time:.1f} seconds ({avg_fps:.1f} fps average)")