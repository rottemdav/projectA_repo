import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from datetime import datetime

from src.io.keypoints_io import load_keypoints_dict_from_json
from src.domain.gait_events_detection import ankle_to_pelvis_distance, gait_event_detection
from plotting.plot_gait_events_detection_timeseries import plot_ankle_to_pelvis_distance, save_figure

DATE = datetime.now().strftime('%Y%m%d_%H%M%S')

input_path = "/home/projects/sipl-prj10496/project_files/data/hrnet_wholebody_output/20260404_174122/HC65_3_keypoints_filtered_20260404_174122_3000_to_5000.json"
# extract the last part of the filename without extension for output naming
video_name = Path(input_path).stem.split("_keypoints")[0]
# extract the running hash id from the parent folder (e.g. 20260404_152056)
run_hash_id = Path(input_path).parent.name
output_path = f"/home/projects/sipl-prj10496/project_files/outputs/hrnet_wholebody_output/{run_hash_id}/{video_name}_gait_events_timeseries.png"
fps = 60.0


def align_x_to_motion_axis(distance_data):
    """Flip x-components so progression direction is consistently positive."""
    mid_pelvis_x = distance_data["mid_pelvis_loc"][:, 0]
    if len(mid_pelvis_x) < 2:
        return distance_data

    # find the frame which the mid_pelvis is smallest (clostest to the left margin)
    leftmost_frame = np.argmin(mid_pelvis_x)

    # if the leftmost frame is larger than the 0, then the motion is from right to left, and we need to flip the x-axis to make it left to right (positive progression)
    if leftmost_frame > 0:
        for key in (
            "left_ankle_distance",
            "right_ankle_distance",
            "mid_pelvis_loc",
            "left_ankle",
            "right_ankle",
        ):
            distance_data[key][:, 0] *= -1.0
        print("Applied x-axis flip: detected right-to-left motion.")
    else:
        print("No x-axis flip: detected left-to-right motion.")

    return distance_data

keypoints_dict = load_keypoints_dict_from_json(input_path, model_type="wholebody")
distance_data = ankle_to_pelvis_distance("wholebody", keypoints_dict)
distance_data = align_x_to_motion_axis(distance_data)
gait_events = gait_event_detection(distance_data, frame_indices=keypoints_dict['frame_indices'])

fig = plot_ankle_to_pelvis_distance(
    distance_data,
    gait_events,
    frame_indices=keypoints_dict['frame_indices'],
    fps=fps,
)
save_figure(fig, output_path)

