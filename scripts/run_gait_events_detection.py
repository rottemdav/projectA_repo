import os
import sys
import pandas as pd
import numpy as np
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from datetime import datetime

from src.io.keypoints_io import load_keypoints_dict_from_json
from src.domain.gait_events_detection import ankle_to_pelvis_distance, gait_event_detection
from src.domain.gait_feature_extraction import calculate_gait_parameters, calculate_spatial_parameters
from plotting.plot_gait_events_detection_timeseries import plot_ankle_to_pelvis_distance, save_figure
from src.models.joint_model_mapping import BODY25_GAIT_KEYPOINTS, WHOLEBODY_GAIT_KEYPOINTS
from src.io.features_io import STEP_EVENTS_SCHEMA, VIDEO_SUMMARY_SCHEMA, build_steps_rows

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Events Detection from HRNet Keypoints")

    parser.add_argument("--plot_distance",
                        type=bool,
                        default=False,
                        help="Whether to plot the ankle to pelvis distance time series with detected gait events.")
    
    parser.add_argument("--input_path",
                        type=str,
                        default=None,
                        help="Path to the input JSON file containing keypoints data.")

    return parser.parse_args()


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

DATE = datetime.now().strftime('%Y%m%d_%H%M%S')

args = parse_args()

input_path = args.input_path if args.input_path else "/home/projects/sipl-prj10496/project_files/data/hrnet_wholebody_output/20260404_174122/HC65_3_keypoints_filtered_20260404_174122_3000_to_5000.json"
# extract the last part of the filename without extension for output naming
video_name = Path(input_path).stem.split("_keypoints")[0]
# extract the running hash id from the parent folder (e.g. 20260404_152056)
run_hash_id = Path(input_path).parent.name
output_path = f"/home/projects/sipl-prj10496/project_files/outputs/hrnet_wholebody_output/{run_hash_id}/{video_name}_gait_events_timeseries.png"
fps = 60.0


keypoints_dict = load_keypoints_dict_from_json(input_path, model_type="wholebody")
distance_data = ankle_to_pelvis_distance("wholebody", keypoints_dict)
distance_data = align_x_to_motion_axis(distance_data)
gait_events = gait_event_detection(distance_data, frame_indices=keypoints_dict['frame_indices'])

if args.plot_distance:
    fig = plot_ankle_to_pelvis_distance(
        distance_data,
        gait_events,
        frame_indices=keypoints_dict['frame_indices'],
        fps=fps,
    )
    save_figure(fig, output_path)

time_vector = np.array(keypoints_dict['frame_indices']) / fps

gait_params = calculate_gait_parameters(keypoints_dict['keypoints'], time_vector, gait_events)
np.set_printoptions(precision=3, suppress=True)

print("Gait Parameters:")
for param_name, param_value in gait_params.items():
    if isinstance(param_value, dict):
        # This handles stepTime, stanceTime, etc.
        print(f"{param_name}:")
        for side, values in param_value.items():
            print(f"  {side}: {values}")
        print() # Adds a blank line for readability
    else:
        # This handles gaitSpeed (a single float)
        print(f"{param_name}: {param_value:.3f}\n")

spatial_params = calculate_spatial_parameters("COCO-WholeBody", keypoints_dict['keypoints'], gait_events)

print("Spatial Parameters:")
for param_name, param_value in spatial_params.items():
    if isinstance(param_value, dict):
        print(f"{param_name}:")
        for side, values in param_value.items():
            print(f"  {side}: {values}")
        print() # Adds a blank line for readability
    else:
        print(f"{param_name}: {param_value:.3f}\n")

## build rows for steps, double support, and video summary tables

# Map event indices -> frame numbers
frame_indices = np.array(keypoints_dict["frame_indices"])
lhs_frames = frame_indices[gait_events["lhs"]]
lto_frames = frame_indices[gait_events["lto"]]
rhs_frames = frame_indices[gait_events["rhs"]]
rto_frames = frame_indices[gait_events["rto"]]

step_rows = []

step_rows += build_steps_rows(
    video_name,
    run_hash_id,
    "left",
    lhs_frames,
    lto_frames,
    gait_params["stepTime"]["left"],
    gait_params["stanceTime"]["left"],
    gait_params["swingTime"]["left"],
    spatial_params["stepLength"]["left"],
)

step_rows += build_steps_rows(
    video_name,
    run_hash_id,
    "right",
    rhs_frames,
    rto_frames,
    gait_params["stepTime"]["right"],
    gait_params["stanceTime"]["right"],
    gait_params["swingTime"]["right"],
    spatial_params["stepLength"]["right"],
)

steps_df = pd.DataFrame(step_rows)

steps_table = pa.Table.from_pandas(steps_df, schema=STEP_EVENTS_SCHEMA, preserve_index=False)

features_dir = f"/home/projects/sipl-prj10496/project_files/outputs/hrnet_wholebody_output/{run_hash_id}"
steps_path = f"{features_dir}/{video_name}_steps.parquet"
pq.write_table(steps_table, steps_path)

summary_df = pd.DataFrame([{
    "video_id": video_name,
    "run_hash_id": run_hash_id,
    "num_frames": int(len(frame_indices)),
    "fps": float(fps),
    "num_steps_left": int(len(gait_params["stepTime"]["left"])),
    "num_steps_right": int(len(gait_params["stepTime"]["right"])),
    "cadence_spm": float(spatial_params.get("cadence", np.nan)),
    "gait_speed_px_per_s": float(gait_params.get("gaitSpeed", np.nan)),
    "mean_step_time_right_s": float(np.nanmean(gait_params["stepTime"]["right"])),
    "mean_step_time_left_s": float(np.nanmean(gait_params["stepTime"]["left"])),
    "mean_step_length_right_px": float(np.nanmean(spatial_params["stepLength"]["right"])),
    "mean_step_length_left_px": float(np.nanmean(spatial_params["stepLength"]["left"])),
    "std_step_time_right_s": float(np.nanstd(gait_params["stepTime"]["right"])),
    "std_step_time_left_s": float(np.nanstd(gait_params["stepTime"]["left"])),
    "std_step_length_right_px": float(np.nanstd(spatial_params["stepLength"]["right"])),
    "std_step_length_left_px": float(np.nanstd(spatial_params["stepLength"]["left"])),
    "schema_version": "v1",
}])

summary_table = pa.Table.from_pandas(summary_df, schema=VIDEO_SUMMARY_SCHEMA, preserve_index=False)
summary_path = f"{features_dir}/{video_name}_summary.parquet"
pq.write_table(summary_table, summary_path)
print(f"Saved step events to {steps_path}")
print(f"Saved video summary to {summary_path}")