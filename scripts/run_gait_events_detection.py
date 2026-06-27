import os
import sys
import pandas as pd
import numpy as np
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import copy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from datetime import datetime

from src.io.keypoints_io import load_keypoints_dict_from_json
from src.domain.gait_events_detection import ankle_to_pelvis_distance, gait_event_detection, infer_forward_axis_sign
from src.domain.gait_feature_extraction import (calculate_gait_parameters, 
                                                calculate_gait_parameters_2, 
                                                calculate_spatial_parameters, 
                                                calculate_angles,
                                                calculate_sagittal_2d_angles,
                                                add_step_direction)
from plotting.plot_gait_events_detection_timeseries import plot_ankle_to_pelvis_distance, save_figure
from src.models.joint_model_mapping import BODY25_GAIT_KEYPOINTS, WHOLEBODY_GAIT_KEYPOINTS
from src.io.features_io import (TEMPORAL_STEP_EVENTS_SCHEMA, VIDEO_SUMMARY_SCHEMA, SPATIAL_STEP_EVENTS_SCHEMA, SPATIAL_EVENT_COLUMNS, STEP_EVENT_COLUMNS, build_angle_step_rows,
                                build_gait_step_rows_from_events, build_gait_step_rows_from_events_2, build_steps_rows, build_spatial_step_rows_from_events)
from src.config import Config  

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
    
    parser.add_argument("--new_output",
                        action='store_true',
                        help="Whether to use the new version of gait parameter calculation and step row building functions.")
    parser.add_argument("--model",
                        type=str,
                        choices=["hrnet", "openpose"],
                        default="hrnet",
                        help="Which pose estimation model was used for the input keypoints (default: hrnet). This will determine the expected keypoint format and which joints are used for gait event detection.")
    parser.add_argument("--heel_strike_extrema",
                        type=str,
                        choices=["min", "max"],
                        default="max",
                        help="Which ankle-to-pelvis signal extremum should be tagged as heel strike. Use 'min' for the current treadmill direction convention, or 'max' if visual validation shows the labels are reversed.")
    
    return parser.parse_args()

def align_x_to_motion_axis(distance_data, forward_axis_sign=None):
    """Flip x-components so forward direction is consistently positive."""
    mid_pelvis_x = distance_data["mid_pelvis_loc"][:, 0]

    if forward_axis_sign is None:
        window_size = max(1, min(len(mid_pelvis_x) // 10, 30))
        start_x = np.nanmedian(mid_pelvis_x[:window_size])
        end_x = np.nanmedian(mid_pelvis_x[-window_size:])
        motion_dx = end_x - start_x
        forward_axis_sign = -1.0 if np.isfinite(motion_dx) and motion_dx < 0 else 1.0

    if forward_axis_sign < 0:
        for key in (
            "left_ankle_distance",
            "right_ankle_distance",
            "mid_pelvis_loc",
            "left_ankle",
            "right_ankle",
        ):
            distance_data[key][:, 0] *= -1.0

        for key in ("left_ankle_dx", "right_ankle_dx"):
            if key in distance_data:
                distance_data[key] *= -1.0

        print("Applied x-axis flip: subject faces decreasing image x.")
    else:
        print("No x-axis flip: subject faces increasing image x.")

    return distance_data

DATE = datetime.now().strftime('%Y%m%d_%H%M%S')

args = parse_args()

input_path = args.input_path if args.input_path else "/home/projects/sipl-prj10496/project_files/data/hrnet_wholebody_output/20260404_174122/HC65_3_keypoints_filtered_20260404_174122_3000_to_5000.json"
# Map CLI model to the exact model strings expected by downstream functions.
if args.model == "hrnet":
    keypoints_model_type = "wholebody"
    gait_model = "wholebody"
    angle_model = "wholeBody"
    sagittal_angle_model = "wholebody"
    spatial_model = "COCO-WholeBody"
    
elif args.model == "openpose":
    keypoints_model_type = "body25"
    gait_model = "body25"
    angle_model = "BODY25"
    sagittal_angle_model = "openpose"
    spatial_model = "BODY25"
else:
    raise ValueError(f"Unsupported model: {args.model}")

# extract the last part of the filename without extension for output naming
video_name = Path(input_path).stem.split("_keypoints")[0]
# extract the running hash id from the parent folder (e.g. 20260404_152056)
run_hash_id = Path(input_path).parent.name
if args.new_output:
    run_hash_id = DATE
Config.set_active_model(args.model)    
Config.OUTPUT_DIR = Config.get_output_dir()

output_path = os.path.join(
    Config.OUTPUT_DIR,
    Config.ACTIVE_MODEL.FILTERED_JSON_FILENAME_FORMAT.format(
        video_name=Config.VIDEO_NAME,
        DATE=Config.DATE,
        out_range=Config.FRAME_RANGE,
    ),
)

#output_path = f"/home/projects/sipl-prj10496/project_files/outputs/hrnet_wholebody_output/{run_hash_id}/{video_name}_gait_events_timeseries.png"
fps = 60.0

plot_output_path = os.path.join(
    Config.OUTPUT_DIR,
    f"{video_name}_gait_events_timeseries_{DATE}.png",
)

keypoints_dict = load_keypoints_dict_from_json(input_path, model_type=keypoints_model_type)
raw_distance_data = ankle_to_pelvis_distance(gait_model, keypoints_dict)

forward_axis_sign = infer_forward_axis_sign(gait_model, keypoints_dict)

angles = calculate_sagittal_2d_angles(
    sagittal_angle_model,
    keypoints_dict["keypoints"],
    forward_axis_sign=forward_axis_sign,
)
print("forward_axis_sign:", forward_axis_sign)

aligned_distance_data = align_x_to_motion_axis(
    copy.deepcopy(raw_distance_data),
    forward_axis_sign=forward_axis_sign,
)

gait_events = gait_event_detection(
    aligned_distance_data,
    frame_indices=keypoints_dict["frame_indices"],
    heel_strike_extrema=args.heel_strike_extrema,
)

angles = calculate_angles(angle_model, keypoints_dict["keypoints"])

#print("Available angles:", angles.keys())
#print("Left knee angles shape:", angles["LKnee"].shape)
#print("First 10 left knee angles:", angles["LKnee"][:10])

if args.plot_distance:
    fig = plot_ankle_to_pelvis_distance(
        aligned_distance_data,
        gait_events,
        frame_indices=keypoints_dict['frame_indices'],
        fps=fps,
    )
    save_figure(fig, plot_output_path)

time_vector = np.array(keypoints_dict['frame_indices']) / fps

gait_params = calculate_gait_parameters_2(keypoints_dict['keypoints'], time_vector, gait_events)

angle_step_rows = build_angle_step_rows(angles, gait_params)
angle_steps_df = pd.DataFrame(angle_step_rows)

#print(angle_steps_df.head())
np.set_printoptions(precision=3, suppress=True)

#print("Gait Parameters:")
#for param_name, param_value in gait_params.items():
#    if isinstance(param_value, dict):
#        print(f"{param_name}:")
#        for side, values in param_value.items():
#            print(f"  {side}: {values}")
#        print()
#    elif isinstance(param_value, np.ndarray):
#        print(f"{param_name}: {param_value}")  # or print summary stats
#    else:
#        print(f"{param_name}: {param_value:.3f}\n")

spatial_params = calculate_spatial_parameters(spatial_model, keypoints_dict['keypoints'], gait_events)

#print("Spatial Parameters:")
#for param_name, param_value in spatial_params.items():
#    if isinstance(param_value, dict):
#        print(f"{param_name}:")
#        for side, values in param_value.items():
#           print(f"  {side}: {values}")
#        print() # Adds a blank line for readability
#    else:
#        print(f"{param_name}: {param_value:.3f}\n")

## build rows for steps, double support, and video summary tables

# Map event indices -> frame numbers
frame_indices = np.array(keypoints_dict["frame_indices"])

lhs_frames = frame_indices[gait_events["lhs"]]
lto_frames = frame_indices[gait_events["lto"]]
rhs_frames = frame_indices[gait_events["rhs"]]
rto_frames = frame_indices[gait_events["rto"]]

step_rows = build_gait_step_rows_from_events_2(
    video_name=video_name,
    run_hash_id=run_hash_id,
    gait_params=gait_params,
)

spatial_rows = build_spatial_step_rows_from_events(
    video_name=video_name,
    run_hash_id=run_hash_id,
    gait_params=gait_params,
    spatial_params=spatial_params,
)

print("Number of step rows:", len(step_rows))
steps_df = pd.DataFrame(step_rows)
steps_df = add_step_direction(steps_df, raw_distance_data, forward_axis_sign)
steps_df = steps_df[STEP_EVENT_COLUMNS]  # Reorder columns to match schema
#print("steps_df:")
#print(steps_df)

spatial_df = pd.DataFrame(spatial_rows, columns=SPATIAL_EVENT_COLUMNS)

steps_table = pa.Table.from_pandas(steps_df, schema=TEMPORAL_STEP_EVENTS_SCHEMA, preserve_index=False)
spatial_table = pa.Table.from_pandas(spatial_df, schema=SPATIAL_STEP_EVENTS_SCHEMA, preserve_index=False)

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

features_dir = Config.OUTPUT_DIR

steps_path = f"{features_dir}/{video_name}_steps.parquet"
spatial_features_path = f"{features_dir}/{video_name}_spatial.parquet"
pq.write_table(steps_table, steps_path)
pq.write_table(spatial_table, spatial_features_path)

left_mask = gait_params["hs_sides"] == "left"
right_mask = gait_params["hs_sides"] == "right"

angle_steps_path = f"{features_dir}/{video_name}_angle_steps.parquet"
angle_steps_df.to_parquet(angle_steps_path, index=False)
print(f"Saved angle step features to {angle_steps_path}")

summary_df = pd.DataFrame([{
    "video_id": video_name,
    "run_hash_id": run_hash_id,
    "num_frames": int(len(frame_indices)),
    "fps": float(fps),
    "num_steps_left": int(np.sum(left_mask)),
    "num_steps_right": int(np.sum(right_mask)),
    "cadence_spm": float(spatial_params.get("cadence", np.nan)),
    "gait_speed_px_per_s": float(gait_params.get("gaitSpeed", np.nan)),
    "mean_step_time_right_s": float(np.nanmean(gait_params["stepTime"][right_mask])),
    "mean_step_time_left_s": float(np.nanmean(gait_params["stepTime"][left_mask])),
    "mean_step_length_right_px": float(np.nanmean(spatial_params["stepLength"]["right"])),
    "mean_step_length_left_px": float(np.nanmean(spatial_params["stepLength"]["left"])),
    "std_step_time_right_s": float(np.nanstd(gait_params["stepTime"][right_mask])),
    "std_step_time_left_s": float(np.nanstd(gait_params["stepTime"][left_mask])),
    "std_step_length_right_px": float(np.nanstd(spatial_params["stepLength"]["right"])),
    "std_step_length_left_px": float(np.nanstd(spatial_params["stepLength"]["left"])),
    "schema_version": "v1",
}])

summary_table = pa.Table.from_pandas(summary_df, schema=VIDEO_SUMMARY_SCHEMA, preserve_index=False)
summary_path = f"{features_dir}/{video_name}_summary.parquet"
pq.write_table(summary_table, summary_path)
print(f"Saved temporal step features to {steps_path}")
print(f"Saved video summary to {summary_path}")
print(f"Saved spatial features to {spatial_features_path}")
