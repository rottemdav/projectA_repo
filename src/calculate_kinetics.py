import cv2
import json
import os
from typing import Dict, Any
from project_files.projectA_repo.src.kinetics_2d_lib import calculate_angles, calculate_gait_parameters, calculate_spatial_parameters

import numpy as np

from project_files.projectA_repo.src.kinetics_2d_lib import WHOLEBODY_KEYPOINTS


def load_keypoints_dict_from_json(json_path: str, person_mode: str = "first") -> Dict[str, Any]:
	"""
	Load keypoints JSON (from hrnet_video_process.py) and return a structured dictionary.

	Expected JSON format per frame:
	{
	  "frame_index": int,
	  "persons": [
		{
		  "bbox": [x1, y1, x2, y2],
		  "body": {"keypoints": [[x, y], ...], "scores": [...]},
		  "feet": {"keypoints": [[x, y], ...], "scores": [...]}
		},
		...
	  ]
	}

	Returns a dictionary with:
	- frame_indices: (N,) int array
	- keypoints: (N, 23, 3) float array [x, y, conf]
	- keypoints_by_name: {name: (N, 3) array}
	- has_person: (N,) bool array
	"""
	with open(json_path, "r") as f:
		frames = json.load(f)

	n_frames = len(frames)
	n_kpts = 23  # 17 body + 6 feet

	keypoints = np.full((n_frames, n_kpts, 3), np.nan, dtype=np.float32)
	frame_indices = np.zeros(n_frames, dtype=np.int32)
	has_person = np.zeros(n_frames, dtype=bool)

	for i, frame in enumerate(frames):
		frame_indices[i] = int(frame.get("frame_index", i))
		persons = frame.get("persons", [])

		if not persons:
			continue

		if person_mode == "largest_bbox":
			best_idx = 0
			best_area = -1.0
			for p_idx, person in enumerate(persons):
				bbox = person.get("bbox")
				if bbox is None or len(bbox) != 4:
					area = 0.0
				else:
					x1, y1, x2, y2 = bbox
					area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
				if area > best_area:
					best_area = area
					best_idx = p_idx
			person = persons[best_idx]
		else:
			person = persons[0]

		body_kp = np.asarray(person["body"]["keypoints"], dtype=np.float32)   # (17, 2)
		body_sc = np.asarray(person["body"]["scores"], dtype=np.float32)       # (17,)
		feet_kp = np.asarray(person["feet"]["keypoints"], dtype=np.float32)    # (6, 2)
		feet_sc = np.asarray(person["feet"]["scores"], dtype=np.float32)        # (6,)

		xy = np.vstack([body_kp, feet_kp])
		conf = np.concatenate([body_sc, feet_sc])[:, None]
		keypoints[i] = np.hstack([xy, conf])
		has_person[i] = True

	keypoints_by_name = {
		WHOLEBODY_KEYPOINTS[k]: keypoints[:, k, :]
		for k in range(n_kpts)
	}

	return {
		"frame_indices": frame_indices,
		"keypoints": keypoints,
		"keypoints_by_name": keypoints_by_name,
		"has_person": has_person,
	}

def draw_angle(frame, text, xy, color=(0, 255, 0)):
	x, y = int(xy[0]), int(xy[1])
	cv2.putText(frame, text, (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_joint_angle_arc(
	frame: np.ndarray,
	p1: np.ndarray,
	center: np.ndarray,
	p2: np.ndarray,
	color=(0, 255, 0),
	thickness: int = 2,
):
	"""Draw the geometric angle at center formed by rays center->p1 and center->p2."""
	v1 = p1 - center
	v2 = p2 - center
	n1 = np.linalg.norm(v1)
	n2 = np.linalg.norm(v2)
	if n1 < 1e-6 or n2 < 1e-6:
		return

	# Radius scales with visible segment lengths for stable visualization.
	radius = int(max(8, min(32, 0.25 * min(n1, n2))))
	a1 = np.arctan2(v1[1], v1[0])
	a2 = np.arctan2(v2[1], v2[0])
	delta = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi  # shortest signed arc

	steps = 24
	arc_angles = a1 + np.linspace(0.0, delta, steps)
	arc_pts = np.stack(
		[
			center[0] + radius * np.cos(arc_angles),
			center[1] + radius * np.sin(arc_angles),
		],
		axis=1,
	).astype(np.int32)

	c = tuple(center.astype(np.int32))
	end1 = tuple((center + (v1 / n1) * radius).astype(np.int32))
	end2 = tuple((center + (v2 / n2) * radius).astype(np.int32))
	cv2.line(frame, c, end1, color, thickness, cv2.LINE_AA)
	cv2.line(frame, c, end2, color, thickness, cv2.LINE_AA)
	cv2.polylines(frame, [arc_pts], False, color, thickness, cv2.LINE_AA)


def overlay_angles_on_video(video_path: str, output_path: str, keypoints: np.ndarray, conf_threshold: float = 0.2):
	"""Draw geometric joint-angle arcs on top of an existing HRNet skeleton video."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Could not open input video: {video_path}")

	fps = cap.get(cv2.CAP_PROP_FPS)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

	# COCO-WholeBody joint indices
	L_SHOULDER, R_SHOULDER = 5, 6
	L_HIP, R_HIP = 11, 12
	L_KNEE, R_KNEE = 13, 14
	L_ANKLE, R_ANKLE = 15, 16
	L_BIG_TOE, R_BIG_TOE = 17, 20

	i = 0
	n_frames = keypoints.shape[0]

	while True:
		ret, frame = cap.read()
		if not ret or i >= n_frames:
			break

		# Hip arcs: shoulder-hip-knee
		if (
			keypoints[i, L_SHOULDER, 2] >= conf_threshold
			and keypoints[i, L_HIP, 2] >= conf_threshold
			and keypoints[i, L_KNEE, 2] >= conf_threshold
		):
			draw_joint_angle_arc(frame, keypoints[i, L_SHOULDER, :2], keypoints[i, L_HIP, :2], keypoints[i, L_KNEE, :2], color=(255, 220, 0))
		if (
			keypoints[i, R_SHOULDER, 2] >= conf_threshold
			and keypoints[i, R_HIP, 2] >= conf_threshold
			and keypoints[i, R_KNEE, 2] >= conf_threshold
		):
			draw_joint_angle_arc(frame, keypoints[i, R_SHOULDER, :2], keypoints[i, R_HIP, :2], keypoints[i, R_KNEE, :2], color=(255, 220, 0))

		# Knee arcs: hip-knee-ankle
		if (
			keypoints[i, L_HIP, 2] >= conf_threshold
			and keypoints[i, L_KNEE, 2] >= conf_threshold
			and keypoints[i, L_ANKLE, 2] >= conf_threshold
		):
			draw_joint_angle_arc(frame, keypoints[i, L_HIP, :2], keypoints[i, L_KNEE, :2], keypoints[i, L_ANKLE, :2], color=(0, 255, 0))
		if (
			keypoints[i, R_HIP, 2] >= conf_threshold
			and keypoints[i, R_KNEE, 2] >= conf_threshold
			and keypoints[i, R_ANKLE, 2] >= conf_threshold
		):
			draw_joint_angle_arc(frame, keypoints[i, R_HIP, :2], keypoints[i, R_KNEE, :2], keypoints[i, R_ANKLE, :2], color=(0, 255, 0))

		# Ankle arcs: knee-ankle-toe
		if (
			keypoints[i, L_KNEE, 2] >= conf_threshold
			and keypoints[i, L_ANKLE, 2] >= conf_threshold
			and keypoints[i, L_BIG_TOE, 2] >= conf_threshold
		):
			draw_joint_angle_arc(frame, keypoints[i, L_KNEE, :2], keypoints[i, L_ANKLE, :2], keypoints[i, L_BIG_TOE, :2], color=(0, 200, 255))
		if (
			keypoints[i, R_KNEE, 2] >= conf_threshold
			and keypoints[i, R_ANKLE, 2] >= conf_threshold
			and keypoints[i, R_BIG_TOE, 2] >= conf_threshold
		):
			draw_joint_angle_arc(frame, keypoints[i, R_KNEE, :2], keypoints[i, R_ANKLE, :2], keypoints[i, R_BIG_TOE, :2], color=(0, 200, 255))

		writer.write(frame)
		i += 1

	cap.release()
	writer.release()
	print(f"Saved angle overlay video to: {output_path}")
	

if __name__ == "__main__":
	data = load_keypoints_dict_from_json("/home/projects/sipl-prj10496/project_files/data/hrnet_wholebody_output/20260403_180607/NL124_3_5_keypoints_20260403_180607_2700_to_3700.json")
	keypoints = data["keypoints"]  # shape: (N, 23, 3)
	angles = calculate_angles("COCO-WholeBody", keypoints)
	input_video = "/home/projects/sipl-prj10496/project_files/data/hrnet_wholebody_output/20260403_180607/NL124_3_5_pose_filtered_20260403_180607_2700_to_3700.mp4"
	output_video = "/home/projects/sipl-prj10496/project_files/data/hrnet_wholebody_output/20260403_180607/NL124_3_5_pose_filtered_20260403_180607_2700_to_3700_angles.mp4"
	overlay_angles_on_video(input_video, output_video, keypoints)

	print("Calculated angles keys:", list(angles.keys()))
	print("Sample LHip first 5 frames:", angles["LHip"][:5])
	