import pyarrow as pa
import numpy as np

TEMPORAL_STEP_EVENTS_SCHEMA = pa.schema([
    ("video_id", pa.string()),
    ("run_hash_id", pa.string()),
    ("global_step_index", pa.int64()),
    ("side", pa.string()),

    ("hs_frame", pa.int64()),
    ("prev_opposite_hs_frame", pa.int64()),
    ("to_frame", pa.int64()),
    ("next_same_side_hs_frame", pa.int64()),

    ("step_time_s", pa.float64()),
    ("stance_time_s", pa.float64()),
    ("swing_time_s", pa.float64()),

    ("relative_foot_dx_px", pa.float64()),
    ("step_direction", pa.string()),


    ("valid", pa.bool_()),
])

SPATIAL_STEP_EVENTS_SCHEMA = pa.schema([
    ("video_id", pa.string()),
    ("run_hash_id", pa.string()),
    ("global_step_index", pa.int64()),
    ("side", pa.string()),

    ("hs_frame", pa.int64()),
    ("prev_opposite_hs_frame", pa.int64()),
    ("to_frame", pa.int64()),
    ("next_same_side_hs_frame", pa.int64()),

    ("step_length_px", pa.float64()),

])

VIDEO_SUMMARY_SCHEMA = pa.schema([
    ("video_id", pa.string()),
    ("run_hash_id", pa.string()),
    ("num_frames", pa.int32()),
    ("fps", pa.float32()),
    ("num_steps_right", pa.int32()),
    ("num_steps_left", pa.int32()),
    ("cadence_spm", pa.float32()),
    ("gait_speed_px_per_s", pa.float32()),
    ("mean_step_time_right_s", pa.float32()),
    ("mean_step_time_left_s", pa.float32()),
    ("mean_step_length_right_px", pa.float32()),
    ("mean_step_length_left_px", pa.float32()),
    ("std_step_time_right_s", pa.float32()),
    ("std_step_time_left_s", pa.float32()),
    ("std_step_length_right_px", pa.float32()),
    ("std_step_length_left_px", pa.float32()),
    ("schema_version", pa.string()),
])

STEP_EVENT_COLUMNS = [
    "video_id",
    "run_hash_id",
    "global_step_index",
    "side",
    "hs_frame",
    "prev_opposite_hs_frame",
    "to_frame",
    "next_same_side_hs_frame",
    "step_time_s",
    "stance_time_s",
    "swing_time_s",
    "relative_foot_dx_px",
    "step_direction",
    "valid",
]

SPATIAL_EVENT_COLUMNS = [
    "video_id",
    "run_hash_id",
    "global_step_index",
    "side",
    "hs_frame",
    "prev_opposite_hs_frame",
    "to_frame",
    "next_same_side_hs_frame",
    "step_length_px",
]

# FIXME: old version, to be removed after testing

def build_steps_rows(video_id, run_hash_id, side, hs_frames, to_frames,
                     step_time, stance_time, swing_time, step_length):
    n = min(len(hs_frames), len(to_frames), len(step_time),
            len(stance_time), len(swing_time), len(step_length))
    rows = []
    for i in range(n):
        rows.append({
            "video_id": video_id,
            "run_hash_id": run_hash_id,
            "frame_index": int(hs_frames[i]),
            "side": side,
            "event_index": i,
            "hs_frame": int(hs_frames[i]),
            "to_frame": int(to_frames[i]),
            "step_time_s": float(step_time[i]),
            "stance_time_s": float(stance_time[i]),
            "swing_time_s": float(swing_time[i]),
            "step_length_px": float(step_length[i]),
            "valid": True,
        })
    return rows

# FIXME end

def compute_stance_swing_for_side(hs_frames, to_frames, fps):
    hs_frames = sorted(hs_frames)
    to_frames = sorted(to_frames)

    rows = []

    for hs in hs_frames:
        # first toe-off after this heel strike
        next_to_candidates = [to for to in to_frames if to > hs]
        if not next_to_candidates:
            continue

        to = next_to_candidates[0]

        # first heel strike after that toe-off
        next_hs_candidates = [h for h in hs_frames if h > to]
        if not next_hs_candidates:
            continue

        next_hs = next_hs_candidates[0]

        rows.append({
            "hs_frame": hs,
            "to_frame": to,
            "next_hs_frame": next_hs,
            "stance_time_s": (to - hs) / fps,
            "swing_time_s": (next_hs - to) / fps,
        })

    return rows

# FIXME: old version, to be removed after testing

def build_gait_step_rows_from_events(video_name, run_hash_id, lhs_frames, rhs_frames, lto_frames, rto_frames, fps):
    rows = []

    #build chronological heel-strike sequence 
    hs_events = []

    for frame in lhs_frames:
        hs_events.append((frame, "left"))

    for frame in rhs_frames:
        hs_events.append((frame, "right"))

    hs_events = sorted(hs_events)

    #compute step time
    step_time_by_hs = {}
    global_step_index = 0

    for i in range (1, len(hs_events)):
        prev_frame, prev_side = hs_events[i-1]
        curr_frame, curr_side = hs_events[i]

        if prev_side == curr_side:
            continue

        step_time_s = (curr_frame - prev_frame) / fps

        step_time_by_hs[(curr_side, curr_frame)] = {
            "step_time_s": step_time_s,
            "prev_opposite_hs_frame": prev_frame,
            "global_step_index": global_step_index,
        }

        global_step_index += 1

    #compute stance/swing for each side
    left_phase = compute_stance_swing_for_side(lhs_frames, lto_frames, fps)
    right_phase = compute_stance_swing_for_side(rhs_frames, rto_frames, fps)

    phase_by_hs = {}

    for row in left_phase:
        phase_by_hs[("left", row["hs_frame"])] = row

    for row in right_phase:
        phase_by_hs[("right", row["hs_frame"])] = row

    print("len(lhs_frames):", len(lhs_frames))
    print("len(rhs_frames):", len(rhs_frames))
    print("len(lto_frames):", len(lto_frames))
    print("len(rto_frames):", len(rto_frames))

    print("len(hs_events):", len(hs_events))
    print("len(step_time_by_hs):", len(step_time_by_hs))
    print("len(left_phase):", len(left_phase))
    print("len(right_phase):", len(right_phase))
    print("len(phase_by_hs):", len(phase_by_hs))

    step_keys = set(step_time_by_hs.keys())
    phase_keys = set(phase_by_hs.keys())

    print("intersection:", len(step_keys & phase_keys))
    print("step only example:", list(step_keys - phase_keys)[:10])
    print("phase only example:", list(phase_keys - step_keys)[:10])

    for hs_frame,side in hs_events:
        step_info = step_time_by_hs.get((side, hs_frame))
        phase_info = phase_by_hs.get((side, hs_frame))

        if step_info is None or phase_info is None:
            continue

        rows.append({
            "video_id": video_name,
            "run_hash_id": run_hash_id,
            "global_step_index": step_info["global_step_index"],
            "side": side,

            "hs_frame": hs_frame,
            "prev_opposite_hs_frame": step_info["prev_opposite_hs_frame"],
            "to_frame": phase_info["to_frame"],
            "next_same_side_hs_frame": phase_info["next_hs_frame"],

            "step_time_s": step_info["step_time_s"],
            "stance_time_s": phase_info["stance_time_s"],
            "swing_time_s": phase_info["swing_time_s"],

            "valid": (
                0.25 <= step_info["step_time_s"] <= 1.5 
                #0.2 <= phase_info["stance_time_s"] <= 2.0 and
                #0.2 <= phase_info["swing_time_s"] <= 2.0
            ),
        })

    return rows

# FIXME end

def build_gait_step_rows_from_events_2(video_name, run_hash_id, gait_params):
    rows = []

    for i in range(len(gait_params["hs_frames"])):
        step_time_sec = gait_params["stepTime"][i]
        stance_time_sec = gait_params["stanceTime"][i]
        swing_time_sec = gait_params["swingTime"][i]

        rows.append({
            #np.array(hs_frames, dtype=int)
            "video_id": str(video_name),
            "run_hash_id": str(run_hash_id),
            "global_step_index": int(i),  # i is chronological
            "side": str(gait_params["hs_sides"][i]),
            "hs_frame": int(gait_params["hs_frames"][i]),
            "prev_opposite_hs_frame": int(gait_params["prev_opposite_hs_frames"][i]),
            "to_frame": int(gait_params["to_frames"][i]),
            "next_same_side_hs_frame": int(gait_params["next_same_side_hs_frames"][i]),
            "step_time_s": float(step_time_sec) if step_time_sec is not None else float("nan"),
            "stance_time_s": float(stance_time_sec) if stance_time_sec is not None else float("nan"),
            "swing_time_s": float(swing_time_sec) if swing_time_sec is not None else float("nan"),
            "valid": 0.25 <= float(step_time_sec) <= 1.5 if step_time_sec is not None else False,
        })

    return rows

def build_spatial_step_rows_from_events(video_name, run_hash_id, gait_params, spatial_params):
    rows = []

    # side-specific arrays
    step_len_left = spatial_params["stepLength"]["left"]
    step_len_right = spatial_params["stepLength"]["right"]

    left_idx = 0
    right_idx = 0

    for i in range(len(gait_params["hs_frames"])):
        side = gait_params["hs_sides"][i]

        if side == "left":
            step_length_px = step_len_left[left_idx] if left_idx < len(step_len_left) else None
            left_idx += 1
        else:
            step_length_px = step_len_right[right_idx] if right_idx < len(step_len_right) else None
            right_idx += 1

        rows.append({
            "video_id": str(video_name),
            "run_hash_id": str(run_hash_id),
            "global_step_index": int(i),
            "side": str(side),
            "hs_frame": int(gait_params["hs_frames"][i]),
            "prev_opposite_hs_frame": int(gait_params["prev_opposite_hs_frames"][i]),
            "to_frame": int(gait_params["to_frames"][i]),
            "next_same_side_hs_frame": int(gait_params["next_same_side_hs_frames"][i]),
            "step_length_px": float(step_length_px) if step_length_px is not None else float("nan"),
        })

    return rows

def build_angle_step_rows(angles, gait_params):
    rows = []

    for i, hs_frame in enumerate(gait_params["hs_frames"]):
        side = gait_params["hs_sides"][i]
        next_same_hs = gait_params["next_same_side_hs_frames"][i]
        toe_off = gait_params["to_frames"][i]

        if next_same_hs < 0:
            continue

        if side == "left":
            hip_key = "LHip"
            knee_key = "LKnee"
            ankle_key = "LAnkle"
        else:
            hip_key = "RHip"
            knee_key = "RKnee"
            ankle_key = "RAnkle"

        start = int(hs_frame)
        end = int(next_same_hs)

        hip_cycle = angles[hip_key][start:end + 1]
        knee_cycle = angles[knee_key][start:end + 1]
        ankle_cycle = angles[ankle_key][start:end + 1]

        row = {
            "global_step_index": int(i),
            "side": str(side),
            "hs_frame": int(hs_frame),
            "to_frame": int(toe_off),
            "next_same_side_hs_frame": int(next_same_hs),

            "hip_angle_at_hs_deg": float(angles[hip_key][start]),
            "knee_angle_at_hs_deg": float(angles[knee_key][start]),
            "ankle_angle_at_hs_deg": float(angles[ankle_key][start]),

            "hip_rom_deg": float(np.nanmax(hip_cycle) - np.nanmin(hip_cycle)),
            "knee_rom_deg": float(np.nanmax(knee_cycle) - np.nanmin(knee_cycle)),
            "ankle_rom_deg": float(np.nanmax(ankle_cycle) - np.nanmin(ankle_cycle)),

            "hip_min_deg": float(np.nanmin(hip_cycle)),
            "hip_max_deg": float(np.nanmax(hip_cycle)),
            "knee_min_deg": float(np.nanmin(knee_cycle)),
            "knee_max_deg": float(np.nanmax(knee_cycle)),
            "ankle_min_deg": float(np.nanmin(ankle_cycle)),
            "ankle_max_deg": float(np.nanmax(ankle_cycle)),
        }

        if toe_off >= 0:
            row.update({
                "hip_angle_at_to_deg": float(angles[hip_key][toe_off]),
                "knee_angle_at_to_deg": float(angles[knee_key][toe_off]),
                "ankle_angle_at_to_deg": float(angles[ankle_key][toe_off]),
            })
        else:
            row.update({
                "hip_angle_at_to_deg": np.nan,
                "knee_angle_at_to_deg": np.nan,
                "ankle_angle_at_to_deg": np.nan,
            })

        rows.append(row)

    return rows