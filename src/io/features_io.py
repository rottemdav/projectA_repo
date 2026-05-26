import pyarrow as pa

STEP_EVENTS_SCHEMA = pa.schema([
    ("video_id", pa.string()),
    ("run_hash_id", pa.string()),
    ("frame_index", pa.int32()),  # use hs_frame as the join key
    ("side", pa.string()),
    ("event_index", pa.int32()),
    ("hs_frame", pa.int32()),
    ("to_frame", pa.int32()),
    ("step_time_s", pa.float32()),
    ("stance_time_s", pa.float32()),
    ("swing_time_s", pa.float32()),
    ("step_length_px", pa.float32()),
    ("valid", pa.bool_()),
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