import pyarrow as pa

STEP_EVENTS_SCHEMA = pa.schema([
    #one row per heel strike event
    ("video_id", pa.string()),
    ("run_hash_id", pa.string()),
    ("frame_index", pa.int32()), #primary key for joining with keypoints data
    ("side", pa.string()),
    ("event_index", pa.int32()),
    ("hs_frame", pa.int32()),
    ("to_frame", pa.int32()),
    ("step_time_sec", pa.float32()),
    ("stance_time_sec", pa.float32()),
    ("swing_time_sec", pa.float32()),
    ("step_length_px", pa.float32()),
    ("step_length_px", pa.float32()),
    ("valid", pa.bool_()),
])

DOUBLE_SUPPORT_SCHEMA = pa.schema([
    #one row per double support event
    ("video_id", pa.string()),
    ("run_hash_id", pa.string()),
    ("frame_index", pa.int32()), #primary key for joining with keypoints data
    ("side", pa.string()),
    ("event_index", pa.int32()),
    ("start_frame", pa.int32()),
    ("end_frame", pa.int32()),
    ("duration_sec", pa.float32())
])

VIDEO_SUMMARY_SCHEMA = pa.schema([
    #one row per video
    ("video_id", pa.string()),
    ("run_hash_id", pa.string()),
    ("num_frames", pa.int32()),
    ("fps", pa.float32()),
    ("num_steps_right", pa.int32()),
    ("num_steps_left", pa.int32()),
    ("cadence_spm", pa.float32()),
    ("gait_speed_px_per_sec", pa.float32()),
    ("mean_step_time_right_sec", pa.float32()),
    ("mean_step_time_left_sec", pa.float32()),
    ("mean_step_length_right_px", pa.float32()),
    ("mean_step_length_left_px", pa.float32()),
    ("std_step_time_right_sec", pa.float32()),
    ("std_step_time_left_sec", pa.float32()),
    ("std_step_length_right_px", pa.float32()),
    ("std_step_length_left_px", pa.float32()),
    ("schema_version", pa.string())
    ])    