import json
import numpy as np
import math

from typing import Any, Dict, List, Tuple
from src.models.joint_model_mapping import WHOLEBODY_KEYPOINTS


def _extract_kpt_block(person: Dict[str, Any], keys: List[str], expected_n: int) -> np.ndarray:
    """
    Return (expected_n, 3) block [x, y, conf], padded with NaN if missing/short.
    Tries multiple aliases in `keys`.
    """
    out = np.full((expected_n, 3), np.nan, dtype=np.float32)

    block = None
    for k in keys:
        if k in person and person[k] is not None:
            block = person[k]
            break

    if block is None:
        return out

    kp = np.asarray(block.get("keypoints", []), dtype=np.float32)
    sc = np.asarray(block.get("scores", []), dtype=np.float32)

    if kp.ndim != 2 or kp.shape[1] < 2:
        return out

    n = min(expected_n, kp.shape[0], sc.shape[0])
    if n > 0:
        out[:n, :2] = kp[:n, :2]
        out[:n, 2] = sc[:n]
    return out


def load_keypoints_dict_from_json(
        
    json_path: str,
    person_mode: str = "first",
    model_type: str = "wholebody",  # "wholebody" -> 133, "body25" -> 25
) -> Dict[str, Any]:
    """
    Returns:
      - frame_indices: (N,)
      - keypoints: (N, K, 3) where K is 133 (wholebody) or 25 (body25)
      - keypoints_by_name: dict[name] -> (N, 3)
      - has_person: (N,)
    """
    with open(json_path, "r") as f:
        frames = json.load(f)

    if model_type == "wholebody":
        # COCO-WholeBody layout: 17 body + 6 foot + 68 face + 21 left hand + 21 right hand
        sections: List[Tuple[List[str], int]] = [
            (["body"], 17),
            (["feet", "foot"], 6),
            (["face"], 68),
            (["left_hand", "lefthand", "hand_left"], 21),
            (["right_hand", "righthand", "hand_right"], 21),
        ]
        n_kpts = 133
        names = WHOLEBODY_KEYPOINTS if len(WHOLEBODY_KEYPOINTS) == 133 else [f"kpt_{i}" for i in range(133)]

    elif model_type in ("body25", "openpose"):
        # Prefer explicit body25 block if present; fallback to body+feet and keep rest NaN.
        sections = [(["body25"], 25)]
        n_kpts = 25
        names = [f"kpt_{i}" for i in range(25)]

    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'wholebody' or 'openpose'.")

    n_frames = len(frames)
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

        if model_type == "body25" and "body25" not in person:
            # fallback from old format: body(17)+feet(6)=23; remaining 2 stay NaN
            body = _extract_kpt_block(person, ["body"], 17)
            feet = _extract_kpt_block(person, ["feet", "foot"], 6)
            stacked = np.vstack([body, feet])  # (23, 3)
            keypoints[i, :stacked.shape[0], :] = stacked
            has_person[i] = True
            continue

        cursor = 0
        for aliases, count in sections:
            block_arr = _extract_kpt_block(person, aliases, count)
            keypoints[i, cursor:cursor + count, :] = block_arr
            cursor += count

        has_person[i] = True

    keypoints_by_name = {names[k]: keypoints[:, k, :] for k in range(n_kpts)}

    return {
        "frame_indices": frame_indices,
        "keypoints": keypoints,  # (frames, keypoints, coordinates)
        "keypoints_by_name": keypoints_by_name,
        "has_person": has_person,
    }

def _json_safe_float(value: Any) -> Any:
    """Convert NaN/Inf float-like values to None for strict JSON output."""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(fv):
        return None
    return fv


def _json_safe_kpt_block(block_kp: np.ndarray, block_sc: np.ndarray) -> Dict[str, List[Any]]:
    """Return JSON-safe keypoint block where non-finite values become null."""
    keypoints_out = [[_json_safe_float(x), _json_safe_float(y)] for x, y in block_kp]
    scores_out = [_json_safe_float(s) for s in block_sc]
    return {"keypoints": keypoints_out, "scores": scores_out}


def save_keypoints_dict_to_json(
    keypoints_dict: Dict[str, Any],
    output_json_path: str,
    model_type: str = "wholebody",
) -> None:
    """
    Saves the keypoints dictionary to a JSON file in the expected format.
    The output format will be compatible with the input format of `load_keypoints_dict_from_json`.
    """

    frame_indices = keypoints_dict["frame_indices"]
    keypoints = keypoints_dict["keypoints"]  # (N, K, 3)
    has_person = keypoints_dict.get("has_person", None)

    if keypoints.ndim != 3 or keypoints.shape[2] != 3:
        raise ValueError(f"Expected keypoints shape (N, K, 3), got {keypoints.shape}")

    n_frames, n_kpts, _ = keypoints.shape

    if model_type == "wholebody":
        sections: List[Tuple[List[str], int]] = [
            (["body"], 17),
            (["feet", "foot"], 6),
            (["face"], 68),
            (["left_hand", "lefthand", "hand_left"], 21),
            (["right_hand", "righthand", "hand_right"], 21),
        ]
        expected_kpts = 133

    elif model_type == "openpose":
        sections = [(["body25"], 25)]
        expected_kpts = 25
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'wholebody' or 'openpose'.")

    if n_kpts != expected_kpts:
        raise ValueError(
            f"Keypoints K dimension mismatch for model_type='{model_type}': "
            f"expected {expected_kpts}, got {n_kpts}"
        )

    if len(frame_indices) != n_frames:
        raise ValueError(
            f"frame_indices length mismatch: expected {n_frames}, got {len(frame_indices)}"
        )

    if has_person is not None and len(has_person) != n_frames:
        raise ValueError(
            f"has_person length mismatch: expected {n_frames}, got {len(has_person)}"
        )
    
    frames = []
    for i in range(n_frames):
        frame_data = {"frame_index": int(frame_indices[i]), "persons": []}
        if has_person is not None and not bool(has_person[i]):
            frames.append(frame_data)
            continue

        person_data = {}
        cursor = 0
        for aliases, count in sections:
            block_kp = keypoints[i, cursor:cursor + count, :2]
            block_sc = keypoints[i, cursor:cursor + count, 2]
            block_dict = _json_safe_kpt_block(block_kp, block_sc)
            for alias in aliases:
                person_data[alias] = block_dict
            cursor += count
        frame_data["persons"].append(person_data)
        frames.append(frame_data)

    with open(output_json_path, "w") as f:
        json.dump(frames, f, indent=2, allow_nan=False)


def load_keypoints_dict_to_json(
    keypoints_dict: Dict[str, Any],
    output_json_path: str,
    model_type: str = "wholebody",
) -> None:
    """Backward-compatible alias for save_keypoints_dict_to_json."""
    save_keypoints_dict_to_json(
        keypoints_dict=keypoints_dict,
        output_json_path=output_json_path,
        model_type=model_type,
    )

