"""
Post-processing script for temporal filtering and residual analysis of pose keypoints.

This script loads keypoints from JSON files (output from mmpose_vitpose_process.py),
applies temporal filtering, and performs residual analysis to determine optimal
cutoff frequencies.

Usage:
    python filter_keypoints_json.py --json-path <path_to_json> --output-dir <output_directory>
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from scipy.signal import filtfilt, butter
from typing import List, Tuple, Dict, Optional


class KeypointPostProcessor:
    """Post-processor for keypoint data with temporal filtering and analysis."""
    
    # WholeBody keypoint indices
    BODY_INDICES = list(range(17))  # 0-16: body keypoints
    FOOT_INDICES = list(range(17, 23))  # 17-22: feet keypoints
    FACE_INDICES = list(range(23, 91))  # 23-90: face keypoints
    LEFT_HAND_INDICES = list(range(91, 112))  # 91-111: left hand
    RIGHT_HAND_INDICES = list(range(112, 133))  # 112-132: right hand
    
    def __init__(self, fs: int = 60, conf_threshold: float = 0.2):
        """
        Initialize the post-processor.
        
        Args:
            fs: Sampling frequency in Hz
            conf_threshold: Confidence threshold for keypoints
        """
        self.fs = fs
        self.conf_threshold = conf_threshold

    def keypoints_to_array(self, all_frames: List[dict]) -> np.ndarray:
        """Convert list of frames with keypoints to a numpy array."""
        num_frames = len(all_frames)
        if num_frames == 0:
            return np.empty((0, 0, 3), dtype=np.float32)

        # Detect which format the JSON is in
        first_frame = all_frames[0]
        if 'persons' in first_frame and len(first_frame['persons']) > 0:
            first_person = first_frame['persons'][0]
            
            # Check if it has full 'keypoints' (simple format)
            if 'keypoints' in first_person:
                # Simple format: persons[0] has 'keypoints' directly
                first_kp = first_person['keypoints']
            elif 'body' in first_person:
                # Complex format: body/feet/face/hands separated
                # Reconstruct full 133-keypoint array from parts
                return self._keypoints_to_array_from_parts(all_frames)
            else:
                raise ValueError(f"Unknown keypoint format. Keys in person: {first_person.keys()}")
        else:
            raise ValueError("No persons found in frames")

        # Process simple format
        if isinstance(first_kp, list):
            first_kp = np.array(first_kp, dtype=np.float32)
        num_keypoints = first_kp.shape[0]
        keypoints_array = np.zeros((num_frames, num_keypoints, 3), dtype=np.float32)

        for t, frame in enumerate(all_frames):
            if len(frame['persons']) > 0:
                kp = frame['persons'][0]['keypoints']
                sc = frame['persons'][0]['scores']
                # Convert to numpy arrays if they're lists
                if isinstance(kp, list):
                    kp = np.array(kp, dtype=np.float32)
                if isinstance(sc, list):
                    sc = np.array(sc, dtype=np.float32)
                keypoints_array[t, :, :2] = kp
                keypoints_array[t, :, 2] = sc
            else:
                keypoints_array[t, :, :] = 0.0  # No detection

        return keypoints_array

    def _keypoints_to_array_from_parts(self, all_frames: List[dict]) -> np.ndarray:
        """Convert frames with separated body/feet/face/hands keypoints to full array."""
        num_frames = len(all_frames)
        # WholeBody has 133 keypoints total
        num_keypoints = 133
        keypoints_array = np.zeros((num_frames, num_keypoints, 3), dtype=np.float32)

        for t, frame in enumerate(all_frames):
            if len(frame['persons']) > 0:
                person = frame['persons'][0]
                
                # Body (0-16)
                if 'body' in person:
                    body_kp = person['body']['keypoints']
                    body_sc = person['body']['scores']
                    if isinstance(body_kp, list):
                        body_kp = np.array(body_kp, dtype=np.float32)
                        body_sc = np.array(body_sc, dtype=np.float32)
                    keypoints_array[t, 0:17, :2] = body_kp
                    keypoints_array[t, 0:17, 2] = body_sc
                
                # Feet (17-22)
                if 'feet' in person:
                    feet_kp = person['feet']['keypoints']
                    feet_sc = person['feet']['scores']
                    if isinstance(feet_kp, list):
                        feet_kp = np.array(feet_kp, dtype=np.float32)
                        feet_sc = np.array(feet_sc, dtype=np.float32)
                    keypoints_array[t, 17:23, :2] = feet_kp
                    keypoints_array[t, 17:23, 2] = feet_sc
                
                # Face (23-90)
                if 'face' in person:
                    face_kp = person['face']['keypoints']
                    face_sc = person['face']['scores']
                    if isinstance(face_kp, list):
                        face_kp = np.array(face_kp, dtype=np.float32)
                        face_sc = np.array(face_sc, dtype=np.float32)
                    keypoints_array[t, 23:91, :2] = face_kp
                    keypoints_array[t, 23:91, 2] = face_sc
                
                # Left hand (91-111)
                if 'left_hand' in person:
                    lh_kp = person['left_hand']['keypoints']
                    lh_sc = person['left_hand']['scores']
                    if isinstance(lh_kp, list):
                        lh_kp = np.array(lh_kp, dtype=np.float32)
                        lh_sc = np.array(lh_sc, dtype=np.float32)
                    keypoints_array[t, 91:112, :2] = lh_kp
                    keypoints_array[t, 91:112, 2] = lh_sc
                
                # Right hand (112-132)
                if 'right_hand' in person:
                    rh_kp = person['right_hand']['keypoints']
                    rh_sc = person['right_hand']['scores']
                    if isinstance(rh_kp, list):
                        rh_kp = np.array(rh_kp, dtype=np.float32)
                        rh_sc = np.array(rh_sc, dtype=np.float32)
                    keypoints_array[t, 112:133, :2] = rh_kp
                    keypoints_array[t, 112:133, 2] = rh_sc
            else:
                keypoints_array[t, :, :] = 0.0  # No detection

        return keypoints_array

    def fill_missing_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Fill missing keypoints (with confidence < threshold) using interpolation."""
        keypoints_filled = keypoints.copy()
        num_frames, num_keypoints, _ = keypoints.shape
        
        for k in range(num_keypoints):
            # Get confidence scores for this keypoint across all frames
            conf_series = keypoints[:, k, 2]
            mask = conf_series < self.conf_threshold  # True where confidence is low
            
            if np.sum(~mask) < 2:
                continue  # Not enough points to interpolate
            
            # Interpolate x and y coordinates
            for d in range(2):  # x and y
                coord_series = keypoints_filled[:, k, d]
                coord_series[mask] = np.interp(
                    np.where(mask)[0],
                    np.where(~mask)[0],
                    coord_series[~mask]
                )
        
        return keypoints_filled

    def butterworth_lpf(self, signal, fc, order=4):
        """Apply Butterworth low-pass filter to a 1D signal (robust for short inputs)."""
        x = np.asarray(signal, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("butterworth_lpf expects a 1D array")

        # If too short for filtfilt padding, return as-is (or you can fallback to lfilter)
        # filtfilt padlen default ~ 3*(max(len(a),len(b))-1)
        nyq = 0.5 * self.fs
        wn = fc / nyq
        wn = min(max(wn, 1e-6), 0.99)
        b, a = butter(order, wn, btype='low')

        padlen = 3 * (max(len(a), len(b)) - 1)
        if x.size <= padlen + 1:
            return x.copy()

        return filtfilt(b, a, x)

    def temporal_filter(self, KP_filled, fc, order=4):
        """Apply temporal Butterworth low-pass filter to keypoints."""
        KP_filled = KP_filled.astype(np.float64)        
        T, J, D = KP_filled.shape
        KP_filtered = np.empty_like(KP_filled)
        
        for j in range(J):
            for d in range(D):
                KP_filtered[:, j, d] = self.butterworth_lpf(KP_filled[:, j, d], fc, order)

        return KP_filtered.astype(np.float32)

    def calc_fc_residual(
        self,
        keypoints_raw,
        filter_func,
        fc_grid,
        score=None,
        conf_threshold=0.2,
        joints=None,
    ):
        """
        Residual analysis with:
        - full-length filtering (preserve time axis)
        - RMS computed only on valid frames
        - elbow (knee) via max distance to line between endpoints
        """
        keypoints_raw = np.asarray(keypoints_raw, dtype=np.float64)
        T, J, D = keypoints_raw.shape
        fcs = np.asarray(list(fc_grid), dtype=np.float64)

        if joints is None:
            joints = np.arange(J, dtype=int)
        else:
            joints = np.asarray(joints, dtype=int)

        if score is not None:
            score = np.asarray(score, dtype=np.float64)
            global_valid_mask = (score >= conf_threshold)
        else:
            global_valid_mask = np.ones((T, J), dtype=bool)

        rms_curves = np.full((J, D, len(fcs)), np.nan, dtype=np.float64)
        knee_fcs = np.full((J, D), np.nan, dtype=np.float64)

        def interp_1d_with_mask(x, valid_mask):
            """Interpolate missing samples but keep length T."""
            x = x.copy()
            idx = np.arange(x.size)
            good = valid_mask & np.isfinite(x)
            if good.sum() < 2:
                return None  # not enough points

            # Fill missing with linear interpolation; endpoints extend
            x[~good] = np.interp(idx[~good], idx[good], x[good])
            return x

        def elbow_max_distance(x, y):
            """
            Find elbow index by max perpendicular distance to line between endpoints.
            x, y must be finite and same length >= 3.
            """
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            # Normalize to [0,1] to avoid scale issues
            x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
            y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

            p1 = np.array([x_n[0], y_n[0]])
            p2 = np.array([x_n[-1], y_n[-1]])
            v = p2 - p1
            v_norm = np.linalg.norm(v) + 1e-12

            # distance from each point to the line p1->p2
            dists = []
            for i in range(len(x_n)):
                p = np.array([x_n[i], y_n[i]])
                # area of parallelogram / base length = perpendicular distance
                dist = np.abs(np.cross(v, p - p1)) / v_norm
                dists.append(dist)

            dists = np.asarray(dists)
            return int(np.argmax(dists))

        for j in joints:
            for d in range(D):
                series = keypoints_raw[:, j, d]
                valid_mask = global_valid_mask[:, j] & np.isfinite(series)

                # Require enough valid points
                if valid_mask.sum() < max(15, int(0.3 * T)):
                    continue

                # Interpolate missing but keep original timing
                series_filled = interp_1d_with_mask(series, valid_mask)
                if series_filled is None:
                    continue

                residuals = np.empty(len(fcs), dtype=np.float64)

                for i, fc in enumerate(fcs):
                    filtered = filter_func(series_filled, fc)
                    r = series_filled - filtered

                    # RMS only on originally-valid frames (pre-interp)
                    rv = r[valid_mask]
                    residuals[i] = np.sqrt(np.mean(rv * rv)) if rv.size else np.nan

                rms_curves[j, d, :] = residuals

                finite = np.isfinite(residuals)
                if finite.sum() >= 3:
                    idxs = np.where(finite)[0]
                    # apply elbow on finite subset
                    elbow_local = elbow_max_distance(fcs[idxs], residuals[idxs])
                    knee_fcs[j, d] = fcs[idxs[elbow_local]]

        recommended_fc = np.nanmedian(knee_fcs) if np.any(np.isfinite(knee_fcs)) else None
        return knee_fcs, fcs, rms_curves, recommended_fc

    def plot_residual_curves(
    self,
    fcs,
    rms_curves,
    save_path=None,
    title=None,
    logy=True,
    show_median=True,
    ):
        """
        Residual plotter focused on spatial dimensions (X,Y) and raw joint curves.

        Accepts:
        - (D,F)     → plots X,Y directly
        - (J,D,F)   → plots all joints (thin), optionally median (thick)

        Dimension convention:
        dim 0 → X
        dim 1 → Y
        dim 2 → ignored
        """
        fcs = np.asarray(fcs, dtype=float)
        R = np.asarray(rms_curves, dtype=float)

        plt.figure(figsize=(9, 5))

        # Colors for X and Y
        dim_cfg = {
            0: dict(color="tab:blue", label="X"),
            1: dict(color="tab:orange", label="Y"),
        }

        if R.ndim == 2:
            # (D,F)
            for d in (0, 1):
                if d >= R.shape[0]:
                    continue
                plt.plot(
                    fcs,
                    R[d],
                    linewidth=2.5,
                    color=dim_cfg[d]["color"],
                    label=f"{dim_cfg[d]['label']} residual",
                )

        elif R.ndim == 3:
            # (J,D,F)
            J, D, F = R.shape

            for d in (0, 1):
                if d >= D:
                    continue

                # plot raw joint curves
                for j in range(J):
                    if not np.all(np.isfinite(R[j, d])):
                        continue
                    plt.plot(
                        fcs,
                        R[j, d],
                        color=dim_cfg[d]["color"],
                        alpha=0.15,
                        linewidth=1.0,
                    )

                # optional median curve
                if show_median:
                    med = np.nanmedian(R[:, d, :], axis=0)
                    plt.plot(
                        fcs,
                        med,
                        color=dim_cfg[d]["color"],
                        linewidth=3.0,
                        label=f"{dim_cfg[d]['label']} median",
                    )

        else:
            raise ValueError(f"Unsupported rms_curves shape: {R.shape}")

        plt.xlabel("Cutoff Frequency (Hz)")
        plt.ylabel("RMS Residual (pixels)")
        plt.title(title if title else "Residual Analysis (raw joint curves)")
        plt.grid(True, alpha=0.3)

        if logy:
            plt.yscale("log")

        plt.legend(loc="best")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved residual plot to: {save_path}")
            plt.close()
        else:
            plt.show()



def load_keypoints_from_json(json_path: str) -> List[dict]:
    """Load keypoints from JSON file."""
    print(f"Loading keypoints from: {json_path}")
    with open(json_path, 'r') as f:
        frames_data = json.load(f)
    print(f"  Loaded {len(frames_data)} frames")
    return frames_data


def save_filtered_keypoints(keypoints_array: np.ndarray, frames_data: List[dict], 
                           output_path: str):
    """Save filtered keypoints to JSON."""
    output_data = []
    for i, frame_data in enumerate(frames_data):
        frame_copy = frame_data.copy()
        if i < len(keypoints_array) and len(frame_data['persons']) > 0:
            filtered_kp = keypoints_array[i]  # (K, 3)
            person = frame_data['persons'][0]
            
            # Check if it's the complex format (with body/feet/face/hands)
            if 'body' in person:
                # Save back to separated format
                frame_copy['persons'][0]['body']['keypoints'] = filtered_kp[0:17, :2].tolist()
                frame_copy['persons'][0]['body']['scores'] = filtered_kp[0:17, 2].tolist()
                
                if 'feet' in person:
                    frame_copy['persons'][0]['feet']['keypoints'] = filtered_kp[17:23, :2].tolist()
                    frame_copy['persons'][0]['feet']['scores'] = filtered_kp[17:23, 2].tolist()
                
                if 'face' in person:
                    frame_copy['persons'][0]['face']['keypoints'] = filtered_kp[23:91, :2].tolist()
                    frame_copy['persons'][0]['face']['scores'] = filtered_kp[23:91, 2].tolist()
                
                if 'left_hand' in person:
                    frame_copy['persons'][0]['left_hand']['keypoints'] = filtered_kp[91:112, :2].tolist()
                    frame_copy['persons'][0]['left_hand']['scores'] = filtered_kp[91:112, 2].tolist()
                
                if 'right_hand' in person:
                    frame_copy['persons'][0]['right_hand']['keypoints'] = filtered_kp[112:133, :2].tolist()
                    frame_copy['persons'][0]['right_hand']['scores'] = filtered_kp[112:133, 2].tolist()
            else:
                # Simple format
                frame_copy['persons'][0]['keypoints'] = filtered_kp[:, :2].tolist()
                frame_copy['persons'][0]['scores'] = filtered_kp[:, 2].tolist()
        
        output_data.append(frame_copy)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved filtered keypoints to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Post-process keypoints with temporal filtering and residual analysis'
    )
    parser.add_argument('--json-path', type=str, required=True,
                       help='Path to input JSON file with keypoints')
    parser.add_argument('--output-dir', type=str, default='./filtered_output',
                       help='Output directory for filtered results')
    parser.add_argument('--cutoff-freq', type=float, default=3.0,
                       help='Cutoff frequency for filtering (Hz)')
    parser.add_argument('--fs', type=int, default=60,
                       help='Sampling frequency (Hz)')
    parser.add_argument('--conf-threshold', type=float, default=0.2,
                       help='Confidence threshold for keypoints')
    parser.add_argument('--analyze-fc', action='store_true',
                       help='Perform residual analysis to find optimal cutoff frequency')
    parser.add_argument('--save-filtered', '--save-filter', action='store_true',
                       help='Save filtered keypoints to JSON')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Keypoint Temporal Filtering & Residual Analysis")
    print("="*70 + "\n")
    
    # Load keypoints
    frames_data = load_keypoints_from_json(args.json_path)
    
    # Initialize post-processor
    post_processor = KeypointPostProcessor(fs=args.fs, conf_threshold=args.conf_threshold)
    
    # Convert to numpy array
    print("\nConverting frames to numpy array...")
    keypoints_array = post_processor.keypoints_to_array(frames_data)
    print(f"  Keypoints shape: {keypoints_array.shape} (frames, keypoints, coordinates)")
    np.set_printoptions(suppress=True, precision=2)
    print(f"  Sample frame 0 keypoints:\n{keypoints_array[0]}")
    
    # Fill missing keypoints
    print("\nFilling missing keypoints...")
    keypoints_filled = post_processor.fill_missing_keypoints(keypoints_array)
    print(f"  Filled missing keypoints (confidence < {args.conf_threshold})")
    
    # Perform residual analysis if requested
    if args.analyze_fc:
        print("\nPerforming residual analysis to find optimal cutoff frequency...")
        fc_grid = np.arange(1.0, 20.5, 0.5)
        foot_idx = KeypointPostProcessor.FOOT_INDICES
        body_idx = KeypointPostProcessor.BODY_INDICES
        main_joints = [*body_idx, *foot_idx]
        
        knee_fcs, fcs, rms_curves, recommended_fc = post_processor.calc_fc_residual(
            keypoints_array, 
            filter_func=post_processor.butterworth_lpf,
            fc_grid=fc_grid, 
            score=keypoints_array[:, :, 2], 
            conf_threshold=args.conf_threshold, 
            joints=main_joints
        )
        
        if recommended_fc is not None:
            print(f"  Recommended cutoff frequency: {recommended_fc:.2f} Hz")
            args.cutoff_freq = recommended_fc
        else:
            print(f"  Could not determine recommended cutoff, using default: {args.cutoff_freq} Hz")
        
        # Plot residual curves
        video_name = os.path.splitext(os.path.basename(args.json_path))[0]
        plot_path = os.path.join(args.output_dir, f'{video_name} residual_curves.png')
        print(f"\nPlotting residual curves...")
        post_processor.plot_residual_curves(
            fcs,
            rms_curves[main_joints, :, :],   # keep (J,D,F)
            save_path=plot_path,
            title="Residual curves (body+feet)",
            logy=True
        )
    
    # Apply temporal filtering
    print(f"\nApplying temporal Butterworth low-pass filter (fc={args.cutoff_freq} Hz)...")
    keypoints_filtered = post_processor.temporal_filter(keypoints_filled, fc=args.cutoff_freq, order=4)
    print(f"  Filtering complete")
    
    # Save filtered keypoints if requested
    if args.save_filtered:
        output_json = os.path.join(args.output_dir, 'filtered_keypoints.json')
        save_filtered_keypoints(keypoints_filtered, frames_data, output_json)
    
    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70 + "\n")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
