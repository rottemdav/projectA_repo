import os
import sys
import cv2
import json
import time
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime
from scipy.signal import filtfilt, butter

class Config:
    """Configuration for OpenPose estimation and post-processing."""
    
    # ============ OpenPose Paths ============
    OPENPOSE_ROOT = "/home/projects/sipl-prj10496/project_files/openpose"
    OPENPOSE_PYTHON_PATH = os.path.join(OPENPOSE_ROOT, "build/python/openpose")
    MODEL_FOLDER = os.path.join(OPENPOSE_ROOT, "models/")
    
    # ============ OpenPose Engine Options ============
    # Can add other pyopenpose params here
    NUMBER_PEOPLE_MAX = 3
    RENDER_POSE = 1
    
    # ============ Tracking & Filtering Parameters ============
    USE_CAMERA_POSITION_FILTER = True
    
    # Camera spatial limits
    CAM1_Y_MAX = 930
    CAM2_X_MIN = 1860
    CAM2_X_MAX = 2380
    CAM3_Y_MAX = 1200
    
    # Tracking logic
    MIN_HIP_CONF = 0.4                  # Minimum MidHip confidence to start an anchor
    CONF_THRESHOLD = 0.3                # Threshold to keep general keypoints
    HIP_DIST_THRESHOLD = 350            # Max allowed distance jump for the hip anchor
    RESET_LAST_KNOWN_AFTER_MISSES = 5   # Number of missed frames before resetting tracker
    
    # ============ Visualization Parameters ============
    DRAW_PREFILTER_PEOPLE = False
    DRAW_POSTFILTER_SELECTED = True
    DRAW_CAM2_X_LIMITS = True
    DRAW_SKELETON = False               # Decide whether to draw the skeleton or not
    
    # ============ Video Processing Parameters ============
    START_FRAME = None    # Frame to start processing from (0 = beginning)
    MAX_FRAMES = None     # Maximum frames to process (None = all frames)
    END_FRAME = None      # Frame to end processing (inclusive, None = till end)

    # ============ I/O Configuration ============
    DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # filename format and directory configuration 
    INPUT_PATH = None
    OUTPUT_DIR = f"/home/projects/sipl-prj10496/project_files/data/openpose_output/{DATE}"

    JSON_FILENAME_FORMAT = f"{{video_name}}_keypoints_{{DATE}}_{{out_range}}.json"
    VIDEO_FILENAME_FORMAT = f"{{video_name}}_pose_{{DATE}}_{{out_range}}.mp4"
    FILTERED_JSON_FILENAME_FORMAT = f"{{video_name}}_keypoints_filtered_{{DATE}}_{{out_range}}.json"
    FILTERED_VIDEO_FILENAME_FORMAT = f"{{video_name}}_pose_filtered_{{DATE}}_{{out_range}}.mp4"
    RESIDUAL_PLOT_FORMAT = f"{{video_name}}_residuals_{{DATE}}_{{out_range}}.png"


class OpenPoseProcessor:
    """
    OpenPose processor class for human pose estimation.
    Mirrors HRNet's WholeBodyPoseProcessor architecture.
    """
    # OpenPose BODY_25 Keypoint structure equivalent
    BODY_INDICES = list(range(0, 15))     
    FACE_INDICES = [0, 15, 16, 17, 18]    
    FOOT_INDICES = [11, 14, 19, 20, 21, 22, 23, 24]
    LEFT_HAND_INDICES = [] # BODY_25 doesn't contain hands natively
    RIGHT_HAND_INDICES = [] 

    def __init__(self):
        # Try importing OpenPose dynamically based on Config path
        try:
            sys.path.insert(0, Config.OPENPOSE_PYTHON_PATH)
            import pyopenpose as op
            self.op = op
        except ImportError as e:
            print(f"Error: Could not find OpenPose library at {Config.OPENPOSE_PYTHON_PATH}")
            raise e

        # Configure OpenPose
        params = dict()
        params["model_folder"] = Config.MODEL_FOLDER
        params["number_people_max"] = Config.NUMBER_PEOPLE_MAX
        params["render_pose"] = Config.RENDER_POSE
        
        try:
            self.opWrapper = self.op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()
        except Exception as e:
            print(f"Error starting OpenPose: {e}")
            sys.exit(1)

    def estimate_pose(self, image: np.ndarray) -> Tuple[list, any]:
        """
        Estimate poses for a single image layout similar to HRNet.
        Returns raw pose keypoints array.
        """
        datum = self.op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))
        
        # poseKeypoints is of shape (N, 25, 3)
        pose_results = datum.poseKeypoints
        return pose_results, datum

    def extract_keypoints(self, pose_results) -> List[dict]:
        """
        Extract keypoints from pose results into a structured dict.
        Mirrors the HRNet output format dicts exactly.
        """
        keypoints_list = []
        if pose_results is None or len(pose_results.shape) < 3:
            return keypoints_list

        for i in range(pose_results.shape[0]):
            kp_full = pose_results[i] # (25, 3)
            keypoints = kp_full[:, :2] # (25, 2)
            scores = kp_full[:, 2]     # (25,)
            
            # Openpose doesn't give a traditional bbox natively without extra steps,
            # but we can infer it broadly, or set to None as HRNet accepts.
            min_x, min_y = np.min(keypoints[scores > 0], axis=0) if np.sum(scores > 0) else (0, 0)
            max_x, max_y = np.max(keypoints[scores > 0], axis=0) if np.sum(scores > 0) else (0, 0)
            bbox = np.array([min_x, min_y, max_x, max_y, 1.0], dtype=np.float32)
            
            keypoints_list.append({
                'keypoints': keypoints,
                'scores': scores,
                'bbox': bbox,
                'body': {'keypoints': keypoints[self.BODY_INDICES], 'scores': scores[self.BODY_INDICES]},
                'feet': {'keypoints': keypoints[self.FOOT_INDICES], 'scores': scores[self.FOOT_INDICES]},
                'face': {'keypoints': keypoints[self.FACE_INDICES], 'scores': scores[self.FACE_INDICES]},
                'left_hand': {'keypoints': [], 'scores': []},
                'right_hand': {'keypoints': [], 'scores': []},
            })
            
        return keypoints_list

    def visualize(self, image: np.ndarray, datum, show: bool = False, draw_face: bool = True) -> np.ndarray:
        """
        Visualize pose estimation results on an image.
        Uses OpenPose native output for now, mirroring the HRNet method's existence.
        """
        if datum is not None and hasattr(datum, 'cvOutputData') and Config.RENDER_POSE:
            vis_image = datum.cvOutputData
        else:
            vis_image = image.copy()
            
        if show:
            cv2.imshow("OpenPose Visualization", vis_image)
            cv2.waitKey(1)
            
        return vis_image

    def process_image(self, image_path: str, output_path: Optional[str] = None, show: bool = False) -> Tuple[list, np.ndarray]:
        """
        Process a single image file.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        pose_results, datum = self.estimate_pose(image)
        vis_image = self.visualize(image, datum, show=show)
        
        if output_path is not None:
            cv2.imwrite(output_path, vis_image)
            
        return pose_results, vis_image

    def process_video(self, video_path: str, output_path: Optional[str] = None, json_output_path: Optional[str] = None,
                      start_frame=0, max_frames=None, end_frame=None, draw_face=True, show=False) -> Tuple[List[Tuple[int, list]], List[dict]]:
        """
        Process a video file, extracting keypoints into JSON dictionaries exactly like HRNet.
        Returns:
            all_results: List of (frame_idx, pose_results)
            all_frames: List of dictionary representations matching HRNet JSON structure
        """
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video stream or file: {video_path}")
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Handle frame ranges
        start_f = start_frame if start_frame is not None else 0
        eff_end_frame = total_frames - 1
        if end_frame is not None:
            eff_end_frame = min(end_frame, total_frames - 1)
        elif max_frames is not None:
            eff_end_frame = min(start_f + max_frames - 1, total_frames - 1)
            
        print(f"Video info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}, processing frames {start_f} to {eff_end_frame}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        # Ensure output directories exist
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        if json_output_path:
            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        
        all_results = []
        all_frames = []
        json_results = []

        frame_idx = start_f
        frames_processed = 0
        start_time = time.time()

        while cap.isOpened() and frame_idx <= eff_end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Estimate pose
            pose_results, datum = self.estimate_pose(frame)
            
            # Visualize
            vis_frame = self.visualize(frame, datum, show=show, draw_face=draw_face)
            all_results.append((frame_idx, pose_results))

            if writer is not None:
                writer.write(vis_frame)

            # Replicate HRNet's dictionary layout
            frame_json = {
                'frame_index': frame_idx,
                'persons': []
            }
            full_frame = {
                'frame_index': frame_idx,
                'timestamp': frame_idx / fps if fps > 0 else 0,
                'persons': []
            }

            keypoints_list = self.extract_keypoints(pose_results)
            for person in keypoints_list:
                frame_json['persons'].append({
                    'bbox': person['bbox'].tolist() if person['bbox'] is not None else None,
                    'body': {
                        'keypoints': person['body']['keypoints'].tolist(),
                        'scores': person['body']['scores'].tolist()
                    },
                    'feet': {
                        'keypoints': person['feet']['keypoints'].tolist(),
                        'scores': person['feet']['scores'].tolist()
                    }
                })
                full_frame['persons'].append({
                    'bbox': person['bbox'].tolist() if person['bbox'] is not None else None,
                    'keypoints': person['keypoints'].tolist(),
                    'scores': person['scores'].tolist()
                })
            
            json_results.append(frame_json)
            all_frames.append(full_frame)

            frame_idx += 1
            frames_processed += 1
            
            if frames_processed % 50 == 0:
                elapsed = time.time() - start_time
                fps_proc = frames_processed / elapsed if elapsed > 0 else 0
                print(f"Processed frame {frame_idx} ({frames_processed} done)... ({fps_proc:.1f} fps)")

        cap.release()
        if writer is not None:
            writer.release()

        # Save all frames to JSON
        if json_output_path:
            with open(json_output_path, 'w') as f:
                json.dump(json_results, f, indent=4)
            print(f"Saved unfiltered keypoints JSON to {json_output_path}")

        return all_results, all_frames

    def get_body_keypoints(self, keypoints_data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract only body keypoints."""
        return keypoints_data['body']['keypoints'], keypoints_data['body']['scores']

    def keypoints_to_array(self, all_frames: List[dict]) -> np.ndarray:
        """
        Convert list of frames with keypoints to a numpy array.
        (N_frames, N_keypoints, 3) just like HRNet. 
        We take the first person found (or empty if none).
        """
        num_frames = len(all_frames)
        if num_frames == 0:
            return np.empty((0, 0, 3), dtype=np.float32)
            
        # Figure out num_keypoints from the first valid frame
        num_keypoints = 25 # Default openpose BODY_25
        for frame in all_frames:
            if len(frame['persons']) > 0:
                first_kp = frame['persons'][0]['keypoints']
                if isinstance(first_kp, list):
                    first_kp = np.array(first_kp, dtype=np.float32)
                num_keypoints = first_kp.shape[0]
                break

        keypoints_array = np.zeros((num_frames, num_keypoints, 3), dtype=np.float32)

        for t, frame in enumerate(all_frames):
            if len(frame['persons']) > 0:
                kp = np.array(frame['persons'][0]['keypoints'], dtype=np.float32)
                sc = np.array(frame['persons'][0]['scores'], dtype=np.float32)
                # Combine coordinates and score
                keypoints_array[t, :, :2] = kp
                keypoints_array[t, :, 2] = sc
                
        return keypoints_array

    def write_and_visualize_filtered_video(
        self,
        all_frames,
        filtered_keypoints,
        video_path: str,
        output_path: str,
        start_frame: int,
        end_frame: int,
        draw_face: bool = True,
        show: bool = False
    ):
        """
        Method for the second pass to render the video after applying filtering logic.
        Allows iterating over video again and writing the filtered array.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        
        frame_to_kp = {}
        for f, kp in zip(all_frames, filtered_keypoints):
            frame_to_kp[f['frame_index']] = kp

        frame_idx = start_frame
        frames_written = 0
        print(f"[Visualization] Starting to write filtered video from frame {start_frame} to {end_frame}...")

        while cap.isOpened() and frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_to_kp:
                kp_arr = np.asarray(frame_to_kp[frame_idx], dtype=np.float32)
                # We will properly map this to OpenPose's drawing function in the future.
                # For now, manually draw points to visualize.
                for point in kp_arr:
                    x, y, score = point[0], point[1], point[2]
                    if score > Config.CONF_THRESHOLD and not np.isnan(x) and not np.isnan(y):
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                        
                writer.write(frame)
            else:
                writer.write(frame)

            frames_written += 1
            frame_idx += 1

        cap.release()
        writer.release()
        print(f"Saved filtered output video to: {output_path}")

class KeypointPostProcessor:
    """Class for post-processing keypoints, e.g., temporal filtering."""
    
    def __init__(self, fs, conf_threshold=0.2):
        self.fs = fs
        self.conf_threshold = conf_threshold

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

    def butterworth_lpf (self, KP_filled, fc, order=4):
        """Apply Butterworth low-pass filter to keypoint time series."""
        nyq = 0.5 * self.fs
        cutoff_freq = fc / nyq
        b, a = butter(order, cutoff_freq, btype='low')
        return filtfilt(b, a, KP_filled)
    
    def temporal_filter(self, KP_filled, fc, order=4):
        """Apply temporal Butterworth low-pass filter to keypoints."""
        KP_filled = KP_filled.astype(np.float64)        
        T, J, D = KP_filled.shape
        KP_filtered = np.empty_like(KP_filled)
        
        for j in range(J):
            for d in range(D):
                KP_filtered[:, j, d] = self.butterworth_lpf(KP_filled[:, j, d], fc, order)

        return KP_filtered.astype(np.float32)

    def calc_fc_residual(self, keypoints_raw, filter_func, fc_grid, score, conf_threshold=0.2, joints=None):
        """
        keypoint_raw: (T,J,D)
        score: (T,J) - confidence scores
        joints: list of joint indices to consider, if None use all

        return: knee_fcs - cutoff per joint, fcs - cutoff for all joints, rms_curves, recommended_fc 
        """

        keypoints_raw = np.asarray(keypoints_raw, dtype=float)
        T,J,D = keypoints_raw.shape
        fcs = np.array(list(fc_grid), dtype=float)
        
        if joints is None:
            joints = list(range(J))
        else:
            joints = np.array(joints, dtype=int)

        rms_curves = np.full((J,D,len(fcs)), np.nan, dtype=float)
        knee_fcs = np.full((J,D), np.nan, dtype=float)

        if score is not None:
            score = np.asarray(score, dtype=float)
            global_valid_mask = (score >= conf_threshold)

        else:
            global_valid_mask = np.ones((T,J), dtype=bool)

        for j in joints:
            for d in range(D):
                keypoint_series = keypoints_raw[:, j, d]
                valid_mask = global_valid_mask[:, j] & np.isfinite(keypoint_series)
                

                if valid_mask.sum() < max(10, int(0.2 * T)):
                    continue  # Not enough valid data

                # Extract valid data points
                valid_data = keypoint_series[valid_mask]
                
                residuals = []
                for fc in fcs:
                    filtered_series = filter_func(valid_data, fc)
                    residual = valid_data - filtered_series
                    rms = np.sqrt(np.mean(residual**2))
                    residuals.append(rms)

                residuals = np.array(residuals, dtype=float)
                rms_curves[j, d, :] = residuals

                # Find knee point in residual curve using maximum curvature
                # The knee is where the curve changes most dramatically
                if len(residuals) > 2:
                    # Calculate second derivative to find maximum curvature
                    diffs = np.diff(residuals)
                    second_diffs = np.diff(diffs)
                    
                    # The knee is where curvature is maximum (most negative second derivative)
                    if len(second_diffs) > 0:
                        knee_idx = np.argmin(second_diffs) + 1  # +1 to align with fcs
                        knee_fcs[j, d] = fcs[knee_idx]
                    
        recommended_fc = np.nanmedian(knee_fcs) if np.any(~np.isnan(knee_fcs)) else None
        return knee_fcs, fcs, rms_curves, recommended_fc
    
    def plot_residual_curves(self, fcs, rms_curves, save_path=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        rms_curves = np.asarray(rms_curves)
        
        # Handle both 1D and 2D arrays
        if rms_curves.ndim == 1:
            plt.plot(fcs, rms_curves, color='gray')
        else:
            # Plot each curve separately
            for i in range(rms_curves.shape[0]):
                plt.plot(fcs, rms_curves[i, :], color='gray', alpha=0.5)
        
        plt.xlabel('Cutoff Frequency (Hz)')
        plt.ylabel('RMS Residual')
        plt.title('RMS Residual vs Cutoff Frequency')
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')    
            plt.close()
        else:
            plt.show()
