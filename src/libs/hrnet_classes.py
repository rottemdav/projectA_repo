
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List, Tuple, Union
from scipy.signal import filtfilt, butter
from tqdm import tqdm
from src.config import Config

# MMPose imports
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample

# MMDetection imports
try:
    from mmdet.apis import inference_detector, init_detector
    HAS_MMDET = True
except (ImportError, ModuleNotFoundError):
    HAS_MMDET = False
    print("Warning: mmdet not found. Person detection will not be available.")

class WholeBodyPoseProcessor:
    """
    WholeBody pose processor class for human pose estimation using MMPose.
    
    COCO-WholeBody outputs 133 keypoints:
    - Body: 17 keypoints (indices 0-16)
    - Feet: 6 keypoints (indices 17-22)  
    - Face: 68 keypoints (indices 23-90)
    - Left hand: 21 keypoints (indices 91-111)
    - Right hand: 21 keypoints (indices 112-132)
    
    This class provides methods for:
    - Initializing WholeBody and detection models
    - Processing single images
    - Processing videos frame by frame
    - Visualizing pose estimation results
    - Extracting specific body parts (face, hands, body)
    """  
    # Keypoint indices for each body part
    BODY_INDICES = list(range(0, 17))      # 17 body keypoints
    FOOT_INDICES = list(range(17, 23))     # 6 foot keypoints
    FACE_INDICES = list(range(23, 91))     # 68 face keypoints
    LEFT_HAND_INDICES = list(range(91, 112))   # 21 left hand keypoints
    RIGHT_HAND_INDICES = list(range(112, 133)) # 21 right hand keypoints
    
    def __init__(
        self, 
        pose_config: str = Config.hrnet.POSE_CONFIG,
        pose_checkpoint: str = Config.hrnet.POSE_CHECKPOINT,
        det_config: str = Config.hrnet.DET_CONFIG,
        det_checkpoint: str = Config.hrnet.DET_CHECKPOINT,
        device: str = Config.DEVICE,
        bbox_thr: float = Config.hrnet.BBOX_THR,
        nms_thr: float = Config.hrnet.NMS_THR,
        vis_kpt_thr: float = Config.hrnet.KPT_THR,
    ):
        """
        Initialize WholeBody pose processor.
        
        Args:
            pose_config: Path to pose estimation config file
            pose_checkpoint: Path or URL to pose estimation checkpoint
            det_config: Path to detection config file  
            det_checkpoint: Path or URL to detection checkpoint
            device: Device to run inference on
            bbox_thr: Bounding box score threshold
            nms_thr: NMS IoU threshold
            kpt_thr: Keypoint confidence threshold
        """
        self.device = device
        self.bbox_thr = bbox_thr
        self.nms_thr = nms_thr
        self.vis_kpt_thr = vis_kpt_thr
        self.det_cat_id = Config.hrnet.DET_CAT_ID
        
        # Change to mmpose root directory for config resolution
        original_dir = os.getcwd()
        os.chdir(Config.hrnet.MMPOSE_ROOT)
        
        try:
            # Initialize pose model first
            print("Initializing WholeBody pose estimation model...")
            self.pose_estimator = init_pose_estimator(
                pose_config, 
                pose_checkpoint, 
                device=device
            )
            
            # Initialize visualizer BEFORE loading mmdet to avoid registry conflicts
            visualizer_cfg = self.pose_estimator.cfg.visualizer.copy()
            visualizer_cfg['radius'] = Config.hrnet.RADIUS
            visualizer_cfg['line_width'] = Config.hrnet.THICKNESS
            self.visualizer = VISUALIZERS.build(visualizer_cfg)
            self.visualizer.set_dataset_meta(
                self.pose_estimator.dataset_meta
            )
            
            if HAS_MMDET:
                print("Initializing detection model...")
                self.detector = init_detector(
                    det_config, 
                    det_checkpoint, 
                    device=device
                )
                self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
            else:
                self.detector = None
                print("Warning: Detection model not available. "
                      "You'll need to provide bounding boxes manually.")
            
            print("Models initialized successfully!")
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def detect_persons(self, image: np.ndarray) -> np.ndarray:
        """
        Detect persons in an image using MMDetection.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            bboxes: Array of bounding boxes [x1, y1, x2, y2]
        """
        if self.detector is None:
            raise RuntimeError("Detection model not available")
        
        det_result = inference_detector(self.detector, image)
        pred_instance = det_result.pred_instances.cpu().numpy()
        
        # Filter by category (person) and score threshold
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
        )
        bboxes = bboxes[
            np.logical_and(
                pred_instance.labels == self.det_cat_id,
                pred_instance.scores > self.bbox_thr
            )
        ]
        
        # Apply NMS
        bboxes = bboxes[nms(bboxes, self.nms_thr), :4]
        
        return bboxes
    
    def estimate_pose(
        self, 
        image: np.ndarray, 
        bboxes: Optional[np.ndarray] = None
    ) -> Tuple[list, any]:
        """
        Estimate poses for detected persons.
        
        Args:
            image: Input image (BGR format)
            bboxes: Optional pre-computed bounding boxes. 
                    If None, will detect persons first.
                    
        Returns:
            pose_results: List of pose estimation results
            data_samples: Merged data samples for visualization
        """
        # Detect persons if bboxes not provided
        if bboxes is None:
            bboxes = self.detect_persons(image)
        
        if len(bboxes) == 0:
            print("No persons detected in the image")
            return [], None
        
        # Run pose estimation
        pose_results = inference_topdown(self.pose_estimator, image, bboxes)
        data_samples = merge_data_samples(pose_results)
        
        return pose_results, data_samples
    
    def visualize(
        self, 
        image: np.ndarray, 
        data_samples,
        show: bool = False,
        draw_bbox: bool = True,
        draw_heatmap: bool = False,
        skeleton_style: str = 'mmpose'
    ) -> np.ndarray:
        """
        Visualize pose estimation results on an image.
        
        Args:
            image: Input image (BGR format)
            data_samples: Pose estimation data samples
            show: Whether to display the image
            draw_bbox: Whether to draw bounding boxes
            draw_heatmap: Whether to draw heatmaps
            skeleton_style: 'mmpose' or 'openpose'
            
        Returns:
            vis_image: Visualized image
        """
        # Convert BGR to RGB for visualization
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            self.visualizer.add_datasample(
                'result',
                img_rgb,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=draw_heatmap,
                draw_bbox=draw_bbox,
                show_kpt_idx=False,
                skeleton_style=skeleton_style,
                show=show,
                wait_time=0,
                kpt_thr=self.vis_kpt_thr
            )
        except Exception as e:
            print(f"[WARNING] Error during visualization: {e}")
            return image  # Return original image on error
        
        # Get the visualization result
        vis_image = self.visualizer.get_image()
        
        if vis_image is None:
            print("[WARNING] Visualizer returned None, using original image")
            return image
        
        # Convert back to BGR for OpenCV
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        return vis_image
    
    def process_image(
        self, 
        image_path: str, 
        output_path: Optional[str] = None,
        show: bool = False
    ) -> Tuple[list, np.ndarray]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization
            show: Whether to display the result
            
        Returns:
            pose_results: List of pose estimation results
            vis_image: Visualized image
        """
        print(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Estimate pose
        pose_results, data_samples = self.estimate_pose(image)
        
        if data_samples is None:
            print("No poses detected")
            return [], image
        
        # Visualize
        vis_image = self.visualize(image, data_samples, show=show)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            print(f"Saved visualization to: {output_path}")
        
        return pose_results, vis_image
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        frame_indices: Optional[List[int]] = None,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        end_frame: Optional[int] = None,
        draw_face: bool = True,
        show: bool = False,
        json_output_path: Optional[str] = None
    ) -> List[Tuple[int, list]]:
        """
        Process a video file.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            frame_indices: Optional list of specific frame indices to process
            start_frame: Frame index to start processing from (default 0)
            max_frames: Maximum number of frames to process (None = all)
            draw_face: Whether to draw face keypoints (default True)
            show: Whether to display frames
            
        Returns:
            all_results: List of (frame_idx, pose_results) tuples
        """
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Seek to start frame if specified
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"Starting from frame {start_frame}")

        # Get video properties after seeking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Compute effective end frame
        if end_frame is not None:
            eff_end_frame = min(end_frame, total_frames - 1)
        elif max_frames is not None:
            eff_end_frame = min(start_frame + max_frames - 1, total_frames - 1)
        else:
            eff_end_frame = total_frames - 1

        print(f"Video info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}, processing frames {start_frame} to {eff_end_frame}")

        # Setup output video writer
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # If not drawing face, set face keypoint scores to 0 to hide them
        face_indices = self.FACE_INDICES if not draw_face else []
        
        all_results = []
        all_frames = []
        json_results = []
#        frame_idx = start_frame
        processed_count = 0
        
        for frame_idx in tqdm(range(start_frame, eff_end_frame + 1), total=eff_end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            # Check if we should process this frame
            should_process = True
            if frame_indices is not None:
                should_process = frame_idx in frame_indices
            # max_frames is now handled by eff_end_frame

            if should_process:
                # Estimate pose
                pose_results, data_samples = self.estimate_pose(frame)

                if data_samples is not None:
                    # Hide face keypoints if draw_face is False
                    if face_indices and hasattr(data_samples, 'pred_instances'):
                        scores = data_samples.pred_instances.keypoint_scores
                        scores[:, face_indices] = 0  # Set face scores to 0 to hide them

                    # Visualize
                    vis_frame = self.visualize(frame, data_samples, show=show)
                    all_results.append((frame_idx, pose_results))
                else:
                    vis_frame = frame

                # Write to output video
                if writer:
                    writer.write(vis_frame)

                # Save JSON keypoints for this frame
                frame_json = {
                    'frame_index': frame_idx,
                    'persons': []
                }

                full_frame = {
                    'frame_index': frame_idx,
                    'timestamp': frame_idx  / fps,
                    'persons': []
                }
                if pose_results:
                    keypoints_list = self.extract_keypoints(pose_results)
                    for person in keypoints_list:
                        frame_json['persons'].append({
                            'bbox': person['bbox'].tolist() if person['bbox'] is not None else None,
                            'body': {'keypoints': person['body']['keypoints'].tolist(), 
                                     'scores': person['body']['scores'].tolist()},
                            'feet': { 'keypoints': person['feet']['keypoints'].tolist(),
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

                processed_count += 1

#                if processed_count % 100 == 0:
#                timestamp = datetime.now().strftime('%H:%M:%S')
#                    print(f"[{timestamp}] Processed frame {frame_idx} ({processed_count} processed, start={start_frame})/{total_frames}")

#           frame_idx += 1
        
        cap.release()
        if writer:
            writer.release()
            print(f"Saved output video to: {output_path}")

        # Save JSON output if requested
        if json_output_path:
            import json
            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
            with open(json_output_path, 'w') as f:
                json.dump(json_results, f)
            print(f"Saved keypoints JSON to: {json_output_path}")

        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Finished processing {processed_count} frames (frames {start_frame} to {eff_end_frame})")
        return all_results, all_frames
    
    def extract_keypoints(self, pose_results: list) -> List[dict]:
        """
        Extract keypoints from pose results in a clean format.
        Args: pose_results: List of pose estimation results
        Returns: keypoints_list: List of dicts with keypoints, scores, and body parts
        """
        
        keypoints_list = []
        
        for result in pose_results:
            pred_instances = result.pred_instances
            
            keypoints = pred_instances.keypoints  # Shape: (N, K, 2)
            scores = pred_instances.keypoint_scores  # Shape: (N, K)
            
            for i in range(len(keypoints)):
                kp = keypoints[i].cpu().numpy() if hasattr(keypoints[i], 'cpu') else keypoints[i]
                sc = scores[i].cpu().numpy() if hasattr(scores[i], 'cpu') else scores[i]
                
                keypoints_list.append({
                    'keypoints': kp,  # All 133 keypoints
                    'scores': sc,
                    'bbox': pred_instances.bboxes[i] if hasattr(pred_instances, 'bboxes') else None,
                    # Extract specific body parts for convenience
                    'body': {'keypoints': kp[self.BODY_INDICES], 'scores': sc[self.BODY_INDICES]},
                    'feet': {'keypoints': kp[self.FOOT_INDICES], 'scores': sc[self.FOOT_INDICES]},
                    'face': {'keypoints': kp[self.FACE_INDICES], 'scores': sc[self.FACE_INDICES]},
                    'left_hand': {'keypoints': kp[self.LEFT_HAND_INDICES], 'scores': sc[self.LEFT_HAND_INDICES]},
                    'right_hand': {'keypoints': kp[self.RIGHT_HAND_INDICES], 'scores': sc[self.RIGHT_HAND_INDICES]},
                })
        
        return keypoints_list
    
    def get_body_keypoints(self, keypoints_data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract only body keypoints (17 keypoints like standard COCO)."""
        return keypoints_data['body']['keypoints'], keypoints_data['body']['scores']
        
    def write_and_visualize_filtered_video( self,
                                            all_frames,
                                            filtered_keypoints,
                                            video_path: str,
                                            output_path: str,
                                            start_frame: int,
                                            end_frame: int,
                                            draw_face: bool = True,
                                            show: bool = False
                                            ):
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
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter failed to open: {output_path}")

        # Mapping: frame_index -> filtered keypoints
        # filtered_keypoints is a numpy array (T, K, 3), all_frames is list of dicts with frame_index
        frame_to_kp = {}
        for f, kp in zip(all_frames, filtered_keypoints):
            frame_to_kp[f['frame_index']] = kp

        face_indices = self.FACE_INDICES if not draw_face else []

        frame_idx = start_frame
        frames_written = 0
        frames_visualized = 0
        print(f"[Visualization] Starting to write filtered video from frame {start_frame} to {end_frame}...")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None:
            # Prefer the last keypoint frame if available, else use video length
            if frame_to_kp:
                end_frame = min(max(frame_to_kp.keys()), total_frames - 1)
            else:
                end_frame = total_frames - 1
        else:
            end_frame = min(end_frame, total_frames - 1)

        if end_frame < start_frame:
            raise ValueError(f"end_frame {end_frame} < start_frame {start_frame}")
        
        for frame_idx in tqdm(range(start_frame, end_frame + 1), total=end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_to_kp:
                kp_arr = np.asarray(frame_to_kp[frame_idx], dtype=np.float32)
                # kp_arr shape is (K, 3) for single person
                if kp_arr.ndim == 2:
                    kp_arr = kp_arr[None, ...]  # (1, K, 3)

                # If you don't have filtered scores, make dummy scores = 1
                scores = np.ones((kp_arr.shape[0], kp_arr.shape[1]), dtype=np.float32)

                # Hide face if requested
                if face_indices:
                    scores[:, face_indices] = 0.0

                # Build a PoseDataSample with pred_instances
                sample = PoseDataSample()
                sample.pred_instances = InstanceData(
                    keypoints=kp_arr,
                    keypoint_scores=scores
                )
                # attach dataset meta so the visualizer knows skeleton definition etc.
                sample.set_metainfo(self.pose_estimator.dataset_meta)

                # merge_data_samples expects list of PoseDataSample
                data_samples = merge_data_samples([sample])

                vis_frame = self.visualize(frame, data_samples, show=show)
                writer.write(vis_frame)
                frames_visualized += 1
            else:
                # No keypoints for this frame, write original frame
                writer.write(frame)

            frames_written += 1
            
            # Print progress every 50 frames
#            if frames_written % 50 == 0:
#                print(f"  [Progress] Written {frames_written} frames (visualized: {frames_visualized}, frame_idx: {frame_idx})")

#            frame_idx += 1

        cap.release()
        writer.release()
        print(f"Saved filtered output video to: {output_path}")
        print(f"  [Summary] Total frames: {frames_written}, Visualized: {frames_visualized}, Original: {frames_written - frames_visualized}")

    def keypoints_to_array(self, all_frames: List[dict]) -> np.ndarray:
        """Convert list of frames with keypoints to a numpy array."""
        num_frames = len(all_frames)
        if num_frames == 0:
            return np.empty((0, 0, 3), dtype=np.float32)

        first_kp = all_frames[0]['persons'][0]['keypoints']
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
    
class KeypointPostProcessor:
    """Class for post-processing keypoints, e.g., temporal filtering."""
    
    def __init__(self, fs, conf_threshold=0.2):
        self.fs = fs
        self.conf_threshold = conf_threshold

    # fixme 3 start : mismatched function location - need to move it elsewhere and delete from here. 
    def keypoints_to_array(self, all_frames: List[dict]) -> np.ndarray:
        """Convert list of frames with keypoints to a numpy array."""
        num_frames = len(all_frames)
        if num_frames == 0:
            return np.empty((0, 0, 3), dtype=np.float32)

        first_kp = all_frames[0]['persons'][0]['keypoints']
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
    # fixme 3 end
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
            
            