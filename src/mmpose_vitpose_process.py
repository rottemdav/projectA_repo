"""
WholeBody Pose Estimation using MMPose

This script demonstrates how to use WholeBody pose estimation models from MMPose.
WholeBody models detect 133 keypoints: body (17), feet (6), face (68), and hands (42).
It supports both image and video inputs and uses MMDetection for person detection.

Requirements:
    - mmpose (with mmengine, mmcv)
    - mmdet (for person detection)
    - torch

Usage:
    python mmpose_vitpose_process.py
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple, Union

# MMPose imports
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

# MMDetection imports
try:
    from mmdet.apis import inference_detector, init_detector
    HAS_MMDET = True
except (ImportError, ModuleNotFoundError):
    HAS_MMDET = False
    print("Warning: mmdet not found. Person detection will not be available.")


# ==================== Configuration ====================
class Config:
    """Configuration for WholeBody pose estimation."""
    
    # Paths - Update these to match your setup
    MMPOSE_ROOT = "/home/projects/sipl-prj10496/project_files/mmpose"
    
    # ============ WholeBody Model Options ============
    # Option 1: HRNet-W48 WholeBody (Best accuracy, 133 keypoints)
    POSE_CONFIG = os.path.join(
        MMPOSE_ROOT, 
        "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288.py"
    )
    POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth"
    
    # Option 2: HRNet-W32 WholeBody (Faster, still good accuracy)
    # POSE_CONFIG = os.path.join(
    #     MMPOSE_ROOT, 
    #     "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py"
    # )
    # POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth"
    
    # Option 3: RTMPose-L WholeBody (Fast and accurate, good for real-time)
    # POSE_CONFIG = os.path.join(
    #     MMPOSE_ROOT, 
    #     "projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py"
    # )
    # POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth"
    
    # Detection model config and checkpoint (RTMDet recommended)
    DET_CONFIG = os.path.join(
        MMPOSE_ROOT, 
        "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
    )
    DET_CHECKPOINT = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    
    # Device
    DEVICE = "cuda:0"  # Use "cpu" if no GPU available
    
    # Detection parameters
    DET_CAT_ID = 0  # Category ID for person in COCO
    BBOX_THR = 0.3  # Bounding box score threshold
    NMS_THR = 0.3   # IoU threshold for NMS
    
    # Visualization parameters
    KPT_THR = 0.3   # Keypoint confidence threshold for visualization
    RADIUS = 8      # Keypoint radius (increased for better visibility)
    THICKNESS = 4   # Skeleton line thickness (increased for better visibility)
    DRAW_FACE = False  # Whether to draw face keypoints (set False to skip face)
    
    # Video processing parameters
    START_FRAME = 2700    # Frame to start processing from (0 = beginning)
    MAX_FRAMES = None     # Maximum frames to process (None = all frames)
    END_FRAME = 3900      # Frame to end processing (inclusive, None = till end)
    
    # Input/Output paths
    INPUT_PATH = "/home/projects/sipl-prj10496/project_files/data/source_videos/NL124/4-3336/GX010030[1].MP4"
    OUTPUT_DIR = "/home/projects/sipl-prj10496/project_files/data/mmpose_hrnet_wholebody_output"


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
        pose_config: str = Config.POSE_CONFIG,
        pose_checkpoint: str = Config.POSE_CHECKPOINT,
        det_config: str = Config.DET_CONFIG,
        det_checkpoint: str = Config.DET_CHECKPOINT,
        device: str = Config.DEVICE,
        bbox_thr: float = Config.BBOX_THR,
        nms_thr: float = Config.NMS_THR,
        kpt_thr: float = Config.KPT_THR,
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
        self.kpt_thr = kpt_thr
        self.det_cat_id = Config.DET_CAT_ID
        
        # Change to mmpose root directory for config resolution
        original_dir = os.getcwd()
        os.chdir(Config.MMPOSE_ROOT)
        
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
            visualizer_cfg['radius'] = Config.RADIUS
            visualizer_cfg['line_width'] = Config.THICKNESS
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
            kpt_thr=self.kpt_thr
        )
        
        # Get the visualization result
        vis_image = self.visualizer.get_image()
        
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
        json_results = []
        frame_idx = start_frame
        processed_count = 0
        
        while frame_idx <= eff_end_frame:
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
                json_results.append(frame_json)

                processed_count += 1

                if processed_count % 10 == 0:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] Processed frame {frame_idx} ({processed_count} processed, start={start_frame})/{total_frames}")

            frame_idx += 1
        
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
        return all_results
    
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
    
    def get_face_keypoints(self, keypoints_data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract only face keypoints (68 keypoints)."""
        return keypoints_data['face']['keypoints'], keypoints_data['face']['scores']
    
    def get_hand_keypoints(self, keypoints_data: dict, hand: str = 'both') -> dict:
        """
        Extract hand keypoints.
        
        Args:
            keypoints_data: Keypoints dict from extract_keypoints
            hand: 'left', 'right', or 'both'
        """
        if hand == 'left':
            return keypoints_data['left_hand']
        elif hand == 'right':
            return keypoints_data['right_hand']
        else:
            return {
                'left': keypoints_data['left_hand'],
                'right': keypoints_data['right_hand']
            }


def main():
    """Main function demonstrating WholeBody pose estimation usage."""
    
    # Check if mmdet is available
    if not HAS_MMDET:
        print("Error: mmdet is required for person detection.")
        print("Install with: pip install mmdet")
        return
    
    # Initialize processor
    processor = WholeBodyPoseProcessor(
        pose_config=Config.POSE_CONFIG,
        pose_checkpoint=Config.POSE_CHECKPOINT,
        det_config=Config.DET_CONFIG,
        det_checkpoint=Config.DET_CHECKPOINT,
        device=Config.DEVICE,
        bbox_thr=Config.BBOX_THR,
        kpt_thr=Config.KPT_THR,
    )
    
    # Example: Process a single image
    # Uncomment and modify paths as needed
    """
    image_path = "/path/to/your/image.jpg"
    output_path = os.path.join(Config.OUTPUT_DIR, "result.jpg")
    pose_results, vis_image = processor.process_image(
        image_path, 
        output_path=output_path,
        show=False
    )
    
    # Extract keypoints in a clean format
    keypoints = processor.extract_keypoints(pose_results)
    for i, kp_data in enumerate(keypoints):
        print(f"Person {i}: {kp_data['keypoints'].shape[0]} keypoints")
        # Access specific body parts
        body_kp, body_scores = processor.get_body_keypoints(kp_data)
        face_kp, face_scores = processor.get_face_keypoints(kp_data)
        hands = processor.get_hand_keypoints(kp_data, 'both')
        print(f"  Body: {body_kp.shape}, Face: {face_kp.shape}")
        print(f"  Left hand: {hands['left']['keypoints'].shape}")
        print(f"  Right hand: {hands['right']['keypoints'].shape}")
    """
    
    # ============ Process Video ============
    video_path = Config.INPUT_PATH
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Use end_frame in output filename if set, else max_frames, else till end
    if Config.END_FRAME is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.END_FRAME}"
    elif Config.MAX_FRAMES is not None:
        out_range = f"{Config.START_FRAME}_to_{Config.START_FRAME + Config.MAX_FRAMES - 1}"
    else:
        out_range = f"{Config.START_FRAME}_to_end"
    output_path = os.path.join(Config.OUTPUT_DIR, f"{video_name}_wholebody_{out_range}.mp4")
    json_output_path = os.path.join(Config.OUTPUT_DIR, f"{video_name}_wholebody_{out_range}.json")

    all_results = processor.process_video(
        video_path,
        output_path=output_path,
        start_frame=Config.START_FRAME,  # Start from this frame
        max_frames=Config.MAX_FRAMES,    # None for all frames
        end_frame=Config.END_FRAME,      # None for till end, or set for explicit range
        draw_face=Config.DRAW_FACE,      # Set False to hide face keypoints
        show=False,
        json_output_path=json_output_path
    )
    
    # Extract and print keypoints for the first frame with detections
    if all_results:
        frame_idx, pose_results = all_results[0]
        keypoints = processor.extract_keypoints(pose_results)
        print(f"\nFrame {frame_idx}: Detected {len(keypoints)} person(s)")
        for i, kp_data in enumerate(keypoints):
            print(f"  Person {i}: {kp_data['keypoints'].shape[0]} keypoints")
    
    print("\n" + "="*60)
    print("WholeBody Pose Processor initialized successfully!")
    print("="*60)
    print("\nCOCO-WholeBody 133 Keypoint Format:")
    print("  Body (0-16):    17 keypoints - standard COCO body") 
    print("  Feet (17-22):    6 keypoints - left/right big toe, small toe, heel")
    print("  Face (23-90):   68 keypoints - facial landmarks")
    print("  Left hand (91-111):  21 keypoints")
    print("  Right hand (112-132): 21 keypoints")
    print("\nBody keypoints (0-16):")
    print("  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear")
    print("  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow")
    print("  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip")
    print("  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle")
    print("\nUsage examples:")
    print("  # Process image:")
    print("  pose_results, vis_img = processor.process_image('image.jpg', 'output.jpg')")
    print("  # Process video:") 
    print("  results = processor.process_video('video.mp4', 'output.mp4')")
    print("  # Extract all keypoints:")
    print("  keypoints = processor.extract_keypoints(pose_results)")
    print("  # Get specific body parts:")
    print("  body_kp, scores = processor.get_body_keypoints(keypoints[0])")
    print("  face_kp, scores = processor.get_face_keypoints(keypoints[0])")
    print("  hands = processor.get_hand_keypoints(keypoints[0], 'both')")


if __name__ == "__main__":
    main()
