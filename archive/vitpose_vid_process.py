import os
import cv2
import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

# ======= variables settings (edit here) =======
FRAME_NUMBER = 850  # Frame to process (0-indexed, use -1 for last frame)
VIDEO_PATH = "/home/projects/sipl-prj10496/project_files/data/source_videos/NL124/3-1786/GX010036[1].MP4"
VIDEO_FILENAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PATH = f"/home/projects/sipl-prj10496/project_files/data/vitpose_output_imgs/NL124/3-1786/{VIDEO_FILENAME}_frame{FRAME_NUMBER}.jpg"

DETECTION_THRESHOLD = 0.3
KP_THRESHOLD = 0.3

#models
DETECTOR_ID = "PekingU/rtdetr_r50vd_coco_o365"
VITPOSE_ID = "usyd-community/vitpose-plus-base"

# Dataset index for ViTPose+ (multi-dataset model with 6 MoE expert heads):
# 0: COCO (17 keypoints)
# 1: AiC
# 2: MPII
# 3: AP-10K
# 4: APT-36K
# 5: COCO-WholeBody (133 keypoints)
DATASET_IDX = 5

# ======= main code  =======
def main():
    # ---- reading frames ----
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle frame selection
    frame_idx = FRAME_NUMBER if FRAME_NUMBER >= 0 else total_frames + FRAME_NUMBER
    frame_idx = max(0, min(frame_idx, total_frames - 1))  # Clamp to valid range
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Cannot read frame {frame_idx} from video file")
        return
    
    print(f"Processing frame {frame_idx} of {total_frames}")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    device = Accelerator().device

    # ---- detect people ----
    det_processor = AutoProcessor.from_pretrained(DETECTOR_ID)
    det_model = RTDetrForObjectDetection.from_pretrained(DETECTOR_ID, device_map=device)

    det_inputs = det_processor(images=image, return_tensors="pt").to(det_model.device)
    with torch.no_grad():
        det_outputs = det_model(**det_inputs)

    det_results = det_processor.post_process_object_detection(
        det_outputs, 
        target_sizes = torch.tensor([(image.height, image.width)]),
        threshold = DETECTION_THRESHOLD
    )[0]

    # Get person boxes (label 0 in COCO)
    person_boxes = det_results["boxes"][det_results["labels"]==0]
    person_boxes = person_boxes.cpu().numpy()
    
    if person_boxes.shape[0] == 0:
        print("No person detected")
        raise RuntimeError("No person detected")
    
    # Keep VOC format for drawing
    person_boxes_voc = person_boxes.copy()
    
    # Convert VOC (x1, y1, x2, y2) --> COCO (x1, y1, w, h) for ViTPose
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    # ---- pose estimation : run ViTPose WholeBody----
    pose_processor = AutoProcessor.from_pretrained(VITPOSE_ID)
    pose_model = VitPoseForPoseEstimation.from_pretrained(VITPOSE_ID, device_map=device)

    pose_inputs = pose_processor(image, boxes=[person_boxes], return_tensors="pt").to(pose_model.device)

    with torch.no_grad():
        # dataset_index must be a tensor of shape (batch_size,) where batch_size = number of images
        dataset_index = torch.tensor([DATASET_IDX], device=pose_model.device)
        pose_outputs = pose_model(**pose_inputs, dataset_index=dataset_index)

    pose_results = pose_processor.post_process_pose_estimation(pose_outputs, boxes=[person_boxes])[0]

    print(f"Detected {len(pose_results)} persons")
    print(f"keypoints per person: {pose_results[0]['keypoints'].shape[0]}")

    # ---- draw keypoints ----
    annotated = frame.copy()

    # Draw bounding boxes for debugging
    for box in person_boxes_voc:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for person in pose_results:
        keypoints = person["keypoints"].cpu().numpy()
        scores = person["scores"].cpu().numpy()

        for kp, score in zip(keypoints, scores):
            if score < KP_THRESHOLD:
                continue
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)

    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
