import sys
import os

# Ensure the correct path is in sys.path relative to this file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from models.openPose_video_process import openpose_pose_estimation
from utils.skeleton_tracking import filter_and_track_person

if __name__ == "__main__":
    # Test video path as defined in your original script
    video_path = "/home/projects/sipl-prj10496/project_files/data/source_videos/HC70/HC70_1.MP4"
    
    start_frame = 300
    end_frame = 350
    
    print(f"Testing OpenPose extraction on {video_path}...")
    print(f"Extracting frames {start_frame} to {end_frame}...")
    
    try:
        # Run the extraction on a short slice to verify output
        keypoints_array = openpose_pose_estimation(video_path, start_frame, end_frame)
        
        print("\n=== EXTRACTION SUCCESS ===")
        print(f"Final Extracted Array Shape: {keypoints_array.shape}")
        
        print("\n=== TESTING TARGET TRACKING PIPELINE ON EXTRACTED DATA ===")
        last_known_hip_x = None
        cam_num = int(video_path.split('_')[-1].split('.')[0]) # Extracts '1' from HC70_1.MP4
        valid_frames = 0
        
        # keypoints_array is shape (N_frames, 25, 3). 
        # We wrap each frame's keypoints in a list to simulate `all_keypoints` for a single frame.
        for i, frame_kp in enumerate(keypoints_array):
            all_keypoints_simulate = [frame_kp.tolist()]
            
            best_person_idx = filter_and_track_person(
                all_keypoints=all_keypoints_simulate,
                last_known_hip_x=last_known_hip_x,
                cam_num=cam_num,
                model_type="openpose",
                use_roi_filter=True,
                min_conf=0.3
            )
            
            if best_person_idx != -1:
                last_known_hip_x = all_keypoints_simulate[best_person_idx][8][0] # Update tracker
                valid_frames += 1
                if i % 10 == 0:
                    print(f"Frame {i:02d}: Target Locked! MidHip X: {last_known_hip_x:.1f}")
            else:
                if i % 10 == 0:
                    print(f"Frame {i:02d}: No valid target found. Tracking anchor frozen.")
                
        print(f"\nTracking complete! Found valid target in {valid_frames}/{len(keypoints_array)} extracted frames.")
        print("Check the 'project_files/data/openpose_output/' directory for the JSON payload and video format.")
    except Exception as e:
        import traceback
        print(f"\n=== EXTRACTION FAILED ===")
        traceback.print_exc()
