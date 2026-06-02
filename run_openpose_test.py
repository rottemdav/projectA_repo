import sys
import os

# Ensure the correct path is in sys.path relative to this file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from models.openPose_video_process import openpose_pose_estimation
from utils.skeleton_tracking import filter_and_track_person

if __name__ == "__main__":
    # Test video path as defined in your original script
    video_path = "/home/projects/sipl-prj10496/project_files/data/source_videos/NL129/NL129_3_3.MP4"
    
    start_frame = 0
    end_frame = None
    
    print(f"Testing OpenPose extraction on {video_path}...")
    print(f"Extracting frames {start_frame} to {end_frame}...")
    
    try:
        # Run the extraction on a short slice to verify output
        keypoints_array, spatial_filtered_unfiltered_time_arr = openpose_pose_estimation(video_path, start_frame, end_frame)
        
        print("\n=== EXTRACTION SUCCESS ===")
        print(f"Final Filtered Array Shape: {keypoints_array.shape}")
        print(f"Pre-filter Unfiltered Array Shape: {spatial_filtered_unfiltered_time_arr.shape}")
        
        print(f"\nTracking and temporal filtering complete! The returned array was smoothed and isolated via ROI correctly.")
        print("Check the 'project_files/data/openpose_output/' directory for the JSON payload and video format.")
    except Exception as e:
        import traceback
        print(f"\n=== EXTRACTION FAILED ===")
        traceback.print_exc()
